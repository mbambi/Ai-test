# =============================================================================
# Entraîneur de Modèles pour l'Analyse de Soudure
# =============================================================================
# 
# Ce module implémente la boucle d'entraînement pour les modèles:
# - U-Net pour la segmentation
# - CNN pour la classification d'homogénéité
#
# Fonctionnalités:
# - Entraînement avec validation
# - Learning rate scheduling
# - Early stopping
# - Logging avec TensorBoard
# - Sauvegarde des checkpoints
#
# =============================================================================

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable
from tqdm import tqdm
import json
from datetime import datetime

# Imports locaux
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.unet import UNet
from models.cnn_classifier import HomogeneityClassifier
from training.dataset import WeldDataset, WeldClassificationDataset, SyntheticWeldDataset


class DiceLoss(nn.Module):
    """
    Dice Loss pour la segmentation binaire.
    
    La Dice Loss est particulièrement efficace pour les problèmes
    de segmentation avec déséquilibre de classes (peu de pixels
    de soudure par rapport au fond).
    
    Formule: DiceLoss = 1 - (2 * intersection + smooth) / (sum + smooth)
    
    Paramètres:
    -----------
    smooth : float
        Terme de lissage pour éviter la division par zéro
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la Dice Loss.
        
        Paramètres:
        -----------
        predictions : torch.Tensor
            Prédictions du modèle (après sigmoid)
        targets : torch.Tensor
            Masques de vérité terrain
            
        Retourne:
        ---------
        torch.Tensor
            Valeur de la loss
        """
        # Appliquer sigmoid si ce n'est pas déjà fait
        if predictions.max() > 1 or predictions.min() < 0:
            predictions = torch.sigmoid(predictions)
        
        # Aplatir les tenseurs
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculer l'intersection
        intersection = (predictions * targets).sum()
        
        # Calculer le coefficient Dice
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combinaison de BCE Loss et Dice Loss.
    
    Cette combinaison tire parti des deux approches:
    - BCE: Optimise chaque pixel indépendamment
    - Dice: Optimise la superposition globale
    
    Paramètres:
    -----------
    bce_weight : float
        Poids de la BCE Loss (défaut: 0.5)
    dice_weight : float
        Poids de la Dice Loss (défaut: 0.5)
    """
    
    def __init__(
        self, 
        bce_weight: float = 0.5, 
        dice_weight: float = 0.5
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calcule la loss combinée."""
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(torch.sigmoid(predictions), targets)
        
        return self.bce_weight * bce + self.dice_weight * dice


class MetricsTracker:
    """
    Suivi des métriques pendant l'entraînement.
    
    Calcule et conserve l'historique des métriques:
    - Loss
    - IoU (Intersection over Union)
    - Dice coefficient
    - Precision/Recall
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Réinitialise toutes les métriques."""
        self.losses = []
        self.ious = []
        self.dices = []
        self.precisions = []
        self.recalls = []
    
    def update(
        self,
        loss: float,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
    ):
        """
        Met à jour les métriques avec un batch.
        
        Paramètres:
        -----------
        loss : float
            Valeur de la loss
        predictions : torch.Tensor
            Prédictions (après sigmoid)
        targets : torch.Tensor
            Vérité terrain
        threshold : float
            Seuil de binarisation
        """
        self.losses.append(loss)
        
        # Binariser les prédictions
        pred_binary = (predictions > threshold).float()
        
        # Aplatir
        pred = pred_binary.view(-1)
        target = targets.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred * target).sum().item()
        fp = (pred * (1 - target)).sum().item()
        fn = ((1 - pred) * target).sum().item()
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-8)
        self.ious.append(iou)
        
        # Dice
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        self.dices.append(dice)
        
        # Precision/Recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        self.precisions.append(precision)
        self.recalls.append(recall)
    
    def get_averages(self) -> Dict[str, float]:
        """Retourne les moyennes des métriques."""
        return {
            'loss': np.mean(self.losses),
            'iou': np.mean(self.ious),
            'dice': np.mean(self.dices),
            'precision': np.mean(self.precisions),
            'recall': np.mean(self.recalls)
        }


class Trainer:
    """
    Classe d'entraînement pour les modèles de soudure.
    
    Gère tout le processus d'entraînement:
    - Boucle d'entraînement et validation
    - Scheduling du learning rate
    - Early stopping
    - Sauvegarde des checkpoints
    - Logging
    
    Paramètres:
    -----------
    model : nn.Module
        Modèle à entraîner
    train_loader : DataLoader
        DataLoader d'entraînement
    val_loader : DataLoader
        DataLoader de validation
    criterion : nn.Module
        Fonction de loss
    optimizer : optim.Optimizer
        Optimiseur
    device : str
        Appareil de calcul
    output_dir : str
        Répertoire de sortie
    model_name : str
        Nom du modèle pour les sauvegardes
        
    Exemple d'utilisation:
    ----------------------
    >>> model = UNet(n_channels=1, n_classes=1)
    >>> trainer = Trainer(model, train_loader, val_loader, ...)
    >>> trainer.train(epochs=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cuda",
        output_dir: str = "./outputs",
        model_name: str = "model",
        scheduler: Optional[Any] = None,
        early_stopping_patience: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        
        # Créer les répertoires
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(
            self.output_dir / "tensorboard" / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Suivi des métriques
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Historique
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rate': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        print(f"Trainer initialisé:")
        print(f"  Device: {device}")
        print(f"  Output: {self.output_dir}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    
    def train(
        self,
        epochs: int,
        save_frequency: int = 5,
        log_frequency: int = 10
    ) -> Dict[str, Any]:
        """
        Lance l'entraînement.
        
        Paramètres:
        -----------
        epochs : int
            Nombre d'epochs
        save_frequency : int
            Fréquence de sauvegarde des checkpoints
        log_frequency : int
            Fréquence de logging dans la console
            
        Retourne:
        ---------
        Dict[str, Any]
            Historique d'entraînement
        """
        print(f"\n{'='*60}")
        print(f"  DÉBUT DE L'ENTRAÎNEMENT - {epochs} epochs")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Entraînement
            train_metrics = self._train_epoch(epoch)
            
            # Validation
            val_metrics = self._validate_epoch(epoch)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Mise à jour de l'historique
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard
            self._log_tensorboard(epoch, train_metrics, val_metrics, current_lr)
            
            # Logging console
            if epoch % log_frequency == 0 or epoch == 1:
                self._print_progress(epoch, epochs, train_metrics, val_metrics, current_lr)
            
            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Sauvegarde du meilleur modèle
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Sauvegarde périodique
            if epoch % save_frequency == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping à l'epoch {epoch}")
                break
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  ENTRAÎNEMENT TERMINÉ")
        print(f"  Temps total: {total_time/60:.1f} minutes")
        print(f"  Meilleure val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        # Sauvegarder l'historique
        self._save_history()
        
        # Fermer TensorBoard
        self.writer.close()
        
        return self.history
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Entraîne le modèle pour une epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False
        )
        
        for batch_images, batch_masks in pbar:
            # Transfert vers le device
            batch_images = batch_images.to(self.device)
            batch_masks = batch_masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_images)
            loss = self.criterion(outputs, batch_masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            with torch.no_grad():
                predictions = torch.sigmoid(outputs)
                self.train_metrics.update(
                    loss.item(),
                    predictions,
                    batch_masks
                )
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return self.train_metrics.get_averages()
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Valide le modèle pour une epoch.
        """
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for batch_images, batch_masks in self.val_loader:
                batch_images = batch_images.to(self.device)
                batch_masks = batch_masks.to(self.device)
                
                outputs = self.model(batch_images)
                loss = self.criterion(outputs, batch_masks)
                
                predictions = torch.sigmoid(outputs)
                self.val_metrics.update(
                    loss.item(),
                    predictions,
                    batch_masks
                )
        
        return self.val_metrics.get_averages()
    
    def _log_tensorboard(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float
    ):
        """Log les métriques dans TensorBoard."""
        self.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        self.writer.add_scalars('IoU', {
            'train': train_metrics['iou'],
            'val': val_metrics['iou']
        }, epoch)
        
        self.writer.add_scalars('Dice', {
            'train': train_metrics['dice'],
            'val': val_metrics['dice']
        }, epoch)
        
        self.writer.add_scalar('Learning Rate', lr, epoch)
    
    def _print_progress(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float
    ):
        """Affiche la progression dans la console."""
        print(f"Epoch {epoch:3d}/{total_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | "
              f"LR: {lr:.2e}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Sauvegarde un checkpoint du modèle."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Sauvegarde régulière
        if not is_best:
            path = self.output_dir / "checkpoints" / f"{self.model_name}_epoch{epoch}.pth"
        else:
            path = self.output_dir / f"{self.model_name}_best.pth"
        
        torch.save(checkpoint, path)
    
    def _save_history(self):
        """Sauvegarde l'historique d'entraînement."""
        history_path = self.output_dir / f"{self.model_name}_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Charge un checkpoint pour reprendre l'entraînement.
        
        Paramètres:
        -----------
        checkpoint_path : str
            Chemin vers le fichier checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        print(f"Checkpoint chargé: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")


def train_model(
    data_dir: str,
    model_type: str = "segmentation",
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    output_dir: str = "./models",
    resume_from: Optional[str] = None
):
    """
    Fonction principale d'entraînement.
    
    Paramètres:
    -----------
    data_dir : str
        Répertoire des données
    model_type : str
        "segmentation", "classification", ou "both"
    epochs : int
        Nombre d'epochs
    batch_size : int
        Taille des batchs
    learning_rate : float
        Taux d'apprentissage
    device : str
        Appareil de calcul
    output_dir : str
        Répertoire de sortie
    resume_from : str, optional
        Chemin du checkpoint pour reprendre
    """
    print(f"\n{'='*60}")
    print(f"  CONFIGURATION DE L'ENTRAÎNEMENT")
    print(f"{'='*60}")
    print(f"  Data: {data_dir}")
    print(f"  Model type: {model_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type in ["segmentation", "both"]:
        print("Entraînement du modèle de segmentation...")
        
        # Vérifier si les données existent, sinon utiliser données synthétiques
        images_dir = os.path.join(data_dir, "images")
        masks_dir = os.path.join(data_dir, "masks")
        
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            # Données réelles
            train_dataset = WeldDataset(
                images_dir, masks_dir,
                augmentation=True
            )
        else:
            print("Données non trouvées, utilisation de données synthétiques")
            train_dataset = SyntheticWeldDataset(n_samples=500)
        
        # Créer une validation set
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Créer le modèle
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        
        # Criterion et optimizer
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            output_dir=output_dir,
            model_name="unet_segmentation",
            scheduler=scheduler,
            early_stopping_patience=15
        )
        
        # Reprendre si demandé
        if resume_from:
            trainer.load_checkpoint(resume_from)
        
        # Entraîner
        trainer.train(epochs=epochs)
    
    if model_type in ["classification", "both"]:
        print("\nEntraînement du modèle de classification...")
        # Implémentation similaire pour le classificateur
        print("(Classification training à implémenter avec données labellisées)")


if __name__ == "__main__":
    # ==========================================================================
    # TEST DE L'ENTRAÎNEMENT
    # ==========================================================================
    
    print("=" * 60)
    print("Test de l'entraînement")
    print("=" * 60)
    
    # Vérifier CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Créer des données synthétiques
    train_dataset = SyntheticWeldDataset(n_samples=100, image_size=(256, 256))
    val_dataset = SyntheticWeldDataset(n_samples=20, image_size=(256, 256))
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Créer le modèle
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    
    # Loss et optimizer
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_dir="./test_training",
        model_name="test_model"
    )
    
    # Entraîner quelques epochs
    print("\nTest d'entraînement (5 epochs)...")
    history = trainer.train(epochs=5, log_frequency=1)
    
    print("\nHistorique:")
    print(f"  Train loss: {history['train_loss']}")
    print(f"  Val loss: {history['val_loss']}")
    
    print("\n✓ Test de l'entraînement réussi!")
