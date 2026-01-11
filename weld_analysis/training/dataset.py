# =============================================================================
# Dataset pour l'Entraînement des Modèles de Soudure
# =============================================================================
# 
# Ce module implémente les classes de dataset pour charger et préparer
# les données d'entraînement:
# - WeldDataset: Pour la segmentation (U-Net)
# - WeldClassificationDataset: Pour la classification d'homogénéité (CNN)
#
# Structure attendue des données:
# dataset/
# ├── images/           # Images de soudure
# │   ├── img001.jpg
# │   └── ...
# ├── masks/            # Masques de segmentation (même nom que les images)
# │   ├── img001.png
# │   └── ...
# └── labels.csv        # Labels de classification (optionnel)
#
# =============================================================================

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Dict, Any
import random

# Import local
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_augmentation import DataAugmentation, WeldSpecificAugmentation


class WeldDataset(Dataset):
    """
    Dataset pour l'entraînement du modèle de segmentation U-Net.
    
    Charge des paires (image, masque) pour entraîner le modèle
    à segmenter les zones de soudure.
    
    Paramètres:
    -----------
    images_dir : str
        Répertoire contenant les images
    masks_dir : str
        Répertoire contenant les masques de segmentation
    image_size : Tuple[int, int]
        Taille de redimensionnement (height, width)
    augmentation : bool
        Si True, applique l'augmentation de données
    transform : Callable, optional
        Transformation personnalisée à appliquer
        
    Exemple d'utilisation:
    ----------------------
    >>> dataset = WeldDataset("data/images", "data/masks", augmentation=True)
    >>> image, mask = dataset[0]
    >>> print(image.shape, mask.shape)  # torch.Size([1, 256, 256])
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        augmentation: bool = False,
        transform: Optional[Callable] = None
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augmentation = augmentation
        self.transform = transform
        
        # Extensions supportées
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Lister les images et vérifier les masques correspondants
        self.samples = self._find_samples()
        
        # Augmentation de données
        if augmentation:
            self.augmenter = WeldSpecificAugmentation(
                p_geometric=0.7,
                p_photometric=0.5,
                p_industrial=0.3
            )
        else:
            self.augmenter = None
        
        print(f"Dataset chargé: {len(self.samples)} échantillons")
    
    def _find_samples(self) -> List[Tuple[Path, Path]]:
        """
        Trouve les paires image-masque valides.
        
        Retourne:
        ---------
        List[Tuple[Path, Path]]
            Liste de (chemin_image, chemin_masque)
        """
        samples = []
        
        # Parcourir les images
        for img_path in self.images_dir.iterdir():
            if img_path.suffix.lower() not in self.image_extensions:
                continue
            
            # Chercher le masque correspondant
            # Le masque peut avoir une extension différente
            mask_name = img_path.stem
            mask_path = None
            
            for ext in self.image_extensions:
                potential_mask = self.masks_dir / f"{mask_name}{ext}"
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            if mask_path is not None:
                samples.append((img_path, mask_path))
            else:
                print(f"Avertissement: Pas de masque pour {img_path.name}")
        
        return sorted(samples, key=lambda x: x[0].name)
    
    def __len__(self) -> int:
        """Retourne le nombre d'échantillons."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Charge et retourne un échantillon.
        
        Paramètres:
        -----------
        idx : int
            Index de l'échantillon
            
        Retourne:
        ---------
        Tuple[torch.Tensor, torch.Tensor]
            (image, masque) sous forme de tenseurs
        """
        img_path, mask_path = self.samples[idx]
        
        # Charger l'image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Impossible de charger: {img_path}")
        
        # Charger le masque
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Impossible de charger: {mask_path}")
        
        # Redimensionner
        image = cv2.resize(image, self.image_size[::-1])
        mask = cv2.resize(mask, self.image_size[::-1])
        
        # Binariser le masque
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Augmentation
        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)
        
        # Transformation personnalisée
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # Conversion en tenseurs
        # Image: normalisation [0, 1] et ajout de la dimension du canal
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0)  # (1, H, W)
        
        # Masque: binarisation [0, 1]
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)
        
        return image_tensor, mask_tensor
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Calcule les poids des échantillons pour le sampling équilibré.
        
        Utile si certaines classes sont sous-représentées.
        
        Retourne:
        ---------
        torch.Tensor
            Poids pour chaque échantillon
        """
        weights = []
        
        for _, mask_path in self.samples:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Plus le masque contient de soudure, plus le poids est élevé
                coverage = np.mean(mask > 127)
                weight = 1.0 + coverage  # Entre 1.0 et 2.0
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


class WeldClassificationDataset(Dataset):
    """
    Dataset pour l'entraînement du classificateur d'homogénéité.
    
    Charge des patches d'images avec leurs labels de classe:
    - 0: Non homogène
    - 1: Partiellement homogène
    - 2: Homogène
    
    Paramètres:
    -----------
    images_dir : str
        Répertoire contenant les images (patches ou images complètes)
    labels_file : str
        Fichier CSV avec les colonnes: filename, label
    image_size : Tuple[int, int]
        Taille des patches (height, width)
    augmentation : bool
        Si True, applique l'augmentation
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_file: str,
        image_size: Tuple[int, int] = (64, 64),
        augmentation: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.augmentation = augmentation
        
        # Charger les labels
        self.labels_df = pd.read_csv(labels_file)
        
        # Vérifier les fichiers existants
        self.samples = self._validate_samples()
        
        # Augmentation
        if augmentation:
            self.augmenter = DataAugmentation(
                p_geometric=0.5,
                p_photometric=0.7
            )
        else:
            self.augmenter = None
        
        # Statistiques des classes
        self._print_class_distribution()
    
    def _validate_samples(self) -> List[Tuple[Path, int]]:
        """
        Valide que les fichiers existent et retourne la liste.
        """
        samples = []
        
        for _, row in self.labels_df.iterrows():
            filename = row['filename']
            label = int(row['label'])
            
            img_path = self.images_dir / filename
            if img_path.exists():
                samples.append((img_path, label))
            else:
                print(f"Avertissement: {filename} non trouvé")
        
        return samples
    
    def _print_class_distribution(self):
        """Affiche la distribution des classes."""
        labels = [s[1] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        
        print("\nDistribution des classes:")
        class_names = ["Non homogène", "Partiel", "Homogène"]
        for cls, count in zip(unique, counts):
            name = class_names[cls] if cls < len(class_names) else f"Classe {cls}"
            print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Charge et retourne un échantillon.
        
        Retourne:
        ---------
        Tuple[torch.Tensor, int]
            (image, label)
        """
        img_path, label = self.samples[idx]
        
        # Charger l'image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Impossible de charger: {img_path}")
        
        # Redimensionner
        image = cv2.resize(image, self.image_size[::-1])
        
        # Augmentation
        if self.augmenter is not None:
            image, _ = self.augmenter(image, None)
        
        # Conversion en tenseur
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calcule les poids des classes pour gérer le déséquilibre.
        
        Retourne:
        ---------
        torch.Tensor
            Poids pour chaque classe
        """
        labels = [s[1] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        
        # Poids inversement proportionnels à la fréquence
        total = len(labels)
        weights = torch.zeros(max(unique) + 1)
        
        for cls, count in zip(unique, counts):
            weights[cls] = total / (len(unique) * count)
        
        return weights


class SyntheticWeldDataset(Dataset):
    """
    Dataset synthétique pour les tests et le prototypage.
    
    Génère des images de soudure synthétiques avec des masques
    correspondants. Utile pour:
    - Tester le pipeline sans données réelles
    - Pré-entraînement avant fine-tuning
    - Validation du code
    
    Paramètres:
    -----------
    n_samples : int
        Nombre d'échantillons à générer
    image_size : Tuple[int, int]
        Taille des images
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.n_samples = n_samples
        self.image_size = image_size
        
        # Pré-calculer les paramètres aléatoires pour reproductibilité
        np.random.seed(42)
        self.params = [self._generate_params() for _ in range(n_samples)]
    
    def _generate_params(self) -> Dict[str, Any]:
        """Génère les paramètres pour une soudure synthétique."""
        h, w = self.image_size
        
        return {
            'weld_y': random.randint(h // 4, 3 * h // 4),
            'weld_width': random.randint(20, 60),
            'weld_length': random.randint(w // 2, w - 20),
            'weld_start': random.randint(10, w // 4),
            'noise_level': random.uniform(10, 30),
            'has_gap': random.random() < 0.2,
            'gap_x': random.randint(w // 3, 2 * w // 3),
            'gap_width': random.randint(5, 15)
        }
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Génère un échantillon synthétique."""
        params = self.params[idx]
        h, w = self.image_size
        
        # Créer l'image de fond (niveaux de gris variables)
        background = np.random.normal(100, 20, (h, w)).astype(np.float32)
        
        # Créer le masque
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Dessiner la soudure (bande horizontale)
        y_start = params['weld_y'] - params['weld_width'] // 2
        y_end = params['weld_y'] + params['weld_width'] // 2
        x_start = params['weld_start']
        x_end = x_start + params['weld_length']
        
        # Limiter aux dimensions de l'image
        y_start = max(0, y_start)
        y_end = min(h, y_end)
        x_end = min(w, x_end)
        
        mask[y_start:y_end, x_start:x_end] = 1.0
        
        # Ajouter une interruption si spécifié
        if params['has_gap']:
            gap_start = params['gap_x']
            gap_end = gap_start + params['gap_width']
            mask[y_start:y_end, gap_start:gap_end] = 0.0
        
        # Créer l'apparence de la soudure sur l'image
        weld_intensity = np.random.normal(170, 15, (h, w)).astype(np.float32)
        image = np.where(mask > 0.5, weld_intensity, background)
        
        # Ajouter du bruit
        noise = np.random.normal(0, params['noise_level'], (h, w))
        image = image + noise
        
        # Clipper et normaliser
        image = np.clip(image, 0, 255).astype(np.float32) / 255.0
        
        # Convertir en tenseurs
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return image_tensor, mask_tensor


def create_data_loaders(
    train_images_dir: str,
    train_masks_dir: str,
    val_images_dir: Optional[str] = None,
    val_masks_dir: Optional[str] = None,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders pour l'entraînement et la validation.
    
    Paramètres:
    -----------
    train_images_dir : str
        Répertoire des images d'entraînement
    train_masks_dir : str
        Répertoire des masques d'entraînement
    val_images_dir : str, optional
        Répertoire des images de validation
        Si non fourni, split automatique
    val_masks_dir : str, optional
        Répertoire des masques de validation
    image_size : Tuple[int, int]
        Taille des images
    batch_size : int
        Taille des batchs
    num_workers : int
        Nombre de workers pour le chargement
    val_split : float
        Proportion pour la validation si pas de répertoire dédié
        
    Retourne:
    ---------
    Tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    if val_images_dir and val_masks_dir:
        # Datasets séparés
        train_dataset = WeldDataset(
            train_images_dir,
            train_masks_dir,
            image_size=image_size,
            augmentation=True
        )
        
        val_dataset = WeldDataset(
            val_images_dir,
            val_masks_dir,
            image_size=image_size,
            augmentation=False
        )
    else:
        # Split automatique
        full_dataset = WeldDataset(
            train_images_dir,
            train_masks_dir,
            image_size=image_size,
            augmentation=False  # Sera ajouté manuellement
        )
        
        # Calculer les tailles
        n_total = len(full_dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        
        # Split aléatoire
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # ==========================================================================
    # TEST DES DATASETS
    # ==========================================================================
    
    print("=" * 60)
    print("Test des datasets de soudure")
    print("=" * 60)
    
    # Test du dataset synthétique
    print("\nTest du dataset synthétique:")
    synth_dataset = SyntheticWeldDataset(n_samples=100, image_size=(256, 256))
    print(f"  Taille: {len(synth_dataset)}")
    
    image, mask = synth_dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Mask unique: {torch.unique(mask)}")
    
    # Test du DataLoader
    print("\nTest du DataLoader:")
    loader = DataLoader(synth_dataset, batch_size=8, shuffle=True)
    
    for batch_images, batch_masks in loader:
        print(f"  Batch images: {batch_images.shape}")
        print(f"  Batch masks: {batch_masks.shape}")
        break
    
    print("\n✓ Test des datasets réussi!")
