# =============================================================================
# Classificateur CNN pour l'Homogénéité des Soudures
# =============================================================================
# 
# Ce module implémente un réseau de neurones convolutif (CNN) pour classifier
# les zones de soudure selon leur homogénéité. L'homogénéité est un critère
# important de qualité qui évalue l'uniformité du cordon de soudure.
#
# Critères d'homogénéité évalués:
# - Uniformité de la texture
# - Régularité des motifs de solidification
# - Absence de porosités, fissures ou inclusions
# - Consistance de la couleur/brillance
#
# Classes de sortie:
# - 0: Non homogène (défauts détectés)
# - 1: Partiellement homogène (défauts mineurs)
# - 2: Homogène (qualité acceptable)
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class ConvBlock(nn.Module):
    """
    Bloc convolutif de base pour le classificateur CNN.
    
    Ce bloc combine:
    - Convolution 2D pour extraire des caractéristiques
    - Batch Normalization pour stabiliser l'entraînement
    - Activation ReLU pour la non-linéarité
    - Dropout optionnel pour la régularisation
    
    Paramètres:
    -----------
    in_channels : int
        Nombre de canaux en entrée
    out_channels : int
        Nombre de canaux en sortie
    kernel_size : int
        Taille du noyau de convolution (défaut: 3)
    stride : int
        Pas de la convolution (défaut: 1)
    padding : int
        Padding autour de l'image (défaut: 1)
    dropout : float
        Taux de dropout (défaut: 0.0, pas de dropout)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Liste des couches du bloc
        layers = [
            # Convolution 2D
            # kernel_size=3 capture les motifs locaux
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False  # Pas de biais car BatchNorm le gère
            ),
            # Normalisation par batch
            # Normalise les activations pour chaque mini-batch
            nn.BatchNorm2d(out_channels),
            # Activation ReLU
            # f(x) = max(0, x) - introduit la non-linéarité
            nn.ReLU(inplace=True)
        ]
        
        # Ajout du dropout si spécifié
        if dropout > 0:
            # Dropout: désactive aléatoirement des neurones pendant l'entraînement
            # Aide à prévenir le surapprentissage
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passage avant du bloc convolutif."""
        return self.block(x)


class HomogeneityClassifier(nn.Module):
    """
    Réseau de neurones convolutif pour classifier l'homogénéité des soudures.
    
    Architecture:
    - 4 blocs convolutifs avec pooling progressif
    - Extraction de caractéristiques globales via AdaptiveAvgPool
    - Couches fully-connected pour la classification finale
    
    L'architecture est conçue pour:
    1. Capturer les textures locales (petits noyaux convolutifs)
    2. Agréger l'information spatiale (pooling)
    3. Détecter les motifs d'homogénéité/hétérogénéité
    
    Paramètres:
    -----------
    n_channels : int
        Nombre de canaux d'entrée (1 pour niveaux de gris, 3 pour RGB)
    n_classes : int
        Nombre de classes de sortie (défaut: 3 pour homogène/partiel/non-homogène)
    dropout : float
        Taux de dropout pour la régularisation (défaut: 0.3)
    
    Exemple d'utilisation:
    ----------------------
    >>> classifier = HomogeneityClassifier(n_channels=1, n_classes=3)
    >>> patch = torch.randn(1, 1, 64, 64)  # Patch de soudure
    >>> class_logits = classifier(patch)
    >>> predicted_class = class_logits.argmax(dim=1)
    """
    
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # =====================================================================
        # EXTRACTEUR DE CARACTÉRISTIQUES (Feature Extractor)
        # =====================================================================
        # Série de blocs convolutifs qui extraient des caractéristiques
        # de plus en plus abstraites
        
        self.features = nn.Sequential(
            # Bloc 1: Détection des contours et textures de base
            # Entrée: (n_channels, H, W) -> Sortie: (32, H/2, W/2)
            ConvBlock(n_channels, 32, kernel_size=3, padding=1),
            ConvBlock(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Réduit par 2
            
            # Bloc 2: Détection de motifs plus complexes
            # Entrée: (32, H/2, W/2) -> Sortie: (64, H/4, W/4)
            ConvBlock(32, 64, kernel_size=3, padding=1, dropout=dropout * 0.5),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloc 3: Caractéristiques de niveau intermédiaire
            # Entrée: (64, H/4, W/4) -> Sortie: (128, H/8, W/8)
            ConvBlock(64, 128, kernel_size=3, padding=1, dropout=dropout * 0.75),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloc 4: Caractéristiques de haut niveau (texture globale)
            # Entrée: (128, H/8, W/8) -> Sortie: (256, H/16, W/16)
            ConvBlock(128, 256, kernel_size=3, padding=1, dropout=dropout),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # =====================================================================
        # POOLING GLOBAL ADAPTATIF
        # =====================================================================
        # Réduit toute taille spatiale à 1x1, permettant des entrées
        # de tailles variables
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # =====================================================================
        # CLASSIFICATEUR (Fully Connected Layers)
        # =====================================================================
        self.classifier = nn.Sequential(
            # Aplatissement implicite: (256, 1, 1) -> (256,)
            nn.Flatten(),
            
            # Première couche FC
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Deuxième couche FC
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            # Couche de sortie
            nn.Linear(64, n_classes)
        )
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialise les poids du réseau selon les meilleures pratiques.
        
        - Convolutions: Initialisation de Kaiming (He)
        - BatchNorm: Poids = 1, Biais = 0
        - Linéaires: Initialisation de Xavier
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming initialization pour ReLU
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm: commencer avec une normalisation neutre
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Xavier initialization pour les couches FC
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du classificateur.
        
        Paramètres:
        -----------
        x : torch.Tensor
            Patch de soudure de forme (batch, n_channels, height, width)
            
        Retourne:
        ---------
        torch.Tensor
            Logits de classification de forme (batch, n_classes)
        """
        # Extraction des caractéristiques
        features = self.features(x)
        
        # Pooling global pour obtenir un vecteur par image
        pooled = self.global_pool(features)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit la classe d'homogénéité avec probabilités.
        
        Paramètres:
        -----------
        x : torch.Tensor
            Patch de soudure
            
        Retourne:
        ---------
        Tuple[torch.Tensor, torch.Tensor]
            - Classes prédites (indices)
            - Probabilités pour chaque classe
        """
        with torch.no_grad():
            logits = self.forward(x)
            # Softmax pour convertir en probabilités
            probabilities = F.softmax(logits, dim=1)
            # Classe avec la probabilité maximale
            predicted_classes = probabilities.argmax(dim=1)
            
        return predicted_classes, probabilities
    
    def get_homogeneity_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule un score d'homogénéité continu entre 0 et 1.
        
        Le score est calculé comme une moyenne pondérée des probabilités:
        - Classe 0 (non homogène): poids 0.0
        - Classe 1 (partiel): poids 0.5
        - Classe 2 (homogène): poids 1.0
        
        Paramètres:
        -----------
        x : torch.Tensor
            Patch de soudure
            
        Retourne:
        ---------
        torch.Tensor
            Score d'homogénéité entre 0 et 1
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            
            # Poids pour chaque classe
            weights = torch.tensor(
                [0.0, 0.5, 1.0], 
                device=probabilities.device
            )
            
            # Score = somme pondérée des probabilités
            score = (probabilities * weights).sum(dim=1)
            
        return score
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les caractéristiques profondes pour analyse.
        
        Utile pour:
        - Visualisation t-SNE/UMAP
        - Clustering de types de soudures
        - Détection d'anomalies
        
        Paramètres:
        -----------
        x : torch.Tensor
            Patch de soudure
            
        Retourne:
        ---------
        torch.Tensor
            Vecteur de caractéristiques de dimension 256
        """
        with torch.no_grad():
            features = self.features(x)
            pooled = self.global_pool(features)
            # Aplatir pour obtenir un vecteur
            feature_vector = pooled.view(pooled.size(0), -1)
            
        return feature_vector


class TextureFeatureExtractor(nn.Module):
    """
    Extracteur de caractéristiques de texture spécialisé.
    
    Ce module utilise des filtres de Gabor et des convolutions
    pour extraire des caractéristiques de texture pertinentes
    pour l'analyse d'homogénéité des soudures.
    
    Les filtres de Gabor sont particulièrement efficaces pour:
    - Détecter les orientations (motifs de solidification)
    - Analyser les fréquences spatiales (régularité)
    - Caractériser les textures locales
    """
    
    def __init__(
        self, 
        n_orientations: int = 8, 
        n_scales: int = 4
    ):
        super().__init__()
        
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        
        # Création des filtres de Gabor comme convolutions fixes
        # Les poids ne sont pas entraînés, ce sont des filtres prédéfinis
        self.gabor_filters = self._create_gabor_bank(
            n_orientations, 
            n_scales
        )
        
        # Convolution pour combiner les réponses de Gabor
        n_gabor_features = n_orientations * n_scales
        self.combine = nn.Sequential(
            nn.Conv2d(n_gabor_features, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def _create_gabor_bank(
        self, 
        n_orientations: int, 
        n_scales: int
    ) -> nn.Conv2d:
        """
        Crée une banque de filtres de Gabor.
        
        Les filtres de Gabor sont des ondelettes sinusoïdales modulées
        par une gaussienne, utilisées pour l'analyse de texture.
        
        Paramètres:
        -----------
        n_orientations : int
            Nombre d'orientations à analyser
        n_scales : int
            Nombre d'échelles (fréquences)
            
        Retourne:
        ---------
        nn.Conv2d
            Couche convolutive avec les filtres de Gabor
        """
        kernel_size = 21
        filters = []
        
        for scale in range(n_scales):
            # Fréquence spatiale
            frequency = 0.1 + scale * 0.1
            sigma = 3 + scale * 2
            
            for orientation in range(n_orientations):
                # Angle en radians
                theta = orientation * np.pi / n_orientations
                
                # Création du filtre de Gabor
                gabor = self._create_gabor_kernel(
                    kernel_size, 
                    sigma, 
                    theta, 
                    frequency
                )
                filters.append(gabor)
        
        # Empiler tous les filtres
        filter_bank = np.stack(filters)[:, np.newaxis, :, :]
        
        # Créer la couche convolutive
        n_filters = n_orientations * n_scales
        conv = nn.Conv2d(
            1, n_filters, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            bias=False
        )
        
        # Charger les filtres de Gabor (non entraînables)
        conv.weight.data = torch.from_numpy(filter_bank).float()
        conv.weight.requires_grad = False
        
        return conv
    
    def _create_gabor_kernel(
        self,
        size: int,
        sigma: float,
        theta: float,
        frequency: float
    ) -> np.ndarray:
        """
        Crée un noyau de Gabor individuel.
        
        Formule: g(x,y) = exp(-x'^2 + gamma^2*y'^2 / 2*sigma^2) * cos(2*pi*f*x')
        où x' et y' sont les coordonnées rotées par theta.
        """
        # Grille de coordonnées
        half = size // 2
        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        xx, yy = np.meshgrid(x, y)
        
        # Rotation des coordonnées
        x_theta = xx * np.cos(theta) + yy * np.sin(theta)
        y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
        
        # Paramètre d'aspect (rapport d'ellipticité)
        gamma = 0.5
        
        # Enveloppe gaussienne
        gaussian = np.exp(
            -(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2)
        )
        
        # Onde sinusoïdale
        sinusoid = np.cos(2 * np.pi * frequency * x_theta)
        
        # Filtre de Gabor
        gabor = gaussian * sinusoid
        
        # Normalisation
        gabor = gabor / np.sqrt(np.sum(gabor ** 2) + 1e-8)
        
        return gabor.astype(np.float32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les caractéristiques de texture.
        
        Paramètres:
        -----------
        x : torch.Tensor
            Image de soudure (batch, 1, H, W)
            
        Retourne:
        ---------
        torch.Tensor
            Caractéristiques de texture (batch, 32, H, W)
        """
        # Appliquer les filtres de Gabor
        gabor_responses = self.gabor_filters(x)
        
        # Prendre la valeur absolue (énergie)
        gabor_energy = torch.abs(gabor_responses)
        
        # Combiner les réponses
        texture_features = self.combine(gabor_energy)
        
        return texture_features


class EnhancedHomogeneityClassifier(nn.Module):
    """
    Classificateur d'homogénéité amélioré avec analyse de texture.
    
    Combine:
    - Extracteur de caractéristiques de texture (filtres de Gabor)
    - CNN pour la classification
    
    Cette architecture est plus efficace pour détecter les subtilités
    de texture dans les soudures.
    """
    
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Extracteur de texture
        self.texture_extractor = TextureFeatureExtractor(
            n_orientations=8, 
            n_scales=4
        )
        
        # CNN sur les caractéristiques de texture
        # Entrée: canaux originaux + caractéristiques de texture
        combined_channels = n_channels + 32  # 32 de texture
        
        self.classifier = HomogeneityClassifier(
            n_channels=combined_channels,
            n_classes=n_classes,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du classificateur amélioré.
        """
        # Extraire les caractéristiques de texture
        texture_features = self.texture_extractor(x)
        
        # Concaténer avec l'image originale
        combined = torch.cat([x, texture_features], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits


if __name__ == "__main__":
    # ==========================================================================
    # TEST DU CLASSIFICATEUR D'HOMOGÉNÉITÉ
    # ==========================================================================
    
    print("=" * 60)
    print("Test du classificateur CNN d'homogénéité de soudures")
    print("=" * 60)
    
    # Création du modèle
    classifier = HomogeneityClassifier(n_channels=1, n_classes=3)
    
    # Affichage de l'architecture
    print("\nArchitecture du modèle:")
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Nombre total de paramètres: {total_params:,}")
    
    # Test avec des patches exemple
    batch_size = 4
    height, width = 64, 64
    
    # Création de patches test
    test_patches = torch.randn(batch_size, 1, height, width)
    print(f"\nPatches d'entrée: {test_patches.shape}")
    
    # Passage avant
    logits = classifier(test_patches)
    print(f"Logits de sortie: {logits.shape}")
    
    # Prédiction
    classes, probs = classifier.predict(test_patches)
    print(f"Classes prédites: {classes}")
    print(f"Probabilités: {probs}")
    
    # Score d'homogénéité
    scores = classifier.get_homogeneity_score(test_patches)
    print(f"Scores d'homogénéité: {scores}")
    
    # Test des caractéristiques
    features = classifier.extract_features(test_patches)
    print(f"Caractéristiques: {features.shape}")
    
    # Test du classificateur amélioré
    print("\n" + "-" * 40)
    print("Test du classificateur amélioré avec texture")
    
    enhanced = EnhancedHomogeneityClassifier(n_channels=1, n_classes=3)
    enhanced_logits = enhanced(test_patches)
    print(f"Logits améliorés: {enhanced_logits.shape}")
    
    print("\n✓ Test du classificateur réussi!")
