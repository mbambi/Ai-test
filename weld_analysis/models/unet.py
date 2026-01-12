# =============================================================================
# Modèle U-Net pour la Segmentation de Zones de Soudure
# =============================================================================
# 
# U-Net est une architecture de réseau de neurones convolutif spécialement
# conçue pour la segmentation d'images biomédicales, mais très efficace
# pour toute tâche de segmentation sémantique.
#
# Architecture:
# - Encodeur (chemin de contraction): Extraction des caractéristiques
# - Décodeur (chemin d'expansion): Reconstruction de la segmentation
# - Connexions de saut (skip connections): Préservation des détails spatiaux
#
# Référence: Ronneberger, O., Fischer, P., & Brox, T. (2015)
# "U-Net: Convolutional Networks for Biomedical Image Segmentation"
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DoubleConv(nn.Module):
    """
    Bloc de double convolution utilisé dans U-Net.
    
    Ce bloc applique deux convolutions successives, chacune suivie de:
    - Batch Normalization: Stabilise l'entraînement et accélère la convergence
    - ReLU: Activation non-linéaire pour introduire de la non-linéarité
    
    Structure: (Conv2d -> BatchNorm2d -> ReLU) x 2
    
    Paramètres:
    -----------
    in_channels : int
        Nombre de canaux en entrée
    out_channels : int
        Nombre de canaux en sortie
    mid_channels : int, optional
        Nombre de canaux intermédiaires (défaut: out_channels)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mid_channels: Optional[int] = None
    ):
        super().__init__()
        
        # Si non spécifié, les canaux intermédiaires = canaux de sortie
        if mid_channels is None:
            mid_channels = out_channels
            
        # Séquence de double convolution
        self.double_conv = nn.Sequential(
            # Première convolution 3x3
            # padding=1 conserve les dimensions spatiales
            nn.Conv2d(
                in_channels, 
                mid_channels, 
                kernel_size=3, 
                padding=1,
                bias=False  # Pas de biais car BatchNorm le gère
            ),
            # Normalisation par batch pour stabiliser l'entraînement
            nn.BatchNorm2d(mid_channels),
            # Activation ReLU (remplace les valeurs négatives par 0)
            nn.ReLU(inplace=True),
            
            # Deuxième convolution 3x3
            nn.Conv2d(
                mid_channels, 
                out_channels, 
                kernel_size=3, 
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc de double convolution.
        
        Paramètres:
        -----------
        x : torch.Tensor
            Tensor d'entrée de forme (batch, channels, height, width)
            
        Retourne:
        ---------
        torch.Tensor
            Tensor de sortie après double convolution
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    Bloc de sous-échantillonnage (downsampling) de l'encodeur U-Net.
    
    Ce bloc réduit les dimensions spatiales par 2 tout en augmentant
    le nombre de canaux, permettant d'extraire des caractéristiques
    de plus haut niveau.
    
    Structure: MaxPool2d -> DoubleConv
    
    Paramètres:
    -----------
    in_channels : int
        Nombre de canaux en entrée
    out_channels : int
        Nombre de canaux en sortie
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            # MaxPooling 2x2: Réduit la taille par 2
            # Garde la valeur maximale dans chaque fenêtre 2x2
            nn.MaxPool2d(kernel_size=2),
            # Double convolution pour extraire les caractéristiques
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc de sous-échantillonnage.
        
        Paramètres:
        -----------
        x : torch.Tensor
            Tensor d'entrée
            
        Retourne:
        ---------
        torch.Tensor
            Tensor sous-échantillonné avec plus de canaux
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Bloc de sur-échantillonnage (upsampling) du décodeur U-Net.
    
    Ce bloc augmente les dimensions spatiales par 2 tout en réduisant
    le nombre de canaux, permettant de reconstruire la segmentation
    avec les détails spatiaux des couches d'encodeur correspondantes.
    
    Deux modes disponibles:
    - Transposed convolution: Appris, plus de paramètres
    - Bilinear interpolation: Plus rapide, moins de paramètres
    
    Paramètres:
    -----------
    in_channels : int
        Nombre de canaux en entrée
    out_channels : int
        Nombre de canaux en sortie
    bilinear : bool
        Si True, utilise l'interpolation bilinéaire
        Si False, utilise la convolution transposée
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bilinear: bool = True
    ):
        super().__init__()
        
        if bilinear:
            # Interpolation bilinéaire: méthode classique de redimensionnement
            # scale_factor=2 double les dimensions spatiales
            self.up = nn.Upsample(
                scale_factor=2, 
                mode='bilinear', 
                align_corners=True
            )
            # Double convolution avec canaux intermédiaires réduits
            self.conv = DoubleConv(
                in_channels, 
                out_channels, 
                mid_channels=in_channels // 2
            )
        else:
            # Convolution transposée: apprend comment faire le upsampling
            # stride=2 double les dimensions spatiales
            self.up = nn.ConvTranspose2d(
                in_channels, 
                in_channels // 2, 
                kernel_size=2, 
                stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Passage avant du bloc de sur-échantillonnage.
        
        Combine le tensor sur-échantillonné avec le tensor correspondant
        de l'encodeur via une connexion de saut (skip connection).
        
        Paramètres:
        -----------
        x1 : torch.Tensor
            Tensor à sur-échantillonner (venant du décodeur)
        x2 : torch.Tensor
            Tensor de l'encodeur pour la skip connection
            
        Retourne:
        ---------
        torch.Tensor
            Tensor fusionné et convolué
        """
        # Sur-échantillonnage de x1
        x1 = self.up(x1)
        
        # Calcul des différences de dimensions pour le padding
        # Nécessaire si les dimensions ne correspondent pas exactement
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        # Padding de x1 pour correspondre aux dimensions de x2
        # Format: (left, right, top, bottom)
        x1 = F.pad(x1, [
            diff_x // 2, 
            diff_x - diff_x // 2,
            diff_y // 2, 
            diff_y - diff_y // 2
        ])
        
        # Concaténation le long de la dimension des canaux
        # C'est la skip connection qui préserve les détails spatiaux
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """
    Couche de sortie du U-Net.
    
    Convolution 1x1 qui réduit le nombre de canaux au nombre de classes
    pour la segmentation (1 pour binaire: soudure/non-soudure).
    
    Paramètres:
    -----------
    in_channels : int
        Nombre de canaux en entrée
    out_channels : int
        Nombre de classes de segmentation
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Convolution 1x1 pour réduire les canaux
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applique la convolution 1x1 finale."""
        return self.conv(x)


class UNet(nn.Module):
    """
    Architecture U-Net complète pour la segmentation de zones de soudure.
    
    L'architecture U-Net est composée de:
    1. Un encodeur qui extrait des caractéristiques à différentes résolutions
    2. Un décodeur qui reconstruit la segmentation à la résolution originale
    3. Des skip connections qui préservent les détails spatiaux
    
    Pour la segmentation de soudures, le modèle produit un masque binaire
    où chaque pixel est classifié comme soudure (1) ou non-soudure (0).
    
    Paramètres:
    -----------
    n_channels : int
        Nombre de canaux d'entrée (1 pour niveaux de gris, 3 pour RGB)
    n_classes : int
        Nombre de classes de sortie (1 pour segmentation binaire)
    bilinear : bool
        Si True, utilise l'interpolation bilinéaire pour le upsampling
    base_features : int
        Nombre de caractéristiques de base (doublé à chaque niveau)
        
    Exemple d'utilisation:
    ----------------------
    >>> model = UNet(n_channels=1, n_classes=1, bilinear=True)
    >>> image = torch.randn(1, 1, 256, 256)  # Batch de 1 image 256x256
    >>> mask = model(image)  # Masque de segmentation prédit
    >>> print(mask.shape)  # torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self, 
        n_channels: int = 1, 
        n_classes: int = 1, 
        bilinear: bool = True,
        base_features: int = 64
    ):
        super().__init__()
        
        # Sauvegarde des paramètres pour référence
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Facteur pour le nombre de canaux dans le dernier bloc down
        # Bilinear utilise moins de canaux car pas de conv transposée
        factor = 2 if bilinear else 1
        
        # =====================================================================
        # ENCODEUR (Chemin de contraction)
        # =====================================================================
        # Chaque bloc Down réduit la résolution par 2 et double les canaux
        
        # Bloc initial: extrait les premières caractéristiques
        # Entrée: (n_channels, H, W) -> Sortie: (64, H, W)
        self.inc = DoubleConv(n_channels, base_features)
        
        # Bloc Down 1: (64, H, W) -> (128, H/2, W/2)
        self.down1 = Down(base_features, base_features * 2)
        
        # Bloc Down 2: (128, H/2, W/2) -> (256, H/4, W/4)
        self.down2 = Down(base_features * 2, base_features * 4)
        
        # Bloc Down 3: (256, H/4, W/4) -> (512, H/8, W/8)
        self.down3 = Down(base_features * 4, base_features * 8)
        
        # Bloc Down 4 (goulot): (512, H/8, W/8) -> (1024 ou 512, H/16, W/16)
        self.down4 = Down(
            base_features * 8, 
            base_features * 16 // factor
        )
        
        # =====================================================================
        # DÉCODEUR (Chemin d'expansion)
        # =====================================================================
        # Chaque bloc Up double la résolution et réduit les canaux de moitié
        
        # Bloc Up 1: (1024, H/16, W/16) + skip -> (512, H/8, W/8)
        self.up1 = Up(
            base_features * 16, 
            base_features * 8 // factor, 
            bilinear
        )
        
        # Bloc Up 2: (512, H/8, W/8) + skip -> (256, H/4, W/4)
        self.up2 = Up(
            base_features * 8, 
            base_features * 4 // factor, 
            bilinear
        )
        
        # Bloc Up 3: (256, H/4, W/4) + skip -> (128, H/2, W/2)
        self.up3 = Up(
            base_features * 4, 
            base_features * 2 // factor, 
            bilinear
        )
        
        # Bloc Up 4: (128, H/2, W/2) + skip -> (64, H, W)
        self.up4 = Up(base_features * 2, base_features, bilinear)
        
        # Couche de sortie: (64, H, W) -> (n_classes, H, W)
        self.outc = OutConv(base_features, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du réseau U-Net.
        
        Paramètres:
        -----------
        x : torch.Tensor
            Image d'entrée de forme (batch, n_channels, height, width)
            
        Retourne:
        ---------
        torch.Tensor
            Masque de segmentation de forme (batch, n_classes, height, width)
            Les valeurs sont des logits (avant sigmoid pour binaire)
        """
        # =====================================================================
        # ENCODEUR: Extraction des caractéristiques à différents niveaux
        # =====================================================================
        
        # Niveau 1: Caractéristiques de base
        x1 = self.inc(x)      # (B, 64, H, W)
        
        # Niveau 2: Caractéristiques de bas niveau
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        
        # Niveau 3: Caractéristiques de niveau intermédiaire
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        
        # Niveau 4: Caractéristiques de haut niveau
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        
        # Niveau 5 (goulot): Caractéristiques les plus abstraites
        x5 = self.down4(x4)   # (B, 1024 ou 512, H/16, W/16)
        
        # =====================================================================
        # DÉCODEUR: Reconstruction avec skip connections
        # =====================================================================
        # Les skip connections (x4, x3, x2, x1) apportent les détails spatiaux
        
        x = self.up1(x5, x4)  # Fusion avec x4
        x = self.up2(x, x3)   # Fusion avec x3
        x = self.up3(x, x2)   # Fusion avec x2
        x = self.up4(x, x1)   # Fusion avec x1
        
        # Couche de sortie: projection vers le nombre de classes
        logits = self.outc(x)
        
        return logits
    
    def predict(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Prédit le masque de segmentation binaire.
        
        Applique sigmoid aux logits et seuille pour obtenir un masque binaire.
        
        Paramètres:
        -----------
        x : torch.Tensor
            Image d'entrée
        threshold : float
            Seuil pour la binarisation (défaut: 0.5)
            
        Retourne:
        ---------
        torch.Tensor
            Masque binaire (0 ou 1)
        """
        # Désactive le gradient pour l'inférence
        with torch.no_grad():
            logits = self.forward(x)
            # Sigmoid pour convertir logits en probabilités [0, 1]
            probs = torch.sigmoid(logits)
            # Binarisation selon le seuil
            mask = (probs > threshold).float()
            
        return mask
    
    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Calcule le nombre de paramètres du modèle.
        
        Retourne:
        ---------
        Tuple[int, int]
            (nombre total de paramètres, nombre de paramètres entraînables)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return total_params, trainable_params


# =============================================================================
# VARIANTES DU U-NET
# =============================================================================

class UNetSmall(UNet):
    """
    Version allégée du U-Net pour des ressources limitées.
    
    Utilise 32 caractéristiques de base au lieu de 64,
    réduisant le nombre de paramètres par 4.
    """
    
    def __init__(
        self, 
        n_channels: int = 1, 
        n_classes: int = 1, 
        bilinear: bool = True
    ):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=bilinear,
            base_features=32  # Réduit de moitié
        )


class AttentionGate(nn.Module):
    """
    Porte d'attention pour améliorer les skip connections.
    
    L'attention permet au modèle de se concentrer sur les régions
    pertinentes de la soudure et d'ignorer le bruit de fond.
    
    Paramètres:
    -----------
    F_g : int
        Nombre de canaux du signal gating (venant du décodeur)
    F_l : int
        Nombre de canaux du signal local (venant de l'encodeur)
    F_int : int
        Nombre de canaux intermédiaires
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        # Transformation du signal gating
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Transformation du signal local
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Calcul des coefficients d'attention
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # Coefficients entre 0 et 1
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self, 
        g: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applique la porte d'attention.
        
        Paramètres:
        -----------
        g : torch.Tensor
            Signal gating (du décodeur, plus profond)
        x : torch.Tensor
            Signal local (de l'encodeur, même niveau)
            
        Retourne:
        ---------
        torch.Tensor
            Signal local pondéré par l'attention
        """
        # Transformation du signal gating
        g1 = self.W_g(g)
        
        # Transformation du signal local
        x1 = self.W_x(x)
        
        # Combinaison additive et ReLU
        psi = self.relu(g1 + x1)
        
        # Calcul des coefficients d'attention
        psi = self.psi(psi)
        
        # Application de l'attention au signal local
        return x * psi


if __name__ == "__main__":
    # ==========================================================================
    # TEST DU MODÈLE U-NET
    # ==========================================================================
    
    print("=" * 60)
    print("Test du modèle U-Net pour segmentation de soudures")
    print("=" * 60)
    
    # Création du modèle
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    
    # Affichage du nombre de paramètres
    total, trainable = model.get_num_parameters()
    print(f"\nNombre de paramètres:")
    print(f"  - Total: {total:,}")
    print(f"  - Entraînables: {trainable:,}")
    
    # Test avec une image exemple
    batch_size = 2
    height, width = 256, 256
    
    # Création d'une image test (niveaux de gris)
    test_image = torch.randn(batch_size, 1, height, width)
    print(f"\nImage d'entrée: {test_image.shape}")
    
    # Passage avant
    output = model(test_image)
    print(f"Sortie (logits): {output.shape}")
    
    # Prédiction binaire
    mask = model.predict(test_image)
    print(f"Masque prédit: {mask.shape}")
    
    print("\n✓ Test du U-Net réussi!")
