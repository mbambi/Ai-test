# =============================================================================
# Module: models
# =============================================================================
# Ce module contient les architectures de réseaux de neurones:
# - U-Net: Pour la segmentation sémantique des zones de soudure
# - CNN Classifier: Pour la classification de l'homogénéité
# =============================================================================

from .unet import UNet
from .cnn_classifier import HomogeneityClassifier

__all__ = ["UNet", "HomogeneityClassifier"]
