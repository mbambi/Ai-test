# =============================================================================
# Projet: Segmentation et Analyse de Zones de Soudure
# =============================================================================
# 
# Ce package fournit des outils pour:
# - La segmentation automatique des zones de soudure (U-Net)
# - L'analyse de la largeur des cordons de soudure
# - L'évaluation de la continuité des soudures
# - La classification de l'homogénéité (CNN)
# - Le calcul de scores de qualité
#
# Auteur: Projet de contrôle qualité automatisé
# Version: 1.0.0
# =============================================================================

__version__ = "1.0.0"
__author__ = "Weld Analysis Team"

# Importation des modules principaux pour faciliter l'accès
from .models.unet import UNet
from .models.cnn_classifier import HomogeneityClassifier
from .utils.image_processing import ImageProcessor
from .utils.video_processing import VideoProcessor
from .analysis.weld_analyzer import WeldAnalyzer
from .pipeline import WeldAnalysisPipeline

# Liste des classes et fonctions exportées
__all__ = [
    "UNet",
    "HomogeneityClassifier", 
    "ImageProcessor",
    "VideoProcessor",
    "WeldAnalyzer",
    "WeldAnalysisPipeline",
]
