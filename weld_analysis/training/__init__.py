# =============================================================================
# Module: training
# =============================================================================
# Ce module contient les outils d'entraînement des modèles:
# - WeldDataset: Chargement des données
# - Trainer: Boucle d'entraînement
# - Métriques et évaluation
# =============================================================================

from .dataset import WeldDataset, WeldClassificationDataset
from .trainer import Trainer, train_model

__all__ = ["WeldDataset", "WeldClassificationDataset", "Trainer", "train_model"]
