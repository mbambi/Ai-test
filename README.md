# 🔧 Projet d'Analyse de Soudure par Intelligence Artificielle

## Vue d'ensemble

Ce repository contient un système complet d'**analyse automatisée de zones de soudure** utilisant des techniques de deep learning pour le contrôle qualité industriel.

## 🎯 Objectifs

- ✅ **Identifier automatiquement** les zones de soudure sur images ou vidéos
- ✅ **Évaluer la largeur**, la continuité et l'homogénéité des soudures
- ✅ **Faciliter le contrôle qualité automatisé** en milieu industriel

## 🧠 Technologies IA utilisées

| Tâche | Modèle | Description |
|-------|--------|-------------|
| Segmentation | **U-Net** | Segmentation sémantique des zones de soudure |
| Classification | **CNN** | Classification de l'homogénéité (3 classes) |

## 📥 Entrées

- Images ou vidéos de soudures
- Éclairage standardisé (recommandé)
- Métadonnées de la caméra (angle, résolution) - optionnel

## 📤 Sorties

- **Masques segmentés** des soudures (images binaires)
- **Mesures géométriques** : largeur moyenne, min, max, écart-type
- **Score de continuité** (0-100%)
- **Score d'homogénéité** (0-100%)
- **Note de qualité** (A, B, C, D, F)
- **Rapports détaillés** (texte, JSON)

## 📂 Structure du projet

```
weld_analysis/
├── models/                  # Modèles de deep learning
│   ├── unet.py             # U-Net pour segmentation
│   └── cnn_classifier.py   # CNN pour classification
├── utils/                   # Utilitaires
│   ├── image_processing.py # Traitement d'images
│   ├── video_processing.py # Traitement vidéo
│   └── data_augmentation.py# Augmentation de données
├── analysis/               # Analyse de qualité
│   ├── weld_analyzer.py    # Mesures et analyse
│   └── quality_scorer.py   # Système de notation
├── training/               # Entraînement
│   ├── dataset.py          # Chargement des données
│   └── trainer.py          # Boucle d'entraînement
├── configs/                # Configuration
├── pipeline.py             # Pipeline complet
├── cli.py                  # Interface CLI
└── requirements.txt        # Dépendances
```

## 🚀 Démarrage rapide

### Installation

```bash
cd weld_analysis
pip install -r requirements.txt
```

### Analyser une image

```bash
python -m weld_analysis analyze image_soudure.jpg --output ./resultats
```

### Analyser une vidéo

```bash
python -m weld_analysis video video_soudure.mp4 --output video_annotee.mp4
```

### Utilisation en Python

```python
from weld_analysis import WeldAnalysisPipeline

pipeline = WeldAnalysisPipeline()
result = pipeline.analyze_image("soudure.jpg")

print(f"Note: {result.quality_grade}")
print(f"Score: {result.quality_score:.1%}")
print(f"Acceptable: {result.is_acceptable}")
```

## 📊 Datasets recommandés

- **GDXray** : Images industrielles publiques de soudures
- **Dataset interne d'atelier** : Images spécifiques à votre processus

## 📖 Documentation complète

Voir le fichier [weld_analysis/README.md](weld_analysis/README.md) pour la documentation complète.

## 📄 Licence

MIT License

---

**Développé pour le contrôle qualité industriel automatisé**
