# 🔧 Système d'Analyse de Soudure par Intelligence Artificielle

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Ce projet implémente un système complet d'**analyse automatisée de zones de soudure** utilisant des techniques de deep learning. Il permet de :

- 🎯 **Segmentation automatique** des zones de soudure sur images et vidéos
- 📏 **Mesure de la largeur** du cordon de soudure
- 🔗 **Analyse de la continuité** (détection des interruptions)
- 🎨 **Évaluation de l'homogénéité** de la texture
- ⭐ **Calcul de scores de qualité** (notation A-F)
- 📊 **Génération de rapports** détaillés

## 🏗️ Architecture

```
weld_analysis/
├── models/                  # Architectures de réseaux de neurones
│   ├── unet.py             # U-Net pour la segmentation
│   └── cnn_classifier.py   # CNN pour la classification d'homogénéité
├── utils/                   # Utilitaires
│   ├── image_processing.py # Traitement d'images
│   ├── video_processing.py # Traitement vidéo
│   └── data_augmentation.py# Augmentation de données
├── analysis/               # Outils d'analyse
│   ├── weld_analyzer.py    # Analyse géométrique et qualitative
│   └── quality_scorer.py   # Système de notation
├── training/               # Entraînement des modèles
│   ├── dataset.py          # Chargement des données
│   └── trainer.py          # Boucle d'entraînement
├── configs/                # Fichiers de configuration
│   └── default_config.yaml
├── pipeline.py             # Pipeline principal d'analyse
├── cli.py                  # Interface en ligne de commande
└── requirements.txt        # Dépendances Python
```

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- CUDA 11.0+ (optionnel, pour l'accélération GPU)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/votre-repo/weld-analysis.git
cd weld-analysis

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r weld_analysis/requirements.txt
```

## 📖 Utilisation

### Interface en Ligne de Commande (CLI)

#### Analyser une image

```bash
python -m weld_analysis.cli analyze image_soudure.jpg
```

Options disponibles :
```bash
python -m weld_analysis.cli analyze image.jpg \
    --output ./resultats \
    --device cuda \
    --model ./models/unet_best.pth \
    --threshold 0.5 \
    --report rapport.txt \
    --json resultats.json \
    --verbose
```

#### Analyser plusieurs images

```bash
python -m weld_analysis.cli analyze ./images/*.jpg \
    --output ./resultats
```

#### Analyser une vidéo

```bash
python -m weld_analysis.cli video video_soudure.mp4 \
    --output video_annotee.mp4 \
    --interval 0.5
```

#### Entraîner un modèle

```bash
python -m weld_analysis.cli train \
    --data ./dataset \
    --model-type segmentation \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.0001 \
    --device cuda \
    --output ./models
```

### Utilisation en Python

```python
from weld_analysis import WeldAnalysisPipeline, PipelineConfig

# Configuration
config = PipelineConfig(
    image_size=(256, 256),
    device="cuda",  # ou "cpu"
    output_dir="./outputs"
)

# Créer le pipeline
pipeline = WeldAnalysisPipeline(
    config=config,
    segmentation_model_path="./models/unet_best.pth"
)

# Analyser une image
result = pipeline.analyze_image("image_soudure.jpg")

# Afficher les résultats
print(f"Note de qualité: {result.quality_grade}")
print(f"Score: {result.quality_score:.1%}")
print(f"Acceptable: {result.is_acceptable}")
print(f"Largeur moyenne: {result.measurements['width_mean']:.1f} px")
print(f"Continuité: {result.continuity['score']:.1%}")
print(f"Homogénéité: {result.homogeneity['score']:.1%}")

# Générer un rapport
report = pipeline.generate_report(result)
print(report)
```

### Analyse détaillée

```python
from weld_analysis import WeldAnalyzer, QualityScorer

# Charger l'image et le masque
import cv2
image = cv2.imread("soudure.jpg", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("masque.png", cv2.IMREAD_GRAYSCALE)

# Analyse géométrique
analyzer = WeldAnalyzer(pixels_per_mm=10.0)
analysis = analyzer.analyze(image, mask)

# Mesures de largeur
print(f"Largeur moyenne: {analysis['measurements'].width_mean:.2f} px")
print(f"Écart-type: {analysis['measurements'].width_std:.2f} px")

# Continuité
print(f"Score de continuité: {analysis['continuity'].continuity_score:.1%}")
print(f"Interruptions: {analysis['continuity'].n_gaps}")

# Homogénéité
print(f"Score d'homogénéité: {analysis['homogeneity'].score:.1%}")
print(f"Défauts: {analysis['homogeneity'].defect_ratio:.1%}")

# Score de qualité
scorer = QualityScorer()
quality = scorer.score(analysis)
print(f"Note: {quality.grade.name} ({quality.overall_score:.1%})")
```

## 🧠 Modèles

### U-Net pour la Segmentation

L'architecture U-Net est utilisée pour la segmentation sémantique des zones de soudure :

- **Encodeur** : Extraction de caractéristiques à différentes résolutions
- **Décodeur** : Reconstruction du masque de segmentation
- **Skip connections** : Préservation des détails spatiaux

```python
from weld_analysis.models import UNet

model = UNet(
    n_channels=1,      # Niveaux de gris
    n_classes=1,       # Segmentation binaire
    bilinear=True,     # Interpolation bilinéaire
    base_features=64   # Caractéristiques de base
)

# Nombre de paramètres
total, trainable = model.get_num_parameters()
print(f"Paramètres: {total:,}")
```

### CNN pour la Classification d'Homogénéité

Un CNN classifie les patches de soudure selon leur homogénéité :

- **Classe 0** : Non homogène (défauts majeurs)
- **Classe 1** : Partiellement homogène (défauts mineurs)
- **Classe 2** : Homogène (qualité acceptable)

```python
from weld_analysis.models import HomogeneityClassifier

classifier = HomogeneityClassifier(
    n_channels=1,
    n_classes=3,
    dropout=0.3
)

# Score d'homogénéité (0-1)
score = classifier.get_homogeneity_score(patch_tensor)
```

## 📊 Métriques de Qualité

### Critères d'évaluation

| Critère | Description | Seuil par défaut |
|---------|-------------|------------------|
| Variation de largeur | Coefficient de variation max | 20% |
| Continuité | Score minimum de continuité | 85% |
| Homogénéité | Score minimum d'homogénéité | 70% |
| Défauts | Ratio maximum de zones défectueuses | 5% |
| Interruptions | Nombre maximum de gaps | 2 |

### Système de notation

| Note | Score | Description |
|------|-------|-------------|
| A | ≥90% | Excellent - Aucun défaut visible |
| B | 80-89% | Bon - Défauts mineurs acceptables |
| C | 70-79% | Acceptable - Défauts modérés |
| D | 60-69% | Marginal - Défauts significatifs |
| F | <60% | Inacceptable - Défauts majeurs |

## 📁 Structure des Données

### Pour l'entraînement

```
dataset/
├── images/           # Images de soudure
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── masks/            # Masques de segmentation
│   ├── img001.png    # Même nom que l'image
│   ├── img002.png
│   └── ...
└── labels.csv        # Labels de classification (optionnel)
```

Format du fichier `labels.csv` :
```csv
filename,label
patch001.png,0
patch002.png,2
patch003.png,1
```

### Métadonnées caméra (optionnel)

```json
{
    "angle": 85.0,
    "resolution": 12.5,
    "distortion": [0.01, -0.02, 0.001, 0.0, 0.0],
    "exposure": "auto"
}
```

## 🔧 Configuration

Créez un fichier de configuration personnalisé :

```yaml
# config.yaml
general:
  device: "cuda"
  
preprocessing:
  image_size: [512, 512]
  
segmentation:
  threshold: 0.6
  min_weld_area: 200
  
analysis:
  pixels_per_mm: 15.0
  quality_criteria:
    min_continuity: 0.90
```

Utilisez-le avec :
```bash
python -m weld_analysis.cli analyze image.jpg --config config.yaml
```

## 📈 Entraînement

### Données synthétiques (pour le prototypage)

```python
from weld_analysis.training import SyntheticWeldDataset
from torch.utils.data import DataLoader

# Créer un dataset synthétique
dataset = SyntheticWeldDataset(n_samples=1000, image_size=(256, 256))
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Entraînement complet

```python
from weld_analysis.training import Trainer, WeldDataset, CombinedLoss
from weld_analysis.models import UNet
import torch.optim as optim

# Dataset
train_dataset = WeldDataset("data/images", "data/masks", augmentation=True)
val_dataset = WeldDataset("data/val_images", "data/val_masks", augmentation=False)

# Modèle
model = UNet(n_channels=1, n_classes=1)

# Trainer
trainer = Trainer(
    model=model,
    train_loader=DataLoader(train_dataset, batch_size=8),
    val_loader=DataLoader(val_dataset, batch_size=8),
    criterion=CombinedLoss(),
    optimizer=optim.AdamW(model.parameters(), lr=1e-4),
    device="cuda",
    output_dir="./checkpoints"
)

# Entraîner
history = trainer.train(epochs=100)
```

## 🗃️ Datasets Recommandés

- **GDXray** : Images industrielles publiques de soudures
- **DAGM** : Dataset de défauts de surface
- Dataset interne d'atelier (à créer)

## 🔍 Bonnes Pratiques

1. **Éclairage standardisé** : Utilisez un éclairage uniforme et reproductible
2. **Résolution adaptée** : 10-20 pixels/mm recommandé
3. **Angle de caméra** : Perpendiculaire ou légèrement incliné (>75°)
4. **Calibration** : Mesurez la résolution pixels/mm avec une mire
5. **Augmentation** : Activez l'augmentation de données pour l'entraînement

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout d'une fonctionnalité'`)
4. Push sur la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📞 Support

Pour toute question ou problème, ouvrez une issue sur GitHub.

---

**Développé avec ❤️ pour le contrôle qualité industriel automatisé**
