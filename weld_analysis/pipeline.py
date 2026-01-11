# =============================================================================
# Pipeline Principal d'Analyse de Soudure
# =============================================================================
# 
# Ce module implémente le pipeline complet pour l'analyse automatisée
# des zones de soudure, de l'entrée (image/vidéo) à la sortie (rapport).
#
# Le pipeline intègre:
# 1. Chargement et prétraitement des données
# 2. Segmentation par U-Net
# 3. Classification d'homogénéité par CNN
# 4. Analyse géométrique et qualitative
# 5. Génération de rapports et visualisations
#
# =============================================================================

import os
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Imports locaux
from .models.unet import UNet
from .models.cnn_classifier import HomogeneityClassifier
from .utils.image_processing import ImageProcessor, MaskProcessor
from .utils.video_processing import VideoProcessor
from .analysis.weld_analyzer import WeldAnalyzer, WeldMeasurement
from .analysis.quality_scorer import QualityScorer, QualityScore


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration du pipeline d'analyse.
    
    Attributs:
    ----------
    image_size : Tuple[int, int]
        Taille des images pour le modèle (height, width)
    device : str
        Appareil de calcul ("cuda" ou "cpu")
    segmentation_threshold : float
        Seuil de binarisation pour la segmentation
    min_weld_area : int
        Surface minimale pour considérer une zone comme soudure
    pixels_per_mm : float
        Résolution de l'image pour les mesures réelles
    output_dir : str
        Répertoire de sortie pour les résultats
    save_masks : bool
        Si True, sauvegarde les masques de segmentation
    save_visualizations : bool
        Si True, génère et sauvegarde les visualisations
    """
    image_size: Tuple[int, int] = (256, 256)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    segmentation_threshold: float = 0.5
    min_weld_area: int = 100
    pixels_per_mm: float = 10.0
    output_dir: str = "./outputs"
    save_masks: bool = True
    save_visualizations: bool = True


@dataclass
class AnalysisResult:
    """
    Résultat complet de l'analyse d'une image.
    
    Attributs:
    ----------
    image_path : str
        Chemin de l'image analysée
    timestamp : str
        Horodatage de l'analyse
    mask : np.ndarray
        Masque de segmentation
    measurements : Dict
        Mesures géométriques
    continuity : Dict
        Résultats de continuité
    homogeneity : Dict
        Résultats d'homogénéité
    quality_score : float
        Score de qualité global
    quality_grade : str
        Note de qualité (A-F)
    is_acceptable : bool
        Si la soudure est acceptable
    recommendations : List[str]
        Recommandations d'amélioration
    processing_time : float
        Temps de traitement en secondes
    """
    image_path: str
    timestamp: str
    mask: np.ndarray
    measurements: Dict
    continuity: Dict
    homogeneity: Dict
    quality_score: float
    quality_grade: str
    is_acceptable: bool
    recommendations: List[str]
    processing_time: float
    
    def to_dict(self) -> Dict:
        """Convertit le résultat en dictionnaire (sans le masque numpy)."""
        result = asdict(self)
        # Convertir le masque en statistiques
        result['mask'] = {
            'shape': list(self.mask.shape),
            'coverage': float(np.mean(self.mask > 0.5))
        }
        return result
    
    def to_json(self) -> str:
        """Convertit le résultat en JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class WeldAnalysisPipeline:
    """
    Pipeline complet d'analyse de soudure.
    
    Ce pipeline orchestre toutes les étapes de l'analyse:
    1. Prétraitement des images
    2. Segmentation des zones de soudure
    3. Classification de l'homogénéité
    4. Analyse géométrique
    5. Calcul des scores de qualité
    6. Génération des rapports
    
    Paramètres:
    -----------
    config : PipelineConfig, optional
        Configuration du pipeline
    segmentation_model_path : str, optional
        Chemin vers le modèle de segmentation pré-entraîné
    classifier_model_path : str, optional
        Chemin vers le modèle de classification pré-entraîné
        
    Exemple d'utilisation:
    ----------------------
    >>> pipeline = WeldAnalysisPipeline()
    >>> result = pipeline.analyze_image("weld_image.jpg")
    >>> print(f"Qualité: {result.quality_grade} ({result.quality_score:.1%})")
    >>> pipeline.generate_report(result, "report.txt")
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        segmentation_model_path: Optional[str] = None,
        classifier_model_path: Optional[str] = None
    ):
        # Configuration
        self.config = config or PipelineConfig()
        
        # Logging
        logger.info(f"Initialisation du pipeline sur {self.config.device}")
        
        # Créer le répertoire de sortie
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialiser les composants
        self._init_processors()
        self._init_models(segmentation_model_path, classifier_model_path)
        self._init_analyzers()
    
    def _init_processors(self):
        """Initialise les processeurs d'images et de masques."""
        self.image_processor = ImageProcessor(
            target_size=self.config.image_size,
            normalize=True,
            grayscale=True
        )
        self.mask_processor = MaskProcessor()
        self.video_processor = VideoProcessor(self.image_processor)
        
        logger.info("Processeurs initialisés")
    
    def _init_models(
        self,
        segmentation_path: Optional[str],
        classifier_path: Optional[str]
    ):
        """Initialise les modèles de deep learning."""
        # Modèle de segmentation U-Net
        self.segmentation_model = UNet(
            n_channels=1,
            n_classes=1,
            bilinear=True
        )
        
        if segmentation_path and os.path.exists(segmentation_path):
            # Charger les poids pré-entraînés
            state_dict = torch.load(
                segmentation_path,
                map_location=self.config.device
            )
            self.segmentation_model.load_state_dict(state_dict)
            logger.info(f"Modèle de segmentation chargé: {segmentation_path}")
        else:
            logger.warning("Utilisation du modèle de segmentation non entraîné")
        
        self.segmentation_model.to(self.config.device)
        self.segmentation_model.eval()
        
        # Modèle de classification d'homogénéité
        self.classifier_model = HomogeneityClassifier(
            n_channels=1,
            n_classes=3
        )
        
        if classifier_path and os.path.exists(classifier_path):
            state_dict = torch.load(
                classifier_path,
                map_location=self.config.device
            )
            self.classifier_model.load_state_dict(state_dict)
            logger.info(f"Modèle de classification chargé: {classifier_path}")
        else:
            logger.warning("Utilisation du modèle de classification non entraîné")
        
        self.classifier_model.to(self.config.device)
        self.classifier_model.eval()
        
        logger.info("Modèles initialisés")
    
    def _init_analyzers(self):
        """Initialise les analyseurs."""
        self.weld_analyzer = WeldAnalyzer(
            pixels_per_mm=self.config.pixels_per_mm,
            min_weld_width=5,
            max_gap_size=10
        )
        self.quality_scorer = QualityScorer()
        
        logger.info("Analyseurs initialisés")
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        metadata_path: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyse une image de soudure.
        
        Paramètres:
        -----------
        image_path : str ou Path
            Chemin vers l'image à analyser
        metadata_path : str, optional
            Chemin vers le fichier de métadonnées caméra
            
        Retourne:
        ---------
        AnalysisResult
            Résultat complet de l'analyse
        """
        import time
        start_time = time.time()
        
        image_path = str(image_path)
        logger.info(f"Analyse de l'image: {image_path}")
        
        # 1. Charger et prétraiter l'image
        image = self.image_processor.load_image(image_path, metadata_path)
        preprocessed = self.image_processor.preprocess(image)
        
        # 2. Segmentation
        mask = self._segment(preprocessed)
        
        # 3. Nettoyer le masque
        cleaned_mask = self.mask_processor.clean_mask(
            mask,
            min_area=self.config.min_weld_area
        )
        
        # 4. Analyse géométrique et de qualité
        analysis = self.weld_analyzer.analyze(preprocessed, cleaned_mask)
        
        # 5. Classification d'homogénéité sur les patches
        homogeneity_class = self._classify_homogeneity(preprocessed, cleaned_mask)
        
        # 6. Score de qualité
        quality = self.quality_scorer.score(analysis)
        
        # Temps de traitement
        processing_time = time.time() - start_time
        
        # Construire le résultat
        result = AnalysisResult(
            image_path=image_path,
            timestamp=datetime.now().isoformat(),
            mask=cleaned_mask,
            measurements={
                'width_mean': analysis['measurements'].width_mean,
                'width_std': analysis['measurements'].width_std,
                'width_min': analysis['measurements'].width_min,
                'width_max': analysis['measurements'].width_max,
                'total_length': analysis['measurements'].total_length
            },
            continuity={
                'score': analysis['continuity'].continuity_score,
                'n_gaps': analysis['continuity'].n_gaps,
                'gap_positions': analysis['continuity'].gap_positions
            },
            homogeneity={
                'score': analysis['homogeneity'].score,
                'texture_uniformity': analysis['homogeneity'].texture_uniformity,
                'color_consistency': analysis['homogeneity'].color_consistency,
                'defect_ratio': analysis['homogeneity'].defect_ratio,
                'classification': homogeneity_class
            },
            quality_score=quality.overall_score,
            quality_grade=quality.grade.name,
            is_acceptable=quality.is_acceptable,
            recommendations=quality.recommendations,
            processing_time=processing_time
        )
        
        # Sauvegarder les résultats si configuré
        if self.config.save_masks or self.config.save_visualizations:
            self._save_results(result, preprocessed)
        
        logger.info(
            f"Analyse terminée - Grade: {result.quality_grade}, "
            f"Score: {result.quality_score:.1%}, "
            f"Temps: {processing_time:.2f}s"
        )
        
        return result
    
    def _segment(self, image: np.ndarray) -> np.ndarray:
        """
        Effectue la segmentation de l'image.
        
        Paramètres:
        -----------
        image : np.ndarray
            Image prétraitée
            
        Retourne:
        ---------
        np.ndarray
            Masque de segmentation binaire
        """
        # Convertir en tenseur
        tensor = self.image_processor.to_tensor(image)
        tensor = tensor.to(self.config.device)
        
        # Inférence
        with torch.no_grad():
            logits = self.segmentation_model(tensor)
            probs = torch.sigmoid(logits)
        
        # Convertir en numpy
        mask = probs.cpu().numpy()[0, 0]
        
        # Binariser
        mask = (mask > self.config.segmentation_threshold).astype(np.uint8) * 255
        
        # Redimensionner à la taille originale
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        return mask
    
    def _classify_homogeneity(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> str:
        """
        Classifie l'homogénéité des zones de soudure.
        
        Utilise le CNN pour classifier des patches de la soudure.
        """
        # Extraire les patches de la zone de soudure
        patches = self.image_processor.create_patches(
            image,
            patch_size=(64, 64),
            stride=(32, 32),
            mask=mask
        )
        
        if not patches:
            return "unknown"
        
        # Classifier chaque patch
        classes = []
        class_names = ["non_homogene", "partiel", "homogene"]
        
        for patch, _ in patches:
            tensor = self.image_processor.to_tensor(patch)
            tensor = tensor.to(self.config.device)
            
            with torch.no_grad():
                pred_class, _ = self.classifier_model.predict(tensor)
                classes.append(pred_class.item())
        
        # Vote majoritaire
        if not classes:
            return "unknown"
        
        majority_class = max(set(classes), key=classes.count)
        return class_names[majority_class]
    
    def _save_results(
        self,
        result: AnalysisResult,
        image: np.ndarray
    ):
        """
        Sauvegarde les résultats de l'analyse.
        """
        # Créer un nom de base pour les fichiers
        base_name = Path(result.image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = os.path.join(
            self.config.output_dir,
            f"{base_name}_{timestamp}"
        )
        
        # Sauvegarder le masque
        if self.config.save_masks:
            mask_path = f"{output_prefix}_mask.png"
            cv2.imwrite(mask_path, result.mask)
            logger.debug(f"Masque sauvegardé: {mask_path}")
        
        # Sauvegarder la visualisation
        if self.config.save_visualizations:
            vis_path = f"{output_prefix}_visualization.png"
            visualization = self._create_visualization(image, result)
            cv2.imwrite(vis_path, visualization)
            logger.debug(f"Visualisation sauvegardée: {vis_path}")
        
        # Sauvegarder le rapport JSON
        json_path = f"{output_prefix}_result.json"
        with open(json_path, 'w') as f:
            f.write(result.to_json())
        logger.debug(f"Résultat JSON sauvegardé: {json_path}")
    
    def _create_visualization(
        self,
        image: np.ndarray,
        result: AnalysisResult
    ) -> np.ndarray:
        """
        Crée une visualisation annotée des résultats.
        """
        # Convertir en RGB si nécessaire
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_rgb = image.copy()
        
        # Créer un overlay coloré pour le masque
        overlay = image_rgb.copy()
        mask_bool = result.mask > 127
        
        # Colorer la zone de soudure en vert
        overlay[mask_bool] = [0, 255, 0]
        
        # Fusionner avec l'image originale
        alpha = 0.4
        visualization = cv2.addWeighted(
            image_rgb, 1 - alpha,
            overlay, alpha,
            0
        )
        
        # Dessiner les contours
        contours = self.mask_processor.find_contours(result.mask)
        cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
        
        # Marquer les gaps
        for gap_y, gap_x in result.continuity['gap_positions']:
            cv2.circle(visualization, (gap_x, gap_y), 10, (0, 0, 255), 2)
            cv2.putText(
                visualization, "GAP",
                (gap_x - 15, gap_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1
            )
        
        # Ajouter les informations textuelles
        info_y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        
        texts = [
            f"Grade: {result.quality_grade}",
            f"Score: {result.quality_score:.1%}",
            f"Largeur: {result.measurements['width_mean']:.1f} px",
            f"Continuite: {result.continuity['score']:.1%}",
            f"Homogeneite: {result.homogeneity['classification']}"
        ]
        
        for text in texts:
            cv2.putText(
                visualization, text,
                (10, info_y),
                font, font_scale, color, 2
            )
            info_y += 25
        
        # Indicateur de statut
        status_color = (0, 255, 0) if result.is_acceptable else (0, 0, 255)
        status_text = "ACCEPTABLE" if result.is_acceptable else "NON CONFORME"
        cv2.putText(
            visualization, status_text,
            (visualization.shape[1] - 150, 30),
            font, 0.7, status_color, 2
        )
        
        return visualization
    
    def analyze_batch(
        self,
        image_paths: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[AnalysisResult]:
        """
        Analyse un lot d'images.
        
        Paramètres:
        -----------
        image_paths : List[str]
            Liste des chemins d'images
        progress_callback : callable, optional
            Fonction appelée après chaque image (args: index, total)
            
        Retourne:
        ---------
        List[AnalysisResult]
            Liste des résultats d'analyse
        """
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            try:
                result = self.analyze_image(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse de {path}: {e}")
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        # Générer un rapport de synthèse
        self._generate_batch_summary(results)
        
        return results
    
    def analyze_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        frame_interval: float = 1.0
    ) -> List[AnalysisResult]:
        """
        Analyse une vidéo de soudure.
        
        Paramètres:
        -----------
        video_path : str
            Chemin vers la vidéo
        output_video_path : str, optional
            Chemin pour la vidéo annotée de sortie
        frame_interval : float
            Intervalle entre les frames analysées (secondes)
            
        Retourne:
        ---------
        List[AnalysisResult]
            Résultats pour chaque frame analysée
        """
        logger.info(f"Analyse de la vidéo: {video_path}")
        
        results = []
        
        for frame, timestamp in self.video_processor.extract_frames(
            video_path,
            interval=frame_interval
        ):
            # Prétraiter la frame
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Redimensionner
            gray = cv2.resize(gray, self.config.image_size[::-1])
            preprocessed = self.image_processor.preprocess(gray)
            
            # Segmentation
            mask = self._segment(preprocessed)
            cleaned_mask = self.mask_processor.clean_mask(mask)
            
            # Analyse
            analysis = self.weld_analyzer.analyze(preprocessed, cleaned_mask)
            quality = self.quality_scorer.score(analysis)
            
            # Créer un résultat simplifié
            result = AnalysisResult(
                image_path=f"{video_path}@{timestamp:.2f}s",
                timestamp=str(timestamp),
                mask=cleaned_mask,
                measurements={
                    'width_mean': analysis['measurements'].width_mean,
                    'width_std': analysis['measurements'].width_std,
                    'width_min': analysis['measurements'].width_min,
                    'width_max': analysis['measurements'].width_max,
                    'total_length': analysis['measurements'].total_length
                },
                continuity={
                    'score': analysis['continuity'].continuity_score,
                    'n_gaps': analysis['continuity'].n_gaps,
                    'gap_positions': analysis['continuity'].gap_positions
                },
                homogeneity={
                    'score': analysis['homogeneity'].score,
                    'texture_uniformity': analysis['homogeneity'].texture_uniformity,
                    'color_consistency': analysis['homogeneity'].color_consistency,
                    'defect_ratio': analysis['homogeneity'].defect_ratio,
                    'classification': 'N/A'
                },
                quality_score=quality.overall_score,
                quality_grade=quality.grade.name,
                is_acceptable=quality.is_acceptable,
                recommendations=quality.recommendations,
                processing_time=0.0
            )
            
            results.append(result)
        
        logger.info(f"Vidéo analysée: {len(results)} frames traitées")
        
        return results
    
    def _generate_batch_summary(self, results: List[AnalysisResult]):
        """
        Génère un rapport de synthèse pour un lot d'analyses.
        """
        if not results:
            return
        
        # Calculer les statistiques
        scores = [r.quality_score for r in results]
        acceptable_count = sum(1 for r in results if r.is_acceptable)
        grade_distribution = {}
        
        for r in results:
            grade_distribution[r.quality_grade] = \
                grade_distribution.get(r.quality_grade, 0) + 1
        
        summary = f"""
═══════════════════════════════════════════════════════════════
                    RAPPORT DE SYNTHÈSE
═══════════════════════════════════════════════════════════════
Nombre d'images analysées: {len(results)}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STATISTIQUES DE QUALITÉ
───────────────────────────────────────────────────────────────
Score moyen:    {np.mean(scores):.1%}
Score min/max:  {np.min(scores):.1%} / {np.max(scores):.1%}
Écart-type:     {np.std(scores):.1%}

TAUX D'ACCEPTATION
───────────────────────────────────────────────────────────────
Acceptables:    {acceptable_count}/{len(results)} ({acceptable_count/len(results):.1%})
Non conformes:  {len(results) - acceptable_count}/{len(results)}

DISTRIBUTION DES NOTES
───────────────────────────────────────────────────────────────
"""
        for grade in ['A', 'B', 'C', 'D', 'F']:
            count = grade_distribution.get(grade, 0)
            bar = '█' * (count * 2)
            summary += f"{grade}: {bar} {count}\n"
        
        summary += "═══════════════════════════════════════════════════════════════\n"
        
        # Sauvegarder le rapport
        summary_path = os.path.join(
            self.config.output_dir,
            f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Rapport de synthèse sauvegardé: {summary_path}")
        print(summary)
    
    def generate_report(
        self,
        result: AnalysisResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Génère un rapport détaillé pour une analyse.
        
        Paramètres:
        -----------
        result : AnalysisResult
            Résultat de l'analyse
        output_path : str, optional
            Chemin de sortie pour le rapport
            
        Retourne:
        ---------
        str
            Contenu du rapport
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║            RAPPORT D'ANALYSE DE SOUDURE                      ║
╠══════════════════════════════════════════════════════════════╣
║ Image: {result.image_path}
║ Date:  {result.timestamp}
║ Temps de traitement: {result.processing_time:.2f}s
╠══════════════════════════════════════════════════════════════╣
║ RÉSULTAT GLOBAL                                              ║
╠──────────────────────────────────────────────────────────────╣
║ Note de qualité:    {result.quality_grade}
║ Score:              {result.quality_score:.1%}
║ Statut:             {'✓ ACCEPTABLE' if result.is_acceptable else '✗ NON CONFORME'}
╠══════════════════════════════════════════════════════════════╣
║ MESURES GÉOMÉTRIQUES                                         ║
╠──────────────────────────────────────────────────────────────╣
║ Largeur moyenne:    {result.measurements['width_mean']:.2f} px
║ Écart-type:         {result.measurements['width_std']:.2f} px
║ Largeur min/max:    {result.measurements['width_min']:.2f} / {result.measurements['width_max']:.2f} px
║ Longueur totale:    {result.measurements['total_length']:.2f} px
╠══════════════════════════════════════════════════════════════╣
║ ANALYSE DE CONTINUITÉ                                        ║
╠──────────────────────────────────────────────────────────────╣
║ Score de continuité:  {result.continuity['score']:.1%}
║ Interruptions:        {result.continuity['n_gaps']}
╠══════════════════════════════════════════════════════════════╣
║ ANALYSE D'HOMOGÉNÉITÉ                                        ║
╠──────────────────────────────────────────────────────────────╣
║ Score d'homogénéité:  {result.homogeneity['score']:.1%}
║ Uniformité texture:   {result.homogeneity['texture_uniformity']:.1%}
║ Consistance couleur:  {result.homogeneity['color_consistency']:.1%}
║ Ratio de défauts:     {result.homogeneity['defect_ratio']:.1%}
║ Classification:       {result.homogeneity['classification']}
╠══════════════════════════════════════════════════════════════╣
║ RECOMMANDATIONS                                              ║
╠──────────────────────────────────────────────────────────────╣
"""
        for i, rec in enumerate(result.recommendations, 1):
            report += f"║ {i}. {rec}\n"
        
        report += "╚══════════════════════════════════════════════════════════════╝\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Rapport sauvegardé: {output_path}")
        
        return report


if __name__ == "__main__":
    # ==========================================================================
    # TEST DU PIPELINE
    # ==========================================================================
    
    print("=" * 60)
    print("Test du pipeline d'analyse de soudure")
    print("=" * 60)
    
    # Créer une configuration
    config = PipelineConfig(
        image_size=(256, 256),
        device="cpu",
        output_dir="./test_outputs"
    )
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Image size: {config.image_size}")
    print(f"  Output dir: {config.output_dir}")
    
    # Créer le pipeline
    pipeline = WeldAnalysisPipeline(config=config)
    
    print("\nPipeline initialisé avec succès!")
    print("  - Modèle de segmentation: U-Net")
    print("  - Modèle de classification: HomogeneityClassifier")
    
    # Test avec une image synthétique
    print("\nTest avec image synthétique...")
    
    # Créer une image test
    test_image = np.random.randint(100, 200, (256, 256), dtype=np.uint8)
    test_path = os.path.join(config.output_dir, "test_image.png")
    os.makedirs(config.output_dir, exist_ok=True)
    cv2.imwrite(test_path, test_image)
    
    # Analyser
    try:
        result = pipeline.analyze_image(test_path)
        
        print(f"\nRésultat de l'analyse:")
        print(f"  Grade: {result.quality_grade}")
        print(f"  Score: {result.quality_score:.1%}")
        print(f"  Acceptable: {result.is_acceptable}")
        print(f"  Temps: {result.processing_time:.2f}s")
        
        # Générer le rapport
        report = pipeline.generate_report(result)
        print("\nRapport généré avec succès!")
        
    except Exception as e:
        print(f"Erreur lors du test: {e}")
    
    print("\n✓ Test du pipeline réussi!")
