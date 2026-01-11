# =============================================================================
# Analyseur de Zones de Soudure
# =============================================================================
# 
# Ce module implémente l'analyse complète des zones de soudure:
# - Mesure de la largeur du cordon de soudure
# - Évaluation de la continuité (détection des interruptions)
# - Analyse de l'homogénéité (uniformité de la texture)
#
# Ces métriques sont essentielles pour le contrôle qualité automatisé
# des soudures industrielles.
#
# =============================================================================

import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import measure, morphology
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch


@dataclass
class WeldMeasurement:
    """
    Structure de données pour les mesures de soudure.
    
    Attributs:
    ----------
    width_mean : float
        Largeur moyenne du cordon de soudure (en pixels)
    width_std : float
        Écart-type de la largeur (variabilité)
    width_min : float
        Largeur minimale détectée
    width_max : float
        Largeur maximale détectée
    continuity_score : float
        Score de continuité entre 0 et 1
    n_gaps : int
        Nombre d'interruptions détectées
    gap_positions : List[Tuple[int, int]]
        Positions des interruptions (y, x)
    total_length : float
        Longueur totale de la soudure en pixels
    """
    width_mean: float
    width_std: float
    width_min: float
    width_max: float
    continuity_score: float
    n_gaps: int
    gap_positions: List[Tuple[int, int]]
    total_length: float


@dataclass
class HomogeneityResult:
    """
    Structure de données pour les résultats d'homogénéité.
    
    Attributs:
    ----------
    score : float
        Score d'homogénéité global entre 0 et 1
    texture_uniformity : float
        Uniformité de la texture
    color_consistency : float
        Consistance des niveaux de gris
    defect_ratio : float
        Ratio de zones défectueuses
    local_scores : np.ndarray
        Carte des scores d'homogénéité locaux
    """
    score: float
    texture_uniformity: float
    color_consistency: float
    defect_ratio: float
    local_scores: np.ndarray


class WeldAnalyzer:
    """
    Analyseur complet de zones de soudure.
    
    Cette classe fournit des méthodes pour analyser les propriétés
    géométriques et de qualité des soudures à partir des masques
    de segmentation et des images originales.
    
    Paramètres:
    -----------
    pixels_per_mm : float
        Résolution de l'image (pixels par millimètre)
        Permet de convertir les mesures en unités réelles
    min_weld_width : int
        Largeur minimale attendue pour une soudure valide (pixels)
    max_gap_size : int
        Taille maximale d'une interruption acceptable (pixels)
        
    Exemple d'utilisation:
    ----------------------
    >>> analyzer = WeldAnalyzer(pixels_per_mm=10.0)
    >>> measurements = analyzer.measure_width(mask)
    >>> continuity = analyzer.analyze_continuity(mask)
    >>> homogeneity = analyzer.analyze_homogeneity(image, mask)
    """
    
    def __init__(
        self,
        pixels_per_mm: float = 1.0,
        min_weld_width: int = 5,
        max_gap_size: int = 10
    ):
        self.pixels_per_mm = pixels_per_mm
        self.min_weld_width = min_weld_width
        self.max_gap_size = max_gap_size
    
    def analyze(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Effectue une analyse complète de la zone de soudure.
        
        Combine toutes les analyses:
        - Mesures géométriques (largeur, longueur)
        - Analyse de continuité
        - Analyse d'homogénéité
        - Score de qualité global
        
        Paramètres:
        -----------
        image : np.ndarray
            Image originale de la soudure
        mask : np.ndarray
            Masque de segmentation binaire
            
        Retourne:
        ---------
        Dict[str, Any]
            Dictionnaire contenant tous les résultats d'analyse
        """
        # S'assurer que le masque est binaire
        mask = (mask > 0.5).astype(np.uint8)
        
        # Mesures géométriques
        measurements = self.measure_width(mask)
        
        # Analyse de continuité
        continuity = self.analyze_continuity(mask)
        
        # Analyse d'homogénéité
        homogeneity = self.analyze_homogeneity(image, mask)
        
        # Score de qualité global
        quality_score = self._compute_quality_score(
            measurements,
            continuity,
            homogeneity
        )
        
        return {
            'measurements': measurements,
            'continuity': continuity,
            'homogeneity': homogeneity,
            'quality_score': quality_score,
            'is_acceptable': quality_score >= 0.6  # Seuil d'acceptation
        }
    
    def measure_width(self, mask: np.ndarray) -> WeldMeasurement:
        """
        Mesure la largeur du cordon de soudure.
        
        La largeur est mesurée perpendiculairement à l'axe principal
        de la soudure, à intervalles réguliers le long du cordon.
        
        Algorithme:
        1. Détection de l'axe principal (squelettisation)
        2. Calcul de la direction locale de la soudure
        3. Mesure perpendiculaire à chaque point
        
        Paramètres:
        -----------
        mask : np.ndarray
            Masque de segmentation binaire
            
        Retourne:
        ---------
        WeldMeasurement
            Structure contenant les mesures de largeur
        """
        # Vérifier que le masque contient des données
        if np.sum(mask) == 0:
            return WeldMeasurement(
                width_mean=0.0,
                width_std=0.0,
                width_min=0.0,
                width_max=0.0,
                continuity_score=0.0,
                n_gaps=0,
                gap_positions=[],
                total_length=0.0
            )
        
        # Squelettisation: réduit la soudure à une ligne d'un pixel
        skeleton = morphology.skeletonize(mask > 0)
        
        # Récupérer les points du squelette
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) == 0:
            return WeldMeasurement(
                width_mean=0.0,
                width_std=0.0,
                width_min=0.0,
                width_max=0.0,
                continuity_score=0.0,
                n_gaps=0,
                gap_positions=[],
                total_length=0.0
            )
        
        # Calcul de la distance transform pour obtenir les largeurs
        # La distance transform donne la distance au bord le plus proche
        distance_map = ndimage.distance_transform_edt(mask)
        
        # La largeur à chaque point du squelette est 2x la distance au bord
        widths = []
        for y, x in skeleton_points:
            width = 2 * distance_map[y, x]
            widths.append(width)
        
        widths = np.array(widths)
        
        # Calcul de la longueur totale (approximée par le nombre de points)
        total_length = len(skeleton_points)
        
        # Analyse de continuité basique
        continuity_result = self.analyze_continuity(mask)
        
        return WeldMeasurement(
            width_mean=np.mean(widths),
            width_std=np.std(widths),
            width_min=np.min(widths),
            width_max=np.max(widths),
            continuity_score=continuity_result.continuity_score,
            n_gaps=continuity_result.n_gaps,
            gap_positions=continuity_result.gap_positions,
            total_length=total_length
        )
    
    def measure_width_profile(
        self,
        mask: np.ndarray,
        n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère un profil de largeur le long de la soudure.
        
        Utile pour visualiser les variations de largeur
        et détecter les zones problématiques.
        
        Paramètres:
        -----------
        mask : np.ndarray
            Masque de segmentation
        n_samples : int
            Nombre de points de mesure
            
        Retourne:
        ---------
        Tuple[np.ndarray, np.ndarray]
            (positions, largeurs) le long de la soudure
        """
        # Distance transform
        distance_map = ndimage.distance_transform_edt(mask)
        
        # Squelette pour les positions
        skeleton = morphology.skeletonize(mask > 0)
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) == 0:
            return np.array([]), np.array([])
        
        # Sous-échantillonner les points
        indices = np.linspace(0, len(skeleton_points) - 1, n_samples).astype(int)
        sampled_points = skeleton_points[indices]
        
        # Calculer les largeurs aux points échantillonnés
        positions = np.arange(n_samples)
        widths = np.array([
            2 * distance_map[y, x] for y, x in sampled_points
        ])
        
        return positions, widths
    
    def analyze_continuity(self, mask: np.ndarray) -> 'ContinuityResult':
        """
        Analyse la continuité de la soudure.
        
        Détecte les interruptions (gaps) dans le cordon de soudure
        qui peuvent indiquer des défauts de soudage.
        
        Algorithme:
        1. Labellisation des composantes connexes
        2. Identification des gaps entre composantes
        3. Calcul du score de continuité
        
        Paramètres:
        -----------
        mask : np.ndarray
            Masque de segmentation binaire
            
        Retourne:
        ---------
        ContinuityResult
            Résultats de l'analyse de continuité
        """
        # Composantes connexes
        labeled, n_labels = ndimage.label(mask)
        
        # Une seule composante = parfaite continuité
        if n_labels <= 1:
            return ContinuityResult(
                continuity_score=1.0 if np.sum(mask) > 0 else 0.0,
                n_gaps=0,
                gap_positions=[],
                component_sizes=[]
            )
        
        # Calculer la taille de chaque composante
        component_sizes = []
        for i in range(1, n_labels + 1):
            size = np.sum(labeled == i)
            component_sizes.append(size)
        
        # La composante principale est la plus grande
        main_component_size = max(component_sizes)
        total_size = sum(component_sizes)
        
        # Score de continuité basé sur la proportion de la composante principale
        continuity_score = main_component_size / total_size
        
        # Trouver les positions des gaps
        gap_positions = self._find_gaps(mask, labeled)
        
        return ContinuityResult(
            continuity_score=continuity_score,
            n_gaps=len(gap_positions),
            gap_positions=gap_positions,
            component_sizes=component_sizes
        )
    
    def _find_gaps(
        self,
        mask: np.ndarray,
        labeled: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Trouve les positions des interruptions dans la soudure.
        
        Utilise la fermeture morphologique pour identifier
        les zones où la soudure est interrompue.
        """
        # Fermeture morphologique pour combler les petits gaps
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.max_gap_size * 2, self.max_gap_size * 2)
        )
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Les gaps sont les zones ajoutées par la fermeture
        gaps = closed.astype(np.int32) - mask.astype(np.int32)
        gaps = (gaps > 0).astype(np.uint8)
        
        # Labelliser les gaps
        gap_labeled, n_gaps = ndimage.label(gaps)
        
        # Trouver le centre de chaque gap
        gap_positions = []
        for i in range(1, n_gaps + 1):
            gap_mask = gap_labeled == i
            coords = np.argwhere(gap_mask)
            if len(coords) > 0:
                center = coords.mean(axis=0).astype(int)
                gap_positions.append((int(center[0]), int(center[1])))
        
        return gap_positions
    
    def analyze_homogeneity(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> HomogeneityResult:
        """
        Analyse l'homogénéité de la zone de soudure.
        
        L'homogénéité évalue l'uniformité de la texture et de l'apparence
        du cordon de soudure. Une soudure homogène indique généralement
        une bonne qualité de soudage.
        
        Métriques analysées:
        1. Uniformité de texture (variance locale)
        2. Consistance des niveaux de gris
        3. Détection de défauts (anomalies locales)
        
        Paramètres:
        -----------
        image : np.ndarray
            Image originale (niveaux de gris de préférence)
        mask : np.ndarray
            Masque de segmentation
            
        Retourne:
        ---------
        HomogeneityResult
            Résultats de l'analyse d'homogénéité
        """
        # Assurer que l'image est en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Appliquer le masque pour isoler la zone de soudure
        mask_bool = mask > 0.5
        
        if np.sum(mask_bool) == 0:
            return HomogeneityResult(
                score=0.0,
                texture_uniformity=0.0,
                color_consistency=0.0,
                defect_ratio=1.0,
                local_scores=np.zeros_like(mask, dtype=np.float32)
            )
        
        # 1. Uniformité de texture via la variance locale
        texture_uniformity = self._compute_texture_uniformity(gray, mask_bool)
        
        # 2. Consistance des niveaux de gris
        color_consistency = self._compute_color_consistency(gray, mask_bool)
        
        # 3. Détection de défauts
        defect_ratio, local_scores = self._detect_defects(gray, mask_bool)
        
        # Score global (moyenne pondérée)
        score = (
            0.4 * texture_uniformity +
            0.3 * color_consistency +
            0.3 * (1 - defect_ratio)
        )
        
        return HomogeneityResult(
            score=score,
            texture_uniformity=texture_uniformity,
            color_consistency=color_consistency,
            defect_ratio=defect_ratio,
            local_scores=local_scores
        )
    
    def _compute_texture_uniformity(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Calcule l'uniformité de texture dans la zone de soudure.
        
        Utilise la variance locale pour évaluer l'uniformité.
        Une variance locale faible indique une texture uniforme.
        """
        # Calcul de la variance locale avec un noyau
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        # Moyenne locale
        local_mean = cv2.filter2D(
            image.astype(np.float32),
            -1,
            kernel
        )
        
        # Variance locale = E[X²] - E[X]²
        local_sq_mean = cv2.filter2D(
            (image.astype(np.float32) ** 2),
            -1,
            kernel
        )
        local_variance = local_sq_mean - local_mean ** 2
        local_variance = np.maximum(local_variance, 0)
        
        # Moyenne de la variance dans la zone de soudure
        weld_variance = local_variance[mask]
        
        if len(weld_variance) == 0:
            return 0.0
        
        # Normaliser: variance élevée = faible uniformité
        mean_variance = np.mean(weld_variance)
        # Convertir en score (0-1), où 1 = parfaitement uniforme
        uniformity = 1.0 / (1.0 + mean_variance / 100)
        
        return float(np.clip(uniformity, 0, 1))
    
    def _compute_color_consistency(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Calcule la consistance des niveaux de gris.
        
        Une soudure de bonne qualité devrait avoir des niveaux
        de gris relativement uniformes.
        """
        # Extraire les pixels de la soudure
        weld_pixels = image[mask]
        
        if len(weld_pixels) == 0:
            return 0.0
        
        # Calculer les statistiques
        mean_intensity = np.mean(weld_pixels)
        std_intensity = np.std(weld_pixels)
        
        # Coefficient de variation normalisé
        cv = std_intensity / (mean_intensity + 1e-6)
        
        # Convertir en score (0-1)
        # CV faible = haute consistance
        consistency = 1.0 / (1.0 + cv * 2)
        
        return float(np.clip(consistency, 0, 1))
    
    def _detect_defects(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Détecte les défauts locaux dans la zone de soudure.
        
        Utilise une approche basée sur les anomalies:
        - Zones significativement plus sombres (porosités, inclusions)
        - Zones significativement plus claires (projections)
        - Zones à texture irrégulière (fissures)
        
        Retourne:
        ---------
        Tuple[float, np.ndarray]
            (ratio de défauts, carte des scores locaux)
        """
        # Calculer les statistiques de la soudure
        weld_pixels = image[mask]
        
        if len(weld_pixels) == 0:
            return 1.0, np.zeros_like(image, dtype=np.float32)
        
        mean_intensity = np.mean(weld_pixels)
        std_intensity = np.std(weld_pixels) + 1e-6
        
        # Créer une carte des anomalies (z-score)
        z_scores = np.abs(image.astype(np.float32) - mean_intensity) / std_intensity
        
        # Appliquer le masque
        z_scores_masked = z_scores * mask
        
        # Seuil pour considérer une anomalie (z-score > 2.5)
        anomaly_threshold = 2.5
        anomalies = (z_scores_masked > anomaly_threshold).astype(np.float32)
        
        # Filtrer les petites anomalies (bruit)
        kernel = np.ones((3, 3), np.uint8)
        anomalies = cv2.morphologyEx(
            (anomalies * 255).astype(np.uint8),
            cv2.MORPH_OPEN,
            kernel
        )
        
        # Calculer le ratio de défauts
        n_anomaly_pixels = np.sum(anomalies > 0)
        n_weld_pixels = np.sum(mask)
        
        defect_ratio = n_anomaly_pixels / n_weld_pixels if n_weld_pixels > 0 else 1.0
        
        # Carte des scores locaux (inverse des z-scores normalisés)
        local_scores = 1.0 / (1.0 + z_scores_masked / 3.0)
        local_scores = local_scores * mask
        
        return float(defect_ratio), local_scores.astype(np.float32)
    
    def _compute_quality_score(
        self,
        measurements: WeldMeasurement,
        continuity: 'ContinuityResult',
        homogeneity: HomogeneityResult
    ) -> float:
        """
        Calcule le score de qualité global.
        
        Combine les différentes métriques en un score unique
        entre 0 (très mauvais) et 1 (excellent).
        """
        # Score de largeur: pénaliser la variabilité excessive
        width_cv = measurements.width_std / (measurements.width_mean + 1e-6)
        width_score = 1.0 / (1.0 + width_cv)
        
        # Score de continuité
        continuity_score = continuity.continuity_score
        
        # Score d'homogénéité
        homogeneity_score = homogeneity.score
        
        # Pondération des scores
        # La continuité est critique, l'homogénéité importante
        quality_score = (
            0.2 * width_score +
            0.4 * continuity_score +
            0.4 * homogeneity_score
        )
        
        return float(np.clip(quality_score, 0, 1))
    
    def generate_report(
        self,
        analysis_result: Dict[str, Any]
    ) -> str:
        """
        Génère un rapport textuel de l'analyse.
        
        Paramètres:
        -----------
        analysis_result : Dict
            Résultat de la méthode analyze()
            
        Retourne:
        ---------
        str
            Rapport formaté
        """
        measurements = analysis_result['measurements']
        continuity = analysis_result['continuity']
        homogeneity = analysis_result['homogeneity']
        quality_score = analysis_result['quality_score']
        is_acceptable = analysis_result['is_acceptable']
        
        # Conversion en mm si le facteur est défini
        width_mean_mm = measurements.width_mean / self.pixels_per_mm
        width_std_mm = measurements.width_std / self.pixels_per_mm
        length_mm = measurements.total_length / self.pixels_per_mm
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║               RAPPORT D'ANALYSE DE SOUDURE                  ║
╠══════════════════════════════════════════════════════════════╣
║ RÉSULTAT GLOBAL: {'ACCEPTABLE ✓' if is_acceptable else 'NON CONFORME ✗'}
║ Score de qualité: {quality_score:.1%}
╠══════════════════════════════════════════════════════════════╣
║ MESURES GÉOMÉTRIQUES                                        ║
╠──────────────────────────────────────────────────────────────╣
║ Largeur moyenne:    {width_mean_mm:.2f} mm
║ Écart-type largeur: {width_std_mm:.2f} mm
║ Largeur min/max:    {measurements.width_min/self.pixels_per_mm:.2f} / {measurements.width_max/self.pixels_per_mm:.2f} mm
║ Longueur totale:    {length_mm:.2f} mm
╠══════════════════════════════════════════════════════════════╣
║ ANALYSE DE CONTINUITÉ                                       ║
╠──────────────────────────────────────────────────────────────╣
║ Score de continuité: {continuity.continuity_score:.1%}
║ Nombre d'interruptions: {continuity.n_gaps}
╠══════════════════════════════════════════════════════════════╣
║ ANALYSE D'HOMOGÉNÉITÉ                                       ║
╠──────────────────────────────────────────────────────────────╣
║ Score d'homogénéité:    {homogeneity.score:.1%}
║ Uniformité de texture:  {homogeneity.texture_uniformity:.1%}
║ Consistance couleur:    {homogeneity.color_consistency:.1%}
║ Ratio de défauts:       {homogeneity.defect_ratio:.1%}
╚══════════════════════════════════════════════════════════════╝
"""
        return report


@dataclass
class ContinuityResult:
    """
    Structure de données pour les résultats de continuité.
    
    Attributs:
    ----------
    continuity_score : float
        Score de continuité entre 0 et 1
    n_gaps : int
        Nombre d'interruptions détectées
    gap_positions : List[Tuple[int, int]]
        Positions des interruptions
    component_sizes : List[int]
        Tailles des composantes connexes
    """
    continuity_score: float
    n_gaps: int
    gap_positions: List[Tuple[int, int]]
    component_sizes: List[int]


class WeldProfileAnalyzer:
    """
    Analyseur de profil de soudure.
    
    Analyse le profil transversal du cordon de soudure pour
    évaluer sa forme (convexe, concave, plate) et détecter
    des anomalies géométriques.
    """
    
    def __init__(self):
        pass
    
    def extract_profile(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        position: int
    ) -> np.ndarray:
        """
        Extrait le profil d'intensité transversal à une position donnée.
        
        Paramètres:
        -----------
        image : np.ndarray
            Image de la soudure
        mask : np.ndarray
            Masque de segmentation
        position : int
            Position le long de la soudure (coordonnée y)
            
        Retourne:
        ---------
        np.ndarray
            Profil d'intensité perpendiculaire à la soudure
        """
        # Extraire la ligne à la position donnée
        row = image[position, :]
        mask_row = mask[position, :]
        
        # Trouver les limites de la soudure
        weld_indices = np.where(mask_row > 0)[0]
        
        if len(weld_indices) == 0:
            return np.array([])
        
        start = weld_indices[0]
        end = weld_indices[-1] + 1
        
        # Extraire le profil
        profile = row[start:end]
        
        return profile
    
    def analyze_profile_shape(
        self,
        profile: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyse la forme du profil.
        
        Détermine si le profil est:
        - Convexe (bombé): forme normale d'une bonne soudure
        - Concave (creux): peut indiquer un manque de métal
        - Plat: soudure de surface
        
        Paramètres:
        -----------
        profile : np.ndarray
            Profil d'intensité
            
        Retourne:
        ---------
        Dict[str, Any]
            Analyse de la forme incluant le type et les métriques
        """
        if len(profile) < 3:
            return {'type': 'unknown', 'curvature': 0.0}
        
        # Normaliser le profil
        profile_norm = profile - profile.min()
        profile_norm = profile_norm / (profile_norm.max() + 1e-6)
        
        # Ajuster une parabole: y = ax² + bx + c
        x = np.arange(len(profile))
        x_centered = x - len(x) / 2
        
        # Régression polynomiale de degré 2
        coeffs = np.polyfit(x_centered, profile_norm, 2)
        a = coeffs[0]  # Coefficient de x²
        
        # Déterminer le type de profil
        if a > 0.01:
            profile_type = 'convex'
        elif a < -0.01:
            profile_type = 'concave'
        else:
            profile_type = 'flat'
        
        # Calculer la courbure (approximation)
        curvature = abs(a)
        
        return {
            'type': profile_type,
            'curvature': float(curvature),
            'coefficients': coeffs.tolist()
        }


if __name__ == "__main__":
    # ==========================================================================
    # TEST DE L'ANALYSEUR DE SOUDURE
    # ==========================================================================
    
    print("=" * 60)
    print("Test de l'analyseur de soudure")
    print("=" * 60)
    
    # Créer une image et un masque de test
    image = np.random.randint(100, 200, (256, 256), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Dessiner une soudure simulée (bande horizontale)
    cv2.rectangle(mask, (20, 100), (236, 150), 255, -1)
    
    # Ajouter une petite interruption
    cv2.rectangle(mask, (100, 100), (110, 150), 0, -1)
    
    print(f"\nImage test: {image.shape}")
    print(f"Masque test: {mask.shape}")
    print(f"Couverture de soudure: {np.mean(mask > 0) * 100:.1f}%")
    
    # Créer l'analyseur
    analyzer = WeldAnalyzer(pixels_per_mm=10.0)
    
    # Effectuer l'analyse complète
    results = analyzer.analyze(image, mask)
    
    # Afficher le rapport
    report = analyzer.generate_report(results)
    print(report)
    
    # Test du profil de soudure
    profile_analyzer = WeldProfileAnalyzer()
    
    profile = profile_analyzer.extract_profile(image, mask, position=125)
    if len(profile) > 0:
        shape_analysis = profile_analyzer.analyze_profile_shape(profile)
        print(f"\nAnalyse du profil:")
        print(f"  Type: {shape_analysis['type']}")
        print(f"  Courbure: {shape_analysis['curvature']:.4f}")
    
    print("\n✓ Test de l'analyseur réussi!")
