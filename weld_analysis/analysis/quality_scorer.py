# =============================================================================
# Système de Notation de Qualité des Soudures
# =============================================================================
# 
# Ce module implémente un système de notation de qualité pour les soudures
# basé sur des critères industriels standards:
# - Notation alphabétique (A, B, C, D, F)
# - Score numérique (0-100)
# - Classification selon les normes industrielles
#
# Les critères de qualité sont configurables selon les exigences
# spécifiques de chaque application industrielle.
#
# =============================================================================

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class QualityGrade(Enum):
    """
    Notes de qualité selon une échelle standard.
    
    A: Excellent - Aucun défaut visible
    B: Bon - Défauts mineurs acceptables
    C: Acceptable - Défauts modérés, à surveiller
    D: Marginal - Défauts significatifs, réparation possible
    F: Inacceptable - Défauts majeurs, rejet
    """
    A = "Excellent"
    B = "Bon"
    C = "Acceptable"
    D = "Marginal"
    F = "Inacceptable"


@dataclass
class QualityCriteria:
    """
    Critères de qualité configurables.
    
    Ces seuils définissent les limites acceptables pour chaque
    métrique de qualité. Ils peuvent être ajustés selon les
    exigences spécifiques de l'application.
    
    Attributs:
    ----------
    width_tolerance : float
        Tolérance sur la variation de largeur (coefficient de variation)
    min_continuity : float
        Score de continuité minimum acceptable
    min_homogeneity : float
        Score d'homogénéité minimum acceptable
    max_defect_ratio : float
        Ratio maximum de zones défectueuses
    max_gaps : int
        Nombre maximum d'interruptions acceptables
    """
    width_tolerance: float = 0.2      # CV max de 20%
    min_continuity: float = 0.85      # 85% minimum
    min_homogeneity: float = 0.70     # 70% minimum
    max_defect_ratio: float = 0.05    # 5% max de défauts
    max_gaps: int = 2                  # 2 interruptions max
    
    # Seuils pour les notes
    grade_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'A': 0.90,  # 90% et plus
        'B': 0.80,  # 80-89%
        'C': 0.70,  # 70-79%
        'D': 0.60,  # 60-69%
        'F': 0.00   # Moins de 60%
    })


@dataclass
class QualityScore:
    """
    Structure de données pour le score de qualité complet.
    
    Attributs:
    ----------
    overall_score : float
        Score global entre 0 et 1
    grade : QualityGrade
        Note alphabétique
    sub_scores : Dict[str, float]
        Scores détaillés par catégorie
    passed_criteria : Dict[str, bool]
        Critères passés/échoués
    recommendations : List[str]
        Recommandations d'amélioration
    is_acceptable : bool
        Si la soudure est globalement acceptable
    """
    overall_score: float
    grade: QualityGrade
    sub_scores: Dict[str, float]
    passed_criteria: Dict[str, bool]
    recommendations: List[str]
    is_acceptable: bool


class QualityScorer:
    """
    Système de notation de qualité pour les soudures.
    
    Cette classe calcule un score de qualité basé sur les mesures
    d'analyse de soudure et les compare aux critères définis.
    
    Paramètres:
    -----------
    criteria : QualityCriteria, optional
        Critères de qualité à utiliser
        Si non fourni, utilise les critères par défaut
    weights : Dict[str, float], optional
        Pondération des différentes catégories de score
        
    Exemple d'utilisation:
    ----------------------
    >>> scorer = QualityScorer()
    >>> quality = scorer.score(analysis_result)
    >>> print(f"Note: {quality.grade.value}")
    >>> print(f"Score: {quality.overall_score:.1%}")
    """
    
    def __init__(
        self,
        criteria: Optional[QualityCriteria] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.criteria = criteria or QualityCriteria()
        
        # Pondération par défaut des catégories
        self.weights = weights or {
            'geometry': 0.25,      # Mesures géométriques
            'continuity': 0.30,   # Continuité du cordon
            'homogeneity': 0.30,  # Homogénéité de la texture
            'defects': 0.15       # Absence de défauts
        }
    
    def score(self, analysis_result: Dict[str, Any]) -> QualityScore:
        """
        Calcule le score de qualité à partir des résultats d'analyse.
        
        Paramètres:
        -----------
        analysis_result : Dict[str, Any]
            Résultat de WeldAnalyzer.analyze()
            
        Retourne:
        ---------
        QualityScore
            Score de qualité complet
        """
        # Extraire les composants de l'analyse
        measurements = analysis_result['measurements']
        continuity = analysis_result['continuity']
        homogeneity = analysis_result['homogeneity']
        
        # Calculer les sous-scores
        sub_scores = self._compute_sub_scores(
            measurements,
            continuity,
            homogeneity
        )
        
        # Vérifier les critères
        passed_criteria = self._check_criteria(
            measurements,
            continuity,
            homogeneity
        )
        
        # Calculer le score global
        overall_score = self._compute_overall_score(sub_scores)
        
        # Déterminer la note
        grade = self._determine_grade(overall_score)
        
        # Générer les recommandations
        recommendations = self._generate_recommendations(
            passed_criteria,
            sub_scores
        )
        
        # Déterminer si acceptable
        is_acceptable = grade in [QualityGrade.A, QualityGrade.B, QualityGrade.C]
        
        return QualityScore(
            overall_score=overall_score,
            grade=grade,
            sub_scores=sub_scores,
            passed_criteria=passed_criteria,
            recommendations=recommendations,
            is_acceptable=is_acceptable
        )
    
    def _compute_sub_scores(
        self,
        measurements: Any,
        continuity: Any,
        homogeneity: Any
    ) -> Dict[str, float]:
        """
        Calcule les sous-scores pour chaque catégorie.
        """
        # Score de géométrie basé sur la variabilité de la largeur
        width_cv = measurements.width_std / (measurements.width_mean + 1e-6)
        geometry_score = 1.0 - min(width_cv / self.criteria.width_tolerance, 1.0)
        
        # Score de continuité (directement depuis l'analyse)
        continuity_score = continuity.continuity_score
        
        # Score d'homogénéité
        homogeneity_score = homogeneity.score
        
        # Score de défauts (inverse du ratio)
        defects_score = 1.0 - min(
            homogeneity.defect_ratio / self.criteria.max_defect_ratio,
            1.0
        )
        
        return {
            'geometry': float(np.clip(geometry_score, 0, 1)),
            'continuity': float(np.clip(continuity_score, 0, 1)),
            'homogeneity': float(np.clip(homogeneity_score, 0, 1)),
            'defects': float(np.clip(defects_score, 0, 1))
        }
    
    def _check_criteria(
        self,
        measurements: Any,
        continuity: Any,
        homogeneity: Any
    ) -> Dict[str, bool]:
        """
        Vérifie si chaque critère de qualité est respecté.
        """
        # Variabilité de la largeur
        width_cv = measurements.width_std / (measurements.width_mean + 1e-6)
        width_ok = width_cv <= self.criteria.width_tolerance
        
        # Continuité
        continuity_ok = continuity.continuity_score >= self.criteria.min_continuity
        
        # Nombre d'interruptions
        gaps_ok = continuity.n_gaps <= self.criteria.max_gaps
        
        # Homogénéité
        homogeneity_ok = homogeneity.score >= self.criteria.min_homogeneity
        
        # Ratio de défauts
        defects_ok = homogeneity.defect_ratio <= self.criteria.max_defect_ratio
        
        return {
            'width_variation': width_ok,
            'continuity': continuity_ok,
            'gaps': gaps_ok,
            'homogeneity': homogeneity_ok,
            'defects': defects_ok
        }
    
    def _compute_overall_score(self, sub_scores: Dict[str, float]) -> float:
        """
        Calcule le score global pondéré.
        """
        total_weight = sum(self.weights.values())
        
        weighted_sum = sum(
            sub_scores[category] * weight
            for category, weight in self.weights.items()
            if category in sub_scores
        )
        
        return weighted_sum / total_weight
    
    def _determine_grade(self, score: float) -> QualityGrade:
        """
        Détermine la note alphabétique basée sur le score.
        """
        thresholds = self.criteria.grade_thresholds
        
        if score >= thresholds['A']:
            return QualityGrade.A
        elif score >= thresholds['B']:
            return QualityGrade.B
        elif score >= thresholds['C']:
            return QualityGrade.C
        elif score >= thresholds['D']:
            return QualityGrade.D
        else:
            return QualityGrade.F
    
    def _generate_recommendations(
        self,
        passed_criteria: Dict[str, bool],
        sub_scores: Dict[str, float]
    ) -> List[str]:
        """
        Génère des recommandations basées sur les critères échoués.
        """
        recommendations = []
        
        # Recommandations basées sur les critères échoués
        if not passed_criteria.get('width_variation', True):
            recommendations.append(
                "Améliorer la régularité de la largeur du cordon. "
                "Vérifier la vitesse de soudage et l'apport de matière."
            )
        
        if not passed_criteria.get('continuity', True):
            recommendations.append(
                "Améliorer la continuité du cordon de soudure. "
                "Vérifier les paramètres d'arc et la préparation des surfaces."
            )
        
        if not passed_criteria.get('gaps', True):
            recommendations.append(
                "Réduire le nombre d'interruptions dans la soudure. "
                "Vérifier l'alimentation en fil et la stabilité de l'arc."
            )
        
        if not passed_criteria.get('homogeneity', True):
            recommendations.append(
                "Améliorer l'homogénéité de la texture. "
                "Vérifier la température de préchauffage et la vitesse de refroidissement."
            )
        
        if not passed_criteria.get('defects', True):
            recommendations.append(
                "Réduire les défauts détectés. "
                "Vérifier la propreté des surfaces et les paramètres de protection gazeuse."
            )
        
        # Recommandations basées sur les sous-scores faibles
        if sub_scores.get('geometry', 1.0) < 0.7:
            if not any('largeur' in r for r in recommendations):
                recommendations.append(
                    "Considérer un contrôle plus strict des paramètres géométriques."
                )
        
        if not recommendations:
            recommendations.append(
                "Qualité satisfaisante. Maintenir les paramètres actuels."
            )
        
        return recommendations
    
    def get_detailed_report(self, quality: QualityScore) -> str:
        """
        Génère un rapport détaillé du score de qualité.
        
        Paramètres:
        -----------
        quality : QualityScore
            Score de qualité calculé
            
        Retourne:
        ---------
        str
            Rapport formaté
        """
        # Symboles pour les critères
        check = "✓"
        cross = "✗"
        
        # Construire le rapport
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                  RAPPORT DE QUALITÉ                          ║
╠══════════════════════════════════════════════════════════════╣
║ NOTE GLOBALE: {quality.grade.name} ({quality.grade.value})
║ Score: {quality.overall_score:.0%}
║ Statut: {'CONFORME' if quality.is_acceptable else 'NON CONFORME'}
╠══════════════════════════════════════════════════════════════╣
║ SCORES DÉTAILLÉS                                            ║
╠──────────────────────────────────────────────────────────────╣
║ Géométrie:    {'█' * int(quality.sub_scores['geometry'] * 10):10s} {quality.sub_scores['geometry']:.0%}
║ Continuité:   {'█' * int(quality.sub_scores['continuity'] * 10):10s} {quality.sub_scores['continuity']:.0%}
║ Homogénéité:  {'█' * int(quality.sub_scores['homogeneity'] * 10):10s} {quality.sub_scores['homogeneity']:.0%}
║ Défauts:      {'█' * int(quality.sub_scores['defects'] * 10):10s} {quality.sub_scores['defects']:.0%}
╠══════════════════════════════════════════════════════════════╣
║ CRITÈRES                                                     ║
╠──────────────────────────────────────────────────────────────╣"""
        
        for criterion, passed in quality.passed_criteria.items():
            symbol = check if passed else cross
            status = "OK" if passed else "ÉCHEC"
            report += f"\n║ {symbol} {criterion.replace('_', ' ').title():20s} {status}"
        
        report += """
╠══════════════════════════════════════════════════════════════╣
║ RECOMMANDATIONS                                              ║
╠──────────────────────────────────────────────────────────────╣"""
        
        for i, rec in enumerate(quality.recommendations, 1):
            # Découper les longues recommandations
            words = rec.split()
            lines = []
            current_line = f"{i}. "
            
            for word in words:
                if len(current_line) + len(word) < 55:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = "   " + word + " "
            lines.append(current_line)
            
            for line in lines:
                report += f"\n║ {line}"
        
        report += """
╚══════════════════════════════════════════════════════════════╝"""
        
        return report
    
    def compare_scores(
        self,
        scores: List[QualityScore]
    ) -> Dict[str, Any]:
        """
        Compare plusieurs scores de qualité.
        
        Utile pour l'analyse de tendances ou la comparaison
        de différentes soudures.
        
        Paramètres:
        -----------
        scores : List[QualityScore]
            Liste de scores à comparer
            
        Retourne:
        ---------
        Dict[str, Any]
            Statistiques comparatives
        """
        if not scores:
            return {}
        
        overall_scores = [s.overall_score for s in scores]
        
        # Distribution des notes
        grade_distribution = {}
        for grade in QualityGrade:
            grade_distribution[grade.name] = sum(
                1 for s in scores if s.grade == grade
            )
        
        # Taux de conformité
        n_acceptable = sum(1 for s in scores if s.is_acceptable)
        acceptance_rate = n_acceptable / len(scores)
        
        # Scores moyens par catégorie
        avg_sub_scores = {}
        for category in scores[0].sub_scores.keys():
            avg_sub_scores[category] = np.mean([
                s.sub_scores[category] for s in scores
            ])
        
        return {
            'n_samples': len(scores),
            'average_score': np.mean(overall_scores),
            'std_score': np.std(overall_scores),
            'min_score': np.min(overall_scores),
            'max_score': np.max(overall_scores),
            'acceptance_rate': acceptance_rate,
            'grade_distribution': grade_distribution,
            'average_sub_scores': avg_sub_scores
        }


class StatisticalQualityControl:
    """
    Contrôle statistique de la qualité (SPC).
    
    Implémente des outils de contrôle statistique pour
    surveiller la qualité des soudures dans le temps.
    """
    
    def __init__(self, target_score: float = 0.85):
        """
        Paramètres:
        -----------
        target_score : float
            Score de qualité cible
        """
        self.target_score = target_score
        self.history: List[float] = []
    
    def add_sample(self, score: float) -> None:
        """Ajoute un échantillon à l'historique."""
        self.history.append(score)
    
    def get_control_limits(
        self,
        n_sigma: float = 3.0
    ) -> Tuple[float, float]:
        """
        Calcule les limites de contrôle.
        
        Paramètres:
        -----------
        n_sigma : float
            Nombre d'écarts-types pour les limites (défaut: 3)
            
        Retourne:
        ---------
        Tuple[float, float]
            (limite inférieure, limite supérieure)
        """
        if len(self.history) < 2:
            return 0.0, 1.0
        
        mean = np.mean(self.history)
        std = np.std(self.history)
        
        lcl = max(0, mean - n_sigma * std)
        ucl = min(1, mean + n_sigma * std)
        
        return lcl, ucl
    
    def is_in_control(self, score: float) -> bool:
        """
        Vérifie si un score est dans les limites de contrôle.
        """
        lcl, ucl = self.get_control_limits()
        return lcl <= score <= ucl
    
    def detect_trend(self, window: int = 7) -> Optional[str]:
        """
        Détecte une tendance dans les scores récents.
        
        Paramètres:
        -----------
        window : int
            Taille de la fenêtre d'analyse
            
        Retourne:
        ---------
        Optional[str]
            "improving", "degrading", ou None si pas de tendance
        """
        if len(self.history) < window:
            return None
        
        recent = self.history[-window:]
        
        # Test de tendance monotone
        increasing = all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
        decreasing = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))
        
        if increasing:
            return "improving"
        elif decreasing:
            return "degrading"
        else:
            return None


if __name__ == "__main__":
    # ==========================================================================
    # TEST DU SYSTÈME DE NOTATION
    # ==========================================================================
    
    print("=" * 60)
    print("Test du système de notation de qualité")
    print("=" * 60)
    
    # Créer des résultats d'analyse simulés
    from .weld_analyzer import WeldMeasurement, ContinuityResult, HomogeneityResult
    
    # Soudure de bonne qualité
    good_measurements = WeldMeasurement(
        width_mean=50.0,
        width_std=5.0,
        width_min=40.0,
        width_max=60.0,
        continuity_score=0.95,
        n_gaps=0,
        gap_positions=[],
        total_length=500.0
    )
    
    good_continuity = ContinuityResult(
        continuity_score=0.95,
        n_gaps=0,
        gap_positions=[],
        component_sizes=[10000]
    )
    
    good_homogeneity = HomogeneityResult(
        score=0.88,
        texture_uniformity=0.90,
        color_consistency=0.85,
        defect_ratio=0.02,
        local_scores=np.ones((100, 100))
    )
    
    good_analysis = {
        'measurements': good_measurements,
        'continuity': good_continuity,
        'homogeneity': good_homogeneity
    }
    
    # Créer le scorer
    scorer = QualityScorer()
    
    # Calculer le score
    quality = scorer.score(good_analysis)
    
    print(f"\nSoudure de bonne qualité:")
    print(f"  Note: {quality.grade.name} ({quality.grade.value})")
    print(f"  Score: {quality.overall_score:.1%}")
    print(f"  Acceptable: {quality.is_acceptable}")
    
    # Afficher le rapport détaillé
    report = scorer.get_detailed_report(quality)
    print(report)
    
    # Test du contrôle statistique
    print("\n" + "-" * 40)
    print("Test du contrôle statistique")
    
    spc = StatisticalQualityControl(target_score=0.85)
    
    # Ajouter des échantillons
    scores = [0.82, 0.85, 0.88, 0.84, 0.86, 0.83, 0.87, 0.85, 0.84, 0.86]
    for s in scores:
        spc.add_sample(s)
    
    lcl, ucl = spc.get_control_limits()
    print(f"Limites de contrôle: [{lcl:.2f}, {ucl:.2f}]")
    
    trend = spc.detect_trend()
    print(f"Tendance détectée: {trend or 'aucune'}")
    
    print("\n✓ Test du système de notation réussi!")
