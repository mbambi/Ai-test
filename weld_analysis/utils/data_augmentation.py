# =============================================================================
# Augmentation de Données pour l'Entraînement
# =============================================================================
# 
# Ce module fournit des transformations d'augmentation de données
# spécifiquement conçues pour les images de soudure:
# - Transformations géométriques (rotation, flip, échelle)
# - Transformations photométriques (contraste, luminosité, bruit)
# - Transformations spécifiques aux images industrielles
#
# L'augmentation de données est cruciale pour:
# - Augmenter la taille effective du dataset
# - Améliorer la généralisation du modèle
# - Rendre le modèle robuste aux variations d'éclairage et de position
#
# =============================================================================

import numpy as np
import cv2
from typing import Tuple, Optional, Callable, List, Dict, Any
import random


class DataAugmentation:
    """
    Classe pour l'augmentation de données des images de soudure.
    
    Cette classe applique des transformations aléatoires aux images
    et masques d'entraînement pour augmenter la diversité des données.
    
    Paramètres:
    -----------
    p_geometric : float
        Probabilité d'appliquer des transformations géométriques (défaut: 0.5)
    p_photometric : float
        Probabilité d'appliquer des transformations photométriques (défaut: 0.5)
    seed : int, optional
        Graine pour la reproductibilité
        
    Exemple d'utilisation:
    ----------------------
    >>> augmenter = DataAugmentation(p_geometric=0.5, p_photometric=0.5)
    >>> augmented_image, augmented_mask = augmenter(image, mask)
    """
    
    def __init__(
        self,
        p_geometric: float = 0.5,
        p_photometric: float = 0.5,
        seed: Optional[int] = None
    ):
        self.p_geometric = p_geometric
        self.p_photometric = p_photometric
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Applique les augmentations à une image et son masque.
        
        Paramètres:
        -----------
        image : np.ndarray
            Image à augmenter
        mask : np.ndarray, optional
            Masque de segmentation correspondant
            
        Retourne:
        ---------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (image augmentée, masque augmenté)
        """
        # Appliquer les transformations géométriques
        if random.random() < self.p_geometric:
            image, mask = self._apply_geometric(image, mask)
        
        # Appliquer les transformations photométriques (image seulement)
        if random.random() < self.p_photometric:
            image = self._apply_photometric(image)
        
        return image, mask
    
    def _apply_geometric(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Applique une transformation géométrique aléatoire.
        
        Transformations possibles:
        - Flip horizontal
        - Flip vertical
        - Rotation (0°, 90°, 180°, 270°)
        - Zoom/Scale
        - Translation légère
        """
        # Liste des transformations à appliquer
        transforms = []
        
        # Flip horizontal (50% de chance)
        if random.random() < 0.5:
            transforms.append(self._flip_horizontal)
        
        # Flip vertical (30% de chance - moins fréquent)
        if random.random() < 0.3:
            transforms.append(self._flip_vertical)
        
        # Rotation par multiples de 90° (40% de chance)
        if random.random() < 0.4:
            transforms.append(self._rotate_90)
        
        # Rotation légère (-15° à +15°) (30% de chance)
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            transforms.append(lambda img, msk: self._rotate(img, msk, angle))
        
        # Zoom (20% de chance)
        if random.random() < 0.2:
            scale = random.uniform(0.9, 1.1)
            transforms.append(lambda img, msk: self._scale(img, msk, scale))
        
        # Appliquer les transformations
        for transform in transforms:
            image, mask = transform(image, mask)
        
        return image, mask
    
    def _flip_horizontal(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Retourne l'image horizontalement.
        
        Cette transformation est valide pour les soudures car
        elles sont généralement symétriques horizontalement.
        """
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)
        return image, mask
    
    def _flip_vertical(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Retourne l'image verticalement.
        
        À utiliser avec précaution car certaines soudures
        peuvent avoir une orientation privilégiée.
        """
        image = np.flipud(image)
        if mask is not None:
            mask = np.flipud(mask)
        return image, mask
    
    def _rotate_90(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Rotation par multiples de 90 degrés.
        
        Choisit aléatoirement entre 90°, 180° ou 270°.
        """
        k = random.choice([1, 2, 3])  # Nombre de rotations de 90°
        image = np.rot90(image, k)
        if mask is not None:
            mask = np.rot90(mask, k)
        return image, mask
    
    def _rotate(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray],
        angle: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Rotation arbitraire autour du centre.
        
        Paramètres:
        -----------
        angle : float
            Angle de rotation en degrés
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Matrice de rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Appliquer à l'image
        image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (w, h),
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # Appliquer au masque
        if mask is not None:
            mask = cv2.warpAffine(
                mask, 
                rotation_matrix, 
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        
        return image, mask
    
    def _scale(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray],
        scale: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Zoom/dézoom centré.
        
        Paramètres:
        -----------
        scale : float
            Facteur d'échelle (>1 = zoom, <1 = dézoom)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Matrice de transformation (scale centré)
        transform_matrix = cv2.getRotationMatrix2D(center, 0, scale)
        
        # Appliquer à l'image
        image = cv2.warpAffine(
            image, 
            transform_matrix, 
            (w, h),
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # Appliquer au masque
        if mask is not None:
            mask = cv2.warpAffine(
                mask, 
                transform_matrix, 
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        
        return image, mask
    
    def _apply_photometric(self, image: np.ndarray) -> np.ndarray:
        """
        Applique des transformations photométriques aléatoires.
        
        Ces transformations simulent les variations d'éclairage
        et de conditions de prise de vue.
        
        Transformations possibles:
        - Ajustement de luminosité
        - Ajustement de contraste
        - Ajout de bruit
        - Variation de gamma
        """
        # Luminosité (60% de chance)
        if random.random() < 0.6:
            delta = random.uniform(-30, 30)
            image = self._adjust_brightness(image, delta)
        
        # Contraste (50% de chance)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            image = self._adjust_contrast(image, factor)
        
        # Bruit gaussien (30% de chance)
        if random.random() < 0.3:
            sigma = random.uniform(5, 15)
            image = self._add_gaussian_noise(image, sigma)
        
        # Correction gamma (25% de chance)
        if random.random() < 0.25:
            gamma = random.uniform(0.8, 1.2)
            image = self._adjust_gamma(image, gamma)
        
        return image
    
    def _adjust_brightness(
        self, 
        image: np.ndarray, 
        delta: float
    ) -> np.ndarray:
        """
        Ajuste la luminosité de l'image.
        
        Paramètres:
        -----------
        delta : float
            Valeur à ajouter à chaque pixel (-255 à +255)
        """
        # Conversion en float pour éviter le dépassement
        adjusted = image.astype(np.float32) + delta
        # Clipper les valeurs
        adjusted = np.clip(adjusted, 0, 255)
        return adjusted.astype(np.uint8)
    
    def _adjust_contrast(
        self, 
        image: np.ndarray, 
        factor: float
    ) -> np.ndarray:
        """
        Ajuste le contraste de l'image.
        
        Paramètres:
        -----------
        factor : float
            Facteur de contraste (>1 = plus de contraste, <1 = moins)
        """
        # Moyenne de l'image
        mean = np.mean(image)
        # Ajustement du contraste autour de la moyenne
        adjusted = (image.astype(np.float32) - mean) * factor + mean
        adjusted = np.clip(adjusted, 0, 255)
        return adjusted.astype(np.uint8)
    
    def _add_gaussian_noise(
        self, 
        image: np.ndarray, 
        sigma: float
    ) -> np.ndarray:
        """
        Ajoute du bruit gaussien à l'image.
        
        Simule le bruit du capteur de la caméra industrielle.
        
        Paramètres:
        -----------
        sigma : float
            Écart-type du bruit
        """
        noise = np.random.normal(0, sigma, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)
    
    def _adjust_gamma(
        self, 
        image: np.ndarray, 
        gamma: float
    ) -> np.ndarray:
        """
        Applique une correction gamma.
        
        La correction gamma simule différentes réponses de capteur
        et conditions d'éclairage.
        
        Paramètres:
        -----------
        gamma : float
            Valeur gamma (>1 = assombrissement, <1 = éclaircissement)
        """
        # Table de correspondance pour accélérer le calcul
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype(np.uint8)
        
        return cv2.LUT(image, table)


class WeldSpecificAugmentation(DataAugmentation):
    """
    Augmentations spécifiques aux images de soudure.
    
    Ajoute des transformations qui simulent des artefacts
    et conditions communes dans l'imagerie industrielle:
    - Reflets métalliques
    - Variations d'exposition dues à la chaleur
    - Éclaboussures et projections
    """
    
    def __init__(
        self,
        p_geometric: float = 0.5,
        p_photometric: float = 0.5,
        p_industrial: float = 0.3,
        seed: Optional[int] = None
    ):
        super().__init__(p_geometric, p_photometric, seed)
        self.p_industrial = p_industrial
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Applique les augmentations incluant les effets industriels."""
        # Appliquer les augmentations de base
        image, mask = super().__call__(image, mask)
        
        # Appliquer les augmentations industrielles
        if random.random() < self.p_industrial:
            image = self._apply_industrial(image)
        
        return image, mask
    
    def _apply_industrial(self, image: np.ndarray) -> np.ndarray:
        """
        Applique des augmentations spécifiques à l'industrie.
        """
        # Reflet métallique (effet de surexposition locale)
        if random.random() < 0.4:
            image = self._add_specular_highlight(image)
        
        # Ombre (zones sous-exposées)
        if random.random() < 0.3:
            image = self._add_shadow(image)
        
        # Petites taches (éclaboussures, poussière)
        if random.random() < 0.2:
            image = self._add_spots(image)
        
        return image
    
    def _add_specular_highlight(self, image: np.ndarray) -> np.ndarray:
        """
        Ajoute un reflet spéculaire (surexposition locale).
        
        Simule le reflet de la lumière sur la surface métallique
        brillante de la soudure.
        """
        h, w = image.shape[:2]
        
        # Position aléatoire du reflet
        center_x = random.randint(0, w - 1)
        center_y = random.randint(0, h - 1)
        
        # Taille du reflet
        radius = random.randint(10, min(h, w) // 4)
        
        # Créer le masque du reflet (gaussien)
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        highlight_mask = np.exp(-(distance ** 2) / (2 * radius ** 2))
        
        # Intensité du reflet
        intensity = random.uniform(30, 80)
        
        # Appliquer le reflet
        if len(image.shape) == 2:
            highlight = highlight_mask * intensity
            image = np.clip(image.astype(np.float32) + highlight, 0, 255)
        else:
            highlight = highlight_mask[:, :, np.newaxis] * intensity
            image = np.clip(image.astype(np.float32) + highlight, 0, 255)
        
        return image.astype(np.uint8)
    
    def _add_shadow(self, image: np.ndarray) -> np.ndarray:
        """
        Ajoute une ombre (zone sous-exposée).
        
        Simule les ombres causées par la géométrie des pièces
        ou l'éclairage non uniforme.
        """
        h, w = image.shape[:2]
        
        # Créer un gradient pour l'ombre
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        if direction == 'horizontal':
            gradient = np.linspace(0, 1, w)
            gradient = gradient[np.newaxis, :].repeat(h, axis=0)
        elif direction == 'vertical':
            gradient = np.linspace(0, 1, h)
            gradient = gradient[:, np.newaxis].repeat(w, axis=1)
        else:
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xv, yv = np.meshgrid(x, y)
            gradient = (xv + yv) / 2
        
        # Inverser aléatoirement
        if random.random() < 0.5:
            gradient = 1 - gradient
        
        # Intensité de l'ombre
        shadow_strength = random.uniform(0.7, 0.95)
        shadow_mask = gradient * (1 - shadow_strength) + shadow_strength
        
        # Appliquer l'ombre
        if len(image.shape) == 2:
            image = image * shadow_mask
        else:
            image = image * shadow_mask[:, :, np.newaxis]
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _add_spots(
        self, 
        image: np.ndarray, 
        n_spots: int = None
    ) -> np.ndarray:
        """
        Ajoute de petites taches aléatoires.
        
        Simule les projections de soudure, poussière ou
        autres artefacts présents sur les surfaces industrielles.
        """
        h, w = image.shape[:2]
        
        if n_spots is None:
            n_spots = random.randint(3, 10)
        
        for _ in range(n_spots):
            # Position et taille de la tache
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            radius = random.randint(1, 5)
            
            # Couleur de la tache (plus sombre ou plus claire)
            if random.random() < 0.5:
                intensity = random.randint(0, 50)
            else:
                intensity = random.randint(200, 255)
            
            # Dessiner la tache
            cv2.circle(image, (x, y), radius, intensity, -1)
        
        return image


class AugmentationPipeline:
    """
    Pipeline configurable d'augmentation de données.
    
    Permet de définir une séquence personnalisée de transformations
    avec des probabilités individuelles.
    
    Exemple d'utilisation:
    ----------------------
    >>> pipeline = AugmentationPipeline([
    ...     ("flip_horizontal", 0.5),
    ...     ("brightness", 0.3),
    ...     ("noise", 0.2),
    ... ])
    >>> augmented = pipeline(image, mask)
    """
    
    def __init__(
        self, 
        transforms: List[Tuple[str, float]],
        seed: Optional[int] = None
    ):
        """
        Paramètres:
        -----------
        transforms : List[Tuple[str, float]]
            Liste de (nom_transformation, probabilité)
        seed : int, optional
            Graine pour la reproductibilité
        """
        self.transforms = transforms
        self.augmenter = DataAugmentation(seed=seed)
        
        # Mapping des noms vers les fonctions
        self.transform_map = {
            "flip_horizontal": self.augmenter._flip_horizontal,
            "flip_vertical": self.augmenter._flip_vertical,
            "rotate_90": self.augmenter._rotate_90,
            "brightness": lambda img, _: (
                self.augmenter._adjust_brightness(img, random.uniform(-30, 30)), 
                None
            ),
            "contrast": lambda img, _: (
                self.augmenter._adjust_contrast(img, random.uniform(0.8, 1.2)), 
                None
            ),
            "noise": lambda img, _: (
                self.augmenter._add_gaussian_noise(img, random.uniform(5, 15)), 
                None
            ),
            "gamma": lambda img, _: (
                self.augmenter._adjust_gamma(img, random.uniform(0.8, 1.2)), 
                None
            ),
        }
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Applique le pipeline d'augmentation.
        """
        for transform_name, probability in self.transforms:
            if random.random() < probability:
                if transform_name in self.transform_map:
                    result = self.transform_map[transform_name](image, mask)
                    if result[1] is not None:
                        image, mask = result
                    else:
                        image = result[0]
        
        return image, mask


if __name__ == "__main__":
    # ==========================================================================
    # TEST DES AUGMENTATIONS DE DONNÉES
    # ==========================================================================
    
    print("=" * 60)
    print("Test des augmentations de données pour soudures")
    print("=" * 60)
    
    # Créer une image et un masque de test
    test_image = np.random.randint(100, 200, (256, 256), dtype=np.uint8)
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(test_mask, (50, 100), (200, 150), 255, -1)
    
    print(f"\nImage test: {test_image.shape}")
    print(f"Masque test: {test_mask.shape}")
    
    # Test de l'augmentation de base
    augmenter = DataAugmentation(p_geometric=0.8, p_photometric=0.8, seed=42)
    
    aug_image, aug_mask = augmenter(test_image.copy(), test_mask.copy())
    print(f"\nAprès augmentation de base:")
    print(f"  Image: {aug_image.shape}, range: [{aug_image.min()}, {aug_image.max()}]")
    print(f"  Masque: {aug_mask.shape}, unique: {np.unique(aug_mask)}")
    
    # Test de l'augmentation spécifique soudure
    weld_augmenter = WeldSpecificAugmentation(
        p_geometric=0.5,
        p_photometric=0.5,
        p_industrial=0.8,
        seed=42
    )
    
    weld_aug_image, weld_aug_mask = weld_augmenter(test_image.copy(), test_mask.copy())
    print(f"\nAprès augmentation soudure:")
    print(f"  Image: {weld_aug_image.shape}, range: [{weld_aug_image.min()}, {weld_aug_image.max()}]")
    
    # Test du pipeline configurable
    pipeline = AugmentationPipeline([
        ("flip_horizontal", 0.5),
        ("brightness", 0.8),
        ("noise", 0.5),
    ])
    
    pipe_image, pipe_mask = pipeline(test_image.copy(), test_mask.copy())
    print(f"\nAprès pipeline personnalisé:")
    print(f"  Image: {pipe_image.shape}")
    
    print("\n✓ Test des augmentations réussi!")
