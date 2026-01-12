# =============================================================================
# Utilitaires de Traitement d'Images pour l'Analyse de Soudures
# =============================================================================
# 
# Ce module fournit des fonctions de prétraitement et post-traitement
# pour les images de soudure, incluant:
# - Normalisation et standardisation de l'éclairage
# - Correction des métadonnées de caméra (angle, résolution)
# - Amélioration du contraste pour les zones de soudure
# - Conversion entre formats d'image et tenseurs
#
# =============================================================================

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional, List, Union, Dict
from pathlib import Path
import json


class ImageProcessor:
    """
    Classe pour le traitement des images de soudure.
    
    Cette classe gère toutes les opérations de prétraitement nécessaires
    pour préparer les images pour les modèles de deep learning:
    - Chargement et sauvegarde d'images
    - Normalisation de l'éclairage
    - Correction de perspective basée sur les métadonnées caméra
    - Conversion vers/depuis les tenseurs PyTorch
    
    Paramètres:
    -----------
    target_size : Tuple[int, int]
        Taille cible pour le redimensionnement (height, width)
    normalize : bool
        Si True, normalise les valeurs de pixels entre 0 et 1
    grayscale : bool
        Si True, convertit les images en niveaux de gris
        
    Exemple d'utilisation:
    ----------------------
    >>> processor = ImageProcessor(target_size=(256, 256), grayscale=True)
    >>> image = processor.load_image("weld_image.jpg")
    >>> tensor = processor.to_tensor(image)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        grayscale: bool = True
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.grayscale = grayscale
        
        # Statistiques de normalisation (calculées sur ImageNet)
        # Utilisées pour les modèles pré-entraînés
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Statistiques pour les niveaux de gris
        self.gray_mean = 0.449
        self.gray_std = 0.226
    
    def load_image(
        self, 
        path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Charge une image depuis le disque avec correction optionnelle.
        
        Paramètres:
        -----------
        path : str ou Path
            Chemin vers le fichier image
        metadata_path : str ou Path, optional
            Chemin vers le fichier JSON de métadonnées caméra
            
        Retourne:
        ---------
        np.ndarray
            Image chargée et prétraitée
        """
        # Lecture de l'image avec OpenCV
        # cv2.IMREAD_UNCHANGED préserve tous les canaux (incluant alpha)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise FileNotFoundError(f"Impossible de charger l'image: {path}")
        
        # Conversion BGR vers RGB (OpenCV charge en BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # Image avec canal alpha
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            # Supprimer le canal alpha
            image = image[:, :, :3]
        
        # Application de la correction basée sur les métadonnées
        if metadata_path is not None:
            metadata = self._load_metadata(metadata_path)
            image = self._apply_camera_correction(image, metadata)
        
        # Conversion en niveaux de gris si demandé
        if self.grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Redimensionnement à la taille cible
        image = self._resize(image)
        
        return image
    
    def _load_metadata(
        self, 
        path: Union[str, Path]
    ) -> Dict:
        """
        Charge les métadonnées de la caméra depuis un fichier JSON.
        
        Les métadonnées attendues incluent:
        - angle: Angle de la caméra par rapport à la surface (degrés)
        - resolution: Résolution de l'image originale (pixels/mm)
        - distortion: Coefficients de distortion de la lentille
        
        Paramètres:
        -----------
        path : str ou Path
            Chemin vers le fichier de métadonnées
            
        Retourne:
        ---------
        Dict
            Dictionnaire des métadonnées
        """
        with open(path, 'r') as f:
            metadata = json.load(f)
        
        # Valeurs par défaut pour les métadonnées manquantes
        default_metadata = {
            'angle': 90.0,          # Vue perpendiculaire par défaut
            'resolution': 1.0,       # 1 pixel/mm par défaut
            'distortion': None,      # Pas de correction de distortion
            'exposure': 'auto'       # Exposition automatique
        }
        
        # Fusionner avec les valeurs par défaut
        for key, value in default_metadata.items():
            if key not in metadata:
                metadata[key] = value
        
        return metadata
    
    def _apply_camera_correction(
        self, 
        image: np.ndarray, 
        metadata: Dict
    ) -> np.ndarray:
        """
        Applique les corrections basées sur les métadonnées caméra.
        
        Cette fonction corrige:
        - La perspective si la caméra n'est pas perpendiculaire
        - La distortion de la lentille si les coefficients sont fournis
        
        Paramètres:
        -----------
        image : np.ndarray
            Image à corriger
        metadata : Dict
            Métadonnées de la caméra
            
        Retourne:
        ---------
        np.ndarray
            Image corrigée
        """
        # Correction de la distortion de la lentille
        if metadata.get('distortion') is not None:
            distortion = np.array(metadata['distortion'])
            h, w = image.shape[:2]
            
            # Matrice de caméra approximative (identité centrée)
            camera_matrix = np.array([
                [w, 0, w / 2],
                [0, h, h / 2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Application de la correction
            image = cv2.undistort(image, camera_matrix, distortion)
        
        # Correction de perspective pour les vues inclinées
        angle = metadata.get('angle', 90.0)
        if abs(angle - 90.0) > 1.0:  # Correction si angle significatif
            image = self._correct_perspective(image, angle)
        
        return image
    
    def _correct_perspective(
        self, 
        image: np.ndarray, 
        angle: float
    ) -> np.ndarray:
        """
        Corrige la perspective pour une vue inclinée.
        
        Utilise une transformation homographique pour simuler
        une vue perpendiculaire à partir d'une vue inclinée.
        
        Paramètres:
        -----------
        image : np.ndarray
            Image à corriger
        angle : float
            Angle de la caméra (90 = perpendiculaire)
            
        Retourne:
        ---------
        np.ndarray
            Image avec perspective corrigée
        """
        h, w = image.shape[:2]
        
        # Calcul du facteur d'étirement basé sur l'angle
        # cos(angle) détermine la compression verticale
        stretch_factor = 1.0 / np.cos(np.radians(90 - angle))
        
        # Points source (coins de l'image)
        src_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # Points destination (après correction)
        # La correction étire l'image verticalement
        margin = w * (stretch_factor - 1) / 4
        dst_points = np.float32([
            [margin, 0],
            [w - margin, 0],
            [w, h],
            [0, h]
        ])
        
        # Calcul de la matrice de transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Application de la transformation
        corrected = cv2.warpPerspective(image, matrix, (w, h))
        
        return corrected
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensionne l'image à la taille cible.
        
        Utilise l'interpolation INTER_AREA pour le sous-échantillonnage
        (meilleure qualité) et INTER_LINEAR pour le sur-échantillonnage.
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Choix de l'interpolation selon le cas
        if h > target_h or w > target_w:
            # Sous-échantillonnage
            interpolation = cv2.INTER_AREA
        else:
            # Sur-échantillonnage
            interpolation = cv2.INTER_LINEAR
        
        resized = cv2.resize(
            image, 
            (target_w, target_h), 
            interpolation=interpolation
        )
        
        return resized
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Applique le prétraitement complet pour améliorer la qualité.
        
        Étapes:
        1. Correction du bruit par filtre bilatéral
        2. Amélioration du contraste par CLAHE
        3. Normalisation de l'éclairage
        
        Paramètres:
        -----------
        image : np.ndarray
            Image à prétraiter
            
        Retourne:
        ---------
        np.ndarray
            Image prétraitée
        """
        # Assurer que l'image est en uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 1. Réduction du bruit avec filtre bilatéral
        # Préserve les contours tout en lissant les zones uniformes
        denoised = cv2.bilateralFilter(
            image,
            d=9,            # Diamètre du voisinage
            sigmaColor=75,  # Filtre dans l'espace des couleurs
            sigmaSpace=75   # Filtre dans l'espace spatial
        )
        
        # 2. Amélioration du contraste avec CLAHE
        # (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=2.0,      # Limite le contraste pour éviter le bruit
            tileGridSize=(8, 8)  # Taille des tuiles pour l'adaptation
        )
        
        if len(denoised.shape) == 2:
            # Image en niveaux de gris
            enhanced = clahe.apply(denoised)
        else:
            # Image couleur: appliquer sur le canal de luminance
            lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 3. Normalisation de l'éclairage par correction gamma
        enhanced = self._normalize_lighting(enhanced)
        
        return enhanced
    
    def _normalize_lighting(
        self, 
        image: np.ndarray, 
        target_mean: float = 128.0
    ) -> np.ndarray:
        """
        Normalise l'éclairage pour avoir une intensité moyenne constante.
        
        Cette correction est importante pour les images industrielles
        où l'éclairage peut varier entre les prises de vue.
        
        Paramètres:
        -----------
        image : np.ndarray
            Image à normaliser
        target_mean : float
            Intensité moyenne cible (défaut: 128 pour centrer)
            
        Retourne:
        ---------
        np.ndarray
            Image avec éclairage normalisé
        """
        # Calcul de l'intensité moyenne actuelle
        current_mean = np.mean(image)
        
        if current_mean < 1.0:
            return image  # Éviter la division par zéro
        
        # Calcul du facteur de correction
        gamma = np.log(target_mean / 255) / np.log(current_mean / 255)
        gamma = np.clip(gamma, 0.5, 2.0)  # Limiter la correction
        
        # Application de la correction gamma
        normalized = np.power(image / 255.0, gamma) * 255
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        return normalized
    
    def to_tensor(
        self, 
        image: np.ndarray, 
        add_batch_dim: bool = True
    ) -> torch.Tensor:
        """
        Convertit une image numpy en tenseur PyTorch.
        
        Transformations appliquées:
        1. Conversion en float32
        2. Normalisation si activée
        3. Réorganisation des dimensions (H, W, C) -> (C, H, W)
        4. Ajout optionnel de la dimension batch
        
        Paramètres:
        -----------
        image : np.ndarray
            Image numpy
        add_batch_dim : bool
            Si True, ajoute une dimension batch
            
        Retourne:
        ---------
        torch.Tensor
            Image sous forme de tenseur
        """
        # Assurer le type float32
        if image.dtype == np.uint8:
            tensor = image.astype(np.float32) / 255.0
        else:
            tensor = image.astype(np.float32)
        
        # Normalisation (moyenne et écart-type)
        if self.normalize:
            if len(tensor.shape) == 2:
                # Niveaux de gris
                tensor = (tensor - self.gray_mean) / self.gray_std
            else:
                # Couleur
                tensor = (tensor - self.mean) / self.std
        
        # Réorganisation des dimensions
        if len(tensor.shape) == 2:
            # (H, W) -> (1, H, W) pour niveaux de gris
            tensor = tensor[np.newaxis, :, :]
        else:
            # (H, W, C) -> (C, H, W) pour couleur
            tensor = tensor.transpose((2, 0, 1))
        
        # Conversion en tenseur PyTorch
        tensor = torch.from_numpy(tensor)
        
        # Ajout de la dimension batch si demandé
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def from_tensor(
        self, 
        tensor: torch.Tensor, 
        denormalize: bool = True
    ) -> np.ndarray:
        """
        Convertit un tenseur PyTorch en image numpy.
        
        Paramètres:
        -----------
        tensor : torch.Tensor
            Tenseur image
        denormalize : bool
            Si True, inverse la normalisation
            
        Retourne:
        ---------
        np.ndarray
            Image numpy
        """
        # Détacher du graphe de calcul et convertir en numpy
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        image = tensor.cpu().numpy()
        
        # Supprimer la dimension batch si présente
        if len(image.shape) == 4:
            image = image[0]
        
        # Réorganisation des dimensions: (C, H, W) -> (H, W, C)
        if image.shape[0] in [1, 3, 4]:
            image = image.transpose((1, 2, 0))
        
        # Dénormalisation
        if denormalize and self.normalize:
            if image.shape[-1] == 1:
                image = image * self.gray_std + self.gray_mean
                image = image.squeeze(-1)
            else:
                image = image * self.std + self.mean
        
        # Conversion en uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
    
    def save_image(
        self, 
        image: np.ndarray, 
        path: Union[str, Path]
    ) -> None:
        """
        Sauvegarde une image sur le disque.
        
        Paramètres:
        -----------
        image : np.ndarray
            Image à sauvegarder
        path : str ou Path
            Chemin de destination
        """
        path = Path(path)
        
        # Assurer que le répertoire existe
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Conversion RGB vers BGR pour OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Sauvegarde
        cv2.imwrite(str(path), image)
    
    def extract_weld_region(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Extrait la région de soudure en utilisant un masque.
        
        Paramètres:
        -----------
        image : np.ndarray
            Image originale
        mask : np.ndarray
            Masque de segmentation binaire
            
        Retourne:
        ---------
        np.ndarray
            Image de la région de soudure avec fond noir
        """
        # S'assurer que le masque est binaire
        mask = (mask > 0.5).astype(np.uint8)
        
        # Appliquer le masque
        if len(image.shape) == 3:
            mask = np.expand_dims(mask, axis=-1)
        
        extracted = image * mask
        
        return extracted
    
    def create_patches(
        self, 
        image: np.ndarray, 
        patch_size: Tuple[int, int] = (64, 64),
        stride: Optional[Tuple[int, int]] = None,
        mask: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Découpe l'image en patches pour l'analyse locale.
        
        Les patches sont utiles pour:
        - Analyse de l'homogénéité locale
        - Augmentation de données
        - Traitement d'images de grande taille
        
        Paramètres:
        -----------
        image : np.ndarray
            Image à découper
        patch_size : Tuple[int, int]
            Taille des patches (height, width)
        stride : Tuple[int, int], optional
            Pas entre les patches (défaut: patch_size)
        mask : np.ndarray, optional
            Si fourni, ne retourne que les patches contenant de la soudure
            
        Retourne:
        ---------
        List[Tuple[np.ndarray, Tuple[int, int]]]
            Liste de (patch, position) où position est (y, x)
        """
        if stride is None:
            stride = patch_size
        
        patches = []
        h, w = image.shape[:2]
        ph, pw = patch_size
        sh, sw = stride
        
        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                # Extraction du patch
                patch = image[y:y + ph, x:x + pw]
                
                # Si un masque est fourni, vérifier le contenu
                if mask is not None:
                    mask_patch = mask[y:y + ph, x:x + pw]
                    # Ignorer les patches sans soudure significative
                    if np.mean(mask_patch) < 0.1:
                        continue
                
                patches.append((patch, (y, x)))
        
        return patches
    
    def reconstruct_from_patches(
        self,
        patches: List[Tuple[np.ndarray, Tuple[int, int]]],
        original_size: Tuple[int, int],
        patch_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Reconstruit une image à partir de patches.
        
        Utilise une moyenne pondérée pour les zones de chevauchement.
        
        Paramètres:
        -----------
        patches : List[Tuple[np.ndarray, Tuple[int, int]]]
            Liste de (patch, position)
        original_size : Tuple[int, int]
            Taille de l'image originale
        patch_size : Tuple[int, int]
            Taille des patches
            
        Retourne:
        ---------
        np.ndarray
            Image reconstruite
        """
        h, w = original_size
        ph, pw = patch_size
        
        # Créer l'image de sortie et le compteur
        reconstructed = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        
        for patch, (y, x) in patches:
            reconstructed[y:y + ph, x:x + pw] += patch.astype(np.float32)
            count[y:y + ph, x:x + pw] += 1
        
        # Éviter la division par zéro
        count = np.maximum(count, 1)
        reconstructed = reconstructed / count
        
        return reconstructed.astype(np.uint8)


class MaskProcessor:
    """
    Classe pour le traitement des masques de segmentation.
    
    Fournit des outils pour:
    - Nettoyage des masques (fermeture, ouverture morphologique)
    - Extraction des contours
    - Analyse des composantes connexes
    """
    
    def __init__(self):
        # Élément structurant pour les opérations morphologiques
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (5, 5)
        )
    
    def clean_mask(
        self, 
        mask: np.ndarray,
        min_area: int = 100
    ) -> np.ndarray:
        """
        Nettoie un masque de segmentation.
        
        Opérations:
        1. Binarisation
        2. Fermeture morphologique (comble les trous)
        3. Ouverture morphologique (supprime le bruit)
        4. Suppression des petites composantes
        
        Paramètres:
        -----------
        mask : np.ndarray
            Masque à nettoyer
        min_area : int
            Surface minimale des régions à conserver
            
        Retourne:
        ---------
        np.ndarray
            Masque nettoyé
        """
        # Assurer la binarisation
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Fermeture: dilatation puis érosion
        # Comble les petits trous dans les régions
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Ouverture: érosion puis dilatation
        # Supprime le bruit et les petites protubérances
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel)
        
        # Suppression des petites composantes
        cleaned = self._remove_small_components(opened, min_area)
        
        return cleaned
    
    def _remove_small_components(
        self, 
        mask: np.ndarray, 
        min_area: int
    ) -> np.ndarray:
        """
        Supprime les composantes connexes de petite taille.
        
        Paramètres:
        -----------
        mask : np.ndarray
            Masque binaire
        min_area : int
            Surface minimale à conserver
            
        Retourne:
        ---------
        np.ndarray
            Masque sans les petites composantes
        """
        # Trouver les composantes connexes
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, 
            connectivity=8
        )
        
        # Créer le masque de sortie
        output = np.zeros_like(mask)
        
        # Conserver les composantes suffisamment grandes
        for i in range(1, num_labels):  # Ignorer le fond (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                output[labels == i] = 255
        
        return output
    
    def find_contours(
        self, 
        mask: np.ndarray
    ) -> List[np.ndarray]:
        """
        Trouve les contours des régions dans le masque.
        
        Paramètres:
        -----------
        mask : np.ndarray
            Masque binaire
            
        Retourne:
        ---------
        List[np.ndarray]
            Liste des contours (chaque contour est un array de points)
        """
        # Assurer le format binaire
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Trouver les contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,      # Seulement les contours externes
            cv2.CHAIN_APPROX_SIMPLE  # Compression des segments
        )
        
        return contours
    
    def get_bounding_boxes(
        self, 
        mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Obtient les boîtes englobantes des régions segmentées.
        
        Paramètres:
        -----------
        mask : np.ndarray
            Masque binaire
            
        Retourne:
        ---------
        List[Tuple[int, int, int, int]]
            Liste de (x, y, width, height) pour chaque région
        """
        contours = self.find_contours(mask)
        boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
        
        return boxes


if __name__ == "__main__":
    # ==========================================================================
    # TEST DES UTILITAIRES DE TRAITEMENT D'IMAGES
    # ==========================================================================
    
    print("=" * 60)
    print("Test des utilitaires de traitement d'images")
    print("=" * 60)
    
    # Créer un processeur d'images
    processor = ImageProcessor(target_size=(256, 256), grayscale=True)
    
    # Créer une image de test synthétique
    test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    print(f"\nImage test créée: {test_image.shape}, dtype: {test_image.dtype}")
    
    # Test du prétraitement
    preprocessed = processor.preprocess(test_image)
    print(f"Image prétraitée: {preprocessed.shape}")
    
    # Test de conversion vers tenseur
    tensor = processor.to_tensor(preprocessed)
    print(f"Tenseur: {tensor.shape}, dtype: {tensor.dtype}")
    
    # Test de conversion inverse
    recovered = processor.from_tensor(tensor)
    print(f"Image récupérée: {recovered.shape}")
    
    # Test de création de patches
    patches = processor.create_patches(test_image, patch_size=(64, 64))
    print(f"Nombre de patches: {len(patches)}")
    
    # Test du processeur de masques
    mask_processor = MaskProcessor()
    
    # Créer un masque de test
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(test_mask, (50, 50), (200, 100), 255, -1)
    
    # Nettoyer le masque
    cleaned = mask_processor.clean_mask(test_mask)
    print(f"Masque nettoyé: {cleaned.shape}")
    
    # Trouver les contours
    contours = mask_processor.find_contours(cleaned)
    print(f"Nombre de contours: {len(contours)}")
    
    # Boîtes englobantes
    boxes = mask_processor.get_bounding_boxes(cleaned)
    print(f"Boîtes englobantes: {boxes}")
    
    print("\n✓ Test des utilitaires réussi!")
