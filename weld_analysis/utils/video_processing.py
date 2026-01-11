# =============================================================================
# Utilitaires de Traitement Vidéo pour l'Analyse de Soudures
# =============================================================================
# 
# Ce module fournit des outils pour traiter les vidéos de soudure:
# - Extraction de frames à intervalles réguliers
# - Détection automatique des zones d'intérêt
# - Suivi temporel des soudures
# - Export des résultats d'analyse en vidéo
#
# =============================================================================

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import torch
from tqdm import tqdm

# Import local
from .image_processing import ImageProcessor


@dataclass
class VideoMetadata:
    """
    Métadonnées d'une vidéo.
    
    Attributs:
    ----------
    width : int
        Largeur des frames en pixels
    height : int
        Hauteur des frames en pixels
    fps : float
        Images par seconde
    total_frames : int
        Nombre total de frames
    duration : float
        Durée en secondes
    codec : str
        Codec vidéo utilisé
    """
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str


class VideoProcessor:
    """
    Classe pour le traitement des vidéos de soudure.
    
    Cette classe permet de:
    - Charger et analyser des vidéos frame par frame
    - Extraire des frames à intervalles réguliers
    - Appliquer des modèles de deep learning sur les frames
    - Générer des vidéos annotées avec les résultats d'analyse
    
    Paramètres:
    -----------
    image_processor : ImageProcessor, optional
        Processeur d'images à utiliser pour chaque frame
        Si non fourni, un processeur par défaut est créé
        
    Exemple d'utilisation:
    ----------------------
    >>> processor = VideoProcessor()
    >>> for frame, timestamp in processor.extract_frames("weld_video.mp4"):
    ...     # Traiter chaque frame
    ...     result = analyze(frame)
    """
    
    def __init__(
        self, 
        image_processor: Optional[ImageProcessor] = None
    ):
        # Utiliser le processeur fourni ou en créer un par défaut
        self.image_processor = image_processor or ImageProcessor()
    
    def get_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extrait les métadonnées d'une vidéo.
        
        Paramètres:
        -----------
        video_path : str
            Chemin vers le fichier vidéo
            
        Retourne:
        ---------
        VideoMetadata
            Métadonnées de la vidéo
        """
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
        
        try:
            # Extraire les propriétés
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Récupérer le codec (4 caractères)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([
                chr((fourcc >> 8 * i) & 0xFF) 
                for i in range(4)
            ])
            
            # Calculer la durée
            duration = total_frames / fps if fps > 0 else 0
            
            return VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                codec=codec
            )
        finally:
            cap.release()
    
    def extract_frames(
        self, 
        video_path: str,
        interval: float = 1.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Générateur qui extrait des frames à intervalles réguliers.
        
        Cette fonction est un générateur pour éviter de charger
        toutes les frames en mémoire simultanément.
        
        Paramètres:
        -----------
        video_path : str
            Chemin vers le fichier vidéo
        interval : float
            Intervalle entre les frames en secondes (défaut: 1.0)
        start_time : float
            Temps de début en secondes (défaut: 0.0)
        end_time : float, optional
            Temps de fin en secondes (défaut: fin de la vidéo)
        max_frames : int, optional
            Nombre maximum de frames à extraire
            
        Yields:
        -------
        Tuple[np.ndarray, float]
            (frame, timestamp) où frame est l'image et timestamp le temps en secondes
        """
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
        
        try:
            # Récupérer les propriétés
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Définir les limites temporelles
            if end_time is None:
                end_time = duration
            
            # Calculer les frames à extraire
            current_time = start_time
            frame_count = 0
            
            while current_time < end_time:
                # Limiter le nombre de frames
                if max_frames is not None and frame_count >= max_frames:
                    break
                
                # Calculer le numéro de frame correspondant
                frame_number = int(current_time * fps)
                
                # Positionner la vidéo
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                # Lire la frame
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convertir BGR vers RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                yield frame, current_time
                
                current_time += interval
                frame_count += 1
                
        finally:
            cap.release()
    
    def process_video(
        self,
        video_path: str,
        model: torch.nn.Module,
        output_path: Optional[str] = None,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Traite une vidéo complète avec un modèle de segmentation.
        
        Pour chaque frame:
        1. Prétraitement de l'image
        2. Inférence avec le modèle
        3. Post-traitement du masque
        4. Optionnellement, sauvegarde de la vidéo annotée
        
        Paramètres:
        -----------
        video_path : str
            Chemin vers la vidéo d'entrée
        model : torch.nn.Module
            Modèle de segmentation (U-Net)
        output_path : str, optional
            Chemin pour la vidéo de sortie annotée
        batch_size : int
            Taille des batchs pour l'inférence
        device : str
            Appareil de calcul ("cuda" ou "cpu")
        show_progress : bool
            Afficher une barre de progression
            
        Retourne:
        ---------
        List[Dict[str, Any]]
            Liste des résultats pour chaque frame analysée
        """
        # Récupérer les métadonnées
        metadata = self.get_metadata(video_path)
        
        # Préparer le modèle
        model = model.to(device)
        model.eval()
        
        # Préparer la sortie vidéo si demandée
        video_writer = None
        if output_path is not None:
            video_writer = self._create_video_writer(
                output_path,
                metadata.width,
                metadata.height,
                metadata.fps
            )
        
        results = []
        
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        
        try:
            # Barre de progression
            frame_iterator = range(metadata.total_frames)
            if show_progress:
                frame_iterator = tqdm(
                    frame_iterator, 
                    desc="Traitement vidéo",
                    total=metadata.total_frames
                )
            
            for frame_idx in frame_iterator:
                # Lire la frame
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convertir BGR vers RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Prétraiter pour le modèle
                processed = self.image_processor.preprocess(frame_rgb)
                tensor = self.image_processor.to_tensor(processed)
                tensor = tensor.to(device)
                
                # Inférence
                with torch.no_grad():
                    if hasattr(model, 'predict'):
                        mask = model.predict(tensor)
                    else:
                        output = model(tensor)
                        mask = torch.sigmoid(output)
                
                # Convertir le masque en numpy
                mask_np = mask.cpu().numpy()[0, 0]
                
                # Redimensionner le masque à la taille originale
                mask_resized = cv2.resize(
                    mask_np,
                    (metadata.width, metadata.height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Stocker les résultats
                timestamp = frame_idx / metadata.fps
                result = {
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'mask': mask_resized,
                    'weld_coverage': np.mean(mask_resized > 0.5)
                }
                results.append(result)
                
                # Écrire la frame annotée si demandé
                if video_writer is not None:
                    annotated = self._annotate_frame(frame, mask_resized)
                    video_writer.write(annotated)
                    
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
        
        return results
    
    def _create_video_writer(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float
    ) -> cv2.VideoWriter:
        """
        Crée un writer vidéo pour sauvegarder les résultats.
        
        Paramètres:
        -----------
        output_path : str
            Chemin de sortie
        width, height : int
            Dimensions de la vidéo
        fps : float
            Images par seconde
            
        Retourne:
        ---------
        cv2.VideoWriter
            Writer configuré
        """
        # Codec MP4V pour une compatibilité maximale
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Créer le répertoire parent si nécessaire
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
        
        return writer
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Annote une frame avec le masque de segmentation.
        
        Superpose le masque en couleur sur l'image originale
        pour visualiser la segmentation.
        
        Paramètres:
        -----------
        frame : np.ndarray
            Frame originale (BGR)
        mask : np.ndarray
            Masque de segmentation (0-1)
        alpha : float
            Transparence de l'overlay (0 = transparent, 1 = opaque)
            
        Retourne:
        ---------
        np.ndarray
            Frame annotée (BGR)
        """
        # Créer un overlay coloré pour le masque
        overlay = np.zeros_like(frame)
        
        # Colorer les zones de soudure en vert
        mask_binary = (mask > 0.5).astype(np.uint8)
        overlay[:, :, 1] = mask_binary * 255  # Canal vert
        
        # Superposer sur l'image originale
        annotated = cv2.addWeighted(
            frame, 1 - alpha * mask_binary.max(),
            overlay, alpha,
            0
        )
        
        # Dessiner le contour
        contours, _ = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
        
        # Ajouter les informations textuelles
        coverage = np.mean(mask_binary) * 100
        text = f"Soudure: {coverage:.1f}%"
        cv2.putText(
            annotated,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return annotated
    
    def detect_weld_events(
        self,
        video_path: str,
        model: torch.nn.Module,
        min_duration: float = 0.5,
        threshold: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> List[Dict[str, Any]]:
        """
        Détecte les événements de soudure dans une vidéo.
        
        Un événement de soudure est défini comme une période où
        la couverture de soudure dépasse un certain seuil.
        
        Paramètres:
        -----------
        video_path : str
            Chemin vers la vidéo
        model : torch.nn.Module
            Modèle de segmentation
        min_duration : float
            Durée minimale d'un événement en secondes
        threshold : float
            Seuil de couverture pour détecter un événement
        device : str
            Appareil de calcul
            
        Retourne:
        ---------
        List[Dict[str, Any]]
            Liste des événements détectés avec:
            - start_time: Début en secondes
            - end_time: Fin en secondes
            - duration: Durée en secondes
            - average_coverage: Couverture moyenne
        """
        # Traiter la vidéo
        results = self.process_video(
            video_path,
            model,
            device=device,
            show_progress=True
        )
        
        # Extraire les timestamps et couvertures
        timestamps = [r['timestamp'] for r in results]
        coverages = [r['weld_coverage'] for r in results]
        
        # Détecter les événements
        events = []
        in_event = False
        event_start = 0
        event_coverages = []
        
        for i, (t, c) in enumerate(zip(timestamps, coverages)):
            if c > threshold and not in_event:
                # Début d'un événement
                in_event = True
                event_start = t
                event_coverages = [c]
            elif c > threshold and in_event:
                # Continuation de l'événement
                event_coverages.append(c)
            elif c <= threshold and in_event:
                # Fin de l'événement
                duration = t - event_start
                if duration >= min_duration:
                    events.append({
                        'start_time': event_start,
                        'end_time': t,
                        'duration': duration,
                        'average_coverage': np.mean(event_coverages)
                    })
                in_event = False
                event_coverages = []
        
        # Gérer l'événement en cours à la fin de la vidéo
        if in_event:
            duration = timestamps[-1] - event_start
            if duration >= min_duration:
                events.append({
                    'start_time': event_start,
                    'end_time': timestamps[-1],
                    'duration': duration,
                    'average_coverage': np.mean(event_coverages)
                })
        
        return events
    
    def create_summary_video(
        self,
        video_path: str,
        events: List[Dict[str, Any]],
        output_path: str,
        padding: float = 1.0
    ) -> None:
        """
        Crée une vidéo résumée contenant seulement les événements de soudure.
        
        Paramètres:
        -----------
        video_path : str
            Chemin vers la vidéo originale
        events : List[Dict[str, Any]]
            Liste des événements à inclure
        output_path : str
            Chemin de sortie
        padding : float
            Temps supplémentaire avant/après chaque événement
        """
        metadata = self.get_metadata(video_path)
        
        # Ouvrir la vidéo source
        cap = cv2.VideoCapture(video_path)
        
        # Créer le writer
        writer = self._create_video_writer(
            output_path,
            metadata.width,
            metadata.height,
            metadata.fps
        )
        
        try:
            for event in events:
                # Calculer les frames à extraire (avec padding)
                start_frame = max(
                    0, 
                    int((event['start_time'] - padding) * metadata.fps)
                )
                end_frame = min(
                    metadata.total_frames,
                    int((event['end_time'] + padding) * metadata.fps)
                )
                
                # Positionner la vidéo
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Extraire les frames
                for _ in range(end_frame - start_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                    
        finally:
            cap.release()
            writer.release()


class FrameBuffer:
    """
    Buffer circulaire pour le traitement en temps réel.
    
    Permet de maintenir un historique de frames pour:
    - Le suivi temporel
    - Le moyennage temporel des prédictions
    - La détection de changements
    
    Paramètres:
    -----------
    buffer_size : int
        Nombre de frames à conserver
    """
    
    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.frames = []
        self.masks = []
        self.timestamps = []
    
    def add(
        self, 
        frame: np.ndarray, 
        mask: np.ndarray, 
        timestamp: float
    ) -> None:
        """
        Ajoute une nouvelle frame au buffer.
        
        Si le buffer est plein, la plus ancienne frame est supprimée.
        """
        self.frames.append(frame)
        self.masks.append(mask)
        self.timestamps.append(timestamp)
        
        # Supprimer les anciennes entrées si nécessaire
        if len(self.frames) > self.buffer_size:
            self.frames.pop(0)
            self.masks.pop(0)
            self.timestamps.pop(0)
    
    def get_temporal_average(self) -> np.ndarray:
        """
        Calcule la moyenne temporelle des masques.
        
        Utile pour réduire le bruit des prédictions.
        """
        if not self.masks:
            return None
        
        return np.mean(self.masks, axis=0)
    
    def detect_change(self, threshold: float = 0.1) -> bool:
        """
        Détecte un changement significatif dans les dernières frames.
        
        Paramètres:
        -----------
        threshold : float
            Seuil de changement (différence de couverture)
            
        Retourne:
        ---------
        bool
            True si un changement significatif est détecté
        """
        if len(self.masks) < 2:
            return False
        
        # Comparer les couvertures de soudure
        current_coverage = np.mean(self.masks[-1] > 0.5)
        previous_coverage = np.mean(self.masks[-2] > 0.5)
        
        return abs(current_coverage - previous_coverage) > threshold
    
    def clear(self) -> None:
        """Vide le buffer."""
        self.frames.clear()
        self.masks.clear()
        self.timestamps.clear()


if __name__ == "__main__":
    # ==========================================================================
    # TEST DES UTILITAIRES DE TRAITEMENT VIDÉO
    # ==========================================================================
    
    print("=" * 60)
    print("Test des utilitaires de traitement vidéo")
    print("=" * 60)
    
    # Créer un processeur vidéo
    processor = VideoProcessor()
    
    print("\nNote: Les tests complets nécessitent une vidéo réelle.")
    print("Cette démonstration montre l'initialisation et le buffer.")
    
    # Test du buffer de frames
    buffer = FrameBuffer(buffer_size=5)
    print(f"\nBuffer créé avec taille: {buffer.buffer_size}")
    
    # Simuler l'ajout de frames
    for i in range(7):
        fake_frame = np.random.rand(100, 100, 3)
        fake_mask = np.random.rand(100, 100)
        buffer.add(fake_frame, fake_mask, i * 0.1)
    
    print(f"Frames dans le buffer: {len(buffer.frames)}")
    
    # Test de la moyenne temporelle
    avg_mask = buffer.get_temporal_average()
    print(f"Masque moyen: {avg_mask.shape}")
    
    # Test de la détection de changement
    change = buffer.detect_change()
    print(f"Changement détecté: {change}")
    
    print("\n✓ Test des utilitaires vidéo réussi!")
