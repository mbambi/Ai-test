#!/usr/bin/env python3
# =============================================================================
# Interface en Ligne de Commande (CLI) pour l'Analyse de Soudure
# =============================================================================
# 
# Ce module fournit une interface utilisateur en ligne de commande pour
# accéder aux fonctionnalités du pipeline d'analyse de soudure.
#
# Commandes disponibles:
# - analyze: Analyse une ou plusieurs images
# - video: Analyse une vidéo
# - train: Lance l'entraînement des modèles
# - evaluate: Évalue les performances sur un dataset de test
#
# Usage:
#   python -m weld_analysis.cli analyze image.jpg
#   python -m weld_analysis.cli video welding.mp4 --output annotated.mp4
#   python -m weld_analysis.cli train --data ./dataset --epochs 100
#
# =============================================================================

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import glob
import json

# Configuration pour éviter les imports avant la définition du chemin
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_parser() -> argparse.ArgumentParser:
    """
    Configure le parseur d'arguments principal.
    
    Retourne:
    ---------
    argparse.ArgumentParser
        Parseur configuré
    """
    parser = argparse.ArgumentParser(
        prog='weld-analysis',
        description="""
        ╔══════════════════════════════════════════════════════════════╗
        ║     SYSTÈME D'ANALYSE DE SOUDURE PAR INTELLIGENCE            ║
        ║                   ARTIFICIELLE                               ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Segmentation, mesure et contrôle qualité automatisés        ║
        ╚══════════════════════════════════════════════════════════════╝
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Sous-commandes
    subparsers = parser.add_subparsers(
        dest='command',
        title='Commandes disponibles',
        description='Utilisez <commande> --help pour plus de détails'
    )
    
    # === Commande: analyze ===
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyse une ou plusieurs images de soudure',
        description='Analyse des images de soudure avec segmentation et scoring'
    )
    
    analyze_parser.add_argument(
        'input',
        nargs='+',
        help='Chemins des images à analyser (supporte les wildcards)'
    )
    
    analyze_parser.add_argument(
        '-o', '--output',
        type=str,
        default='./outputs',
        help='Répertoire de sortie pour les résultats (défaut: ./outputs)'
    )
    
    analyze_parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Appareil de calcul (défaut: auto)'
    )
    
    analyze_parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Chemin vers le modèle de segmentation pré-entraîné'
    )
    
    analyze_parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Seuil de segmentation (défaut: 0.5)'
    )
    
    analyze_parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Désactive la génération des visualisations'
    )
    
    analyze_parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Chemin pour le rapport textuel'
    )
    
    analyze_parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Chemin pour l\'export JSON des résultats'
    )
    
    analyze_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mode verbeux avec détails supplémentaires'
    )
    
    # === Commande: video ===
    video_parser = subparsers.add_parser(
        'video',
        help='Analyse une vidéo de soudure',
        description='Analyse frame par frame d\'une vidéo de soudage'
    )
    
    video_parser.add_argument(
        'input',
        type=str,
        help='Chemin de la vidéo à analyser'
    )
    
    video_parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Chemin pour la vidéo annotée de sortie'
    )
    
    video_parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Intervalle entre les frames analysées en secondes (défaut: 1.0)'
    )
    
    video_parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Appareil de calcul'
    )
    
    video_parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Chemin vers le modèle pré-entraîné'
    )
    
    video_parser.add_argument(
        '--summary',
        type=str,
        default=None,
        help='Chemin pour le rapport de synthèse'
    )
    
    # === Commande: train ===
    train_parser = subparsers.add_parser(
        'train',
        help='Entraîne les modèles de segmentation ou classification',
        description='Entraînement des modèles U-Net et CNN'
    )
    
    train_parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Chemin vers le répertoire de données d\'entraînement'
    )
    
    train_parser.add_argument(
        '--model-type',
        type=str,
        choices=['segmentation', 'classification', 'both'],
        default='both',
        help='Type de modèle à entraîner (défaut: both)'
    )
    
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Nombre d\'epochs d\'entraînement (défaut: 100)'
    )
    
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Taille des batchs (défaut: 8)'
    )
    
    train_parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Taux d\'apprentissage initial (défaut: 1e-4)'
    )
    
    train_parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Appareil de calcul'
    )
    
    train_parser.add_argument(
        '--output',
        type=str,
        default='./models',
        help='Répertoire pour sauvegarder les modèles (défaut: ./models)'
    )
    
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Chemin du checkpoint pour reprendre l\'entraînement'
    )
    
    # === Commande: evaluate ===
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Évalue les performances sur un dataset de test',
        description='Calcule les métriques de performance'
    )
    
    eval_parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Chemin vers le répertoire de données de test'
    )
    
    eval_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Chemin vers le modèle à évaluer'
    )
    
    eval_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Chemin pour le rapport d\'évaluation'
    )
    
    eval_parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Appareil de calcul'
    )
    
    return parser


def resolve_device(device: str) -> str:
    """
    Résout le choix de l'appareil de calcul.
    
    Paramètres:
    -----------
    device : str
        'cuda', 'cpu', ou 'auto'
        
    Retourne:
    ---------
    str
        'cuda' ou 'cpu'
    """
    import torch
    
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def expand_paths(patterns: List[str]) -> List[str]:
    """
    Étend les patterns glob en liste de fichiers.
    
    Paramètres:
    -----------
    patterns : List[str]
        Liste de chemins ou patterns
        
    Retourne:
    ---------
    List[str]
        Liste des fichiers trouvés
    """
    files = []
    for pattern in patterns:
        if '*' in pattern or '?' in pattern:
            # Pattern glob
            matches = glob.glob(pattern, recursive=True)
            files.extend(matches)
        else:
            # Fichier direct
            if os.path.exists(pattern):
                files.append(pattern)
            else:
                print(f"Avertissement: Fichier non trouvé: {pattern}")
    
    return sorted(set(files))


def cmd_analyze(args: argparse.Namespace):
    """
    Exécute la commande d'analyse d'images.
    """
    from .pipeline import WeldAnalysisPipeline, PipelineConfig
    
    # Résoudre les fichiers d'entrée
    files = expand_paths(args.input)
    
    if not files:
        print("Erreur: Aucun fichier trouvé")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  ANALYSE DE SOUDURE - {len(files)} fichier(s)")
    print(f"{'='*60}\n")
    
    # Configuration
    device = resolve_device(args.device)
    
    config = PipelineConfig(
        device=device,
        output_dir=args.output,
        segmentation_threshold=args.threshold,
        save_visualizations=not args.no_viz
    )
    
    # Créer le pipeline
    pipeline = WeldAnalysisPipeline(
        config=config,
        segmentation_model_path=args.model
    )
    
    # Analyser les images
    results = []
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Analyse: {file_path}")
        
        try:
            result = pipeline.analyze_image(file_path)
            results.append(result)
            
            # Affichage du résultat
            status = "✓" if result.is_acceptable else "✗"
            print(f"  {status} Grade: {result.quality_grade}, "
                  f"Score: {result.quality_score:.1%}, "
                  f"Temps: {result.processing_time:.2f}s")
            
            if args.verbose:
                print(f"    Largeur: {result.measurements['width_mean']:.1f} px")
                print(f"    Continuité: {result.continuity['score']:.1%}")
                print(f"    Homogénéité: {result.homogeneity['score']:.1%}")
                
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
    
    # Rapport textuel
    if args.report and results:
        report_content = ""
        for result in results:
            report_content += pipeline.generate_report(result) + "\n\n"
        
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\nRapport sauvegardé: {args.report}")
    
    # Export JSON
    if args.json and results:
        json_data = [r.to_dict() for r in results]
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"Export JSON sauvegardé: {args.json}")
    
    # Résumé
    if len(results) > 1:
        acceptable = sum(1 for r in results if r.is_acceptable)
        print(f"\n{'='*60}")
        print(f"  RÉSUMÉ: {acceptable}/{len(results)} acceptables "
              f"({acceptable/len(results):.1%})")
        print(f"{'='*60}")


def cmd_video(args: argparse.Namespace):
    """
    Exécute la commande d'analyse vidéo.
    """
    from .pipeline import WeldAnalysisPipeline, PipelineConfig
    
    print(f"\n{'='*60}")
    print(f"  ANALYSE VIDÉO: {args.input}")
    print(f"{'='*60}\n")
    
    device = resolve_device(args.device)
    
    config = PipelineConfig(
        device=device,
        output_dir=os.path.dirname(args.output) if args.output else './outputs'
    )
    
    pipeline = WeldAnalysisPipeline(
        config=config,
        segmentation_model_path=args.model
    )
    
    # Analyser la vidéo
    results = pipeline.analyze_video(
        args.input,
        output_video_path=args.output,
        frame_interval=args.interval
    )
    
    # Résumé
    if results:
        scores = [r.quality_score for r in results]
        acceptable = sum(1 for r in results if r.is_acceptable)
        
        print(f"\n{'='*60}")
        print(f"  RÉSUMÉ VIDÉO")
        print(f"{'='*60}")
        print(f"  Frames analysées: {len(results)}")
        print(f"  Score moyen: {sum(scores)/len(scores):.1%}")
        print(f"  Acceptables: {acceptable}/{len(results)} ({acceptable/len(results):.1%})")
        
        if args.output:
            print(f"  Vidéo annotée: {args.output}")


def cmd_train(args: argparse.Namespace):
    """
    Exécute la commande d'entraînement.
    """
    from .training.trainer import train_model
    
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT DES MODÈLES")
    print(f"{'='*60}")
    print(f"  Données: {args.data}")
    print(f"  Type: {args.model_type}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    device = resolve_device(args.device)
    
    train_model(
        data_dir=args.data,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        output_dir=args.output,
        resume_from=args.resume
    )


def cmd_evaluate(args: argparse.Namespace):
    """
    Exécute la commande d'évaluation.
    """
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION DU MODÈLE")
    print(f"{'='*60}")
    print(f"  Données: {args.data}")
    print(f"  Modèle: {args.model}")
    print(f"{'='*60}\n")
    
    # L'évaluation sera implémentée dans le module training
    print("Évaluation en cours...")
    print("(Module d'évaluation à implémenter)")


def main():
    """
    Point d'entrée principal du CLI.
    """
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Router vers la commande appropriée
    commands = {
        'analyze': cmd_analyze,
        'video': cmd_video,
        'train': cmd_train,
        'evaluate': cmd_evaluate
    }
    
    if args.command in commands:
        try:
            commands[args.command](args)
        except KeyboardInterrupt:
            print("\n\nInterruption par l'utilisateur")
            sys.exit(1)
        except Exception as e:
            print(f"\nErreur: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
