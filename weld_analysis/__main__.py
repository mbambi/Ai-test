#!/usr/bin/env python3
# =============================================================================
# Point d'entrée principal du module weld_analysis
# =============================================================================
# 
# Permet d'exécuter le module directement avec:
#   python -m weld_analysis [commande] [options]
#
# Exemple:
#   python -m weld_analysis analyze image.jpg
#   python -m weld_analysis video welding.mp4
#   python -m weld_analysis train --data ./dataset
#
# =============================================================================

from .cli import main

if __name__ == "__main__":
    main()
