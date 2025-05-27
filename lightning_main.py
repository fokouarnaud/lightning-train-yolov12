#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv12-Face Lightning.ai - Point d'entrée principal
Implémentation simplifiée et optimisée pour Lightning.ai

Author: Cedric
Date: 2025-05-26
"""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

from src.train import YOLOv12FaceTrainer
from src.data_manager import DataManager
from src.model_manager import ModelManager
from src.lightning_utils import LightningLogger

def setup_logging(log_level: str = "INFO") -> None:
    """Configure le système de logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/logs/training.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis un fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='YOLOv12-Face Training sur Lightning.ai',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Chemin vers le fichier de configuration'
    )
    
    # Entraînement
    parser.add_argument(
        '--model-size', 
        type=str, 
        default='s',
        choices=['n', 's', 'm', 'l', 'x'],
        help='Taille du modèle YOLOv12'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Nombre d\'epochs d\'entraînement'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=16,
        help='Taille du batch'
    )
    
    parser.add_argument(
        '--img-size', 
        type=int, 
        default=640,
        help='Taille des images d\'entrée'
    )
    
    # Données
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='widerface',
        choices=['widerface', 'custom'],
        help='Dataset à utiliser'
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='./datasets',
        help='Chemin vers les données'
    )
    
    # Modes d'exécution
    parser.add_argument(
        '--mode', 
        type=str, 
        default='train',
        choices=['train', 'eval', 'export', 'all'],
        help='Mode d\'exécution'
    )
    
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Chemin vers un checkpoint pour reprendre l\'entraînement'
    )
    
    # Lightning.ai spécifique
    parser.add_argument(
        '--save-period', 
        type=int, 
        default=10,
        help='Fréquence de sauvegarde (epochs)'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Niveau de logging'
    )
    
    return parser.parse_args()

def main():
    """Fonction principale"""
    print("=" * 80)
    print("🚀 YOLOv12-Face Lightning.ai Training Pipeline")
    print("=" * 80)
    
    # Parse des arguments
    args = parse_args()
    
    # Configuration du logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Chargement de la configuration
        logger.info(f"📄 Chargement de la configuration: {args.config}")
        config = load_config(args.config)
        
        # Override de la config avec les arguments CLI
        config['model']['size'] = args.model_size
        config['training']['epochs'] = args.epochs
        config['training']['batch_size'] = args.batch_size
        config['training']['img_size'] = args.img_size
        config['data']['dataset'] = args.dataset
        config['data']['path'] = args.data_path
        
        # Initialisation des composants
        logger.info("🔧 Initialisation des composants...")
        
        # Gestionnaire de données
        data_manager = DataManager(config['data'])
        
        # Gestionnaire de modèles
        model_manager = ModelManager(config['model'])
        
        # Logger Lightning.ai
        lightning_logger = LightningLogger(
            project_name="yolov12-face",
            save_period=args.save_period
        )
        
        # Trainer principal
        trainer = YOLOv12FaceTrainer(
            config=config,
            data_manager=data_manager,
            model_manager=model_manager,
            logger=lightning_logger
        )
        
        # Exécution selon le mode
        if args.mode in ['train', 'all']:
            logger.info("🏋️ Démarrage de l'entraînement...")
            trainer.train(resume_from=args.resume)
        
        if args.mode in ['eval', 'all']:
            logger.info("📊 Évaluation du modèle...")
            trainer.evaluate()
        
        if args.mode in ['export', 'all']:
            logger.info("📦 Export du modèle...")
            trainer.export(format='onnx')
        
        logger.info("✅ Pipeline terminé avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {str(e)}")
        raise
    
    finally:
        print("=" * 80)
        print("📈 Résultats disponibles dans ./outputs/")
        print("📊 Logs disponibles dans ./outputs/logs/")
        print("=" * 80)

if __name__ == "__main__":
    main()
