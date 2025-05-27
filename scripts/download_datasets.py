#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de téléchargement des datasets pour YOLOv12-Face
Compatible avec WIDERFace et datasets personnalisés Google Drive
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_manager import DataManager

def setup_logging(verbose: bool = False):
    """Configure le système de logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('download_datasets.log')
        ]
    )

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Télécharge et prépare les datasets pour YOLOv12-Face',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='widerface',
        choices=['widerface', 'custom'],
        help='Type de dataset à télécharger'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./datasets',
        help='Répertoire de sortie pour les datasets'
    )
    
    parser.add_argument(
        '--google-drive-id',
        type=str,
        help='ID Google Drive pour dataset personnalisé'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Forcer le téléchargement même si les fichiers existent'
    )
    
    parser.add_argument(
        '--skip-conversion',
        action='store_true',
        help='Ignorer la conversion au format YOLO'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Valider le dataset existant sans télécharger'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mode verbose avec plus de logs'
    )
    
    return parser.parse_args()

def download_widerface(data_manager: DataManager, force: bool = False) -> bool:
    """
    Télécharge et prépare WIDERFace
    
    Args:
        data_manager: Gestionnaire de données
        force: Forcer le téléchargement
        
    Returns:
        True si succès
    """
    logger = logging.getLogger(__name__)
    logger.info("📥 Téléchargement de WIDERFace...")
    
    try:
        # Vérifier si déjà téléchargé
        if not force and data_manager._is_dataset_ready():
            logger.info("✅ WIDERFace déjà préparé")
            return True
        
        # Préparer le dataset
        data_yaml_path = data_manager.prepare_dataset()
        
        if data_yaml_path.exists():
            logger.info(f"✅ WIDERFace préparé: {data_yaml_path}")
            
            # Afficher les statistiques
            stats = data_manager.get_dataset_stats()
            logger.info("📊 Statistiques du dataset:")
            logger.info(f"   Train: {stats['train_images']} images, {stats['train_labels']} labels")
            logger.info(f"   Val: {stats['val_images']} images, {stats['val_labels']} labels")
            
            return True
        else:
            logger.error("❌ Échec de la préparation de WIDERFace")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement WIDERFace: {e}")
        return False

def download_custom_dataset(data_manager: DataManager, google_drive_id: str, force: bool = False) -> bool:
    """
    Télécharge un dataset personnalisé depuis Google Drive
    
    Args:
        data_manager: Gestionnaire de données
        google_drive_id: ID du fichier Google Drive
        force: Forcer le téléchargement
        
    Returns:
        True si succès
    """
    logger = logging.getLogger(__name__)
    logger.info(f"📥 Téléchargement du dataset personnalisé: {google_drive_id}")
    
    try:
        # Ajouter l'ID Google Drive à la config
        data_manager.config['google_drive_id'] = google_drive_id
        
        # Vérifier si déjà téléchargé
        if not force and data_manager._is_dataset_ready():
            logger.info("✅ Dataset personnalisé déjà préparé")
            return True
        
        # Préparer le dataset
        data_yaml_path = data_manager.prepare_dataset()
        
        if data_yaml_path.exists():
            logger.info(f"✅ Dataset personnalisé préparé: {data_yaml_path}")
            
            # Afficher les statistiques
            stats = data_manager.get_dataset_stats()
            logger.info("📊 Statistiques du dataset:")
            logger.info(f"   Train: {stats['train_images']} images, {stats['train_labels']} labels")
            logger.info(f"   Val: {stats['val_images']} images, {stats['val_labels']} labels")
            
            return True
        else:
            logger.error("❌ Échec de la préparation du dataset personnalisé")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement dataset personnalisé: {e}")
        return False

def validate_dataset(data_manager: DataManager) -> bool:
    """
    Valide un dataset existant
    
    Args:
        data_manager: Gestionnaire de données
        
    Returns:
        True si le dataset est valide
    """
    logger = logging.getLogger(__name__)
    logger.info("🔍 Validation du dataset...")
    
    try:
        from src.utils import check_dataset_health
        
        dataset_path = Path(data_manager.config['path'])
        health_report = check_dataset_health(dataset_path)
        
        logger.info(f"📋 Statut du dataset: {health_report['status']}")
        
        if health_report['issues']:
            logger.warning("⚠️ Problèmes détectés:")
            for issue in health_report['issues']:
                logger.warning(f"   - {issue}")
        
        if health_report['recommendations']:
            logger.info("💡 Recommandations:")
            for rec in health_report['recommendations']:
                logger.info(f"   - {rec}")
        
        # Afficher les statistiques
        if health_report['stats']:
            logger.info("📊 Statistiques:")
            for split, stats in health_report['stats'].items():
                logger.info(f"   {split}: {stats['images']} images, {stats['labels']} labels")
        
        return health_report['status'] in ['healthy', 'warning']
        
    except Exception as e:
        logger.error(f"❌ Erreur validation: {e}")
        return False

def create_sample_custom_config():
    """Crée un exemple de configuration pour dataset personnalisé"""
    logger = logging.getLogger(__name__)
    
    sample_config = """
# Configuration exemple pour dataset personnalisé
# ==============================================

# Votre dataset doit être organisé comme suit:
# dataset.zip
# ├── train/
# │   ├── images/
# │   │   ├── image1.jpg
# │   │   └── image2.jpg
# │   └── labels/
# │       ├── image1.txt
# │       └── image2.txt
# └── val/
#     ├── images/
#     │   ├── image3.jpg
#     │   └── image4.jpg
#     └── labels/
#         ├── image3.txt
#         └── image4.txt

# Les fichiers labels doivent être au format YOLO:
# class x_center y_center width height
# Exemple pour une face (class 0):
# 0 0.5 0.5 0.3 0.4

# Étapes pour utiliser votre dataset:
# 1. Compresser votre dataset en .zip
# 2. Uploader sur Google Drive
# 3. Récupérer l'ID du fichier (dans l'URL de partage)
# 4. Lancer: python scripts/download_datasets.py --dataset custom --google-drive-id YOUR_ID

# URL Google Drive exemple:
# https://drive.google.com/file/d/1ABC123DEF456GHI/view?usp=sharing
# L'ID est: 1ABC123DEF456GHI
"""
    
    sample_file = Path("custom_dataset_example.txt")
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_config)
    
    logger.info(f"📝 Exemple de configuration créé: {sample_file}")

def main():
    """Fonction principale"""
    args = parse_args()
    
    print("📥 YOLOv12-Face Dataset Downloader")
    print("=" * 50)
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Configuration du DataManager
    data_config = {
        'dataset': args.dataset,
        'path': args.output_dir
    }
    
    data_manager = DataManager(data_config)
    
    success = True
    
    try:
        if args.validate_only:
            # Mode validation seulement
            logger.info("🔍 Mode validation uniquement")
            success = validate_dataset(data_manager)
            
        elif args.dataset == 'widerface':
            # Téléchargement WIDERFace
            success = download_widerface(data_manager, args.force_download)
            
        elif args.dataset == 'custom':
            # Téléchargement dataset personnalisé
            if not args.google_drive_id:
                logger.error("❌ --google-drive-id requis pour dataset personnalisé")
                create_sample_custom_config()
                return 1
            
            success = download_custom_dataset(
                data_manager, 
                args.google_drive_id, 
                args.force_download
            )
        
        # Validation finale
        if success and not args.skip_conversion:
            logger.info("🔍 Validation finale...")
            success = validate_dataset(data_manager)
    
    except KeyboardInterrupt:
        logger.info("⏹️ Téléchargement interrompu par l'utilisateur")
        return 1
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        return 1
    
    # Résumé final
    print("\n" + "=" * 50)
    if success:
        print("✅ Dataset préparé avec succès!")
        print(f"📁 Chemin: {Path(args.output_dir).absolute()}")
        print("\n📝 Prochaines étapes:")
        print("1. python lightning_main.py --config config.yaml --mode train")
        print("2. Vérifier les résultats dans outputs/")
    else:
        print("❌ Échec de la préparation du dataset")
        print("🔍 Vérifiez les logs pour plus de détails")
        return 1
    
    print("=" * 50)
    return 0

if __name__ == "__main__":
    sys.exit(main())
