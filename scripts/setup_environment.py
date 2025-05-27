#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de configuration de l'environnement YOLOv12-Face pour Lightning.ai
Configure l'environnement optimal pour l'entraînement cloud
"""

import os
import sys
import logging
import subprocess
import platform
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.lightning_utils import setup_lightning_environment
from src.utils import set_seed, setup_directories

def setup_logging():
    """Configure le système de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('setup.log')
        ]
    )

def check_python_version():
    """Vérifie la version de Python"""
    logger = logging.getLogger(__name__)
    
    version = sys.version_info
    logger.info(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ requis")
        return False
    
    logger.info("✅ Version Python compatible")
    return True

def install_requirements():
    """Installe les dépendances"""
    logger = logging.getLogger(__name__)
    logger.info("📦 Installation des dépendances...")
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error(f"❌ Fichier requirements.txt non trouvé: {requirements_file}")
        return False
    
    try:
        # Mise à jour de pip d'abord
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        logger.info("✅ pip mis à jour")
        
        # Installation des dépendances
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                      check=True, capture_output=True)
        logger.info("✅ Dépendances installées")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur installation: {e}")
        return False

def setup_project_structure():
    """Crée la structure du projet"""
    logger = logging.getLogger(__name__)
    logger.info("📁 Configuration de la structure du projet...")
    
    base_dir = Path(__file__).parent.parent
    
    # Dossiers à créer
    directories = [
        "outputs/models",
        "outputs/logs", 
        "outputs/exports",
        "outputs/results",
        "datasets/downloads",
        "datasets/widerface",
        "datasets/custom",
        "configs/models",
        "notebooks/outputs",
        "temp"
    ]
    
    try:
        setup_directories(base_dir, directories)
        logger.info("✅ Structure du projet créée")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur création structure: {e}")
        return False

def check_gpu_availability():
    """Vérifie la disponibilité du GPU"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"🚀 GPU détecté: {gpu_name}")
            logger.info(f"   Nombre de GPUs: {gpu_count}")
            logger.info(f"   Mémoire GPU: {gpu_memory:.1f}GB")
            
            # Test rapide
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x)
            logger.info("✅ Test GPU réussi")
            
            return True
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 Apple Silicon GPU détecté")
            return True
            
        else:
            logger.warning("⚠️ Aucun GPU détecté, utilisation du CPU")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur vérification GPU: {e}")
        return False

def configure_environment_variables():
    """Configure les variables d'environnement optimales"""
    logger = logging.getLogger(__name__)
    logger.info("⚙️ Configuration des variables d'environnement...")
    
    # Variables d'optimisation pour Lightning.ai
    env_vars = {
        # PyTorch optimizations
        'TORCH_BACKENDS_CUDNN_BENCHMARK': 'True',
        'TORCH_BACKENDS_CUDNN_DETERMINISTIC': 'False',
        'CUDA_LAUNCH_BLOCKING': '0',
        
        # Memory optimizations
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4',
        
        # Lightning.ai specific
        'LIGHTNING_CLOUD_URL': 'https://lightning.ai',
        'WANDB_SILENT': 'true',  # Réduire les logs W&B
        
        # Reproductibilité
        'PYTHONHASHSEED': '42',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        
        # Optimisations diverses
        'TOKENIZERS_PARALLELISM': 'false',  # Éviter les warnings
        'TRANSFORMERS_CACHE': './temp/transformers',
        'HF_HOME': './temp/huggingface'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.debug(f"  {key} = {value}")
    
    logger.info("✅ Variables d'environnement configurées")

def verify_installation():
    """Vérifie que l'installation est correcte"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Vérification de l'installation...")
    
    try:
        # Test des imports critiques
        import torch
        import torchvision
        import ultralytics
        import cv2
        import numpy as np
        import yaml
        
        logger.info(f"✅ PyTorch {torch.__version__}")
        logger.info(f"✅ TorchVision {torchvision.__version__}")
        logger.info(f"✅ Ultralytics {ultralytics.__version__}")
        logger.info(f"✅ OpenCV {cv2.__version__}")
        logger.info(f"✅ NumPy {np.__version__}")
        
        # Test de création d'un modèle simple
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Modèle léger pour test
        logger.info("✅ YOLO model loading test")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import manquant: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur vérification: {e}")
        return False

def create_config_templates():
    """Crée des templates de configuration"""
    logger = logging.getLogger(__name__)
    logger.info("📝 Création des templates de configuration...")
    
    base_dir = Path(__file__).parent.parent
    configs_dir = base_dir / "configs"
    
    # Template de configuration d'entraînement rapide
    quick_config = {
        'project': {
            'name': 'yolov12-face-quick-test',
            'version': '1.0.0'
        },
        'model': {
            'size': 'n',  # Nano pour test rapide
            'num_classes': 1,
            'class_names': ['face']
        },
        'training': {
            'epochs': 10,  # Peu d'epochs pour test
            'batch_size': 8,
            'img_size': 416,  # Plus petit pour test
            'lr0': 0.01
        },
        'data': {
            'dataset': 'widerface',
            'path': './datasets',
            'cache': False  # Pas de cache pour test
        },
        'lightning': {
            'accelerator': 'auto',
            'devices': 1,
            'precision': '16-mixed'
        }
    }
    
    # Template de configuration pour production
    production_config = {
        'project': {
            'name': 'yolov12-face-production',
            'version': '1.0.0'
        },
        'model': {
            'size': 's',  # Small pour production
            'num_classes': 1,
            'class_names': ['face']
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'lr0': 0.001
        },
        'data': {
            'dataset': 'widerface',
            'path': './datasets',
            'cache': True
        },
        'lightning': {
            'accelerator': 'gpu',
            'devices': 1,
            'precision': '16-mixed'
        }
    }
    
    # Sauvegarder les templates
    import yaml
    
    with open(configs_dir / "quick_test.yaml", 'w') as f:
        yaml.dump(quick_config, f, default_flow_style=False)
    
    with open(configs_dir / "production.yaml", 'w') as f:
        yaml.dump(production_config, f, default_flow_style=False)
    
    logger.info("✅ Templates de configuration créés")

def main():
    """Fonction principale de setup"""
    print("🚀 Configuration YOLOv12-Face Lightning.ai")
    print("=" * 50)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    success = True
    
    # Étapes de configuration
    steps = [
        ("Vérification Python", check_python_version),
        ("Installation des dépendances", install_requirements),
        ("Structure du projet", setup_project_structure),
        ("Variables d'environnement", configure_environment_variables),
        ("Vérification GPU", check_gpu_availability),
        ("Vérification installation", verify_installation),
        ("Templates de configuration", create_config_templates)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n📋 {step_name}...")
        try:
            if not step_func():
                logger.error(f"❌ Échec: {step_name}")
                success = False
        except Exception as e:
            logger.error(f"❌ Erreur {step_name}: {e}")
            success = False
    
    # Configuration Lightning.ai spécifique
    try:
        setup_lightning_environment()
        set_seed(42)
        logger.info("✅ Configuration Lightning.ai terminée")
    except Exception as e:
        logger.error(f"❌ Erreur configuration Lightning: {e}")
        success = False
    
    # Résumé final
    print("\n" + "=" * 50)
    if success:
        print("✅ Configuration terminée avec succès!")
        print("\n📝 Prochaines étapes:")
        print("1. python lightning_main.py --config configs/quick_test.yaml")
        print("2. python scripts/download_datasets.py")
        print("3. python lightning_main.py --mode train")
    else:
        print("❌ Configuration échouée, vérifiez les logs")
        return 1
    
    print("=" * 50)
    return 0

if __name__ == "__main__":
    sys.exit(main())
