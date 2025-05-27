#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 YOLOv12-Face Lightning.ai - Quick Start Script
Script de démarrage rapide pour l'entraînement YOLOv12-Face
"""

import os
import sys
import time
import argparse
from pathlib import Path

def print_banner():
    """Affiche la bannière du projet"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 🚀 YOLOv12-Face Lightning.ai                 ║
    ║                                                              ║
    ║    Détection Faciale Haute Performance sur Infrastructure    ║
    ║                           Cloud                              ║
    ║                                                              ║
    ║    ⚡ Powered by Lightning.ai ⚡                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_installation():
    """Vérifie rapidement l'installation"""
    print("🔍 Vérification rapide de l'installation...")
    
    try:
        # Test imports critiques
        import torch
        import ultralytics
        import yaml
        
        # Test modules du projet
        sys.path.append(str(Path(__file__).parent))
        from src import YOLOv12FaceTrainer, DataManager, ModelManager
        
        print("✅ Installation vérifiée")
        return True
        
    except ImportError as e:
        print(f"❌ Import manquant: {e}")
        print("🔧 Exécutez: python scripts/setup_environment.py")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def interactive_config():
    """Configuration interactive"""
    print("\n⚙️ Configuration Interactive")
    print("-" * 40)
    
    config = {}
    
    # Choix du modèle
    models = {
        '1': ('n', 'Nano - Ultra-rapide (120 FPS, 6.5MB)'),
        '2': ('s', 'Small - Recommandé (85 FPS, 22MB)'),
        '3': ('m', 'Medium - Haute précision (60 FPS, 52MB)'),
        '4': ('l', 'Large - Précision max (45 FPS, 88MB)')
    }
    
    print("🎯 Choisissez la taille du modèle:")
    for key, (size, desc) in models.items():
        print(f"  {key}. YOLOv12{size} - {desc}")
    
    while True:
        choice = input("Votre choix (1-4) [2]: ").strip() or "2"
        if choice in models:
            config['model_size'] = models[choice][0]
            break
        print("⚠️ Choix invalide, réessayez")
    
    # Nombre d'epochs
    print(f"\n📈 Nombre d'epochs d'entraînement:")
    print(f"  - Test rapide: 20 epochs (~10 min)")
    print(f"  - Entraînement standard: 100 epochs (~1-2h)")
    print(f"  - Production: 150+ epochs (~3-4h)")
    
    while True:
        try:
            epochs = input("Epochs [100]: ").strip() or "100"
            config['epochs'] = int(epochs)
            if config['epochs'] > 0:
                break
            print("⚠️ Le nombre d'epochs doit être positif")
        except ValueError:
            print("⚠️ Veuillez entrer un nombre valide")
    
    # Batch size
    print(f"\n📦 Taille du batch:")
    print(f"  - Petit GPU/CPU: 8")
    print(f"  - GPU standard: 16")
    print(f"  - GPU puissant: 32")
    
    while True:
        try:
            batch = input("Batch size [16]: ").strip() or "16"
            config['batch_size'] = int(batch)
            if config['batch_size'] > 0:
                break
            print("⚠️ La taille du batch doit être positive")
        except ValueError:
            print("⚠️ Veuillez entrer un nombre valide")
    
    # Taille d'image
    print(f"\n🖼️ Taille des images:")
    print(f"  - Rapide: 416px")
    print(f"  - Standard: 640px")
    print(f"  - Haute qualité: 832px")
    
    img_sizes = {'1': 416, '2': 640, '3': 832}
    print("Votre choix:")
    for key, size in img_sizes.items():
        print(f"  {key}. {size}px")
    
    while True:
        choice = input("Choix (1-3) [2]: ").strip() or "2"
        if choice in img_sizes:
            config['img_size'] = img_sizes[choice]
            break
        print("⚠️ Choix invalide")
    
    return config

def create_custom_config(config, config_path):
    """Crée un fichier de configuration personnalisé"""
    
    custom_config = f"""# Configuration personnalisée YOLOv12-Face
# Générée automatiquement par quick_start.py

project:
  name: "yolov12-face-custom"
  version: "1.0.0"
  description: "Configuration personnalisée YOLOv12-Face"
  author: "User"

model:
  size: "{config['model_size']}"
  num_classes: 1
  class_names: ["face"]

training:
  epochs: {config['epochs']}
  batch_size: {config['batch_size']}
  img_size: {config['img_size']}
  optimizer: "AdamW"
  lr0: 0.001
  lrf: 0.01

data:
  dataset: "widerface"
  path: "./datasets"
  cache: true

lightning:
  accelerator: "auto"
  devices: 1
  precision: "16-mixed"
  save_every_n_epochs: 10

output:
  base_dir: "./outputs"
  run_name: "custom_{{timestamp}}"

environment:
  seed: 42
  num_workers: 4
"""
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(custom_config)
    
    print(f"✅ Configuration sauvegardée: {config_path}")

def estimate_training_time(config):
    """Estime le temps d'entraînement"""
    
    # Facteurs de temps (minutes par epoch)
    time_factors = {
        'n': 0.5,  # Nano
        's': 1.0,  # Small 
        'm': 2.0,  # Medium
        'l': 3.5   # Large
    }
    
    # Facteur taille d'image
    img_factor = {
        416: 0.7,
        640: 1.0,
        832: 1.5
    }
    
    # Facteur batch size (inverse)
    batch_factor = 16 / config['batch_size']
    
    base_time = time_factors.get(config['model_size'], 1.0)
    img_time = img_factor.get(config['img_size'], 1.0)
    
    estimated_minutes = config['epochs'] * base_time * img_time * batch_factor
    
    hours = int(estimated_minutes // 60)
    minutes = int(estimated_minutes % 60)
    
    return hours, minutes

def show_summary(config):
    """Affiche un résumé de la configuration"""
    print("\n📋 RÉSUMÉ DE LA CONFIGURATION")
    print("=" * 50)
    print(f"🎯 Modèle: YOLOv12{config['model_size']}")
    print(f"📈 Epochs: {config['epochs']}")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"🖼️ Taille d'image: {config['img_size']}px")
    
    hours, minutes = estimate_training_time(config)
    if hours > 0:
        print(f"⏱️ Temps estimé: ~{hours}h{minutes:02d}min")
    else:
        print(f"⏱️ Temps estimé: ~{minutes}min")
    
    # Recommandations
    print(f"\n💡 Recommandations:")
    if config['model_size'] == 'n':
        print(f"   🚀 Excellent choix pour tests rapides et déploiement mobile")
    elif config['model_size'] == 's':
        print(f"   ⚖️ Bon équilibre vitesse/précision, recommandé pour la production")
    elif config['model_size'] in ['m', 'l']:
        print(f"   🎯 Haute précision, idéal pour applications critiques")
    
    if config['epochs'] < 50:
        print(f"   ⚠️ Peu d'epochs: bon pour tests, considérez plus pour production")
    elif config['epochs'] > 200:
        print(f"   ⏳ Beaucoup d'epochs: excellent pour précision maximale")

def run_training(config_path):
    """Lance l'entraînement"""
    print(f"\n🚀 Lancement de l'entraînement...")
    print(f"📄 Configuration: {config_path}")
    
    # Commande d'entraînement
    cmd = f"python lightning_main.py --config {config_path}"
    print(f"💻 Commande: {cmd}")
    
    confirm = input(f"\n▶️ Lancer l'entraînement maintenant? (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes', 'oui']:
        print(f"🏋️ Entraînement en cours...")
        print(f"📊 Suivez les métriques dans outputs/logs/")
        print(f"⏹️ Utilisez Ctrl+C pour arrêter")
        
        try:
            os.system(cmd)
        except KeyboardInterrupt:
            print(f"\n⏹️ Entraînement interrompu par l'utilisateur")
    else:
        print(f"📝 Pour lancer plus tard:")
        print(f"   {cmd}")

def main():
    """Fonction principale"""
    print_banner()
    
    parser = argparse.ArgumentParser(description='Quick Start YOLOv12-Face')
    parser.add_argument('--skip-check', action='store_true', help='Ignorer la vérification')
    parser.add_argument('--auto-config', type=str, help='Utiliser une config existante')
    args = parser.parse_args()
    
    # Vérification de l'installation
    if not args.skip_check and not check_installation():
        print(f"\n🔧 Installation requise:")
        print(f"   python scripts/setup_environment.py")
        return 1
    
    # Configuration
    if args.auto_config:
        config_path = args.auto_config
        if not Path(config_path).exists():
            print(f"❌ Configuration non trouvée: {config_path}")
            return 1
        print(f"✅ Utilisation de la configuration: {config_path}")
    else:
        # Configuration interactive
        config = interactive_config()
        show_summary(config)
        
        # Sauvegarder la configuration
        config_path = "custom_config.yaml"
        create_custom_config(config, config_path)
    
    # Vérifier les données
    datasets_dir = Path("datasets")
    if not datasets_dir.exists() or not any(datasets_dir.iterdir()):
        print(f"\n📥 Dataset requis:")
        print(f"   python scripts/download_datasets.py --dataset widerface")
        
        download = input(f"Télécharger maintenant? (y/N): ").strip().lower()
        if download in ['y', 'yes', 'oui']:
            print(f"📥 Téléchargement en cours...")
            os.system("python scripts/download_datasets.py --dataset widerface")
        else:
            print(f"⚠️ Téléchargez les données avant l'entraînement")
            return 1
    
    # Lancer l'entraînement
    run_training(config_path)
    
    print(f"\n🎉 Quick Start terminé!")
    print(f"📊 Vérifiez les résultats dans outputs/")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n👋 Quick Start interrompu")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        sys.exit(1)
