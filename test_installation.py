#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test rapide pour YOLOv12-Face Lightning.ai
Vérifie que l'installation et la configuration sont correctes
"""

import sys
import os
import time
import traceback
from pathlib import Path

def test_imports():
    """Teste tous les imports nécessaires"""
    print("🔍 Test des imports...")
    
    tests = [
        ("Python standard", ["os", "sys", "pathlib", "time", "json", "yaml"]),
        ("Scientifique", ["numpy", "pandas", "matplotlib", "seaborn"]),
        ("Vision", ["cv2", "PIL"]),
        ("Deep Learning", ["torch", "torchvision"]),
        ("YOLO", ["ultralytics"]),
        ("Projet", ["src"])
    ]
    
    results = {}
    
    for category, modules in tests:
        print(f"  📦 {category}:")
        category_results = []
        
        for module in modules:
            try:
                if module == "src":
                    # Test import du projet
                    sys.path.append(str(Path(__file__).parent))
                    import src
                    from src import YOLOv12FaceTrainer, DataManager, ModelManager
                else:
                    __import__(module)
                
                print(f"    ✅ {module}")
                category_results.append(True)
                
            except ImportError as e:
                print(f"    ❌ {module}: {e}")
                category_results.append(False)
            except Exception as e:
                print(f"    ⚠️ {module}: {e}")
                category_results.append(False)
        
        results[category] = category_results
    
    # Résumé
    total_tests = sum(len(tests) for _, tests in tests)
    passed_tests = sum(sum(results.values(), []))
    
    print(f"\n📊 Résumé imports: {passed_tests}/{total_tests} réussis")
    return passed_tests == total_tests

def test_gpu():
    """Teste la disponibilité du GPU"""
    print("\n🚀 Test GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"  ✅ GPU disponible: {device_name}")
            print(f"  📊 Nombre de GPUs: {device_count}")
            print(f"  💾 Mémoire GPU: {memory:.1f}GB")
            
            # Test simple
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x)
            print(f"  ✅ Test calcul GPU réussi")
            
            return True
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ✅ Apple Silicon GPU (MPS) disponible")
            return True
            
        else:
            print(f"  ⚠️ Aucun GPU détecté, utilisation CPU")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur test GPU: {e}")
        return False

def test_directories():
    """Teste la structure des dossiers"""
    print("\n📁 Test structure des dossiers...")
    
    required_dirs = [
        "src",
        "configs", 
        "scripts",
        "notebooks",
        "datasets",
        "outputs"
    ]
    
    base_dir = Path(__file__).parent
    results = []
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
            results.append(True)
        else:
            print(f"  ❌ {dir_name}/ manquant")
            results.append(False)
    
    # Créer les dossiers manquants
    missing_dirs = [required_dirs[i] for i, exists in enumerate(results) if not exists]
    if missing_dirs:
        print(f"  🔧 Création des dossiers manquants...")
        for dir_name in missing_dirs:
            (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
            print(f"    ✅ {dir_name}/ créé")
    
    return True

def test_config():
    """Teste le chargement des configurations"""
    print("\n⚙️ Test configurations...")
    
    config_files = ["config.yaml", "configs/quick_test.yaml", "configs/production.yaml"]
    base_dir = Path(__file__).parent
    
    try:
        import yaml
        
        for config_file in config_files:
            config_path = base_dir / config_file
            
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    print(f"  ✅ {config_file}")
                except Exception as e:
                    print(f"  ❌ {config_file}: {e}")
                    return False
            else:
                print(f"  ⚠️ {config_file}: non trouvé")
        
        return True
        
    except ImportError:
        print(f"  ❌ PyYAML non disponible")
        return False

def test_ultralytics():
    """Teste l'installation d'Ultralytics YOLO"""
    print("\n🎯 Test Ultralytics YOLO...")
    
    try:
        from ultralytics import YOLO
        
        # Test de création d'un modèle simple
        print("  🔄 Test création modèle...")
        model = YOLO('yolov8n.pt')  # Modèle léger pour test
        print("  ✅ Modèle YOLO créé")
        
        # Test d'information du modèle
        try:
            info = model.info()
            print("  ✅ Informations modèle récupérées")
        except:
            print("  ⚠️ Info modèle non disponible (normal)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur Ultralytics: {e}")
        return False

def test_project_modules():
    """Teste les modules du projet"""
    print("\n🏗️ Test modules du projet...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        
        # Test DataManager
        print("  🔄 Test DataManager...")
        from src.data_manager import DataManager
        
        data_config = {
            'dataset': 'widerface',
            'path': './datasets'
        }
        data_manager = DataManager(data_config)
        print("  ✅ DataManager initialisé")
        
        # Test ModelManager
        print("  🔄 Test ModelManager...")
        from src.model_manager import ModelManager
        
        model_config = {
            'size': 's',
            'num_classes': 1,
            'class_names': ['face']
        }
        model_manager = ModelManager(model_config)
        print("  ✅ ModelManager initialisé")
        
        # Test LightningLogger
        print("  🔄 Test LightningLogger...")
        from src.lightning_utils import LightningLogger
        
        logger = LightningLogger("test", 10)
        print("  ✅ LightningLogger initialisé")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur modules projet: {e}")
        traceback.print_exc()
        return False

def test_quick_run():
    """Teste un run rapide du pipeline"""
    print("\n⚡ Test run rapide...")
    
    try:
        # Test du script principal
        main_script = Path(__file__).parent / "lightning_main.py"
        
        if main_script.exists():
            print("  ✅ Script principal trouvé")
            
            # Test import du script principal
            sys.path.append(str(Path(__file__).parent))
            import lightning_main
            print("  ✅ Script principal importé")
            
            return True
        else:
            print("  ❌ Script principal non trouvé")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur test run: {e}")
        return False

def create_test_report(results):
    """Crée un rapport de test"""
    print("\n📋 RAPPORT DE TEST")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Tests réussis: {passed_tests}/{total_tests}")
    print(f"Taux de réussite: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDétail:")
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    if passed_tests == total_tests:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Le projet est prêt à être utilisé")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests échoués")
        print("🔧 Vérifiez les erreurs ci-dessus")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 YOLOv12-Face Lightning.ai - Tests Système")
    print("=" * 60)
    
    start_time = time.time()
    
    # Liste des tests à exécuter
    tests = [
        ("Imports", test_imports),
        ("GPU", test_gpu), 
        ("Dossiers", test_directories),
        ("Configurations", test_config),
        ("Ultralytics", test_ultralytics),
        ("Modules Projet", test_project_modules),
        ("Run Rapide", test_quick_run)
    ]
    
    results = {}
    
    # Exécuter tous les tests
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Erreur inattendue dans {test_name}: {e}")
            results[test_name] = False
    
    # Créer le rapport
    success = create_test_report(results)
    
    # Temps d'exécution
    duration = time.time() - start_time
    print(f"\n⏱️ Tests terminés en {duration:.2f}s")
    
    if success:
        print("\n🚀 Prochaines étapes:")
        print("1. python scripts/setup_environment.py")
        print("2. python scripts/download_datasets.py --dataset widerface")
        print("3. python lightning_main.py --config configs/quick_test.yaml")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
