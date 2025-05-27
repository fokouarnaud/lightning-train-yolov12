#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'export des modèles YOLOv12-Face
Export vers ONNX, TensorRT, CoreML, etc. pour production et mobile
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
import torch

def setup_logging(verbose: bool = False):
    """Configure le système de logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('export_models.log')
        ]
    )

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Export des modèles YOLOv12-Face vers différents formats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Chemin vers le modèle PyTorch (.pt)'
    )
    
    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        default=['onnx'],
        choices=['onnx', 'torchscript', 'coreml', 'tflite', 'engine', 'pb', 'saved_model'],
        help='Formats d\'export à générer'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/exports',
        help='Répertoire de sortie pour les modèles exportés'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Taille des images d\'entrée'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Taille de batch pour l\'export'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device pour l\'export'
    )
    
    parser.add_argument(
        '--half',
        action='store_true',
        help='Utiliser la précision FP16'
    )
    
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Export avec tailles dynamiques (ONNX)'
    )
    
    parser.add_argument(
        '--simplify',
        action='store_true',
        default=True,
        help='Simplifier le modèle ONNX'
    )
    
    parser.add_argument(
        '--optimize-for-mobile',
        action='store_true',
        help='Optimisations spécifiques mobile'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmarker les modèles exportés'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mode verbose avec plus de logs'
    )
    
    return parser.parse_args()

def validate_model_path(model_path: str) -> bool:
    """Valide le chemin du modèle"""
    logger = logging.getLogger(__name__)
    
    path = Path(model_path)
    
    if not path.exists():
        logger.error(f"❌ Modèle non trouvé: {model_path}")
        return False
    
    if not path.suffix == '.pt':
        logger.error(f"❌ Format de modèle non supporté: {path.suffix}")
        return False
    
    # Vérifier que le fichier n'est pas vide
    if path.stat().st_size == 0:
        logger.error(f"❌ Fichier modèle vide: {model_path}")
        return False
    
    logger.info(f"✅ Modèle valide: {model_path}")
    return True

def export_onnx(model: YOLO, output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Export vers ONNX
    
    Args:
        model: Modèle YOLO
        output_dir: Répertoire de sortie
        args: Arguments de ligne de commande
        
    Returns:
        Informations sur l'export
    """
    logger = logging.getLogger(__name__)
    logger.info("📦 Export ONNX...")
    
    try:
        start_time = time.time()
        
        # Paramètres d'export ONNX
        export_args = {
            'format': 'onnx',
            'imgsz': args.img_size,
            'device': args.device,
            'half': args.half,
            'dynamic': args.dynamic,
            'simplify': args.simplify,
            'opset': 12,  # Version ONNX
        }
        
        # Optimisations pour mobile
        if args.optimize_for_mobile:
            export_args.update({
                'opset': 11,  # Meilleure compatibilité mobile
                'simplify': True,
                'dynamic': False,  # Tailles fixes pour mobile
            })
        
        # Export
        exported_model = model.export(**export_args)
        export_time = time.time() - start_time
        
        # Informations sur le fichier exporté
        model_path = Path(exported_model)
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        
        logger.info(f"✅ Export ONNX terminé: {model_path.name}")
        logger.info(f"   Taille: {model_size:.2f} MB")
        logger.info(f"   Temps: {export_time:.2f}s")
        
        return {
            'format': 'onnx',
            'path': str(model_path),
            'size_mb': model_size,
            'export_time': export_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur export ONNX: {e}")
        return {
            'format': 'onnx',
            'success': False,
            'error': str(e)
        }

def export_torchscript(model: YOLO, output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Export vers TorchScript"""
    logger = logging.getLogger(__name__)
    logger.info("📦 Export TorchScript...")
    
    try:
        start_time = time.time()
        
        exported_model = model.export(
            format='torchscript',
            imgsz=args.img_size,
            device=args.device,
            half=args.half
        )
        
        export_time = time.time() - start_time
        model_path = Path(exported_model)
        model_size = model_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ Export TorchScript terminé: {model_path.name}")
        logger.info(f"   Taille: {model_size:.2f} MB")
        logger.info(f"   Temps: {export_time:.2f}s")
        
        return {
            'format': 'torchscript',
            'path': str(model_path),
            'size_mb': model_size,
            'export_time': export_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur export TorchScript: {e}")
        return {
            'format': 'torchscript',
            'success': False,
            'error': str(e)
        }

def export_coreml(model: YOLO, output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Export vers CoreML (iOS)"""
    logger = logging.getLogger(__name__)
    logger.info("📦 Export CoreML...")
    
    try:
        start_time = time.time()
        
        # CoreML est optimisé pour iOS
        exported_model = model.export(
            format='coreml',
            imgsz=args.img_size,
            device='cpu',  # CoreML nécessite CPU
            half=False,    # CoreML gère automatiquement la précision
            nms=True       # Inclure NMS dans le modèle
        )
        
        export_time = time.time() - start_time
        model_path = Path(exported_model)
        
        # CoreML produit un dossier .mlmodel
        if model_path.is_dir():
            model_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
        else:
            model_size = model_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ Export CoreML terminé: {model_path.name}")
        logger.info(f"   Taille: {model_size:.2f} MB")
        logger.info(f"   Temps: {export_time:.2f}s")
        
        return {
            'format': 'coreml',
            'path': str(model_path),
            'size_mb': model_size,
            'export_time': export_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur export CoreML: {e}")
        return {
            'format': 'coreml',
            'success': False,
            'error': str(e)
        }

def export_tflite(model: YOLO, output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Export vers TensorFlow Lite (Android)"""
    logger = logging.getLogger(__name__)
    logger.info("📦 Export TensorFlow Lite...")
    
    try:
        start_time = time.time()
        
        # TFLite pour Android
        export_args = {
            'format': 'tflite',
            'imgsz': args.img_size,
            'device': 'cpu',
            'int8': args.optimize_for_mobile,  # Quantification INT8 pour mobile
        }
        
        exported_model = model.export(**export_args)
        export_time = time.time() - start_time
        model_path = Path(exported_model)
        model_size = model_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ Export TFLite terminé: {model_path.name}")
        logger.info(f"   Taille: {model_size:.2f} MB")
        logger.info(f"   Temps: {export_time:.2f}s")
        
        return {
            'format': 'tflite',
            'path': str(model_path),
            'size_mb': model_size,
            'export_time': export_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur export TFLite: {e}")
        return {
            'format': 'tflite',
            'success': False,
            'error': str(e)
        }

def benchmark_model(model_path: str, img_size: int, device: str) -> Dict[str, float]:
    """
    Benchmark d'un modèle exporté
    
    Args:
        model_path: Chemin vers le modèle
        img_size: Taille des images
        device: Device de test
        
    Returns:
        Métriques de performance
    """
    logger = logging.getLogger(__name__)
    logger.info(f"⏱️ Benchmark: {Path(model_path).name}")
    
    try:
        import numpy as np
        
        # Créer des données de test
        dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
        
        # Benchmark selon le format
        if model_path.endswith('.onnx'):
            return benchmark_onnx(model_path, dummy_input)
        elif model_path.endswith('.pt'):
            return benchmark_torchscript(model_path, dummy_input, device)
        else:
            logger.warning(f"⚠️ Benchmark non supporté pour: {model_path}")
            return {}
            
    except Exception as e:
        logger.error(f"❌ Erreur benchmark: {e}")
        return {}

def benchmark_onnx(model_path: str, dummy_input: np.ndarray) -> Dict[str, float]:
    """Benchmark d'un modèle ONNX"""
    try:
        import onnxruntime as ort
        
        # Créer la session ONNX
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(5):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(20):
            start = time.time()
            session.run(None, {input_name: dummy_input})
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        fps = 1000 / avg_time
        
        return {
            'avg_inference_ms': avg_time,
            'fps': fps,
            'min_inference_ms': min(times) * 1000,
            'max_inference_ms': max(times) * 1000
        }
        
    except ImportError:
        logging.getLogger(__name__).warning("⚠️ onnxruntime non installé")
        return {}
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Erreur benchmark ONNX: {e}")
        return {}

def benchmark_torchscript(model_path: str, dummy_input: np.ndarray, device: str) -> Dict[str, float]:
    """Benchmark d'un modèle TorchScript"""
    try:
        # Charger le modèle TorchScript
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        
        # Convertir l'input
        tensor_input = torch.from_numpy(dummy_input).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(tensor_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                model(tensor_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        fps = 1000 / avg_time
        
        return {
            'avg_inference_ms': avg_time,
            'fps': fps,
            'min_inference_ms': min(times) * 1000,
            'max_inference_ms': max(times) * 1000
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Erreur benchmark TorchScript: {e}")
        return {}

def main():
    """Fonction principale"""
    args = parse_args()
    
    print("📦 YOLOv12-Face Model Exporter")
    print("=" * 50)
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Valider le modèle d'entrée
    if not validate_model_path(args.model_path):
        return 1
    
    # Créer le répertoire de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger le modèle
    try:
        logger.info(f"🔄 Chargement du modèle: {args.model_path}")
        model = YOLO(args.model_path)
        logger.info("✅ Modèle chargé")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
        return 1
    
    # Mappage des fonctions d'export
    export_functions = {
        'onnx': export_onnx,
        'torchscript': export_torchscript,
        'coreml': export_coreml,
        'tflite': export_tflite,
    }
    
    # Exporter vers les formats demandés
    export_results = []
    
    for format_name in args.formats:
        logger.info(f"\n📦 Export {format_name.upper()}...")
        
        if format_name in export_functions:
            result = export_functions[format_name](model, output_dir, args)
            export_results.append(result)
        else:
            logger.warning(f"⚠️ Format non supporté: {format_name}")
            try:
                # Essayer l'export générique ultralytics
                exported_model = model.export(
                    format=format_name,
                    imgsz=args.img_size,
                    device=args.device,
                    half=args.half
                )
                
                model_path = Path(exported_model)
                model_size = model_path.stat().st_size / (1024 * 1024)
                
                result = {
                    'format': format_name,
                    'path': str(model_path),
                    'size_mb': model_size,
                    'success': True
                }
                export_results.append(result)
                logger.info(f"✅ Export {format_name} terminé")
                
            except Exception as e:
                logger.error(f"❌ Erreur export {format_name}: {e}")
                export_results.append({
                    'format': format_name,
                    'success': False,
                    'error': str(e)
                })
    
    # Benchmark des modèles exportés
    if args.benchmark:
        logger.info("\n⏱️ Benchmark des modèles exportés...")
        for result in export_results:
            if result.get('success') and 'path' in result:
                bench_result = benchmark_model(
                    result['path'], 
                    args.img_size, 
                    args.device
                )
                result.update(bench_result)
    
    # Résumé final
    print("\n" + "=" * 50)
    print("📋 RÉSUMÉ DES EXPORTS")
    print("=" * 50)
    
    successful_exports = 0
    total_exports = len(export_results)
    
    for result in export_results:
        format_name = result['format'].upper()
        
        if result.get('success'):
            successful_exports += 1
            print(f"✅ {format_name}")
            print(f"   📁 {result.get('path', 'N/A')}")
            print(f"   📊 {result.get('size_mb', 0):.2f} MB")
            
            if 'avg_inference_ms' in result:
                print(f"   ⏱️  {result['avg_inference_ms']:.2f}ms ({result['fps']:.1f} FPS)")
        else:
            print(f"❌ {format_name}")
            print(f"   Erreur: {result.get('error', 'Inconnue')}")
        print()
    
    print(f"📊 Succès: {successful_exports}/{total_exports}")
    
    if successful_exports > 0:
        print("\n📝 Prochaines étapes:")
        print("1. Tester les modèles exportés")
        print("2. Intégrer dans votre application")
        print("3. Vérifier les performances en production")
    
    print("=" * 50)
    return 0 if successful_exports > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
