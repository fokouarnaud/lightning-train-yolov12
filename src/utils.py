#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires généraux pour YOLOv12-Face Lightning.ai
Fonctions communes et helpers
"""

import os
import logging
import random
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Définit la graine pour la reproductibilité
    
    Args:
        seed: Valeur de la graine
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"🌱 Graine définie: {seed}")

def setup_directories(base_dir: Union[str, Path], subdirs: List[str]) -> None:
    """
    Crée une structure de dossiers
    
    Args:
        base_dir: Dossier de base
        subdirs: Liste des sous-dossiers à créer
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Structure créée: {base_path}")

def get_device() -> torch.device:
    """
    Détermine le meilleur device disponible
    
    Returns:
        Device PyTorch optimal
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"🚀 GPU détecté: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("🍎 Apple Silicon GPU détecté")
    else:
        device = torch.device('cpu')
        logger.info("💻 Utilisation du CPU")
    
    return device

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Compte les paramètres d'un modèle
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Tuple (paramètres totaux, paramètres entraînables)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 Paramètres: {total_params:,} total, {trainable_params:,} entraînables")
    return total_params, trainable_params

def format_time(seconds: float) -> str:
    """
    Formate une durée en secondes
    
    Args:
        seconds: Durée en secondes
        
    Returns:
        Chaîne formatée (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def format_size(bytes_size: int) -> str:
    """
    Formate une taille en bytes
    
    Args:
        bytes_size: Taille en bytes
        
    Returns:
        Chaîne formatée (KB, MB, GB)
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

def validate_image_formats(image_dir: Path, valid_extensions: List[str] = None) -> List[Path]:
    """
    Valide et filtre les images par format
    
    Args:
        image_dir: Dossier contenant les images
        valid_extensions: Extensions valides (par défaut: jpg, jpeg, png)
        
    Returns:
        Liste des chemins d'images valides
    """
    if valid_extensions is None:
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    valid_images = []
    
    for ext in valid_extensions:
        valid_images.extend(image_dir.glob(f"*{ext}"))
        valid_images.extend(image_dir.glob(f"*{ext.upper()}"))
    
    logger.info(f"📸 {len(valid_images)} images valides trouvées dans {image_dir}")
    return sorted(valid_images)

def resize_image_keep_aspect(image: np.ndarray, target_size: int) -> Tuple[np.ndarray, float]:
    """
    Redimensionne une image en gardant les proportions
    
    Args:
        image: Image OpenCV (numpy array)
        target_size: Taille cible (carré)
        
    Returns:
        Tuple (image redimensionnée, facteur d'échelle)
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Padding pour obtenir une image carrée
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return padded, scale

def draw_detections(image: np.ndarray, detections: List[dict], 
                   class_names: List[str] = None, 
                   confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Dessine les détections sur une image
    
    Args:
        image: Image OpenCV
        detections: Liste des détections [{'bbox': [x1,y1,x2,y2], 'conf': float, 'class': int}]
        class_names: Noms des classes
        confidence_threshold: Seuil de confiance minimum
        
    Returns:
        Image avec les détections dessinées
    """
    if class_names is None:
        class_names = ['face']
    
    result_image = image.copy()
    h, w = image.shape[:2]
    
    colors = [
        (255, 0, 0),    # Rouge
        (0, 255, 0),    # Vert
        (0, 0, 255),    # Bleu
        (255, 255, 0),  # Jaune
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    for det in detections:
        conf = det.get('conf', 0)
        if conf < confidence_threshold:
            continue
        
        bbox = det.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        class_id = det.get('class', 0)
        
        # Couleur selon la classe
        color = colors[class_id % len(colors)]
        
        # Dessiner la boîte
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Label avec confiance
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        label = f"{class_name}: {conf:.2f}"
        
        # Taille du texte
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                    font_scale, thickness)
        
        # Fond du texte
        cv2.rectangle(result_image, (x1, y1 - text_h - baseline - 4), 
                     (x1 + text_w, y1), color, -1)
        
        # Texte
        cv2.putText(result_image, label, (x1, y1 - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return result_image

def create_training_plots(metrics_history: List[dict], save_path: Path) -> None:
    """
    Crée des graphiques de suivi de l'entraînement
    
    Args:
        metrics_history: Historique des métriques
        save_path: Chemin de sauvegarde
    """
    if not metrics_history:
        logger.warning("⚠️ Aucune métrique à tracer")
        return
    
    # Extraire les données
    epochs = [entry['epoch'] for entry in metrics_history]
    
    # Métriques communes
    metrics_to_plot = ['loss', 'mAP50', 'mAP50-95', 'precision', 'recall']
    available_metrics = {}
    
    for metric in metrics_to_plot:
        values = []
        for entry in metrics_history:
            metric_dict = entry.get('metrics', {})
            # Essayer différentes variantes du nom de métrique
            value = metric_dict.get(metric) or metric_dict.get(f'val/{metric}') or metric_dict.get(f'train/{metric}')
            if value is not None:
                values.append(value)
            else:
                values.append(None)
        
        if any(v is not None for v in values):
            available_metrics[metric] = values
    
    if not available_metrics:
        logger.warning("⚠️ Aucune métrique reconnue trouvée")
        return
    
    # Créer les graphiques
    n_metrics = len(available_metrics)
    cols = 2
    rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if rows == 1:
        axes = [axes] if n_metrics == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (metric_name, values) in enumerate(available_metrics.items()):
        ax = axes[i]
        
        # Filtrer les valeurs None
        valid_indices = [j for j, v in enumerate(values) if v is not None]
        valid_epochs = [epochs[j] for j in valid_indices]
        valid_values = [values[j] for j in valid_indices]
        
        if valid_values:
            ax.plot(valid_epochs, valid_values, marker='o', linewidth=2, markersize=4)
            ax.set_title(f'{metric_name.upper()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
            # Ajouter la valeur finale
            if valid_values:
                final_value = valid_values[-1]
                ax.text(0.02, 0.98, f'Final: {final_value:.4f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Masquer les axes inutilisés
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Graphiques sauvegardés: {save_path}")

def check_dataset_health(dataset_path: Path) -> dict:
    """
    Vérifie la santé d'un dataset YOLO
    
    Args:
        dataset_path: Chemin vers le dataset
        
    Returns:
        Rapport de santé du dataset
    """
    health_report = {
        'status': 'healthy',
        'issues': [],
        'stats': {},
        'recommendations': []
    }
    
    # Vérifier la structure
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    missing_dirs = []
    
    for req_dir in required_dirs:
        dir_path = dataset_path / req_dir
        if not dir_path.exists():
            missing_dirs.append(req_dir)
    
    if missing_dirs:
        health_report['status'] = 'unhealthy'
        health_report['issues'].append(f"Dossiers manquants: {missing_dirs}")
    
    # Statistiques des fichiers
    for split in ['train', 'val']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if images_dir.exists() and labels_dir.exists():
            images = validate_image_formats(images_dir)
            labels = list(labels_dir.glob('*.txt'))
            
            health_report['stats'][split] = {
                'images': len(images),
                'labels': len(labels),
                'ratio': len(labels) / len(images) if images else 0
            }
            
            # Vérifier la correspondance images/labels
            if len(images) != len(labels):
                health_report['issues'].append(
                    f"{split}: {len(images)} images vs {len(labels)} labels"
                )
                health_report['status'] = 'warning'
    
    # Recommandations
    total_images = sum(stats.get('images', 0) for stats in health_report['stats'].values())
    if total_images < 100:
        health_report['recommendations'].append("Dataset très petit, considérer plus d'images")
    elif total_images < 1000:
        health_report['recommendations'].append("Dataset petit, résultats limités possibles")
    
    logger.info(f"🏥 Santé du dataset: {health_report['status']}")
    return health_report

def cleanup_temp_files(base_dir: Path, patterns: List[str] = None) -> None:
    """
    Nettoie les fichiers temporaires
    
    Args:
        base_dir: Dossier de base
        patterns: Motifs de fichiers à supprimer
    """
    if patterns is None:
        patterns = ['*.tmp', '*.log', '*.cache', '__pycache__', '.DS_Store']
    
    deleted_count = 0
    
    for pattern in patterns:
        for file_path in base_dir.rglob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Impossible de supprimer {file_path}: {e}")
    
    logger.info(f"🧹 {deleted_count} éléments temporaires supprimés")

def calculate_model_flops(model: torch.nn.Module, input_size: Tuple[int, int, int, int]) -> int:
    """
    Calcule approximativement les FLOPs d'un modèle
    
    Args:
        model: Modèle PyTorch
        input_size: Taille d'entrée (batch, channels, height, width)
        
    Returns:
        Nombre approximatif de FLOPs
    """
    try:
        # Estimation simple basée sur les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimation grossière: ~2 FLOPs par paramètre par pixel
        batch_size, channels, height, width = input_size
        pixels = height * width
        estimated_flops = total_params * pixels * 2
        
        logger.info(f"📊 FLOPs estimés: {estimated_flops / 1e9:.2f}G")
        return estimated_flops
        
    except Exception as e:
        logger.warning(f"⚠️ Erreur calcul FLOPs: {e}")
        return 0

def export_config_summary(config: dict, output_path: Path) -> None:
    """
    Exporte un résumé de configuration lisible
    
    Args:
        config: Configuration complète
        output_path: Chemin de sortie
    """
    summary_lines = [
        "# YOLOv12-Face Configuration Summary",
        "=" * 50,
        "",
        f"**Projet**: {config.get('project', {}).get('name', 'N/A')}",
        f"**Version**: {config.get('project', {}).get('version', 'N/A')}",
        f"**Auteur**: {config.get('project', {}).get('author', 'N/A')}",
        "",
        "## Modèle",
        f"- Taille: YOLOv12{config.get('model', {}).get('size', 'N/A')}",
        f"- Classes: {config.get('model', {}).get('num_classes', 'N/A')}",
        f"- Noms: {', '.join(config.get('model', {}).get('class_names', []))}",
        "",
        "## Entraînement",
        f"- Epochs: {config.get('training', {}).get('epochs', 'N/A')}",
        f"- Batch size: {config.get('training', {}).get('batch_size', 'N/A')}",
        f"- Image size: {config.get('training', {}).get('img_size', 'N/A')}",
        f"- Optimiseur: {config.get('training', {}).get('optimizer', 'N/A')}",
        f"- Learning rate: {config.get('training', {}).get('lr0', 'N/A')}",
        "",
        "## Données",
        f"- Dataset: {config.get('data', {}).get('dataset', 'N/A')}",
        f"- Chemin: {config.get('data', {}).get('path', 'N/A')}",
        f"- Cache: {config.get('data', {}).get('cache', 'N/A')}",
        "",
        "## Lightning.ai",
        f"- Accélérateur: {config.get('lightning', {}).get('accelerator', 'N/A')}",
        f"- Devices: {config.get('lightning', {}).get('devices', 'N/A')}",
        f"- Précision: {config.get('lightning', {}).get('precision', 'N/A')}",
        ""
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info(f"📄 Résumé de configuration exporté: {output_path}")
