#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires spécifiques à Lightning.ai
Optimisations et intégrations pour l'environnement cloud
"""

import os
import logging
import time
import json
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

logger = logging.getLogger(__name__)

class LightningLogger:
    """
    Logger personnalisé pour Lightning.ai
    Gère le monitoring, les métriques et la sauvegarde
    """
    
    def __init__(self, project_name: str, save_period: int = 10):
        """
        Initialise le logger Lightning.ai
        
        Args:
            project_name: Nom du projet
            save_period: Période de sauvegarde (epochs)
        """
        self.project_name = project_name
        self.save_period = save_period
        self.start_time = time.time()
        
        # Métriques de suivi
        self.metrics_history = []
        self.system_metrics = []
        
        # Chemins de logging
        self.log_dir = Path("outputs/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / f"{project_name}_metrics.json"
        self.system_file = self.log_dir / f"{project_name}_system.json"
        
        logger.info(f"⚡ LightningLogger initialisé pour {project_name}")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log les métriques d'entraînement"""
        timestamp = time.time()
        
        metric_entry = {
            'epoch': epoch,
            'timestamp': timestamp,
            'elapsed_time': timestamp - self.start_time,
            'metrics': metrics
        }
        
        self.metrics_history.append(metric_entry)
        
        # Sauvegarder périodiquement
        if epoch % self.save_period == 0:
            self._save_metrics()
        
        logger.info(f"📊 Epoch {epoch} - Métriques: {metrics}")
    
    def log_system_info(self) -> Dict[str, Any]:
        """Log les informations système Lightning.ai"""
        system_info = {
            'timestamp': time.time(),
            'gpu_info': self._get_gpu_info(),
            'memory_info': self._get_memory_info(),
            'cpu_info': self._get_cpu_info(),
            'disk_info': self._get_disk_info()
        }
        
        self.system_metrics.append(system_info)
        return system_info
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Récupère les informations GPU"""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            try:
                gpu_info.update({
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(),
                    'memory_allocated': torch.cuda.memory_allocated(),
                    'memory_reserved': torch.cuda.memory_reserved(),
                    'max_memory_allocated': torch.cuda.max_memory_allocated()
                })
            except Exception as e:
                logger.warning(f"⚠️ Erreur récupération info GPU: {e}")
        
        return gpu_info
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Récupère les informations mémoire"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Récupère les informations CPU"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
    
    def _get_disk_info(self) -> Dict[str, float]:
        """Récupère les informations disque"""
        disk = psutil.disk_usage('/')
        return {
            'total_gb': disk.total / (1024**3),
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent': (disk.used / disk.total) * 100
        }
    
    def _save_metrics(self) -> None:
        """Sauvegarde les métriques sur disque"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            with open(self.system_file, 'w') as f:
                json.dump(self.system_metrics, f, indent=2)
                
            logger.debug(f"💾 Métriques sauvegardées")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde métriques: {e}")
    
    def get_best_metrics(self) -> Optional[Dict[str, Any]]:
        """Retourne les meilleures métriques"""
        if not self.metrics_history:
            return None
        
        # Trouver la meilleure mAP (ou autre métrique principale)
        best_map = 0
        best_metrics = None
        
        for entry in self.metrics_history:
            metrics = entry.get('metrics', {})
            current_map = metrics.get('mAP50', metrics.get('val/mAP50', 0))
            
            if current_map > best_map:
                best_map = current_map
                best_metrics = entry
        
        return best_metrics
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Génère un rapport de résumé"""
        total_time = time.time() - self.start_time
        best_metrics = self.get_best_metrics()
        
        summary = {
            'project_name': self.project_name,
            'total_training_time': total_time,
            'total_epochs': len(self.metrics_history),
            'best_metrics': best_metrics,
            'system_info': self.system_metrics[-1] if self.system_metrics else None,
            'average_time_per_epoch': total_time / len(self.metrics_history) if self.metrics_history else 0
        }
        
        # Sauvegarder le rapport
        report_file = self.log_dir / f"{self.project_name}_summary.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Rapport de résumé généré: {report_file}")
        return summary

class LightningOptimizer:
    """
    Optimiseur spécifique à Lightning.ai
    Gère les optimisations de performance et de ressources
    """
    
    @staticmethod
    def optimize_environment() -> None:
        """Optimise l'environnement Lightning.ai"""
        logger.info("⚡ Optimisation de l'environnement Lightning.ai")
        
        # Variables d'environnement pour optimiser PyTorch
        optimizations = {
            'CUDA_LAUNCH_BLOCKING': '0',  # Optimiser les lancements CUDA
            'TORCH_CUDNN_V8_API_ENABLED': '1',  # Activer cuDNN v8
            'PYTHONHASHSEED': '42',  # Reproductibilité
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',  # Optimiser cuBLAS
        }
        
        for key, value in optimizations.items():
            os.environ[key] = value
            logger.debug(f"  {key} = {value}")
        
        # Optimisations PyTorch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimiser pour des tailles d'entrée fixes
            torch.backends.cudnn.deterministic = False  # Permet plus d'optimisations
            torch.cuda.empty_cache()  # Nettoyer le cache GPU
        
        logger.info("✅ Optimisations appliquées")
    
    @staticmethod
    def get_optimal_batch_size(model_size: str, img_size: int, available_memory_gb: float) -> int:
        """
        Calcule la taille de batch optimale selon les ressources
        
        Args:
            model_size: Taille du modèle (n, s, m, l, x)
            img_size: Taille des images
            available_memory_gb: Mémoire disponible en GB
            
        Returns:
            Taille de batch recommandée
        """
        # Estimation de la mémoire requise par image (en MB)
        memory_per_image = {
            'n': {'640': 50, '512': 32, '416': 20},
            's': {'640': 80, '512': 50, '416': 32},
            'm': {'640': 120, '512': 75, '416': 48},
            'l': {'640': 160, '512': 100, '416': 64},
            'x': {'640': 200, '512': 125, '416': 80}
        }
        
        img_key = str(img_size) if str(img_size) in ['640', '512', '416'] else '640'
        mem_per_img = memory_per_image.get(model_size, memory_per_image['s'])[img_key]
        
        # Calculer la batch size optimale (garder 20% de marge)
        available_memory_mb = available_memory_gb * 1024 * 0.8
        optimal_batch_size = int(available_memory_mb / mem_per_img)
        
        # Limiter entre 1 et 64
        optimal_batch_size = max(1, min(64, optimal_batch_size))
        
        logger.info(f"📊 Batch size optimal calculé: {optimal_batch_size}")
        logger.info(f"   Mémoire disponible: {available_memory_gb:.1f}GB")
        logger.info(f"   Mémoire par image: {mem_per_img}MB")
        
        return optimal_batch_size
    
    @staticmethod
    def monitor_resources() -> Dict[str, float]:
        """Monitore l'utilisation des ressources"""
        resources = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
        
        if torch.cuda.is_available():
            resources.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024**3),
                'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024**3)
            })
        
        return resources

class LightningCheckpoint:
    """
    Gestionnaire de checkpoints optimisé pour Lightning.ai
    """
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialise le gestionnaire de checkpoints
        
        Args:
            checkpoint_dir: Répertoire des checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Gestionnaire de checkpoints: {checkpoint_dir}")
    
    def save_checkpoint(self, model_path: str, epoch: int, metrics: Dict[str, float]) -> str:
        """
        Sauvegarde un checkpoint avec métadonnées
        
        Args:
            model_path: Chemin vers le modèle
            epoch: Epoch actuel
            metrics: Métriques de l'epoch
            
        Returns:
            Chemin vers le checkpoint sauvegardé
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch:03d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Copier le modèle vers le checkpoint
        import shutil
        shutil.copy2(model_path, checkpoint_path)
        
        # Créer un fichier de métadonnées
        metadata = {
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics,
            'model_path': str(model_path),
            'checkpoint_path': str(checkpoint_path)
        }
        
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Checkpoint sauvegardé: {checkpoint_name}")
        return str(checkpoint_path)
    
    def get_best_checkpoint(self, metric_name: str = 'mAP50') -> Optional[str]:
        """
        Trouve le meilleur checkpoint selon une métrique
        
        Args:
            metric_name: Nom de la métrique à optimiser
            
        Returns:
            Chemin vers le meilleur checkpoint
        """
        best_value = 0
        best_checkpoint = None
        
        for metadata_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                metrics = metadata.get('metrics', {})
                value = metrics.get(metric_name, 0)
                
                if value > best_value:
                    best_value = value
                    best_checkpoint = metadata.get('checkpoint_path')
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur lecture métadonnées {metadata_file}: {e}")
        
        if best_checkpoint:
            logger.info(f"🏆 Meilleur checkpoint: {best_checkpoint} ({metric_name}={best_value:.4f})")
        
        return best_checkpoint
    
    def cleanup_old_checkpoints(self, keep_best: int = 3) -> None:
        """
        Nettoie les anciens checkpoints en gardant les meilleurs
        
        Args:
            keep_best: Nombre de meilleurs checkpoints à conserver
        """
        logger.info(f"🧹 Nettoyage des checkpoints (garder {keep_best} meilleurs)")
        
        # TODO: Implémenter le nettoyage intelligent
        # Garder les N meilleurs + le plus récent + certains jalons
        
        logger.info("✅ Nettoyage terminé")

def setup_lightning_environment() -> None:
    """
    Configuration complète de l'environnement Lightning.ai
    """
    logger.info("🚀 Configuration de l'environnement Lightning.ai")
    
    # Optimisations générales
    LightningOptimizer.optimize_environment()
    
    # Vérification des ressources
    resources = LightningOptimizer.monitor_resources()
    logger.info(f"💻 Ressources système: CPU={resources['cpu_percent']:.1f}%, RAM={resources['memory_percent']:.1f}%")
    
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU disponible: {torch.cuda.get_device_name()}")
        logger.info(f"   Mémoire GPU: {torch.cuda.memory_allocated()/(1024**3):.2f}GB allouée")
    else:
        logger.warning("⚠️ GPU non disponible, utilisation du CPU")
    
    logger.info("✅ Environnement Lightning.ai configuré")
