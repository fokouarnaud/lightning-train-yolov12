#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'entraînement YOLOv12-Face optimisé pour Lightning.ai
Utilise ultralytics YOLO avec des optimisations cloud
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

from ultralytics import YOLO
import torch
import yaml

from .lightning_utils import LightningLogger
from .data_manager import DataManager
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class YOLOv12FaceTrainer:
    """
    Trainer principal pour YOLOv12-Face sur Lightning.ai
    Simplifie l'interface d'ultralytics pour une utilisation optimisée
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_manager: DataManager,
        model_manager: ModelManager,
        logger: LightningLogger
    ):
        """
        Initialise le trainer
        
        Args:
            config: Configuration complète du projet
            data_manager: Gestionnaire des données
            model_manager: Gestionnaire des modèles
            logger: Logger Lightning.ai
        """
        self.config = config
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.lightning_logger = logger
        
        # Chemins de sortie
        self.output_dir = Path(config['output']['base_dir'])
        self.models_dir = Path(config['output']['models_dir'])
        self.logs_dir = Path(config['output']['logs_dir'])
        
        # Créer les dossiers de sortie
        self._create_output_dirs()
        
        # Initialiser le modèle YOLO
        self.model = None
        self._init_model()
        
        logger.info(f"✅ YOLOv12FaceTrainer initialisé")
        logger.info(f"📁 Sortie: {self.output_dir}")
        logger.info(f"🎯 Modèle: YOLOv12{config['model']['size']}")
    
    def _create_output_dirs(self) -> None:
        """Crée les dossiers de sortie nécessaires"""
        for dir_path in [self.output_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _init_model(self) -> None:
        """Initialise le modèle YOLO"""
        model_config = self.config['model']
        
        # Charger le modèle (pré-entraîné ou architecture vide)
        if model_config.get('pretrained'):
            model_path = model_config['pretrained']
            logger.info(f"🔄 Chargement du modèle pré-entraîné: {model_path}")
            self.model = YOLO(model_path)
        else:
            # Créer un modèle depuis l'architecture YAML
            yaml_path = model_config.get('yaml_path', f"yolov12{model_config['size']}.yaml")
            logger.info(f"🏗️ Création du modèle depuis: {yaml_path}")
            self.model = YOLO(yaml_path)
        
        # Adapter le nombre de classes si nécessaire
        if hasattr(self.model.model, 'yaml'):
            self.model.model.yaml['nc'] = model_config['num_classes']
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Lance l'entraînement du modèle
        
        Args:
            resume_from: Chemin vers un checkpoint pour reprendre l'entraînement
            
        Returns:
            Dictionnaire avec les résultats d'entraînement
        """
        logger.info("🚀 Démarrage de l'entraînement YOLOv12-Face")
        
        # Préparer les données
        data_yaml_path = self.data_manager.prepare_dataset()
        
        # Configuration d'entraînement
        train_config = self.config['training']
        lightning_config = self.config['lightning']
        
        # Paramètres d'entraînement optimisés pour Lightning.ai
        train_args = {
            # Données et modèle
            'data': str(data_yaml_path),
            'epochs': train_config['epochs'],
            'batch': train_config['batch_size'],
            'imgsz': train_config['img_size'],
            
            # Optimiseur et learning rate
            'optimizer': train_config['optimizer'],
            'lr0': train_config['lr0'],
            'lrf': train_config['lrf'],
            'momentum': train_config['momentum'],
            'weight_decay': train_config['weight_decay'],
            
            # Régularisation
            'box': train_config['box'],
            'cls': train_config['cls'],
            'obj': train_config['obj'],
            
            # Seuils
            'conf': train_config['conf_thres'],
            'iou': train_config['iou_thres'],
            
            # Augmentation
            'hsv_h': self.config['data']['augmentation']['hsv_h'],
            'hsv_s': self.config['data']['augmentation']['hsv_s'],
            'hsv_v': self.config['data']['augmentation']['hsv_v'],
            'degrees': self.config['data']['augmentation']['degrees'],
            'translate': self.config['data']['augmentation']['translate'],
            'scale': self.config['data']['augmentation']['scale'],
            'mosaic': self.config['data']['augmentation']['mosaic'],
            'mixup': self.config['data']['augmentation']['mixup'],
            
            # Sortie
            'project': str(self.output_dir),
            'name': self._generate_run_name(),
            'exist_ok': True,
            'save': True,
            'save_period': lightning_config.get('save_every_n_epochs', 10),
            
            # Optimisations Lightning.ai
            'cache': self.config['data'].get('cache', True),
            'device': self._get_device(),
            'workers': self.config['environment']['num_workers'],
            'amp': lightning_config['precision'] in ['16', '16-mixed'],
            
            # Monitoring
            'verbose': True,
            'plots': True,
        }
        
        # Reprendre l'entraînement si demandé
        if resume_from:
            train_args['resume'] = resume_from
            logger.info(f"🔄 Reprise de l'entraînement depuis: {resume_from}")
        
        # Gestion des ressources pour Lightning.ai
        self._optimize_for_lightning()
        
        # Lancer l'entraînement
        start_time = time.time()
        try:
            logger.info(f"🏋️ Entraînement en cours...")
            logger.info(f"📊 Paramètres: {train_config['epochs']} epochs, batch={train_config['batch_size']}, img={train_config['img_size']}")
            
            results = self.model.train(**train_args)
            
            training_time = time.time() - start_time
            logger.info(f"✅ Entraînement terminé en {training_time:.2f}s")
            
            # Sauvegarder les résultats
            self._save_training_results(results, training_time)
            
            return {
                'success': True,
                'results': results,
                'training_time': training_time,
                'best_model_path': str(self.model.trainer.best),
                'last_model_path': str(self.model.trainer.last)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur pendant l'entraînement: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
    
    def evaluate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Évalue le modèle sur le dataset de validation
        
        Args:
            model_path: Chemin vers le modèle à évaluer (utilise le meilleur si None)
            
        Returns:
            Dictionnaire avec les résultats d'évaluation
        """
        logger.info("📊 Évaluation du modèle")
        
        # Utiliser le meilleur modèle par défaut
        if model_path is None and hasattr(self.model, 'trainer'):
            model_path = self.model.trainer.best
        
        # Charger le modèle à évaluer
        if model_path and os.path.exists(model_path):
            eval_model = YOLO(model_path)
            logger.info(f"📈 Évaluation du modèle: {model_path}")
        else:
            eval_model = self.model
            logger.info("📈 Évaluation du modèle actuel")
        
        # Configuration d'évaluation
        eval_config = self.config['evaluation']
        
        try:
            # Lancer l'évaluation
            results = eval_model.val(
                data=str(self.data_manager.data_yaml_path),
                imgsz=self.config['training']['img_size'],
                conf=eval_config['conf_thres'],
                iou=eval_config['iou_thres'],
                save_txt=eval_config.get('save_txt', True),
                save_conf=eval_config.get('save_conf', True),
                save_json=eval_config.get('save_json', True),
                plots=eval_config.get('save_plots', True),
                project=str(self.output_dir),
                name='evaluation',
                exist_ok=True
            )
            
            logger.info("✅ Évaluation terminée")
            return {
                'success': True,
                'results': results,
                'metrics': results.results_dict if hasattr(results, 'results_dict') else {}
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur pendant l'évaluation: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def export(self, model_path: Optional[str] = None, format: str = 'onnx') -> Dict[str, Any]:
        """
        Exporte le modèle vers différents formats
        
        Args:
            model_path: Chemin vers le modèle à exporter
            format: Format d'export (onnx, torchscript, coreml, etc.)
            
        Returns:
            Dictionnaire avec les informations d'export
        """
        logger.info(f"📦 Export du modèle en format {format}")
        
        # Utiliser le meilleur modèle par défaut
        if model_path is None and hasattr(self.model, 'trainer'):
            model_path = self.model.trainer.best
        
        # Charger le modèle à exporter
        if model_path and os.path.exists(model_path):
            export_model = YOLO(model_path)
            logger.info(f"📤 Export du modèle: {model_path}")
        else:
            export_model = self.model
            logger.info("📤 Export du modèle actuel")
        
        # Configuration d'export
        export_config = self.config['export']
        
        try:
            # Paramètres d'export selon le format
            export_args = {
                'format': format,
                'imgsz': self.config['training']['img_size'],
                'device': self._get_device(),
                'half': self.config['lightning']['precision'] in ['16', '16-mixed'],
                'simplify': True,  # Simplifier le modèle ONNX
                'workspace': 4,    # Workspace pour TensorRT
                'verbose': True
            }
            
            # Options spécifiques ONNX
            if format == 'onnx':
                onnx_config = export_config.get('onnx', {})
                export_args.update({
                    'opset': onnx_config.get('opset', 12),
                    'simplify': onnx_config.get('simplify', True),
                    'dynamic': onnx_config.get('dynamic', False)
                })
            
            # Lancer l'export
            exported_model = export_model.export(**export_args)
            
            # Déplacer vers le dossier d'export
            export_dir = Path(self.config['output']['exports_dir'])
            export_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"✅ Export terminé: {exported_model}")
            
            return {
                'success': True,
                'exported_model': str(exported_model),
                'format': format,
                'size_mb': os.path.getsize(exported_model) / (1024 * 1024) if os.path.exists(exported_model) else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur pendant l'export: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': format
            }
    
    def _get_device(self) -> str:
        """Détermine le device optimal pour Lightning.ai"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _generate_run_name(self) -> str:
        """Génère un nom unique pour ce run"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_size = self.config['model']['size']
        return f"yolov12{model_size}_{timestamp}"
    
    def _optimize_for_lightning(self) -> None:
        """Optimisations spécifiques à Lightning.ai"""
        # Optimisations mémoire
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Configuration des workers
        os.environ['PYTHONPATH'] = self.config['environment']['pythonpath']
        
        # Graine pour la reproductibilité
        if 'seed' in self.config['environment']:
            torch.manual_seed(self.config['environment']['seed'])
        
        logger.info("⚡ Optimisations Lightning.ai appliquées")
    
    def _save_training_results(self, results: Any, training_time: float) -> None:
        """Sauvegarde les résultats d'entraînement"""
        results_dict = {
            'training_time': training_time,
            'config': self.config,
            'model_info': {
                'size': self.config['model']['size'],
                'num_classes': self.config['model']['num_classes'],
                'img_size': self.config['training']['img_size']
            }
        }
        
        # Sauvegarder en YAML
        results_file = self.output_dir / 'training_results.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(results_dict, f, default_flow_style=False)
        
        logger.info(f"💾 Résultats sauvegardés: {results_file}")
