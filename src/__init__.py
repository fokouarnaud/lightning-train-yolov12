#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package YOLOv12-Face Lightning.ai
Modules principaux pour l'entraînement optimisé sur cloud
"""

__version__ = "1.0.0"
__author__ = "Cedric"
__description__ = "YOLOv12-Face optimisé pour Lightning.ai"

from .train import YOLOv12FaceTrainer
from .data_manager import DataManager
from .model_manager import ModelManager
from .lightning_utils import LightningLogger, LightningOptimizer, setup_lightning_environment

__all__ = [
    'YOLOv12FaceTrainer',
    'DataManager', 
    'ModelManager',
    'LightningLogger',
    'LightningOptimizer',
    'setup_lightning_environment'
]
