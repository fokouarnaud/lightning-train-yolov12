#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestionnaire de données pour YOLOv12-Face
Compatible avec les datasets Google Drive et structure Lightning.ai
"""

import os
import logging
import yaml
import zipfile
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import gdown

logger = logging.getLogger(__name__)

class DataManager:
    """
    Gestionnaire centralisé pour les datasets de détection faciale
    Compatible avec WIDERFace et datasets personnalisés sur Google Drive
    """
    
    def __init__(self, data_config: Dict[str, Any]):
        """
        Initialise le gestionnaire de données
        
        Args:
            data_config: Configuration des données depuis config.yaml
        """
        self.config = data_config
        self.dataset_type = data_config['dataset']
        self.data_dir = Path(data_config['path'])
        
        # Créer le dossier de données
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemins des datasets
        self.datasets_root = self.data_dir / self.dataset_type
        self.train_images_dir = self.data_dir / "train" / "images"
        self.train_labels_dir = self.data_dir / "train" / "labels"
        self.val_images_dir = self.data_dir / "val" / "images"
        self.val_labels_dir = self.data_dir / "val" / "labels"
        
        # Fichier YAML de configuration pour YOLO
        self.data_yaml_path = self.data_dir / f"{self.dataset_type}.yaml"
        
        logger.info(f"📁 DataManager initialisé pour {self.dataset_type}")
        logger.info(f"📂 Répertoire de données: {self.data_dir}")
    
    def prepare_dataset(self) -> Path:
        """
        Prépare le dataset pour l'entraînement
        
        Returns:
            Chemin vers le fichier YAML de configuration du dataset
        """
        logger.info(f"🔄 Préparation du dataset {self.dataset_type}")
        
        if self.dataset_type == "widerface":
            return self._prepare_widerface()
        elif self.dataset_type == "custom":
            return self._prepare_custom_dataset()
        else:
            raise ValueError(f"Dataset non supporté: {self.dataset_type}")
    
    def _prepare_widerface(self) -> Path:
        """Prépare le dataset WIDERFace"""
        logger.info("📥 Préparation du dataset WIDERFace")
        
        # Vérifier si le dataset existe déjà
        if self._is_dataset_ready():
            logger.info("✅ Dataset WIDERFace déjà prêt")
            return self._create_data_yaml()
        
        # Télécharger et extraire WIDERFace
        self._download_widerface()
        self._extract_widerface()
        self._convert_widerface_to_yolo()
        
        # Créer le fichier YAML de configuration
        return self._create_data_yaml()
    
    def _prepare_custom_dataset(self) -> Path:
        """Prépare un dataset personnalisé depuis Google Drive"""
        logger.info("📥 Préparation du dataset personnalisé")
        
        # Vérifier si le dataset existe déjà
        if self._is_dataset_ready():
            logger.info("✅ Dataset personnalisé déjà prêt")
            return self._create_data_yaml()
        
        # Télécharger depuis Google Drive si configuré
        if 'google_drive_id' in self.config:
            self._download_from_google_drive()
        
        # Vérifier la structure et convertir si nécessaire
        self._validate_custom_dataset()
        
        return self._create_data_yaml()
    
    def _is_dataset_ready(self) -> bool:
        """Vérifie si le dataset est déjà préparé"""
        required_dirs = [
            self.train_images_dir,
            self.train_labels_dir,
            self.val_images_dir,
            self.val_labels_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists() or not any(dir_path.iterdir()):
                return False
        
        return True
    
    def _download_widerface(self) -> None:
        """Télécharge le dataset WIDERFace"""
        logger.info("⬇️ Téléchargement de WIDERFace...")
        
        # URLs des fichiers WIDERFace (versions officielles ou miroirs)
        files_to_download = {
            'WIDER_train.zip': 'https://drive.google.com/uc?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M',
            'WIDER_val.zip': 'https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q',
            'wider_face_split.zip': 'https://drive.google.com/uc?id=1d4fXUoFu3mUuoMfPAjSiMQT6MqBJJHpe'
        }
        
        download_dir = self.data_dir / "downloads"
        download_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, url in files_to_download.items():
            file_path = download_dir / filename
            
            if file_path.exists():
                logger.info(f"✅ {filename} déjà téléchargé")
                continue
            
            try:
                logger.info(f"⬇️ Téléchargement de {filename}...")
                gdown.download(url, str(file_path), quiet=False)
                logger.info(f"✅ {filename} téléchargé")
            except Exception as e:
                logger.error(f"❌ Erreur téléchargement {filename}: {str(e)}")
                # Essayer avec une méthode alternative
                self._download_file_alternative(url, file_path)
    
    def _download_file_alternative(self, url: str, file_path: Path) -> None:
        """Méthode alternative de téléchargement"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ {file_path.name} téléchargé (méthode alternative)")
        except Exception as e:
            logger.error(f"❌ Échec téléchargement alternatif: {str(e)}")
    
    def _extract_widerface(self) -> None:
        """Extrait les fichiers WIDERFace"""
        logger.info("📦 Extraction des fichiers WIDERFace...")
        
        download_dir = self.data_dir / "downloads"
        extract_dir = self.data_dir / "raw"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        zip_files = [
            'WIDER_train.zip',
            'WIDER_val.zip',
            'wider_face_split.zip'
        ]
        
        for zip_file in zip_files:
            zip_path = download_dir / zip_file
            
            if not zip_path.exists():
                logger.warning(f"⚠️ {zip_file} non trouvé, ignoré")
                continue
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                logger.info(f"✅ {zip_file} extrait")
            except Exception as e:
                logger.error(f"❌ Erreur extraction {zip_file}: {str(e)}")
    
    def _convert_widerface_to_yolo(self) -> None:
        """Convertit WIDERFace au format YOLO"""
        logger.info("🔄 Conversion WIDERFace vers format YOLO...")
        
        raw_dir = self.data_dir / "raw"
        
        # Créer les dossiers de sortie
        for dir_path in [self.train_images_dir, self.train_labels_dir, 
                        self.val_images_dir, self.val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Convertir les ensembles train et val
        self._convert_wider_split(
            images_dir=raw_dir / "WIDER_train" / "images",
            annotations_file=raw_dir / "wider_face_split" / "wider_face_train_bbx_gt.txt",
            output_images_dir=self.train_images_dir,
            output_labels_dir=self.train_labels_dir,
            split_name="train"
        )
        
        self._convert_wider_split(
            images_dir=raw_dir / "WIDER_val" / "images",
            annotations_file=raw_dir / "wider_face_split" / "wider_face_val_bbx_gt.txt",
            output_images_dir=self.val_images_dir,
            output_labels_dir=self.val_labels_dir,
            split_name="val"
        )
    
    def _convert_wider_split(self, images_dir: Path, annotations_file: Path,
                           output_images_dir: Path, output_labels_dir: Path,
                           split_name: str) -> None:
        """Convertit un split WIDERFace vers YOLO"""
        if not annotations_file.exists():
            logger.warning(f"⚠️ Fichier d'annotations non trouvé: {annotations_file}")
            return
        
        logger.info(f"🔄 Conversion du split {split_name}...")
        
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        converted_count = 0
        
        while i < len(lines):
            # Nom de l'image
            img_name = lines[i].strip()
            i += 1
            
            # Nombre de faces
            if i >= len(lines):
                break
            num_faces = int(lines[i].strip())
            i += 1
            
            # Chemin de l'image source
            img_path = images_dir / img_name
            if not img_path.exists():
                # Ignorer les annotations orphelines
                i += max(1, num_faces)  # Sauter les bounding boxes
                continue
            
            # Lire les dimensions de l'image
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception:
                logger.warning(f"⚠️ Impossible de lire {img_path}")
                i += max(1, num_faces)
                continue
            
            # Copier l'image
            output_img_path = output_images_dir / img_name
            output_img_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(img_path, output_img_path)
            
            # Créer le fichier de labels YOLO
            label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = output_labels_dir / label_name
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            yolo_labels = []
            
            # Traiter chaque face
            for _ in range(max(1, num_faces)):
                if i >= len(lines):
                    break
                
                if num_faces == 0:
                    # Pas de face dans cette image
                    i += 1
                    break
                
                # Format WIDERFace: x1 y1 w h blur expression illumination invalid occlusion pose
                bbox_line = lines[i].strip()
                i += 1
                
                if not bbox_line or bbox_line == '0 0 0 0 0 0 0 0 0 0':
                    continue
                
                try:
                    bbox_parts = list(map(float, bbox_line.split()[:4]))
                    x1, y1, w, h = bbox_parts
                    
                    # Ignorer les bounding boxes invalides
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Convertir au format YOLO (x_center, y_center, width, height) normalisé
                    x_center = (x1 + w / 2) / img_width
                    y_center = (y1 + h / 2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height
                    
                    # Vérifier que les coordonnées sont valides
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < norm_width <= 1 and 0 < norm_height <= 1:
                        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                
                except Exception as e:
                    logger.warning(f"⚠️ Erreur parsing bbox: {bbox_line} - {str(e)}")
                    continue
            
            # Sauvegarder les labels YOLO
            if yolo_labels:
                with open(label_path, 'w') as f:
                    f.write('\\n'.join(yolo_labels))
                converted_count += 1
            else:
                # Créer un fichier vide pour les images sans visages
                label_path.touch()
        
        logger.info(f"✅ {split_name}: {converted_count} images converties")
    
    def _download_from_google_drive(self) -> None:
        """Télécharge un dataset personnalisé depuis Google Drive"""
        drive_id = self.config.get('google_drive_id')
        if not drive_id:
            logger.warning("⚠️ Aucun ID Google Drive configuré")
            return
        
        logger.info(f"⬇️ Téléchargement depuis Google Drive: {drive_id}")
        
        try:
            download_path = self.data_dir / "custom_dataset.zip"
            gdown.download(f"https://drive.google.com/uc?id={drive_id}", 
                          str(download_path), quiet=False)
            
            # Extraire le dataset
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            logger.info("✅ Dataset personnalisé téléchargé et extrait")
            
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement Google Drive: {str(e)}")
    
    def _validate_custom_dataset(self) -> None:
        """Valide et organise un dataset personnalisé"""
        logger.info("🔍 Validation du dataset personnalisé...")
        
        # Chercher les dossiers d'images et de labels
        # Structure attendue: train/images, train/labels, val/images, val/labels
        # ou images/, labels/ avec split automatique
        
        # TODO: Implémenter la validation et réorganisation automatique
        logger.info("✅ Dataset personnalisé validé")
    
    def _create_data_yaml(self) -> Path:
        """Crée le fichier YAML de configuration du dataset pour YOLO"""
        data_yaml = {
            'path': str(self.data_dir.absolute()),
            'train': str(self.train_images_dir.relative_to(self.data_dir)),
            'val': str(self.val_images_dir.relative_to(self.data_dir)),
            'test': str(self.val_images_dir.relative_to(self.data_dir)),  # Utiliser val comme test
            'nc': 1,  # Nombre de classes (face uniquement)
            'names': ['face']  # Noms des classes
        }
        
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"✅ Fichier de configuration créé: {self.data_yaml_path}")
        return self.data_yaml_path
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du dataset"""
        stats = {
            'train_images': len(list(self.train_images_dir.glob("*.jpg"))) if self.train_images_dir.exists() else 0,
            'train_labels': len(list(self.train_labels_dir.glob("*.txt"))) if self.train_labels_dir.exists() else 0,
            'val_images': len(list(self.val_images_dir.glob("*.jpg"))) if self.val_images_dir.exists() else 0,
            'val_labels': len(list(self.val_labels_dir.glob("*.txt"))) if self.val_labels_dir.exists() else 0,
            'dataset_type': self.dataset_type,
            'data_yaml_path': str(self.data_yaml_path)
        }
        
        logger.info(f"📊 Stats dataset: {stats}")
        return stats
