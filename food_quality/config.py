import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    """Base application configuration."""
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-freshvision-secret-key-cse-cmv')
    
    # Database
    INSTANCE_DIR = BASE_DIR / 'instance'
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URI', f"sqlite:///{INSTANCE_DIR / 'food_quality.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Uploads
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024))  # 10MB
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
    ALLOWED_MIME_TYPES = {
        'image/jpeg',
        'image/png',
        'image/webp'
    }
    
    # Machine Learning
    MODELS_DIR = BASE_DIR / 'models'
    MODEL_PATH = MODELS_DIR / 'food_quality_model.keras'
    FALLBACK_MODEL_PATH = MODELS_DIR / 'food_quality_model.h5'
    METRICS_PATH = MODELS_DIR / 'model_metrics.json'
    CLASS_INDICES_PATH = MODELS_DIR / 'class_indices.json'
    UNCERTAINTY_THRESHOLD = float(os.environ.get('UNCERTAINTY_THRESHOLD', 0.65))
    MODEL_VERSION = 'v1.0.0-MobileNetV2'
    
    # Image input shape
    IMAGE_SIZE = (224, 224)
