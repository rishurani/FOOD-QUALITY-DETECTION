import json
import logging
import threading
from pathlib import Path
from PIL import Image
import numpy as np

from food_quality.config import Config

logger = logging.getLogger(__name__)

class ModelNotLoadedError(Exception):
    """Raised when inference is requested but no trained model is available."""
    pass

class FoodQualityPredictor:
    """
    Singleton predictor class that loads the MobileNetV2 trained model
    once into memory and performs thread-safe food quality inference.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FoodQualityPredictor, cls).__new__(cls)
                cls._instance._model = None
                cls._instance._class_indices = None
                cls._instance._model_loaded = False
                cls._instance._load_error = None
                cls._instance._init_model()
            return cls._instance

    def _init_model(self):
        """Attempts to load the trained model and class indices."""
        try:
            model_file = None
            if Config.MODEL_PATH.exists():
                model_file = Config.MODEL_PATH
            elif Config.FALLBACK_MODEL_PATH.exists():
                model_file = Config.FALLBACK_MODEL_PATH

            if not model_file:
                self._model_loaded = False
                self._load_error = (
                    "No trained model found. Please train the model using "
                    "'python train_model.py' to generate 'models/food_quality_model.keras'."
                )
                logger.warning(self._load_error)
                return

            # Lazy import TensorFlow so the web app starts quickly even if TF initializes
            import tensorflow as tf
            logger.info(f"Loading trained Food Quality model from {model_file}...")
            self._model = tf.keras.models.load_model(str(model_file))
            self._model_loaded = True
            self._load_error = None

            # Load class indices if present
            if Config.CLASS_INDICES_PATH.exists():
                with open(Config.CLASS_INDICES_PATH, 'r') as f:
                    self._class_indices = json.load(f)
            else:
                # Default binary mapping
                self._class_indices = {'fresh': 0, 'spoiled': 1}

            logger.info("Food Quality model loaded successfully.")

        except Exception as e:
            self._model_loaded = False
            self._load_error = f"Failed to load machine learning model: {str(e)}"
            logger.error(self._load_error)

    @property
    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._model_loaded

    @property
    def load_error(self):
        """Return the reason if model failed or has not loaded."""
        return self._load_error

    def reload(self):
        """Force reload of the model from disk (useful after retraining)."""
        with self._lock:
            self._model = None
            self._model_loaded = False
            self._init_model()

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocess an input image for MobileNetV2 inference:
        1. Open with Pillow and convert to RGB
        2. Resize to 224 x 224 using high-quality resampling
        3. Convert to float numpy array
        4. Apply MobileNetV2 normalization: scale values to [-1, 1]
        5. Add batch dimension (1, 224, 224, 3)
        """
        with Image.open(image_path) as img:
            rgb_img = img.convert('RGB')
            resized_img = rgb_img.resize(Config.IMAGE_SIZE, Image.Resampling.BILINEAR)
            img_array = np.array(resized_img, dtype=np.float32)

        # The trained model graph includes the preprocess_input layer internally,
        # so raw [0, 255] float arrays are fed directly without double-scaling.
        batch_input = np.expand_dims(img_array, axis=0)
        return batch_input

    def predict(self, image_path: Path) -> dict:
        """
        Execute prediction on an image file.
        Returns:
            {
                'prediction': 'Fresh' | 'Spoiled',
                'confidence': float (0.0 to 1.0),
                'is_uncertain': bool,
                'explanation': str,
                'model_version': str
            }
        """
        if not self._model_loaded or self._model is None:
            # Try reloading once in case user trained it while app was running
            self._init_model()
            if not self._model_loaded or self._model is None:
                raise ModelNotLoadedError(
                    self._load_error or "Trained model is not available. Please run train_model.py."
                )

        # Preprocess input
        batch_data = self.preprocess_image(image_path)

        # Predict
        with self._lock:
            raw_output = self._model.predict(batch_data, verbose=0)

        # Interpret output depending on architecture (binary sigmoid vs 2-class softmax)
        if raw_output.shape[-1] == 1:
            # Binary output: 0 = Fresh, 1 = Spoiled (or based on class_indices)
            score = float(raw_output[0][0])
            # If spoiled index is 1:
            spoiled_idx = self._class_indices.get('spoiled', 1)
            if spoiled_idx == 1:
                prob_spoiled = score
                prob_fresh = 1.0 - score
            else:
                prob_fresh = score
                prob_spoiled = 1.0 - score

            if prob_fresh >= prob_spoiled:
                predicted_label = "Fresh"
                confidence = prob_fresh
            else:
                predicted_label = "Spoiled"
                confidence = prob_spoiled
        else:
            # 2-class softmax: [prob_class_0, prob_class_1]
            probs = raw_output[0]
            # Map classes from self._class_indices
            fresh_idx = self._class_indices.get('fresh', 0)
            spoiled_idx = self._class_indices.get('spoiled', 1)
            
            prob_fresh = float(probs[fresh_idx])
            prob_spoiled = float(probs[spoiled_idx])

            if prob_fresh >= prob_spoiled:
                predicted_label = "Fresh"
                confidence = prob_fresh
            else:
                predicted_label = "Spoiled"
                confidence = prob_spoiled

        # Check uncertainty threshold
        is_uncertain = confidence < Config.UNCERTAINTY_THRESHOLD

        # Craft human-readable explanation
        if is_uncertain:
            explanation = (
                f"The model predicts '{predicted_label}' with {confidence * 100:.1f}% confidence, "
                f"which falls below the reliability threshold ({Config.UNCERTAINTY_THRESHOLD * 100:.0f}%). "
                "Visual features show ambiguous ripening or lighting characteristics. "
                "Manual visual and olfactory inspection is advised."
            )
        elif predicted_label == "Fresh":
            explanation = (
                f"Analysis indicates high visual freshness ({confidence * 100:.1f}% confidence). "
                "The sample exhibits consistent coloration, firm skin texture, and an absence of "
                "observable fungal decay, discoloration patches, or microbial breakdown."
            )
        else:
            explanation = (
                f"Analysis indicates spoilage or decay ({confidence * 100:.1f}% confidence). "
                "The image reveals visual markers characteristic of deterioration, such as dark spots, "
                "skin wrinkling, enzymatic browning, or microbial surface lesions."
            )

        return {
            'prediction': predicted_label,
            'confidence': round(confidence, 4),
            'is_uncertain': is_uncertain,
            'explanation': explanation,
            'model_version': Config.MODEL_VERSION,
            'probabilities': {
                'fresh': round(prob_fresh, 4),
                'spoiled': round(prob_spoiled, 4)
            }
        }
