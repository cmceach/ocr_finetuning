"""Training modules for DocTR detection and recognition models"""

from .train_detection import train_detection_model
from .train_recognition import train_recognition_model
from .train_full_pipeline import train_full_pipeline
from .metrics import OCRMetrics

__all__ = [
    "train_detection_model",
    "train_recognition_model",
    "train_full_pipeline",
    "OCRMetrics",
]

