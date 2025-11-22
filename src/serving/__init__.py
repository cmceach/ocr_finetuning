"""Serving modules for OCR model inference"""

from .ocr_server import create_app
from .model_loader import load_model_for_inference

__all__ = ["create_app", "load_model_for_inference"]

