"""Evaluation modules for OCR models"""

from .compare_models import compare_with_azure_di
from .evaluate_ocr import evaluate_ocr_model

__all__ = ["compare_with_azure_di", "evaluate_ocr_model"]

