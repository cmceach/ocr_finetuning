"""Evaluation modules for OCR models"""

from .compare_models import compare_with_azure_di
from .evaluate_ocr import (
    evaluate_ocr_model,
    evaluate_from_file,
    evaluate_nemotron_model,
    evaluate_nemotron_from_file,
    compare_nemotron_with_baseline,
    parse_nemotron_output_for_eval,
    calculate_text_similarity,
)

__all__ = [
    "compare_with_azure_di",
    "evaluate_ocr_model",
    "evaluate_from_file",
    "evaluate_nemotron_model",
    "evaluate_nemotron_from_file",
    "compare_nemotron_with_baseline",
    "parse_nemotron_output_for_eval",
    "calculate_text_similarity",
]

