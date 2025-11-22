"""Model loading utilities for inference"""

from pathlib import Path
from typing import Optional
from doctr.models import ocr_predictor
from ..utils.config_loader import ConfigLoader


def load_model_for_inference(
    model_path: Optional[str] = None,
    det_model_path: Optional[str] = None,
    reco_model_path: Optional[str] = None,
    config: Optional[ConfigLoader] = None,
):
    """
    Load trained model for inference.

    Args:
        model_path: Path to full pipeline model (if available)
        det_model_path: Path to detection model weights
        reco_model_path: Path to recognition model weights
        config: Configuration loader instance

    Returns:
        OCR predictor instance
    """
    if config is None:
        config = ConfigLoader()

    det_config = config.get("detection", {})
    reco_config = config.get("recognition", {})

    det_arch = det_config.get("architecture", "db_resnet50")
    reco_arch = reco_config.get("architecture", "crnn_vgg16_bn")

    # Load model
    predictor = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=False)

    # Load custom weights if provided
    if det_model_path and Path(det_model_path).exists():
        if det_model_path.startswith("file://"):
            det_model_path = det_model_path[7:]
        predictor.det_predictor.model.load_weights(det_model_path)

    if reco_model_path and Path(reco_model_path).exists():
        if reco_model_path.startswith("file://"):
            reco_model_path = reco_model_path[7:]
        predictor.reco_predictor.model.load_weights(reco_model_path)

    return predictor

