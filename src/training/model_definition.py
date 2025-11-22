"""Custom DocTR model definitions and utilities"""

from typing import Optional, Dict, Any
import torch
from doctr.models import ocr_predictor
from doctr.models.detection import DBNet, LinkNet
from doctr.models.recognition import CRNN, SAR, Master


def get_detection_model(architecture: str = "db_resnet50", pretrained: bool = True):
    """
    Get detection model by architecture name.

    Args:
        architecture: Model architecture name (db_resnet50, craft_resnet50, linknet_resnet18)
        pretrained: Whether to load pretrained weights

    Returns:
        Detection model instance
    """
    # DocTR uses ocr_predictor which includes both detection and recognition
    # For detection-only, we'll use the predictor but focus on detection
    predictor = ocr_predictor(det_arch=architecture, reco_arch=None, pretrained=pretrained)
    return predictor.det_predictor.model


def get_recognition_model(architecture: str = "crnn_vgg16_bn", pretrained: bool = True):
    """
    Get recognition model by architecture name.

    Args:
        architecture: Model architecture name (crnn_vgg16_bn, sar_resnet31, master)
        pretrained: Whether to load pretrained weights

    Returns:
        Recognition model instance
    """
    predictor = ocr_predictor(det_arch=None, reco_arch=architecture, pretrained=pretrained)
    return predictor.reco_predictor.model


def get_full_pipeline_model(
    det_arch: str = "db_resnet50",
    reco_arch: str = "crnn_vgg16_bn",
    pretrained: bool = True,
):
    """
    Get full OCR pipeline model (detection + recognition).

    Args:
        det_arch: Detection architecture name
        reco_arch: Recognition architecture name
        pretrained: Whether to load pretrained weights

    Returns:
        OCR predictor instance
    """
    return ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=pretrained)


def load_custom_weights(model, weights_path: str):
    """
    Load custom weights into a model.

    Args:
        model: Model instance
        weights_path: Path to weights file
    """
    if weights_path.startswith("file://"):
        weights_path = weights_path[7:]
    model.load_weights(weights_path)

