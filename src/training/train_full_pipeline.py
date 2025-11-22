"""End-to-end training orchestrator for full OCR pipeline"""

from typing import Optional
from ..utils.config_loader import ConfigLoader
from .train_detection import train_detection_model
from .train_recognition import train_recognition_model


def train_full_pipeline(
    config: Optional[ConfigLoader] = None,
    train_detection: bool = True,
    train_recognition: bool = True,
):
    """
    Train full OCR pipeline (detection + recognition).

    Args:
        config: Configuration loader instance
        train_detection: Whether to train detection model
        train_recognition: Whether to train recognition model

    Returns:
        Dictionary with paths to trained models
    """
    if config is None:
        config = ConfigLoader()

    pipeline_mode = config.get("pipeline_mode", "full")

    results = {}

    # Train detection if needed
    if train_detection and pipeline_mode in ["detection", "full"]:
        print("Training detection model...")
        det_model_path = train_detection_model(config=config)
        results["detection_model"] = det_model_path

    # Train recognition if needed
    if train_recognition and pipeline_mode in ["recognition", "full"]:
        print("Training recognition model...")
        reco_model_path = train_recognition_model(config=config)
        results["recognition_model"] = reco_model_path

    print("Full pipeline training completed!")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train full OCR pipeline")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--client-config", type=str, help="Path to client config file")
    parser.add_argument("--detection-only", action="store_true", help="Train only detection")
    parser.add_argument("--recognition-only", action="store_true", help="Train only recognition")

    args = parser.parse_args()

    config = ConfigLoader(args.config, args.client_config)
    train_full_pipeline(
        config=config,
        train_detection=not args.recognition_only,
        train_recognition=not args.detection_only,
    )

