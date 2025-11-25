"""End-to-end training orchestrator for full OCR pipeline"""

from typing import Optional
from ..utils.config_loader import ConfigLoader
from .train_detection import train_detection_model
from .train_recognition import train_recognition_model
from .train_nemotron import train_nemotron_model


def train_full_pipeline(
    config: Optional[ConfigLoader] = None,
    train_detection: bool = True,
    train_recognition: bool = True,
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    image_base_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    Train full OCR pipeline (detection + recognition) or Nemotron Parse model.

    Args:
        config: Configuration loader instance
        train_detection: Whether to train detection model (for DocTR mode)
        train_recognition: Whether to train recognition model (for DocTR mode)
        train_data_path: Path to training data (for Nemotron mode)
        val_data_path: Path to validation data (for Nemotron mode)
        image_base_dir: Base directory for images (for Nemotron mode)
        output_dir: Output directory for models

    Returns:
        Dictionary with paths to trained models
    """
    if config is None:
        config = ConfigLoader()

    pipeline_mode = config.get("pipeline_mode", "full")

    results = {}

    # Nemotron Parse training (VLM-based document understanding)
    if pipeline_mode == "nemotron":
        print("Training Nemotron Parse model...")
        nemotron_model_path = train_nemotron_model(
            config=config,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            image_base_dir=image_base_dir,
            output_dir=output_dir,
        )
        results["nemotron_model"] = nemotron_model_path
        print("Nemotron Parse training completed!")
        return results

    # DocTR-based training (detection + recognition)
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
    parser.add_argument("--nemotron", action="store_true", help="Train Nemotron Parse model")

    # Nemotron-specific arguments
    parser.add_argument("--train-data", type=str, help="Path to training data (for Nemotron)")
    parser.add_argument("--val-data", type=str, help="Path to validation data (for Nemotron)")
    parser.add_argument("--image-base-dir", type=str, help="Base directory for images")
    parser.add_argument("--output-dir", type=str, help="Output directory for models")

    args = parser.parse_args()

    config = ConfigLoader(args.config, args.client_config)

    # Override pipeline mode if --nemotron flag is used
    if args.nemotron:
        # Temporarily override config
        original_mode = config.get("pipeline_mode", "full")
        config._config["pipeline_mode"] = "nemotron"

    train_full_pipeline(
        config=config,
        train_detection=not args.recognition_only,
        train_recognition=not args.detection_only,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        image_base_dir=args.image_base_dir,
        output_dir=args.output_dir,
    )

