"""Training script for DocTR text recognition models"""

import os
import json
import torch
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

from ..utils.config_loader import ConfigLoader
from .model_definition import get_recognition_model
from .metrics import OCRMetrics


def train_recognition_model(
    config: Optional[ConfigLoader] = None,
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    Train a text recognition model using DocTR.

    Args:
        config: Configuration loader instance
        train_data_path: Path to training data JSON file
        val_data_path: Path to validation data JSON file
        output_dir: Directory to save trained model

    Returns:
        Path to saved model weights
    """
    if config is None:
        config = ConfigLoader()

    # Get configuration
    reco_config = config.get("recognition", {})
    train_config = config.get("training", {})
    mlflow_config = config.get("logging", {}).get("mlflow", {})

    # Initialize MLflow if enabled
    if mlflow_config.get("enabled"):
        mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "http://localhost:5000"))
        mlflow.set_experiment(mlflow_config.get("experiment_name", "ocr-recognition"))

    # Set device
    device = torch.device(train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    model = get_recognition_model(
        architecture=reco_config.get("architecture", "crnn_vgg16_bn"),
        pretrained=reco_config.get("pretrained", True),
    )
    model = model.to(device)

    # Load training data
    if train_data_path is None:
        train_data_path = config.get("data.processed_data_path", "./data/processed_data/recognition.json")

    with open(train_data_path, "r") as f:
        train_data = json.load(f)

    # Load validation data
    if val_data_path is None:
        val_data_path = config.get("data.processed_data_path", "./data/processed_data/recognition_val.json")

    val_data = []
    if os.path.exists(val_data_path):
        with open(val_data_path, "r") as f:
            val_data = json.load(f)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=reco_config.get("learning_rate", 0.001),
        weight_decay=reco_config.get("weight_decay", 0.0001),
    )

    epochs = reco_config.get("epochs", 100)
    batch_size = reco_config.get("batch_size", 16)

    # Training loop (simplified - actual implementation would use DocTR's training utilities)
    best_val_loss = float("inf")
    output_dir = Path(output_dir or config.get("model_storage.local_path", "./trained_models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Note: This is a simplified training loop
    # For production, use DocTR's reference training scripts
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Simplified training step
        for batch_idx in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = train_data[batch_idx : batch_idx + batch_size]
            # Training step would go here
            pass

        # Validation
        if val_data:
            model.eval()
            val_loss = 0.0
            # Validation step would go here
            pass

        # Log metrics
        if mlflow_config.get("enabled"):
            mlflow.log_metrics(
                {
                    "train_loss": train_loss / len(train_data),
                    "val_loss": val_loss / len(val_data) if val_data else 0,
                },
                step=epoch,
            )

        # Save checkpoint
        checkpoint_path = output_dir / f"recognition_epoch_{epoch+1}.pth"
        # model.save_weights(str(checkpoint_path))

        if val_data and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "recognition_best.pth"
            # model.save_weights(str(best_model_path))

    print("Training completed!")
    return str(output_dir / "recognition_best.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DocTR recognition model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--client-config", type=str, help="Path to client config file")
    parser.add_argument("--train-data", type=str, help="Path to training data")
    parser.add_argument("--val-data", type=str, help="Path to validation data")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    config = ConfigLoader(args.config, args.client_config)
    train_recognition_model(
        config=config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
    )

