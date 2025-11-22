"""Model export and versioning utilities"""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def export_model(
    model_path: str,
    output_dir: str,
    model_name: str,
    version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export and version a trained model.

    Args:
        model_path: Path to model file or directory
        output_dir: Output directory for exported model
        model_name: Name of the model
        version: Version string (defaults to timestamp)
        metadata: Additional metadata to include

    Returns:
        Path to exported model
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create versioned directory
    version_dir = output_dir / model_name / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy model files
    if model_path.is_file():
        shutil.copy2(model_path, version_dir / model_path.name)
    elif model_path.is_dir():
        shutil.copytree(model_path, version_dir / model_path.name, dirs_exist_ok=True)

    # Save metadata
    metadata_dict = {
        "model_name": model_name,
        "version": version,
        "export_date": datetime.now().isoformat(),
        **(metadata or {}),
    }

    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=2)

    # Create symlink to latest version
    latest_link = output_dir / model_name / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(version)

    print(f"Model exported to: {version_dir}")
    return str(version_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export and version a trained model")
    parser.add_argument("--model-path", required=True, help="Path to model file or directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--version", help="Version string")
    parser.add_argument("--metadata", help="Path to metadata JSON file")

    args = parser.parse_args()

    metadata = None
    if args.metadata:
        with open(args.metadata, "r") as f:
            metadata = json.load(f)

    export_model(
        args.model_path,
        args.output_dir,
        args.model_name,
        args.version,
        metadata,
    )

