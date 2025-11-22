"""Convert Label Studio JSON to DocTR native format"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from .label_studio_loader import LabelStudioLoader


class DocTRConverter:
    """Convert Label Studio exports to DocTR training format"""

    def __init__(self, label_studio_loader: LabelStudioLoader):
        """
        Initialize DocTR converter.

        Args:
            label_studio_loader: LabelStudioLoader instance with loaded data
        """
        self.loader = label_studio_loader

    def convert_for_detection(self) -> List[Dict[str, Any]]:
        """
        Convert to DocTR detection format.

        Returns:
            List of dictionaries with image paths and bounding boxes
        """
        tasks = self.loader.get_tasks()
        detection_data = []

        for task in tasks:
            image_path = self.loader.get_image_path(task)
            if not image_path:
                continue

            regions = self.loader.extract_text_regions(task)
            if not regions:
                continue

            # DocTR detection format: list of polygons (4 points for each bbox)
            boxes = []
            for region in regions:
                bbox = region["bbox"]
                # Convert to polygon format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                polygon = [
                    [bbox[0], bbox[1]],  # top-left
                    [bbox[2], bbox[1]],  # top-right
                    [bbox[2], bbox[3]],  # bottom-right
                    [bbox[0], bbox[3]],  # bottom-left
                ]
                boxes.append(polygon)

            detection_data.append({"image_path": image_path, "boxes": boxes})

        return detection_data

    def convert_for_recognition(self) -> List[Dict[str, Any]]:
        """
        Convert to DocTR recognition format.

        Returns:
            List of dictionaries with image paths, cropped regions, and text labels
        """
        tasks = self.loader.get_tasks()
        recognition_data = []

        for task in tasks:
            image_path = self.loader.get_image_path(task)
            if not image_path:
                continue

            regions = self.loader.extract_text_regions(task)
            if not regions:
                continue

            # Load image to get dimensions
            try:
                if image_path.startswith("http"):
                    import requests
                    from io import BytesIO

                    response = requests.get(image_path)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(image_path)
                img_width, img_height = img.size
            except Exception:
                continue

            for region in regions:
                bbox = region["bbox"]
                text = region.get("text", "").strip()

                if not text:
                    continue

                # Normalize coordinates to [0, 1] range
                normalized_bbox = [
                    bbox[0] / img_width,  # x1
                    bbox[1] / img_height,  # y1
                    bbox[2] / img_width,  # x2
                    bbox[3] / img_height,  # y2
                ]

                recognition_data.append(
                    {
                        "image_path": image_path,
                        "bbox": normalized_bbox,
                        "text": text,
                    }
                )

        return recognition_data

    def convert_for_full_pipeline(self) -> Dict[str, Any]:
        """
        Convert to DocTR full pipeline format (detection + recognition).

        Returns:
            Dictionary with detection and recognition data
        """
        return {
            "detection": self.convert_for_detection(),
            "recognition": self.convert_for_recognition(),
        }

    def save_detection(self, output_path: str):
        """
        Save detection format data to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        data = self.convert_for_detection()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_recognition(self, output_path: str):
        """
        Save recognition format data to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        data = self.convert_for_recognition()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_full_pipeline(self, output_dir: str):
        """
        Save full pipeline data to directory.

        Args:
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = self.convert_for_full_pipeline()
        with open(output_dir / "detection.json", "w", encoding="utf-8") as f:
            json.dump(data["detection"], f, indent=2, ensure_ascii=False)
        with open(output_dir / "recognition.json", "w", encoding="utf-8") as f:
            json.dump(data["recognition"], f, indent=2, ensure_ascii=False)

    @staticmethod
    def from_label_studio_json(
        json_path: str, output_path: str, format_type: str = "detection"
    ):
        """
        Convert Label Studio JSON directly to DocTR format file.

        Args:
            json_path: Path to Label Studio JSON export
            output_path: Path to output JSON file or directory
            format_type: Format type - 'detection', 'recognition', or 'full'
        """
        loader = LabelStudioLoader(json_path)
        converter = DocTRConverter(loader)

        if format_type == "detection":
            converter.save_detection(output_path)
        elif format_type == "recognition":
            converter.save_recognition(output_path)
        elif format_type == "full":
            converter.save_full_pipeline(output_path)
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

