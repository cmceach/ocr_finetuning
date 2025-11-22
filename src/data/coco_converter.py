"""Convert Label Studio JSON to COCO format"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from .label_studio_loader import LabelStudioLoader


class COCOConverter:
    """Convert Label Studio exports to COCO format"""

    def __init__(self, label_studio_loader: LabelStudioLoader):
        """
        Initialize COCO converter.

        Args:
            label_studio_loader: LabelStudioLoader instance with loaded data
        """
        self.loader = label_studio_loader
        self.coco_data = {
            "info": {
                "description": "OCR Dataset converted from Label Studio",
                "version": "1.0",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "text", "supercategory": "text"}],
        }
        self.image_id_map = {}
        self.annotation_id = 1

    def convert(self) -> Dict[str, Any]:
        """
        Convert Label Studio data to COCO format.

        Returns:
            COCO format dictionary
        """
        tasks = self.loader.get_tasks()

        for task_idx, task in enumerate(tasks):
            image_path = self.loader.get_image_path(task)
            if not image_path:
                continue

            # Add image entry
            image_id = task_idx + 1
            self.image_id_map[image_path] = image_id

            # Get image dimensions (try to load image)
            width, height = self._get_image_dimensions(image_path)

            self.coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": Path(image_path).name,
                    "width": width,
                    "height": height,
                }
            )

            # Add annotations
            regions = self.loader.extract_text_regions(task)
            for region in regions:
                bbox = region["bbox"]
                # COCO format: [x, y, width, height] (top-left corner + size)
                coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

                self.coco_data["annotations"].append(
                    {
                        "id": self.annotation_id,
                        "image_id": image_id,
                        "category_id": 1,  # text category
                        "bbox": coco_bbox,
                        "area": coco_bbox[2] * coco_bbox[3],
                        "iscrowd": 0,
                        "text": region.get("text", ""),
                        "confidence": region.get("confidence", 1.0),
                    }
                )
                self.annotation_id += 1

        return self.coco_data

    def _get_image_dimensions(self, image_path: str) -> tuple:
        """
        Get image dimensions.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (width, height)
        """
        try:
            from PIL import Image

            # Handle URL paths
            if image_path.startswith("http"):
                import requests
                from io import BytesIO

                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_path)

            return img.size  # Returns (width, height)
        except Exception:
            # Default dimensions if image cannot be loaded
            return (1024, 1024)

    def save(self, output_path: str):
        """
        Save COCO format data to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        coco_data = self.convert()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def from_label_studio_json(json_path: str, output_path: str):
        """
        Convert Label Studio JSON directly to COCO format file.

        Args:
            json_path: Path to Label Studio JSON export
            output_path: Path to output COCO JSON file
        """
        loader = LabelStudioLoader(json_path)
        converter = COCOConverter(loader)
        converter.save(output_path)

