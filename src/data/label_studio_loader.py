"""Label Studio JSON export loader"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np


class LabelStudioLoader:
    """Load and parse Label Studio JSON exports"""

    def __init__(self, json_path: str):
        """
        Initialize Label Studio loader.

        Args:
            json_path: Path to Label Studio JSON export file
        """
        self.json_path = Path(json_path)
        self.data = self._load_json()

    def _load_json(self) -> List[Dict[str, Any]]:
        """Load JSON file"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks from the export"""
        return self.data

    def get_annotations(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract annotations from a task.

        Args:
            task: Task dictionary from Label Studio export

        Returns:
            List of annotation dictionaries
        """
        annotations = []
        if "annotations" in task:
            for ann in task["annotations"]:
                if "result" in ann:
                    annotations.extend(ann["result"])
        elif "predictions" in task:
            # Use predictions if annotations not available
            for pred in task["predictions"]:
                if "result" in pred:
                    annotations.extend(pred["result"])
        return annotations

    def get_image_path(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Get image path from task data.

        Args:
            task: Task dictionary

        Returns:
            Image path or None
        """
        data = task.get("data", {})
        # Label Studio can store image in different fields
        for key in ["image", "ocr", "img", "image_url"]:
            if key in data:
                return data[key]
        return None

    def extract_text_regions(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract text regions (bounding boxes and text) from a task.

        Args:
            task: Task dictionary

        Returns:
            List of text region dictionaries with keys: bbox, text, confidence
        """
        annotations = self.get_annotations(task)
        regions = []

        for ann in annotations:
            if ann.get("type") == "rectangle" or ann.get("type") == "labels":
                # Get bounding box coordinates
                value = ann.get("value", {})
                original_width = ann.get("original_width", 1)
                original_height = ann.get("original_height", 1)

                # Convert percentage to pixel coordinates
                x = value.get("x", 0) / 100.0 * original_width
                y = value.get("y", 0) / 100.0 * original_height
                width = value.get("width", 0) / 100.0 * original_width
                height = value.get("height", 0) / 100.0 * original_height

                # Get text transcription if available
                text = None
                for related_ann in annotations:
                    if (
                        related_ann.get("type") == "textarea"
                        and related_ann.get("from_name") == ann.get("from_name")
                    ):
                        text_value = related_ann.get("value", {})
                        if "text" in text_value:
                            text = text_value["text"][0] if isinstance(text_value["text"], list) else text_value["text"]

                regions.append(
                    {
                        "bbox": [x, y, x + width, y + height],  # [x1, y1, x2, y2]
                        "text": text or "",
                        "confidence": ann.get("score", 1.0),
                        "label": value.get("labels", [None])[0] if value.get("labels") else None,
                    }
                )

        return regions

    def to_dict(self) -> Dict[str, Any]:
        """Convert loaded data to dictionary format"""
        tasks_data = []
        for task in self.data:
            image_path = self.get_image_path(task)
            regions = self.extract_text_regions(task)
            tasks_data.append({"image_path": image_path, "regions": regions})

        return {"tasks": tasks_data, "total": len(tasks_data)}

