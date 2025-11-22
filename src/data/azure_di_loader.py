"""Load and process Azure Document Intelligence results"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from azure.ai.documentintelligence.models import AnalyzeResult
from ..utils.azure_utils import AzureUtils


class AzureDILoader:
    """Load and convert Azure Document Intelligence results"""

    def __init__(self, result: Optional[AnalyzeResult] = None, json_path: Optional[str] = None):
        """
        Initialize Azure DI loader.

        Args:
            result: AnalyzeResult object from Azure DI
            json_path: Path to saved JSON result file
        """
        if result:
            self.result_dict = result.as_dict()
        elif json_path:
            with open(json_path, "r", encoding="utf-8") as f:
                self.result_dict = json.load(f)
        else:
            raise ValueError("Either result or json_path must be provided")

    def extract_text_regions(self) -> List[Dict[str, Any]]:
        """
        Extract text regions from Azure DI result.

        Returns:
            List of text region dictionaries with bbox, text, confidence
        """
        regions = []
        pages = self.result_dict.get("pages", [])

        for page in pages:
            page_width = page.get("width", 1)
            page_height = page.get("height", 1)

            # Extract words
            words = page.get("words", [])
            for word in words:
                bbox = word.get("polygon", [])
                if len(bbox) >= 4:
                    # Convert polygon to bounding box [x1, y1, x2, y2]
                    x_coords = [point.get("x", 0) for point in bbox]
                    y_coords = [point.get("y", 0) for point in bbox]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)

                    regions.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "text": word.get("content", ""),
                            "confidence": word.get("confidence", 1.0),
                            "page": page.get("pageNumber", 1),
                        }
                    )

        return regions

    def extract_lines(self) -> List[Dict[str, Any]]:
        """
        Extract text lines from Azure DI result.

        Returns:
            List of line dictionaries with bbox, text, confidence
        """
        lines = []
        pages = self.result_dict.get("pages", [])

        for page in pages:
            page_lines = page.get("lines", [])
            for line in page_lines:
                bbox = line.get("polygon", [])
                if len(bbox) >= 4:
                    x_coords = [point.get("x", 0) for point in bbox]
                    y_coords = [point.get("y", 0) for point in bbox]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)

                    lines.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "text": line.get("content", ""),
                            "confidence": line.get("confidence", 1.0),
                            "page": page.get("pageNumber", 1),
                        }
                    )

        return lines

    def to_label_studio_format(self, image_url: str) -> Dict[str, Any]:
        """
        Convert Azure DI result to Label Studio prediction format.

        Args:
            image_url: URL or path to the image

        Returns:
            Label Studio task format dictionary
        """
        regions = self.extract_lines()  # Use lines for better granularity
        predictions = []

        for idx, region in enumerate(regions):
            bbox = region["bbox"]
            # Get image dimensions (assume from first page)
            pages = self.result_dict.get("pages", [])
            if pages:
                page = pages[0]
                width = page.get("width", 1024)
                height = page.get("height", 1024)
            else:
                width, height = 1024, 1024

            # Convert to percentage coordinates
            x_percent = (bbox[0] / width) * 100
            y_percent = (bbox[1] / height) * 100
            width_percent = ((bbox[2] - bbox[0]) / width) * 100
            height_percent = ((bbox[3] - bbox[1]) / height) * 100

            prediction_id = f"pred_{idx}"
            predictions.extend(
                [
                    {
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        "value": {
                            "x": x_percent,
                            "y": y_percent,
                            "width": width_percent,
                            "height": height_percent,
                            "rotation": 0,
                        },
                        "id": prediction_id,
                        "from_name": "bbox",
                        "to_name": "image",
                        "type": "rectangle",
                    },
                    {
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        "value": {
                            "x": x_percent,
                            "y": y_percent,
                            "width": width_percent,
                            "height": height_percent,
                            "rotation": 0,
                            "text": [region["text"]],
                        },
                        "id": prediction_id,
                        "from_name": "transcription",
                        "to_name": "image",
                        "type": "textarea",
                    },
                ]
            )

        return {
            "data": {"image": image_url},
            "predictions": [
                {
                    "model_version": "azure-document-intelligence",
                    "result": predictions,
                    "score": 0.9,  # Default confidence score
                }
            ],
        }

    @staticmethod
    def analyze_and_convert_to_label_studio(
        document_path: str,
        image_url: str,
        model_id: str = "prebuilt-layout",
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze document with Azure DI and convert to Label Studio format.

        Args:
            document_path: Path to document file
            image_url: URL or path to the image for Label Studio
            model_id: Azure DI model ID
            endpoint: Azure DI endpoint (optional, uses env var if not provided)
            key: Azure DI API key (optional, uses env var if not provided)

        Returns:
            Label Studio task format dictionary
        """
        client = AzureUtils.get_document_intelligence_client(endpoint, key)
        result = AzureUtils.analyze_document(client, document_path, model_id)

        loader = AzureDILoader(result=result)
        return loader.to_label_studio_format(image_url)

