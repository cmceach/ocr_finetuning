"""Tests for data conversion modules"""

import pytest
import json
import tempfile
from pathlib import Path
from src.data.label_studio_loader import LabelStudioLoader
from src.data.coco_converter import COCOConverter
from src.data.doctr_converter import DocTRConverter


def test_coco_converter():
    """Test COCO converter"""
    # Create sample Label Studio JSON
    sample_data = [
        {
            "id": 1,
            "data": {"image": "test.jpg"},
            "annotations": [
                {
                    "result": [
                        {
                            "original_width": 1000,
                            "original_height": 1000,
                            "value": {
                                "x": 10,
                                "y": 20,
                                "width": 30,
                                "height": 40,
                            },
                            "type": "rectangle",
                        },
                        {
                            "original_width": 1000,
                            "original_height": 1000,
                            "value": {
                                "x": 10,
                                "y": 20,
                                "width": 30,
                                "height": 40,
                                "text": ["Test"],
                            },
                            "type": "textarea",
                        },
                    ]
                }
            ],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_path = f.name

    try:
        loader = LabelStudioLoader(temp_path)
        converter = COCOConverter(loader)
        coco_data = converter.convert()

        assert "images" in coco_data
        assert "annotations" in coco_data
        assert len(coco_data["images"]) > 0
    finally:
        Path(temp_path).unlink()


def test_doctr_converter():
    """Test DocTR converter"""
    # Similar test structure as COCO converter
    sample_data = [
        {
            "id": 1,
            "data": {"image": "test.jpg"},
            "annotations": [
                {
                    "result": [
                        {
                            "original_width": 1000,
                            "original_height": 1000,
                            "value": {
                                "x": 10,
                                "y": 20,
                                "width": 30,
                                "height": 40,
                            },
                            "type": "rectangle",
                        },
                        {
                            "original_width": 1000,
                            "original_height": 1000,
                            "value": {
                                "x": 10,
                                "y": 20,
                                "width": 30,
                                "height": 40,
                                "text": ["Test"],
                            },
                            "type": "textarea",
                        },
                    ]
                }
            ],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_path = f.name

    try:
        loader = LabelStudioLoader(temp_path)
        converter = DocTRConverter(loader)
        detection_data = converter.convert_for_detection()
        recognition_data = converter.convert_for_recognition()

        assert len(detection_data) > 0
        assert len(recognition_data) > 0
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])

