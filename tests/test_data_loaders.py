"""Tests for data loading modules"""

import pytest
import json
import tempfile
from pathlib import Path
from src.data.label_studio_loader import LabelStudioLoader


def test_label_studio_loader():
    """Test Label Studio loader"""
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
                            "from_name": "bbox",
                            "to_name": "image",
                        },
                        {
                            "original_width": 1000,
                            "original_height": 1000,
                            "value": {
                                "x": 10,
                                "y": 20,
                                "width": 30,
                                "height": 40,
                                "text": ["Hello World"],
                            },
                            "type": "textarea",
                            "from_name": "transcription",
                            "to_name": "image",
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
        tasks = loader.get_tasks()
        assert len(tasks) == 1

        regions = loader.extract_text_regions(tasks[0])
        assert len(regions) > 0
        assert regions[0]["text"] == "Hello World"
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])

