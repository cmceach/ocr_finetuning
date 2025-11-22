"""Comprehensive OCR evaluation utilities"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..training.metrics import OCRMetrics
from ..utils.config_loader import ConfigLoader


def evaluate_ocr_model(
    model,
    test_data: List[Dict[str, Any]],
    config: Optional[ConfigLoader] = None,
) -> Dict[str, float]:
    """
    Evaluate OCR model on test data.

    Args:
        model: Trained OCR model (DocTR predictor)
        test_data: List of test samples with image paths and ground truth
        config: Configuration loader instance

    Returns:
        Dictionary with evaluation metrics
    """
    if config is None:
        config = ConfigLoader()

    from doctr.io import DocumentFile

    all_pred_bboxes = []
    all_gt_bboxes = []
    all_pred_texts = []
    all_gt_texts = []

    for sample in test_data:
        image_path = sample.get("image_path")
        gt_regions = sample.get("regions", [])

        # Run inference
        doc = DocumentFile.from_images(image_path)
        result = model(doc)

        # Extract predictions
        pred_bboxes = []
        pred_texts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Convert DocTR geometry to bbox format
                        geometry = word.geometry
                        bbox = [
                            geometry[0][0],  # x1
                            geometry[0][1],  # y1
                            geometry[1][0],  # x2
                            geometry[1][1],  # y2
                        ]
                        pred_bboxes.append(bbox)
                        pred_texts.append(word.value)

        # Extract ground truth
        gt_bboxes = [r["bbox"] for r in gt_regions]
        gt_texts = [r.get("text", "") for r in gt_regions]

        all_pred_bboxes.append(pred_bboxes)
        all_gt_bboxes.append(gt_bboxes)
        all_pred_texts.extend(pred_texts)
        all_gt_texts.extend(gt_texts)

    # Calculate detection metrics
    detection_metrics = []
    for pred_bboxes, gt_bboxes in zip(all_pred_bboxes, all_gt_bboxes):
        metrics = OCRMetrics.calculate_detection_metrics(pred_bboxes, gt_bboxes)
        detection_metrics.append(metrics)

    # Aggregate detection metrics
    avg_detection_metrics = {
        "precision": sum(m["precision"] for m in detection_metrics) / len(detection_metrics),
        "recall": sum(m["recall"] for m in detection_metrics) / len(detection_metrics),
        "f1_score": sum(m["f1_score"] for m in detection_metrics) / len(detection_metrics),
    }

    # Calculate recognition metrics
    recognition_metrics = OCRMetrics.calculate_recognition_metrics(all_pred_texts, all_gt_texts)

    # Combine all metrics
    return {
        **avg_detection_metrics,
        **recognition_metrics,
    }


def evaluate_from_file(
    model,
    test_data_path: str,
    config: Optional[ConfigLoader] = None,
) -> Dict[str, float]:
    """
    Evaluate OCR model from test data file.

    Args:
        model: Trained OCR model
        test_data_path: Path to test data JSON file
        config: Configuration loader instance

    Returns:
        Dictionary with evaluation metrics
    """
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    return evaluate_ocr_model(model, test_data, config)

