"""Tests for training modules"""

import pytest
from src.training.metrics import OCRMetrics


def test_iou_calculation():
    """Test IoU calculation"""
    bbox1 = [0, 0, 10, 10]
    bbox2 = [5, 5, 15, 15]

    iou = OCRMetrics.calculate_iou(bbox1, bbox2)
    assert 0 <= iou <= 1
    assert iou > 0  # Should have some overlap


def test_detection_metrics():
    """Test detection metrics calculation"""
    pred_bboxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    gt_bboxes = [[1, 1, 11, 11], [21, 21, 31, 31]]

    metrics = OCRMetrics.calculate_detection_metrics(pred_bboxes, gt_bboxes)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1


def test_character_accuracy():
    """Test character accuracy calculation"""
    pred_text = "Hello World"
    gt_text = "Hello World"

    accuracy = OCRMetrics.calculate_character_accuracy(pred_text, gt_text)
    assert accuracy == 1.0

    pred_text = "Hallo World"
    accuracy = OCRMetrics.calculate_character_accuracy(pred_text, gt_text)
    assert 0 < accuracy < 1.0


if __name__ == "__main__":
    pytest.main([__file__])

