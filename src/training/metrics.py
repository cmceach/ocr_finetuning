"""Custom OCR metrics for evaluation"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score


class OCRMetrics:
    """OCR-specific metrics calculator"""

    @staticmethod
    def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU score between 0 and 1
        """
        # Calculate intersection
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def match_bboxes(
        pred_bboxes: List[List[float]], gt_bboxes: List[List[float]], iou_threshold: float = 0.5
    ) -> Tuple[List[int], List[int]]:
        """
        Match predicted bounding boxes to ground truth boxes.

        Args:
            pred_bboxes: List of predicted bounding boxes
            gt_bboxes: List of ground truth bounding boxes
            iou_threshold: IoU threshold for matching

        Returns:
            Tuple of (matched_pred_indices, matched_gt_indices)
        """
        matched_pred = []
        matched_gt = []
        used_gt = set()

        for pred_idx, pred_bbox in enumerate(pred_bboxes):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_bbox in enumerate(gt_bboxes):
                if gt_idx in used_gt:
                    continue

                iou = OCRMetrics.calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                matched_pred.append(pred_idx)
                matched_gt.append(best_gt_idx)
                used_gt.add(best_gt_idx)

        return matched_pred, matched_gt

    @staticmethod
    def calculate_detection_metrics(
        pred_bboxes: List[List[float]],
        gt_bboxes: List[List[float]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calculate detection metrics (precision, recall, F1).

        Args:
            pred_bboxes: List of predicted bounding boxes
            gt_bboxes: List of ground truth bounding boxes
            iou_threshold: IoU threshold for matching

        Returns:
            Dictionary with precision, recall, and f1_score
        """
        if len(gt_bboxes) == 0:
            if len(pred_bboxes) == 0:
                return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
            else:
                return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        matched_pred, matched_gt = OCRMetrics.match_bboxes(
            pred_bboxes, gt_bboxes, iou_threshold
        )

        tp = len(matched_pred)
        fp = len(pred_bboxes) - tp
        fn = len(gt_bboxes) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    @staticmethod
    def calculate_character_accuracy(pred_text: str, gt_text: str) -> float:
        """
        Calculate character-level accuracy.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text

        Returns:
            Character accuracy between 0 and 1
        """
        if len(gt_text) == 0:
            return 1.0 if len(pred_text) == 0 else 0.0

        pred_chars = list(pred_text.lower().replace(" ", ""))
        gt_chars = list(gt_text.lower().replace(" ", ""))

        if len(gt_chars) == 0:
            return 1.0 if len(pred_chars) == 0 else 0.0

        correct = sum(1 for p, g in zip(pred_chars, gt_chars) if p == g)
        return correct / max(len(pred_chars), len(gt_chars))

    @staticmethod
    def calculate_word_accuracy(pred_text: str, gt_text: str) -> float:
        """
        Calculate word-level accuracy.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text

        Returns:
            Word accuracy between 0 and 1
        """
        pred_words = pred_text.lower().split()
        gt_words = gt_text.lower().split()

        if len(gt_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0

        correct = sum(1 for p, g in zip(pred_words, gt_words) if p == g)
        return correct / max(len(pred_words), len(gt_words))

    @staticmethod
    def calculate_recognition_metrics(
        pred_texts: List[str], gt_texts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate recognition metrics across multiple text predictions.

        Args:
            pred_texts: List of predicted texts
            gt_texts: List of ground truth texts

        Returns:
            Dictionary with character_accuracy and word_accuracy
        """
        if len(pred_texts) != len(gt_texts):
            raise ValueError("pred_texts and gt_texts must have the same length")

        char_accuracies = [
            OCRMetrics.calculate_character_accuracy(p, g) for p, g in zip(pred_texts, gt_texts)
        ]
        word_accuracies = [
            OCRMetrics.calculate_word_accuracy(p, g) for p, g in zip(pred_texts, gt_texts)
        ]

        return {
            "character_accuracy": np.mean(char_accuracies),
            "word_accuracy": np.mean(word_accuracies),
        }

