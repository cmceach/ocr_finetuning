"""Comprehensive OCR evaluation utilities"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import torch
from tqdm import tqdm

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


# =============================================================================
# Nemotron Parse Evaluation
# =============================================================================


def parse_nemotron_output_for_eval(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse Nemotron Parse output into structured regions for evaluation.
    
    Supports both Nemotron Parse format (<x1><y1><x2><y2>text<class>)
    and legacy format (<label><bbox>x1,y1,x2,y2</bbox>text</label>).

    Args:
        raw_text: Raw output from Nemotron Parse model

    Returns:
        List of region dictionaries with text, bbox, and label
    """
    regions = []

    # Pattern for Nemotron Parse format: <x1><y1><x2><y2>text<class>
    # Coordinates are normalized floats, class is a word
    nemotron_pattern = r'<([\d.]+)><([\d.]+)><([\d.]+)><([\d.]+)>(.*?)<(\w+)>'
    matches = re.findall(nemotron_pattern, raw_text, re.DOTALL)

    for match in matches:
        x1, y1, x2, y2, text, label = match
        try:
            bbox = [float(x1), float(y1), float(x2), float(y2)]
        except (ValueError, AttributeError):
            bbox = None

        regions.append({
            "text": text.strip(),
            "bbox": bbox,
            "label": label,
        })

    # If no Nemotron format found, try legacy format: <label><bbox>x1,y1,x2,y2</bbox>text</label>
    if not regions:
        legacy_pattern = r'<(\w+)><bbox>([\d.,]+)</bbox>(.*?)</\1>'
        legacy_matches = re.findall(legacy_pattern, raw_text, re.DOTALL)
        
        for match in legacy_matches:
            label, bbox_str, text = match
            try:
                bbox = [float(x.strip()) for x in bbox_str.split(",")]
            except (ValueError, AttributeError):
                bbox = None

            regions.append({
                "text": text.strip(),
                "bbox": bbox,
                "label": label,
            })

    # If still no structured output found, extract plain text
    if not regions:
        # Clean up special tokens and extract text blocks
        clean_text = re.sub(r'<[^>]+>', '', raw_text)
        lines = clean_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line:
                regions.append({"text": line, "bbox": None, "label": None})

    return regions


def calculate_text_similarity(pred_text: str, gt_text: str) -> Dict[str, float]:
    """
    Calculate text similarity metrics between prediction and ground truth.

    Args:
        pred_text: Predicted text
        gt_text: Ground truth text

    Returns:
        Dictionary with similarity metrics
    """
    # Character-level metrics
    pred_chars = list(pred_text)
    gt_chars = list(gt_text)

    # Levenshtein distance (edit distance)
    edit_distance = _levenshtein_distance(pred_text, gt_text)
    max_len = max(len(pred_text), len(gt_text), 1)
    char_accuracy = 1.0 - (edit_distance / max_len)

    # Word-level metrics
    pred_words = pred_text.split()
    gt_words = gt_text.split()

    word_edit_distance = _levenshtein_distance_list(pred_words, gt_words)
    max_word_len = max(len(pred_words), len(gt_words), 1)
    word_accuracy = 1.0 - (word_edit_distance / max_word_len)

    # Exact match
    exact_match = float(pred_text.strip() == gt_text.strip())

    return {
        "char_accuracy": char_accuracy,
        "word_accuracy": word_accuracy,
        "exact_match": exact_match,
        "edit_distance": edit_distance,
        "word_edit_distance": word_edit_distance,
    }


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _levenshtein_distance_list(list1: List[str], list2: List[str]) -> int:
    """Calculate Levenshtein distance between two lists of strings"""
    if len(list1) < len(list2):
        return _levenshtein_distance_list(list2, list1)

    if len(list2) == 0:
        return len(list1)

    previous_row = range(len(list2) + 1)
    for i, item1 in enumerate(list1):
        current_row = [i + 1]
        for j, item2 in enumerate(list2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (item1 != item2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


@torch.inference_mode()
def evaluate_nemotron_model(
    model,
    processor,
    test_data: List[Dict[str, Any]],
    config: Optional[ConfigLoader] = None,
    max_new_tokens: int = 4096,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate Nemotron Parse model on test data.

    Args:
        model: Nemotron Parse model
        processor: Model processor
        test_data: List of test samples with image_path and target_text
        config: Configuration loader instance
        max_new_tokens: Maximum tokens to generate
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with evaluation metrics and detailed results
    """
    if config is None:
        config = ConfigLoader()

    device = next(model.parameters()).device

    all_results = []
    all_char_accuracies = []
    all_word_accuracies = []
    all_exact_matches = []

    iterator = tqdm(test_data, desc="Evaluating") if show_progress else test_data

    for sample in iterator:
        image_path = sample.get("image_path")
        gt_text = sample.get("target_text", "")

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Prepare inputs
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            # Decode
            pred_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Parse outputs
            pred_regions = parse_nemotron_output_for_eval(pred_text)
            gt_regions = parse_nemotron_output_for_eval(gt_text)

            # Extract text for comparison
            pred_full_text = " ".join(r["text"] for r in pred_regions if r["text"])
            gt_full_text = " ".join(r["text"] for r in gt_regions if r["text"])

            # Calculate metrics
            similarity = calculate_text_similarity(pred_full_text, gt_full_text)

            all_char_accuracies.append(similarity["char_accuracy"])
            all_word_accuracies.append(similarity["word_accuracy"])
            all_exact_matches.append(similarity["exact_match"])

            all_results.append({
                "image_path": image_path,
                "prediction": pred_text,
                "ground_truth": gt_text,
                "pred_regions": pred_regions,
                "gt_regions": gt_regions,
                "metrics": similarity,
            })

        except Exception as e:
            all_results.append({
                "image_path": image_path,
                "error": str(e),
            })

    # Aggregate metrics
    num_samples = len(all_char_accuracies)
    metrics = {
        "num_samples": num_samples,
        "num_errors": len([r for r in all_results if "error" in r]),
        "avg_char_accuracy": sum(all_char_accuracies) / num_samples if num_samples > 0 else 0,
        "avg_word_accuracy": sum(all_word_accuracies) / num_samples if num_samples > 0 else 0,
        "exact_match_rate": sum(all_exact_matches) / num_samples if num_samples > 0 else 0,
    }

    return {
        "metrics": metrics,
        "detailed_results": all_results,
    }


def evaluate_nemotron_from_file(
    model,
    processor,
    test_data_path: str,
    config: Optional[ConfigLoader] = None,
    output_path: Optional[str] = None,
    max_new_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Evaluate Nemotron Parse model from test data file.

    Args:
        model: Nemotron Parse model
        processor: Model processor
        test_data_path: Path to test data JSON file
        config: Configuration loader instance
        output_path: Optional path to save detailed results
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with evaluation metrics
    """
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    results = evaluate_nemotron_model(
        model, processor, test_data, config, max_new_tokens
    )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {output_path}")

    return results


def compare_nemotron_with_baseline(
    nemotron_model,
    processor,
    test_data: List[Dict[str, Any]],
    baseline_model=None,
    config: Optional[ConfigLoader] = None,
) -> Dict[str, Any]:
    """
    Compare Nemotron Parse with a baseline model (e.g., Azure DI or DocTR).

    Args:
        nemotron_model: Nemotron Parse model
        processor: Nemotron processor
        test_data: Test data samples
        baseline_model: Baseline model (DocTR predictor or None for Azure DI)
        config: Configuration loader

    Returns:
        Comparison results
    """
    # Evaluate Nemotron
    print("Evaluating Nemotron Parse...")
    nemotron_results = evaluate_nemotron_model(
        nemotron_model, processor, test_data, config
    )

    comparison = {
        "nemotron": nemotron_results["metrics"],
    }

    # Evaluate baseline if provided
    if baseline_model is not None:
        print("Evaluating baseline model...")
        baseline_results = evaluate_ocr_model(baseline_model, test_data, config)
        comparison["baseline"] = baseline_results

        # Calculate improvements
        if "avg_char_accuracy" in nemotron_results["metrics"]:
            nemotron_char = nemotron_results["metrics"]["avg_char_accuracy"]
            baseline_char = baseline_results.get("char_accuracy", 0)
            comparison["improvement"] = {
                "char_accuracy_delta": nemotron_char - baseline_char,
                "char_accuracy_relative": (nemotron_char - baseline_char) / baseline_char if baseline_char > 0 else 0,
            }

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate OCR models")
    parser.add_argument("--model-type", choices=["doctr", "nemotron"], required=True,
                        help="Type of model to evaluate")
    parser.add_argument("--model-path", type=str, help="Path to model or adapter")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output", type=str, help="Path to save results")
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    config = ConfigLoader(args.config) if args.config else None

    if args.model_type == "nemotron":
        from ..training.nemotron_model import load_finetuned_nemotron

        model, processor = load_finetuned_nemotron(
            adapter_path=args.model_path,
        )
        results = evaluate_nemotron_from_file(
            model, processor, args.test_data, config, args.output
        )
    else:
        from ..serving.model_loader import load_model_for_inference

        model = load_model_for_inference(model_path=args.model_path, config=config)
        results = evaluate_from_file(model, args.test_data, config)

    print("\nEvaluation Results:")
    print(json.dumps(results.get("metrics", results), indent=2))

