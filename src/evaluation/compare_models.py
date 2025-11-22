"""Compare DocTR models with Azure Document Intelligence"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..data.azure_di_loader import AzureDILoader
from ..utils.azure_utils import AzureUtils
from .evaluate_ocr import evaluate_ocr_model
from ..utils.config_loader import ConfigLoader


def compare_with_azure_di(
    doctr_model,
    test_data: List[Dict[str, Any]],
    config: Optional[ConfigLoader] = None,
) -> Dict[str, Any]:
    """
    Compare DocTR model performance with Azure Document Intelligence baseline.

    Args:
        doctr_model: Trained DocTR model
        test_data: List of test samples
        config: Configuration loader instance

    Returns:
        Dictionary with comparison results
    """
    if config is None:
        config = ConfigLoader()

    # Evaluate DocTR model
    doctr_metrics = evaluate_ocr_model(doctr_model, test_data, config)

    # Evaluate Azure DI
    azure_di_config = config.get("azure_di", {})
    model_id = azure_di_config.get("model_id", "prebuilt-layout")

    client = AzureUtils.get_document_intelligence_client()
    azure_results = []

    for sample in test_data:
        image_path = sample.get("image_path")
        try:
            result = AzureUtils.analyze_document(client, image_path, model_id)
            loader = AzureDILoader(result=result)
            regions = loader.extract_text_regions()

            azure_results.append(
                {
                    "image_path": image_path,
                    "regions": regions,
                }
            )
        except Exception as e:
            print(f"Error processing {image_path} with Azure DI: {e}")
            continue

    # Calculate Azure DI metrics
    # Convert Azure DI results to same format as test data for evaluation
    azure_test_data = []
    for sample, azure_result in zip(test_data, azure_results):
        azure_test_data.append(
            {
                "image_path": sample["image_path"],
                "regions": [
                    {
                        "bbox": r["bbox"],
                        "text": r["text"],
                        "confidence": r.get("confidence", 1.0),
                    }
                    for r in azure_result["regions"]
                ],
            }
        )

    # Note: Azure DI evaluation would need a mock model wrapper
    # For now, return the metrics separately
    return {
        "doctr": doctr_metrics,
        "azure_di": {
            "note": "Azure DI metrics calculation requires model wrapper implementation",
            "samples_processed": len(azure_results),
        },
        "comparison": {
            "doctr_precision": doctr_metrics.get("precision", 0),
            "doctr_recall": doctr_metrics.get("recall", 0),
            "doctr_f1": doctr_metrics.get("f1_score", 0),
            "doctr_char_accuracy": doctr_metrics.get("character_accuracy", 0),
            "doctr_word_accuracy": doctr_metrics.get("word_accuracy", 0),
        },
    }


def compare_from_file(
    doctr_model,
    test_data_path: str,
    config: Optional[ConfigLoader] = None,
) -> Dict[str, Any]:
    """
    Compare models from test data file.

    Args:
        doctr_model: Trained DocTR model
        test_data_path: Path to test data JSON file
        config: Configuration loader instance

    Returns:
        Dictionary with comparison results
    """
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    return compare_with_azure_di(doctr_model, test_data, config)

