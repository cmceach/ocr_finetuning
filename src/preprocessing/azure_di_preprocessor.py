"""Use Azure Document Intelligence for pre-annotation assistance"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..data.azure_di_loader import AzureDILoader
from ..utils.azure_utils import AzureUtils


class AzureDIPreprocessor:
    """Preprocess documents using Azure Document Intelligence for pre-annotation"""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        model_id: str = "prebuilt-layout",
    ):
        """
        Initialize Azure DI preprocessor.

        Args:
            endpoint: Azure Document Intelligence endpoint
            key: Azure Document Intelligence API key
            model_id: Model ID to use for analysis
        """
        self.client = AzureUtils.get_document_intelligence_client(endpoint, key)
        self.model_id = model_id

    def preannotate_document(self, document_path: str, image_url: str) -> Dict[str, Any]:
        """
        Pre-annotate a document using Azure DI.

        Args:
            document_path: Path to document file
            image_url: URL or path to image for Label Studio

        Returns:
            Label Studio task format dictionary with predictions
        """
        return AzureDILoader.analyze_and_convert_to_label_studio(
            document_path, image_url, self.model_id
        )

    def preannotate_batch(
        self, document_paths: List[str], image_urls: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Pre-annotate a batch of documents.

        Args:
            document_paths: List of document file paths
            image_urls: List of corresponding image URLs

        Returns:
            List of Label Studio task format dictionaries
        """
        if len(document_paths) != len(image_urls):
            raise ValueError("document_paths and image_urls must have the same length")

        results = []
        for doc_path, img_url in zip(document_paths, image_urls):
            try:
                result = self.preannotate_document(doc_path, img_url)
                results.append(result)
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                continue

        return results

    def save_preannotations(
        self, preannotations: List[Dict[str, Any]], output_path: str
    ):
        """
        Save pre-annotations to JSON file for Label Studio import.

        Args:
            preannotations: List of Label Studio task dictionaries
            output_path: Path to output JSON file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(preannotations, f, indent=2, ensure_ascii=False)

