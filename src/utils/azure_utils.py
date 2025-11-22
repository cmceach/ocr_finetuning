"""Azure SDK helper utilities"""

import os
from typing import Optional
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat


class AzureUtils:
    """Utility class for Azure service interactions"""

    @staticmethod
    def get_document_intelligence_client(
        endpoint: Optional[str] = None, key: Optional[str] = None
    ) -> DocumentIntelligenceClient:
        """
        Create and return a Document Intelligence client.

        Args:
            endpoint: Azure Document Intelligence endpoint URL
            key: Azure Document Intelligence API key

        Returns:
            DocumentIntelligenceClient instance
        """
        if endpoint is None:
            endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        if key is None:
            key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not endpoint or not key:
            raise ValueError(
                "Azure Document Intelligence endpoint and key must be provided "
                "either as arguments or environment variables"
            )

        credential = AzureKeyCredential(key)
        return DocumentIntelligenceClient(endpoint=endpoint, credential=credential)

    @staticmethod
    def get_default_credential() -> DefaultAzureCredential:
        """
        Get default Azure credential for managed identity or service principal.

        Returns:
            DefaultAzureCredential instance
        """
        return DefaultAzureCredential()

    @staticmethod
    def analyze_document(
        client: DocumentIntelligenceClient,
        document_path: str,
        model_id: str = "prebuilt-layout",
        content_format: Optional[DocumentContentFormat] = None,
    ) -> dict:
        """
        Analyze a document using Azure Document Intelligence.

        Args:
            client: DocumentIntelligenceClient instance
            document_path: Path to document file
            model_id: Model ID to use (default: prebuilt-layout)
            content_format: Optional content format (e.g., DocumentContentFormat.MARKDOWN)

        Returns:
            Analysis result as dictionary
        """
        with open(document_path, "rb") as f:
            request = AnalyzeDocumentRequest()
            if content_format:
                request.output_content_format = content_format

            poller = client.begin_analyze_document(model_id, body=f, output_content_format=content_format)
            result = poller.result()

            return result.as_dict()

