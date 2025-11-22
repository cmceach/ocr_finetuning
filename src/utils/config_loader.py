"""Configuration management utilities"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration files"""

    def __init__(self, config_path: Optional[str] = None, client_config: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to default configuration file
            client_config: Path to client-specific configuration file
        """
        # Load environment variables
        load_dotenv()

        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"

        # Load default configuration
        with open(config_path, "r") as f:
            self.default_config = yaml.safe_load(f)

        # Load client-specific configuration if provided
        self.config = self.default_config.copy()
        if client_config:
            with open(client_config, "r") as f:
                client_config_dict = yaml.safe_load(f)
                self.config = self._merge_config(self.default_config, client_config_dict)

        # Override with environment variables
        self._apply_env_overrides()

    def _merge_config(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration"""
        # Azure Document Intelligence
        if os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"):
            self.config.setdefault("azure_di", {})["endpoint"] = os.getenv(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
            )
        if os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"):
            self.config.setdefault("azure_di", {})["key"] = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if os.getenv("AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID"):
            self.config.setdefault("azure_di", {})["model_id"] = os.getenv(
                "AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID"
            )

        # Label Studio
        if os.getenv("LABEL_STUDIO_URL"):
            self.config.setdefault("label_studio", {})["url"] = os.getenv("LABEL_STUDIO_URL")
        if os.getenv("LABEL_STUDIO_API_KEY"):
            self.config.setdefault("label_studio", {})["api_key"] = os.getenv("LABEL_STUDIO_API_KEY")

        # MLflow
        if os.getenv("MLFLOW_TRACKING_URI"):
            self.config.setdefault("logging", {}).setdefault("mlflow", {})["tracking_uri"] = os.getenv(
                "MLFLOW_TRACKING_URI"
            )
        if os.getenv("MLFLOW_EXPERIMENT_NAME"):
            self.config.setdefault("logging", {}).setdefault("mlflow", {})["experiment_name"] = os.getenv(
                "MLFLOW_EXPERIMENT_NAME"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_config(self) -> Dict[str, Any]:
        """Get full configuration dictionary"""
        return self.config.copy()


def load_config(config_path: Optional[str] = None, client_config: Optional[str] = None) -> ConfigLoader:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to default configuration file
        client_config: Path to client-specific configuration file

    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path, client_config)

