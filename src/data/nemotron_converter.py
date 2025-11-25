"""Data converter and dataset for Nemotron Parse finetuning"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

from .label_studio_loader import LabelStudioLoader


class NemotronDataConverter:
    """Convert Label Studio annotations to Nemotron Parse training format"""

    def __init__(
        self,
        image_base_dir: Optional[str] = None,
        max_image_size: int = 1024,
    ):
        """
        Initialize the converter.

        Args:
            image_base_dir: Base directory for resolving relative image paths
            max_image_size: Maximum dimension for image resizing
        """
        self.image_base_dir = Path(image_base_dir) if image_base_dir else None
        self.max_image_size = max_image_size

    def convert_from_label_studio(
        self,
        json_path: str,
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert Label Studio export to Nemotron training format.

        Args:
            json_path: Path to Label Studio JSON export
            output_path: Optional path to save converted data

        Returns:
            List of training samples in Nemotron format
        """
        loader = LabelStudioLoader(json_path)
        samples = []

        for task in loader.get_tasks():
            image_path = loader.get_image_path(task)
            if not image_path:
                continue

            # Resolve image path
            resolved_path = self._resolve_image_path(image_path)
            if not resolved_path or not os.path.exists(resolved_path):
                continue

            # Extract text regions
            regions = loader.extract_text_regions(task)

            # Convert to Nemotron format (structured text output)
            target_text = self._format_nemotron_output(regions)

            samples.append({
                "image_path": str(resolved_path),
                "target_text": target_text,
                "regions": regions,
                "task_id": task.get("id"),
            })

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)

        return samples

    def _resolve_image_path(self, image_path: str) -> Optional[str]:
        """Resolve image path, handling various Label Studio path formats"""
        # Handle Label Studio local storage paths
        if image_path.startswith("/data/local-files/"):
            image_path = image_path.replace("/data/local-files/?d=", "")

        # Handle URL paths
        if image_path.startswith("http://") or image_path.startswith("https://"):
            # For now, skip URLs - they need to be downloaded first
            return None

        # Handle absolute paths
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path

        # Try resolving relative to base directory
        if self.image_base_dir:
            full_path = self.image_base_dir / image_path
            if full_path.exists():
                return str(full_path)

        return image_path if os.path.exists(image_path) else None

    def _format_nemotron_output(
        self,
        regions: List[Dict[str, Any]],
        include_bboxes: bool = True,
    ) -> str:
        """
        Format regions into Nemotron Parse output format.

        Nemotron Parse outputs structured annotations with text, bounding boxes,
        and semantic classes in reading order.

        Args:
            regions: List of text regions with bbox, text, and label
            include_bboxes: Whether to include bounding box coordinates

        Returns:
            Formatted target text string
        """
        # Sort regions by reading order (top-to-bottom, left-to-right)
        sorted_regions = sorted(
            regions,
            key=lambda r: (r["bbox"][1], r["bbox"][0])  # Sort by y, then x
        )

        output_parts = []
        for region in sorted_regions:
            text = region.get("text", "").strip()
            if not text:
                continue

            if include_bboxes:
                bbox = region["bbox"]
                label = region.get("label", "text")
                # Format: <class>text</class> with bbox as metadata
                # Using Nemotron's structured output format
                output_parts.append(
                    f"<{label}><bbox>{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}</bbox>{text}</{label}>"
                )
            else:
                output_parts.append(text)

        return "\n".join(output_parts)

    def convert_from_jsonl(
        self,
        jsonl_path: str,
        image_key: str = "image",
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Convert JSONL format data to Nemotron training format.

        Args:
            jsonl_path: Path to JSONL file
            image_key: Key for image path in JSONL
            text_key: Key for target text in JSONL

        Returns:
            List of training samples
        """
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                image_path = data.get(image_key)
                target_text = data.get(text_key)

                if image_path and target_text:
                    resolved_path = self._resolve_image_path(image_path)
                    if resolved_path:
                        samples.append({
                            "image_path": resolved_path,
                            "target_text": target_text,
                        })

        return samples


class NemotronDataset(Dataset):
    """PyTorch Dataset for Nemotron Parse finetuning"""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        processor: Any,
        max_length: int = 4096,
        image_size: Tuple[int, int] = (1024, 1024),
    ):
        """
        Initialize the dataset.

        Args:
            samples: List of sample dictionaries with image_path and target_text
            processor: Nemotron processor/tokenizer for encoding
            max_length: Maximum sequence length for text
            image_size: Target image size (height, width)
        """
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load and preprocess image
        image = Image.open(sample["image_path"]).convert("RGB")

        # Get target text
        target_text = sample["target_text"]

        # Process using the model's processor
        # Nemotron Parse uses an encoder-decoder architecture
        encoding = self.processor(
            images=image,
            text=target_text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Add labels (same as decoder input for autoregressive training)
        if "input_ids" in item:
            item["labels"] = item["input_ids"].clone()

        return item


class NemotronDataCollator:
    """Data collator for Nemotron Parse training"""

    def __init__(
        self,
        processor: Any,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
    ):
        """
        Initialize the collator.

        Args:
            processor: Model processor/tokenizer
            padding: Padding strategy
            max_length: Maximum length for padding
            pad_to_multiple_of: Pad to multiple of this value
            label_pad_token_id: Token ID for padding labels (typically -100 to ignore in loss)
        """
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of features"""
        # Separate pixel values and text inputs
        pixel_values = []
        labels_list = []
        has_labels = "labels" in features[0]

        for feature in features:
            if "pixel_values" in feature:
                pixel_values.append(feature["pixel_values"])
            if has_labels:
                labels_list.append(feature["labels"])

        batch = {}

        # Stack pixel values
        if pixel_values:
            batch["pixel_values"] = torch.stack(pixel_values)

        # Pad text sequences
        text_features = [{k: v for k, v in f.items() if k not in ["pixel_values", "labels"]}
                        for f in features]

        if text_features and text_features[0]:
            # Find max length in batch
            max_len = max(f["input_ids"].size(0) for f in features if "input_ids" in f)
            if self.max_length:
                max_len = min(max_len, self.max_length)
            if self.pad_to_multiple_of:
                max_len = ((max_len + self.pad_to_multiple_of - 1)
                          // self.pad_to_multiple_of * self.pad_to_multiple_of)

            # Pad input_ids and attention_mask
            padded_input_ids = []
            padded_attention_mask = []

            pad_token_id = getattr(self.processor, "pad_token_id", 0)

            for feature in features:
                if "input_ids" in feature:
                    input_ids = feature["input_ids"]
                    attention_mask = feature.get("attention_mask",
                                                  torch.ones_like(input_ids))

                    padding_length = max_len - input_ids.size(0)
                    if padding_length > 0:
                        input_ids = torch.cat([
                            input_ids,
                            torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype)
                        ])
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.zeros(padding_length, dtype=attention_mask.dtype)
                        ])

                    padded_input_ids.append(input_ids[:max_len])
                    padded_attention_mask.append(attention_mask[:max_len])

            if padded_input_ids:
                batch["input_ids"] = torch.stack(padded_input_ids)
                batch["attention_mask"] = torch.stack(padded_attention_mask)

        # Pad labels
        if has_labels and labels_list:
            max_label_len = max(l.size(0) for l in labels_list)
            if self.max_length:
                max_label_len = min(max_label_len, self.max_length)

            padded_labels = []
            for labels in labels_list:
                padding_length = max_label_len - labels.size(0)
                if padding_length > 0:
                    labels = torch.cat([
                        labels,
                        torch.full((padding_length,), self.label_pad_token_id,
                                  dtype=labels.dtype)
                    ])
                padded_labels.append(labels[:max_label_len])

            batch["labels"] = torch.stack(padded_labels)

        return batch


def create_train_val_split(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split samples into training and validation sets.

    Args:
        samples: List of all samples
        val_ratio: Ratio of validation samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples)
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(samples))
    val_size = int(len(samples) * val_ratio)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    return train_samples, val_samples


def load_nemotron_dataset(
    data_path: str,
    processor: Any,
    image_base_dir: Optional[str] = None,
    val_ratio: float = 0.1,
    max_length: int = 4096,
    seed: int = 42,
) -> Tuple[NemotronDataset, NemotronDataset]:
    """
    Load and prepare Nemotron datasets from Label Studio export or JSONL.

    Args:
        data_path: Path to data file (JSON or JSONL)
        processor: Model processor
        image_base_dir: Base directory for images
        val_ratio: Validation split ratio
        max_length: Maximum sequence length
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    converter = NemotronDataConverter(image_base_dir=image_base_dir)

    # Detect format and load
    if data_path.endswith(".jsonl"):
        samples = converter.convert_from_jsonl(data_path)
    else:
        samples = converter.convert_from_label_studio(data_path)

    # Split
    train_samples, val_samples = create_train_val_split(
        samples, val_ratio=val_ratio, seed=seed
    )

    # Create datasets
    train_dataset = NemotronDataset(
        samples=train_samples,
        processor=processor,
        max_length=max_length,
    )
    val_dataset = NemotronDataset(
        samples=val_samples,
        processor=processor,
        max_length=max_length,
    )

    return train_dataset, val_dataset

