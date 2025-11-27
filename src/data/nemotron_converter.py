"""Data converter and dataset for Nemotron Parse finetuning"""

import json
import os
import re
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
        validate: bool = True,
        normalize_bboxes: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convert Label Studio export to Nemotron training format.

        Args:
            json_path: Path to Label Studio JSON export
            output_path: Optional path to save converted data
            validate: Whether to validate each conversion
            normalize_bboxes: Whether to normalize bbox coordinates to 0-1 range

        Returns:
            List of training samples in Nemotron format
        """
        loader = LabelStudioLoader(json_path)
        samples = []
        validation_results = []

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

            # Get image dimensions for bbox normalization
            image_width = None
            image_height = None
            if normalize_bboxes:
                try:
                    with Image.open(resolved_path) as img:
                        image_width, image_height = img.size
                except Exception:
                    # If we can't read image, check if coordinates are already normalized
                    # (all values <= 1.0)
                    if regions:
                        sample_bbox = regions[0].get("bbox", [])
                        if sample_bbox and all(coord <= 1.0 for coord in sample_bbox):
                            # Coordinates appear to already be normalized
                            image_width = None
                            image_height = None

            # Convert to Nemotron format (structured text output)
            target_text = self._format_nemotron_output(
                regions,
                image_width=image_width,
                image_height=image_height,
            )

            # Validate conversion if requested
            validation_result = None
            if validate:
                validation_result = self.validate_conversion(
                    regions,
                    target_text,
                    image_width=image_width,
                    image_height=image_height,
                )
                validation_results.append({
                    "task_id": task.get("id"),
                    "image_path": str(resolved_path),
                    "validation": validation_result,
                })

            samples.append({
                "image_path": str(resolved_path),
                "target_text": target_text,
                "regions": regions,
                "task_id": task.get("id"),
                "image_width": image_width,
                "image_height": image_height,
                "validation": validation_result,
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
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> str:
        """
        Format regions into Nemotron Parse output format.

        Nemotron Parse outputs structured annotations with text, bounding boxes,
        and semantic classes in reading order. The format uses:
        - Spatial tokens for coordinates: <x1><y1><x2><y2> (normalized 0-1)
        - Semantic class tokens: <class>
        - Text content in markdown format

        Format: <x1><y1><x2><y2>text_content<class>

        Args:
            regions: List of text regions with bbox, text, and label
            include_bboxes: Whether to include bounding box coordinates
            image_width: Image width for normalizing bbox coordinates (if not already normalized)
            image_height: Image height for normalizing bbox coordinates (if not already normalized)

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
                label = region.get("label") or "text"
                
                # Normalize bbox coordinates to 0-1 range if needed
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # If coordinates appear to be in pixel space, normalize them
                if image_width and image_height:
                    x1 = x1 / image_width
                    y1 = y1 / image_height
                    x2 = x2 / image_width
                    y2 = y2 / image_height
                elif any(coord > 1.0 for coord in [x1, y1, x2, y2]):
                    # Coordinates appear to be in pixel space but no image dimensions provided
                    # Keep as-is but warn (validation will catch this)
                    pass
                
                # Nemotron Parse format: <x1><y1><x2><y2>text<class>
                # Coordinates are formatted with 4 decimal places
                output_parts.append(
                    f"<{x1:.4f}><{y1:.4f}><{x2:.4f}><{y2:.4f}>{text}<{label}>"
                )
            else:
                output_parts.append(text)

        return "\n".join(output_parts)

    def _parse_nemotron_output(self, formatted_text: str) -> List[Dict[str, Any]]:
        """
        Parse Nemotron formatted output back to regions for validation.
        
        Nemotron Parse format: <x1><y1><x2><y2>text<class>
        Where x1,y1,x2,y2 are normalized coordinates (0-1 range)
        
        Args:
            formatted_text: Formatted text string from _format_nemotron_output
            
        Returns:
            List of parsed regions with text, bbox, and label
        """
        regions = []
        
        # Pattern for Nemotron Parse format: <x1><y1><x2><y2>text<class>
        # Coordinates are floating point numbers, class is a word
        pattern = r'<([\d.]+)><([\d.]+)><([\d.]+)><([\d.]+)>(.*?)<(\w+)>'
        matches = re.findall(pattern, formatted_text, re.DOTALL)
        
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
        
        # If no structured output found, try legacy format: <label><bbox>...</bbox>text</label>
        if not regions:
            legacy_pattern = r'<(\w+)><bbox>([\d.,]+)</bbox>(.*?)</\1>'
            legacy_matches = re.findall(legacy_pattern, formatted_text, re.DOTALL)
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
        
        # If still no structured output found, extract plain text lines
        if not regions:
            lines = formatted_text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line:
                    regions.append({"text": line, "bbox": None, "label": None})
        
        return regions

    def validate_conversion(
        self,
        original_regions: List[Dict[str, Any]],
        formatted_text: str,
        tolerance: float = 0.1,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate that Label Studio regions are correctly converted to Nemotron format.
        
        Args:
            original_regions: Original regions from Label Studio
            formatted_text: Formatted output text from _format_nemotron_output
            tolerance: Tolerance for bbox coordinate comparison (fraction of bbox size)
            image_width: Image width for normalizing original bbox coordinates
            image_height: Image height for normalizing original bbox coordinates
            
        Returns:
            Dictionary with validation results including:
            - is_valid: Whether conversion is valid
            - errors: List of error messages
            - warnings: List of warning messages
            - stats: Statistics about the conversion
            - examples: Example comparisons
        """
        errors = []
        warnings = []
        stats = {
            "original_regions": len(original_regions),
            "original_regions_with_text": 0,
            "parsed_regions": 0,
            "matched_regions": 0,
            "bbox_mismatches": 0,
            "text_mismatches": 0,
            "label_mismatches": 0,
            "missing_regions": 0,
            "extra_regions": 0,
        }
        
        # Count original regions with text
        original_with_text = [r for r in original_regions if r.get("text", "").strip()]
        stats["original_regions_with_text"] = len(original_with_text)
        
        # Parse formatted output back to regions
        parsed_regions = self._parse_nemotron_output(formatted_text)
        stats["parsed_regions"] = len(parsed_regions)
        
        # Check if we can parse the format correctly
        if not parsed_regions and formatted_text.strip():
            errors.append("Formatted text could not be parsed back to regions")
        
        # Helper to normalize bbox coordinates
        def normalize_bbox(bbox, width, height):
            if not bbox or not width or not height:
                return bbox
            return [
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height,
            ]
        
        # Normalize original regions if image dimensions provided
        original_normalized = []
        for r in original_with_text:
            bbox = r.get("bbox")
            if bbox and image_width and image_height:
                # Check if bbox appears to need normalization (coords > 1)
                if any(coord > 1.0 for coord in bbox):
                    normalized_bbox = normalize_bbox(bbox, image_width, image_height)
                else:
                    normalized_bbox = bbox
            else:
                normalized_bbox = bbox
            original_normalized.append({
                **r,
                "bbox": normalized_bbox,
                "original_bbox": bbox,
            })
        
        # Compare original regions with parsed regions
        # Sort both for comparison (by bbox position)
        original_sorted = sorted(
            original_normalized,
            key=lambda r: (r["bbox"][1] if r.get("bbox") else 0, r["bbox"][0] if r.get("bbox") else 0)
        )
        parsed_sorted = sorted(
            parsed_regions,
            key=lambda r: (r["bbox"][1] if r.get("bbox") else 0, r["bbox"][0] if r.get("bbox") else 0)
        )
        
        # Try to match regions by bbox proximity
        matched_indices = set()
        for i, orig_region in enumerate(original_sorted):
            orig_bbox = orig_region.get("bbox")
            orig_text = orig_region.get("text", "").strip()
            orig_label = orig_region.get("label") or "text"
            
            if not orig_bbox:
                warnings.append(f"Original region {i} has no bbox")
                continue
            
            # Find closest matching parsed region by bbox
            best_match_idx = None
            best_distance = float('inf')
            
            for j, parsed_region in enumerate(parsed_sorted):
                if j in matched_indices:
                    continue
                
                parsed_bbox = parsed_region.get("bbox")
                if not parsed_bbox:
                    continue
                
                # Calculate bbox center distance (both should be normalized now)
                orig_center = ((orig_bbox[0] + orig_bbox[2]) / 2, (orig_bbox[1] + orig_bbox[3]) / 2)
                parsed_center = ((parsed_bbox[0] + parsed_bbox[2]) / 2, (parsed_bbox[1] + parsed_bbox[3]) / 2)
                distance = ((orig_center[0] - parsed_center[0])**2 + (orig_center[1] - parsed_center[1])**2)**0.5
                
                # Calculate average bbox size for tolerance
                orig_width = orig_bbox[2] - orig_bbox[0]
                orig_height = orig_bbox[3] - orig_bbox[1]
                parsed_width = parsed_bbox[2] - parsed_bbox[0]
                parsed_height = parsed_bbox[3] - parsed_bbox[1]
                avg_size = ((orig_width + orig_height + parsed_width + parsed_height) / 4)
                
                # Check if bboxes are close enough (tolerance as fraction of average size)
                max_distance = tolerance * avg_size if avg_size > 0 else tolerance * 100
                if distance < max_distance:
                    if distance < best_distance:
                        best_match_idx = j
                        best_distance = distance
            
            if best_match_idx is not None:
                matched_indices.add(best_match_idx)
                parsed_region = parsed_sorted[best_match_idx]
                stats["matched_regions"] += 1
                
                # Compare bbox
                parsed_bbox = parsed_region.get("bbox")
                if parsed_bbox:
                    bbox_diff = [
                        abs(orig_bbox[k] - parsed_bbox[k]) for k in range(4)
                    ]
                    # Use relative tolerance based on bbox size
                    avg_size = sum([
                        orig_bbox[2] - orig_bbox[0],
                        orig_bbox[3] - orig_bbox[1],
                        parsed_bbox[2] - parsed_bbox[0],
                        parsed_bbox[3] - parsed_bbox[1]
                    ]) / 4
                    max_diff = tolerance * avg_size if avg_size > 0 else tolerance * 10
                    if any(d > max_diff for d in bbox_diff):
                        stats["bbox_mismatches"] += 1
                        warnings.append(
                            f"Region {i} bbox mismatch: "
                            f"original={orig_bbox}, parsed={parsed_bbox}, diff={bbox_diff}"
                        )
                
                # Compare text
                parsed_text = parsed_region.get("text", "").strip()
                if orig_text != parsed_text:
                    stats["text_mismatches"] += 1
                    warnings.append(
                        f"Region {i} text mismatch: "
                        f"original='{orig_text[:50]}...', parsed='{parsed_text[:50]}...'"
                    )
                
                # Compare label
                parsed_label = parsed_region.get("label", "text")
                if orig_label != parsed_label:
                    stats["label_mismatches"] += 1
                    warnings.append(
                        f"Region {i} label mismatch: "
                        f"original='{orig_label}', parsed='{parsed_label}'"
                    )
            else:
                stats["missing_regions"] += 1
                errors.append(
                    f"Original region {i} not found in parsed output: "
                    f"text='{orig_text[:50]}...', bbox={orig_bbox}"
                )
        
        # Check for extra regions in parsed output
        stats["extra_regions"] = len(parsed_sorted) - len(matched_indices)
        if stats["extra_regions"] > 0:
            warnings.append(
                f"Found {stats['extra_regions']} extra regions in parsed output "
                "that don't match original regions"
            )
        
        # Validate format structure
        format_errors = self._validate_format_structure(formatted_text)
        errors.extend(format_errors)
        
        # Determine overall validity
        is_valid = len(errors) == 0 and stats["matched_regions"] == stats["original_regions_with_text"]
        
        # Prepare examples
        examples = {
            "original_sample": original_sorted[:3] if original_sorted else [],
            "formatted_text_sample": formatted_text[:500] if formatted_text else "",
            "parsed_sample": parsed_sorted[:3] if parsed_sorted else [],
        }
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "examples": examples,
        }
    
    def _validate_format_structure(self, formatted_text: str) -> List[str]:
        """
        Validate the structure of formatted text matches expected Nemotron Parse pattern.
        
        Expected format: <x1><y1><x2><y2>text<class>
        Where coordinates are normalized (0-1 range)
        
        Args:
            formatted_text: Formatted text to validate
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not formatted_text.strip():
            return ["Formatted text is empty"]
        
        lines = formatted_text.strip().split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches Nemotron Parse format: <x1><y1><x2><y2>text<class>
            nemotron_pattern = r'^<([\d.]+)><([\d.]+)><([\d.]+)><([\d.]+)>(.*?)<(\w+)>$'
            match = re.match(nemotron_pattern, line, re.DOTALL)
            
            if not match:
                # Also check legacy format: <label><bbox>...</bbox>text</label>
                legacy_pattern = r'^<(\w+)><bbox>([\d.,]+)</bbox>(.*?)</\1>$'
                legacy_match = re.match(legacy_pattern, line, re.DOTALL)
                
                if legacy_match:
                    # Legacy format is valid but not preferred
                    continue
                
                # Check if it's plain text (which is also valid for no-bbox mode)
                if not re.match(r'^[^<>]+$', line):
                    errors.append(
                        f"Line {i} does not match expected Nemotron format: '{line[:100]}...'"
                    )
                continue
            
            x1, y1, x2, y2, text, label = match.groups()
            
            # Validate bbox coordinates
            try:
                coords = [float(x1), float(y1), float(x2), float(y2)]
                
                # Check if coordinates are in valid range (should be 0-1 for normalized)
                # Allow slightly out of bounds for edge cases
                for j, coord in enumerate(coords):
                    if coord < -0.1 or coord > 1.1:
                        errors.append(
                            f"Line {i} coordinate {j} ({coord}) appears to be non-normalized "
                            f"(expected 0-1 range). Consider providing image dimensions."
                        )
                
                # Check x2 > x1 and y2 > y1
                if coords[2] < coords[0]:
                    errors.append(f"Line {i}: x2 ({coords[2]}) should be >= x1 ({coords[0]})")
                if coords[3] < coords[1]:
                    errors.append(f"Line {i}: y2 ({coords[3]}) should be >= y1 ({coords[1]})")
                    
            except (ValueError, AttributeError) as e:
                errors.append(
                    f"Line {i} coordinate parsing error: '{x1},{y1},{x2},{y2}' - {str(e)}"
                )
            
            # Validate class label is not empty
            if not label:
                errors.append(f"Line {i}: empty class label")
        
        return errors

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


def validate_label_studio_conversion(
    samples: List[Dict[str, Any]],
    max_samples_to_check: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate Label Studio to Nemotron conversion for a batch of samples.
    
    Args:
        samples: List of converted samples (from convert_from_label_studio)
        max_samples_to_check: Maximum number of samples to validate (None = all)
        verbose: Whether to print detailed validation results
        
    Returns:
        Dictionary with aggregate validation results
    """
    if max_samples_to_check:
        samples_to_check = samples[:max_samples_to_check]
    else:
        samples_to_check = samples
    
    total_samples = len(samples_to_check)
    valid_samples = 0
    invalid_samples = 0
    all_errors = []
    all_warnings = []
    aggregate_stats = {
        "total_original_regions": 0,
        "total_parsed_regions": 0,
        "total_matched_regions": 0,
        "total_bbox_mismatches": 0,
        "total_text_mismatches": 0,
        "total_label_mismatches": 0,
        "total_missing_regions": 0,
        "total_extra_regions": 0,
    }
    
    for i, sample in enumerate(samples_to_check):
        validation = sample.get("validation")
        if not validation:
            # Re-validate if not already validated
            converter = NemotronDataConverter()
            regions = sample.get("regions", [])
            target_text = sample.get("target_text", "")
            validation = converter.validate_conversion(regions, target_text)
        
        if validation["is_valid"]:
            valid_samples += 1
        else:
            invalid_samples += 1
            all_errors.extend([
                f"Sample {i} (task_id={sample.get('task_id')}): {err}"
                for err in validation["errors"]
            ])
        
        all_warnings.extend([
            f"Sample {i} (task_id={sample.get('task_id')}): {warn}"
            for warn in validation["warnings"]
        ])
        
        # Aggregate stats
        stats = validation["stats"]
        for key in aggregate_stats:
            aggregate_stats[key] += stats.get(key, 0)
    
    # Calculate percentages
    validation_rate = valid_samples / total_samples if total_samples > 0 else 0
    
    result = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "validation_rate": validation_rate,
        "errors": all_errors[:20],  # Limit to first 20 errors
        "warnings": all_warnings[:50],  # Limit to first 50 warnings
        "aggregate_stats": aggregate_stats,
        "error_count": len(all_errors),
        "warning_count": len(all_warnings),
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Label Studio to Nemotron Conversion Validation")
        print("=" * 60)
        print(f"Total samples checked: {total_samples}")
        print(f"Valid samples: {valid_samples} ({validation_rate:.1%})")
        print(f"Invalid samples: {invalid_samples} ({1-validation_rate:.1%})")
        print(f"\nAggregate Statistics:")
        print(f"  Original regions: {aggregate_stats['total_original_regions']}")
        print(f"  Parsed regions: {aggregate_stats['total_parsed_regions']}")
        print(f"  Matched regions: {aggregate_stats['total_matched_regions']}")
        print(f"  Bbox mismatches: {aggregate_stats['total_bbox_mismatches']}")
        print(f"  Text mismatches: {aggregate_stats['total_text_mismatches']}")
        print(f"  Label mismatches: {aggregate_stats['total_label_mismatches']}")
        print(f"  Missing regions: {aggregate_stats['total_missing_regions']}")
        print(f"  Extra regions: {aggregate_stats['total_extra_regions']}")
        
        if all_errors:
            print(f"\nErrors ({len(all_errors)} total, showing first 20):")
            for err in all_errors[:20]:
                print(f"  - {err}")
        
        if all_warnings:
            print(f"\nWarnings ({len(all_warnings)} total, showing first 20):")
            for warn in all_warnings[:20]:
                print(f"  - {warn}")
        
        print("=" * 60)
    
    return result


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

