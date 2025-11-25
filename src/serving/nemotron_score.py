"""Azure ML scoring script for Nemotron Parse inference"""

import os
import io
import json
import logging
import base64
from typing import Dict, Any, List, Optional

import torch
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global model objects
model = None
processor = None


def init():
    """Initialize the model for inference - called once when endpoint starts"""
    global model, processor

    logger.info("Initializing Nemotron Parse model...")

    # Import here to avoid issues during Azure ML environment setup
    from src.training.nemotron_model import (
        load_finetuned_nemotron,
        NEMOTRON_PARSE_MODEL_ID,
    )

    # Get model path from Azure ML
    model_path = os.environ.get("AZUREML_MODEL_DIR", "./trained_models/nemotron")
    adapter_path = os.path.join(model_path, "adapter") if os.path.exists(os.path.join(model_path, "adapter")) else None
    merged_path = os.path.join(model_path, "merged") if os.path.exists(os.path.join(model_path, "merged")) else None

    # Configuration from environment
    base_model_id = os.environ.get("NEMOTRON_MODEL_ID", NEMOTRON_PARSE_MODEL_ID)
    use_flash_attention = os.environ.get("USE_FLASH_ATTENTION", "true").lower() == "true"
    torch_dtype_str = os.environ.get("TORCH_DTYPE", "bfloat16")

    torch_dtype = torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float16

    logger.info(f"Model path: {model_path}")
    logger.info(f"Adapter path: {adapter_path}")
    logger.info(f"Merged path: {merged_path}")

    # Load model
    model, processor = load_finetuned_nemotron(
        base_model_id=base_model_id,
        adapter_path=adapter_path,
        merged_model_path=merged_path,
        torch_dtype=torch_dtype,
    )

    model.eval()
    logger.info("Model loaded successfully!")


def run(raw_data: str) -> str:
    """
    Run inference on input data.

    Args:
        raw_data: JSON string with request data

    Returns:
        JSON string with prediction results
    """
    global model, processor

    try:
        # Parse input
        data = json.loads(raw_data)

        # Handle different input formats
        if "image_base64" in data:
            # Base64 encoded image
            image_bytes = base64.b64decode(data["image_base64"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif "image_url" in data:
            # URL - download image
            import requests
            response = requests.get(data["image_url"], timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            return json.dumps({
                "error": "No image provided. Use 'image_base64' or 'image_url'"
            })

        # Get generation parameters
        max_new_tokens = data.get("max_new_tokens", 4096)
        temperature = data.get("temperature", 0.0)
        top_k = data.get("top_k", 1)

        # Run inference
        device = next(model.parameters()).device
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "top_k": top_k,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode output
        generated_text = processor.batch_decode(outputs, skip_special_tokens=False)[0]

        # Parse structured output
        regions = _parse_output(generated_text)

        return json.dumps({
            "raw_text": generated_text,
            "regions": regions,
            "model_version": os.environ.get("MODEL_VERSION", "1.0.0"),
        })

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return json.dumps({"error": str(e)})


def _parse_output(raw_text: str) -> List[Dict[str, Any]]:
    """Parse Nemotron output into structured regions"""
    import re

    regions = []
    pattern = r'<(\w+)><bbox>([\d.,]+)</bbox>(.*?)</\1>'
    matches = re.findall(pattern, raw_text, re.DOTALL)

    for match in matches:
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

    # Fallback for plain text
    if not regions:
        clean_text = re.sub(r'<[^>]+>', '', raw_text)
        for line in clean_text.strip().split("\n"):
            if line.strip():
                regions.append({"text": line.strip()})

    return regions

