"""FastAPI server for Nemotron Parse inference"""

import os
import io
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import torch

from ..utils.config_loader import ConfigLoader
from ..training.nemotron_model import (
    NEMOTRON_PARSE_MODEL_ID,
    load_finetuned_nemotron,
    get_nemotron_processor,
)


class NemotronRequest(BaseModel):
    """Nemotron Parse request model"""

    image_url: Optional[str] = None
    max_new_tokens: int = Field(default=4096, ge=1, le=9000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=1, ge=1)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class TextRegion(BaseModel):
    """Extracted text region"""

    text: str
    bbox: Optional[List[float]] = None
    label: Optional[str] = None
    confidence: Optional[float] = None


class NemotronResponse(BaseModel):
    """Nemotron Parse response model"""

    raw_text: str
    regions: List[TextRegion]
    model_version: str
    processing_time_ms: float


class NemotronModelManager:
    """Manager for Nemotron Parse model loading and inference"""

    def __init__(
        self,
        model_id: str = NEMOTRON_PARSE_MODEL_ID,
        adapter_path: Optional[str] = None,
        merged_model_path: Optional[str] = None,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        use_vllm: bool = False,
    ):
        """
        Initialize model manager.

        Args:
            model_id: Base model ID
            adapter_path: Path to LoRA adapter
            merged_model_path: Path to merged model
            device: Device to use
            torch_dtype: Model dtype
            use_vllm: Whether to use vLLM for inference
        """
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.merged_model_path = merged_model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_vllm = use_vllm

        self.model = None
        self.processor = None
        self.vllm_model = None

    def load(self):
        """Load model and processor"""
        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self):
        """Load model using transformers"""
        self.model, self.processor = load_finetuned_nemotron(
            base_model_id=self.model_id,
            adapter_path=self.adapter_path,
            merged_model_path=self.merged_model_path,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )
        self.model.eval()

    def _load_vllm(self):
        """Load model using vLLM for efficient inference"""
        try:
            from vllm import LLM, SamplingParams

            model_path = self.merged_model_path or self.model_id
            self.vllm_model = LLM(
                model=model_path,
                max_num_seqs=64,
                limit_mm_per_prompt={"image": 1},
                dtype="bfloat16" if self.torch_dtype == torch.bfloat16 else "float16",
                trust_remote_code=True,
            )
            self.processor = get_nemotron_processor(model_path)

        except ImportError:
            raise ImportError(
                "vLLM is required for vLLM backend. "
                "Install with: pip install vllm"
            )

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_k: int = 1,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate text from image.

        Args:
            image: PIL Image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty

        Returns:
            Generated text
        """
        if self.use_vllm:
            return self._generate_vllm(
                image, max_new_tokens, temperature, top_k, repetition_penalty
            )
        else:
            return self._generate_transformers(
                image, max_new_tokens, temperature, top_k, repetition_penalty
            )

    def _generate_transformers(
        self,
        image: Image.Image,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
    ) -> str:
        """Generate using transformers"""
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode
        generated_text = self.processor.batch_decode(
            outputs, skip_special_tokens=False
        )[0]

        return generated_text

    def _generate_vllm(
        self,
        image: Image.Image,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
    ) -> str:
        """Generate using vLLM"""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            skip_special_tokens=False,
        )

        prompts = [{
            "prompt": "",
            "multi_modal_data": {"image": image},
        }]

        outputs = self.vllm_model.generate(prompts, sampling_params)
        return outputs[0].outputs[0].text


def parse_nemotron_output(raw_text: str) -> List[TextRegion]:
    """
    Parse Nemotron Parse output into structured regions.
    
    Supports both Nemotron Parse format (<x1><y1><x2><y2>text<class>)
    and legacy format (<label><bbox>x1,y1,x2,y2</bbox>text</label>).

    Args:
        raw_text: Raw output from model

    Returns:
        List of TextRegion objects
    """
    regions = []

    # Pattern for Nemotron Parse format: <x1><y1><x2><y2>text<class>
    nemotron_pattern = r'<([\d.]+)><([\d.]+)><([\d.]+)><([\d.]+)>(.*?)<(\w+)>'
    matches = re.findall(nemotron_pattern, raw_text, re.DOTALL)

    for match in matches:
        x1, y1, x2, y2, text, label = match
        try:
            bbox = [float(x1), float(y1), float(x2), float(y2)]
        except (ValueError, AttributeError):
            bbox = None

        regions.append(TextRegion(
            text=text.strip(),
            bbox=bbox,
            label=label,
        ))

    # If no Nemotron format found, try legacy format
    if not regions:
        legacy_pattern = r'<(\w+)><bbox>([\d.,]+)</bbox>(.*?)</\1>'
        legacy_matches = re.findall(legacy_pattern, raw_text, re.DOTALL)

        for match in legacy_matches:
            label, bbox_str, text = match
            try:
                bbox = [float(x.strip()) for x in bbox_str.split(",")]
            except (ValueError, AttributeError):
                bbox = None

            regions.append(TextRegion(
                text=text.strip(),
                bbox=bbox,
                label=label,
            ))

    # If still no structured output found, treat as plain text
    if not regions:
        # Try to extract text blocks separated by newlines
        lines = raw_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line:
                regions.append(TextRegion(text=line))

    return regions


# Global model manager
_model_manager: Optional[NemotronModelManager] = None


def get_model_manager() -> NemotronModelManager:
    """Get the global model manager"""
    global _model_manager
    if _model_manager is None:
        raise RuntimeError("Model not loaded. Call load_model first.")
    return _model_manager


def create_nemotron_app(
    model_id: str = NEMOTRON_PARSE_MODEL_ID,
    adapter_path: Optional[str] = None,
    merged_model_path: Optional[str] = None,
    config: Optional[ConfigLoader] = None,
    use_vllm: bool = False,
) -> FastAPI:
    """
    Create FastAPI application for Nemotron Parse inference.

    Args:
        model_id: Base model ID
        adapter_path: Path to LoRA adapter
        merged_model_path: Path to merged finetuned model
        config: Configuration loader instance
        use_vllm: Whether to use vLLM for inference

    Returns:
        FastAPI application instance
    """
    global _model_manager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for model loading"""
        global _model_manager

        # Load config overrides
        if config:
            model_id_config = config.get("nemotron.model_id", model_id)
        else:
            model_id_config = model_id

        # Initialize model manager
        _model_manager = NemotronModelManager(
            model_id=model_id_config,
            adapter_path=adapter_path,
            merged_model_path=merged_model_path,
            use_vllm=use_vllm,
        )

        print(f"Loading Nemotron Parse model: {model_id_config}")
        if adapter_path:
            print(f"  Adapter path: {adapter_path}")
        if merged_model_path:
            print(f"  Merged model path: {merged_model_path}")

        _model_manager.load()
        print("Model loaded successfully!")

        yield

        # Cleanup
        _model_manager = None

    app = FastAPI(
        title="Nemotron Parse Inference API",
        version="1.0.0",
        description="Document understanding and OCR using NVIDIA Nemotron Parse",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "nemotron-parse-inference"}

    @app.get("/ready")
    async def readiness_check():
        """Readiness check endpoint"""
        return {
            "status": "ready",
            "model_loaded": _model_manager is not None and _model_manager.model is not None,
        }

    @app.get("/model-info")
    async def model_info():
        """Get information about the loaded model"""
        manager = get_model_manager()
        return {
            "model_id": manager.model_id,
            "adapter_path": manager.adapter_path,
            "merged_model_path": manager.merged_model_path,
            "using_vllm": manager.use_vllm,
        }

    @app.post("/predict", response_model=NemotronResponse)
    async def predict(
        file: UploadFile = File(...),
        max_new_tokens: int = Query(default=4096, ge=1, le=9000),
        temperature: float = Query(default=0.0, ge=0.0, le=2.0),
        top_k: int = Query(default=1, ge=1),
        repetition_penalty: float = Query(default=1.1, ge=1.0, le=2.0),
    ):
        """
        Extract text from uploaded document image.

        Args:
            file: Uploaded image file
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty

        Returns:
            Extracted text with regions
        """
        import time

        start_time = time.time()

        try:
            # Read and validate file
            contents = await file.read()

            # Check file type
            valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
            file_ext = Path(file.filename).suffix.lower() if file.filename else ""

            if file_ext not in valid_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Supported: {', '.join(valid_extensions)}"
                )

            # Load image
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Get model manager
            manager = get_model_manager()

            # Generate
            raw_text = manager.generate(
                image=image,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            # Parse output
            regions = parse_nemotron_output(raw_text)

            processing_time = (time.time() - start_time) * 1000

            return NemotronResponse(
                raw_text=raw_text,
                regions=regions,
                model_version=os.getenv("MODEL_VERSION", "1.0.0"),
                processing_time_ms=processing_time,
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during inference: {str(e)}"
            )

    @app.post("/predict/batch")
    async def predict_batch(
        files: List[UploadFile] = File(...),
        max_new_tokens: int = Query(default=4096, ge=1, le=9000),
        temperature: float = Query(default=0.0, ge=0.0, le=2.0),
    ):
        """
        Extract text from multiple document images.

        Args:
            files: List of uploaded image files
            max_new_tokens: Maximum tokens per image
            temperature: Sampling temperature

        Returns:
            List of extraction results
        """
        results = []

        for file in files:
            try:
                response = await predict(
                    file=file,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                results.append({
                    "filename": file.filename,
                    "result": response.model_dump(),
                })
            except HTTPException as e:
                results.append({
                    "filename": file.filename,
                    "error": e.detail,
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                })

        return {"results": results}

    @app.post("/extract-text")
    async def extract_text_only(file: UploadFile = File(...)):
        """
        Simple endpoint to extract just the text from a document.

        Args:
            file: Uploaded image file

        Returns:
            Plain text extraction
        """
        response = await predict(file=file)
        # Concatenate all region texts
        full_text = "\n".join(r.text for r in response.regions if r.text)
        return {
            "text": full_text,
            "processing_time_ms": response.processing_time_ms,
        }

    return app


# For running directly
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Nemotron Parse Inference Server")
    parser.add_argument("--model-id", type=str, default=NEMOTRON_PARSE_MODEL_ID,
                        help="Base model ID")
    parser.add_argument("--adapter-path", type=str, help="Path to LoRA adapter")
    parser.add_argument("--merged-model-path", type=str, help="Path to merged model")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM backend")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    args = parser.parse_args()

    app = create_nemotron_app(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        merged_model_path=args.merged_model_path,
        use_vllm=args.use_vllm,
    )

    uvicorn.run(app, host=args.host, port=args.port)

