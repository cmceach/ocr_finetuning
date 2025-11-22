"""FastAPI server for OCR inference"""

import os
import io
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from doctr.io import DocumentFile
from .model_loader import load_model_for_inference
from ..utils.config_loader import ConfigLoader


class OCRRequest(BaseModel):
    """OCR request model"""

    image_url: Optional[str] = None
    batch: Optional[List[str]] = None


class OCRResponse(BaseModel):
    """OCR response model"""

    pages: List[Dict[str, Any]]
    model_version: str


def create_app(
    model_path: Optional[str] = None,
    det_model_path: Optional[str] = None,
    reco_model_path: Optional[str] = None,
    config: Optional[ConfigLoader] = None,
) -> FastAPI:
    """
    Create FastAPI application for OCR inference.

    Args:
        model_path: Path to full pipeline model
        det_model_path: Path to detection model weights
        reco_model_path: Path to recognition model weights
        config: Configuration loader instance

    Returns:
        FastAPI application instance
    """
    app = FastAPI(title="OCR Inference API", version="1.0.0")

    # Load model
    predictor = load_model_for_inference(
        model_path=model_path,
        det_model_path=det_model_path,
        reco_model_path=reco_model_path,
        config=config,
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "ocr-inference"}

    @app.get("/ready")
    async def readiness_check():
        """Readiness check endpoint"""
        return {"status": "ready", "model_loaded": predictor is not None}

    @app.post("/predict", response_model=OCRResponse)
    async def predict(file: UploadFile = File(...)):
        """
        Predict OCR results from uploaded image or PDF.

        Args:
            file: Uploaded file (image or PDF)

        Returns:
            OCR results in DocTR format
        """
        try:
            # Read file content
            contents = await file.read()

            # Load document
            if file.filename.endswith(".pdf"):
                doc = DocumentFile.from_pdf(io.BytesIO(contents))
            else:
                doc = DocumentFile.from_images(io.BytesIO(contents))

            # Run inference
            result = predictor(doc)

            # Convert to JSON-serializable format
            pages = []
            for page in result.pages:
                page_dict = {
                    "page_idx": page.page_idx,
                    "dimensions": page.dimensions,
                    "blocks": [],
                }

                for block in page.blocks:
                    block_dict = {
                        "geometry": block.geometry,
                        "lines": [],
                    }

                    for line in block.lines:
                        line_dict = {
                            "geometry": line.geometry,
                            "words": [],
                        }

                        for word in line.words:
                            word_dict = {
                                "value": word.value,
                                "confidence": word.confidence,
                                "geometry": word.geometry,
                            }
                            line_dict["words"].append(word_dict)

                        block_dict["lines"].append(line_dict)

                    page_dict["blocks"].append(block_dict)

                pages.append(page_dict)

            return OCRResponse(
                pages=pages,
                model_version=os.getenv("MODEL_VERSION", "1.0.0"),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    @app.post("/predict/batch")
    async def predict_batch(files: List[UploadFile] = File(...)):
        """
        Predict OCR results from multiple uploaded files.

        Args:
            files: List of uploaded files

        Returns:
            List of OCR results
        """
        results = []
        for file in files:
            try:
                result = await predict(file)
                results.append({"filename": file.filename, "result": result.dict()})
            except Exception as e:
                results.append({"filename": file.filename, "error": str(e)})

        return {"results": results}

    return app


# For running directly
if __name__ == "__main__":
    import uvicorn

    config = ConfigLoader()
    app = create_app(config=config)

    uvicorn.run(app, host="0.0.0.0", port=8000)

