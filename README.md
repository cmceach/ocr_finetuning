# OCR Fine-Tuning Pipeline Template

A flexible, production-ready repository template for OCR fine-tuning workflows supporting DocTR, NVIDIA Nemotron Parse v1.1, Label Studio, and Azure Document Intelligence, deployable to Azure Container Instances and Azure ML.

## Features

- **Multiple OCR Models**: Support for DocTR (detection + recognition) and NVIDIA Nemotron Parse v1.1 (VLM-based document understanding)
- **Flexible Pipeline**: Toggle between detection-only, recognition-only, full OCR pipeline, or Nemotron Parse
- **Parameter-Efficient Finetuning**: LoRA and QLoRA support for Nemotron Parse to reduce GPU memory requirements
- **Label Studio Integration**: Import/export annotations, convert to COCO format
- **Azure Document Intelligence**: Baseline comparison and pre-annotation assistance
- **Multiple Training Backends**: Support for local GPU training and Azure ML compute
- **Azure Deployment**: Ready-to-deploy containerized inference servers (CPU and GPU)
- **Auto-Scaling**: Production-ready auto-scaling configurations for Azure ML endpoints
- **MLOps Ready**: MLflow integration for experiment tracking

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (required for Nemotron Parse training/inference, optional for DocTR)
  - **Nemotron Parse**: Minimum 16GB VRAM (V100/T4), recommended 24GB+ (A100)
  - **DocTR**: 8GB+ VRAM sufficient
- Azure account (for Azure DI and deployment)
- Label Studio instance (for annotation)
- HuggingFace account (for accessing Nemotron Parse model)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ocr_finetuning
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Azure and Label Studio credentials
```

### Basic Usage

#### 1. Prepare Data for Label Studio

```python
from src.preprocessing.azure_di_preprocessor import AzureDIPreprocessor

preprocessor = AzureDIPreprocessor()
preannotations = preprocessor.preannotate_batch(
    document_paths=["doc1.pdf", "doc2.pdf"],
    image_urls=["http://labelstudio:8080/data/upload/doc1.pdf", ...]
)
preprocessor.save_preannotations(preannotations, "preannotations.json")
```

#### 2. Convert Label Studio Export to Training Format

```python
from src.data.doctr_converter import DocTRConverter
from src.data.label_studio_loader import LabelStudioLoader

loader = LabelStudioLoader("label_studio_export.json")
converter = DocTRConverter(loader)
converter.save_full_pipeline("./data/processed_data")
```

#### 3. Train Model

**DocTR Pipeline:**
```bash
# Local training
bash training/scripts/train_local.sh \
    --config config/default_config.yaml \
    --client-config config/client_configs/example_client.yaml \
    --mode full

# Or use Python directly
python -m src.training.train_full_pipeline \
    --config config/default_config.yaml \
    --client-config config/client_configs/example_client.yaml
```

**Nemotron Parse (VLM-based):**
```bash
# Train with LoRA (recommended - memory efficient)
python -m src.training.train_nemotron \
    --train-data data/labeled_data/train.json \
    --val-data data/labeled_data/val.json \
    --output-dir trained_models/nemotron \
    --learning-rate 2e-5 \
    --batch-size 4

# Train with QLoRA (4-bit quantization - for limited GPU memory)
python -m src.training.train_nemotron \
    --train-data data/labeled_data/train.json \
    --qlora \
    --batch-size 2

# Or use the pipeline
python -m src.training.train_full_pipeline \
    --nemotron \
    --train-data data/labeled_data/train.json
```

#### 4. Evaluate Model

**DocTR:**
```python
from src.serving.model_loader import load_model_for_inference
from src.evaluation.evaluate_ocr import evaluate_from_file

model = load_model_for_inference(
    det_model_path="trained_models/detection_best.pth",
    reco_model_path="trained_models/recognition_best.pth"
)
metrics = evaluate_from_file(model, "data/evaluation_data/test.json")
```

**Nemotron Parse:**
```python
from src.training.nemotron_model import load_finetuned_nemotron
from src.evaluation.evaluate_ocr import evaluate_nemotron_from_file

model, processor = load_finetuned_nemotron(
    adapter_path="trained_models/nemotron/final/adapter"
)
results = evaluate_nemotron_from_file(
    model, processor, "data/evaluation_data/test.json"
)
print(f"Character Accuracy: {results['metrics']['avg_char_accuracy']:.2%}")
print(f"Word Accuracy: {results['metrics']['avg_word_accuracy']:.2%}")
```

#### 5. Deploy to Azure

**DocTR (CPU):**
```bash
bash deployment/azure/deploy_container.sh \
    --resource-group your-resource-group \
    --container-name ocr-inference \
    --registry-name your-registry \
    --image-name ocr-inference:latest
```

**Nemotron Parse (GPU):**
```bash
# Deploy to Azure Container Instances with GPU
bash deployment/azure/deploy_nemotron_aci.sh \
    --resource-group your-resource-group \
    --registry-name your-registry \
    --gpu-sku V100 \
    --adapter-path /models/adapter  # Optional: path to finetuned adapter

# Or deploy to Azure ML Managed Endpoint (recommended for production)
az ml online-endpoint create --file deployment/azure/nemotron_ml_endpoint.yaml
az ml online-deployment create --file deployment/azure/nemotron_ml_deployment.yaml --all-traffic
```

## Project Structure

```
ocr_finetuning/
├── data/                    # Data directories
├── notebooks/               # Jupyter notebooks for analysis
├── src/                     # Source code
│   ├── data/               # Data loaders and converters
│   │   ├── nemotron_converter.py  # Nemotron dataset preparation
│   │   └── ...
│   ├── preprocessing/      # Data preprocessing
│   ├── training/          # Training scripts
│   │   ├── train_nemotron.py      # Nemotron training
│   │   ├── nemotron_model.py      # Model utilities
│   │   └── ...
│   ├── evaluation/        # Evaluation utilities
│   ├── serving/           # Inference servers
│   │   ├── nemotron_server.py     # Nemotron FastAPI server
│   │   ├── nemotron_score.py      # Azure ML scoring script
│   │   └── ...
│   └── utils/             # Utilities
├── training/              # Training configs and scripts
│   └── configs/
│       └── nemotron_config.yaml   # Nemotron training config
├── serving/               # Serving Dockerfiles and configs
├── deployment/            # Deployment scripts
│   └── azure/
│       ├── deploy_nemotron_aci.sh      # ACI GPU deployment
│       ├── nemotron_ml_job.yaml        # Azure ML training job
│       ├── nemotron_ml_endpoint.yaml   # ML endpoint config
│       ├── nemotron_ml_deployment.yaml # ML deployment config
│       ├── nemotron_dockerfile         # GPU Docker image
│       └── autoscale_profiles.yaml     # Auto-scaling profiles
└── config/                # Configuration files
```

## Configuration

The template uses YAML-based configuration files:

- `config/default_config.yaml`: Default configuration
- `config/client_configs/`: Client-specific configurations

Key configuration options:
- `pipeline_mode`: `detection`, `recognition`, `full`, or `nemotron`
- `training.backend`: `local` or `azure_ml`
- `detection.architecture`: Model architecture for detection
- `recognition.architecture`: Model architecture for recognition
- `nemotron.use_lora`: Enable LoRA for parameter-efficient finetuning
- `nemotron.use_qlora`: Enable 4-bit quantization (QLoRA) for memory efficiency
- `nemotron.learning_rate`: Learning rate (default: 2e-5)
- `nemotron.batch_size`: Batch size per device (default: 4)

See `training/configs/nemotron_config.yaml` for full Nemotron configuration options.

## Label Studio Integration

### Export Format

Label Studio exports should be in JSON format with the following structure:
- Image data in `data.image` or `data.ocr`
- Annotations with bounding boxes and text transcriptions

### Converting to COCO Format

```python
from src.data.coco_converter import COCOConverter
from src.data.label_studio_loader import LabelStudioLoader

loader = LabelStudioLoader("export.json")
converter = COCOConverter(loader)
converter.save("output_coco.json")
```

## Azure Document Intelligence

### Baseline Evaluation

Use Azure DI to establish a baseline for comparison:

```python
from src.evaluation.compare_models import compare_with_azure_di

comparison = compare_with_azure_di(model, test_data)
print(f"DocTR F1: {comparison['doctr']['f1_score']}")
```

### Pre-annotation

Use Azure DI to pre-annotate documents in Label Studio:

```python
from src.preprocessing.azure_di_preprocessor import AzureDIPreprocessor

preprocessor = AzureDIPreprocessor()
task = preprocessor.preannotate_document("document.pdf", "http://labelstudio:8080/data/upload/doc.pdf")
```

## Training

### DocTR Training

**Local Training:**
```bash
bash training/scripts/train_local.sh \
    --config config/default_config.yaml \
    --mode full
```

**Azure ML Training:**
```bash
bash training/scripts/train_azure_ml.sh \
    --workspace-name your-workspace \
    --resource-group your-resource-group \
    --compute-cluster your-cluster
```

### Nemotron Parse Training

**Local Training (LoRA - Recommended):**
```bash
python -m src.training.train_nemotron \
    --config training/configs/nemotron_config.yaml \
    --train-data data/labeled_data/train.json \
    --val-data data/labeled_data/val.json \
    --output-dir trained_models/nemotron
```

**Local Training (QLoRA - Memory Efficient):**
```bash
python -m src.training.train_nemotron \
    --train-data data/labeled_data/train.json \
    --qlora \
    --batch-size 2 \
    --learning-rate 2e-5
```

**Azure ML Training:**
```bash
az ml job create \
    --file deployment/azure/nemotron_ml_job.yaml \
    --resource-group your-resource-group \
    --workspace-name your-workspace
```

**Training Tips:**
- Use LoRA (`--use-lora`) for 10-20x memory reduction vs full finetuning
- Use QLoRA (`--qlora`) for 4-bit quantization - works on 16GB GPUs
- Recommended batch size: 4-8 with gradient accumulation
- Learning rate: 2e-5 works well for LoRA, adjust based on your data

## Deployment

### Local Testing

**DocTR Server:**
```bash
cd serving
docker-compose up
```

**Nemotron Parse Server:**
```bash
# Start with base model
python -m src.serving.nemotron_server \
    --host 0.0.0.0 \
    --port 8000

# Start with finetuned adapter
python -m src.serving.nemotron_server \
    --adapter-path trained_models/nemotron/final/adapter \
    --host 0.0.0.0 \
    --port 8000

# Use vLLM backend for faster inference (optional)
python -m src.serving.nemotron_server \
    --adapter-path trained_models/nemotron/final/adapter \
    --use-vllm
```

### Azure Container Instances

**DocTR (CPU):**
```bash
bash deployment/azure/deploy_container.sh \
    --resource-group rg-ocr \
    --container-name ocr-api \
    --registry-name yourregistry \
    --cpu 2 \
    --memory 4
```

**Nemotron Parse (GPU):**
```bash
bash deployment/azure/deploy_nemotron_aci.sh \
    --resource-group rg-ocr \
    --registry-name yourregistry \
    --gpu-sku V100 \
    --cpu 4 \
    --memory 16 \
    --adapter-path /models/adapter  # Optional
```

### Azure ML Managed Endpoints

**Nemotron Parse (Production):**
```bash
# Create endpoint
az ml online-endpoint create \
    --file deployment/azure/nemotron_ml_endpoint.yaml \
    --resource-group rg-ocr \
    --workspace-name ws-ocr

# Create deployment with auto-scaling
az ml online-deployment create \
    --file deployment/azure/nemotron_ml_deployment.yaml \
    --resource-group rg-ocr \
    --workspace-name ws-ocr \
    --all-traffic
```

See `deployment/azure/autoscale_profiles.yaml` for recommended auto-scaling configurations.

## API Usage

### DocTR API

Once deployed, the DocTR inference API provides:

- `POST /predict`: Single document OCR
- `POST /predict/batch`: Batch processing
- `GET /health`: Health check
- `GET /ready`: Readiness check

Example:
```bash
curl -X POST http://your-container-ip:8000/predict \
    -F "file=@document.pdf"
```

### Nemotron Parse API

**Endpoints:**
- `POST /predict`: Extract text from document image
- `POST /predict/batch`: Batch processing
- `POST /extract-text`: Simple text extraction (no structured output)
- `GET /health`: Health check
- `GET /ready`: Readiness check
- `GET /model-info`: Model information

**Request Examples:**

**File Upload:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@document.png" \
  -F "max_new_tokens=4096" \
  -F "temperature=0.0"
```

**Python Client:**
```python
import requests

with open("document.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"max_new_tokens": 4096, "temperature": 0.0}
    )
result = response.json()
for region in result["regions"]:
    print(f"{region['label']}: {region['text']}")
```

**Azure ML Endpoint:**
```python
import requests
import base64

# Encode image
with open("document.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Call endpoint
response = requests.post(
    "https://<endpoint>.inference.ml.azure.com/score",
    headers={"Authorization": "Bearer <api-key>"},
    json={
        "image_base64": image_base64,
        "max_new_tokens": 4096
    }
)
result = response.json()
```

**Response Format:**
```json
{
  "raw_text": "<text><bbox>10,20,200,50</bbox>Invoice #12345</text>...",
  "regions": [
    {
      "text": "Invoice #12345",
      "bbox": [10.0, 20.0, 200.0, 50.0],
      "label": "text"
    }
  ],
  "model_version": "1.0.0",
  "processing_time_ms": 3542.5
}
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
isort src/
```

## Contributing

This is a template repository. Customize it for your specific needs:

1. Update configuration files for your use case
2. Modify training scripts based on your data format
3. Adjust deployment settings for your infrastructure
4. Add client-specific configurations in `config/client_configs/`

## License

MIT License

## Deployment

### Label Studio Deployment

For deploying Label Studio on Azure, see:
- [Label Studio Azure Deployment Guide](deployment/azure/LABEL_STUDIO_AZURE_DEPLOYMENT.md)
- [Local Label Studio Setup](LABEL_STUDIO_SETUP.md)

Quick deployment options:
- **Azure Container Instances**: `./deployment/azure/label_studio_aci_deploy.sh`
- **Azure App Service**: `./deployment/azure/label_studio_app_service_deploy.sh`

### OCR Model Deployment

**DocTR Models:**
See the [deployment scripts](deployment/azure/) for deploying trained DocTR models to Azure Container Instances.

**Nemotron Parse Models:**
- **Azure Container Instances (GPU)**: `deployment/azure/deploy_nemotron_aci.sh`
- **Azure ML Managed Endpoint**: `deployment/azure/nemotron_ml_endpoint.yaml` and `nemotron_ml_deployment.yaml`
- **Auto-Scaling**: See `deployment/azure/autoscale_profiles.yaml` for production configurations

**GPU Requirements:**
- Minimum: V100 16GB or T4 16GB
- Recommended: A100 40GB+ for production workloads
- See auto-scale profiles for cost-optimized configurations

## Model Comparison

| Feature | DocTR | Nemotron Parse |
|---------|-------|----------------|
| **Architecture** | Detection + Recognition (2-stage) | Vision-Language Model (end-to-end) |
| **Model Size** | ~50-100M params | ~900M params |
| **GPU Memory** | 8GB+ | 16GB+ (LoRA), 24GB+ (full) |
| **Training** | Separate detection/recognition | Unified finetuning |
| **Output** | Bounding boxes + text | Structured text with bboxes |
| **Best For** | Traditional OCR, simple layouts | Complex documents, structured data |
| **Finetuning** | Full model | LoRA/QLoRA supported |

## Nemotron Parse Quick Reference

**Training:**
```bash
# LoRA finetuning (recommended)
python -m src.training.train_nemotron --train-data train.json --use-lora

# QLoRA (4-bit, memory efficient)
python -m src.training.train_nemotron --train-data train.json --qlora
```

**Serving:**
```bash
# Local server
python -m src.serving.nemotron_server --adapter-path ./adapter

# Azure ML endpoint
az ml online-deployment create --file deployment/azure/nemotron_ml_deployment.yaml
```

**Request Format:**
- File upload: `curl -F "file=@doc.png" http://localhost:8000/predict`
- Base64 (Azure ML): `{"image_base64": "...", "max_new_tokens": 4096}`

## Support

For issues and questions, please open an issue in the repository.

