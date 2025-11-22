# OCR Fine-Tuning Pipeline Template

A flexible, production-ready repository template for OCR fine-tuning workflows supporting DocTR, Label Studio, and Azure Document Intelligence, deployable to Azure Container Instances.

## Features

- **Flexible Pipeline**: Toggle between detection-only, recognition-only, or full OCR pipeline
- **Label Studio Integration**: Import/export annotations, convert to COCO format
- **Azure Document Intelligence**: Baseline comparison and pre-annotation assistance
- **Multiple Training Backends**: Support for local GPU training and Azure ML compute
- **Azure Deployment**: Ready-to-deploy containerized inference server
- **MLOps Ready**: MLflow integration for experiment tracking

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (for training, optional for inference)
- Azure account (for Azure DI and deployment)
- Label Studio instance (for annotation)

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

#### 4. Evaluate Model

```python
from src.serving.model_loader import load_model_for_inference
from src.evaluation.evaluate_ocr import evaluate_from_file

model = load_model_for_inference(
    det_model_path="trained_models/detection_best.pth",
    reco_model_path="trained_models/recognition_best.pth"
)
metrics = evaluate_from_file(model, "data/evaluation_data/test.json")
```

#### 5. Deploy to Azure Container Instances

```bash
bash deployment/azure/deploy_container.sh \
    --resource-group your-resource-group \
    --container-name ocr-inference \
    --registry-name your-registry \
    --image-name ocr-inference:latest
```

## Project Structure

```
ocr_finetuning/
├── data/                    # Data directories
├── notebooks/               # Jupyter notebooks for analysis
├── src/                     # Source code
│   ├── data/               # Data loaders and converters
│   ├── preprocessing/      # Data preprocessing
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation utilities
│   ├── serving/           # Inference server
│   └── utils/             # Utilities
├── training/              # Training configs and scripts
├── serving/               # Serving Dockerfiles and configs
├── deployment/            # Deployment scripts
└── config/                # Configuration files
```

## Configuration

The template uses YAML-based configuration files:

- `config/default_config.yaml`: Default configuration
- `config/client_configs/`: Client-specific configurations

Key configuration options:
- `pipeline_mode`: `detection`, `recognition`, or `full`
- `training.backend`: `local` or `azure_ml`
- `detection.architecture`: Model architecture for detection
- `recognition.architecture`: Model architecture for recognition

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

### Local Training

```bash
bash training/scripts/train_local.sh \
    --config config/default_config.yaml \
    --mode full
```

### Azure ML Training

```bash
bash training/scripts/train_azure_ml.sh \
    --workspace-name your-workspace \
    --resource-group your-resource-group \
    --compute-cluster your-cluster
```

## Deployment

### Local Testing

```bash
cd serving
docker-compose up
```

### Azure Container Instances

```bash
bash deployment/azure/deploy_container.sh \
    --resource-group rg-ocr \
    --container-name ocr-api \
    --registry-name yourregistry \
    --cpu 2 \
    --memory 4
```

## API Usage

Once deployed, the inference API provides:

- `POST /predict`: Single document OCR
- `POST /predict/batch`: Batch processing
- `GET /health`: Health check
- `GET /ready`: Readiness check

Example:
```bash
curl -X POST http://your-container-ip:8000/predict \
    -F "file=@document.pdf"
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

See the [deployment scripts](deployment/azure/) for deploying trained OCR models to Azure Container Instances.

## Support

For issues and questions, please open an issue in the repository.

