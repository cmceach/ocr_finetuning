# Running Label Studio Locally

## Quick Start

### Option 1: Using the provided script

```bash
# Start Label Studio on default port 8080
./scripts/start_label_studio.sh

# Or specify a custom port
./scripts/start_label_studio.sh 9000
```

### Option 2: Manual start

```bash
# Activate conda environment
conda activate ocr_finetuning

# Start Label Studio
label-studio start --port 8080 --data-dir ./label_studio_data
```

## First Time Setup

1. **Start Label Studio** using one of the methods above
2. **Open your browser** and navigate to `http://localhost:8080`
3. **Create an account** - You'll be prompted to create a user account on first launch
4. **Create a project** - Click "Create Project" and configure your labeling interface

## Configuration for OCR Projects

### Recommended Labeling Interface for OCR

When creating a new project, use this labeling configuration:

```xml
<View>
  <Image name="image" value="$image"/>
  
  <Labels name="label" toName="image">
    <Label value="Text" background="green"/>
    <Label value="Handwriting" background="blue"/>
  </Labels>
  
  <Rectangle name="bbox" toName="image" strokeWidth="3"/>
  
  <TextArea name="transcription" toName="image"
            editable="true"
            perRegion="true"
            required="true"
            maxSubmissions="1"
            rows="5"
            placeholder="Recognized Text"
            displayMode="region-list"
            />
</View>
```

## Importing Data

### From your OCR pipeline:

1. **Prepare pre-annotations** (optional):
   ```python
   from src.preprocessing.azure_di_preprocessor import AzureDIPreprocessor
   
   preprocessor = AzureDIPreprocessor()
   preannotations = preprocessor.preannotate_batch(
       document_paths=["doc1.pdf", "doc2.pdf"],
       image_urls=["http://localhost:8080/data/upload/doc1.pdf", ...]
   )
   preprocessor.save_preannotations(preannotations, "preannotations.json")
   ```

2. **Import into Label Studio**:
   - Go to your project in Label Studio
   - Click "Import" button
   - Upload your JSON file with pre-annotations (or raw images)

## Exporting Annotations

### Export from Label Studio UI:

1. Go to your project
2. Click "Export" button
3. Select "JSON" format
4. Download the export file

### Export using Python SDK:

```python
from label_studio_sdk import Client

# Connect to Label Studio
ls = Client(url='http://localhost:8080', api_key='YOUR_API_KEY')

# Get project
project = ls.get_project(1)  # Replace with your project ID

# Export annotations
export = project.export_tasks(format='JSON')
```

## API Key

To get your API key:
1. Log into Label Studio
2. Go to Account & Settings (click your username in top right)
3. Navigate to "Access Token" section
4. Copy your API key

## Data Storage

- **Default location**: `./label_studio_data/` (created in project root)
- **Database**: SQLite (default) - stored in `label_studio_data/`
- **Uploaded files**: Stored in `label_studio_data/media/upload/`

## Common Commands

```bash
# Start with custom port
label-studio start --port 9000

# Start with PostgreSQL (requires setup)
label-studio start --database postgresql://user:pass@localhost/dbname

# Start with custom host
label-studio start --host 0.0.0.0 --port 8080

# Reset database (WARNING: deletes all data)
label-studio reset-db

# Show help
label-studio --help
```

## Troubleshooting

### Port already in use
```bash
# Find process using port 8080
lsof -i :8080

# Kill the process or use a different port
label-studio start --port 9000
```

### Permission errors
```bash
# Make sure label_studio_data directory is writable
chmod -R 755 ./label_studio_data
```

### Import errors
- Ensure your JSON format matches Label Studio's expected format
- Check that image URLs are accessible from Label Studio
- For local files, use absolute paths or serve them via a web server

## Integration with OCR Pipeline

See the notebooks:
- `notebooks/03_label_studio_prep.ipynb` - Prepare data for Label Studio
- `notebooks/01_data_analysis.ipynb` - Analyze exported annotations

