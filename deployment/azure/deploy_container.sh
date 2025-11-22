#!/bin/bash
# Deploy OCR model to Azure Container Instances

set -e

# Default values
RESOURCE_GROUP=""
CONTAINER_NAME="ocr-inference"
IMAGE_NAME="ocr-inference:latest"
REGISTRY_NAME=""
LOCATION="eastus"
CPU=2
MEMORY=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --registry-name)
            REGISTRY_NAME="$2"
            shift 2
            ;;
        --location)
            LOCATION="$2"
            shift 2
            ;;
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check Azure CLI
if ! command -v az &> /dev/null; then
    echo "Azure CLI not found. Please install it first."
    exit 1
fi

# Login to Azure (if not already logged in)
az account show > /dev/null 2>&1 || az login

# Create resource group if it doesn't exist
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" 2>/dev/null || true

# Build and push image to Azure Container Registry if registry is provided
if [ -n "$REGISTRY_NAME" ]; then
    echo "Building and pushing image to Azure Container Registry..."
    az acr build --registry "$REGISTRY_NAME" --image "$IMAGE_NAME" --file serving/Dockerfile .
    FULL_IMAGE_NAME="${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}"
else
    FULL_IMAGE_NAME="$IMAGE_NAME"
fi

# Create container instance
echo "Creating Azure Container Instance..."
az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --image "$FULL_IMAGE_NAME" \
    --cpu "$CPU" \
    --memory "${MEMORY}Gi" \
    --registry-login-server "${REGISTRY_NAME}.azurecr.io" \
    --ip-address Public \
    --ports 8000 \
    --environment-variables \
        MODEL_VERSION=1.0.0 \
    --secure-environment-variables \
        AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="${AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT}" \
        AZURE_DOCUMENT_INTELLIGENCE_KEY="${AZURE_DOCUMENT_INTELLIGENCE_KEY}"

# Get container IP
CONTAINER_IP=$(az container show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --query ipAddress.ip --output tsv)

echo "Container deployed successfully!"
echo "Container IP: $CONTAINER_IP"
echo "API endpoint: http://${CONTAINER_IP}:8000"

