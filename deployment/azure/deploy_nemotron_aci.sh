#!/bin/bash
# Deploy Nemotron Parse model to Azure Container Instances with GPU
#
# Requirements:
# - Azure CLI installed and logged in
# - Azure Container Registry (ACR) set up
# - GPU quota available in your subscription (V100 or better recommended)
#
# Usage:
#   ./deploy_nemotron_aci.sh \
#       --resource-group mygroup \
#       --registry-name myregistry \
#       --adapter-path /path/to/adapter  # Optional: path to LoRA adapter

set -e

# Default values
RESOURCE_GROUP=""
CONTAINER_NAME="nemotron-parse-inference"
IMAGE_NAME="nemotron-parse:latest"
REGISTRY_NAME=""
LOCATION="eastus"
GPU_COUNT=1
GPU_SKU="V100"  # Options: K80, P100, V100, T4
CPU=4
MEMORY=16
ADAPTER_PATH=""
USE_VLLM="false"

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
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --gpu-sku)
            GPU_SKU="$2"
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
        --adapter-path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --use-vllm)
            USE_VLLM="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --resource-group    Azure resource group name (required)"
            echo "  --registry-name     Azure Container Registry name (required)"
            echo "  --container-name    Container instance name (default: nemotron-parse-inference)"
            echo "  --image-name        Docker image name (default: nemotron-parse:latest)"
            echo "  --location          Azure region (default: eastus)"
            echo "  --gpu-count         Number of GPUs (default: 1)"
            echo "  --gpu-sku           GPU SKU: K80, P100, V100, T4 (default: V100)"
            echo "  --cpu               CPU cores (default: 4)"
            echo "  --memory            Memory in GB (default: 16)"
            echo "  --adapter-path      Path to LoRA adapter in blob storage"
            echo "  --use-vllm          Use vLLM backend for inference"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$RESOURCE_GROUP" ]; then
    echo "Error: --resource-group is required"
    exit 1
fi

if [ -z "$REGISTRY_NAME" ]; then
    echo "Error: --registry-name is required"
    exit 1
fi

# Check Azure CLI
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI not found. Please install it first."
    echo "Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Login to Azure (if not already logged in)
echo "Checking Azure login status..."
az account show > /dev/null 2>&1 || az login

# Create resource group if it doesn't exist
echo "Creating resource group (if needed)..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" 2>/dev/null || true

# Build and push image to Azure Container Registry
echo "Building and pushing image to Azure Container Registry..."
echo "This may take several minutes..."

az acr build \
    --registry "$REGISTRY_NAME" \
    --image "$IMAGE_NAME" \
    --file deployment/azure/nemotron_dockerfile \
    --platform linux/amd64 \
    .

FULL_IMAGE_NAME="${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}"

# Get ACR credentials
echo "Getting ACR credentials..."
ACR_USERNAME=$(az acr credential show --name "$REGISTRY_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$REGISTRY_NAME" --query passwords[0].value -o tsv)

# Build environment variables
ENV_VARS="MODEL_VERSION=1.0.0"
if [ "$USE_VLLM" = "true" ]; then
    ENV_VARS="$ENV_VARS USE_VLLM=true"
fi

# Build command with optional adapter path
CMD_ARGS="--host 0.0.0.0 --port 8000"
if [ -n "$ADAPTER_PATH" ]; then
    CMD_ARGS="$CMD_ARGS --adapter-path $ADAPTER_PATH"
fi
if [ "$USE_VLLM" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --use-vllm"
fi

# Create container instance with GPU
echo "Creating Azure Container Instance with GPU..."
az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --image "$FULL_IMAGE_NAME" \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --gpu-count "$GPU_COUNT" \
    --gpu-sku "$GPU_SKU" \
    --registry-login-server "${REGISTRY_NAME}.azurecr.io" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --ip-address Public \
    --ports 8000 \
    --environment-variables $ENV_VARS \
    --command-line "python -m src.serving.nemotron_server $CMD_ARGS" \
    --restart-policy OnFailure

# Wait for container to be running
echo "Waiting for container to start..."
az container wait \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --created

# Get container details
CONTAINER_IP=$(az container show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --query ipAddress.ip --output tsv)

CONTAINER_STATE=$(az container show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --query containers[0].instanceView.currentState.state --output tsv)

echo ""
echo "=========================================="
echo "Nemotron Parse Deployment Complete!"
echo "=========================================="
echo ""
echo "Container Name: $CONTAINER_NAME"
echo "Container IP: $CONTAINER_IP"
echo "Container State: $CONTAINER_STATE"
echo ""
echo "API Endpoints:"
echo "  Health:    http://${CONTAINER_IP}:8000/health"
echo "  Predict:   http://${CONTAINER_IP}:8000/predict"
echo "  Docs:      http://${CONTAINER_IP}:8000/docs"
echo ""
echo "Test with:"
echo "  curl -X POST http://${CONTAINER_IP}:8000/predict -F 'file=@document.png'"
echo ""
echo "View logs with:"
echo "  az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""

