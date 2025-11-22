#!/bin/bash
# Deploy Label Studio to Azure Container Instances

set -e

# Default values
RESOURCE_GROUP=""
CONTAINER_NAME="label-studio"
IMAGE_NAME="heartexlabs/label-studio:latest"
LOCATION="eastus"
CPU=2
MEMORY=4
PORT=8080
STORAGE_ACCOUNT_NAME=""
STORAGE_CONTAINER_NAME="label-studio-data"

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
        --storage-account)
            STORAGE_ACCOUNT_NAME="$2"
            shift 2
            ;;
        --storage-container)
            STORAGE_CONTAINER_NAME="$2"
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

# Create storage account and container if provided
if [ -n "$STORAGE_ACCOUNT_NAME" ]; then
    echo "Creating Azure Storage account..."
    az storage account create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$STORAGE_ACCOUNT_NAME" \
        --location "$LOCATION" \
        --sku Standard_LRS \
        2>/dev/null || echo "Storage account may already exist"
    
    # Get storage account key
    STORAGE_KEY=$(az storage account keys list \
        --resource-group "$RESOURCE_GROUP" \
        --account-name "$STORAGE_ACCOUNT_NAME" \
        --query "[0].value" -o tsv)
    
    # Create container
    az storage container create \
        --name "$STORAGE_CONTAINER_NAME" \
        --account-name "$STORAGE_ACCOUNT_NAME" \
        --account-key "$STORAGE_KEY" \
        2>/dev/null || echo "Container may already exist"
fi

# Create container instance
echo "Creating Azure Container Instance..."
az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --image "$IMAGE_NAME" \
    --cpu "$CPU" \
    --memory "${MEMORY}Gi" \
    --ip-address Public \
    --ports $PORT \
    --environment-variables \
        LABEL_STUDIO_HOST="http://localhost:$PORT" \
        LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
    --command-line "label-studio start --host 0.0.0.0 --port $PORT"

# Get container IP
CONTAINER_IP=$(az container show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --query ipAddress.ip --output tsv)

echo ""
echo "=========================================="
echo "Label Studio deployed successfully!"
echo "=========================================="
echo "Container IP: $CONTAINER_IP"
echo "Access URL: http://${CONTAINER_IP}:${PORT}"
echo ""
echo "Note: Data is stored in the container and will be lost if the container is deleted."
echo "For persistent storage, use Azure Storage or Azure Files."
echo ""
echo "To view logs:"
echo "  az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --follow"
echo ""
echo "To stop the container:"
echo "  az container stop --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
echo "To delete the container:"
echo "  az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"

