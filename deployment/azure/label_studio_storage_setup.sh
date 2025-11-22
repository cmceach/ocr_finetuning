#!/bin/bash
# Set up Azure Storage for Label Studio persistent data

set -e

# Default values
RESOURCE_GROUP=""
STORAGE_ACCOUNT_NAME=""
LOCATION="eastus"
CONTAINER_NAME="label-studio-data"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        --storage-account)
            STORAGE_ACCOUNT_NAME="$2"
            shift 2
            ;;
        --location)
            LOCATION="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
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

# Create storage account
echo "Creating Azure Storage account..."
az storage account create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$STORAGE_ACCOUNT_NAME" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2

# Get storage account key
STORAGE_KEY=$(az storage account keys list \
    --resource-group "$RESOURCE_GROUP" \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --query "[0].value" -o tsv)

# Create container for Label Studio data
echo "Creating storage container..."
az storage container create \
    --name "$CONTAINER_NAME" \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --account-key "$STORAGE_KEY" \
    --public-access off

echo ""
echo "=========================================="
echo "Azure Storage setup completed!"
echo "=========================================="
echo "Storage Account: $STORAGE_ACCOUNT_NAME"
echo "Container: $CONTAINER_NAME"
echo ""
echo "To use this storage with Label Studio, configure:"
echo "  STORAGE_TYPE=azure"
echo "  STORAGE_AZURE_ACCOUNT_NAME=$STORAGE_ACCOUNT_NAME"
echo "  STORAGE_AZURE_ACCOUNT_KEY=$STORAGE_KEY"
echo "  STORAGE_AZURE_CONTAINER_NAME=$CONTAINER_NAME"
echo ""
echo "Note: Keep your storage account key secure!"

