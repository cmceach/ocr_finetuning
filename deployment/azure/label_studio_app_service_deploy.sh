#!/bin/bash
# Deploy Label Studio to Azure App Service

set -e

# Default values
RESOURCE_GROUP=""
APP_NAME="label-studio-app"
LOCATION="eastus"
APP_SERVICE_PLAN="label-studio-plan"
SKU="B1"  # Basic tier
IMAGE_NAME="heartexlabs/label-studio:latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        --app-name)
            APP_NAME="$2"
            shift 2
            ;;
        --location)
            LOCATION="$2"
            shift 2
            ;;
        --plan-name)
            APP_SERVICE_PLAN="$2"
            shift 2
            ;;
        --sku)
            SKU="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
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

# Create App Service Plan
echo "Creating App Service Plan..."
az appservice plan create \
    --name "$APP_SERVICE_PLAN" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --is-linux \
    --sku "$SKU" \
    2>/dev/null || echo "App Service Plan may already exist"

# Create Web App with Docker image
echo "Creating Web App..."
az webapp create \
    --resource-group "$RESOURCE_GROUP" \
    --plan "$APP_SERVICE_PLAN" \
    --name "$APP_NAME" \
    --deployment-container-image-name "$IMAGE_NAME" \
    2>/dev/null || echo "Web App may already exist"

# Configure app settings
echo "Configuring app settings..."
az webapp config appsettings set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --settings \
        WEBSITES_PORT=8080 \
        LABEL_STUDIO_HOST="https://${APP_NAME}.azurewebsites.net" \
        LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true

# Configure container settings
echo "Configuring container..."
az webapp config set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --always-on true \
    --linux-fx-version "DOCKER|${IMAGE_NAME}"

# Get app URL
APP_URL=$(az webapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --query defaultHostName --output tsv)

echo ""
echo "=========================================="
echo "Label Studio deployed successfully!"
echo "=========================================="
echo "App Name: $APP_NAME"
echo "Access URL: https://${APP_URL}"
echo ""
echo "Note: It may take a few minutes for the app to start."
echo ""
echo "To view logs:"
echo "  az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo ""
echo "To stop the app:"
echo "  az webapp stop --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo ""
echo "To delete the app:"
echo "  az webapp delete --resource-group $RESOURCE_GROUP --name $APP_NAME --yes"

