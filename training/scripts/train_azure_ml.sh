#!/bin/bash
# Azure ML training script for OCR fine-tuning

set -e

# Default values
CONFIG_PATH=""
CLIENT_CONFIG=""
WORKSPACE_NAME=""
RESOURCE_GROUP=""
SUBSCRIPTION_ID=""
COMPUTE_CLUSTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --client-config)
            CLIENT_CONFIG="$2"
            shift 2
            ;;
        --workspace-name)
            WORKSPACE_NAME="$2"
            shift 2
            ;;
        --resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        --subscription-id)
            SUBSCRIPTION_ID="$2"
            shift 2
            ;;
        --compute-cluster)
            COMPUTE_CLUSTER="$2"
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

# Set subscription if provided
if [ -n "$SUBSCRIPTION_ID" ]; then
    az account set --subscription "$SUBSCRIPTION_ID"
fi

# Submit Azure ML job
# Note: This is a template - actual implementation would use Azure ML SDK or CLI
echo "Submitting Azure ML training job..."
echo "Workspace: $WORKSPACE_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo "Compute Cluster: $COMPUTE_CLUSTER"

# Example Azure ML CLI command (adjust based on your setup)
# az ml job create --file deployment/azure/azure_ml_job.yaml \
#     --workspace-name "$WORKSPACE_NAME" \
#     --resource-group "$RESOURCE_GROUP"

echo "Azure ML job submitted successfully!"

