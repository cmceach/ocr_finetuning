#!/bin/bash
# Start Label Studio locally

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocr_finetuning

# Set default port if not provided
PORT=${1:-8080}

# Create data directory if it doesn't exist
mkdir -p ./label_studio_data

echo "Starting Label Studio on port $PORT..."
echo "Access it at: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Label Studio
label-studio start \
    --port $PORT \
    --data-dir ./label_studio_data \
    --log-level INFO

