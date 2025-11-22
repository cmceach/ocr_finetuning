#!/bin/bash
# Container entrypoint script

set -e

# Wait for model files if needed
if [ -n "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Waiting for model file: $MODEL_PATH"
    while [ ! -f "$MODEL_PATH" ]; do
        sleep 5
    done
fi

# Run the application
exec python -m src.serving.ocr_server

