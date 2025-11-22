#!/bin/bash
# Local training script for OCR fine-tuning

set -e

# Default values
CONFIG_PATH=""
CLIENT_CONFIG=""
TRAIN_DATA=""
VAL_DATA=""
OUTPUT_DIR="./trained_models"
MODE="full"

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
        --train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --val-data)
            VAL_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run training based on mode
case $MODE in
    detection)
        python -m src.training.train_detection \
            --config "$CONFIG_PATH" \
            --client-config "$CLIENT_CONFIG" \
            --train-data "$TRAIN_DATA" \
            --val-data "$VAL_DATA" \
            --output-dir "$OUTPUT_DIR"
        ;;
    recognition)
        python -m src.training.train_recognition \
            --config "$CONFIG_PATH" \
            --client-config "$CLIENT_CONFIG" \
            --train-data "$TRAIN_DATA" \
            --val-data "$VAL_DATA" \
            --output-dir "$OUTPUT_DIR"
        ;;
    full)
        python -m src.training.train_full_pipeline \
            --config "$CONFIG_PATH" \
            --client-config "$CLIENT_CONFIG"
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

echo "Training completed successfully!"

