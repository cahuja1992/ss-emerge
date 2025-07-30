#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Define paths relative to the script's location
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd) # Go up one level from scripts/
SRC_DIR="$PROJECT_ROOT/src"
CONFIG_DIR="$PROJECT_ROOT/configs"
PRETRAINED_MODELS_BASE_DIR="$PROJECT_ROOT/pretrained_models"
FINETUNED_MODELS_BASE_DIR="$PROJECT_ROOT/finetuned_models"
RESULTS_DIR="$PROJECT_ROOT/results"

# --- Experiment-specific parameters (from thesis table) ---
# SS-EMERGE
SS_EMERGE_CONFIG="experiment1_ss_emerge_seed.yaml"
SS_EMERGE_PRETRAIN_EPOCHS=3288
SS_EMERGE_FINETUNE_EPOCHS=100

# ResNet Baseline
RESNET_CONFIG="experiment1_resnet_seed.yaml"
RESNET_FINETUNE_EPOCHS=100 # ResNet is fully supervised here, no pretrain phase

# --- Parse Arguments ---
GPU_ID=0 # Default GPU ID
if [ "$#" -gt 0 ]; then
    GPU_ID="$1"
fi

echo "--- Starting Experiment 1: Baseline Model Comparison (GPU: $GPU_ID) ---"

# --- Create necessary directories ---
mkdir -p "$PRETRAINED_MODELS_BASE_DIR"
mkdir -p "$FINETUNED_MODELS_BASE_DIR"
mkdir -p "$RESULTS_DIR"

# --- Helper function to find the latest checkpoint ---
# Arguments: $1=model_dir, $2=prefix, $3=expected_epochs (for exact match, fallback to latest)
get_latest_checkpoint_path() {
    local model_dir="$1"
    local prefix="$2"
    local expected_epochs="$3"
    local expected_path="$model_dir/${prefix}_epoch_${expected_epochs}.pth"

    if [ -f "$expected_path" ]; then
        echo "$expected_path"
        return 0
    fi

    # Fallback: find the one with the highest epoch number
    local latest_checkpoint=$(find "$model_dir" -maxdepth 1 -type f -name "${prefix}*.pth" | \
                              sort -V | tail -n 1) # sort -V for natural sort (epoch_9 vs epoch_10)

    if [ -n "$latest_checkpoint" ]; then
        echo "$latest_checkpoint"
        return 0
    else
        return 1 # Not found
    fi
}


# --- Run SS-EMERGE Variant ---
echo -e "\n=== Running SS-EMERGE Variant ==="
SS_EMERGE_EXP_NAME="Experiment1_SS_EMERGE_SEED"
SS_EMERGE_PRETRAINED_DIR="$PRETRAINED_MODELS_BASE_DIR/$SS_EMERGE_EXP_NAME"
SS_EMERGE_FINETUNED_DIR="$FINETUNED_MODELS_BASE_DIR/$SS_EMERGE_EXP_NAME"
mkdir -p "$SS_EMERGE_PRETRAINED_DIR" "$SS_EMERGE_FINETUNED_DIR"

# Phase 1: Pretraining
echo "--- Phase 1: Pretraining SS-EMERGE ---"
python "$SRC_DIR/pretrain.py" \
    --config "$CONFIG_DIR/$SS_EMERGE_CONFIG" \
    --gpu "$GPU_ID"

SS_EMERGE_PRETRAINED_MODEL_PATH=$(get_latest_checkpoint_path "$SS_EMERGE_PRETRAINED_DIR" "${SS_EMERGE_EXP_NAME}_encoder" "$SS_EMERGE_PRETRAIN_EPOCHS")
if [ -z "$SS_EMERGE_PRETRAINED_MODEL_PATH" ]; then
    echo "Error: SS-EMERGE pretrained model not found. Exiting."
    exit 1
fi
echo "Using SS-EMERGE pretrained model: $SS_EMERGE_PRETRAINED_MODEL_PATH"

# Phase 2: Finetuning
echo "--- Phase 2: Finetuning SS-EMERGE ---"
python "$SRC_DIR/finetune.py" \
    --config "$CONFIG_DIR/$SS_EMERGE_CONFIG" \
    --pretrained_model_path "$SS_EMERGE_PRETRAINED_MODEL_PATH" \
    --gpu "$GPU_ID" \
    --save_dir "$SS_EMERGE_FINETUNED_DIR"

SS_EMERGE_FINETUNED_MODEL_PATH=$(get_latest_checkpoint_path "$SS_EMERGE_FINETUNED_DIR" "${SS_EMERGE_EXP_NAME}_finetuned" "$SS_EMERGE_FINETUNE_EPOCHS")
if [ -z "$SS_EMERGE_FINETUNED_MODEL_PATH" ]; then
    echo "Error: SS-EMERGE finetuned model not found. Exiting."
    exit 1
fi
echo "Using SS-EMERGE finetuned model: $SS_EMERGE_FINETUNED_MODEL_PATH"

# Phase 3: Evaluation
echo "--- Phase 3: Evaluating SS-EMERGE ---"
python "$SRC_DIR/evaluate.py" \
    --config "$CONFIG_DIR/$SS_EMERGE_CONFIG" \
    --model_path "$SS_EMERGE_FINETUNED_MODEL_PATH" \
    --gpu "$GPU_ID" | tee "$RESULTS_DIR/${SS_EMERGE_EXP_NAME}_results.txt" # Capture output to results file


# --- Run ResNet Baseline Variant ---
echo -e "\n=== Running ResNet Baseline Variant ==="
RESNET_EXP_NAME="Experiment1_ResNet_SEED"
RESNET_FINETUNED_DIR="$FINETUNED_MODELS_BASE_DIR/$RESNET_EXP_NAME"
mkdir -p "$RESNET_FINETUNED_DIR"

# ResNet is fully supervised, so no separate pretraining phase.
# It directly goes to finetuning (training from scratch).
echo "--- Phase 1: Training ResNet (from scratch) ---"
python "$SRC_DIR/finetune.py" \
    --config "$CONFIG_DIR/$RESNET_CONFIG" \
    --pretrained_model_path "NONE" \
    --gpu "$GPU_ID" \
    --save_dir "$RESNET_FINETUNED_DIR"

RESNET_FINETUNED_MODEL_PATH=$(get_latest_checkpoint_path "$RESNET_FINETUNED_DIR" "${RESNET_EXP_NAME}_finetuned" "$RESNET_FINETUNE_EPOCHS")
if [ -z "$RESNET_FINETUNED_MODEL_PATH" ]; then
    echo "Error: ResNet finetuned model not found. Exiting."
    exit 1
fi
echo "Using ResNet finetuned model: $RESNET_FINETUNED_MODEL_PATH"

# Phase 2: Evaluation
echo "--- Phase 2: Evaluating ResNet ---"
python "$SRC_DIR/evaluate.py" \
    --config "$CONFIG_DIR/$RESNET_CONFIG" \
    --model_path "$RESNET_FINETUNED_MODEL_PATH" \
    --gpu "$GPU_ID" | tee "$RESULTS_DIR/${RESNET_EXP_NAME}_results.txt" # Capture output to results file


echo -e "\n--- Experiment 1: Baseline Model Comparison Complete ---"
echo "Results are saved in the '$RESULTS_DIR' directory."