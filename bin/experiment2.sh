#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SRC_DIR="$PROJECT_ROOT/src"
CONFIG_DIR="$PROJECT_ROOT/configs"
PRETRAINED_MODELS_BASE_DIR="$PROJECT_ROOT/pretrained_models"
FINETUNED_MODELS_BASE_DIR="$PROJECT_ROOT/finetuned_models"
RESULTS_DIR="$PROJECT_ROOT/results"

# --- Parse Arguments ---
GPU_ID=0 # Default GPU ID
if [ "$#" -gt 0 ]; then
    GPU_ID="$1"
fi

echo "--- Starting Experiment 2: Self-Supervised vs. Fully Supervised Training (GPU: $GPU_ID) ---"

# --- Create necessary base directories ---
mkdir -p "$PRETRAINED_MODELS_BASE_DIR"
mkdir -p "$FINETUNED_MODELS_BASE_DIR"
mkdir -p "$RESULTS_DIR"

# --- Helper function to find the latest checkpoint ---
# Arguments: $1=model_dir, $2=prefix, $3=expected_epochs
get_latest_checkpoint_path() {
    local model_dir="$1"
    local prefix="$2"
    local expected_epochs="$3"
    local expected_path="$model_dir/${prefix}_epoch_${expected_epochs}.pth"

    if [ -f "$expected_path" ]; then
        echo "$expected_path"
        return 0
    fi

    local latest_checkpoint=$(find "$model_dir" -maxdepth 1 -type f -name "${prefix}*.pth" | \
                              sort -V | tail -n 1) # sort -V for natural sort (epoch_9 vs epoch_10)

    if [ -n "$latest_checkpoint" ]; then
        echo "$latest_checkpoint"
        return 0
    else
        return 1 # Not found
    fi
}


# --- Define all experiment variants to run ---
# Format: "config_filename:run_pretraining_flag:pretrain_epochs:finetune_epochs"
# run_pretraining_flag: "true" if pretrain.py should be called, "false" for fully supervised from scratch
EXPERIMENT_VARIANTS=(
    "experiment2_ss_emerge_seed_10_ssl.yaml:true:3288:100"
    "experiment2_ss_emerge_seed_50_ssl.yaml:true:3288:100"
    "experiment2_ss_emerge_seed_100_ssl.yaml:true:3288:100"
    "experiment2_ss_emerge_seed_100_fs.yaml:false:0:100" # Pretrain epochs 0 as N/A

    "experiment2_ss_emerge_seed_iv_10_ssl.yaml:true:3288:100"
    "experiment2_ss_emerge_seed_iv_50_ssl.yaml:true:3288:100"
    "experiment2_ss_emerge_seed_iv_100_ssl.yaml:true:3288:100"
    "experiment2_ss_emerge_seed_iv_100_fs.yaml:false:0:100"
)

# --- Loop through and run each variant ---
for variant in "${EXPERIMENT_VARIANTS[@]}"; do
    IFS=':' read -r config_file_name run_pretraining pretrain_epochs finetune_epochs <<< "$variant"
    
    # Read experiment_name from the config file itself
    EXP_CONFIG_PATH="$CONFIG_DIR/$config_file_name"
    EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$EXP_CONFIG_PATH'))['experiment_name'])")

    echo -e "\n=== Running Experiment Variant: $EXP_NAME ==="
    
    # Define experiment-specific output directories
    CURRENT_PRETRAINED_DIR="$PRETRAINED_MODELS_BASE_DIR/$EXP_NAME"
    CURRENT_FINETUNED_DIR="$FINETUNED_MODELS_BASE_DIR/$EXP_NAME"
    mkdir -p "$CURRENT_PRETRAINED_DIR" "$CURRENT_FINETUNED_DIR"

    PRETRAINED_ENCODER_PATH="NONE" # Default for finetune.py if not pretraining
    
    # Phase 1: Self-Supervised Pretraining (if enabled for this variant)
    if [ "$run_pretraining" = "true" ]; then
        echo "--- Phase 1: Pretraining ---"
        python "$SRC_DIR/pretrain.py" \
            --config "$EXP_CONFIG_PATH" \
            --gpu "$GPU_ID"
        
        PRETRAINED_ENCODER_PATH=$(get_latest_checkpoint_path "$CURRENT_PRETRAINED_DIR" "${EXP_NAME}_encoder" "$pretrain_epochs")
        if [ -z "$PRETRAINED_ENCODER_PATH" ]; then
            echo "Error: Pretrained model for $EXP_NAME not found. Exiting."
            exit 1
        fi
        echo "Using pretrained encoder: $PRETRAINED_ENCODER_PATH"
    else
        echo "--- Skipping Pretraining (Fully Supervised) ---"
    fi

    # Phase 2: Finetuning
    echo "--- Phase 2: Finetuning ---"
    python "$SRC_DIR/finetune.py" \
        --config "$EXP_CONFIG_PATH" \
        --pretrained_model_path "$PRETRAINED_ENCODER_PATH" \
        --gpu "$GPU_ID" \
        --save_dir "$CURRENT_FINETUNED_DIR" # Override config's save_dir to be specific

    FINETUNED_MODEL_PATH=$(get_latest_checkpoint_path "$CURRENT_FINETUNED_DIR" "${EXP_NAME}_finetuned" "$finetune_epochs")
    if [ -z "$FINETUNED_MODEL_PATH" ]; then
        echo "Error: Finetuned model for $EXP_NAME not found. Exiting."
        exit 1
    fi
    echo "Using finetuned model: $FINETUNED_MODEL_PATH"

    # Phase 3: Evaluation
    echo "--- Phase 3: Evaluation ---"
    python "$SRC_DIR/evaluate.py" \
        --config "$EXP_CONFIG_PATH" \
        --model_path "$FINETUNED_MODEL_PATH" \
        --gpu "$GPU_ID" | tee "$RESULTS_DIR/${EXP_NAME}_results.txt" # Capture output to results file

done

echo -e "\n--- Experiment 2: Self-Supervised vs. Fully Supervised Training Complete ---"
echo "Results are saved in the '$RESULTS_DIR' directory."