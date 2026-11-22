#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Project/fl_moe}"
SCRIPT_DIR="$PROJECT_ROOT/algorithms"
DATASET="cifar100"
NUM_CLIENTS=5
ROUNDS=100
LOCAL_STEPS=5
ALPHA=0.1
SEED=0
LR=0.01
OUTPUT_DIR="$PROJECT_ROOT/outputs/${DATASET}"
LAUNCH_LOG_DIR="$OUTPUT_DIR/launcher_logs"

EXPERIMENTS=(
    "uniform|0|uniform.py|"
    "rfedags|1|rfedags.py|"
)

mkdir -p "$LAUNCH_LOG_DIR"

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name gpu script extra_args <<< "$exp"
    if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
        echo "[ERROR] Missing: $SCRIPT_DIR/$script" >&2
        exit 1
    fi
done

launch_experiment() {
    local name="$1"
    local gpu="$2"
    local script="$3"
    local log_file="$LAUNCH_LOG_DIR/${name}.log"
    local pid_file="$LAUNCH_LOG_DIR/${name}.pid"
    local -a cmd=(
        env CUDA_VISIBLE_DEVICES="$gpu"
        python "$SCRIPT_DIR/$script"
        --dataset "$DATASET"
        --num-clients "$NUM_CLIENTS"
        --rounds "$ROUNDS"
        --local-steps "$LOCAL_STEPS"
        --alpha "$ALPHA"
        --seed "$SEED"
        --lr "$LR"
        --momentum 0
        --weight-decay 0
        --data-augmentation true
        --output-dir "$OUTPUT_DIR"
        --run-name "$name"
    )

    echo "[Launch] $name | GPU=$gpu | script=$script"
    nohup "${cmd[@]}" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    echo "         PID=$pid"
    echo "         log=$log_file"
}

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name gpu script extra_args <<< "$exp"
    launch_experiment "$name" "$gpu" "$script"
done

echo
echo "Dataset     : $DATASET"
echo "Clients     : $NUM_CLIENTS"
echo "Rounds      : $ROUNDS"
echo "Local Steps : $LOCAL_STEPS"
echo "Alpha       : $ALPHA"
echo "Seed        : $SEED"
echo "Output      : $OUTPUT_DIR"
echo "Logs        : $LAUNCH_LOG_DIR"
