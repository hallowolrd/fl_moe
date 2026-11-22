
#!/usr/bin/env bash
set -u

# ============================================================
# 联邦学习多算法并行启动脚本
# 数据集 / Architecture / 轮数 / 对比算法 / GPU 均在顶部配置
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Project/fl_moe}"
SCRIPT_DIR="$PROJECT_ROOT/algorithms"

# ------------------------------------------------------------
# 实验公共配置
# ------------------------------------------------------------
DATASET="cifar10"
DIRICHLET_SAMPLE_BALANCE="balanced"


LR=0.01
MOMENTUM=0
WEIGHT_DECAY=0

ROUNDS=50
ALPHA=0.1
SEED=0

NUM_CLIENTS=50

OUTPUT_DIR="$PROJECT_ROOT/outputs_clients/${NUM_CLIENTS}"
LAUNCH_LOG_DIR="$OUTPUT_DIR/launcher_logs"

# ------------------------------------------------------------
# 对比算法配置
#
# 格式：
#   "名称|GPU|Python脚本|算法专属参数"
#
# 以后增删算法、修改 GPU 或算法专属参数，只需要改这里。
# ------------------------------------------------------------
EXPERIMENTS=(
    "uniform|3|uniform.py|"
    "rfedags|3|rfedags.py|"
)

mkdir -p "$LAUNCH_LOG_DIR"

# ------------------------------------------------------------
# 检查算法脚本是否存在
# ------------------------------------------------------------
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name gpu script extra_args <<< "$exp"

    if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
        echo "[ERROR] Missing: $SCRIPT_DIR/$script" >&2
        exit 1
    fi
done

# ------------------------------------------------------------
# 启动单个实验
# ------------------------------------------------------------
launch_experiment() {
    local name="$1"
    local gpu="$2"
    local script="$3"
    local extra_args="${4:-}"

    local log_file="$LAUNCH_LOG_DIR/${name}.log"
    local pid_file="$LAUNCH_LOG_DIR/${name}.pid"

    local -a extra_args_array=()
    if [[ -n "$extra_args" ]]; then
        read -r -a extra_args_array <<< "$extra_args"
    fi

    echo "[Launch] $name | GPU=$gpu | script=$script"

    local -a cmd=(
        env CUDA_VISIBLE_DEVICES="$gpu"
        python "$SCRIPT_DIR/$script"
        --dataset "$DATASET"
        --num-clients "$NUM_CLIENTS"
        --dirichlet-sample-balance "$DIRICHLET_SAMPLE_BALANCE"
        --rounds "$ROUNDS"
        --alpha "$ALPHA"
        --seed "$SEED"
        --lr "$LR"
        --momentum "$MOMENTUM"
        --weight-decay "$WEIGHT_DECAY"
        --data-augmentation true
    )

    # 添加算法专属参数
    if (( ${#extra_args_array[@]} > 0 )); then
        cmd+=("${extra_args_array[@]}")
    fi

    # 输出目录和实验名称
    cmd+=(
        --output-dir "$OUTPUT_DIR"
        --run-name "$name"
    )

    nohup "${cmd[@]}" > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"

    echo "         PID=$pid"
    echo "         log=$log_file"
}

# ------------------------------------------------------------
# 并行启动全部实验
# ------------------------------------------------------------
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name gpu script extra_args <<< "$exp"
    launch_experiment "$name" "$gpu" "$script" "$extra_args"
done

# ------------------------------------------------------------
# 启动信息
# ------------------------------------------------------------
echo
echo "============================================================"
echo "All experiments have been launched in parallel."
echo "Dataset       : $DATASET"
echo "Partition     : dirichlet / $DIRICHLET_SAMPLE_BALANCE"
echo "Clients       : $NUM_CLIENTS"
echo "Rounds        : $ROUNDS"
echo "Alpha         : $ALPHA"
echo "Seed          : $SEED"
echo "Learning Rate : $LR"
echo "Momentum      : $MOMENTUM"
echo "Weight Decay  : $WEIGHT_DECAY"
echo "Augmentation  : Enabled"
echo "Output        : $OUTPUT_DIR"
echo "Logs          : $LAUNCH_LOG_DIR"
echo "============================================================"
echo

echo "GPU mapping:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name gpu script extra_args <<< "$exp"
    echo "  $name -> GPU $gpu"
done

echo
echo "Process IDs:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name gpu script extra_args <<< "$exp"
    pid_file="$LAUNCH_LOG_DIR/${name}.pid"

    if [[ -f "$pid_file" ]]; then
        echo "  $name -> $(cat "$pid_file")"
    fi
done

echo
echo "Check processes:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name gpu script extra_args <<< "$exp"
    pid_file="$LAUNCH_LOG_DIR/${name}.pid"

    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        ps -p "$pid" -o pid=,stat=,cmd= || true
    fi
done

echo
echo "Check GPU usage:"
echo "  nvidia-smi"

echo
echo "Check logs:"
echo "  tail -f $LAUNCH_LOG_DIR/*.log"
