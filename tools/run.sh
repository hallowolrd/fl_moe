#!/usr/bin/env bash
set -u

# ============================================================
# 联邦学习多算法并行启动脚本
# 数据集 / Backbone / 轮数 / 对比算法 / GPU 均在顶部配置
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Project/clean}"
SCRIPT_DIR="$PROJECT_ROOT/a"

# ------------------------------------------------------------
# 实验公共配置
# ------------------------------------------------------------
DATASET="cifar100"
BACKBONE="vgg11_075"
ROUNDS=100
ALPHA=0.1
SEED=0

OUTPUT_DIR="$PROJECT_ROOT/outputs_tkfac/${DATASET}_${BACKBONE}"
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
    "fisher|0|tkfac_expert.py|"
    "fed_moe|0|fed_moe_style_expert.py|"
    "fedmoe_da|1|fedmoe_da_style_expert.py|"
    "somfed|1|somfed_style_expert.py|"
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

    nohup env CUDA_VISIBLE_DEVICES="$gpu" \
        python "$SCRIPT_DIR/$script" \
        --dataset "$DATASET" \
        --backbone "$BACKBONE" \
        --rounds "$ROUNDS" \
        --alpha "$ALPHA" \
        --seed "$SEED" \
        "${extra_args_array[@]}" \
        --output-dir "$OUTPUT_DIR" \
        > "$log_file" 2>&1 &

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
echo "Dataset : $DATASET"
echo "Backbone: $BACKBONE"
echo "Rounds  : $ROUNDS"
echo "Alpha   : $ALPHA"
echo "Seed    : $SEED"
echo "Output  : $OUTPUT_DIR"
echo "Logs    : $LAUNCH_LOG_DIR"
echo "============================================================"
echo

echo "GPU mapping:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name gpu script extra_args <<< "$exp"
    echo "  $name -> GPU $gpu"
done

echo
echo "Check processes:"
echo "  ps -fp \$(cat $LAUNCH_LOG_DIR/*.pid 2>/dev/null)"

echo
echo "Check GPU usage:"
echo "  nvidia-smi"