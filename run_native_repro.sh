#!/usr/bin/env bash
# DeepSeek-V4.1-Flash 仓库实现版 (native) 一键运行脚本
#
# 用法:
#   bash run_native_repro.sh              # 自动完成全部准备, 然后跑 普通生成 + DSpark
#   bash run_native_repro.sh normal       # 只跑普通生成
#   bash run_native_repro.sh dspark       # 只跑 DSpark
#
# 运行前提 (脚本不代管, 请自行保证):
#   - 已激活含 PyTorch / torch-npu 等依赖的 Python 环境 (python 在 PATH 上)
#   - 已 source CANN 环境 (ASCEND_HOME_PATH 已设置)
#
# 目录约定:
#   3rdparty/   外部代码仓, 以 Git 子模块管理, 固定提交由主仓记录
#   build/      构建产物、远端下载与缓存 (已 gitignore)
#
# 所有准备步骤幂等: 已就位的子模块/构建产物/模型文件会直接跳过。
#
# 可用环境变量覆盖默认值:
#   MODEL_SOURCE              模型文件目录  (默认 <仓库>/build/model)
#   ASCEND_RT_VISIBLE_DEVICES  指定 NPU 卡号 (默认自动选择 HBM 占用最低的卡)
#   OPS_CPU_NUMBER            算子编译并行数 (默认 32)

set -euo pipefail

usage() {
    echo "用法: bash ${0##*/} [normal|dspark|all]"
    echo "默认按顺序运行普通生成和 DSpark。"
}

MODE="${1:-all}"
case "$MODE" in
    normal|dspark|all) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd -P)}"
NATIVE_DIR="$REPO_ROOT/native"
BUILD_DIR="$REPO_ROOT/build"
MODEL_SOURCE="${MODEL_SOURCE:-$BUILD_DIR/model}"
export OPS_CPU_NUMBER="${OPS_CPU_NUMBER:-32}"

info() { printf '\033[32m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -d "$NATIVE_DIR" ]] || die "未找到 $NATIVE_DIR (REPO_ROOT=$REPO_ROOT, 可用 REPO_ROOT 覆盖)"

if command -v python >/dev/null 2>&1; then
    PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    die "PATH 上没有 python, 请先激活 Python 环境"
fi

# ---- 1. 前置环境检查 (只检查, 不设置) ----
info "Python: $($PYTHON --version 2>&1) ($($PYTHON -c 'import sys; print(sys.executable)'))"
[[ -n "${ASCEND_HOME_PATH:-}" ]] || die "未检测到 CANN 环境 (ASCEND_HOME_PATH 未设置), 请先 source 本机 CANN 的 set_env.sh"
info "CANN: $ASCEND_HOME_PATH"
$PYTHON -c 'import torch, torch_npu, transformers, datasets, tokenizers, cannbotdsl, compressed_tensors, PIL, yaml' >/dev/null 2>&1 \
    || die "Python 环境缺少关键依赖, 请自行安装: $PYTHON -m pip install -r native/requirements.txt"

# ---- 2. 子模块 ----
if [[ -f "$REPO_ROOT/3rdparty/cann-recipes-infer/executor/offline/offline_inference.py" \
   && -d "$REPO_ROOT/3rdparty/ops-transformer/experimental/attention/compressor_v2" \
   && -f "$REPO_ROOT/3rdparty/cannbot-dsl/packaging/native/operator_groups.toml" ]]; then
    info "3rdparty/ 子模块已就位"
else
    info "初始化 3rdparty/ 子模块 (cann-recipes-infer, ops-transformer, cannbot-dsl)..."
    git -C "$REPO_ROOT" submodule update --init --recursive
fi

# ---- 3. Python 适配层 ----
if [[ -d "$BUILD_DIR/python_ops/experimental" && -d "$BUILD_DIR/python_ops/current_provider" ]]; then
    info "Python 适配层已就位 (build/python_ops)"
else
    info "构建 Python 适配层..."
    bash "$NATIVE_DIR/prepare_python_ops.sh" \
        "$REPO_ROOT/3rdparty/ops-transformer" "$REPO_ROOT/3rdparty/cannbot-dsl" "$BUILD_DIR/python_ops"
fi

# ---- 4. 设备算子 + C++ 扩展 ----
if [[ -f "$BUILD_DIR/native_build/opp/vendors/customize/bin/set_env.bash" ]]; then
    info "设备算子与 custom_ops 扩展已就位 (build/native_build)"
else
    info "构建设备算子 (OPS_CPU_NUMBER=$OPS_CPU_NUMBER, 约 5-30 分钟)..."
    bash "$NATIVE_DIR/build_repro_ops.sh" "$REPO_ROOT/3rdparty/ops-transformer" "$BUILD_DIR/native_build"
fi

# ---- 5. 模型配置 ----
if [[ -f "$MODEL_SOURCE/config.json" && -f "$MODEL_SOURCE/chat_template.jinja" ]]; then
    info "模型配置已就位: $MODEL_SOURCE"
else
    info "下载模型 config.json + chat_template.jinja..."
    revision=314113f8a7dadb388215396ba5fbee2c610a04d6
    base="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision"
    mkdir -p "$MODEL_SOURCE"
    curl --fail --fail-early --location --retry 2 --remote-name-all \
        --output-dir "$MODEL_SOURCE" "$base/"{config.json,chat_template.jinja}
fi

# ---- 6. 运行时环境 ----
# customize 的 set_env.bash 直接追加以下变量, 先兜底避免 set -u 下 unbound 报错
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# shellcheck source=/dev/null
source "$BUILD_DIR/native_build/opp/vendors/customize/bin/set_env.bash" >/dev/null 2>&1
export PYTHONPATH="$BUILD_DIR/python_ops/experimental:$BUILD_DIR/python_ops/current_provider:$BUILD_DIR/native_build/ascendc/torch_ops_extension${PYTHONPATH:+:$PYTHONPATH}"
export CANNBOTDSL_CACHE_DIR="$BUILD_DIR/cache/dsl"
export TORCH_EXTENSIONS_DIR="$BUILD_DIR/cache/torch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "$CANNBOTDSL_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

info "导入自检 (torch_npu / cann_ops_transformer / custom_ops / DS41 ops)..."
$PYTHON -c 'import torch_npu, cann_ops_transformer.ops, cann_ops_transformer_experimental, custom_ops; from cann_ops_transformer.ops.ds41 import compressor, mixed_quant_sparse_flash_mla' >/dev/null 2>&1 \
    || die "导入自检失败, 请检查 CANN ($ASCEND_HOME_PATH) 是否提供 cann_ops_nn / cann_ops_transformer"

# ---- 7. 选择 NPU 卡 ----
pick_idle_card() {
    local count id rate best_id="" best_rate=""
    count=$(npu-smi info -l 2>/dev/null | awk -F: '/Total Count/ {gsub(/[^0-9]/, "", $2); print $2}')
    [[ "$count" =~ ^[0-9]+$ ]] || return 1
    for ((id = 0; id < count; id++)); do
        rate=$(npu-smi info -t usages -i "$id" 2>/dev/null \
            | awk -F: '/HBM Usage Rate/ {gsub(/[^0-9]/, "", $2); print $2}')
        [[ "$rate" =~ ^[0-9]+$ ]] || continue
        if [[ -z "$best_rate" || "$rate" -lt "$best_rate" ]]; then
            best_rate=$rate
            best_id=$id
        fi
    done
    [[ -n "$best_id" ]] && echo "$best_id"
}

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    info "使用指定的 NPU 卡: $ASCEND_RT_VISIBLE_DEVICES"
else
    card=$(pick_idle_card) || true
    [[ -n "${card:-}" ]] || die "无法自动选择 NPU 卡, 请手动 export ASCEND_RT_VISIBLE_DEVICES=<卡号>"
    export ASCEND_RT_VISIBLE_DEVICES="$card"
    info "自动选择当前最空闲的 NPU 卡: $card"
fi

# ---- 8. 运行复现 ----
LOG_DIR="$BUILD_DIR/logs"
mkdir -p "$LOG_DIR"

run_repro() {
    local label=$1
    shift
    local log="$LOG_DIR/${label}.log"
    local rc=0 start=$SECONDS
    info "运行 $label (日志: $log)..."
    $PYTHON "$NATIVE_DIR/executor_repro.py" --source "$MODEL_SOURCE" "$@" 2>&1 | tee "$log" || rc=$?
    local last
    last=$(grep -o '{"result".*}' "$log" | tail -1 || true)
    if [[ $rc -eq 0 && "${last:-}" == *'"result": "pass"'* ]]; then
        info "$label 通过 ($((SECONDS - start))s): ${last:-请查看日志}"
        return 0
    fi
    die "$label 失败 (exit=$rc), 请查看日志: $log"
}

case "$MODE" in
    normal) run_repro normal ;;
    dspark) run_repro dspark --dspark ;;
    all)    run_repro normal
            run_repro dspark --dspark ;;
esac

info "完成: 全部通过的运行模式 = $MODE"
