#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
用法：run_upstream_repro.sh [--device cpu|npu[:编号]] [--source 目录]

默认在 CPU 上运行，复用 build/inference/ 中的官方源码，缺少时按固定版本下载。
--source 指定已有的四文件源码目录，相对路径从当前工作目录解析。
--device npu:0 使用当前进程可见的首张昇腾卡。
EOF
}

argument_error() {
    printf '%s\n' "$1" >&2
    usage >&2
    exit 2
}

repo_root=$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")
source_dir="$repo_root/build/inference"
explicit_source=false
device=cpu
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --source|--device)
            option=$1
            [[ $# -ge 2 && -n $2 && $2 != -* ]] || argument_error "$option 缺少参数"
            value=$2
            shift 2
            ;;
        --source=*|--device=*)
            option=${1%%=*}
            value=${1#*=}
            [[ -n $value ]] || argument_error "$option 缺少参数"
            shift
            ;;
        *) argument_error "未知参数：$1" ;;
    esac
    if [[ $option == --source ]]; then
        source_dir=$value
        explicit_source=true
    else
        [[ $value == cpu || $value =~ ^npu(:[0-9]+)?$ ]] || argument_error "不支持的设备：$value"
        device=$value
    fi
done

files=(model.py engram.py vision.py image_processor.py)
collect_missing() {
    local file
    missing=()
    for file in "${files[@]}"; do
        if [[ ! -f "$source_dir/$file" || ! -s "$source_dir/$file" ]]; then
            missing+=("$file")
        fi
    done
}

download_missing() (
    # Lock the directory itself. Another launcher may finish while we wait.
    mkdir -p -- "$source_dir"
    exec 9<"$source_dir"
    flock -x 9
    collect_missing
    (( ${#missing[@]} )) || return 0

    local revision=314113f8a7dadb388215396ba5fbee2c610a04d6
    local base="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision/inference"
    local staging file
    staging=$(mktemp -d -- "$source_dir/.download.XXXXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    for file in "${missing[@]}"; do
        printf '下载 %s\n' "$file" >&2
        curl --fail --location --silent --show-error --retry 2 --connect-timeout 15 --max-time 120 \
            --output "$staging/$file" "$base/$file"
        if [[ ! -s "$staging/$file" ]]; then
            printf '下载的文件为空：%s\n' "$file" >&2
            return 1
        fi
    done
    for file in "${missing[@]}"; do
        mv -fT -- "$staging/$file" "$source_dir/$file"
    done
)

collect_missing
if [[ $explicit_source == true ]] && (( ${#missing[@]} )); then
    printf '目录 %s 中缺少以下文件或文件为空：\n' "$source_dir" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

if command -v python >/dev/null; then
    python=python
elif command -v python3 >/dev/null; then
    python=python3
else
    printf '找不到 Python，请先激活包含 PyTorch 的 Python 环境。\n' >&2
    exit 1
fi
if (( ${#missing[@]} )); then
    download_missing
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$python" -B "$repo_root/upstream/upstream_repro.py" --device "$device" --source="$source_dir"
