#!/usr/bin/env bash
# Stage the Python bridges and the DS41 operator sources in a private directory.
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "Usage: bash prepare_python_ops.sh OPS_TRANSFORMER_CHECKOUT CANNBOT_DSL_CHECKOUT NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
ops_checkout=$(realpath "$1")
dsl_checkout=$(realpath "$2")
output=$(realpath -m "$3")
if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    exit 1
fi
mkdir -p "$(dirname "$output")"
work=$(mktemp -d "$(dirname "$output")/.ds41-python.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
mkdir -p "$work/source" "$work/package/current_provider/ops"

ops_rev=ef931ccf753524513fedc6efb26eff72ab670a08
dsl_rev=a095097ea4c66711b606c985c2a76e48ab9b0c3e
names=(compressor_v2 indexer_prologue_k_dsl indexer_prologue_qw_dsl mixed_quant_sparse_flash_mla_dsl quant_lightning_indexer_dsl quant_sparse_lightning_indexer_dsl)
paths=(torch_extension mc2/common/op_kernel)
for name in "${names[@]}"; do
    paths+=("experimental/attention/$name/torch_extension")
done
git -C "$ops_checkout" archive "$ops_rev" "${paths[@]}" | tar -x -C "$work/source"
(
    cd "$work/source/torch_extension"
    export TORCH_EXTENSION_VENDOR=experimental TORCH_EXTENSION_EXPERIMENTAL=TRUE
    TORCH_EXTENSION_OPS=$(IFS=,; echo "${names[*]}")
    export TORCH_EXTENSION_OPS
    python setup.py bdist_wheel --dist-dir "$work/wheels"
)
python -m pip install --no-index --no-deps --target "$work/package/experimental" "$work"/wheels/*.whl

python - "$dsl_checkout" "$dsl_rev" "$work/package/current_provider/ops" <<'PY'
import subprocess
import sys
import tomllib
from pathlib import Path

checkout, revision, target = sys.argv[1:]
package = Path(target)

def read_source(path):
    return subprocess.check_output(["git", "-C", checkout, "show", f"{revision}:{path}"])

manifest = tomllib.loads(read_source("packaging/native/operator_groups.toml").decode())
for name in manifest["groups"]["ds41"]:
    for path in manifest["operators"][name]:
        (package / Path(path).name).write_bytes(read_source(f"samples/{path}"))
(package / "__init__.py").touch()
PY
mv -T "$work/package" "$output"
