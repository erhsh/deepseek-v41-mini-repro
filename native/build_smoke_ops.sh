#!/usr/bin/env bash
# Build the Ascend 950 recipe operators and the upstream CompressorV2 in one
# private vendor directory. Run after activating PyTorch and sourcing CANN.
# Usage: bash build_smoke_ops.sh /path/to/ops-transformer /new/build-directory
# Set RECIPES_ROOT if the pinned framework checkout is outside DS41_WORK.
# Tested ops-transformer revision: ef931ccf753524513fedc6efb26eff72ab670a08
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: $0 OPS_TRANSFORMER_CHECKOUT NEW_BUILD_DIRECTORY" >&2
    exit 2
fi
upstream=$(realpath "$1")/experimental/attention/compressor_v2
test -f "$upstream/op_kernel/compressor_v2.cpp"
: "${ASCEND_HOME_PATH:?Source your CANN environment first}"
repo=${RECIPES_ROOT:-"${DS41_WORK:-$HOME/dsv41-mini-work}/cann-recipes-infer"}
if [[ ! -f "$repo/ops/ascendc/build.sh" ]]; then
    echo "Set RECIPES_ROOT to the cann-recipes-infer directory containing ops/ascendc/build.sh." >&2
    exit 1
fi
mkdir "$2"
build_dir=$(cd "$2" && pwd)
cp -R "$repo/ops/ascendc" "$build_dir/ascendc"
work="$build_dir/ascendc"
op="$work/src/compressor_v2"
mkdir "$op"
cp -R "$upstream/op_host" "$upstream/op_kernel" "$op/"

# Register the original upstream implementation with the recipe build targets.
cat > "$op/CMakeLists.txt" <<'CMAKE'
add_ops_compile_options(OP_NAME CompressorV2 OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -mllvm -cce-aicore-hoist-movemask=false)
target_sources(op_host_aclnn PRIVATE op_host/compressor_v2_def.cpp)
target_sources(optiling PRIVATE op_host/compressor_v2_tiling_register.cpp op_host/arch22/compressor_tiling.cpp op_host/arch35/compressor_tiling.cpp)
target_sources(opsproto PRIVATE op_host/compressor_v2_infershape.cpp)
foreach(target IN ITEMS optiling opsproto op_host_aclnn)
    target_include_directories(${target} PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include/op_common ${ASCEND_CANN_PACKAGE_PATH}/include/experiment)
endforeach()
set_property(TARGET optiling opsproto op_host_aclnn PROPERTY CXX_STANDARD 17)
target_compile_options(optiling PRIVATE -std=c++17)
CMAKE

cd "$work"
OPS_CPU_NUMBER=${OPS_CPU_NUMBER:-4} bash build.sh -c ascend950 \
    -n 'hc_pre;moe_gating_top_k_hash;kv_compress_epilog_v2;swiglu_group_quant;compressor_v2'
mkdir "$build_dir/opp"
bash output/CANN-custom_ops-*-linux.*.run --quiet --install-path="$build_dir/opp"
cd torch_ops_extension
MAX_JOBS=${MAX_JOBS:-4} USE_NINJA=1 python setup.py build_ext --inplace
printf 'source %q\n' "$build_dir/opp/vendors/customize/bin/set_env.bash"
printf 'export PYTHONPATH=%q:"${PYTHONPATH:-}"\n' "$work/torch_ops_extension"
