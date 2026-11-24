#!/usr/bin/env bash
# Build the five Ascend 950 operators and their PyTorch extension in a private directory.
# Run after activating PyTorch and sourcing CANN.
set -euo pipefail

native_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$native_dir/.." && pwd -P)
source "$native_dir/source_revisions.sh"
case $# in
    1)
        ops_root="$repo_root/3rdparty/ops-transformer"
        output=$1
        ;;
    2)
        ops_root=$1
        output=$2
        ;;
    *)
        echo "Usage: bash $0 [OPS_TRANSFORMER_CHECKOUT] NEW_BUILD_DIRECTORY" >&2
        exit 2
        ;;
esac

recipes="$repo_root/3rdparty/cann-recipes-infer"
if [[ ! -f "$recipes/ops/ascendc/build.sh" ]]; then
    echo "Missing cann-recipes-infer. From the repository root, run: git submodule update --init --recursive" >&2
    exit 1
fi
compressor_tree="$OPS_TRANSFORMER_REV:experimental/attention/compressor_v2"
if ! git -C "$ops_root" cat-file -e "$compressor_tree/op_kernel/compressor_v2.cpp" 2>/dev/null; then
    echo "Missing ops-transformer revision $OPS_TRANSFORMER_REV under $ops_root. From the repository root, run: git submodule update --init --recursive" >&2
    exit 1
fi
: "${ASCEND_HOME_PATH:?Source your CANN environment first}"
# The upstream build passes its working directory unquoted to rm and mkdir.
build_dir=$(realpath -m -- "$output")
case "$build_dir" in
    *[[:space:]]*|*'*'*|*'?'*|*'['*)
        echo "Choose a build directory without whitespace or shell wildcards: $build_dir" >&2
        exit 2
        ;;
    "$recipes"|"$recipes"/*)
        echo "Choose a build directory outside the cann-recipes-infer submodule." >&2
        exit 2
        ;;
esac
mkdir -- "$output"
work="$build_dir/ascendc"
mkdir "$work"
recipe_ops=(hc_pre moe_gating_top_k_hash kv_compress_epilog_v2 swiglu_group_quant)
sources=(CMakeLists.txt build.sh version.info cmake src/utils torch_ops_extension/setup.py torch_ops_extension/custom_ops)
for name in "${recipe_ops[@]}"; do
    sources+=("src/$name/CMakeLists.txt" "src/$name/op_host" "src/$name/op_kernel")
done
git -C "$recipes" archive HEAD:ops/ascendc -- "${sources[@]}" | tar -x -C "$work"
op="$work/src/compressor_v2"
mkdir "$op"
git -C "$ops_root" archive "$compressor_tree" -- op_host op_kernel | tar -x -C "$op"

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
op_names=$(IFS=';'; echo "${recipe_ops[*]};compressor_v2")
OPS_CPU_NUMBER=${OPS_CPU_NUMBER:-4} bash build.sh -c ascend950 -n "$op_names"
mkdir "$build_dir/opp"
bash output/CANN-custom_ops-*-linux.*.run --quiet --install-path="$build_dir/opp"
cd torch_ops_extension
MAX_JOBS=${MAX_JOBS:-4} USE_NINJA=1 python setup.py build_ext --inplace
printf 'source %q\n' "$build_dir/opp/vendors/customize/bin/set_env.bash"
printf 'export PYTHONPATH=%q:"${PYTHONPATH:-}"\n' "$work/torch_ops_extension"
