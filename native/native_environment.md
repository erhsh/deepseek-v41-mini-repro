# 在 Ascend 950 上复建运行环境

以下步骤面向 Linux x86_64、Python 3.12 和 Ascend 950PR。按顺序在同一个 Bash 终端运行。单卡也会初始化 HCCL。实测主机为 Ubuntu 24.04.4 LTS，使用驱动 25.7.rc1、CANN Toolkit 9.2.0 和同版本 Ascend 950 OPP。先按本机产品、架构和驱动安装 Toolkit、设备算子包及编译工具，确认 `npu-smi info` 能识别设备。Python 需有 venv 支持和同版本的开发头文件（`Python.h`，Ubuntu 软件包为 `python3.12-venv`、`python3.12-dev`）。编译用到 Git、tar、GCC/G++（实测 13.3，支持 C++17）、CMake（项目要求 3.16 及以上）、make 和 Ninja。系统环境配套见[昇腾软件安装说明](https://www.hiascend.com/document)。

## 1. 准备可复用的工作目录

进入本说明所在的 `native/` 目录执行。将 `CANN_ENV` 替换为 Toolkit 实际提供的环境脚本。选择一个可长期使用的 `DS41_WORK` 目录，设备算子安装脚本会记录本机绝对路径。

```bash
export NATIVE_DIR="$PWD"
export DS41_WORK="${DS41_WORK:-$HOME/dsv41-mini-work}"
export CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
mkdir -p "$DS41_WORK"
python3.12 -m venv "$DS41_WORK/venv"
source "$DS41_WORK/venv/bin/activate"
source "$CANN_ENV"

python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.12.0+cpu'
python -m pip install -r "$NATIVE_DIR/requirements.txt"
```

`torch-npu` 为 CPU 版 PyTorch 注册昇腾后端。CANN 环境提供 `cann_ops_nn` 和基础 `cann_ops_transformer`，TorchAir 随 torch-npu 提供。固定版本参见本目录的 [requirements.txt](requirements.txt)。TorchAir 在导入时会读取 protobuf 的包元数据。单卡分配 KV 缓存时，框架会间接导入 requests。CANN 的 opc/TBE 编译链会导入 decorator、SciPy 和 attrs。框架上游的模型 requirements 对应另一套软件组合。环境准备完后可检查关键版本。

```bash
python -c 'import torch, torch_npu, torchair, cann_ops_nn, cann_ops_transformer; print(torch.__version__, torch_npu.__version__)'
```

## 2. 获取三个固定版本的源码

```bash
export RECIPES_ROOT="$DS41_WORK/cann-recipes-infer"
git clone https://gitcode.com/easy-cann/cann-recipes-infer.git "$RECIPES_ROOT"
git -C "$RECIPES_ROOT" checkout --detach 2239704f489094605cf6db9bf9d5837841754c1c
git clone https://gitcode.com/cann/ops-transformer.git "$DS41_WORK/ops-transformer"
git -C "$DS41_WORK/ops-transformer" checkout --detach ef931ccf753524513fedc6efb26eff72ab670a08
git clone https://gitcode.com/cann/cannbot-dsl.git "$DS41_WORK/cannbot-dsl"
git -C "$DS41_WORK/cannbot-dsl" checkout --detach a095097ea4c66711b606c985c2a76e48ab9b0c3e
```

框架 checkout 中的 `executor/` 提供模型注册、调度和加载，`models/deepseek_v4_1/` 提供模型与部署 YAML，`module/` 提供共享量化和专家层，`ops/ascendc/` 提供设备算子和 C++ 扩展。模型运行时仍需通过 `PYTHONPATH` 找到它。后两个源码仓用于构建。

## 3. 构建 Python 适配层和设备算子

```bash
bash "$NATIVE_DIR/prepare_python_ops.sh" \
  "$DS41_WORK/ops-transformer" "$DS41_WORK/cannbot-dsl" "$DS41_WORK/python_ops"
bash "$NATIVE_DIR/build_smoke_ops.sh" \
  "$DS41_WORK/ops-transformer" "$DS41_WORK/native_build"
```

两个命令各自创建一个新输出目录，可在后续运行中复用同一套产物。

| 产物 | 内容与用途 |
|---|---|
| `python_ops/experimental/` | ops-transformer 的 6 个 PyTorch 适配器，注册 CompressorV2、索引器 K/QW 投影、QLI/QSLI 和混合量化稀疏注意力接口。 |
| `python_ops/current_provider/ops/` | 从 cannbot-dsl 固定提交的 DS41 清单提取的 42 个 Python 源文件，供 CANNBot-DSL 即时编译。 |
| `native_build/opp/` | Ascend 950 设备算子包，包含 HcPre、MoE 门控 top-k、KV 压缩写回、SwiGLU 量化，以及从 ops-transformer 编入的 CompressorV2。 |
| `native_build/ascendc/torch_ops_extension/` | `custom_ops` 扩展，按本机 Python、PyTorch、torch-npu 和 CANN 的二进制接口编译。 |

换主机或更换安装目录时，在目标位置重新安装设备算子包。Python、torch-npu 或 CANN 版本变更时重新构建扩展。运行时还会即时编译部分 DSL 和 C++ 内核，编译器和可写缓存目录需要保留。

## 4. 加载并检查运行环境

新开终端时重新设置 `DS41_WORK` 和 `CANN_ENV`，然后执行下列命令。让 DS41 的 `ops` 包在 Python 搜索路径上排在框架 checkout 前面。

```bash
source "$DS41_WORK/venv/bin/activate"
source "$CANN_ENV"
source "$DS41_WORK/native_build/opp/vendors/customize/bin/set_env.bash"
export RECIPES_ROOT="$DS41_WORK/cann-recipes-infer"
export PYTHONPATH="$DS41_WORK/python_ops/experimental:$DS41_WORK/python_ops/current_provider:$DS41_WORK/native_build/ascendc/torch_ops_extension:$RECIPES_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CANNBOTDSL_CACHE_DIR="$DS41_WORK/cache/dsl"
export TORCH_EXTENSIONS_DIR="$DS41_WORK/cache/torch"
export OMP_NUM_THREADS=1
mkdir -p "$CANNBOTDSL_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
python -c 'import torch_npu, cann_ops_transformer.ops, cann_ops_transformer_experimental, custom_ops; from cann_ops_transformer.ops.ds41 import compressor, mixed_quant_sparse_flash_mla; print("DS41 imports OK")'
```

接下来按[运行说明](README.md)准备两个模型文件、选择一张卡并执行整网测试。
