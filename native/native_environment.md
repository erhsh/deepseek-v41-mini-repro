# 在 Ascend 950 上复建运行环境

以下步骤面向 Linux x86_64、Python 3.12 和 Ascend 950PR。按顺序在同一个 Bash 终端运行。单卡也会初始化 HCCL。实测主机为 Ubuntu 24.04.4 LTS，使用驱动 25.7.rc1、CANN Toolkit 9.2.0 和同版本 Ascend 950 OPP。先按本机产品、架构和驱动安装 Toolkit、设备算子包及编译工具，确认 `npu-smi info` 能识别设备。Python 需有 venv 支持和同版本的开发头文件（`Python.h`，Ubuntu 软件包为 `python3.12-venv`、`python3.12-dev`）。编译用到 Git、tar、GCC/G++（实测 13.3，支持 C++17）、CMake（项目要求 3.16 及以上）、make 和 Ninja。系统环境配套见[昇腾软件安装说明](https://www.hiascend.com/document)。

完成第 1 节后，可在主仓根目录执行 `bash run_native_repro.sh`，由脚本初始化子模块、构建算子、准备模型文件并运行复现。也可按下文分步执行。外部代码仓以 Git 子模块管理，位于 `3rdparty/`。构建产物、模型文件与缓存统一放在主仓 `build/`（已加入 `.gitignore`）。[启动脚本](../run_native_repro.sh)

## 1. 准备可复用的 Python 环境

从主仓根目录或 `native/` 执行以下命令。将 `CANN_ENV` 替换为本机实际的 Toolkit 环境脚本。上游设备算子构建脚本要求构建目录路径不含空白和通配符，仓库默认路径均满足。

```bash
export NATIVE_DIR="$(git rev-parse --show-toplevel)/native"
export CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
python3.12 -m venv .venv
source .venv/bin/activate
source "$CANN_ENV"

python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.12.0+cpu'
python -m pip install -r "$NATIVE_DIR/requirements.txt"
```

`torch-npu` 为 CPU 版 PyTorch 注册昇腾后端。CANN 环境提供 `cann_ops_nn` 和基础 `cann_ops_transformer`，TorchAir 随 torch-npu 提供。固定版本参见本目录的 [requirements.txt](requirements.txt)。TorchAir 在导入时会读取 protobuf 的包元数据。单卡分配 KV 缓存时，框架会间接导入 requests。CANN 的 opc/TBE 编译链会导入 decorator、SciPy 和 attrs。环境准备完后可检查关键版本。

```bash
python -c 'import torch, torch_npu, torchair, cann_ops_nn, cann_ops_transformer; print(torch.__version__, torch_npu.__version__)'
```

## 2. 准备固定版本的源码

三个外部代码仓均以 Git 子模块管理。递归克隆时随主仓一并检出；普通克隆从主仓根目录或 `native/` 执行以下命令，Git 会检出主仓记录的固定提交。

```bash
git -C "$(git rev-parse --show-toplevel)" submodule update --init --recursive
```

| 来源及固定提交 | 本地位置 | 用途 |
|---|---|---|
| [cann-recipes-infer · 2239704](https://gitcode.com/cann/cann-recipes-infer/tree/2239704f489094605cf6db9bf9d5837841754c1c) | `3rdparty/cann-recipes-infer` | 模型、执行器、部署 YAML、四个设备算子和 PyTorch C++ 扩展。 |
| [ops-transformer · ef931cc](https://gitcode.com/cann/ops-transformer/tree/ef931ccf753524513fedc6efb26eff72ab670a08) | `3rdparty/ops-transformer` | CompressorV2 设备算子和六个 DS41 PyTorch 适配器。 |
| [cannbot-dsl · a095097](https://gitcode.com/cann/cannbot-dsl/tree/a095097ea4c66711b606c985c2a76e48ab9b0c3e) | `3rdparty/cannbot-dsl` | DS41 的 Python 算子与元数据实现。安装的 `cannbot-dsl==0.8.2` 提供编译和运行这些实现的环境。 |

三个项目根目录各有 `LICENSE`，cann-recipes-infer 的设备算子另附 `ops/LICENSE`。cann-recipes-infer 和 ops-transformer 的第三方声明在各自根目录的 `Third_Party_Open_Source_Software_Notice`。

## 3. 构建 Python 适配层和设备算子

从主仓根目录或 `native/` 执行以下命令。两个命令各自创建一个新输出目录，产物位于主仓 `build/` 下，可在后续运行中复用同一套产物。

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
bash "$NATIVE_DIR/prepare_python_ops.sh" "$REPO_ROOT/build/python_ops"
bash "$NATIVE_DIR/build_repro_ops.sh" "$REPO_ROOT/build/native_build"
```

| 产物 | 内容与用途 |
|---|---|
| `build/python_ops/experimental/` | ops-transformer 的 6 个 PyTorch 适配器，注册 CompressorV2、索引器 K/QW 投影、QLI/QSLI 和混合量化稀疏注意力接口。 |
| `build/python_ops/current_provider/ops/` | 从 cannbot-dsl 固定提交的 DS41 清单提取的 42 个 Python 源文件，供 CANNBot-DSL 即时编译。 |
| `build/native_build/opp/` | Ascend 950 设备算子包，包含 HcPre、MoE 门控 top-k、KV 压缩写回、SwiGLU 量化，以及从 ops-transformer 编入的 CompressorV2。 |
| `build/native_build/ascendc/torch_ops_extension/` | `custom_ops` 扩展，按本机 Python、PyTorch、torch-npu 和 CANN 的二进制接口编译。 |

换主机时在目标机器重新构建设备算子包和 C++ 扩展，扩展会使用本机 CPU 指令集。若只更换安装目录，需在目标位置重新安装设备算子包。Python、torch-npu 或 CANN 版本变更时重新构建扩展。运行时还会即时编译部分 DSL 和 C++ 内核，编译器和可写缓存目录需要保留。

## 4. 加载并检查运行环境

新开终端时重新激活 Python 环境并 source CANN，然后执行下列命令。入口会从子模块加载框架。这里设置两个 Python 算子包与本机 C++ 扩展的搜索路径。

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
source "$CANN_ENV"
source "$REPO_ROOT/build/native_build/opp/vendors/customize/bin/set_env.bash"
export PYTHONPATH="$REPO_ROOT/build/python_ops/experimental:$REPO_ROOT/build/python_ops/current_provider:$REPO_ROOT/build/native_build/ascendc/torch_ops_extension${PYTHONPATH:+:$PYTHONPATH}"
export CANNBOTDSL_CACHE_DIR="$REPO_ROOT/build/cache/dsl"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/build/cache/torch"
export OMP_NUM_THREADS=1
mkdir -p "$CANNBOTDSL_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
python -c 'import torch_npu, cann_ops_transformer.ops, cann_ops_transformer_experimental, custom_ops; from cann_ops_transformer.ops.ds41 import compressor, mixed_quant_sparse_flash_mla; print("DS41 imports OK")'
```

接下来准备模型配置和聊天模板。子模块、模型文件、Python 包和构建产物就位后，可在该主机离线运行普通生成和 DSpark。首次推理会即时编译部分算子。[运行说明](README.md)
