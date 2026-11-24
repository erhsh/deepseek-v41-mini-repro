# DeepSeek-V4.1-Flash 上游参考实现的微缩版

在 Linux x86_64 CPU 或单张昇腾 NPU 上，运行官方模型的 6 层文本主干、1 层视觉编码器和 3 层 DSpark 草稿模型。入口在内存中构造分词器、图文输入和测试权重，并用 PyTorch 张量运算实现 6 个上游 CUDA TileLang 接口。

从仓库根目录执行 `bash run_upstream_repro.sh` 即可在 CPU 上复现。脚本会补齐 `build/inference/` 中缺少的模型文件，文件齐全后可离线运行。已有完整源码目录时，可传入 `--source DIR`，相对路径从当前工作目录解析。用 `--device npu:0` 可在可见的首张昇腾卡上运行。[启动脚本](../run_upstream_repro.sh)

也可以只复制本目录，直接运行 Python 入口。下列命令从本 README 所在目录执行。

## 1. 取得四个官方源文件

`--source` 接受一个包含四个 Python 文件的目录。`model.py` 定义主干和 DSpark，`engram.py` 提供 Engram 哈希，`vision.py` 定义视觉编码器与对齐层，`image_processor.py` 提供图像输入类型和标记。这四个文件合计 81,963 字节，固定到本次验证的版本。[官方源码](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6/inference) · [上游 MIT 许可证](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/LICENSE)

```bash
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
base="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision/inference"
mkdir -p ../build/inference
curl --fail --fail-early --location --retry 2 --remote-name-all \
  --output-dir ../build/inference "$base/"{model,engram,vision,image_processor}.py
```

脚本默认从主仓根的 `build/inference/` 读取源码，默认在 CPU 上运行。已有相同版本源码时，可直接将其 `inference/` 路径传给脚本。合成图像和分词器覆盖这次请求所需的数据，因此模型权重分片和官方 tokenizer 文件均可省略。

## 2. 在 CPU 上最小复现

以下组合已在隔离的 Python 3.12.3 环境运行。PyTorch 负责张量与低精度数据类型，tokenizers 构建合成词表，NumPy、SymPy 和 Pillow 由官方源码导入。四个通用包的实测版本列在 [requirements.txt](requirements.txt)，PyTorch 按设备单独安装。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.10.0+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 python upstream_repro.py --device cpu
```

## 3. 在昇腾 NPU 上运行（可选）

新开 Bash 终端，在已安装配套驱动和 CANN 的主机上激活独立的 NPU 虚拟环境，并加载相应的 Toolkit 环境脚本。实测平台为 Linux x86_64、Ascend 950PR、CANN Toolkit 9.3.0、Python 3.12、PyTorch 2.12.0+cpu 和 torch_npu 2.12.0.post2。以下 `CANN_ENV` 指向该版本 Toolkit 的实际安装路径。物理卡编号由 `ASCEND_RT_VISIBLE_DEVICES` 选择。脚本在该进程中访问的首张卡为 `npu:0`。

```bash
python3.12 -m venv .venv-npu
source .venv-npu/bin/activate
source "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
python -m pip install 'torch==2.12.0+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install 'torch-npu==2.12.0.post2' -r requirements.txt
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  python upstream_repro.py --device npu:0
```

## 通过条件

最后一行 JSON 含 `"result": "pass"`。`stages` 记录位置 0、13、14 的输出，分别对应 13-token 图文预填充和连续两步解码。三步的 logits 形状都是 `[1, 128]`。`calls` 包含视觉、Engram、压缩 KV、稀疏索引、专家、草稿层及六个张量算子接口的调用次数。测试检查输出形状、有限值和执行覆盖，生成的 token 来自合成权重。
