# DeepSeek-V4.1-Flash 微缩复现

两套入口都使用合成图文和在内存中初始化的权重。只有 CPU 时，可以从上游参考版开始。要实际执行 cann-recipes-infer 的融合算子，使用 Ascend 950PR 上的仓库实现版。

| 运行条件 | [上游参考版](upstream/README.md) | [仓库实现版](native/README.md) |
|---|---|---|
| 执行入口 | `upstream/upstream_smoke.py` 直接构造官方 `Transformer` | `native/executor_smoke.py` 通过 `OfflineInference.generate()` 执行 |
| 从官方模型下载 | 4 个文件，`inference/{model,engram,vision,image_processor}.py`，共约 80 KiB | 2 个文件，`config.json` 和 `chat_template.jinja`，共约 18 KiB |
| 源码位置 | 入口自带 6 个 PyTorch 张量实现 | 模型、执行器与两个算子项目均来自 `3rdparty/` 下的 Git 子模块 |
| Python 环境 | PyTorch、tokenizers、NumPy、SymPy、Pillow | 匹配的 PyTorch / torch-npu、Transformers、tokenizers、datasets、量化库、DSL 运行时等，另有 CANN 提供的 Python 包 |
| 已测硬件路径 | Linux x86_64 CPU 或单卡 Ascend 950PR | Linux x86_64 + 单卡 Ascend 950PR，CANN 9.2.0、HCCL 和可用的本机编译工具 |
| 编译与缓存 | CPU 流程直接使用公开 Python wheels | 构建 5 个设备算子、1 个 C++ 扩展、6 个 Python 适配器，准备 DS41 的 42 个算子源文件，产物位于主仓 `build/`（已 gitignore），首次运行还会即时编译 |
| 通过条件 | 一次图文预填充和两步解码，位置为 0、13、14，输出 `"result": "pass"` | 普通生成或 DSpark 模式均输出 `"result": "pass"`。后者还会报告草稿验证轮数 |

两版获取的官方文件均固定在 `314113f8a7dadb388215396ba5fbee2c610a04d6`。随机权重只用于检查执行路径与数值是否有限。启动时无需完整权重分片、分片索引、官方分词器或外部数据集。[官方模型文件](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6)

## 克隆方式

只运行上游参考版可以普通克隆，然后按 [upstream/README.md](upstream/README.md) 准备四个官方文件。它独立于 `native/` 和子模块。

运行昇腾仓库实现版时，把 `REPRO_SOURCE` 设为主仓的 Git 地址或本机绝对路径，在目标父目录克隆并进入工作目录。

```bash
git clone --recurse-submodules "$REPRO_SOURCE" dsv41-mini-run
cd dsv41-mini-run
```

在已有的普通克隆中，从主仓根目录初始化子模块。

```bash
git submodule update --init --recursive
```

`native/executor_smoke.py` 提供微缩配置、合成输入、权重初始化和验收逻辑。Git 在 `3rdparty/` 下检出固定版本的 cann-recipes-infer、ops-transformer 和 cannbot-dsl 三个子模块。在就绪的 Python 与 CANN 环境下，可用主仓根的 `run_native_smoke.sh` 一键完成准备与运行；也可按顺序完成[环境搭建与算子构建](native/native_environment.md)和[模型文件准备与推理](native/README.md)。

本项目的许可证和第三方声明位于主仓根目录。子模块和仓外依赖保留各自的许可证与声明。[本项目许可证](LICENSE) · [第三方声明](Third_Party_Open_Source_Software_Notice)

## 已验证范围

上游版在两个全新的 Python 3.12 虚拟环境中通过测试。CPU 使用 PyTorch 2.10.0+cpu，Ascend 950PR 使用 PyTorch 2.12.0+cpu、torch-npu 2.12.0.post2 和 CANN 9.3.0，两者读取同一组四个官方文件。

昇腾仓库实现版从新的递归克隆构建了 5 个设备算子和 PyTorch C++ 扩展，在 Ascend 950PR、CANN 9.2.0 上分别完成普通生成和 DSpark。普通生成输出 3 个 token，DSpark 完成 4 轮草稿核验并输出 5 个 token。实测两条路径各有 80 和 81 个带源码文件的框架模块从子模块加载，构建和推理结束后子模块工作树干净。上游参考版也在未初始化子模块的普通克隆中通过了 CPU 测试。

本用例检查图文预填充、解码、输出形状和有限值。正式模型的输出质量需要真实权重验证，多卡通信需要目标拓扑验证。[昇腾环境复建说明](native/native_environment.md)
