# DeepSeek-V4.1-Flash 微缩复现

两套入口都使用合成图文和在内存中初始化的权重。只有 CPU 时，可以从上游参考版开始。要实际执行 cann-recipes-infer 的融合算子，使用 Ascend 950PR 上的仓库实现版。

| 运行条件 | [上游参考版](upstream/README.md) | [仓库实现版](native/README.md) |
|---|---|---|
| 执行入口 | `upstream/upstream_smoke.py` 直接构造官方 `Transformer` | `native/executor_smoke.py` 通过 `OfflineInference.generate()` 执行 |
| 从官方模型下载 | 4 个文件，`inference/{model,engram,vision,image_processor}.py`，共约 80 KiB | 2 个文件，`config.json` 和 `chat_template.jinja`，共约 18 KiB |
| 额外框架源码 | 入口自带 6 个 PyTorch 张量实现 | 固定版本的 cann-recipes-infer、ops-transformer、cannbot-dsl |
| Python 环境 | PyTorch、tokenizers、NumPy、SymPy、Pillow | 匹配的 PyTorch / torch-npu、Transformers、tokenizers、datasets、量化库、DSL 运行时等，另有 CANN 提供的 Python 包 |
| 已测硬件路径 | Linux x86_64 CPU 或单卡 Ascend 950PR | Linux x86_64 + 单卡 Ascend 950PR，CANN 9.2.0、HCCL 和可用的本机编译工具 |
| 编译与缓存 | CPU 流程直接使用公开 Python wheels | 构建 5 个设备算子、1 个 C++ 扩展、6 个 Python 适配器，准备 DS41 的 42 个算子源文件，首次运行还会即时编译 |
| 通过条件 | 一次图文预填充和两步解码，位置为 0、13、14，输出 `"result": "pass"` | 普通生成或 DSpark 模式均输出 `"result": "pass"`。后者还会报告草稿验证轮数 |

两版获取的官方文件均固定在 `314113f8a7dadb388215396ba5fbee2c610a04d6`。随机权重只用于检查执行路径与数值是否有限。启动时无需完整权重分片、分片索引、官方分词器或外部数据集。[官方模型文件](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6)

## 独立仓库包含什么

保留本 README、`upstream/` 和 `native/` 即可复现两种实现。二者各有运行入口与分设备的依赖说明，也可以单独取用。源码包附带原仓的 LICENSE 和第三方软件声明。DeepSeek 的四个源文件、CANN 安装包、设备算子和模型文件按各自说明获取。

原工作区的 `checks/` 是单模块诊断工具，历史进度、权重下载记录、展示页、虚拟环境、已编译二进制和缓存都可以留在工作区。仓库实现版在运行时从独立的框架 checkout 导入模型。算子包安装位置、设备编号、缓存路径由复现者设置。

## 已验证范围

上游版在两个全新的 Python 3.12 虚拟环境中通过测试。CPU 使用 PyTorch 2.10.0+cpu，Ascend 950PR 使用 PyTorch 2.12.0+cpu、torch-npu 2.12.0.post2 和 CANN 9.3.0，两者读取同一组四个官方文件。

仓库实现版使用独立 Python 3.12 环境、固定版本的干净框架 checkout 及重新构建的算子。在预装驱动、CANN 9.2.0 和编译器的 950PR 主机上，普通生成得到了 3 个 token，DSpark 完成 4 轮草稿核验并生成 5 个 token。两次都只读取两个官方文件。Python 依赖、编译产物和缓存均可由复现者在目标机器生成。在其他产品、架构或软件版本上需要重新验证。这些测试验收流程覆盖、张量形状和有限值。正式模型的输出质量需要真实权重验证。[昇腾环境复建说明](native/native_environment.md)
