# DeepSeek-V4.1-Flash 仓库实现微缩版

[executor_repro.py](executor_repro.py) 调用 cann-recipes-infer 的 `OfflineInference.generate()`，用合成图文和测试权重在一张 Ascend 950PR 上执行预填充和解码。文本主干为 6 层、隐藏维度 5120，视觉编码器为 1 层。`--dspark` 增加 3 层草稿模型和主模型验证。

从仓库根目录执行 `bash run_native_repro.sh`，默认依次运行普通生成和 DSpark。也可指定 `normal` 或 `dspark` 单独运行一种模式。[启动脚本](../run_native_repro.sh)

入口按自身路径从 `3rdparty/cann-recipes-infer/` 导入模型和执行器。首次使用需保留主仓的 Git 克隆，以便取得固定版本的子模块。首次运行前需激活 Python 与 CANN 环境。[环境复建说明](native_environment.md)

## 运行

1. 在同一 Bash 终端加载环境复建说明最后一节的 Python 环境和算子包。
2. 从主仓根目录或 `native/` 执行以下命令。
   `--source` 指向仅需包含 `config.json` 和 `chat_template.jinja` 的模型文件目录。首次下载后可复用，已有这两个文件时可直接将 `MODEL_SOURCE` 设为所在目录。

```bash
export NATIVE_DIR="$(git rev-parse --show-toplevel)/native"
export MODEL_SOURCE="${MODEL_SOURCE:-$(git rev-parse --show-toplevel)/build/model}"
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
base="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision"
mkdir -p "$MODEL_SOURCE"
curl --fail --fail-early --location --retry 2 --remote-name-all \
  --output-dir "$MODEL_SOURCE" "$base/"{config.json,chat_template.jinja}
```

3. 选择一张可用物理卡。入口也会读取 `MODEL_SOURCE` 环境变量，未设置时默认使用主仓 `build/model`。下例使用卡 0，脚本内部访问可见设备 `npu:0`。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python "$NATIVE_DIR/executor_repro.py" --source "$MODEL_SOURCE"
python "$NATIVE_DIR/executor_repro.py" --source "$MODEL_SOURCE" --dspark
```

脚本从子模块读取 `models/deepseek_v4_1/config/deepseek_v4_1_flash_rank_8_8ep_multimodal.yaml`，也可以用 `--recipe` 传入绝对路径。运行时覆盖为单 rank、假权重、普通专家路由和单卡 Engram。临时分词器、测试权重和输入均由脚本生成。单卡流程无需下载正式权重、词表、分片索引或图文数据集。固定的框架版本在导入执行器时就会导入 `datasets`，因此 Python 依赖仍包含这个包。当前选择 Ascend C 和普通专家路由，可省去 TileKernels、DeepGEMM 和离线权重转换工具。框架探测可选 TileLang 实现时会打印 `No module named 'tile_kernels'`，本配方继续使用 Ascend C 路径。

模型检查通过且 HCCL 通信组正常关闭后，脚本输出 `"result": "pass"` 并以状态码 0 退出。`calls.vision` 和 `calls.prefill` 均为 1。开启 DSpark 时，`speculative_forwards` 会记录验证轮数。该用例检查执行路径、输出形状与有限非零 logits。真实权重的精度、吞吐与八卡融合通信需另行测试。

在共享 950PR 主机上曾遇到 HCCL 退组时报 `507018`。若设备日志同时显示 `libscatter_aicpu_kernel.so` 缺少 `HcclLaunchAicpuCacheEvictKernel`，请核对该设备实际加载的 AICPU 算子包与主机 CANN 版本。遇到退组异常时，入口会以非零状态退出。
