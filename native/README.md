# DeepSeek-V4.1-Flash 仓库实现微缩版

[executor_smoke.py](executor_smoke.py) 调用 cann-recipes-infer 的 `OfflineInference.generate()`，用合成图文和测试权重在一张 Ascend 950PR 上执行预填充和解码。文本主干为 6 层、隐藏维度 5120，视觉编码器为 1 层。`--dspark` 增加 3 层草稿模型和主模型验证。

## 运行

1. 在 [Ascend 环境复建说明](native_environment.md) 中完成环境准备，并执行最后一节加载 Python 包和算子。已有同版本环境时可直接加载。
2. 在本说明所在目录执行以下命令，获取固定版本的模型配置和聊天模板。

```bash
export MODEL_SOURCE="${MODEL_SOURCE:-$HOME/dsv41-mini-work/model}"
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
base="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision"
mkdir -p "$MODEL_SOURCE"
curl --fail --fail-early --location --retry 2 --remote-name-all \
  --output-dir "$MODEL_SOURCE" "$base/"{config.json,chat_template.jinja}
```

3. 选择一张可用物理卡。入口也会读取 `MODEL_SOURCE` 环境变量，默认目录与上面的下载命令一致。下例使用卡 0，脚本内部访问可见设备 `npu:0`。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python executor_smoke.py --source "$MODEL_SOURCE"
python executor_smoke.py --source "$MODEL_SOURCE" --dspark
```

脚本从框架 checkout 定位 `models/deepseek_v4_1/config/deepseek_v4_1_flash_rank_8_8ep_multimodal.yaml`，也可以用 `--recipe` 传入绝对路径。运行时覆盖为单 rank、假权重、普通专家路由和单卡 Engram。临时分词器、测试权重和输入均由脚本生成。单卡流程无需下载正式权重、词表、分片索引或图文数据集。固定的框架版本在导入执行器时就会导入 `datasets`，因此 Python 依赖仍包含这个包。当前选择 Ascend C 和普通专家路由，可省去 TileKernels、DeepGEMM 和离线权重转换工具。框架探测可选 TileLang 实现时会打印 `No module named 'tile_kernels'`，本配方继续使用 Ascend C 路径。

输出 JSON 中 `result` 为 `pass`，`calls.vision` 和 `calls.prefill` 均为 1。开启 DSpark 时，`speculative_forwards` 会记录验证轮数。该用例检查执行路径、输出形状与有限非零 logits。真实权重的精度、吞吐与八卡融合通信需另行测试。
