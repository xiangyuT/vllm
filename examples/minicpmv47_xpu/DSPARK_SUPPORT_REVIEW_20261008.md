# MiniCPM V4.7 DSpark 权重与 XPU 支持调查

> 2026-10-08 GPU 验证更新：已在物理卡 0、1 加载 DSpark，补齐 non-causal attention 编译配置并通过 24 项原生检查。[GSM8K 20 题](DSPARK_GSM8K20_20261008.md) 中，基线和 DSpark 均为 19/20（95%），错题相同；GPU 已释放。长输出分叉和性能记录见 [双卡开发验证](DSPARK_GPU_VALIDATION_20261008.md)。下文保留静态调查阶段的证据边界。

2026-10-08。DSpark 权重已下载、校验并解包到原有模型目录，继续使用用户指定的本地 `1008` 标注。该包是 **5 层 DSpark 草稿模型**，需要搭配 MiniCPM 主模型使用。当前 vLLM 已有原生 Qwen3 DSpark 路径，检查出的张量与主模型结构匹配；**必须使用 V2 runner，XPU 端到端兼容性和性能尚未验证**。本轮仅下载、读取源码和执行 CPU 配置结构检查，没有使用 GPU。

## 下载结果

模型目录：[MiniCPM-V-4.7-35B-A3B-DSpark-1008](/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-DSpark-1008)。目录包含 `config.json`、`model.safetensors`、原始 tar 和下载校验记录。

| 项目 | 结果 |
| --- | --- |
| 用户提供的来源 | [MiniCPM-V-4.7-35B-A3B-DSpark.tar](https://data-transfer-huawei.obs.cn-north-4.myhuaweicloud.com/caitianchi/v47/1006/release/MiniCPM-V-4.7-35B-A3B-DSpark.tar) |
| 来源路径与本地标注 | 来源仍是 `1006/release`；本地标注 `1008`，不改写来源版本 |
| OBS Last-Modified | 2026-10-08 06:10:50 UTC |
| tar 大小 | 778,465,280 bytes，约 742.4 MiB |
| ETag | `4c4fc75d4e7b334a484245ee08a65184-8` |
| OBS CRC64 与本地完整对象 CRC64 | 均为 `666953002590505572`，匹配 |
| tar SHA-256 | `6e8d85e0bef394da3f76a237899e5cc1a08519e3a6f2ecaa55fe474e2e3467ae` |
| safetensors | 1 个文件，62 个 BF16 张量，389,223,169 个存储参数 |
| 完成时间 | 2026-10-08 09:45:01 UTC，即北京时间 17:45:01 |

下载完成后检查了 tar 路径、文件类型和覆盖冲突，再解包；safetensors 的 shape、dtype、字节范围和文件长度一致。完整下载记录见 [download-manifest.json](dspark-20261008/download-manifest.json)，张量头见 [safetensors-headers.json](dspark-20261008/safetensors-headers.json)。旧主模型和已下载的主模型 `1008` 目录均保留。

按用户后续要求，主模型 `1008` 与 DSpark `1008` 两个目录及其文件均调整为 `intel:intel`，目录模式 `775`、普通文件 `664`。xiangyu 不属于 intel 组，因此另加 xiangyu 的读写 ACL；目录默认 ACL 同时允许 intel 与 xiangyu 编辑后续新建文件。无设备挂载的权限辅助容器分别以 UID 1000 和 UID 1002 验证了全部文件的读写权限，并实际检查双方新建文件后的交叉编辑；临时检查文件已删除。详见 [权限检查结果](dspark-20261008/permissions-validation.log)。旧 `-hf` 目录未修改。

## 源码基线

实际阅读的本地整合树是 `vllm-functional`，分支 `dev/minicpmv47-main-functional-20261008`，基于 `95307ac6d4cd8be2c1e38f236e9104c569af9025`。该树已有本任务的 #58674、#60539 和 BF16 补丁；本轮没有修改这些源码或 kernels，也没有提交和推送。

本轮另查 upstream main，固定到 `7816afdb259dbb18f9e98d11b05194dd3446d0fa`，提交时间为 2026-10-08 09:33:41 UTC。对本报告涉及的 speculative 配置、Qwen3 DSpark/DFlash 模型、runner、graph、RoPE、XPU 后端和 DSpark E2E 测试路径进行比较，只有 `vllm/config/speculative.py` 将 `Self` 从 `typing_extensions` 改为从 `typing` 导入；这些路径的行为没有变化。此比较不代表整个 main 已重新集成。

## 权重与主模型的结构关系

| 项目 | DSpark 包 | MiniCPM 主模型及代码匹配 |
| --- | --- | --- |
| 原生架构 | `Qwen3DSparkModel`，`model_type=qwen3` | registry 映射到 `Qwen3DSparkForCausalLM` |
| 草稿骨干 | 5 层 full attention，8 个 Q head、2 个 KV head，head dim 256 | TP=2 时 Q/KV head 均可整除；不同于主模型 30 层 GDN、10 层 full attention |
| hidden size / 词表 | 2048 / 248144 | 旧主模型与主模型 1008 均一致 |
| 目标辅助层 | `[3,11,19,27,35]` | 主模型 40 层中这五层均为 full attention |
| 辅助特征投影 | `fc.weight=[2048,10240]` | 五层各 2048 维拼接，输入宽度 10240；capture 索引按代码转换为 `(4,12,20,28,36)` |
| embedding / lm_head | 包内没有这两组权重 | loader 支持从 target 的 language model 共享；等词表、等 hidden size 满足结构条件 |
| Markov head | 两组 `[248144,256]` BF16 权重 | 原生 `DSparkMarkovHead` 支持完整词表，两个权重在各 TP rank 复制 |
| confidence head | 权重 `[1,2304]`，bias `[1]` | 2048 hidden + 256 Markov embedding；代码将此 head 建为 FP32，权重加载时转换 dtype |
| block size | 15 | 未指定预测数时从 config 推导 `num_speculative_tokens=15` |
| 草稿 RoPE | `rope_dim=256`，section `[44,44,40]` | 显式 rope_dim 优先于 partial factor，section 和为 128，满足 rotary_dim/2 |

草稿 RoPE 与主模型的 rotary dim 64 不同，这不构成代码中的尺寸冲突：两个模型分别创建位置编码。归档中的 `auto_map` 指向 `dspark.DSparkDraftModel`，但不含 `dspark.py`；vLLM 的原生架构注册可处理这个包，直接用 Transformers `AutoModel` 加载自定义代码则是另一条路径。

[CPU 结构检查](dspark-20261008/static-contract-check.json) 通过。检查从源码 AST 提取两个纯配置函数，验证辅助层索引与 FC 输入宽度，并检查配置及张量头；未导入 Torch，未加载模型，也未执行完整 `VllmConfig` 初始化。尺寸匹配不能证明该草稿对旧主模型权重的训练配对或接受率。

## 当前支持路径与待验证边界

| 路径 | 静态结论 | 影响 |
| --- | --- | --- |
| runner | V1 的 unsupported feature 列表明确包含 DSpark；V2 factory 可创建 `DSparkSpeculator`，XPU worker 可选择 `XPUModelRunnerV2` | 之前基准的 `VLLM_USE_V2_MODEL_RUNNER=0` 不能直接沿用，需改为 `1` |
| MiniCPM 辅助 hidden states | 当前整合的 #58674 通过 language model 委托提供 EAGLE3 接口；V2 为 DSpark 开启辅助状态输出 | 有五层特征接入路径，仍需实际验证捕获顺序和加载 |
| 草稿 attention | DSpark 复用 DFlash，默认五层均为 non-causal；FlashAttentionBackend 声明支持 non-causal | 未发现该路径的 XPU 平台硬性禁用，但具体 XPU kernel 组合仍需执行验证 |
| 主模型 GDN 验证 | V2 hybrid state 传递 accepted-token 数和 draft-token 数；XPU GDN wrapper 与原生 kernel 有 speculative state 路径 | 有验证和状态更新代码；15-token DSpark 与 MiniCPM 组合的拒绝、回滚及续写尚未验证 |
| 文本及多模态位置 | DFlash/DSpark 使用一维逻辑 context/query positions；MRotaryEmbedding 接受一维输入，主模型另有多模态位置路径 | 文本的维度接口可接入；图片、视频、canvas 位置和语义不能由此宣告已支持 |
| graph | 草稿继承 DFlash graph 策略：不捕获 PIECEWISE；target decode 为 FULL 且 draft attention 满足要求时尝试 FULL_DECODE_ONLY，否则草稿 eager | 之前的 PIECEWISE 成功只覆盖普通主模型 decode；不代表 DSpark 草稿已捕获 |
| 量化 | draft 使用自己的 quantization 配置，默认 `None`；dtype 来自 target | 主模型 online FP8 不会自动量化这份 BF16 草稿，需分开记录两者配置 |
| confidence | checkpoint 含 confidence head；运行时仅在 `enable_adaptive_verification=true` 时使用 | 加载 head 不等于开启 adaptive verification，默认关闭 |
| upstream 测试 | DSpark E2E 已覆盖部分 Qwen、Gemma 模型的接受率及准确率 | 没有 MiniCPM 4.7 + XPU + 该 checkpoint 的现成验证证据 |

核心本地入口为 [qwen3_dspark.py](vllm-functional/vllm/model_executor/models/qwen3_dspark.py)、[DSparkSpeculator](vllm-functional/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)、[draft loader](vllm-functional/vllm/v1/worker/gpu/spec_decode/dspark/utils.py)、[V2 runner](vllm-functional/vllm/v1/worker/gpu/model_runner.py)、[XPU adapter](vllm-functional/vllm/v1/worker/xpu_model_runner.py) 和 [GDN interface](kernels-bf16/csrc/xpu/gdn_attn/gdn_attn_interface.cpp)。算法说明见 [upstream DSpark 文档](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md)。

## 与现有性能优化的关系

DSpark 每轮增加一次草稿骨干和从左到右的 Markov 采样，主模型批量验证多个候选，收益取决于接受的 token 数是否抵消额外工作。当前没有 DSpark TPOT、TTFT 或接受率测量。

默认 15 个候选使单请求主模型验证 query length 为 **16**。当前 #60539 的 MoE、norm、投影配对、QKV 和 all-reduce pass 多数只匹配 compile range 终点不超过 **8** 的区域；原生 FP8 GEMV 快路径又要求 **M=1**。因此完整 16 行验证通常不满足这些快路径条件，会使用其他实现。拒绝后缩短的批次能否命中，需要实际 trace；不能把此前普通单 token decode 的收益直接套到 DSpark。

草稿本身是 dense attention/MLP，不满足 MiniCPM 专用 MoE 融合结构，默认也没有 FP8 量化。完整词表 Markov W2 单份 BF16 有 127,049,728 bytes，且各 rank 复制；当前顺序采样每个预测位置都计算 Markov bias。block 15 会增加顺序阶段的工作，不保证相比小 block 更快。

已有 [Graph 性能记录](XPU_GRAPH_EXPERIMENT_20261008.md) 中，普通 decode 的 NONE / PIECEWISE TPOT 为 17.838 / 11.963 ms/token，使用的是 V1 runner、旧主模型和本任务 BF16 overlay。DSpark 强制 V2，因此后续对照应先取得 **同一 V2 runner、同一主模型权重** 的非 speculative 基线，再比较 DSpark；上述数值只能作为历史参考。

## 后续功能验证配置要点

下面是由源码推导的配置片段，尚未启动服务。主模型应另指定其完整目录，DSpark 目录用于 `speculative_config.model`。

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

```json
{
  "method": "dspark",
  "model": "/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-DSpark-1008",
  "num_speculative_tokens": 15,
  "draft_tensor_parallel_size": 2,
  "enable_adaptive_verification": false
}
```

首轮可先采用 graph NONE、关闭 prefix caching，检查权重加载、短文本、长上下文、拒绝后 GDN 状态与继续生成；之后再分别检查 adaptive verification、graph 和多模态。草稿继续保留训练 block 15，改变 block 长度属于另一项需要验证接受率的实验。当前没有改动 checkpoint 配置，没有占用 GPU。
