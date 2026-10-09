# MiniCPM V4.7 DSpark 双卡开发验证

> 2026-10-08 后续性能更新：用户已认可 GSM8K 得分并继续授权卡 0、1。编译融合与 Graph 模式的新测量、源码修复和本轮释放记录统一见 [独立性能文档](DSPARK_PERFORMANCE_20261008.md)。下文保留先前 eager 实验与当时的设备释放记录。

> 2026-10-08 trace 修正：下文“主模型 causal attention 继续使用原装实现”的覆盖范围判断已被后续 trace 修正；适配也覆盖主模型 speculative 验证展开后的单 query 调用。原数据和文字保留，新证据见 [attention 覆盖范围修正](DSPARK_PERFORMANCE_20261008.md#2026-10-08-attention-覆盖范围修正)。

> 2026-10-08 GSM8K 更新：按用户要求，以评测得分确认正确性。基线和 DSpark 的 [20 题串行对照](DSPARK_GSM8K20_20261008.md) 均为 19/20（95%），错题相同，无截断；GPU 0、1 已释放。下文的 token 分叉保留为诊断证据。

2026-10-08。本轮使用物理 GPU 0、1，搭配主模型 1008 和 DSpark 草稿 1008。DSpark 已运行，原装镜像缺失的 non-causal attention 配置已通过局部 native 覆盖补齐，24 项数值检查通过。2K/256 场景 TPOT 中位数从基线 50.973 降为 DSpark 35.134 ms/token。长输出的分叉作为数值诊断保留，正确性按后续 GSM8K 得分确认。此前的静态结论见 [DSpark 支持调查](DSPARK_SUPPORT_REVIEW_20261008.md)。

## 设备与环境

| 物理卡 | PCI | Sysman UUID | Torch 逻辑卡 |
| --- | --- | --- | --- |
| 0，Arc Pro B70 / 0xE223 | `0000:18:00.0` | `00000000-0000-0018-0000-0000e2238086` | 0 |
| 1，Arc Pro B70 / 0xE223 | `0000:36:00.0` | `00000000-0000-0036-0000-0000e2238086` | 1 |

启动前两张卡仅有 xpu-smi 查询进程。容器 `codex-minicpm47-dspark-gpu01-20261008` 的完整 ID 为 `ac93ad8d263c5c2275a38160a25156d9b1b51588f20a79adf60026b22f5fb6fc`，使用镜像 `sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566`，基于 upstream v0.31.0。执行通过 `ZE_AFFINITY_MASK=0,1` 限定到上述两卡，Torch UUID 与 Sysman UUID 已核对。主、草稿模型均只读挂载。[设备映射](dspark-gpu01-20261008/torch-identity.json)、[容器创建记录](dspark-gpu01-20261008/container-created.json)

源码使用 `vllm-functional`，分支 `dev/minicpmv47-main-functional-20261008`，基于 `95307ac6d4cd8be2c1e38f236e9104c569af9025`，加本任务已整合的 #58674、#60539、BF16 补丁。首轮 eager 使用镜像原装 kernels，没有加载 BF16 原生性能 overlay；不据此声称运行了完整 #622 wheel。

## 当前实验状态

| 模式 | 请求和 token 对照诊断 | TTFT 中位数 ms | TPOT 中位数 ms/token | 总时延中位数 ms |
| --- | --- | --- | --- | --- |
| V2，非 speculative，eager，graph NONE | 6/6 通过 | 160.694 | 50.973 | 13158.963 |
| V2，DSpark 15，eager，graph NONE | 6 项 HTTP 200；5/6 token 对照一致 | — | — | — |
| V2，DSpark 15，补齐原生 attention，graph NONE | 5/6 的 64-token 对照一致；256-token 长输出有分叉 | 160.922 | 35.134 | 9119.993 |

两组共同配置为 BF16 激活、主模型 online FP8、TP=2、prefix caching 关闭、并发 1；max model len 8704、max batched tokens 4096、显存占用比例 0.80。草稿维持 BF16、TP=2、15-token block、adaptive verification 关闭。[固定实验参数](dspark-gpu01-20261008/PLAN.json)

基线三个短答案为 `408`、`北京`、`Apple`；512/2048/8192 上下文均成功生成 64 token。[基线功能结果](dspark-gpu01-20261008/baseline_eager-correctness.json)

时延场景为 2048 输入 / 256 输出，两次预热和三次测量；[基线逐样本](dspark-gpu01-20261008/baseline_eager-samples.json)、[基线摘要](dspark-gpu01-20261008/baseline_eager-summary.json)。

DSpark 的三次有效测量接受率均为 5.1389%，平均接受长度均为 1.7708（含 bonus）：每请求 144 轮验证、2160 个候选、111 个接受 token。TPOT 比基线低 31.07%，总时延低 30.69%；这是当前配置下的开发性能对照。256-token 两组文本不同，按用户明确的要求不将此作为硬门禁，正确性另按 GSM8K 数值答案评分。[DSpark 逐样本](dspark-gpu01-20261008/dspark_native-samples.json)、[时延及输出对照](dspark-gpu01-20261008/latency-comparison.json)

DSpark 首轮日志确认 `(4,12,20,28,36)` 五个辅助层已启用，加载含 confidence head 的草稿，额外使用主模型 embedding/lm_head。六项请求累计 112 轮 draft、1680 个候选、87 个接受 token；接受率 5.18%，含 bonus 的平均接受长度 1.78。它们是混合短请求及上下文检查的统计，不代表性能场景的接受率。[功能结果](dspark-gpu01-20261008/dspark_eager-correctness.json)、[日志](dspark-gpu01-20261008/server-dspark_eager.log)、[metrics](dspark-gpu01-20261008/dspark-eager-functional-metrics.txt)

8K 首次分叉在输出第 23 个索引位置：基线接续逗号并解释截断内容，DSpark 接续句号并给出另一段说明。首轮另有明确的配置缺口：原装 attention module 未编译 `256,true,false,false,false,false`（paged non-causal、全局窗口、无 sink/LSE），运行时回退到 PyTorch reference。该轮没有采集 DSpark 时延；补齐后再记录上表的开发时延和分叉诊断。

## 原生 attention 补齐

采用 kernels 已支持的自定义编译配置，在无设备挂载的 `codex-minicpm47-bf16-build-20261008` 构建两项：chunk prefill `256,true,false,false,false,false`，paged decode `8,256,64,false,false,false`。mixed 路径先运行 prefill，再调用 masked decode，因此两项都必须存在，即使本批查询均为 15-token 草稿。

产物使用独立 `_minicpm_dspark_fa2` op namespace 和独立共享库 SONAME，避免覆盖镜像原装 attention module。运行适配仅将 head-dim 256、paged、non-causal、无窗口/sink/LSE 的调用交给新增模块；主模型 causal attention 继续使用原装实现。它是本任务的局部运行覆盖，没有安装完整 kernels wheel 或生成新的运行镜像。[适配代码](dspark-gpu01-20261008/dspark_attention.py)、[编译配置](dspark-gpu01-20261008/chunk-prefill-dspark.conf)、[构建脚本](dspark-gpu01-20261008/build-dspark-attention.sh)

| 产物 | SHA-256 |
| --- | --- |
| `_minicpm_dspark_fa2.abi3.so` | `3025d2cc09c83eed580ea22fbcf6e523842bb2ac9793b6847ae9f6d45fbd385e` |
| `libminicpm_dspark_attn_kernels.so` | `60b962f71c686b8763377fdd3daf92d227143e03cf18452e31db38c8173f5a5c` |

以仓库已有 paged-attention 测试及其参考函数为基础，检查 BF16/FP16、query 7/15/16、KV 32/512/2048/8192、GQA 4:1、head 256、block 64，共 12/12 项通过。容差沿用已有测试的 atol=0.02、rtol=0.01；同时要求原生调用成功且没有新增 fallback warning。[检查结果](dspark-gpu01-20261008/native-attention-results.json)、[日志](dspark-gpu01-20261008/native-attention-tests.log)

实际模型日志中的 cache shape 为 `[390,704,1,256]`，query 为 `[15,4,256]`，两个 TP rank 均进入原生路径，没有缺失配置的回退。物理 page size 是 704；kernel 的 64 配置对应 dispatch 的 KV tile bucket。按实际 page size 704 补测相同矩阵，又有 12/12 项通过。[704 检查结果](dspark-gpu01-20261008/native-attention-block704-results.json)

补齐原生 attention 后，五项 64-token 输出仍与基线一致，8K 仍从 index 23 分叉；三次 32-token 诊断输出相同。DSpark 下该位置句号与逗号的 logprob 均为 `-0.6982667446`。重新启动基线后三次输出也相同，逗号 logprob 为 `-0.5800447464`，句号为 `-0.8300447464`，仍选择原基线的逗号。候选排序对计算模式变化敏感，具体来自 GDN state、attention 或矩阵计算的哪一层尚未隔离，不能只归因于 all-reduce 或宣告状态完全等价。[DSpark 诊断](dspark-gpu01-20261008/dspark_native-8192-diagnostics.json)、[基线复测](dspark-gpu01-20261008/baseline_recheck-8192-diagnostics.json)

2K 的前 64 token 一致，128-token 诊断确认首次分叉为 index 71（文本字符索引 119）：基线选择“接收” token 108843，logprob=-0.8417034，另一候选“收” token 96150 为 -0.9667034；DSpark 下两者均为 -0.9218878，选择较小 ID 的“收”，随后生成“收悉”。各模式三次诊断输出都稳定。两个分叉点均表现出候选排序的数值敏感性，当前没有隔离到具体内核；按用户要求保留为诊断，不作为评测硬门禁。[首次分叉对照](dspark-gpu01-20261008/2048-first-difference-diagnostics.json)

最初仅编译 chunk tuple 的检查触发了缺失 paged tuple 的回退，未计作原生通过；该次记录保留在 `native-attention-attempt1*`。CPU 构建失败及修正日志也保留，最终成功构建见 [attention-build-attempt4.log](dspark-gpu01-20261008/attention-build-attempt4.log)。

功能检查包括三个短答案，以及 512、2048、8192 token 上下文各生成 64 token，temperature=0、seed=7，与本轮 V2 基线比较 token IDs。性能属于开发证据，每组两次预热加三次测量，保留逐样本 TTFT、TPOT、总时延和 speculative metrics。其设备、权重、runner、输入长度与此前 V1 32K/512 记录不同，不能直接作为历史优化收益比较。

## 后续 roofline 和 batch 调优参考

用户提供的 [PR6 v0.31 batch roofline 文档](/home/intel/xiangyu/tmp/BATCH_ROOFLINE_PR6_V031.md) 使用 Qwen3.6、FP16、GPU 1/2、32K/256、prefix cache 命中的纯 decode，以及 NONE / FULL_DECODE_ONLY 的 B=1/2/4/8 对照。短请求预热、检查真实 graph/compile cache 命中、记录 preemption 和分 batch 比较的方法可用于后续实验。此处保留当前 2048/256、prefix cache 关闭的三次测量结果，不替换为该文档的数值。

DSpark 15 的主模型验证有 16 个位置，但它们属于同一序列，不能直接代入文档中 16 个独立序列的 GDN 状态与 paged KV 模型。后续成本模型还需包含草稿 dense 层、Markov 全词表投影、双卡通信、实际接受长度及 target 验证路径。文档的 batch 带宽效率和同步 penalty 是依据其已有测量校准的参数，MiniCPM 与 DSpark 需要自己的 trace 和形状证据。

## GPU 释放状态

本轮时延与数值诊断结束后，GPU 已于 2026-10-08 11:06:45 UTC 释放，两个 worker 均完成清理，卡 0、1 仅有查询进程。[卡 0](dspark-gpu01-20261008/gpu0-after.json)、[卡 1](dspark-gpu01-20261008/gpu1-after.json)。主动关闭 engine 后 output handler 记录了 EngineDeadError，该日志在全部请求结束后，不属于测量阶段的请求失败。

随后用户要求 GSM8K 20 题，已重新进行设备空闲检查并使用同一授权的卡 0、1；该后续阶段的最终释放状态见 [GSM8K 评测文档](DSPARK_GSM8K20_20261008.md)。
