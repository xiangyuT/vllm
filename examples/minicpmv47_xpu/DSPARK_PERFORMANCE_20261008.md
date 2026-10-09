# MiniCPM V4.7 DSpark 编译和 XPU Graph 性能收集

## 2026-10-09 接受率定位：Chat template、短 GSM8K 与上下文长度

**本机标准 chat GSM8K20 已重新达到 78.26% 接受率、19/20 得分；32K 文学续写加 chat template 后仍只有 8.96%。** 模板会影响接受率与评分，但本次结果没有支持“缺模板就是 32K 低接受率的全部原因”。这轮保持 1008 配对、主模型 online FP8/BF16 激活、草稿 BF16、k3 FULL、TP2、prefix cache 关闭、temperature=0 和 seed=7；没有修改模型/kernel 源码、翻转采样布局或复用评测响应。[冻结参数](dspark-32k-performance-20261009/chat-template-probe/PLAN.json)、[完整汇总](dspark-32k-performance-20261009/chat-template-probe/summary.json)

| 本轮短题测试 | 输入 token 范围 | 接受率 accepted/proposed | 平均接受长度 | GSM8K 得分 |
| --- | --- | --- | --- | --- |
| 同 20 题、5-shot、原始 completion | 702–787 | **67.44%** | 3.023 | 18/20 |
| 同 20 题、5-shot、主模型 chat template | 714–799 | **78.26%** | 3.348 | **19/20** |

GSM 使用自然 EOS、max_tokens=1024、min_tokens=0、thinking 关闭、串行执行；两组统一采用上游 GSM evaluator 的 stop 列表 `Question / Assistant: / <|separator|>`，均无长度截断。Chat 共提出 3,423 个草稿 token、接受 2,679 个、1,141 轮；raw 共提出 2,838、接受 1,914、946 轮。指标按累计计数加权，未将各题接受率简单平均。这里沿用 API 的 `AL=1+accepted/steps` 口径，末端停止裁剪可使其与实际 completion_tokens/steps 略有区别。[Chat 全部新响应](dspark-32k-performance-20261009/chat-template-probe/gsm-chat-results.json)、[Raw 全部新响应](dspark-32k-performance-20261009/chat-template-probe/gsm-raw-results.json)

对长文本增加了指令与模板的分离对照。三组均保留相同的 32,768-token 正文，统一固定生成 256 token、ignore_eos=true；全部通过 completion 接口传入 token ID。第三组 token ID 来自运行服务实际 `/tokenize(messages)` 的 chat 渲染，避免把接口/输出解析器差异混作模板效果。各组一次预热、三次有效响应，不报告时延收益。

| 同一 32K 正文的包装 | 实际输入 token | 接受率 | 平均接受长度 |
| --- | --- | --- | --- |
| 原始正文，无指令/模板 | 32,768 | 7.93% | 1.238 |
| 相同正文 + 明确续写指令，无模板 | 32,790 | 12.30% | 1.369 |
| 相同正文和指令 + 主模型 chat template | 32,802 | 8.96% | 1.269 |

Chat 参数为 add_generation_prompt=true、enable_thinking=false。正文没有为了容纳包装而额外截短；所以本表的“32K”是正文长度，而非包装后总长度。[服务端实际渲染 token ID](dspark-32k-performance-20261009/chat-template-probe/tokenize-responses.json)、[三组原始响应](dspark-32k-performance-20261009/chat-template-probe/summary.json)

又固定原文结束位置 32,768，只改变可见正文窗口；同样的 chat 续写指令、256 固定输出，各组预热后保留三次响应：

| 可见正文窗口 | 原文 token 区间（右端不包含） | 包装后输入 token | 接受率 | 平均接受长度 |
| --- | --- | --- | --- | --- |
| 最近 2K | [30,720, 32,768) | 2,082 | **18.70%** | 1.561 |
| 最近 8K | [24,576, 32,768) | 8,226 | **13.89%** | 1.417 |
| 完整 32K | [0, 32,768) | 32,802 | **8.96%** | 1.269 |

32K 行明确复用上一阶段三次原始响应作为参照，没有重复计成新的测量。这个对照支持长度/历史内容影响接受率，但目标模型和草稿都看到了更短的历史，绝对位置和生成内容也可能变化；**它不是只限制草稿 attention 的实验，不能据此宣称 draft-only 4K window 已修复问题**。[窗口参数和数据](dspark-32k-performance-20261009/suffix-context-probe/summary.json)

### 用户指定 subagent 的社区原始资料调查

| 社区模型与条件 | 公开结果 | 口径/限制 |
| --- | --- | --- |
| DeepSeek 官方 Qwen3-8B DSpark，k7、T1、non-thinking、固定 block、禁用 confidence scheduler | GSM8K AL=6.17；聊天 AL=3.21–3.72 | 作者 AL 含 bonus；固定 7 proposals、1 bonus 且忽略末端裁剪时，约对应 GSM 73.9%、聊天 31.6%–38.9% |
| OpenBMB MiniCPM5-2B DSpark，k7、T0、BF16、自然 EOS | Math AL=6.0496、Code 6.1106、General 4.1585 | 不同于本任务 MiniCPM-V4.7；只直接报告 AL，不把它当百分比 |
| Dogacel/Qwen3-8B-DSpark，k7、T1、SPEED-Bench coding、单 H100、Chat API | AL=2.33，接受率18.9% | 作者报告自己的训练未完成；不是本任务模型的预期值 |

原始来源：[DSpark 作者论文 §4](https://arxiv.org/html/2607.05147v1#S4)、[OpenBMB 模型卡](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark#evaluation)、[Dogacel 模型卡与命令](https://huggingface.co/Dogacel/Qwen3-8B-DSpark#benchmark)。本机短 GSM 的 70% 目标已有实测支持；这些社区结果也显示任务、模型配对、草稿长度和采样条件会影响结果，不能直接移植另一模型的接受率。

社区另有两个诊断线索：[#56792](https://github.com/vllm-project/vllm/issues/56792) 报告某个未具名 multimodal-OCR checkpoint 的训练布局为 fill-in，修正 sample_from_anchor 后接受率上升。但 [DeepSeek 官方 Qwen3-8B config](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7/blob/main/config.json) 同样缺此字段，论文却采用 anchor-first，因此本机字段缺失不证明当前 True 错误。本轮保持原布局。另一个 [mlx-dspark 作者记录](https://github.com/ARahim3/mlx-dspark/blob/main/CHANGELOG.md#0200--2026-09-28--speculation-holds-at-long-context-185-at-32k-gemma-4-12b-325-red-hats-qwen38-dspark-head) 在其他模型上限制 drafter 最近 4096 行的 cross-attention、保留完整 target 上下文后改善了长上下文表现；没有成对接受率表，仍只是本任务下一步受控实验的线索。[结构化社区结果与限制](dspark-32k-performance-20261009/chat-template-probe/community-findings.json)

本轮累计完成 40 次新 GSM 评测、9 次包装对照、6 次新短窗口响应；预热和历史 32K 参照分开记录。已核查逐轮 accepted/proposed、histogram、评分、输入来源和源码散列；没有新增运行代码或 kernel 修改，`vllm-functional` 的 26 个既有整合文件保持原散列。[核验](dspark-32k-performance-20261009/chat-template-probe/validation.json)。两轮诊断容器均已停止、端口关闭，root Sysman 查询确认 GPU 0/1 无进程；社区 subagent 只做网络/源码阅读，没有使用 GPU。[模板阶段释放](dspark-32k-performance-20261009/chat-template-probe/release.json)、[窗口阶段释放](dspark-32k-performance-20261009/suffix-context-probe/release.json)

> 2026-10-09 低接受率调查：#58674 当前 head 为 `e578cb0e5277cda4e38d9306dfb5ed0c59440616`，本机停在 `4f5977e4c23c55f1d10b12a687443bbb262a27eb`；漏补集中在多模态参数读取、canvas metadata 缓存保留及其测试。DSpark/DFlash、Qwen3.5 和 draft graph 的七个关键文件与当前 PR head 逐字节一致，未发现这两个追加提交带来能解释纯文本低接受率的核心行为修复。补丁仅做应用检查，未改运行源码、未使用 GPU。[核查结果](pr58674-update-audit-20261009/summary.json)、[待补三文件 diff](pr58674-update-audit-20261009/fix-ci.patch)。权重挂载仍指向本地标注 1008 的主模型及 DSpark 草稿，具体来源与配置身份见 [权重核查](pr58674-update-audit-20261009/weights.json)；接受率问题仍未定位。

<!-- real-text-32k-fixed256-performance:start -->
## 2026-10-09 真实正文 32K / 固定 256 token 性能测试

本轮按用户指定，截取《战争与和平》正文开头的 **32,768 个主模型 tokenizer token**，使用原始 completion 续写，按用户要求忽略 EOS 并固定生成 256 token。直接传入 token ID，服务端输入精确为 32,768 token，没有 chat template、重复填充或正文之外的续写指令。[固定输入/参数](dspark-32k-performance-20261009/fixed256/PLAN.json)、[正文前缀](dspark-32k-performance-20261009/fixed256/prompt.txt)

| 配置 | TTFT 中位数 ms | 平均 decode TPOT 中位数 ms/token | 总完成时间中位数 ms | 草稿接受率 | 平均接受长度 |
| --- | --- | --- | --- | --- | --- |
| 普通 FULL，无 DSpark | 2815.433 | 12.345 | 5962.533 | — | — |
| DSpark k3 FULL，BF16 草稿 | 3045.710 | 25.227 | 9480.524 | 7.93% | 1.238 |
| DSpark k3 FULL，FP8 Markov/草稿词表 | 3033.914 | 23.593 | 9050.136 | 10.15% | 1.305 |

各组先完成四次短前缀预热、两次完整 32K 前缀预热，再采三次有效样本，保留全部结果，不删除较慢样本。三组统一 `min_tokens=max_tokens=256`、`ignore_eos=true`，temperature=0、seed=7；屏蔽提前 EOS，实际输出长度必须为 256 token。Prefix cache 关闭，单并发，TP2，物理 GPU 0/1，V2，BF16 激活、主模型 online FP8、FULL。为输入和生成保留空间，`max_model_len=33280`；每轮最多 4096 个 batched token，32K prefill 分块执行。模型加载、编译、capture 和预热均不计入有效样本。

TTFT 从客户端发请求到首个带生成 token ID 的 SSE chunk；文本首 token 时间另行保存。Decode TPOT 为 `(DONE 时间−首个生成 token chunk 时间)/(实际输出 token 数−首 chunk token 数)`，是包含客户端/服务流式开销的平均指标；DSpark 一个 chunk 可含多个 token，不能把 chunk 间隔当逐 token ITL。总完成时间直接按请求起止测量，不用各项中位数相加。

普通 FULL 与两组 DSpark 都保留当前 #60539/BF16 source/native 覆盖；DSpark 同时使用既有 native non-causal paged attention。两组 DSpark 都是 k3，差别仅为草稿 Markov/词表权重格式和投影后端。三组均不在测量期间启用 profiler、路由 snapshot 或额外 GPU event。此为单份正文、单个截断位置的开发对照，不是完整数据集或 formal acceptance。

- 普通 FULL，无 DSpark：三个有效请求的输出 token 数为 256/256/256；[全部样本](dspark-32k-performance-20261009/fixed256/baseline_FULL/samples.json)、[实际启动参数](dspark-32k-performance-20261009/fixed256/baseline_FULL/launch.json)。两个 rank 的实际 XPUGraph replay 已核验。
- DSpark k3 FULL，BF16 草稿：三个有效请求的输出 token 数为 256/256/256；[全部样本](dspark-32k-performance-20261009/fixed256/k3_bf16_FULL/samples.json)、[实际启动参数](dspark-32k-performance-20261009/fixed256/k3_bf16_FULL/launch.json)。两个 rank 的实际 XPUGraph replay 已核验。
- DSpark k3 FULL，FP8 Markov/草稿词表：三个有效请求的输出 token 数为 256/256/256；[全部样本](dspark-32k-performance-20261009/fixed256/k3_heads_fp8_FULL/samples.json)、[实际启动参数](dspark-32k-performance-20261009/fixed256/k3_heads_fp8_FULL/launch.json)。两个 rank 的实际 XPUGraph replay 已核验。

相对普通 FULL，正值表示时延下降、负值表示上升：
- DSpark k3 FULL，BF16 草稿：decode TPOT 降幅 -104.34%，总完成时间降幅 -59.00%，TTFT 降幅 -8.18%。
- DSpark k3 FULL，FP8 Markov/草稿词表：decode TPOT 降幅 -91.11%，总完成时间降幅 -51.78%，TTFT 降幅 -7.76%。
- FP8 Markov/草稿词表相对本轮 k3 BF16：decode TPOT 降幅 6.48%，总完成时间降幅 4.54%。这些比较不沿用旧合成负载的收益百分比。[完整比较](dspark-32k-performance-20261009/fixed256/comparison.json)

本轮推理容器已停止、31065 端口已关闭；GPU 0/1 的 root Sysman 查询结果保存在 [释放记录](dspark-32k-performance-20261009/fixed256/release.json)。未停止其他容器或重置设备。运行仓库和 native kernel 没有新增修改，未创建提交、变更版本或写远程仓库。[源码/镜像身份](dspark-32k-performance-20261009/fixed256/source-identity.json)

<!-- real-text-32k-fixed256-performance:end -->
<!-- real-text-32k-performance:start -->
## 2026-10-09 真实正文 32K 自然 EOS 诊断（历史）

此前自然 EOS 请求已完成，但 DSpark 输出 31 token、普通 FULL 输出 256 token，完整请求总耗时不能作为同等输出长度的加速比较。用户已要求忽略 EOS，最新固定 256 token 对照见上节。

本轮按用户指定，截取《战争与和平》正文开头的 **32,768 个主模型 tokenizer token**，使用原始 completion 作自然续写。直接传入 token ID，服务端输入精确为 32,768 token，没有 chat template、重复填充或正文之外的续写指令。[固定输入/参数](dspark-32k-performance-20261009/PLAN.json)、[正文前缀](dspark-32k-performance-20261009/prompt.txt)

| 配置 | TTFT 中位数 ms | 平均 decode TPOT 中位数 ms/token | 总完成时间中位数 ms | 草稿接受率 | 平均接受长度 |
| --- | --- | --- | --- | --- | --- |
| 普通 FULL，无 DSpark | 2794.347 | 11.590 | 5750.404 | — | — |
| DSpark k3 FULL，BF16 草稿 | 2985.498 | 26.689 | 3786.009 | 6.17% | 1.185 |
| DSpark k3 FULL，FP8 Markov/草稿词表 | 3033.660 | 26.948 | 3842.113 | 6.17% | 1.185 |

各组先完成四次短前缀预热、两次完整 32K 前缀预热，再采三次有效样本，保留全部结果，不删除较慢样本。输出最多 256 token，temperature=0、seed=7、`ignore_eos=false`，不设最小输出长度。Prefix cache 关闭，单并发，TP2，物理 GPU 0/1，V2，BF16 激活、主模型 online FP8、FULL。为输入和生成保留空间，`max_model_len=33280`；每轮最多 4096 个 batched token，32K prefill 分块执行。模型加载、编译、capture 和预热均不计入有效样本。

TTFT 从客户端发请求到首个带生成 token ID 的 SSE chunk；文本首 token 时间另行保存。Decode TPOT 为 `(DONE 时间−首个生成 token chunk 时间)/(实际输出 token 数−首 chunk token 数)`，是包含客户端/服务流式开销的平均指标；DSpark 一个 chunk 可含多个 token，不能把 chunk 间隔当逐 token ITL。总完成时间直接按请求起止测量，不用各项中位数相加。

普通 FULL 与两组 DSpark 都保留当前 #60539/BF16 source/native 覆盖；DSpark 同时使用既有 native non-causal paged attention。两组 DSpark 都是 k3，差别仅为草稿 Markov/词表权重格式和投影后端。三组均不在测量期间启用 profiler、路由 snapshot 或额外 GPU event。此为单份正文、单个截断位置的开发对照，不是完整数据集或 formal acceptance。

- 普通 FULL，无 DSpark：三个有效请求的输出 token 数为 256/256/256；[全部样本](dspark-32k-performance-20261009/baseline_FULL/samples.json)、[实际启动参数](dspark-32k-performance-20261009/baseline_FULL/launch.json)。两个 rank 的实际 XPUGraph replay 已核验。
- DSpark k3 FULL，BF16 草稿：三个有效请求的输出 token 数为 31/31/31；[全部样本](dspark-32k-performance-20261009/k3_bf16_FULL/samples.json)、[实际启动参数](dspark-32k-performance-20261009/k3_bf16_FULL/launch.json)。两个 rank 的实际 XPUGraph replay 已核验。
- DSpark k3 FULL，FP8 Markov/草稿词表：三个有效请求的输出 token 数为 31/31/31；[全部样本](dspark-32k-performance-20261009/k3_heads_fp8_FULL/samples.json)、[实际启动参数](dspark-32k-performance-20261009/k3_heads_fp8_FULL/launch.json)。两个 rank 的实际 XPUGraph replay 已核验。

三组实际输出长度不同：不报告完整请求总耗时的加速百分比。三组所有有效请求的前 30 个 token 相同，第 31 个 token 处 DSpark 输出 EOS；原始百分比计算和完整响应保留为历史诊断，不能升级为固定 256 token 的性能结论。[历史原始比较](dspark-32k-performance-20261009/comparison.json)

本轮推理容器已停止、31065 端口已关闭；GPU 0/1 的 root Sysman 查询结果保存在 [释放记录](dspark-32k-performance-20261009/release.json)。未停止其他容器或重置设备。运行仓库和 native kernel 没有新增修改，未创建提交、变更版本或写远程仓库。[源码/镜像身份](dspark-32k-performance-20261009/source-identity.json)

<!-- real-text-32k-performance:end -->
> **2026-10-09 输入方法修正：** 下文所有 2K/256 性能表、阶段 trace、路由分布和相应 E2E roofline ratio 均来自同一份编号重复资料，使用 `ignore_eos=true` 强制续写。这些历史测量保留为合成负载诊断，不能据此认定真实任务上的 DSpark 收益或推荐最佳配置；包括 5.28%、17.36% 在内的收益需要用有实际语义的输入重新验证。GSM8K 得分与精确 shape 的 kernel 微基准仍保留各自的证据意义。
>
> 用户已指定后续使用超过 256k 的连续长文本，在指定位置截断。已收集 Project Gutenberg 的《战争与和平》英文完整正文：3,201,654 个字符，按 1008 主模型 tokenizer 为 781,349 token（正文计数，不含 chat template）。[原文、tokenizer 身份和校验值](benchmark-inputs/war-and-peace-2600/manifest.json)。随后完成独立的短输入接受率核验，结果见下节；未来完整长上下文测试的单位、截取方式和具体位置仍待确定。历史 `recommended-profile.json` 是旧实验参数快照，不再作为真实文本性能已经验证的推荐。

## 2026-10-09 《战争与和平》接受率小样本核验

**本次短输入检查没有出现接受率过高或接近饱和的情况。** 在相同 k3 FULL 配置下，六段原始 completion 的汇总 draft 接受率为 **22.51%**、平均接受长度 **1.675**；六段 chat 续写为 **20.82%**、平均接受长度 **1.625**。这是输入行为诊断，不报告性能收益，不作为完整长上下文接受率或语料训练成员关系的证明。[汇总](dspark-input-acceptance-20261009/summary.json)、[完整响应与逐轮数据](dspark-input-acceptance-20261009/results.json)

| 正文截断位置，原文 tokenizer token 索引 | 原始 completion 接受率 | Chat 续写接受率 |
| --- | --- | --- |
| 2,048 | 12.86% | 15.24% |
| 65,536 | 13.41% | 18.18% |
| 131,072 | 18.81% | 22.15% |
| 262,144 | 32.32% | 23.94% |
| 524,288 | 27.19% | 27.19% |
| 700,000 | 40.87% | 19.78% |

每个位置仅取其前面的 **2,048 个连续 token 窗口**，没有把从书首到该位置的全部正文送入模型。因此表中的 262,144/700,000 是原文采样位置，**不是本次推理上下文长度**。原始 completion 直接传入 token ID 列表，输入确认为 2,048 token；chat 另加入自然续写指令和模板，实际输入为 2,081–2,082 token。六个窗口不重叠，内容、参考后文和哈希均保存在 [输入](dspark-input-acceptance-20261009/cases.json)。

参数：1008 主/草稿权重，TP2，BF16 激活，主模型 online FP8，草稿 backbone BF16、Markov 与草稿词表 FP8 oneDNN，k3、FULL、prefix cache 关闭，串行，temperature=0、seed=7。输出最多 256 token，`ignore_eos=false`、无最小输出长度；12 次请求中 11 次达到长度上限，最后一次 chat 自然 EOS，输出 238 token。预热请求独立保存且排除。此次沿用用户此前对卡 0/1 的 DSpark 开发授权，仅做输入核验；没有启动逐 kernel profiler，也没有选择后续完整长文本测试的上下文长度。[冻结参数](dspark-input-acceptance-20261009/PLAN.json)、[实际启动参数](dspark-input-acceptance-20261009/launch.json)

原始 completion 汇总 918 轮、提出 2,754 个草稿 token、接受 620 个，三候选全部接受的轮次比例为 9.91%；chat 为 935 轮、提出 2,805 个、接受 584 个，全部接受比例为 5.13%。逐轮 accepted/proposed 数组、直方图和累计计数已交叉验证。指标在末端输出截断前统计；保留每请求 0–3 token 的 preclip 差值，不将它当作实际发出的 token 数。

输出与原文后续 512 个 token 的精确前缀匹配，原始 completion 最长为 11 token、chat 最长为 4 token；允许在参考后文中移动匹配位置后，最长连续重合分别为 11/5 token。抽查生成内容为段落续写，未见长段原文逐字复述；这不排除不同版本、译本或其他训练内容的影响。仅凭文本知名度不能推定本配置接受率很高。

作为不同任务的历史参考，同一 FP8 head 配置的 GSM8K20 自然作答接受率为 78.46%、平均接受长度 3.354，明显高于本次文学续写。历史评分响应可支持这个接受率比较，但不升级为新的性能对照。[历史 GSM8K 响应](dspark-optimization-20261009/gsm8k/k3_heads_fp8-results.json)

当前可将这份正文保留为低接受率续写输入，后续在用户指定的完整前缀长度下重新测量接受率、性能和 trace，不能直接把此次 2K 窗口结果外推到超长上下文。此次推理容器已停止，root Sysman 查询确认 GPU 0/1 无进程，31065 端口关闭；未停止其他容器、未重置设备。[释放记录](dspark-input-acceptance-20261009/release.json)。运行仓库的 26 个已有修改文件哈希保持一致，没有新增运行代码修改、提交、版本变更或远程写入。[源码核验](dspark-input-acceptance-20261009/runtime-preservation.json)

> 2026-10-09 Roofline 定向三轮迭代完成：推荐 k3 FULL、Markov 与草稿词表权重 online FP8 / BF16 激活、原 oneDNN 投影，TPOT **8.530 ms/token**，比新鲜 k3 BF16 的 9.005 降低 **5.28%**。四组各三次有效样本、各新跑 GSM8K **19/20、零截断**。已实现全部阶段的分析模型和 ratio 计算，补齐硬件/通信参考、真实路由及 FULL 阶段 event trace；GPU 0/1 已释放，推理容器和端口 31065 已关闭。

## 2026-10-09 Roofline 定向 Markov / 词表迭代

| 本轮配置 | TPOT 中位数 ms/token | 相对新鲜 k3 BF16 的降幅 | GSM8K |
| --- | --- | --- | --- |
| k3 BF16 Markov，新鲜对照 | 9.005 | — | 19/20，无截断 |
| k3 Markov online FP8，原 GEMV | 9.015 | -0.11% | 19/20，无截断 |
| k3 Markov online FP8，原 oneDNN | 8.822 | 2.04% | 19/20，无截断 |
| k3 Markov + 草稿词表 online FP8，原 oneDNN | **8.530** | **5.28%** | **19/20，无截断** |

前两轮只改变草稿 Markov 权重的运行时格式和投影后端，第三轮另对草稿词表投影增加 FP8 buffer。主模型权重/词表、草稿 backbone、2K/256 输入、BF16 激活、TP2、FULL、seed=7 和 warmup/三样本规则保持一致。FP8 词表 buffers 挂在草稿模型上，即使 LM head 与主模型共享，主模型 module 和原权重仍不改动。最后一张 TP 分片的 48 行 padding 不参加 scale 计算，FP8 padding 设为零。

保留 BF16 原权重：Markov FP8 buffer 约 60.59 MiB/rank，词表另增加 242.38 MiB/rank，总计约 **302.97 MiB/rank**，不宣称模型常驻内存下降。各组接受长度会随舍入略变；最终配置与新鲜对照的三次汇总提出/接受/轮数恰好一致（1578/239/526），单次顺序并非严格同输出比较。三样本中较慢的样本保留，没有作为异常值排除。[本轮完整比较](dspark-optimization-20261009/comparison.json)

精确 Markov shape `[1,256] × [256,248192]` 的图内微基准在 GPU 0/1 分别为：BF16 217.112/216.661 μs；原 FP8 GEMV 230.118/230.091 μs；最佳一维读取 GEMV 204.021/204.069 μs；原 oneDNN FP8 131.036/131.023 μs。四种 KS/RG 调度候选比原 FP8 GEMV 更慢，未选用。FP8 各候选使用实际 dequantized 权重的 FP32 product、BF16-rounded 输出作独立 reference，并在 Graph 前后检查。[原路径](dspark-optimization-20261009/micro-markov-generic.json)、[KS/RG 候选](dspark-optimization-20261009/micro-policies.json)、[一维读取及 oneDNN](dspark-optimization-20261009/micro-flat.json)

原 GEMV 的源码调度注明按 B70、K=2048 测得；Markov 的 K=256、大 N 工作负载没有获得该调度的收益。这一结论有确切 op/shape、源码和微基准支持。oneDNN 的 FP8 单次逻辑流量约 64.03 MB，比 BF16 的 127.57 MB 小；608 GB/s 单遍读取/写入模型下，条件 kernel ratio 从原 FP8 GEMV 的约 45.8% 提高至 oneDNN 的约 80.4%。这些是逻辑 tensor 流量对应的模型 ratio，尚非实测 DRAM 利用率。[独立 roofline 图](dspark-roofline-20261009/markov-roofline.svg)

真实 routing 在单独请求开启 CPU snapshot，未计入性能样本：64 轮 × 40 层，包含被拒绝的全部四个验证位置；平均每层 21.459375 个不同 expert，范围 10–32；两 rank 原始 JSON 的 SHA-256 均为 `1f7e95efbbdf9ce2f1b4dc21cfd87b4991dd9673f7f7d2087b06fdb64dcbb557`。这是请求前 64 轮的实测参考，不是三次测量整个窗口的 route distribution。模型脚本已用该 profile 的实测 expert 数替换原 `[8,32]` 情景，保留 KV/cache 假设范围。[rank 0](dspark-optimization-20261009/routing-k3_markov_fp8/rank0.json)、[rank 1](dspark-optimization-20261009/routing-k3_markov_fp8/rank1.json)

实现保持独立 source/native 覆盖，未构建完整 wheel、变更版本或写远程仓库。原通用路径脚本已另存 [源码快照](dspark-optimization-20261009/source-generic/manifest.json)；隔离 native 使用独立 namespace 避免 device image kernel 名冲突。[KS/RG 构建身份](dspark-optimization-20261009/markov-tuning-manifest.json)、[一维读取/oneDNN 构建身份](dspark-optimization-20261009/markov-flat-manifest.json)

### FULL 阶段 trace 与当前 ratio

为了避免旧 PTI 的 Graph completion event 问题，本轮在独立请求中用 XPU event 包围真实 XPUGraph replay，逐 rank 保留 128 个事件（主/草稿各 64），不把附加 trace 的时间计入三次性能样本。它记录完整 graph 阶段时间，而不是每个 kernel 的 device trace；同步请求改变提交节奏，时间包含通信等待，不能把阶段中位数之和当作未扰动的请求关键路径。[阶段原始数据/对照](dspark-optimization-20261009/phase-comparison.json)

| 配置 | 主模型阶段中位数，rank 0/1 ms | 草稿阶段中位数，rank 0/1 ms |
| --- | --- | --- |
| Markov FP8 oneDNN | 8.570 / 8.674 | 3.355 / 3.361 |
| Markov + 草稿词表 FP8 oneDNN | 8.575 / 8.630 | 2.925 / 2.978 |

第三轮收益与草稿阶段耗时下降一致，主模型验证阶段基本稳定。当前整个 DSpark round 仍主要受主模型约 8.6 ms 的验证成本限制；这些数据没有把主模型里的 MoE/GDN 各自剩余成本进一步隔离。

| 当前配置 | 实测 TPOT ms/token | 608 GB/s 参数下的模型 TPOT ms/token | 条件 roofline ratio |
| --- | --- | --- | --- |
| 原 Markov FP8 GEMV，实际 routing | 9.015 | 4.981–5.899 | 55.25–65.43% |
| Markov FP8 oneDNN，实际 routing | 8.822 | 4.981–5.899 | 56.47–66.87% |
| Markov + 草稿词表 FP8 oneDNN，实际 routing | **8.530** | **4.690–5.607** | **54.98–65.74%** |

第三轮自己的 64 轮 routing 在两个 rank 同样字节一致，平均 unique expert 为 21.345703125、范围 10–32。上表分别使用各配置自己的 route 数据，原始计数/哈希包含在模型输入记录中。ratio 区间仍取决于共享 KV/cache 和 elementwise 物化假设，不是实测 DRAM 利用率或确定可移除成本；dtype 改变也改变理论工作量，不能只按 ratio 大小判断快慢。[全部配置/组件/逐样本计算](dspark-roofline-20261009/results.json)、[模型与实测图](dspark-roofline-20261009/dspark-model-comparison.svg)

### 复现与释放

[推荐参数与全部环境开关](dspark-optimization-20261009/recommended-profile.json) 保存完整 CLI、FULL/pass/capture 参数及权重格式，使用原 v0.31 派生镜像 `sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566`。实际实现为 [草稿 FP8 覆盖](dspark-optimization-20261009/markov_fp8.py) 与 [原 oneDNN C++ 入口绑定构建](dspark-optimization-20261009/build-flat.py)，通过独立 `markov-flat.so` 避开通用 GEMV 分派；采集器在正常性能请求不做 GPU snapshot/event 记录。

本轮共 12 个有效性能样本、80 个新 GSM8K 响应；模型加载、编译、Graph capture、warmup、microbenchmark 和附加审计请求均排除。所有四个推理容器已停止，使用 root、host PID namespace 的 Sysman 查询确认卡 0/1 已无本轮 worker，端口 31065 关闭；未硬件重置。[完整释放记录](dspark-optimization-20261009/final-release.json)

基础 vLLM 提交仍为 `95307ac6d4cd8be2c1e38f236e9104c569af9025`，kernels 仍为 `bee39d9299538ce6e82a947443790450aa570f3c` 之上的既有 BF16/M16 开发树；26 个整合文件、原 native 库与既有改动保留。本轮只新增分析/实验 artifacts 并更新本文，没有 commit、push、包/镜像版本变更。Python syntax/Ruff、数值 reference/Graph 检查和 GSM20 均完成；没有宣称 clean wheel/image 或完整数据集验收。[源码和 native 保存核验](dspark-optimization-20261009/runtime-preservation.json)

本次将分析模型、微基准、未选用的调度/读取候选、oneDNN 后端选择及词表权重优化作为一轮完整交付。k7/k15 的新双-head FP8 未执行，不把当前 2K/256、k3 的收益外推为所有 block/上下文的通用结论。

> 2026-10-09 迭代完成：3 候选 FULL 的 TPOT 为 9.010 ms/token，比同日非 speculative FULL 的 10.902 低 17.36%；7 候选为 9.182，补齐 M=16 MoE 融合的 15 候选为 10.756。四组均有三次有效样本和 GSM8K 19/20、零截断，Native router 355 项 GPU 检查通过。GPU 0、1 已释放。本轮 patch 与可复现配置见下节，旧数据和旧释放记录保留。

> 2026-10-09 Roofline 续作：每轮分析模型和十种配置的 ratio 计算已实现，新增 GPU 0/1 带宽、BF16 GEMM、TP 通信三样本校准已完成并释放 GPU。最新 k3 FULL 的模型 TPOT 为 4.249–6.851 ms/token，ratio 为 47.16–76.04%；这是显式路由/cache 假设下的分析区间，完整模型的实测流量验证仍未完成。当前结果、公式和重建脚本见下节。此前“尚无模型”的静态核查保留为历史状态。

## 2026-10-09 Roofline 分析模型与硬件参考

[模型脚本](dspark-roofline-20261009/model.py) 与 [完整组件/逐样本结果](dspark-roofline-20261009/results.json) 覆盖主模型 dense、router、routed/shared expert、GDN/conv/rollback、KV attention、vocab、greedy/rejection，以及草稿 context projection/KV、五层 backbone、non-causal attention、vocab、串行 Markov 和 TP 通信。使用本任务权重配置、直接关联的历史 op shape 和当前验证行数，不迁移 PR6 的 batch 效率 penalty。

GPU 0/1 的 PCI/UUID 和上轮一致；每项四次调用预热后采三块，seed=7。BF16 4096³ GEMM 为 170.44 / 177.05 TFLOP/s；512 MiB 输入和 512 MiB 输出的 Graph 流式 copy 为 528.22 / 529.99 GB/s；实际 Markov shape 的 BF16 GEMM 为 217.076 / 217.222 μs。[计算参考](dspark-roofline-20261009/compute-reference2.json)

小消息 all-reduce 的 Graph 参考为约 9–11 μs/次，普通提交约 25–27 μs/次；full-vocab gather 随行数从约 20 μs 增至 191 μs。两 rank 均独立保留三个样本块；参考使用平衡到达，未把旧 trace 的 peer wait 当作带宽成本。[rank 0](dspark-roofline-20261009/collective-reference-rank0.json)、[rank 1](dspark-roofline-20261009/collective-reference-rank1.json)

默认表使用 [Intel B70 官方](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html) 608 GB/s 显存带宽、22.94 TFLOP/s FP32 vector 参数与较慢卡的 BF16 matrix 实测参考；结果同时提供 copy 带宽参考版本。矩阵参考不是每个 shape 的保证上限，copy 参考也不是通用纯读取上限。未拟合 E2E penalty，ratio 超过 1 不裁剪。

每个组件采用 `count * max(useful_FLOPs/P, scenario_bytes/BW)`；沿主/草稿和串行 Markov 的依赖顺序累加，再加入每卡 81 次主 all-reduce、11 次草稿 all-reduce及主/草稿各一次 vocab gather 的平衡通信参考。不相加两个 rank 的时间。MoE 每层 unique expert 在 `[8,min(256,8*M)]`；KV 的共享窗口读取与每 query/head 重扫分别构成两种逻辑流量场景，elementwise 也分别考虑融合复用与显式物化。区间端点是模型情景，不是置信区间、实测 DRAM counters或保证的物理上下界。

| 配置 | 实测 TPOT ms/token | 模型 TPOT 区间 ms/token | Ratio 区间 |
| --- | --- | --- | --- |
| 10-09 baseline FULL | 10.902 | 3.822–4.081 | 35.05–37.44% |
| 10-09 DSpark k3 FULL | 9.010 | 4.249–6.851 | 47.16–76.04% |
| 10-09 DSpark k7 FULL | 9.182 | 4.022–8.658 | 43.80–94.30% |
| 10-09 DSpark k15、M16 FULL | 10.756 | 4.463–12.609 | 41.50–117.23% |
| 10-08 baseline NONE | 18.754 | 5.198–5.458 | 27.72–29.10% |
| 10-08 baseline PIECEWISE | 11.190 | 3.822–4.081 | 34.15–36.47% |
| 10-08 baseline FULL | 10.908 | 3.822–4.081 | 35.03–37.41% |
| 10-08 DSpark k15 NONE | 30.292 | 5.250–13.610 | 17.33–44.93% |
| 10-08 DSpark k15 PIECEWISE | 11.413 | 4.581–12.940 | 40.14–113.38% |
| 10-08 DSpark k15 FULL | 13.137 | 5.481–15.484 | 41.72–117.87% |

Ratio 为每个实测样本的 `modeled_decode_ms / measured_decode_ms`，表取三个 ratio 的中位数。逐样本用报告的实际验证轮数乘每轮模型成本，再除以 SSE 首 token 后的 255 个实际输出 token。源码中 metrics 的 observe 先于 max_tokens 截断，旧 k15 最后一轮有一个额外计数，最新 k7 的一个样本有两个。因此采用 `255/num_spec_steps`，保留终端 preclip overshoot，不直接用全请求统计平均接受长度。每个样本 `content_chunks=num_spec_steps+1` 支持首输出对应 prefill、其后的窗口包含这些验证轮。[截断源码](vllm-functional/vllm/v1/core/sched/scheduler.py:2140)

高端情景超过 100% 表明“expert 最大不重叠、KV 每 query/head 重读”的联合成本假设没有描述这些实测结果；这不能证明实测不可能，也不能把 100%-ratio 当作确定可移除成本。模型尚未包含真实 padded DPAS 工作量、SFU/归约吞吐限制、实际 host/kernel-launch latency和完整 FULL 时间线，故仍标为条件分析模型，`empirical_full_model_roofline_validated=false`。

校准中保留了失败尝试：首个容器 entrypoint 错误；第一轮第二卡捕获空 Graph；两次通信初始化缺少 DRM 路径/节点。有效参考仅使用显式 device stream 修复后的 `compute-reference2.json` 及完整 DRM 挂载、仅 0/1 affinity 下通过输出检查的双 rank 通信结果。没有把失败样本用于 roof。[选择和排除说明](dspark-roofline-20261009/calibration-audit.json)、[校准释放记录](dspark-roofline-20261009/calibration-release.json)

重建：`bf16-host-tools/.venv/bin/python dspark-roofline-20261009/model.py`，仅 CPU 读取保留结果。第一轮优化将验证 Markov 权重 online FP8 / BF16 激活，随后结合实际 routing 与 trace 继续迭代；本段尚无新候选的性能或评分结果。

2026-10-08。用户依据先前 GSM8K 同 20 题均为 19/20（95%）接受当前正确性，继续采集性能；逐 token 一致性不作为门禁。本轮使用物理 GPU 0、1、1008 主模型和 DSpark 草稿，完成 V2 runner 的六组编译/Graph 性能采集，共 18 个有效样本，GPU 已释放。本轮各编译模式均通过四道短请求，未另行重复 GSM8K 20 题。

## 2026-10-09 开发迭代

设备、1008 主/草稿权重、BF16 + online FP8、TP=2、V2 runner、2K/256 prompt、seed=7、prefix caching 关闭和三样本协议保持一致。每组新建独立容器，四道短请求及两次完整预热不计入三次测量，随后执行同一 GSM8K 5-shot 前 20 题，max_tokens=1024、自然 EOS、串行执行，无结果复用。[本轮参数](dspark-iteration-20261009/PLAN.json)、[GSM 参数](dspark-iteration-20261009/gsm8k/PLAN.json)

| 配置 | 主 / 草稿 FULL query 长度 | TTFT 中位数 ms | TPOT 中位数 ms/token | 总时延中位数 ms | GSM8K |
| --- | --- | --- | --- | --- | --- |
| DSpark 3，既有 native 覆盖 | 4 / 3 | 163.795 | 9.010 | 2461.714 | 19/20，零截断 |
| DSpark 7，既有 native 覆盖 | 8 / 7 | 169.479 | 9.182 | 2511.377 | 19/20，零截断 |
| DSpark 15，M=16 MoE/router 覆盖 | 16 / 15 | 163.523 | 10.756 | 2906.031 | 19/20，零截断 |
| 同日非 speculative FULL | 1 / — | 162.143 | 10.902 | 2942.847 | 19/20，零截断 |

三候选的主模型和草稿均在两个 rank 确认真实 XPUGraph replay，尚未启动 XPU profiler，避免此前 PTI graph event 问题影响开发采集。[3 候选](dspark-iteration-20261009/k3_FULL-replay-evidence.json)、[7 候选](dspark-iteration-20261009/k7_FULL-replay-evidence.json)、[M=16](dspark-iteration-20261009/m16_FULL-replay-evidence.json)

同日基线三次中位数为 10.902 ms/token；3 候选降低 17.36%，7 候选降低 15.78%，M=16 融合的 15 候选降低 1.34%。当前固定 2K/256 输入选择 3 候选 FULL，主模型 capture=4、草稿 capture=3，配置写在 [可复现参数](dspark-iteration-20261009/recommended-profile.json)。这不是所有工作负载的通用 block 大小；较高接受长度的请求可能更适合 7 候选。所有组的三样本与 warmup 排除已核验。[本轮完整比较](dspark-iteration-20261009/comparison.json)

M=16 组比旧 15 候选 FULL 的 13.137 降至 10.756 ms/token，同时平均接受长度由 1.8286 升到 2.2368，不能把整个端到端变化归因于新 router 的独立速度。默认融合阈值保留为 8，M=16 支持保持显式选择。

四组重新执行同一 GSM8K 20 题，共 80 个新响应，均 19/20、HTTP 200、无截断，仍只错索引 12（盈利年份）。准确性跑批的单次 wall time 已保存在逐题记录中，不据此声称三样本 GSM 性能收益。[本轮评分对照](dspark-iteration-20261009/gsm8k/comparison.json)

本轮采集和检查结束后已停止四个 GPU 容器，卡 0、1 只剩 xpu-smi 查询进程；未执行硬件重置，没有保留推理服务。[本轮最终释放记录](dspark-iteration-20261009/final-release.json)

### M=16 实现与检查

`moe_router_decode.cpp` 将支持上限从 8 扩展到 16，仍以最多 4 行为一个 work-group 的行组，扩展全局读完成计数器槽位，最后一个 work-group 负责复位。每组寄存器/SLM 行数保持不变；plain 和 residual-add/RMSNorm router 共用这一调度。Python helper 的支持说明同步为 16，已有测试加入 9/15/16、跨流连续调用及 >16 拒绝检查。默认 MoE 融合阈值仍为 8。

使用新增 `m16-router.so` 覆盖两项既有 `_moe_C` op；生成产物使用独立 `vllm::moe_m16` namespace 和 C++ entrypoint，避免与已有 native device image 的 kernel 名冲突。它是原 v0.31 派生镜像中的局部 native 覆盖，没有构建完整 wheel。M=16 模型实验显式开启 `xpu_moe_shared_fusion_max_token_num=16` 和 `[8,16]` compile endpoints，保留原小范围的其他 pass；日志确认 `(9,16)` 编译范围的 MoE + input norm 融合已命中。

355 项 GPU 检查通过，涵盖 FP16/BF16、不同 expert/hidden/top-k、renormalize、原地输出、连续/双流调用和边界拒绝。前两次各 354 通过、1 项失败：原测试把 BF16 norm 与 FP32 reference 的一个 BF16 ULP 差异按 FP16 容差检查，随后该 norm 差异又传播进投影 reference。相同输入在旧版 native 和 M=16 native 均产生同一差异，五次串行重放与双流结果一致；诊断未观察到计数器或流不稳定。测试现在按 BF16 epsilon 检查 norm，再用已经验证的、实际舍入的 norm 输出独立检查投影及 routing；Top-k ID/权重原有比较没有放宽。失败记录全部保留。[GPU 通过记录](dspark-iteration-20261009/router-gpu-tests-attempt3.log)、[新版精度诊断](dspark-iteration-20261009/router-precision-diagnostic.log)、[旧版精度诊断](dspark-iteration-20261009/router-precision-original.log)

源码为 kernels 分支 `dev/minicpmv47-bf16-kernels-20261008`，基础提交 `bee39d9299538ce6e82a947443790450aa570f3c`；本轮只增量修改三个既有文件，旧 BF16 改动保留。vLLM 的 26 个整合文件仍匹配上轮源码散列。clang-format、Ruff、YAPF 和 diff 检查通过；没有 commit、push、版本或权重变更。[完整增量 diff](dspark-iteration-20261009/m16-router.patch)、[源码/native 身份](dspark-iteration-20261009/implementation-manifest.json)、[vLLM 身份](dspark-iteration-20261009/runtime-source-manifest.json)

## 2026-10-08 2K 输入和 256 输出

| 模式 | 主模型 Graph | 草稿 Graph | TTFT 中位数 ms | TPOT 中位数 ms/token | 总时延中位数 ms | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 非 speculative eager | NONE | — | 160.694 | 50.973 | 13158.963 | 先前同环境三次采集 |
| DSpark eager，已补齐 attention | NONE | NONE | 160.922 | 35.134 | 9119.993 | 先前同环境三次采集 |
| 非 speculative 编译融合 | NONE | — | 163.182 | 18.754 | 4945.549 | 三次采集完成 |
| DSpark 编译，pass 开关开启 | NONE | NONE | 164.422 | 30.292 | 7889.161 | 三次采集完成 |
| 非 speculative 编译融合 | PIECEWISE | — | 162.962 | 11.190 | 3016.445 | 三次采集完成；实际 replay 已确认 |
| DSpark 编译，pass 开关开启 | PIECEWISE | NONE | 163.672 | 11.413 | 3073.996 | 三次采集完成；主模型实际 replay 已确认 |
| 非 speculative 编译融合 | FULL_DECODE_ONLY | — | 162.765 | 10.908 | 2944.362 | 三次采集完成；两 rank 实际 XPUGraph replay 已确认 |
| DSpark 编译，pass 开关开启 | FULL_DECODE_ONLY | FULL_DECODE_ONLY | 177.779 | 13.137 | 3526.632 | 三次采集完成；主模型和草稿两 rank 实际 XPUGraph replay 已确认 |

先前 eager 数据见 [双卡开发验证](DSPARK_GPU_VALIDATION_20261008.md)，正确性得分见 [GSM8K 20 题](DSPARK_GSM8K20_20261008.md)。新编译组使用相同 prompt，SHA-256 为 `122dab552738423ce9ceef0667c4c0820aaf4ec8322f26c18227b9ebe9e04073`，并发 1、BF16 激活、主模型 online FP8、TP=2、关闭 prefix caching。DSpark 固定 15 个候选、草稿 BF16、关闭 adaptive verification。

本轮为纯文本，image/video 限额均为 0。每组先做四道短请求预热，再做两次完整 2K/256 预热和三次有效测量，temperature=0、seed=7、强制输出 256 token。报告三次测量中位数，保留所有样本。模型加载、编译、capture、预热和 profiler 请求不计入时延。[实验参数](dspark-performance-20261008/PLAN.json)

TTFT 为请求开始到首次非空内容 SSE；TPOT 为首次内容到 SSE DONE 的时长除以 255。DSpark SSE 可一次包含多个 token，此指标是端到端平均 token 时延，不是单次 kernel 时延。

## 编译配置和身份

编译 mode=3、Inductor，开启 #60539 六项 XPU pass，加载此前已验证的 BF16 原生性能覆盖。DSpark 另加载已通过 24 项检查的 non-causal attention 覆盖；这仍是局部 native 覆盖，不代表构建了完整 #622 wheel。

源码基础提交为 `95307ac6d4cd8be2c1e38f236e9104c569af9025`，工作树 `vllm-functional`，分支 `dev/minicpmv47-main-functional-20261008`。初始 baseline NONE 使用既有 #58674、#60539、BF16 整合树，[初始源码和 native 身份](dspark-performance-20261008/source-native-manifest.json)。DSpark 编译首次启动失败后，在两个 Python 文件中修复了异构 attention head geometry 的规则注册；native 二进制不变，后续模式使用修复后的树。

## 编译注册兼容修复

DSpark 使 static attention context 同时含主模型与草稿的 8/4 个本地 Q head。QKV pass 的搜索 pattern 将整数作为 wildcard，两种 geometry 的拓扑因此相同，但各自 shape guard 和 replacement 参数不同。PyTorch 对同一 matcher 的重复拓扑检查抛错，首次 DSpark NONE 尚未进入推理或性能采集。

修复采用临时 matcher 分别注册各 geometry/normalize 变体，再将带原 shape guard 的规则合并到主 matcher，不丢弃第二组规则，也不更改融合数学。CPU FakeTensor 回归测试确认 FP16/BF16 下两个 geometry 均能融合到正确的 `num_q_heads`，2/2 通过；Ruff、`git diff --check` 通过。[完整修复 diff](dspark-performance-20261008/registration-fix/registration-fix.patch)、[源码散列](dspark-performance-20261008/registration-fix/manifest.json)、[CPU 测试](dspark-performance-20261008/registration-fix/pytest.log)

原启动失败日志在 `dspark-gpu01-20261008/performance-dspark_NONE.log`，重试使用独立 `dspark_NONE_retry` 标签。单 geometry 的 baseline NONE 保留原采集与双 rank trace，不将失败启动计作性能样本。

修复后主模型和草稿均完成编译，但自动生成的 8-token warmup 在五分钟内没有完成，两个 worker 持续忙于等待，尚未隔离根因。该启动尝试没有性能样本，保留日志和设备统计。对 15 候选的实际 16-token 验证，<=8 的 MoE/QKV/norm/all-reduce pass 不适用；`dspark_NONE_nosmall` 关闭 `fuse_xpu_moe_shared`，避免其加入 8-token compile range，其余开关和 native 覆盖保持原配置。该参数差异单独标注，不能称为六项 pass 在 DSpark 验证中全部生效。[修复后源码身份](dspark-performance-20261008/source-manifest-after-fix.json)

baseline NONE 运行容器 ID 为 `ac93ad8d263c5c2275a38160a25156d9b1b51588f20a79adf60026b22f5fb6fc`。后续改用每个模式的独立容器和 IPC 环境，完整 ID 保存在对应 `<label>-container.json`。镜像 ID 均为 `sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566`。使用 `ZE_AFFINITY_MASK=0,1`，对应 PCI `0000:18:00.0`、`0000:36:00.0`；每阶段再次核对 Torch 映射，模型只读挂载。[设备映射](dspark-performance-20261008/torch-identity.json)

单请求 baseline capture size 为 `[1]`，DSpark PIECEWISE 为 `[16]`，DSpark FULL_DECODE_ONLY 为 `[1,16]`：主模型 uniform verify length=16，草稿 query length=15，V2 manager 按各自 query length 向上取整生成 descriptor。FULL 的配置让两者均可生成单请求 graph，避免只设置 `[16]` 时草稿被取整到 30 token、因 max_num_seqs=1 而没有合格 graph。实际运行模式以日志、metrics 和 trace 为准。

## Dummy warmup 修复与失败记录

`dspark_NONE_nosmall` 停在 XCCL 初始化 all-reduce；停止服务后两卡独立小算子正常。独立容器 `62843003aaaf1b9a65ff094564bfc737d27c819e7de88bab2a783deadb94a5e6` 的 `dspark_NONE_fresh` 能完成通信、加载和编译，但 4096-token dummy warmup 后报 `UR_RESULT_ERROR_DEVICE_LOST`，未接收真实请求。这些启动尝试没有有效性能样本，日志与关闭记录全部保留。未执行硬件重置。

静态检查发现 V2 dummy input 将所有行标为 decode，导致开启 speculative 时 4096-token 预热行进入 GDN speculative 路径；该路径的 state 与 token index 只为 16-token 验证准备。修复按 decode query length 分类长 dummy 行，真实请求路径不变，16-token capture 保持 speculative decode。CPU 测试检查 4096/16/8-token 与混合行，以及送入 Mamba metadata 的 `-1/0` 标识，共 6/6 通过，Ruff 和 diff 检查通过。修复后的 `dspark_NONE_fixed` 完成 4096/8-token 预热、kernel warmup、四道短请求与五次完整请求，未再出现设备丢失。[修复 diff](dspark-performance-20261008/dummy-warmup-fix/dummy-warmup-fix.patch)、[测试](dspark-performance-20261008/dummy-warmup-fix/pytest.log)、[后续源码身份](dspark-performance-20261008/source-manifest-after-dummy-fix.json)

后续 DSpark 重试恢复六项 pass 开关，不沿用失败尝试的 `fuse_xpu_moe_shared=false` 参数。

## 已完成模式的证据

主模型编译 NONE 的四轮 decode trace 在每个 rank 含 40 次融合 QKV、160 次融合 MoE、156 次 residual norm+FP8 GEMM、120 次 gated norm+FP8 GEMM、4 次 FP8 GEMM pair 和 324 次 inplace all-reduce。GDN output allocation pass 在该图没有命中，不能把六项配置开关都算作收益来源。[双 rank trace 计数](dspark-performance-20261008/baseline_NONE-trace-summary-complete.json)

PIECEWISE 使用 `[1]` capture，日志统计实际 1-token decode 为 PIECEWISE，2048-token prefill 为 NONE；四道短请求通过。TPOT 比编译 NONE 低 40.33%。这是同一 BF16 native 覆盖下的 Graph 对照；eager 参考组没有加载该 BF16 性能覆盖，不能把 eager 到编译组的全部差值归因于编译或单个 PR。[三次采集摘要](dspark-performance-20261008/baseline_PIECEWISE_fresh-summary.json)、[全部样本](dspark-performance-20261008/baseline_PIECEWISE_fresh-samples.json)

PIECEWISE 独立的四轮 decode profiler，在每个 rank 记录 164 次 `urEnqueueGraphExp` 与 164 次 `zeCommandListAppendGraphExt`，即每轮 41 段实际 XPU Graph replay。[trace 摘要](dspark-performance-20261008/baseline_PIECEWISE_fresh-trace-summary.json)

#60539 的四项主要融合 pass 与 inplace all-reduce 受 ≤8-token compile range 限制；DSpark 15 候选的主模型验证是 16 token，草稿是 15 token。DSpark NONE 的四轮 decode trace 已确认每个 rank 的 QKV/norm/MoE/FP8 pair 融合与 inplace all-reduce 事件均为 0，普通 `vllm::all_reduce` 为 368；GDN 与 non-causal draft attention 仍执行。六项开关开启不等于这些规则在 16-token 验证路径生效。[DSpark trace 摘要](dspark-performance-20261008/dspark_NONE_fixed-trace-summary.json)

DSpark 编译 NONE 三次采集的接受率为 7.9202%，平均接受长度 2.1880；其 TPOT 为 30.292 ms/token。先前 eager DSpark 的接受率为 5.1389%、平均接受长度 1.7708，因此 35.134 到 30.292 的变化同时含数值/输出造成的接受长度变化，不能当作单个 kernel 或编译 pass 的独立收益。[摘要](dspark-performance-20261008/dspark_NONE_fixed-summary.json)、[样本与每请求统计](dspark-performance-20261008/dspark_NONE_fixed-samples.json)

DSpark PIECEWISE 主模型真实 16-token 验证命中 PIECEWISE，草稿 manager 按代码策略保持 NONE。三次 TPOT 中位数为 11.413 ms/token，接受率 7.9202%、平均接受长度 2.1880，与编译 NONE 一致。[摘要](dspark-performance-20261008/dspark_PIECEWISE-summary.json)

DSpark PIECEWISE 的独立 profiler 请求尚未完成就关闭了服务，未产出有效 trace；日志同时含 PTI graph completion event 超时。这次附加 trace 不算通过，不重写失败记录。三次有效时延在 profiler 启动前已完成；主模型运行模式由 capture 日志和真实请求 Graph stats 支持，草稿 NONE 由其 manager 策略支持。

主模型 FULL_DECODE_ONLY 首次容器停在 XCCL 初始化，未加载模型、没有性能样本；停止后两卡独立小算子正常。原配置的 `baseline_FULL_retry` 启动成功，完成三次有效采集，TPOT 10.908 ms/token，比 NONE 低 41.84%、比 PIECEWISE 低 2.52%。两 rank 的首轮 replay 日志均确认 `ModelCudaGraphManager`、`num_tokens=1`、`num_reqs=1`、`graph_type=XPUGraph`，捕获一个 graph。为避免重复触发 PTI 的 graph event 阻塞，此 FULL 组未启动 XPU profiler；replay 证据来自只记录首轮的 wrapper 和真实请求 Graph stats。[摘要](dspark-performance-20261008/baseline_FULL_retry-summary.json)

DSpark FULL 在两 rank 均确认主模型 `num_tokens=16` 和草稿 `num_tokens=15` 的实际 XPUGraph replay，各 manager 捕获一个 graph。该组没有开启 XPU profiler。三次 TPOT 中位数 13.137 ms/token；接受率 5.5238%、平均接受长度 1.8286，比 PIECEWISE 的 2.1880 低。因此它的端到端 TPOT 高 15.11%，不能据此认定 FULL graph 自身开销更高。[采集摘要](dspark-performance-20261008/dspark_FULL-summary.json)、[主/草稿 replay 证据](dspark-performance-20261008/dspark_FULL-replay-evidence.json)、[主模型 replay 证据](dspark-performance-20261008/baseline_FULL_retry-replay-evidence.json)

| DSpark 模式 | 三次总接受候选 | 三次总提出候选 | 三次总验证轮数 | 接受率 | 平均接受长度（含 bonus） |
| --- | --- | --- | --- | --- | --- |
| NONE | 417 | 5265 | 351 | 7.9202% | 2.1880 |
| PIECEWISE | 417 | 5265 | 351 | 7.9202% | 2.1880 |
| FULL_DECODE_ONLY | 348 | 6300 | 420 | 5.5238% | 1.8286 |

同一 2K/256 开发输入上，非 speculative 最快为 FULL（10.908），DSpark 15 最快为 PIECEWISE（11.413 ms/token），当前没有观察到 DSpark 相对最佳非 speculative 配置的收益。这不外推到 GSM8K 或其他接受率更高的请求。所有组均为零 preemption；三次有效样本、两次完整预热排除、256-token usage 与中位数计算已核验。[完整对照及核验结果](dspark-performance-20261008/comparison.json)

### 2026-10-08 attention 覆盖范围修正

先前关于 native attention“仅覆盖草稿，主模型保持 stock”的判断范围过窄。新 trace 记录每 rank 四轮主模型验证为 40 次新增 op：Q shape `[16,8,256]`，cu-seqlens 长度 17，`max_seqlen_q=1`、`causal=False`；另有 20 次草稿调用，Q shape `[15,4,256]`、cu-seqlens 长度 2。新增适配按 non-causal paged 调用条件匹配，因此也覆盖展开成单 query 行的主模型 speculative 验证。旧源码、日志与测量不改写，当前覆盖范围以此 trace 为准。这不表示装入了完整 #622 wheel，也不改变已完成 GSM8K 的测量事实。

`_spec_decode_varlen_fwd` 为每个展开 query 设置 `seqused_k = kv_len - q_len + j + 1`，用逐位置 KV 长度保留 causal 边界，因此底层 flag 为 False 不表示主模型变成 bidirectional attention。其主模型与草稿两类新增 op 的形状和标量参数均来自实际 trace。

## 2026-10-08 Trace 与 kernel shape 分析

本节仅分析已有 trace 和源码，没有使用 GPU、修改运行代码或重写原测量。完整逐 kernel/shape 归因见 [分析结果](dspark-performance-20261008/shape-analysis.json)，可用 [CPU 分析脚本](dspark-performance-20261008/analyze-shapes.py) 重建。泛称 `gemm_kernel` 的事件通过 External id，或 correlation → runtime External id，连接到实际 CPU op 的 shape、dtype、stride；两 rank 的 DSpark 各 1236 个 GEMM 事件均完成这条关联，没有按相邻算子猜 shape。

### 覆盖缺口有证据，但不等于所有大 shape kernel 都没有优化

1. 主模型 M=16 验证不进入 ≤8 的融合范围，QKV/norm/FP8 pair/MoE shared/inplace all-reduce 都没有实际命中。[PR #60539](https://github.com/vllm-project/vllm/pull/60539) 也明确将这些 pass 限于 ≤8；这是当前实测代码和配置的覆盖边界。
2. FP8 GEMV 的实际自动分派比图融合边界更窄：`fp8_gemv_w8a16_supported` 在 `M != 1` 时返回 False。M=16 的 FP8 linears 由本轮覆盖退回原 oneDNN GEMM。内核正确性支持 M≤8，并不表示 M=2/4/8 已开启更快的自动分派。[分派条件](kernels-bf16/csrc/xpu/gemv/xe_2/fp8_gemv_xe2.cpp)、[覆盖入口](performance-20261008/native-bindings.cpp)
3. 草稿 dense backbone 为 M=15、BF16 权重；Markov head 为 M=1、BF16 权重。这两类都不满足 W8A16 的 FP8 权重前提。BF16 激活已经补齐，剩余限制来自行数和权重格式。
4. 普通 MoE 已使用 `w8a16_policy_m_16` 的原生 grouped GEMM，GDN 已使用专用 spec kernels，attention 也进入新增原生路径；不能笼统称为“没有优化”或 reference fallback。缺口主要是本轮小 M 融合/GEMV覆盖，以及 DSpark 新增计算成本。

MoE 的 expert/shared 融合核心本身有 tile=16，并允许最多 512 routed rows；但一体化入口的 router GEMV/RMSNorm native kernel 有明确 `M<=8` 检查。单独把 `xpu_moe_shared_fusion_max_token_num` 改成 16 会让入口调用不支持的 router，不能视作可直接启用的优化。需要扩展 M=16 router 或提供正确的多行 routing 路径，再验证融合收益。[router 检查](kernels-bf16/csrc/moe/moe_router_decode.cpp)、[融合核心](kernels-bf16/csrc/xpu/grouped_gemm/xe_2/moe_shared_fused_decode_xe2.cpp)

### DSpark NONE 的设备计算热点

每个 trace 仅四轮稳态 decode。下表是每 rank 的设备 kernel duration 累加除以四，属于 profiler 内的计算事件统计；不加两个 rank，也不作为未启用 profiler 的端到端时延。普通 baseline 一轮生成一个 token，DSpark 一轮验证 16 个位置并计算 15 个草稿，工作量不同。

| 计算部分 | 实际 shape / 路径 | DSpark 每轮设备事件累计 ms（rank 0 / 1） | 说明 |
| --- | --- | --- | --- |
| Routed MoE 两次 GEMM | `w8a16_policy_m_16`，每轮 80 次 | 6.954 / 6.949 | 当前用普通 grouped GEMM；同轮的 shared linears、gate、routing 分开运行 |
| Markov bias | BF16 `[1,256] × [256,248192]`，每轮 15 次 | 3.297 / 3.295 | 全词表投影串行依赖上一位置 token；主模型 FP8 GEMV 不覆盖它 |
| GDN delta-rule | native spec，30 层 × 16 位置 | 2.395 / 2.393 | kernel 内顺序推进 token，并保存每个位置的 rollback SSM state |
| 主模型 FP8 linears | M=16，190 次 GEMM/轮 | 2.039 / 2.029 | 包含共享专家 linears；实际为 oneDNN GEMM，没有新 GEMV 或 norm/GEMM 融合 |
| 草稿 chunk attention | head=256、query=15、paged non-causal | 1.119 / 1.120 | 已有 native kernel，未观察到缺失配置回退 |
| 主模型全词表 logits | BF16 `[16,2048] × [2048,124096]` | 0.879 / 0.878 | 验证 16 个位置的 TP 分片投影 |
| 草稿全词表 logits | BF16 `[15,2048] × [2048,124096]` | 0.873 / 0.873 | 在 Markov 的 15 次 bias 之前，另做一次 15 行投影 |
| GDN causal conv | native spec | 0.659 / 0.659 | 支持 speculative state 处理 |
| 草稿 dense backbone | M=15，BF16，5 层 | 0.475 / 0.475 | 是现有 BF16 GEMM 路径，不能据此断言更换 kernel 必然更快 |

Markov 权重矩阵实际 padded vocab 为 248192，约 121.19 MiB 的 BF16 权重；这个数字是逻辑 tensor 大小，不是假定每轮的 DRAM 流量。其 15 次 projection 合计占该 trace 非通信 kernel duration 累计约 14.7–14.8%。源码中 `disable_tp=True` 将该 head 复制到两个 rank，避免每个 Markov 位置增加跨卡通信；它是 DSpark 特有、可明确归因的计算热点。[head 实现](vllm-functional/vllm/model_executor/models/qwen3_dspark.py)、[15 次串行采样](vllm-functional/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)

GDN 并非缺少 16-token 实现：`gated_delta_rule_spec_kernel` 已在 GPU 上逐位置递推，并把每个位置的 SSM state 写入独立 cache slot，供下一轮依据接受长度 rollback。普通 baseline 的 delta-rule 四轮平均约 0.148/0.149 ms，每轮 16 位置的 spec 路径约 2.395/2.393 ms，增长大致对应实际 token 工作量。不能把这直接判定为同 shape kernel 性能退化。[递推与状态保存](kernels-bf16/csrc/xpu/gdn_attn/gated_delta_rule.hpp)

### 通信与 profiler 影响

普通 baseline 每轮 81 次 inplace all-reduce，DSpark 每轮 92 次普通 all-reduce；trace 的 DSpark 分布为 81 次 `[16,2048]` 和 11 次 `[15,2048]` BF16，payload 从单 token 的 4 KiB 增为 64/60 KiB。DSpark 同时丢失 inplace pass，跨卡调用次数和 tensor 大小均增加。

但 all-reduce kernel duration 在 DSpark rank 0/1 为约 11.97/80.12 ms 每轮，差异非常大，包含等待另一 rank 到达的成本。该 trace 四轮跨度约 514 ms，明显受到 profiler 与 host 提交影响，不能把 80 ms 当作正常服务的纯 PCIe 传输成本或带宽瓶颈。需要单独的通信/提交分析才能量化可移除成本。

### FULL 相对 PIECEWISE 的下降首先表现为接受长度变化

| 模式 | 每请求报告验证轮数 | 平均接受长度 | TPOT ms/token | decode 窗口 / 报告轮数的中位数 ms |
| --- | --- | --- | --- | --- |
| DSpark PIECEWISE | 117 | 2.1880 | 11.413 | 24.875 |
| DSpark FULL | 140 | 1.8286 | 13.137 | 23.928 |

最后一列来自三次未启用 profiler 的 SSE decode 窗口，除以该请求的 `num_spec_steps`，是近似每轮端到端成本，包含采样、调度和输出边界，不是 kernel 时间。FULL 每轮约低 3.80%，但输出同样 256 token 需要多 19.66% 的轮数，组合表现为 TPOT 高 15.11%。两模式的验证/草稿 shape 仍是 16/15，当前证据不支持把这个 FULL 下降归咎于某个 shape kernel 变慢。没有有效 FULL XPU trace，接受长度变化来自哪项浮点/状态/路由差异也未隔离，不能指定某个 kernel 为因果来源。

### 后续调优顺序

> 历史计划更新（2026-10-09）：以下 3/7 候选和 M=16 router/MoE 实验已完成，结果见文首开发迭代。Markov 独立优化及完整 roofline 仍未完成，以下保留为 2026-10-08 的分析记录。

先试 3/7 候选等较短 block，使主模型验证长度回到 4/8，并确认融合实际命中及接受长度/评分；这只对齐图融合和 router 的范围，M>1 的线性投影仍不会自动获得 M=1 GEMV 的全部收益。然后针对 M=16 router + expert/shared 融合，以及 Markov 的 BF16 全词表 projection / bias / sampling 提交成本做独立优化。源码已有 `dspark_draft_topk` 的 gathered Markov bias 路径，可作为另一个候选，但它会改变 proposal 和接受率，需要用 GSM8K 得分和性能一起验证，不能默认打开。

本轮没有实现或执行这些候选。当前可确认的结论是：小行数优化覆盖不足、DSpark 的新增计算，以及此输入较低的平均接受长度共同限制收益；FULL 对 PIECEWISE 的端到端下降则主要表现为验证轮数增加。

## 2026-10-09 Roofline 建模核查（静态）

> 历史状态（2026-10-09）：本节写于模型实现和硬件参考采集之前。当前分析模型与 ratio 已实现，见文首 Roofline 分析模型一节；这里保留当时的缺项核查，不再作为当前“尚无模型”的结论。

**当前没有完整 DSpark roofline 建模，也没有可报告的 DSpark roofline ratio。** 本次核查已有脚本、原始 trace、源码及四组最新采集，仅使用 host Python 标准库，没有启动容器、推理或 GPU profiler。结构化结果在 [roofline-status.json](dspark-iteration-20261009/roofline-status.json)，缺失的理论时延和 ratio 保持 `null`。

### 已有证据与缺项

| 模型部分 | 已有证据 | 尚未完成的部分 |
| --- | --- | --- |
| 主模型 dense / FP8 linears、草稿 BF16 backbone 与 vocab head | 旧 NONE trace 的实际 op、shape、dtype、stride 及耗时 | 分别按实际执行精度/指令路径建立 FLOPs、量化/激活流量和计算上限；最新 FULL 配置的对应 kernel 证据 |
| Routed MoE / router / shared expert | 旧 M=16 kernel 归因；新 M=16 router 检查与模型采集 | 每层实际路由、unique expert 数、权重复用与 padding 工作量；独立 microbenchmark 和带宽模型 |
| GDN / causal conv | 逐位置递推及 rollback state 保存的源码，旧 NONE 设备耗时 | 初始 state 读取、每位置 state 写回、FP32 运算和顺序依赖的模型 |
| Full attention / KV | 主验证与草稿调用的实际 geometry、KV 展开源码 | 同一序列多个 query 对共享 KV 的逻辑读取、cache 复用和分层流量模型 |
| Markov projection / sampling | 本次补充投影的局部 FLOPs/逻辑 bytes；旧 NONE 的两个 rank 原始 GEMM 事件 | 实际 DRAM/L2 流量或可验证的复用模型、同设备 BF16 上限；embedding、bias、argmax、remap 和提交成本 |
| TP 通信 / Graph / host | 旧 NONE 的调用数与 payload；最新 FULL 的实际 replay 证据 | 区分通信传输、peer wait 与提交开销，处理两个 rank 的重叠和依赖；有效 FULL 时间线 |
| 端到端 token 效率 | 每组三次 SSE 测量、接受候选/轮数、评分 | 完整一轮模型与相同 decode 窗口的有效输出 token 数对齐 |

`analyze-shapes.py` 目前生成 shape 归因、kernel 时间累计和端到端每轮成本代理，没有计算 FLOPs、硬件上限或 roofline ratio。`comparison.json` 的 `tpot_reduction_vs_same_day_baseline_pct` 是对照组性能变化；旧 trace 的 `fraction_of_noncommunication_kernel_sum` 是时间占比，两者均不能替代 roofline ratio。

最新四组均未启动 XPU profiler；配置中存在 `--profiler-config` 不等于采到了 trace。旧 DSpark PIECEWISE 的附加 profiler 请求失败，FULL 也没有有效设备 trace。旧 NONE 归因不能直接证明最新 k3/k7/M=16 FULL 各 kernel 的 roofline 效率。

| 最新配置 | 实测 TPOT 中位数 ms/token | 统计平均接受长度（含 bonus） | Roofline TPOT | Roofline ratio |
| --- | --- | --- | --- | --- |
| 非 speculative FULL | 10.902 | — | 未建模 | 未计算 |
| DSpark k=3 FULL | 9.010 | 1.4740 | 未建模 | 未计算 |
| DSpark k=7 FULL | 9.182 | 1.8305 | 未建模 | 未计算 |
| DSpark k=15、M=16 融合 FULL | 10.756 | 2.2368 | 未建模 | 未计算 |

### PR6 batch 模型为何不能直接套用

用户提供的 [PR6 batch roofline 文档](/home/intel/xiangyu/tmp/BATCH_ROOFLINE_PR6_V031.md) 已有 Qwen3.6、FP16、TP2、32K、多独立序列的模型和 ratio。其 580 GB/s、156 TFLOP/s、随 batch 变化的 `hw_eff`、pipeline/Graph/sync penalty，以及均匀随机 top-8 路由都是该参考模型的参数或假设；文档明确其中一些 penalty 按实测修正过。这里没有把它们作为当前 DSpark 的已验证上限。

DSpark 的 `k+1` 是一个序列的验证位置，不是 `k+1` 个独立序列。源码显示 GDN 先读一次初始 state，随后逐位置递推并保存 rollback state；主 attention 把各 query 展开为单 query 行，但这些行指向同一序列的物理 KV blocks。因此不能直接照参考模型把独立 state/KV 流量乘以 `k+1`；attention 的重复逻辑读取也不等于相同倍数的 DRAM 读取。实际 MoE expert 重叠同样尚未验证，不能照搬均匀随机 unique-expert 公式。

当前另有 BF16 草稿、主模型 FP8 存储/BF16 激活、GDN 中 FP32 运算、串行 Markov 采样和不同接受长度。计算 roof 必须匹配实际 kernel 执行路径，而不是仅根据权重 dtype 选一个统一峰值。

### Markov 投影的局部分析模型

本次重新读取两个 rank 的原始 gzip trace，核验原 SHA-256，并通过 External id / runtime correlation 复核实际 `aten::mm`：BF16 `[1,256] × [256,248192]`，stride `[[256,1],[1,256]]`。每个 rank 四轮共有 60 次，即每轮 15 次；padded vocab 248192 来自 trace。以下是从确切矩阵维度推导的工作量，不是新增硬件测量。

采用每个乘加计两次 FLOP 的 `2MNK` 约定，且假设 A/W 各读一次、C 写一次、不读初始 C：

```text
M=1, K=256, N=248192, BF16=2 bytes
F = 2*M*N*K                 = 127074304 FLOPs / projection
D_logical = 2*(M*K+K*N+M*N) = 127571200 bytes / projection
I_logical = F/D_logical     = 0.996105 FLOP/byte
W_logical = 2*K*N           = 127074304 bytes = 121.1875 MiB
```

rank 0/1 的投影平均设备时间分别为 219.790 / 219.696 μs，每轮 15 次累计为 3.297 / 3.295 ms。上述 `D_logical` 是一次读写假设下的 tensor 流量，不是实测 DRAM bytes，不包括 bias 加法、argmax 等其他 kernel。即便 15 次调用重复使用同一权重，也不能在没有 cache 证据时宣称每轮只从 DRAM 读一次，或必须读 15 次。

本次仅得到低算术强度的局部模型；没有将参考文档带宽代入算一个近似 ratio，也没有据此证明 Markov kernel 已接近 roof 或必有多少优化空间。

### Ratio 的计算口径与后续调查方向

标准 roofline 将运算吞吐约束为 `min(P, BW*I)`；相同 kernel 可换成时间口径 `T_roof=max(F/P, D/BW)`，再计算 `ratio=T_roof/T_measured`。F、D、P、BW 和实测时间必须对应相同 kernel、shape、执行精度和存储层；有分层流量时分别使用对应层的 bandwidth roof。[Roofline 原始报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)、[Berkeley Lab 说明](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/)

端到端需要沿主模型验证、草稿、串行 Markov、采样及通信的依赖关系建立一轮时间模型，并处理资源争用/重叠，不能相加两个 rank 或 CPU/GPU 的包含时间。若固定观察到的接受行为，且计数与时窗对齐，可用 `TPOT_roof(k)≈T_round_roof(k)/A(k)`，再与同口径实测 TPOT 比较；这里的 `A(k)` 是该窗口每轮实际输出 token 数。表中的全请求统计平均接受长度还不能直接视为 SSE 首 token 到 DONE 窗口的精确分母。硬件效率和 proposal 接受效率要分别报告，不能把换 block 后的接受长度收益算成 kernel roofline 收益。

后续调查可先覆盖 Markov、MoE、GDN/KV 三类主要成本，再补通信和每轮关键路径。分析模型可以先提供显式假设的估计，实际流量/复用与设备上限未验证时应保留这一证据等级；最新 FULL kernel 时间线仍是单独缺项。本次没有运行校准、microbenchmark 或新的模型性能采集，历史结果与运行代码保持原样。

## 对照范围

这是开发性能采集。用户提供的 [PR6 batch roofline 文档](/home/intel/xiangyu/tmp/BATCH_ROOFLINE_PR6_V031.md) 使用不同模型、FP16、32K/256 和 prefix cache 命中场景；本轮借鉴其短请求预热及 Graph 确认方法，不直接引用其数值作为 MiniCPM/DSpark 的 roofline。

此前 V1、旧权重、GPU 0/3 的 NONE / PIECEWISE 数据保留为历史记录。当前测量的主模型权重、runner、物理卡对、上下文长度不同，不能将两组数值相除来认定独立 PR 收益。

## GPU 释放与交付

六组采集结束后服务已关闭，卡 0、1 仅有 xpu-smi 查询进程，本任务所有运行过的 GPU 容器均已停止，端口 31065 已无监听。未做硬件重置。[最终释放记录及完整容器 ID](dspark-performance-20261008/final-release.json)

本轮源码增量为两个 QKV 注册文件及四个 dummy warmup/测试文件，diff 与 CPU 检查见前文两个修复目录。基础提交仍为 `95307ac6d4cd8be2c1e38f236e9104c569af9025`，分支仍为 `dev/minicpmv47-main-functional-20261008`；26 个整合源码文件与三项 native 产物均匹配已记录散列。历史源码、失败记录和其他工作树的既有改动保留；没有提交、推送或版本变更。
