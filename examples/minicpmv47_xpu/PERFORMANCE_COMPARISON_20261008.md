# MiniCPM V4.7 当前性能采集及历史对照

> 2026-10-08 Graph 后续实验：不同 Graph 模式的性能集中记录于 [独立实验文档](XPU_GRAPH_EXPERIMENT_20261008.md)：本轮 NONE / PIECEWISE 为 17.838 / 11.963 ms/token；FULL 未通过全部严格 token 对照，未采正式性能。本文原始采样数字保持不变。

> 2026-10-08 Graph 静态调查：当前栈已具备 XPU Graph 接入，本文各组仍明确使用 graph NONE；17.724 ms/token 不是 graph 收益数据。实现、默认设置和模型限制见 [静态分析记录](XPU_GRAPH_STATIC_ANALYSIS_20261008.md)。

2026-10-08。本轮使用旧权重、物理卡 0/3、TP=2、BF16 激活和 online FP8，采集 32,768 输入 / 512 输出、bsz=1 的端到端延迟。显式开启 #60539 优化并加载配套 BF16 原生 overlay 后，**TPOT 中位数为 17.724 ms/token，比本轮普通编译的 53.998 ms/token 低 67.18%**；两个 rank 的运行 trace 确认五类优化实际执行。GDN 输出分配改写未命中。

## 性能结果

下表每行均为三次正式请求的中位数。TTFT、总时长单位为 ms，TPOT 为 ms/token；各指标分别取中位数。

| 配置 | TTFT | TPOT | 请求总时长 |
| --- | ---: | ---: | ---: |
| 历史 v0.30.0 eager，2026-09-28 | 2719.789 | 51.928 | 29254.829 |
| 历史 v0.30.0 compile，2026-09-28 | 2775.341 | 55.641 | 31209.444 |
| 当前 main / v0.31.0 镜像，eager | 2756.117 | 50.171 | 28393.628 |
| 当前 main / v0.31.0 镜像，普通 compile | 2805.503 | 53.998 | 30398.282 |
| 当前 main，compile + #60539 开关 + BF16 原生 overlay | **2769.714** | **17.724** | **11826.634** |

本轮优化组相对普通 compile，TPOT 降低 **67.18%**，总时长降低 **61.09%**，TTFT 降低 **1.28%**；相对当前 eager，TPOT 降低 **64.67%**。改善主要体现在 decode。普通 compile 的 TPOT 仍比当前 eager 高 **7.63%**；编译模式本身不会自动开启该 PR 的优化。

与历史记录的**观察差值**：当前 eager 的 TPOT 低 3.38%，普通 compile 低 2.95%；优化组比历史 compile 低 68.15%，比历史 eager 低 65.87%。历史文档使用物理卡 0/1、Transformers 5.16.1、v0.30.0 及另一份 prompt；当前使用卡 0/3、Transformers 5.17.0、冻结 main。历史 eager 的 runner 选择没有完整证据。这些差值可用于历史对照，不能作为只改变 PR 的因果收益。

历史数据来源为 [原支持状态文档](/home/intel/xiangyu/copilot_workspace/vllm-minicpmv47-xpu-v030/MINICPM_V47_XPU_SUPPORT_STATUS.md)，原文件保持不变；[提取的历史样本](performance-20261008/historical.json)保留来源 SHA-256。

### 本轮正式样本

| 配置 / 样本 | TTFT | TPOT | 总时长 |
| --- | ---: | ---: | ---: |
| eager / 1 | 2757.974 | 50.421 | 28523.166 |
| eager / 2 | 2755.499 | 50.112 | 28362.807 |
| eager / 3 | 2756.117 | 50.171 | 28393.628 |
| 普通 compile / 1 | 2805.395 | 54.176 | 30489.381 |
| 普通 compile / 2 | 2805.503 | 53.998 | 30398.282 |
| 普通 compile / 3 | 2808.633 | 53.581 | 30188.410 |
| 优化 compile / 1 | 2767.759 | 17.614 | 11768.334 |
| 优化 compile / 2 | 2769.940 | 17.833 | 11882.582 |
| 优化 compile / 3 | 2769.714 | 17.724 | 11826.634 |

原始请求结果和全部预热样本：[eager](performance-20261008/eager-samples.json)、[普通 compile](performance-20261008/compile-samples.json)、[优化 compile](performance-20261008/compile_bf16_overlay-samples.json)。

## #60539 是否实际用上

eager 使用编译模式 NONE；普通 compile 使用 mode=3、Inductor、graph NONE，六个 XPU 优化开关保持默认关闭。**这两组不是 PR 优化组。** 第三组显式开启：

```json
{
  "fuse_xpu_moe_shared": true,
  "fuse_xpu_qkv_norm_rope": true,
  "fuse_xpu_fp8_gemm_pair": true,
  "fuse_xpu_norm_fp8_gemm": true,
  "xpu_gdn_output_alloc": true,
  "xpu_inplace_all_reduce": true
}
```

优化组增加小 token 编译范围，启动配置实际记录 `compile_ranges_endpoints=[8,4096]`。编译日志包含五类非零图改写；重复层会复用编译子图，因此唯一图改写数量不能直接当作模型层覆盖数。[图改写统计](performance-20261008/fusion-match-counts.json)、[编译日志](performance-20261008/debug-run/server-compile_bf16_overlay.log)

采样结束后单独调用 Torch CPU/XPU profiler，使用相同 32K 输入、强制输出 16 token。两份 rank trace 均包含最后一块 prefill 和三个 M=1 decode step；下表按 CPU op 的实际输入 shape 只统计 M=1，**没有把 profiler 耗时纳入性能样本**。两个 rank 的计数一致：

| 优化 | 每个 rank 三步 decode 的实际调用 | 确认结果 |
| --- | --- | --- |
| MoE + shared expert + 输入 norm/router | `xpu_moe_shared_fused_resadd_norm`、router norm、原生 MoE 各 120 次 | 每步 40 层均执行融合路径 |
| QKV split + norm + M-RoPE | `qkv_split_norm_rope` 30 次 | 每步 10 个 full-attention 层 |
| FP8 投影 pair | 普通 pair 3 次 + residual norm pair 87 次 | 每步 30 个 GDN 投影 pair，后者已被进一步融合 |
| norm + FP8 投影 | gated norm 90 次；residual norm pair 87 次；residual norm single 30 次 | 每步 30 个 gated norm 和 39 个 residual norm 投影 |
| 原地 all-reduce | `xpu_all_reduce_` 243 次 | 每步 81 次，含 embedding 归约 |
| GDN 输出分配 | 编译 pass 改写数为 0 | **未命中** |

XPU kernel 事件还包含 BF16 的 `router_kernel`、`SpGatherName/SpGemmName/SpSiluMulName/SpFinalizeName`、`Fp8GemvKernel`、norm GEMV 和 QKV kernel；模板参数明确含 `bfloat16` / `cutlass::bfloat16_t`。这提供了真实模型执行的 BF16 原生内核证据。[trace 摘要及 shape](performance-20261008/trace-summary.json)、[原始 trace](performance-20261008/traces)、[捕获请求](performance-20261008/profile-request.json)

GDN 分配未命中的原因可以从本次 Dynamo 分图中确认：`torch.zeros` 分配位于前一个子图，`gdn_attention_core_xpu` 位于独立 `submod_1`；当前 pass 要求同一图中存在相连的 allocation 和 auto-functionalized op。源码仍使用零填充，本轮没有补改此行为。[实际分图](performance-20261008/computation-graph-rank0.py:765)、[pass 匹配条件](vllm-functional/vllm/compilation/passes/utility/xpu_utility.py:161)

## 配套内核与版本边界

模型源码为 [vllm-functional](vllm-functional)，开发分支 `dev/minicpmv47-main-functional-20261008`，基础 main：`95307ac6d4cd8be2c1e38f236e9104c569af9025`。整合 #60539 head `8f3e461beebc37d665a2cc2ba805adea1617dc34`、模型 #58674 head `4f5977e4c23c55f1d10b12a687443bbb262a27eb` 和本地 BF16 扩展；文件散列与上一轮功能验证一致，未再修改模型源码。[完整源码 diff](functional-20261008/vllm-main-integrated.patch)

BF16 kernels 工作树基础 #622 head 为 `bee39d9299538ce6e82a947443790450aa570f3c`，本轮复用此前编译、验证的 BF16 生产源对象，补编译 QKV 源和临时 op 注册层。性能 overlay 同时注册新增 norm/pair/MoE/router/QKV op，并给原装 FP8 linear 添加 M=1 GEMV 分发；不符合 GEMV 条件的输入直接调用原装 oneDNN C++ 实现，避免 dispatcher 递归 fallback。[构建脚本](performance-20261008/build-performance-addon.py)、[注册层](performance-20261008/native-bindings.cpp)、[构建日志](performance-20261008/build-addon.log)、[产物散列](performance-20261008/addon-sha256.txt)

**性能数字代表 #60539 开关、本地 BF16 扩展和局部 #622 原生 overlay 的组合，不能拆成单项收益，也不代表完整 #622 wheel。** Attention/paged-decode 及其他原装内核保留基镜像实现；本轮没有完整候选 wheel、干净候选镜像或逐项消融结果。

新增 GEMV 分发、oneDNN fallback，以及 MiniCPM 实际 TP=2 head/rotary shape 的 BF16 QKV 检查 4/4 通过。[检查结果](performance-20261008/addon-validation.json) 三个服务配置均在性能采集前回答 `408`、`北京`；原生 BF16 内核上一轮的 35/35 数值检查仍是独立证据，不能替代模型准确率评测。

基镜像为 upstream `vllm/vllm-openai-xpu:v0.31.0`，ID `sha256:d03c3b8fd3c522ef6bc30a5d81848dacdd5a688bb27e79553a4e79ff5f7e87c5`；实际开发镜像 ID `sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566`，保留 Torch 2.14.0+xpu、Transformers 5.17.0 和原装 kernels 0.1.15.4。运行容器完整 ID `3f9720eff5b248b58e8d555e0e7839f7db70c6834c2010eabaccfa00a9a148f3`。[容器身份](performance-20261008/container-identity.json)

## 采样口径与保留事项

每组独立启动，均固定 V1 runner、graph NONE、旧权重、相同输入和设备。服务参数为 max_model_len=33280、max_num_batched_tokens=4096、max_num_seqs=1、gpu_memory_utilization=0.82、图片/视频配额 0、关闭 prefix caching。两次完整 32K/512 预热后，顺序采集三次；模型加载、编译、预热和 trace 均排除。temperature=0、seed=7、enable_thinking=false、ignore_eos=true、min_tokens=max_tokens=512。十五次请求均核对实际 prompt/output 为 32768/512，并收到 512 个非空正文 SSE 片段。[冻结采样计划](performance-20261008/PLAN.json)、[客户端](performance-20261008/bench-chat.py)、[优化启动参数](performance-20261008/compile_bf16_overlay-launch.json)

TTFT 为发出 HTTP 请求到首个非空正文片段；TPOT 为首个正文片段到 SSE `[DONE]` 的时长除以 511。均为本机端到端时延，不能当作 GPU kernel self-time。当前固定 prompt UTF-8 SHA-256 为 `b3d6cc02119ef32717118e1850a6bbd7b2e2ef0ee9bc056415ac0caaa8463a46`，与历史输入不同；三组当前采样使用同一文件，并分别通过服务端 `/tokenize` 核对长度。[prompt](performance-20261008/prompt.txt)、[输入清单](performance-20261008/prompt-manifest.json)

早期优化诊断运行使用全局 DEBUG，FlashAttention 每步重复打印 stride 信息；该次完整数据保留于 [debug-run](performance-20261008/debug-run)，**排除于上表和差值**。最终优化组使用全局 INFO，仅编译 pass 开启定向 DEBUG。一次客户端误连到正在关闭的旧服务所得 HTTP 500 也保留为启动时序错误，未产生正式样本。

证据级别为开发端到端性能采集，未做交错 B/C/B、系统准确率评测或单项消融。卡 0/3 的 UUID、PCI 与 Torch 映射在各阶段前核对；占用检查通过后才启动，其他卡没有执行任务。采集结束后停止本轮服务并释放设备：[卡 0](performance-20261008/gpu0-after.json)、[卡 3](performance-20261008/gpu3-after.json)仅有采集用 `xpu-smi`。旧权重、1008 权重、原 v0.30 工作树和历史文档保留，未提交或推送代码，未改包版本。
