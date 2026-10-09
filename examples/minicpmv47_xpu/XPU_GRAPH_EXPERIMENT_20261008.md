# MiniCPM V4.7：不同 XPU Graph 模式的实测

2026-10-08。本轮使用旧权重、物理卡 0/3、TP=2、BF16 激活、online FP8 和此前已验证的局部原生优化 overlay。**PIECEWISE 已完成实际捕获/重放，TPOT 中位数为 11.963 ms/token，比本轮 NONE 的 17.838 ms/token 低 32.94%。** FULL_DECODE_ONLY 也完成捕获/重放，但 2K 续写存在可复现的 token/概率差异，未采集其正式性能。

GPU 阶段已于 **2026-10-08 16:56:51（北京时间）**结束。测试服务和容器已停止，卡 0、卡 3 已释放；结束占用记录仅包含采集用 `xpu-smi`。[卡 0](graph-experiment-20261008/gpu0-after.json)、[卡 3](graph-experiment-20261008/gpu3-after.json)、[结束时间](graph-experiment-20261008/completed-device-phase.txt)

## 本轮不同模式的性能

32,768 输入 / 512 输出，单并发。每组两次完整预热、三次正式样本；下表各指标独立取中位数。TTFT、总时长为 ms，TPOT 为 ms/token。

| 请求配置 | 实际 decode 模式 | 严格 token 对照 | TTFT | TPOT | 请求总时长 |
| --- | --- | --- | ---: | ---: | ---: |
| NONE | NONE | 7/7，建立参考输出 | 2768.798 | 17.838 | 11884.692 |
| PIECEWISE | PIECEWISE | **7/7 与 NONE 一致** | 2769.178 | **11.963** | **8884.322** |
| FULL_DECODE_ONLY | FULL | 6/7；2K 续写第 11 个 token 分叉 | 未采集 | 未采集 | 未采集 |

PIECEWISE 相对本轮 NONE：TPOT 降低 **32.94%**，总时长降低 **25.25%**，TTFT 变化 **+0.014%**。prefill 保持无 graph 路径，收益体现在 decode。本轮只覆盖单并发、capture size=1；不能推广到其他 batch、capture size、多模态或所有上下文。

### 正式样本

| 模式 / 样本 | TTFT | TPOT | 请求总时长 |
| --- | ---: | ---: | ---: |
| NONE / 1 | 2768.019 | 17.898 | 11913.677 |
| NONE / 2 | 2768.798 | 17.838 | 11884.258 |
| NONE / 3 | 2773.171 | 17.831 | 11884.692 |
| PIECEWISE / 1 | 2767.587 | 11.963 | 8880.557 |
| PIECEWISE / 2 | 2769.178 | 11.967 | 8884.322 |
| PIECEWISE / 3 | 2773.951 | 11.963 | 8887.200 |

十次预热/正式请求均为 HTTP 200，实际 prompt/output 为 32768/512，各有 512 个非空正文 SSE 片段。PIECEWISE 的五次 512-token 输出文本与对应 NONE 输出全部一致。[NONE 原始数据](graph-experiment-20261008/NONE-samples.json)、[PIECEWISE 原始数据](graph-experiment-20261008/PIECEWISE-samples.json)、[输出对照](graph-experiment-20261008/PIECEWISE-output-equivalence.json)

### 上一轮其他运行模式，仅作参考

以下来自上一轮独立启动，不混入本轮 Graph 差值计算：

| 上一轮模式 | TTFT | TPOT | 请求总时长 |
| --- | ---: | ---: | ---: |
| 当前 main eager，原装 kernels | 2756.117 | 50.171 | 28393.628 |
| 当前 main 普通 compile，原装 kernels | 2805.503 | 53.998 | 30398.282 |
| compile + #60539 开关 + BF16 overlay，graph NONE | 2769.714 | 17.724 | 11826.634 |

本轮 NONE 为 17.838 ms/token，比上轮 17.724 高约 0.65%。这里只作运行间一致性参考。v0.30.0 的历史 eager/compile TPOT 分别为 51.928/55.641 ms/token，设备组合、prompt 和软件环境与当前不同，不单独归因给 Graph 或某个 PR。[上一轮性能及历史对照](PERFORMANCE_COMPARISON_20261008.md)

## 捕获和重放的实际证据

PIECEWISE 和 FULL_DECODE_ONLY 两组日志均记录捕获完成；`--cudagraph-metrics` 的运行表确认 M=1 的实际 decode mode 分别为 PIECEWISE 和 FULL，较长 prefill 为 NONE。因此这不是只接受了启动参数的检查。[运行模式证据](graph-experiment-20261008/graph-runtime-proof.json)、[PIECEWISE 服务日志](graph-experiment-20261008/server-PIECEWISE.log)、[FULL 服务日志](graph-experiment-20261008/server-FULL_DECODE_ONLY.log)

各阶段开始前确认设备空闲，并检查 Torch UUID 的字节序映射：逻辑 0 对应物理卡 0 / PCI `0000:18:00.0`，逻辑 1 对应物理卡 3 / PCI `0000:cc:00.0`。其他两张卡没有执行任务。[设备映射](graph-experiment-20261008/torch-PIECEWISE-identity.json)

两种 Graph 模式的日志中出现了负的 graph pool memory 估算/记账值。本轮保留日志，没有把这些负值解释为实际显存节省，也没有据此调整内存配置。服务均可正常加载；这里仅报告端到端时延。

## FULL_DECODE_ONLY 的输出差异

三条短题为乘法、中文首都、英文翻译；四条续写使用 512、2048、8192、32768 个实际 prompt token，并强制输出 64 个 token。NONE 和 PIECEWISE 的七项全部通过；FULL 的短题及 512/8192/32768 续写均与 NONE token IDs 完全一致，**2048 续写在 zero-based 位置 10 分叉**。两种回答均为正常中文，不能将这个现象直接称为乱码或状态损坏。[所有 FULL 检查](graph-experiment-20261008/FULL_DECODE_ONLY-correctness.json)

在 FULL 上连续重试三次，输出完全一致，并稳定重现该分叉。随后重启 NONE，连续三次也与原 NONE 参考一致。在分叉之前生成的十个 token 相同；第 11 个 token 的报告概率为：

| 模式 | `仅`，token 96725 的 logprob | `在`，token 95772 的 logprob | 贪心选择 |
| --- | ---: | ---: | --- |
| NONE，三次一致 | -1.931219 | -2.056219 | `仅` |
| FULL，三次一致 | -1.972602 | -1.972602 | `在` |

观察到的是 FULL 改变了这一位置的概率关系，随后在并列最高分处分叉；BF16 数值差异可以作为解释候选，但尚未逐层对齐 hidden state/attention 输出，**不能确认具体根因**。这也不能反向证明 GDN/KV 更新必然有错。本轮按开始前保存的严格 token 对照口径，未将 FULL 列入通过配置，也未采其正式时延。[FULL 重试及 logprobs](graph-experiment-20261008/FULL_DECODE_ONLY-2048-diagnostics.json)、[NONE 回查](graph-experiment-20261008/NONE_RECHECK-2048-diagnostics.json)

## 可复现的配置与软件身份

共同参数：旧权重 `/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-hf` 只读挂载；TP=2、BF16、online FP8、V1 runner、mode=3 / Inductor、max_model_len=33280、max_num_batched_tokens=4096、max_num_seqs=1、gpu_memory_utilization=0.82、关闭 prefix caching、图片/视频配额均为 0。候选 Graph 设置 `cudagraph_capture_sizes=[1]`、`max_cudagraph_capture_size=1`；其余模型/优化开关一致。

#60539 的六项开关均开启，使用同一局部 BF16 原生 overlay；GDN 分配 pass 在当前分图中仍未命中。本轮没有修改模型或 kernel 源码，也没有产出完整候选 wheel。模型基础 main 为 `95307ac6d4cd8be2c1e38f236e9104c569af9025`，整合 #60539 head `8f3e461beebc37d665a2cc2ba805adea1617dc34`、模型 #58674 head `4f5977e4c23c55f1d10b12a687443bbb262a27eb` 及 BF16 扩展；kernels 基础 #622 head 为 `bee39d9299538ce6e82a947443790450aa570f3c`。源码和原生产物散列与上一轮相符。[源码 preflight](graph-experiment-20261008/source-preflight.txt)

运行镜像为 upstream v0.31.0 派生的 `sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566`；Torch 2.14.0+xpu、oneCCL 2022.1.2、原装 kernels 0.1.15.4。overlay SHA-256 为 `4ca392356f7e8e671a37db5157f64e62242d84134bb1705ee397d4313608011a`。容器完整 ID 为 `3f9720eff5b248b58e8d555e0e7839f7db70c6834c2010eabaccfa00a9a148f3`，结束状态 exited。[最终容器身份](graph-experiment-20261008/container-final.json)

测试使用同一输入文件，SHA-256 `b3d6cc02119ef32717118e1850a6bbd7b2e2ef0ee9bc056415ac0caaa8463a46`，temperature=0、seed=7、enable_thinking=false、ignore_eos=true、min_tokens=max_tokens=512。TTFT 从 HTTP 请求发出计至首个非空正文片段；TPOT 为首个正文片段到 SSE `[DONE]` 的时长除以 511。采集排除模型加载、编译、Graph 内存预估/捕获和预热。[采样计划](graph-experiment-20261008/PLAN.json)、[客户端](graph-experiment-20261008/bench-chat.py)、[PIECEWISE 启动参数](graph-experiment-20261008/PIECEWISE-launch.json)、[FULL 启动参数](graph-experiment-20261008/FULL_DECODE_ONLY-launch.json)

证据为开发阶段的小规模对照，组间独立启动、顺序采样，未做交错 B/C/B 或系统准确率评测。旧权重、1008 权重、原 v0.30 工作树、原始性能/静态分析记录保留；没有提交、推送、升级依赖或改变包版本。
