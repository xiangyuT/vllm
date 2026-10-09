# XPU Graph 当前支持情况：静态分析

> 2026-10-08 后续实验：已在卡 0/3 完成 NONE、PIECEWISE 和 FULL_DECODE_ONLY 的验证。PIECEWISE 输出对照通过、TPOT 11.963 ms/token；FULL 捕获/重放成功，但有一项 token 对照分叉，未采正式性能。结果集中在独立的 [模式性能实验文档](XPU_GRAPH_EXPERIMENT_20261008.md)。下文保留先前静态阶段的证据边界。

2026-10-08。**当前 vLLM main + Torch 2.14 已接入并默认允许 XPU Graph，不再是早期“默认关闭、实验性单卡”的状态。** 旧环境变量 `VLLM_XPU_ENABLE_XPU_GRAPH` 已删除。当前 MiniCPM 的 TP=2 配置在代码上可以进入 graph 捕获路径，但本任务的 BF16 原生 overlay、混合 GDN/attention 状态和双卡 replay 正确性尚未验证；此前 17.724 ms/token 数据明确使用 graph NONE。

本轮仅阅读源码、包元数据、ELF 依赖信息、CI 配置和上游 PR/issue。没有导入 Torch 来创建上下文，没有调用 device_count/设备查询、创建 XPU tensor、capture/replay、启动推理服务或运行 GPU 测试。

## 版本与证据范围

| 对象 | 本次读取身份 |
| --- | --- |
| 已测本地 vLLM 基础 main | `95307ac6d4cd8be2c1e38f236e9104c569af9025`，开发树 `dev/minicpmv47-main-functional-20261008` |
| 调查时远端 vLLM main | `0f112d180033f3757cf283723f5593a40b02ed82`，2026-10-08 07:33:28 UTC |
| 镜像 Torch | `2.14.0+xpu`，源码 `08187d9e0fba026dc8217405802ab5381dc88d90`，`torch.version.xpu=20260100` |
| Torch 固定的 torch-xpu-ops 依赖 | `9022a83389905509033d02cda8005c21be9c976c` |
| 镜像通信与运行库元数据 | oneccl / oneccl-devel `2022.1.2`，intel-sycl-rt / intel-cmplr-lib-ur `2026.1.1` |
| 原装 XPU kernels | `0.1.15.4`；候选 BF16 源码基础 #622 head `bee39d9299538ce6e82a947443790450aa570f3c` |

对比上述两个 main 提交，检查的 11 个 graph 相关核心文件无差异，包括 XPU platform/runner、graph wrapper、编译配置、FlashAttention/GDN backend 和通信代码。[远端身份](graph-static-20261008/upstream-main.json)、[差异检查一](graph-static-20261008/graph-relevant-upstream-diff.txt)、[差异检查二](graph-static-20261008/graph-more-upstream-diff.txt)、[运行库元数据](graph-static-20261008/runtime-package-metadata.txt)

元数据读取使用已有无 GPU 构建容器 `fb05fa28e23be50b5f71ed365bf75890a9b4240b158785cf7cf3d453abce0b2a`，其 devices 列表为空、device request 为空且没有 `/dev/dri` 挂载。开发镜像仍为 `sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566`。没有启动上一轮已停止的 GPU 容器。

## 上游演进与旧结论为何不再适用

| PR | 合并日期 | 对 graph 支持的影响 |
| --- | --- | --- |
| [#34482](https://github.com/vllm-project/vllm/pull/34482) | 2026-02-25 | 首次接入 XPU capture/replay；当时说明不支持 distributed，FlashAttention 限 PIECEWISE |
| [#38193](https://github.com/vllm-project/vllm/pull/38193) | 2026-03-26 | 因驱动和稳定性限制默认关闭，引入 `VLLM_XPU_ENABLE_XPU_GRAPH` |
| [#43043](https://github.com/vllm-project/vllm/pull/43043) | 2026-05-19 | 允许 XpuCommunicator 进入 graph_capture，移除 TP>1 和 FlashAttention 的旧平台级硬禁用/强制回退 |
| [#49419](https://github.com/vllm-project/vllm/pull/49419) | 2026-07-25 | 增加实验性、单卡限制的提示 |
| [#51600](https://github.com/vllm-project/vllm/pull/51600) | **2026-09-24** | 配合 Torch 2.14 默认开启，删除旧环境开关及单卡实验性提示，sleep mode 时仍回退 NONE |

#51600 的合并提交 `dcfc17e0b1cb4a4bb4eeca9a75c82b82d776f25e` 已包含在本地冻结 main 中。不能沿用早期 PR 的“只支持单卡”说明来描述当前代码。[当前改动](graph-static-20261008/pr-51600-files.jsonl)

上游 hardware feature 表仍把 Intel GPU 的 CUDA graph 列标为不支持，与当前接入代码和 #51600 不一致。这里以冻结源码和已合并 PR 判断接口/配置能力，不把表格的旧标记解释为 API 不存在，也不把默认开启解释为本模型已经通过正确性验证。[上游功能矩阵](https://github.com/vllm-project/vllm/blob/0f112d180033f3757cf283723f5593a40b02ed82/docs/features/README.md)

## Torch 底层：有原生 capture/replay

已安装的 `torch/xpu/graphs.py` 导出 `XPUGraph`、`graph`、`graph_pool_handle`、`is_current_stream_capturing` 和 `make_graphed_callables`；C++ header 定义 modifiable/executable command graph，并实现 capture_begin/end、instantiate、replay、pool。不是仅留着空的 Python 名称。[实际安装文件](graph-static-20261008/torch-xpu-graphs.py)、[C++ header](graph-static-20261008/XPUGraph.h)、[官方 API 文档](https://docs.pytorch.org/docs/main/generated/torch.xpu.graphs.XPUGraph.html)

本镜像 Torch commit 的 `XPUGraph.cpp` 明确规定：编译器版本 >=2026.1.0 且设备不是 PVC 时，设置 `enable_native_recording`；BMG 位于该条件分支。对外仍通过 SYCL `command_graph` API，底层启用 native recording，因此不能仅凭类型名仍是 SYCL 就判断它是早期旧实现。#51600 将这个变化称为 Level Zero Graph。PVC 仍走另一个分支，不能把 BMG 的改进直接推广到 PVC。[固定 Torch 源码](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/xpu/XPUGraph.cpp#L111)

源码还约束 capture 必须在同一 stream 结束，使用 graph 专用 memory pool，replay 复用捕获的地址；空图会发出警告。这些条件需要实际运行验证。宿主包元数据为 Level Zero GPU driver `25.48.36300.8`、loader `1.26.2`、IGC `2.24.8`，但本轮未查询驱动的 graph aspect 或运行 capture，不能由包版本单独证明 native graph 可用。[宿主包版本](graph-static-20261008/host-driver-package-versions.txt)

## vLLM 如何使用它

`supports_xpu_graph()` 只检查 Torch >=2.11.0.dev，没有检查每个模型/内核是否可捕获。当前 Torch 2.14 静态满足版本条件；接口存在也已由安装文件确认。[门禁](vllm-functional/vllm/utils/torch_utils.py:1050)

XPUModelRunner V1、V2 和 MM encoder runner 均执行 `_torch_cuda_wrapper()`，把 `torch.cuda.CUDAGraph/graph/graph_pool_handle` 映射到 XPU 等价接口，复用 `CUDAGraphWrapper`。该 wrapper 捕获固定 batch descriptor 的工作并在后续调用 `replay()`，不是只启用 Inductor 编译。[映射](vllm-functional/vllm/v1/worker/xpu_model_runner.py:54)、[捕获](vllm-functional/vllm/compilation/cuda_graph.py:314)、[重放](vllm-functional/vllm/compilation/cuda_graph.py:361)

控制 graph 的当前有效设置是 `compilation_config.cudagraph_mode`。O0 默认 NONE，O1 默认 PIECEWISE，O2/O3 默认 FULL_AND_PIECEWISE；最终模式还会经过模型、backend 和其他配置的规范化。`--enforce-eager` 同时关闭编译和 graph；单独指定 graph NONE 可以保留 mode=3 编译。XPU sleep mode 也会关闭 graph。[优化级别](vllm-functional/vllm/config/vllm.py:253)、[eager 覆盖](vllm-functional/vllm/config/vllm.py:1746)、[sleep mode](vllm-functional/vllm/platforms/xpu.py:340)

## TP=2 与通信

当前平台代码没有“world_size>1 自动禁用”的分支。`parallel_state.graph_capture` 接受 XpuCommunicator，TP/PP/DP graph context 共用指定的 capture stream。原装 all-reduce 进入 Torch XCCL；#60539 的原地 all-reduce 同样仍依赖这个后端。[通信接入](vllm-functional/vllm/distributed/parallel_state.py:673)

与镜像 Torch 精确对应的 torch-xpu-ops 源码规定：捕获 collective 要求 oneCCL >=2022.0.0，并在 capture 期间避免普通 process-group 状态/event 查询。已安装 oneCCL 2022.1.2 满足这项版本门槛，ELF 依赖也指向该环境的 `libccl.so.2`。**这是必要条件通过，不是双卡正确性结论。** oneCCL 官方 2022.0 release 已说明支持 scale-up SYCL graph；不能据此推广到任意 scale-out、transport 或驱动组合。[固定 XCCL 源码](https://github.com/intel/torch-xpu-ops/blob/9022a83389905509033d02cda8005c21be9c976c/src/xccl/ProcessGroupXCCL.cpp#L151)、[oneCCL release](https://github.com/uxlfoundation/oneCCL/releases/tag/2022.0.0)

上游有专门的 XPU Graph CI job，声明执行 dispatcher、mode、breakable 和 encoder graph 测试，也有两卡 e2e job 及不带 enforce-eager 的 TP=2 MXFP4 例子。本轮仅检查 CI 定义，未检查其最近执行结果；这些定义没有证明 MiniCPM 4.7 + 本地 BF16 overlay 的覆盖。[Graph job](vllm-functional/.buildkite/intel_jobs/test-intel.yaml:182)

## 对 MiniCPM 4.7 当前路径的具体判断

| 模式 | 静态含义 | 当前模型边界 |
| --- | --- | --- |
| NONE | 不捕获，保留可选 mode=3 编译 | 上轮优化组 17.724 ms/token 的已测配置 |
| PIECEWISE | 捕获编译子图，attention/GDN 等分图边界可留在捕获之外 | 可以分析为较小捕获范围的候选；仍可能包含 TP collective 和新增 BF16 op |
| FULL_DECODE_ONLY | 整体捕获 decode，prefill 不捕获 | 与单 token decode 的 GDN 静态能力匹配；通信、状态、所有被包含的 op 都需要可捕获 |
| FULL_AND_PIECEWISE | decode FULL，其他 batch PIECEWISE | O2/O3 默认组合；实际规范化由 backend 最低能力决定 |
| FULL | 所有 batch 整体捕获 | 本模型 GDN 不支持 full prefill，不应预期原样保留该设置 |

MiniCPM 语言部分有 30 个 GDN 和 10 个 full-attention 层。当前 FlashAttention V2 与 GDN metadata builder 均声明 `UNIFORM_BATCH`；GDN 的 full capture builder 明确只支持 decode。于是即便去掉早期“FlashAttention 只能 PIECEWISE”的平台限制，仍不能把整个 hybrid 模型当作任意 prefill/batch 均可 FULL。[FA 能力](vllm-functional/vllm/v1/attention/backends/flash_attn.py:620)、[GDN 能力](vllm-functional/vllm/v1/attention/backends/gdn_attn.py:91)、[GDN decode capture](vllm-functional/vllm/v1/attention/backends/gdn_attn.py:717)、[模式回退逻辑](vllm-functional/vllm/config/compilation.py:1478)

本地 #60539/BF16 overlay 已有非 graph 的真实 decode 证据，但 graph 需要额外区分三个静态条件：

- GEMV/norm 使用当前 XPU queue 发起 SYCL 工作，MoE 使用 q.submit；GEMV 的 enqueue extension header 也包含 graph 接口。没有看到这些路径显式禁用 graph，但这不足以证明自定义 op 与 native recording 兼容。
- MoE/router workspace 按 M 持久缓存；router scratch 按 queue 首次 `malloc_device`，随后 `memset(...).wait()`。必须在实际 capture stream 上先初始化，默认 stream 上的预热不能替代。当前 runner 在 graph_capture context 中做同 stream 的预热并同步，具备相应机制；覆盖到每种候选 shape/queue 仍未实测。[router 初始化](kernels-bf16/csrc/moe/moe_router_decode.cpp:716)、[capture warmup](vllm-functional/vllm/v1/worker/gpu_model_runner.py:6870)
- GDN 的 recurrent state、state index 和 KV metadata 必须在 replay 间正确更新；capture/replay 固定地址并不等于这些状态自动正确。当前 GDN builder 为 decode 准备持久 metadata buffer，具备实现路径；本地模型加 overlay 的状态更新尚未验证。

因此，静态结论为“**具备尝试 PIECEWISE / FULL_DECODE_ONLY 等模式的代码和依赖条件**”，不是“该组合已稳定可用”。Graph 主要有机会减少 Python/dispatcher/launch 提交开销；不能从上轮无 graph 的 TPOT 推导额外加速比例，也不能把通信和权重读取的时间视为会随 graph 自动消失。

## 已知问题如何解读

[B70 双卡 issue #48327](https://github.com/vllm-project/vllm/issues/48327) 使用 Torch 2.12、Gemma FP8，报告 graph 下出现乱码；该 issue 仍 open，但维护者在 2026-09-28 已建议使用 Torch 2.14 后的最新代码，并说明解决了一些问题。不能把旧环境的现象直接当作当前 MiniCPM 的已知必现问题，也不能把这个回复当作本模型的通过记录。

[PVC TP=2 issue #48946](https://github.com/vllm-project/vllm/issues/48946) 使用 Torch 2.12 / oneCCL 2021.17，涉及 collective 算法、空图和 decode 状态。其环境明显早于本镜像，而 PVC 的 native recording 分支也不同；仅作为捕获/重放与普通编译需要分别校验的历史例子。[Windows fused-SDPA issue #5069](https://github.com/intel/torch-xpu-ops/issues/5069) 也是另一硬件/后端组合，不能用来断言 Linux vLLM 的自定义 FlashAttention 是否通过。

本轮静态分析已完成。没有修改 vLLM、kernels 或运行配置，没有提交、推送、升级依赖或使用 GPU。后续若另行授权运行，应在保持旧权重、卡 0/3、BF16 online FP8、runner 和优化开关一致的条件下，分别确认捕获/replay、基本输出和 GDN 状态随上下文增长的正确性，再测 graph 的性能；这些验证本轮均未执行。
