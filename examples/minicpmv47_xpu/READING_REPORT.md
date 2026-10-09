# MiniCPM V4.7 XPU 主分支迁移与性能 PR 阅读结果

> 2026-10-09 PR #58674 更新核查：当前 head 为 `e578cb0e5277cda4e38d9306dfb5ed0c59440616`，本机集成仍停在 `4f5977e4c23c55f1d10b12a687443bbb262a27eb`。后续 `6147c09b8b98a931123ceaf113e0fdc514499fda` 补充多模态 kwargs API 兼容和缓存元数据保留，随后合并 main。三文件补丁 `git apply --check` 通过但未应用。所核查的 DSpark/Qwen3.5/RoPE/worker 核心路径在合并 main 前后没有行为差异，七个关键文件与当前 PR head 逐字节一致；这批漏补没有解释当前纯文本低接受率。下文 PR head 和阶段状态为历史记录；[当前完整核查](pr58674-update-audit-20261009/summary.json)、[未应用补丁](pr58674-update-audit-20261009/fix-ci.patch)。本轮无 GPU 执行或运行源码修改。

> 2026-10-08 GSM8K 更新：按数值答案得分验证，基线和 DSpark 的同 20 题串行结果均为 19/20（95%），无截断、错题相同；详见 [评测记录](DSPARK_GSM8K20_20261008.md)。GPU 0、1 已释放。

> 2026-10-08 DSpark 双卡开发更新：已用物理卡 0、1 验证 1008 主模型和草稿，补齐原生 attention 配置。2K/256 eager 时延已采集，长输出一致性仍有限制；详见 [DSpark 开发验证](DSPARK_GPU_VALIDATION_20261008.md)。

> 2026-10-08 DSpark 更新：草稿权重已按 1008 标注下载、CRC64 校验并解包。当前有原生支持路径，需 V2 runner；本轮未使用 GPU。权重结构、XPU 支持边界及对 #60539 优化的影响见 [DSpark 调查](DSPARK_SUPPORT_REVIEW_20261008.md)。

> 2026-10-08 Graph 实验更新：PIECEWISE 已在双卡真实 decode 中使用，并通过本轮输出对照；NONE / PIECEWISE TPOT 为 17.838 / 11.963 ms/token。FULL 有一项可复现的 token 分叉。不同模式结果见 [独立实验文档](XPU_GRAPH_EXPERIMENT_20261008.md)，GPU 已释放。

> 2026-10-08 Graph 静态调查：Torch 2.14 和当前 main 已默认接入 XPU Graph，旧环境开关及单卡实验性提示已移除；本任务 graph 模式下的 BF16 overlay/双卡正确性仍未验证。详见 [XPU Graph 静态分析](XPU_GRAPH_STATIC_ANALYSIS_20261008.md)。

> 2026-10-08 性能更新：已采集 eager、普通 compile 和 #60539/BF16 原生 overlay 优化组；TPOT 中位数分别为 50.171、53.998、17.724 ms/token。真实 decode 确认五类优化执行，详见 [性能采集与历史对照](PERFORMANCE_COMPARISON_20261008.md)。

> 2026-10-08 后续更新：已完成本地 main/模型/性能 PR 整合、旧权重双卡文本检查及新增 BF16 内核小规模 GPU 检查；新权重已按 1008 标注下载校验和解包。当前结果见 [首轮功能验证](FUNCTIONAL_VALIDATION_20261008.md)，下文保留初始阅读阶段记录。

> 2026-10-08 补充：下文保留原 PR 快照的 BF16 限制与历史实测。现已在两个独立本地开发分支补充 BF16 源码；当前实现和 CPU 检查见 [BF16 实现记录](BF16_IMPLEMENTATION.md)，GPU 正确性与性能仍待验证。

2026-10-08。本轮范围是源码阅读、补丁整理和基于 upstream v0.31.0 镜像的开发容器准备，不使用 GPU。迁移需要同时处理 **MiniCPM 4.7 模型支持、vLLM 编译优化和配套 XPU kernels**；只更新 main 或只挑性能 PR 都不构成完整支持。源码存档目的地已由用户指定为 `xiangyuT/vllm`。

## 历史准备与当前差异

历史记录位于 [MINICPM_V47_XPU_SUPPORT_STATUS.md](/home/intel/xiangyu/copilot_workspace/vllm-minicpmv47-xpu-v030/MINICPM_V47_XPU_SUPPORT_STATUS.md)。原工作树在 `dev/minicpmv47-fp8-030`，完整提交为 `f3b9e6bea5e4a64f08db3d316506881f00bfd0bc`，基于 v0.30.0 的 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。17 个文件的回移包含模型、解析器、测试与文档，历史状态文件本身是未跟踪文件。

记录中的纯文本测试使用 BF16、online FP8、TP=2；eager 和 compile 的 32K/512 TPOT 中位数分别为 51.928 与 55.641 ms。它们来自不同服务启动，runner 也未完全对齐，不能作为本次 PR 的收益基线。图片和视频成功依赖隔离环境中的 Transformers 开发版及只读处理器元数据覆盖；原始部署仍有处理器导入、混排顺序和视频语义方面的缺口。

这轮固定的 main 是 `95307ac6d4cd8be2c1e38f236e9104c569af9025`，不是浮动的 `main` 名称。该提交仍没有 `MiniCPMV4_7ForConditionalGeneration` 注册；[模型支持 PR #58674](https://github.com/vllm-project/vllm/pull/58674) 仍未合并。其当前 head 已增加“不读取 4.7 已删除的 drop_vision_last_layer 字段”的修复，并补充 EAGLE-3 接口测试，不能直接把旧回移当作当前 PR 的等价版本。

## PR 身份与依赖

| 对象 | 完整 head | 集成含义 |
| --- | --- | --- |
| [wyc55069407/vllm #6](https://github.com/wyc55069407/vllm/pull/6) | `8f3e461beebc37d665a2cc2ba805adea1617dc34` | fork 中的性能 PR，11 个提交 |
| [vllm-project/vllm #60539](https://github.com/vllm-project/vllm/pull/60539) | `8f3e461beebc37d665a2cc2ba805adea1617dc34` | 同一 head 的 upstream PR，只集成一次 |
| [vllm-project/vllm-xpu-kernels #622](https://github.com/vllm-project/vllm-xpu-kernels/pull/622) | `bee39d9299538ce6e82a947443790450aa570f3c` | 12 个提交，提供前一 PR 调用的原生 op |
| [vllm-project/vllm #58674](https://github.com/vllm-project/vllm/pull/58674) | `4f5977e4c23c55f1d10b12a687443bbb262a27eb` | MiniCPM 4.7 模型、canvas M-RoPE 与解析器支持 |

vLLM 性能 PR 的共同基点是 `a4b14e57b6d5ad25a2215eff93d6f93ba05005df`；kernels main 与 PR 基点均为 `bdf9ac02f55293b6217a1a47e0a497d580474e23`。每个 PR 的提交顺序、文件列表与补丁 SHA-256 保存在 [source-review-manifest.json](evidence/source-review-manifest.json)。

`git apply --check` 的结果：性能 PR 在固定的 vLLM main 上通过，kernels PR 在固定的 kernels main 上通过；模型 PR 有两处上下文冲突，分别在 `tests/models/registry.py` 和 `vllm/reasoning/__init__.py`。这只检查文本能否应用，不证明各提交逐个 cherry-pick、模块导入或运行兼容。

模型 PR 还会改写 4.6 的多模态处理与 `embed_multimodal`，后续适配应核对当前 main 的 `use_vit_merger` 张量归并逻辑与 PR 的图像、视频分开处理。代码仍先拼接图像嵌入、再拼接视频嵌入，视觉塔仍以 `{"image"}` 标记；历史记录中的反序混排与 video-only 限额问题不能宣告已经解决。

## 性能改动在 MiniCPM 4.7 上的适用条件

只读检查本地权重的 `config.json`，文本部分有 40 层，其中 30 层 GDN、10 层 full attention；hidden size 为 2048，256 个专家中每 token 选择 8 个，routed 与 shared expert intermediate size 均为 512。该结构与 MoE pass 的主要形状门禁一致。模型代码把语言部分交给 Qwen3.5，因此这些 decode 优化在结构上有适用性；权重加载后的 dtype、scale、runner 与实际图匹配仍须确认。

| 改动 | 减少的工作与适用条件 | 对旧 BF16 配置的影响 |
| --- | --- | --- |
| MoE 与 shared expert 融合，含 router 和输入 residual-add norm | 合并 routed/shared 调度、复用 workspace，路由 GEMV 与 softmax/top-k 融合；需 FP16 激活、per-tensor FP8、2048/256/top-8、匹配专家布局及无 EP/EPLB/LoRA 等 | BF16 会明确禁用这项 pass；换 FP16 后需要重新检查正确性 |
| QKV split、Q/K Gemma RMSNorm 与 M-RoPE 融合 | full attention 的 decode 范围换成 SYCL op，支持 FP16/BF16 和匹配的 rotary geometry；当前 main 已可能使用融合 Triton op，因此收益不能按旧 eager 的多个独立 op 推算 | 可以匹配 BF16，但本轮未确认实际命中 |
| GDN 两个 FP8 输入投影配对 | 同一 hidden input 的 qkvz 与 ba 投影合并；单行 FP16 的 GEMV 可以一次 launch 完成两组输出 | BF16 或多行输入保留两个 oneDNN GEMM，不能假定配对必然加速 |
| norm 与 FP8 投影融合 | GDN gated norm 与 out projection，以及 Gemma residual-add norm 与输入投影合并；单行 FP16 可融合为 GEMV，多行 FP16 使用单独 SYCL norm 再调 oneDNN | BF16 走通用 norm 与 GEMM 回退，不能声称命中了 FP16 快路径 |
| in-place TP all-reduce | 仅替换单一使用者、连续、无别名的新中间张量，减少 clone 与包装开销；归约仍存在 | 无 FP16 专属门禁；不减少通信字节数，也不消除 TP 归约 |
| GDN output 从 zero-fill 改为 empty | 在 op 完整写出有效行并补零 dummy/padding 的前提下，删除冗余清零；需图中能看见分配与消费者 | 无 BF16 专属门禁；需要核对 splitting 配置和实际匹配 |
| kernels 内部的 FP8 decode GEMV dispatch | `fp8_gemm_w8a16` 自动尝试 BMG、FP16、M=1、FP8-e4m3、scalar scale、对齐与布局合格的路径；其他情况回到 oneDNN | BF16 不命中；此项无需 vLLM 编译 pass，在 eager 下也可能生效 |
| paged decode split-K reduction 按 head dimension 分块 | 增加小 batch decode 的 reduction 并行度；仅当 full attention backend 实际调用该 Xe2 paged kernel 时相关 | 需检查 backend 与真实调用，不能从上下文长度推断收益 |

MoE 的“一个 op”内部仍有多个 GPU kernel，不应把它计作一次 GPU launch。kernels PR 也没有给所有小 batch 开启新的 FP8 GEMV：dispatch 明确只选择 M=1，M=2..8 保留 oneDNN。视觉塔和视频预处理没有被这组性能 PR 专门优化，图片/视频 TTFT 不能直接套用 decode 的预期。

这些判断来自源码，**没有本轮性能数值或加速百分比**。PR 作者报告的测试平台是 Arc Pro B70，不能把那里的结果直接当成 MiniCPM 4.7 的端到端验证。

主要代码位置：

- [vLLM 编译 pass](vllm-pr-review/vllm/compilation/passes/fusion/xpu_fusion.py)：online FP8 门禁、MoE 全层检查、QKV 与 norm 图匹配。
- [vLLM 工具 pass](vllm-pr-review/vllm/compilation/passes/utility/xpu_utility.py)：别名安全与 GDN output 分配。
- [XPU op 包装](vllm-pr-review/vllm/_xpu_ops.py)：MoE 权重与 router 检查、cached plan、直接 process-group all-reduce。
- [GEMV dispatch](kernels-pr-review/csrc/xpu/gemv/xe_2/fp8_gemv_xe2.cpp)：BMG/FP16/M=1 与布局门禁。
- [norm GEMV](kernels-pr-review/csrc/xpu/gemv/xe_2/norm_fp8_gemv_xe2.cpp)：融合、独立 SYCL norm 与 ATen 回退。
- [MoE wrapper](kernels-pr-review/vllm_xpu_kernels/moe_shared_fused_interface.py)：持久 workspace、router 与专家入口。
- [paged decode reduction](kernels-pr-review/csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp)：head dimension 分块与只在 tile 0 写 softmax LSE。

## 开发容器与后续集成

基础镜像为本机已有的 `vllm/vllm-openai-xpu:v0.31.0`：完整 image ID 为 `sha256:d03c3b8fd3c522ef6bc30a5d81848dacdd5a688bb27e79553a4e79ff5f7e87c5`，仓库 digest 为 `sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7`。Dockerfile 使用 digest 固定基线。

原装 Python 环境是 Python 3.12.3、vLLM 0.31.0+xpu、Torch 2.14.0+xpu、Transformers 5.17.0、vllm-xpu-kernels 0.1.15.4、triton 3.8.0+xpu。当前 main 的 Torch、kernels 与 triton pins 与这个基线一致；其 Transformers 范围放宽为 `>=5.16.1,<5.20.0`，原装 vLLM 元数据则要求 `<5.18.0`，后续升级处理器依赖应在源码安装配置里解决。

静态检查确认原装 Transformers 没有 4.7 原生 processor 文件，也没有 `BASE_VIDEO_PROCESSOR_DOCSTRING`；本地权重自带的视频处理器仍导入这个常量，因此镜像升级本身没有消除历史多模态导入问题。最新模型 PR 已避免读取原生 4.7 Config 中删除的字段，但处理器选择与元数据兼容仍是独立工作。

用于读取基线元数据的容器为 `codex-minicpm47-v031-prep-20261008`，完整 ID 为 `517aab95a6ccc1b8ca5c6ee15fd35360d125d3d4586d999539ee6d1a024406ce`。该准备容器完成读取后停止，保留容器及证据。

[Dockerfile.dev](Dockerfile.dev) 在同一 upstream 镜像上补充 oneAPI 2026.1.1 编译器，保留原装 Python 包。kernels README、Dockerfile 与 CI 使用 2026.1；其 AGENTS.md 中的 2026.0 说明较旧，按该文件“优先当前 build metadata”的规则采用 2026.1。[build-dev-image.sh](build-dev-image.sh) 保留网络代理参数和独立构建日志；第一次构建未传入宿主机代理，在取 key 的阶段主动中止，第二次传入代理后成功，两个日志均保留。

已构建开发镜像 `sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566`，并通过 [create-dev-container.sh](create-dev-container.sh) 创建了 `codex-minicpm47-v031-dev-20261008`，完整容器 ID 为 `300e7c7dd6e1fd558d9e81a2c8318bef6462b39602d31f9346327d674c2b837a`。源码目录只读挂载，无 GPU device request、无 `/dev/dri`、无模型挂载。该容器当前安装的仍是 upstream v0.31.0 的原装 vLLM 与 kernels，main 和 PR 快照只用于阅读，没有配置源码 overlay 或安装 PR wheel。

验证确认 `icpx` 为 2026.1.1，包含 SYCL 和 enqueue-functions 头文件的 C++ 源码通过 `-fsycl -c` 编译；没有执行编译产物或 GPU 程序。六项核心 Python 包版本与原镜像一致。完整验证输出在 [dev-container-validation.txt](evidence/dev-container-validation.txt)，设备和挂载配置在 [dev-container-identity.json](evidence/dev-container-identity.json)。进入容器可用 `docker exec -it codex-minicpm47-v031-dev-20261008 /bin/bash`。

后续源码整合顺序为：固定 vLLM main 的独立开发分支，适配当前 #58674，集成 #60539 一次，再从独立 kernels 分支集成 #622 并构建。两个仓库的源码、wheel 与容器记录必须分别固定，不能仅靠相同的 package version 认定包含 PR。源码存档使用用户指定的 `xiangyuT/vllm`；本轮没有创建整合提交或远程推送。

编译优化配置中六个开关默认关闭，需显式启用；PR 示例使用 mode 3、Inductor 和 `splitting_ops=[]`。它们主要覆盖 compile range 终点不超过 8 token 的范围，GDN allocation pass 除外。不应同时使用 `--enforce-eager`；清空 splitting 的运行和图模式兼容性须在后续阶段验证。

## 本轮检查与边界

29 个 PR 改动的 Python 文件通过 AST 语法解析；两组性能补丁通过静态应用检查，模型补丁的两处冲突已记录。语法与补丁检查没有导入 vLLM/Torch，也没有执行模型代码。

原有 v0.30 工作树、状态文件、模型权重与用户媒体保留。本轮新增内容均在此独立目录，源码 checkout 是固定提交的只读阅读基线；PR review 目录是原始 head 的源码快照，没有把补丁应用到 main。尚未运行 GPU kernel 正确性、服务启动、模型评测、trace 或性能 A/B。
