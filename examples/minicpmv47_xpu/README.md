# MiniCPM-V 4.7 / DSpark XPU 开发归档

2026-10-09：本目录保存当前开发代码对应的实验源码、配置和历史结果。
仓库根目录是当前集成实现；本目录保留原始实验文件及路径，历史文档的
“当前”“未提交”等表述均指各自记录时的状态。

配套 kernels 分支：
[xiangyuT/vllm-xpu-kernels:dev/minicpmv47-dspark-xpu-20261009](https://github.com/xiangyuT/vllm-xpu-kernels/tree/dev/minicpmv47-dspark-xpu-20261009)。

## 源码与依赖

| 项目 | 固定 revision / 状态 |
| --- | --- |
| vLLM main 基线 | `95307ac6d4cd8be2c1e38f236e9104c569af9025`，2026-10-08 快照 |
| XPU fusion PR #60539 | `8f3e461beebc37d665a2cc2ba805adea1617dc34`，已按文件集成 |
| MiniCPM PR #58674 | `4f5977e4c23c55f1d10b12a687443bbb262a27eb`，已按文件集成 |
| kernels PR #622 | `bee39d9299538ce6e82a947443790450aa570f3c`，配套分支的基线 |
| SYCL-TLA | `87f6850680a580654b9ea2c80dbc01aeb36ad231` |
| 运行环境 | upstream 0.31.0 image、Torch 2.14.0+xpu、Transformers 5.17.0、oneAPI 2026.1.1 |

根目录另含 BF16 MoE/GEMV/norm 图融合、QKV op 注册修复，以及 V2 speculative
dummy warmup 的 GDN 状态边界修复。保存归档前运行工作区的 26 个集成文件；
Ruff 仅调整一个测试文件的 generator expression 排版，其 AST 完全一致。
其余文件逐字节一致，整理前的完整差异见 [original-source.patch](original-source.patch)。
原始来源与发布源码的 SHA-256 分别记录在 [archive-manifest.json](archive-manifest.json)。

#58674 后续审计 head 为 `e578cb0e5277cda4e38d9306dfb5ed0c59440616`；
[fix-ci.patch](pr58674-update-audit-20261009/fix-ci.patch) 保留为待应用补丁，
尚未纳入根目录代码或历史测量。

## 实验入口

- [性能与接受率总表](DSPARK_PERFORMANCE_20261008.md)：不同 graph 模式、迭代、32K 实际文本及 GSM8K。
- [BF16 实现](BF16_IMPLEMENTATION.md)与[首轮功能验证](FUNCTIONAL_VALIDATION_20261008.md)。
- [DSpark 支持调查](DSPARK_SUPPORT_REVIEW_20261008.md)与[GPU 验证](DSPARK_GPU_VALIDATION_20261008.md)。
- [XPU graph 实验](XPU_GRAPH_EXPERIMENT_20261008.md)。
- `dspark-gpu01-20261008/`：head 256、paged non-causal attention 适配及构建配置。
- `dspark-iteration-20261009/`：M16 router 编译、运行覆盖和 graph replay 审计。
- `dspark-optimization-20261009/`：FP8 Markov / draft vocabulary 覆盖、独立 native 实验及 microbenchmark。
- `dspark-roofline-20261009/`：分析模型、实测参考、ratio 计算和绘图源码。
- `benchmark-inputs/war-and-peace-2600/`：真实长文本的收集脚本和输入来源清单。
- `dspark-32k-performance-20261009/fixed256/`：固定输入 32768 token、输出 256 token、ignore EOS 的采集脚本与结果摘要。
- `dspark-32k-performance-20261009/chat-template-probe/`：GSM8K 与长文本的模板诊断。
- `dspark-32k-performance-20261009/suffix-context-probe/`：同一文本终点的 2K/8K/32K 历史长度诊断。

## 最新结果与 block size

DSpark checkpoint `block_size=15`。最新 32K、chat-template/GSM8K 测试显式配置
`num_speculative_tokens=3`，因此实际每轮草稿长度为 3；这组接受率不代表默认
k=15 配置。现有代码只在未指定 `num_speculative_tokens` 时读取 checkpoint
block size。变更草稿长度对接受率的影响仍需要同输入对照。
证据见 [dspark-block-size.json](dspark-block-size.json)。

attention 专用编译配置的 page size 为 64；它与 DSpark 草稿 block size 独立。
target 的 hybrid attention/Mamba 分配还会调整 attention block size，启动日志
记录为 576，不能把这些数值混为同一参数。

| 最新实际文本配置，32768 输入 / 固定 256 输出 | TPOT ms/token | 草稿 token 接受率 |
| --- | --- | --- |
| 普通解码 FULL | 12.345 | 不适用 |
| DSpark k=3 BF16 FULL | 25.227 | 7.93% |
| DSpark k=3 FP8 Markov/vocab FULL | 23.593 | 10.15% |

上述性能块各有三次测量，排除 warmup。GSM8K 串行 20 题的最新诊断使用 k=3、
BF16 draft、温度 0：raw 接受率 67.44%，得分 18/20；chat template 接受率
78.26%，得分 19/20。32K 文本添加模板后为 8.96%，长文本低接受率尚未解决。
历史重复文本测量与真实文本测量分别保留。所有结果为开发证据。

## 复现边界

`Dockerfile.dev` 与准备脚本保存 upstream 0.31.0 基础镜像及编译工具链准备方式。
实验通过源码和独立 `.so` 覆盖运行；尚未构建、验证包含全部修改的完整 kernels
wheel 或干净运行镜像。仅安装这两个仓库不能自动启用所有实验覆盖。

脚本保持原样，包含原主机路径、容器名、端口和 device selector。原容器布局为
`/workspace/preparation/{vllm-functional,kernels-bf16,...}`，构建工作区为
`/workspace/build/kernels`；迁移时需要重建该布局或调整脚本路径。模型只读挂载为
`/minicpm` 和 `/dspark`，源码包不含权重。设备选择需按当前主机映射与占用状态确定。

权重目录标签 `1008` 是用户指定的本地标注，下载 URL 位于 `1006/release`。
尚无 draft 训练时所用 target 权重的精确 fingerprint，不能由目录名证明配对。

归档不包含模型/tar、共享库/对象文件、依赖/build tree、原始大型 trace、完整
语料和输入 token 数组、完整模型响应、完整容器 inspect/API dump。它们保留在
原本机工作区；历史文档指向这些文件的链接用于定位本机证据。归档保留了轻量
结果摘要、原始源码哈希和构建入口。验证记录见 [archive-validation.json](archive-validation.json)。

本次归档不使用 GPU。核心代码执行静态检查及 CPU 回归；历史实验脚本通过语法
检查，并保留原样。它们按生产源码 Ruff 规则扫描仍有 497 项风格/静态问题，
未对历史快照执行全套 lint 整理、GPU 回归或重新测量。

核心源码 Ruff、typos、markdownlint、SPDX、测试 CI wiring 等适用检查通过。
完整 pre-commit 的 `mypy-3.10` 仍有 22 个错误：15 个 Optional 属性类型错误、
4 个测试类型声明/动态接口错误，以及 3 个 multimodal `modality` 参数 API 错误。
后三项对应上述尚未应用的 #58674 fix-ci 更新。原始日志保留在
[core-hooks.log](archive-checks/core-hooks.log)。这次提交保存开发快照，commit hook
显式跳过已失败的 `mypy-3.10`；它不构成全套 CI 通过。

首次在 commit hook 内初始化 markdownlint 依赖时，新 archive worktree 的 index
被依赖仓库 index 覆盖；已仅重建该新 worktree 的 index、重新暂存，并从 worktree
直接运行完整检查。原开发工作区与源码未受影响。

归档后的下一轮实验：固定同一组 GSM8K chat 和实际长文本输入，对照 k=3/7/15，
同时记录逐位置接受率、平均接受长度、TPOT 和 graph 命中，确认草稿长度与训练
block size 的关系。本次整理尚未执行该对照。
