# MiniCPM V4.7 首轮双卡功能验证

> 2026-10-08 后续更新：已补采 32K/512 性能，并在真实 decode trace 中确认五类 #60539 优化执行；GDN 分配改写未命中。结果与历史对照见 [性能记录](PERFORMANCE_COMPARISON_20261008.md)。下文保留首轮功能阶段的原始证据边界。

2026-10-08。使用用户授权的物理 GPU 0 和 GPU 3、旧权重完成 TP=2 文本检查；新增 BF16 内核完成独立数值检查。新权重已下载、校验和解包，本轮没有用于推理。没有进行性能比较。

## 已测结果

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 旧权重 TP=2、BF16 激活、online FP8、eager | 3/3 请求 HTTP 200，输出 `408`、`北京`、`Apple` | [请求及响应](functional-20261008/chat-results.json)、[服务日志](functional-20261008/server-eager-old-attempt2.log) |
| 新增原生 BF16 内核及 FP16 对照 | 35/35 项通过：20 项 BF16 数值、14 项 FP16 数值、1 项混合 dtype 拒绝 | [结果](functional-20261008/bf16-kernel-results.json)、[运行脚本](functional-20261008/check-bf16-kernels.py) |
| MiniCPM 4.7 模型 CPU 单测 | 11 项通过，含模型接口、vision hook、canvas M-RoPE | [日志](functional-20261008/model-cpu-tests.log) |
| 集成源码静态检查 | Ruff、Python AST、`git diff --check` 通过 | [Ruff](functional-20261008/ruff-check.log)、[源码清单](functional-20261008/source-manifest.json) |

短请求使用 `max_model_len=2048`、`max_num_batched_tokens=2048`、`max_num_seqs=1`、`gpu_memory_utilization=0.75`；temperature=0、请求 seed=7、关闭 thinking、max_tokens=64。图片和视频配额均为 0。三条短答案仅构成功能检查，不能代表完整模型准确率。

原生检查覆盖 M=1/4 的投影 pair、gated norm、residual norm、router 和 router norm，以及 I=256 的 routed/shared MoE；BF16 MoE 检查 M=1 默认 tile 和 M=4 tile16。BF16 GEMV 精确保留 `+131072`、`-131072`、`2^-100`，residual norm 保留大于 FP16 范围的 residual。测试采用仓库已有 FP32 参考计算和相应容差。

## 实际运行的软件和源码

开发镜像基于 upstream `vllm/vllm-openai-xpu:v0.31.0`，基镜像 ID 为 `sha256:d03c3b8fd3c522ef6bc30a5d81848dacdd5a688bb27e79553a4e79ff5f7e87c5`；添加编译器后的开发镜像 ID 为 `sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566`。Python 依赖保持基镜像版本，包括 Torch 2.14.0+xpu、Transformers 5.17.0、kernels 0.1.15.4。

模型服务使用 [vllm-functional](vllm-functional) 的源码覆盖，分支 `dev/minicpmv47-main-functional-20261008`，基础 main 为 `95307ac6d4cd8be2c1e38f236e9104c569af9025`。本地整合 PR #60539 head `8f3e461beebc37d665a2cc2ba805adea1617dc34`、模型 PR #58674 head `4f5977e4c23c55f1d10b12a687443bbb262a27eb` 和已有 BF16 fusion 补丁；两个模型 PR 上下文冲突在 registry 中解决。共 22 个变更文件，未提交、未推送。[完整 diff](functional-20261008/vllm-main-integrated.patch)、[文件散列及基础提交](functional-20261008/source-manifest.json)

**模型短请求使用基镜像的原装 kernels，编译模式为 NONE，不能据此声称新增 BF16 fusion 已在模型中命中。** 独立内核检查通过临时 `bf16-diagnostic.so` 注册新增 op，复用 [kernels-bf16](kernels-bf16) 已编译的生产源对象。该工作树分支为 `dev/minicpmv47-bf16-kernels-20261008`，基础 PR #622 head 为 `bee39d9299538ce6e82a947443790450aa570f3c`。单 GEMV 使用独立诊断 op 名，避免替换已有 dispatcher fallback；多行 norm/pair 使用原装 FP8 GEMM fallback。[构建脚本](functional-20261008/build-addon.py)、[构建日志](functional-20261008/build-addon-attempt2.log)、[产物 SHA-256](functional-20261008/native/addon-sha256.txt)、[BF16 源码 diff](bf16-evidence/kernels-bf16-final.patch)

这没有产出完整候选 `_xpu_C`、wheel 或候选运行镜像。尚未验证模型编译融合命中、图片/视频推理、全套内核矩阵、新权重推理或性能收益；本轮按用户要求优先快速完成初步功能检查。

## 设备与容器

| 物理 Sysman 卡 | PCI | Sysman UUID | Torch 逻辑索引 |
| --- | --- | --- | --- |
| 0，Arc Pro B70 | `0000:18:00.0` | `00000000-0000-0018-0000-0000e2238086` | 0 |
| 3，Arc Pro B70 | `0000:cc:00.0` | `00000000-0000-00cc-0000-0000e2238086` | 1 |

运行容器 `codex-minicpm47-functional-gpu03-attempt2-20261008` 的完整 ID 为 `3f9720eff5b248b58e8d555e0e7839f7db70c6834c2010eabaccfa00a9a148f3`。挂载 `/dev/dri` 和只读 by-path，通过 `ZE_AFFINITY_MASK=0,3` 限定执行设备；启动前和内核阶段前均核对 Torch UUID 与物理映射，其他两张卡没有创建执行任务。[身份](functional-20261008/container-attempt2-identity.json)、[Torch 映射](functional-20261008/torch-attempt2-identity.json)

首次只挂载两个卡的节点导致 oneCCL drmfd 初始化失败，未加载模型；相关容器与[失败日志](functional-20261008/server-eager-old-attempt1.log)保留。修正完整设备节点可见性后，双卡通信、模型加载及请求通过。内核检查只在 Torch 逻辑卡 0 执行。

本轮测试容器停止后，释放这轮模型与内核进程。结束占用记录见 [卡 0](functional-20261008/gpu0-after.json)、[卡 3](functional-20261008/gpu3-after.json)；查询中的 `xpu-smi` 是采集进程。

## 1008 标注权重

保存至 `/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-1008`，与旧权重同级。归档 70,446,213,120 bytes，OBS CRC64 为 `1232283036859224851`，与完整对象计算值一致；SHA-256 为 `23578937c178340399fb5b25d972442cb7650030a5e5a06cb539edbefef29dbb`。

解包后 16 个 safetensors 分片、1152 个 BF16 张量，索引名称、shape/dtype 字节数和 data offsets 均通过结构检查。顶层 config 字段与旧权重未发现差异；这不代表 tensor 内容相同。`1008` 是用户指定的本地标注，来源 URL 仍是 `1006/release`。[下载及校验清单](model-download-1008/download-manifest.json)

旧权重只读挂载，原 v0.30 工作树和历史状态文档保持原状；历史文档 SHA-256 仍为 `0ec9c322f9d8511657d3d262c5f96e8e0423a7600a5d298f71b808849d2050a9`。未改变包版本，未执行 Git commit、push 或 PR 写入。
