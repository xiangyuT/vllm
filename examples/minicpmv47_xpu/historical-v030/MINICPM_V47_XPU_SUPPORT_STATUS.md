# MiniCPM-V 4.7 35B-A3B：vLLM v0.30.0 / Intel XPU 支持状态

更新日期：2026-09-28。本文件仅保存在本地工作树，不属于已发布的 dev 分支提交。

## 结论

| 输入场景 | 当前回移代码 + 原始容器 | 隔离诊断环境的实测 | 结论边界 |
| --- | --- | --- | --- |
| 纯文本 | 可运行 | XPU 0/1、TP=2、online FP8、eager，聊天请求返回正常文本 | 只做了少量功能检查，没有系统准确率评测 |
| 单图、高清图片 | `AutoProcessor` 导入失败 | 修复处理器选择后，单图及五切片图片请求返回 HTTP 200，颜色检查正确 | 不能说原始镜像开箱即用 |
| 纯视频画面 | 同上 | 两帧短视频及用户提供的 43.4 秒视频均可推理 | 抽帧语义与权重自带处理器不完全一致；音频未处理 |
| 图片与视频混排 | 同上 | 图片在前的样例可用，视频在前的样例出现归属错误 | 有确定的嵌入顺序错配风险，不能宣布混排已完整支持 |

### 测试环境与持久改动

- 本地源码：`/home/intel/xiangyu/copilot_workspace/vllm-minicpmv47-xpu-v030`，分支 `dev/minicpmv47-fp8-030`，提交 `f3b9e6bea5`。该提交基于 vLLM `v0.30.0`，回移 MiniCPM-V 4.7 的模型、解析器、测试及支持模型文档，共涉及 17 个文件。
- 权重：`/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-hf`。索引引用的 17 个 safetensors 分片和 1,152 个张量均已做结构完整性检查；没有官方哈希可供逐字节核对。
- 容器基础镜像：`vllm/vllm-openai-xpu:v0.30.0`，原装 Transformers `5.16.1`。设备仅使用 XPU 0、1；服务配置为 TP=2、`--dtype bfloat16`、`--quantization fp8`、`--enforce-eager`。启动日志选中 `XPUW8A16FP8LinearKernel` 和 XPU FP8 MoE 后端。以上不是 TP=4 或 XPU Graph 的验证结果。
- 镜像中的 CLI 必须通过 `PYTHONPATH=/workspace/vllm` 指向回移源码。容器需要 `/dev/dri`，并只读挂载 `/dev/dri/by-path` 以供 oneCCL 的 drmfd 回退。不要把 vLLM 的 `--device xpu` 当作启动选项；该版本把它解释为设备 ID。
- 本地权重、原镜像均未修改。本文提及的 Transformers 升级和处理器配置覆盖仅在一次性容器中测试，**不在已发布分支里**。

## 纯文本：已跑通，不等于准确率过关

在原始 Transformers `5.16.1` 镜像中，限制图片和视频各为 0，服务成功加载权重。一次通过的配置为：

```bash
PYTHONPATH=/workspace/vllm ZE_AFFINITY_MASK=0,1 \
vllm serve /minicpm \
  --served-model-name MiniCPM-V-4.7-35B-A3B \
  --trust-remote-code --model-impl vllm \
  --tensor-parallel-size 2 --dtype bfloat16 --quantization fp8 \
  --enforce-eager --max-model-len 2048 \
  --max-num-batched-tokens 2048 --max-num-seqs 1 \
  --gpu-memory-utilization 0.75 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 127.0.0.1 --port 31047
```

在聊天请求中设置 `chat_template_kwargs: {"enable_thinking": false}` 后，`17 × 24` 得到 `408`，中国首都的问题得到 `北京`。回移代码的定向模型和解析器测试共 43 项通过。这里的判断是“能加载、能生成基本正确的文本”；**没有完成**正式文本准确率评测。下面另有一次 32K 输入的单并发延迟实测，不能推广到其他长度或并发。已有测试服务已停止。

### 纯文本 32,768 输入 / 512 输出、bsz=1 延迟

2026-09-28，在**原始 Transformers 5.16.1 镜像**和同一回移源码上，另启动一次 XPU 0/1、TP=2、BF16、online FP8、eager 的纯文本服务：`--limit-mm-per-prompt '{"image":0,"video":0}' --max-model-len 33280 --max-num-batched-tokens 4096 --max-num-seqs 1 --gpu-memory-utilization 0.82 --no-enable-prefix-caching`。后一个参数避免相同输入的多轮测试命中前缀缓存；服务日志显示选中了 XPU FP8 线性核及 MoE 后端，前缀缓存命中率为 0。没有使用多模态处理器的临时覆盖。

固定输入由开头的“以下是编号资料，仅用于测试长上下文推理延迟。末尾请简短回答。”及编号重复行组成，逐字截取到**连同聊天模板恰好 32,768 个 prompt token**；`/tokenize` 及每次请求的 `usage.prompt_tokens` 均核对了这个数。输入 UTF-8 SHA-256 为 `e9da807d9b42f7c01fc65c0bde50756656a56db1b96085af7bf022848e5b55ea`。聊天请求设置 `enable_thinking=false`、`temperature=0`、`min_tokens=max_tokens=512`、`ignore_eos=true` 和流式输出；两次相同长度的预热后，顺序运行三次，任何时候都只有一个请求。三次 `usage.completion_tokens` 均为 512，均收到 512 个带正文的流式输出片段。

TTFT 为客户端发送请求至第一个非空正文片段的时间，包含本机 HTTP 与服务端预处理；TPOT 为第一个正文片段到流式 `[DONE]` 的时间除以 511。单位均为毫秒：

| 正式测试 | TTFT | TPOT | 请求总时长 |
| --- | ---: | ---: | ---: |
| 1 | 2719.789 | 51.928 | 29254.829 |
| 2 | 2719.122 | 52.436 | 29514.128 |
| 3 | 2724.660 | 51.873 | 29231.516 |
| **中位数** | **2719.789** | **51.928** | **29254.829** |

这是单节点本机客户端、bsz=1 的端到端时延，不是纯 GPU kernel 时间；只代表该确定性测试输入及这次热态。强制输出 512 token 是为测量延迟，不能据此评价长上下文回答质量。测试服务已停止。

### XPU 上开启 Inductor 编译（纯文本）

2026-09-28，基于上述原始镜像、相同权重及回移源码，在一次性容器中设置 `VLLM_USE_V2_MODEL_RUNNER=0` 和 `VLLM_XPU_ENABLE_XPU_GRAPH=0`，以 XPU 0/1、TP=2、BF16、online FP8 启动：

```bash
PYTHONPATH=/workspace/vllm ZE_AFFINITY_MASK=0,1 \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_XPU_ENABLE_XPU_GRAPH=0 \
vllm serve /minicpm --trust-remote-code --model-impl vllm \
  --host 127.0.0.1 --port 31055 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization fp8 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --max-model-len 33280 --max-num-batched-tokens 4096 \
  --max-num-seqs 1 --gpu-memory-utilization 0.82 \
  --no-enable-prefix-caching --optimization-level 2 \
  --compilation-config '{"mode":3,"backend":"inductor","cudagraph_mode":"NONE"}'
```

**不能加 `--enforce-eager`**：它会把编译模式强制改为 `NONE`，即使同时指定了上面的编译参数。此次启动日志实际记录 `CompilationMode.VLLM_COMPILE: 3`、`backend='inductor'`、`CUDAGraphMode.NONE`，编译了 `(1, 4096)` 的图，保存 AOT 编译函数，并报告 `torch.compile took 61.48 s in total`；两个 worker 均明确跳过 graph capture。FP8 线性核和 XPU FP8 MoE 后端也均被选中。这些日志和请求成功共同确认运行了编译路径，**不是只接受了命令行参数**。

同一服务上，短文本问题“17 乘以 24 等于多少？只回答数字。”返回 `408`。再次使用上一节 SHA-256 相同的输入，服务端确认 prompt 为 32,768 token；设置相同的流式、`temperature=0`、`enable_thinking=false`、`ignore_eos=true`、`min_tokens=max_tokens=512`，bsz=1 顺序运行两次预热、三次正式请求。五次均为 HTTP 200、512 个实际输出 token；正式测试的客户端时延按上一节相同口径计算，单位毫秒：

| 编译配置正式测试 | TTFT | TPOT | 请求总时长 |
| --- | ---: | ---: | ---: |
| 1 | 2777.033 | 55.641 | 31209.444 |
| 2 | 2775.341 | 56.509 | 31651.255 |
| 3 | 2774.535 | 55.631 | 31202.114 |
| **中位数** | **2775.341** | **55.641** | **31209.444** |

前一节的 eager 数字来自**另一次服务启动**，因此不能拿两组中位数直接相减来归因编译收益或回退；本节只证明指定配置下编译能启动、基础回答正确、32K 输入与 512 输出能完成。没有验证编译配置下的图片、视频输入或长上下文准确率。测试服务已停止。

### 32K/512 延迟的静态路径分析与历史数据

以下均为 bsz=1 的 TPOT（ms）；MiniCPM-V 两项是上面两次本机启动的实测中位数，Qwen3.6-35B-A3B 四项由用户提供，**不是本轮在相同 harness、模型及配置下重测**：

| 模型 | v0.30.0 eager | v0.30.0 compile | llm-scaler-vllm:0.26.0-b2 | v0.26.0 upstreamable |
| --- | ---: | ---: | ---: | ---: |
| MiniCPM-V 4.7 35B-A3B，text only | 51.928 | 55.641 | 未测 | 未测 |
| Qwen3.6 35B，online FP8（用户数据） | 50.513 | 53.577 | 15.470 | 46.401 |

MiniCPM 的 compile - eager **观察差值**为 +3.713 ms/token（约 +7.15%）；用户提供的 Qwen3.6 v0.30 两列差值为 +3.064 ms/token（约 +6.07%）。两个模型都观察到 v0.30 编译列更慢，是值得核查的共同现象，但 MiniCPM 的两次服务启动未完全控制热态与节点负载，且额外强制了 V1 model runner，不能把 +3.713 ms 直接归因于 Inductor。Qwen3.6 的 b2 为 15.470 ms，比其 upstreamable 的 46.401 ms 低约 3 倍；**没有对应镜像的这轮调用栈或对齐配置，不把该差距归因于 graph、单个 kernel 或编译开关**。

这次 MiniCPM **纯文本 decode** 的调用路径可由源码确认：

- 模型构造把语言部分实例化为 `Qwen3_5ForCausalLM`，前向直接进入其 `.model`，不执行视觉塔：`vllm/model_executor/models/minicpmv4_6.py:1197-1232,1442-1456`。权重 `text_config.layer_types` 是 40 层：30 层 GDN、10 层 full attention；每层均有 256 选 8 的 MoE，且有独立 shared expert。这些是**调用次数与结构**，不是各部分的已测耗时。
- 主要重复工作候选：GDN 的 FP8 投影、XPU 自定义 recurrent/conv op、输出投影（`qwen_gdn_linear_attn.py:976-1023`，`_xpu_ops.py:137-211`）；full attention 的 QKV、Q/K norm、RoPE、KV 读取和输出投影（`qwen3_next.py:271-467`）。XPU 的 QK norm + RoPE + gate 专用融合门禁要求 CUDA，故不能假定在这 10 层命中（`qwen3_next.py:372-393`）。每层的 routed MoE + shared expert 分开执行（`qwen3_next.py:128-267`）；在线 FP8 线性实际走 XPU W8A16 op，FP8 MoE 走 XPU backend（`kernels/linear/scaled_mm/xpu.py:122-180`，`fused_moe/experts/xpu_moe.py:34-173`）。这些项在 bsz=1 下可能受小 GEMV、launch 和权重读取影响，**没有 trace 不给出百分比分摊或 roofline 结论**。
- TP=2 还需跨卡归约：GDN/attention 的 `RowParallelLinear` 默认归约输出（`qwen_gdn_linear_attn.py:496-503`，`qwen3_next.py:313-328`，`linear.py:1771-1778`），MoE 默认在非 sequence-parallel 路径做最终归约（`fused_moe/layer.py:118-120,230-236`，`fused_moe/runner/moe_runner.py:482-512`）。v0.30 的 XPU 平台明确不支持若干编译期融合，包括 allreduce+RMSNorm 和 GEMM+通信（`platforms/xpu.py:313-333`）。不能仅凭源码判断归约占总 TPOT 的比例。
- **编译与 eager 并非仅切换 graph**：`--enforce-eager` 把 mode 设为 NONE；mode 3 会默认禁用 custom ops（`config/vllm.py:1546-1552,1608-1615`），XPU 编译优先选 `native` 而非 eager 的 `vllm_c`（`platforms/xpu.py:484-494`）。`GemmaRMSNorm` 在前者走 native IR，在后者若 wheel 提供 fused op 则走 XPU fused kernel（`model_executor/custom_op.py:174-203`，`layers/layernorm.py:130-194`）。GDN XPU op 是独立的编译分图边界（`config/compilation.py:767-787`）；graph mode NONE 且 XPU graph 环境变量关闭，编译不能消除每步所有 kernel/通信提交（`platforms/xpu.py:293-305`）。这些差异足以解释为何“成功编译”**不保证变快**，但哪项导致本次差值仍须逐核/host trace。
- 此次编译启动额外设置 `VLLM_USE_V2_MODEL_RUNNER=0`；之前 eager 启动没有显式设置这个环境变量。代码默认会视 Triton 与特性支持情况选择 V2，XPU worker 进而实例化不同的 V1/V2 runner（`config/vllm.py:674-718`，`v1/worker/xpu_worker.py:168-194`）。此前没有留存 runner 选择的日志证据，所以这也是**未隔离的变量**；不能把观察差值纯粹当作 compile vs eager A/B。

若要给各项明确耗时及可比的编译回退结论，需要在两组都固定同一 runner、时钟/设备、提示词和热态的条件下交错重测，并在真实 decode 中记录 `gpu_model_runner` 的 preprocess/forward/sample、XPU kernel 名与次数、逐核时间、allreduce 时间及 host 空档；不能用单独的 kernel 自耗时替代 TPOT。

## 图片和视频：成功路径是诊断配置，不是当前分支的默认行为

### 原始镜像的阻塞

权重自带 `video_processing_minicpmv4_7.py` 导入 `transformers.video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING`；镜像内的 `5.16.1` 没有这个仅用于文档装饰器的常量，`AutoProcessor.from_pretrained` 报 `ImportError`。因此即使只发送图片，也无法在原始处理器配置下直接加载。测试过的发布版 `5.17.0` 以及当时的源码主分支也没有该常量；**只升级 Transformers 不能消除权重自带处理器的这一导入问题**。

曾在一次性容器里将此文档常量临时设为空字符串，以便排查后续图像和视频路径。这是诊断补丁，不是产品修复；补丁与生成的测试媒体均已删除。

### 无文档常量补丁的另一条实测路径

在一次性容器中安装 Transformers `5.18.0.dev0`（源码提交 `07338b6c74a578868368e6e549dea83414e4b8cb`），使用其内置 `MiniCPMV4_7Processor` 和内置的 4.6 图像、视频处理器。同时**只在容器中以只读文件挂载覆盖**两份权重元数据：

1. `preprocessor_config.json`：将 `image_processor_type` 指向 `MiniCPMV4_6ImageProcessor`，移除指向权重自带图像/视频代码的 `auto_map`；其余字段保持不变。
2. `processor_config.json`：保留 `processor_class=MiniCPMV4_7Processor`，移除指向权重自带处理器代码的 `auto_map`。

保持原始 `config.json` 不变，并保留 `--trust-remote-code`，使模型仍加载权重自带 Config。若直接使用 Transformers 的原生 Config，当前回移模型初始化会因缺少 `drop_vision_last_layer` 失败。上述元数据覆盖**没有写回原始权重，也没有提交到 dev 分支**。

多模态服务还需 `--limit-mm-per-prompt '{"image":1,"video":1}'`，以及与纯文本相同的 TP=2、FP8、eager、源码挂载和 XPU 设备设置。43.4 秒本地视频的实测另设 `--media-io-kwargs '{"video":{"num_frames":8}}'`、`--max-model-len 6144`、`--max-num-batched-tokens 4096`、`--gpu-memory-utilization 0.82`，并仅向服务开放只读挂载的视频目录。该服务只监听本机回环地址，现已停止。

### 实际输出与可推断范围

| 测试输入 | 实测结果 | 注意事项 |
| --- | --- | --- |
| 64×64 红图、绿图 | 分别回答 `Red`、`Green` | 单图基本功能成立 |
| 1024×768 红图 | 处理为五个切片，回答 `Red` | 不代表所有分辨率或细节识别准确 |
| 两帧红→蓝视频 | 回答 `Red Blue` | 仅验证短视频画面 |
| 2 fps、两秒的红绿蓝黄四帧视频 | 问末帧，回答 `Yellow` | 证实 vLLM 看到了全部四帧；这与参考抽帧不同 |
| MMMU `validation_Accounting_1` 单图选择题 | 答 `B`，与标注相同 | 无图、白图也答 `B`，单凭选择题命中不能证明读图 |
| 同一张 MMMU 图片中的三个表格值 | 原图答 `$10, $2, $14`；白图和无图答另一组错误值 | 支持“该用例确实利用了画面”的判断；追问不属于官方计分 |
| 本地 `05f39ac7d571df55.mp4`，43.4 秒 | 视频描述提到户外石板路、建筑与水、人物动作及白色→黑色→粉色服装变化 | 与抽样画面基本一致；简述漏掉最后的白上衣、蓝色牛仔下装，单独追问末帧则答对“白色、蓝色” |
| 同一视频的三个时间点服装颜色 | 画面参照为白、黑、粉；模型答白、白、粉 | 2/3，不能算作完整时序理解成功；单独问中段时视频和对应单帧均答“黑” |

上表的 43.4 秒本地视频为 1408×2560、30 fps、共 1,302 帧，另有 AAC 音轨。此次 `num_frames=8` 的实际 vLLM 解码索引是 `[0, 185, 371, 557, 743, 929, 1115, 1301]`；视频描述请求约占 4,843 个 prompt token。视频原文件只读使用、未经转码，**音轨未作为模型输入**。描述题由本机自行拟定，不是 Video-MME-v2 官方题，也没有据此计算基准分数。无视频对照误将场景描述为室内，进一步说明带视频的回答确实受到画面影响。

### 同一次启动：纯文本、用户图片和用户视频

2026-09-28，在**同一个服务进程**上依次发送以下三个聊天请求，没有重启或切换配置。使用 XPU 0/1、TP=2、online FP8、BF16、eager、`max_model_len=6144`、`max_num_batched_tokens=4096`、`max_num_seqs=1`、`gpu_memory_utilization=0.82`、`limit_mm_per_prompt={"image":1,"video":1}` 和 `media_io_kwargs={"video":{"num_frames":8}}`。使用前述 Transformers `5.18.0.dev0` 固定提交和两份只读处理器元数据覆盖；用户媒体以只读文件挂载，`--allowed-local-media-path /media`。三次均设置 `temperature=0`、`max_tokens=256` 和 `chat_template_kwargs={"enable_thinking":false}`。启动日志实际选中 `XPUW8A16FP8LinearKernel` 与 XPU FP8 MoE 后端。每条请求均返回 HTTP 200、`finish_reason=stop`。

1. **纯文本输入**：`17 乘以 24 等于多少？只回答数字。`；模型原始回答：`408`；prompt tokens：27。
2. **图片输入**：`/home/intel/xiangyu/copilot_workspace/image.png`；提问：`请简要描述图片中的主体，并指出胸前的英文文字。`；prompt tokens：92。

   ![本次输入图片：白色 intel 机器人](../image.png)

   模型原始回答：

   > 图片中的主体是一个白色的宇航员造型玩偶，它有着银色的手套和靴子，头盔面罩上显示着蓝色的发光符号。
   >
   > 玩偶胸前的英文文字是：**intel**

3. **视频输入**：`/home/intel/xiangyu/copilot_workspace/tmp.mp4`；提问：`请只根据画面，简要描述视频中的人物、场景，以及他做了什么。不要推断声音。`；prompt tokens：3,705。

   <video controls preload="metadata" width="640" src="../tmp.mp4"></video>

   如果本地 Markdown 阅读器不支持内嵌视频，可[打开同一个本地视频文件](../tmp.mp4)。模型原始回答：

   > 画面中，一个戴着灰色鸭舌帽、身穿紫色西装外套的男孩站在一个红色的帐篷旁。背景是一个集市，可以看到摆满书籍的摊位和其他路人。
   >
   > 男孩手里拿着一枚金币，他先是低头看着金币，随后抬起头，眼神显得有些犹豫和纠结，并转头向一侧张望。

图片原文件是 292×470 的白色 `intel` 机器人；视频原文件长 6.833 秒、1280×720、24 fps，显式均匀取 8 帧。这验证了同一部署下三种输入都能生成回答，**不保证描述的每个细节准确**；音轨没有作为输入。原始权重、用户媒体和已发布分支均未修改。
上述图片预览和视频播放器使用指向工作区原文件的相对路径；在本地查看时需保持文档与媒体的相对位置，媒体未复制进仓库或上传。

## 已知缺口和未验证项

1. **交付配置缺失**：已发布的 dev 分支只有 vLLM 回移代码，没有 Transformers 开发版依赖固定方式、两份权重元数据覆盖或正式处理器兼容修复。在原始镜像和原始权重上直接开多模态仍会遇到前述 `ImportError`。
2. **视频抽帧语义不同**：权重自带视频处理器对普通短视频默认约每秒取一帧；vLLM 的 `VideoMediaIO` 默认最多均匀取 32 帧。校准样本的参考索引为 `[0, 2]`，vLLM 实际为 `[0, 1, 2, 3]`。本地长视频的 8 帧是显式设置，不是模型默认的 1 fps 取样；不能把其结果当作原生处理器语义下的结果。
3. **图像/视频混排顺序**：`minicpmv4_6.py` 的 `embed_multimodal` 无论输入顺序如何均先拼图像、再拼视频；`models/utils.py` 按展平顺序填充提示词的多模态占位符。结构性对照验证了反序时的映射风险。绿静态图单独和“图在前”混排时答 `Green`，但“视频在前、绿图在后”的混排请求把静态图答成 `Red`。不能从此推断所有混排请求均失败。
4. **视频专用服务限额**：视觉塔由 `_mark_tower_model(..., {"image"})` 标记，仅凭图片限额决定是否跳过。`image=0, video>0` 的服务有跳过视频所需视觉塔的代码级风险；尚未以独立服务复验。已跑通的服务始终设 `image=1, video=1`。
5. **高级视频处理不等价**：vLLM 视频路径逐帧调用图像处理器，没有实现权重视频处理器 `stack_frames>1` 的逐秒复合帧；CPU 对照中，原生 4.6 视频处理器与权重自带 4.7 处理器在 `stack_frames=2` 时两个中间视觉单元顺序互换。视频路径还固定使用 `get_video_max_slice_num()`，没有像图像路径那样采纳请求级 `videos_kwargs.max_slice_nums`。
6. **提示词与位置字段存在差异**：默认单图和视频的视觉张量、`target_sizes` 在 CPU 对照中逐元素一致，但权重自带处理器的视频提示词比主分支原生处理器多 `<image_id>0</image_id>` 三个 token；自带处理器给出 `image_bounds`、`target_sizes_mrope` 等字段，原生处理器返回 `mm_token_type_ids`。vLLM 4.7 另从占位符和尺寸计算 canvas M-RoPE；所测样本无相关回退告警，不构成所有形状和顺序下的正确性证明。
7. **质量与覆盖面**：没有做完整文本或多模态准确率评测，也没有验证视频音频、字幕、多视频、大批量并发、TP>2、图模式或所有长视频分辨率。高分辨率视频每帧可产生多个切片；本地视频原始分辨率约 578 个视觉占位 token/帧，默认 32 帧可能远超所测上下文长度，应在请求前明确预算并核对实际解码帧数。

**当前可确认的是**：纯文本在原始容器上完成 TP=2 online FP8 的基础功能验证；图片和纯视频画面在隔离的处理器兼容配置下完成端到端验证；原始部署、反序图文/视频混排和完整评测仍未达到可交付的验证标准。所有诊断服务均已停止。
