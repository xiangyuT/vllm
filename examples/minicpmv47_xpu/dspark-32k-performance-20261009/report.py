"""Rebuild the current documentation section from retained sample summaries."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent / "DSPARK_PERFORMANCE_20261008.md"
START = "<!-- real-text-32k-performance:start -->"
END = "<!-- real-text-32k-performance:end -->"
NAMES = {
    "baseline_FULL": "普通 FULL，无 DSpark",
    "k3_bf16_FULL": "DSpark k3 FULL，BF16 草稿",
    "k3_heads_fp8_FULL": "DSpark k3 FULL，FP8 Markov/草稿词表",
}


def main():
    plan = json.loads((ROOT / "PLAN.json").read_text())
    lines = [
        START,
        "## 2026-10-09 真实正文 32K 自然 EOS 诊断（历史）",
        "",
        "此前自然 EOS 请求已完成，但 DSpark 输出 31 token、普通 FULL 输出 256 token，完整请求总耗时不能作为同等输出长度的加速比较。用户已要求忽略 EOS，最新固定 256 token 对照见上节。",
        "",
        "本轮按用户指定，截取《战争与和平》正文开头的 **32,768 个主模型 tokenizer token**，使用原始 completion 作自然续写。直接传入 token ID，服务端输入精确为 32,768 token，没有 chat template、重复填充或正文之外的续写指令。[固定输入/参数](dspark-32k-performance-20261009/PLAN.json)、[正文前缀](dspark-32k-performance-20261009/prompt.txt)",
        "",
        "| 配置 | TTFT 中位数 ms | 平均 decode TPOT 中位数 ms/token | 总完成时间中位数 ms | 草稿接受率 | 平均接受长度 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    summaries = []
    for profile in plan["profiles"]:
        path = ROOT / profile / "summary.json"
        if not path.exists():
            lines.append(f"| {NAMES[profile]} | 待采集 | — | — | — | — |")
            continue
        summary = json.loads(path.read_text())
        summaries.append(summary)
        medians = summary["medians"]
        spec = summary.get("speculative")
        acceptance = f"{100 * spec['draft_acceptance_rate']:.2f}%" if spec else "—"
        length = f"{spec['mean_acceptance_length']:.3f}" if spec else "—"
        lines.append(
            f"| {NAMES[profile]} | {medians['ttft_ms']:.3f} | {medians['tpot_ms']:.3f} | {medians['total_ms']:.3f} | {acceptance} | {length} |"
        )
    lines += [
        "",
        "各组先完成四次短前缀预热、两次完整 32K 前缀预热，再采三次有效样本，保留全部结果，不删除较慢样本。输出最多 256 token，temperature=0、seed=7、`ignore_eos=false`，不设最小输出长度。Prefix cache 关闭，单并发，TP2，物理 GPU 0/1，V2，BF16 激活、主模型 online FP8、FULL。为输入和生成保留空间，`max_model_len=33280`；每轮最多 4096 个 batched token，32K prefill 分块执行。模型加载、编译、capture 和预热均不计入有效样本。",
        "",
        "TTFT 从客户端发请求到首个带生成 token ID 的 SSE chunk；文本首 token 时间另行保存。Decode TPOT 为 `(DONE 时间−首个生成 token chunk 时间)/(实际输出 token 数−首 chunk token 数)`，是包含客户端/服务流式开销的平均指标；DSpark 一个 chunk 可含多个 token，不能把 chunk 间隔当逐 token ITL。总完成时间直接按请求起止测量，不用各项中位数相加。",
        "",
        "普通 FULL 与两组 DSpark 都保留当前 #60539/BF16 source/native 覆盖；DSpark 同时使用既有 native non-causal paged attention。两组 DSpark 都是 k3，差别仅为草稿 Markov/词表权重格式和投影后端。三组均不在测量期间启用 profiler、路由 snapshot 或额外 GPU event。此为单份正文、单个截断位置的开发对照，不是完整数据集或 formal acceptance。",
        "",
    ]
    for summary in summaries:
        profile = summary["profile"]
        tokens = "/".join(str(n) for n in summary["completion_tokens"])
        lines.append(
            f"- {NAMES[profile]}：三个有效请求的输出 token 数为 {tokens}；[全部样本](dspark-32k-performance-20261009/{profile}/samples.json)、[实际启动参数](dspark-32k-performance-20261009/{profile}/launch.json)。两个 rank 的实际 XPUGraph replay 已核验。"
        )
    comparison = ROOT / "comparison.json"
    if comparison.exists():
        lines += [
            "",
            "三组实际输出长度不同：不报告完整请求总耗时的加速百分比。三组所有有效请求的前 30 个 token 相同，第 31 个 token 处 DSpark 输出 EOS；原始百分比计算和完整响应保留为历史诊断，不能升级为固定 256 token 的性能结论。[历史原始比较](dspark-32k-performance-20261009/comparison.json)",
        ]
    release = ROOT / "release.json"
    if release.exists():
        state = json.loads(release.read_text())
        if not state["running"] and not state["port_31065_listening"]:
            lines += [
                "",
                "本轮推理容器已停止、31065 端口已关闭；GPU 0/1 的 root Sysman 查询结果保存在 [释放记录](dspark-32k-performance-20261009/release.json)。未停止其他容器或重置设备。运行仓库和 native kernel 没有新增修改，未创建提交、变更版本或写远程仓库。[源码/镜像身份](dspark-32k-performance-20261009/source-identity.json)",
            ]
    else:
        lines += ["", "本轮采集仍在进行；已完成配置的阶段释放记录与样本分开保存。"]
    lines += ["", END, ""]
    section = "\n".join(lines)
    existing = DOC.read_text()
    if START in existing:
        before, rest = existing.split(START, 1)
        _, after = rest.split(END, 1)
        DOC.write_text(before + section + after.lstrip("\n"))
    else:
        first, rest = existing.split("\n", 1)
        DOC.write_text(first + "\n\n" + section + "\n" + rest.lstrip("\n"))
    print("Updated 32K performance section:", len(summaries), "completed profiles")


if __name__ == "__main__":
    main()
