"""Standalone figures from retained measurements and explicit traffic models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "dspark-optimization-20261009"
flops = 2 * 256 * 248192
bf16_bytes = 2 * (256 + 256 * 248192 + 248192)
fp8_bytes = 256 * 248192 + 2 * (256 + 248192) + 4
generic = json.loads((EXP / "micro-markov-generic.json").read_text())
flat = json.loads((EXP / "micro-flat.json").read_text())
bf = next(r for r in generic if r["logical_device"] == 0 and r["name"] == "bf16")
fg = next(
    r for r in generic if r["logical_device"] == 0 and r["name"] == "online_fp8_generic"
)
fl = min(
    (r for r in flat if r["logical_device"] == 0 and r["config"] < 6),
    key=lambda r: r["median_us"],
)
od = next(r for r in flat if r["logical_device"] == 0 and r["config"] == 6)
fig, ax = plt.subplots(figsize=(8.4, 5.0))
intensity = np.logspace(-1, 3, 300)
ax.loglog(
    intensity,
    np.minimum(608 * intensity / 1000, 170.4386),
    label="608 GB/s + BF16 matrix reference",
    color="#334155",
)
ax.loglog(
    intensity,
    np.minimum(608 * intensity / 1000, 22.94),
    ls="--",
    label="608 GB/s + published FP32 vector peak",
    color="#94a3b8",
)
for label, row, traffic, dy in [
    ("BF16 oneDNN", bf, bf16_bytes, 0.15),
    ("FP8 generic GEMV", fg, fp8_bytes, -0.15),
    ("FP8 flat GEMV", fl, fp8_bytes, 0.07),
    ("FP8 oneDNN", od, fp8_bytes, 0.17),
]:
    x = flops / traffic
    y = flops / (row["median_us"] * 1e-6) / 1e12
    ratio = max(flops / 170.4386e12, traffic / 608e9) / (row["median_us"] * 1e-6)
    ax.scatter(
        x,
        y,
        s=48,
        zorder=4,
        label=f"{label}: {row['median_us']:.1f} us; ratio {ratio:.1%}",
    )
ax.set(
    xlabel="Logical arithmetic intensity (FLOP/byte)",
    ylabel="Useful performance (TFLOP/s)",
    title="Markov projection: M=1, K=256, N=248192",
    ylim=(0.05, 220),
)
ax.grid(True, which="both", alpha=0.18)
ax.legend(fontsize=8, loc="lower right")
fig.text(
    0.08,
    0.015,
    "GPU 0 median of three blocks. Single-pass tensor traffic; DRAM counters were not measured.",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(ROOT / "markov-roofline.svg")
fig.savefig(ROOT / "markov-roofline.png", dpi=180)
plt.close(fig)
data = json.loads((ROOT / "results.json").read_text())
profiles = [
    r
    for r in data["profiles"]
    if r["measurement_date_group"] == "roofline_directed_20261009"
]
fig, ax = plt.subplots(figsize=(8.8, 4.4))
for i, p in enumerate(profiles):
    m = p["models"][0]["median_modeled_tpot_ms"]
    ax.plot([m["low"], m["high"]], [i, i], lw=6, color="#64748b", alpha=0.6)
    ax.scatter(p["measured_tpot_median_ms"], i, color="#ef4444", s=38, zorder=5)
ax.set_yticks(range(len(profiles)), [r["label"] for r in profiles])
ax.invert_yaxis()
ax.set(
    xlabel="TPOT (ms/token)",
    title="DSpark: modeled traffic scenarios and measured TPOT",
)
ax.grid(axis="x", alpha=0.2)
fig.text(
    0.04,
    0.01,
    "Gray: model scenario range; red: three-sample measured median. These are not confidence intervals.",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(ROOT / "dspark-model-comparison.svg")
fig.savefig(ROOT / "dspark-model-comparison.png", dpi=180)
print("Saved standalone SVG/PNG figures.")
