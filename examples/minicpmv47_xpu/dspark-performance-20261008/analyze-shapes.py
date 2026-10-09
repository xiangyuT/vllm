"""Analyze retained CPU/XPU traces without importing Torch or accessing devices."""

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def gemm_owner(op):
    args = op.get("args", {})
    dims = args.get("Input Dims", [])
    types = args.get("Input type", [])
    if len(dims) < 2:
        return "unattributed"
    a, b = dims[:2]
    if op.get("name") == "_xpu_C::fp8_gemm_w8a16" and a[0] == 16:
        return "target_fp8_linears_m16"
    if op.get("name") != "aten::mm" or types[:2] != ["c10::BFloat16"] * 2:
        return "other_or_unattributed"
    if a == [1, 256] and b == [256, 248192]:
        return "draft_markov_bf16_m1"
    if a == [16, 2048] and b in [[2048, 256], [2048, 1]]:
        return "target_router_and_shared_gate_m16"
    if a == [16, 2048] and b == [2048, 124096]:
        return "target_vocab_projection_m16"
    if a == [15, 2048] and b == [2048, 124096]:
        return "draft_vocab_projection_m15"
    if a[0] == 15:
        return "draft_dense_backbone_m15"
    if (a, b) in [([16, 10240], [10240, 2048]), ([16, 2048], [2048, 2560])]:
        return "draft_context_projection_m16"
    if a == [1, 2048] and b == [2048, 124096]:
        return "baseline_vocab_projection_m1"
    return "other_or_unattributed"


results = []
for label in ["baseline_NONE", "dspark_NONE_fixed"]:
    for path in sorted((ROOT / "traces" / label).glob("*.gz")):
        events = json.load(gzip.open(path, "rt"))["traceEvents"]
        cpu = {e["args"]["External id"]: e for e in events
               if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})}
        runtime = {e["args"].get("correlation"): e for e in events
                   if e.get("cat") in ["xpu_runtime", "xpu_driver"]
                   and e.get("args", {}).get("External id")}
        kernels = [e for e in events if e.get("cat") == "kernel"]
        per_name = defaultdict(lambda: {"count": 0, "sum_us": 0.0})
        per_owner = defaultdict(lambda: {"count": 0, "sum_us": 0.0, "shapes": Counter()})
        matrix_rows = defaultdict(lambda: {"count": 0, "sum_us": 0.0})
        for e in kernels:
            per_name[e["name"]]["count"] += 1
            per_name[e["name"]]["sum_us"] += e["dur"]
            if e["name"] != "gemm_kernel":
                continue
            args = e["args"]
            rt = runtime.get(args.get("correlation"), {})
            ext = args.get("External id") or rt.get("args", {}).get("External id")
            op = cpu.get(ext, {})
            owner = gemm_owner(op)
            shape = json.dumps(op.get("args", {}).get("Input Dims", []))
            per_owner[owner]["count"] += 1
            per_owner[owner]["sum_us"] += e["dur"]
            per_owner[owner]["shapes"][shape] += 1
            key = (op.get("name", "unattributed"), shape,
                   json.dumps(op.get("args", {}).get("Input type", [])),
                   json.dumps(op.get("args", {}).get("Input Strides", [])))
            matrix_rows[key]["count"] += 1
            matrix_rows[key]["sum_us"] += e["dur"]
        comm = [e for e in kernels if e["name"].startswith("oneccl")
                or "allgatherv" in e["name"]]
        noncomm_us = sum(e["dur"] for e in kernels) - sum(e["dur"] for e in comm)
        result = {
            "label": label, "trace": str(path.relative_to(ROOT.parent)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "profiled_iterations": 4,
            "cpu_op_count": sum(e.get("cat") == "cpu_op" for e in events),
            "kernel_count": len(kernels),
            "noncommunication_kernel_sum_ms_per_iteration": noncomm_us / 4000,
            "communication_kernel_sum_ms_per_iteration": sum(e["dur"] for e in comm) / 4000,
            "gemm_attribution": dict(per_owner),
            "matrix_shapes": [{"op": k[0], "shapes": json.loads(k[1]),
                               "types": json.loads(k[2]),
                               "strides": json.loads(k[3]), **v}
                              for k, v in matrix_rows.items()],
            "kernel_names": dict(per_name),
            "allreduce_cpu_shapes": dict(Counter(json.dumps(e["args"].get("Input Dims"))
                  for e in events if e.get("cat") == "cpu_op"
                  and e["name"] in ["vllm::all_reduce", "vllm::xpu_all_reduce_"])),
        }
        for v in result["gemm_attribution"].values():
            v["ms_per_iteration"] = v["sum_us"] / 4000
            v["fraction_of_noncommunication_kernel_sum"] = v["sum_us"] / noncomm_us
        results.append(result)

rounds = {}
for label in ["dspark_NONE_fixed", "dspark_PIECEWISE", "dspark_FULL"]:
    samples = [s for s in json.loads((ROOT / (label + "-samples.json")).read_text())
               if s["phase"] == "measured"]
    rows = []
    for s in samples:
        spec = s["request_metrics"]["speculative_decoding"]
        decode_ms = s["tpot_ms"] * 255
        rows.append({"index": s["index"], "decode_window_ms": decode_ms,
                     "speculative_rounds": spec["num_spec_steps"],
                     "decode_window_ms_per_reported_round": decode_ms / spec["num_spec_steps"]})
    rounds[label] = {"samples": rows, "median_ms_per_reported_round": statistics.median(
                    r["decode_window_ms_per_reported_round"] for r in rows)}

output = {
    "evidence_class": "static_analysis_of_retained_development_traces",
    "method": "GPU gemm -> External id (or correlation -> runtime External id) -> CPU op with exact shapes/types; kernel-name sums kept separately",
    "limits": ["four profiled iterations per trace, not three independent E2E measurements",
               "do not sum across ranks, add CPU and GPU durations, or treat communication waits as bandwidth",
               "no valid DSpark PIECEWISE/FULL XPU trace; per-round E2E proxy is not kernel time",
               "baseline iteration emits one token; DSpark iteration verifies 16 queries and drafts 15"],
    "traces": results, "end_to_end_round_proxy": rounds,
    "markov_weight_logical_bytes": 256 * 248192 * 2,
}
(ROOT / "shape-analysis.json").write_text(json.dumps(output, indent=2) + "\n")
for r in results:
    print(r["label"], Path(r["trace"]).name)
    for owner, v in r["gemm_attribution"].items():
        print(owner, v["count"], round(v["ms_per_iteration"], 4))
print("End-to-end round proxies", {k: v["median_ms_per_reported_round"] for k, v in rounds.items()})
