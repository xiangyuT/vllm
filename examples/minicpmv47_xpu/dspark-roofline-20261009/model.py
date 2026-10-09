"""Reproducible DSpark round roofline from local configuration and evidence.

Logical DRAM scenarios are explicit; neither expert routing nor cache traffic is
invented. Balanced collectives are measured references, not model peer waits.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parent
MAIN = Path("/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-1008/config.json")
DRAFT = Path("/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-DSpark-1008/config.json")
SPEC_URL = "https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html"


def read(path):
    return json.loads(path.read_text())


def hardware():
    rows = read(ROOT / "compute-reference2.json")["rows"]
    bw = min(
        r["reference_bandwidth_bytes_per_second"]
        for r in rows
        if r["kind"] == "stream_copy"
    )
    matrix = min(
        r["reference_flop_per_second"]
        for r in rows
        if r["kind"] == "bf16_gemm_reference" and r["mnk"] == [4096] * 3
    )
    return {
        "dram_peak_bytes_per_second": 608e9,
        "dram_stream_copy_reference_bytes_per_second": bw,
        "bf16_matrix_reference_flop_per_second": matrix,
        "fp32_vector_published_flop_per_second": 22.94e12,
        "published_source": SPEC_URL,
        "reference_policy": "use slower of the two measured GPUs; no fitted efficiency penalties",
        "limits": [
            "copy bandwidth is a read+write attainable reference, not a universal read ceiling",
            "matrix reference comes from BF16 4096 cubed, not a proof every shape attains it",
            "FP8 weight storage does not imply FP8 compute throughput",
        ],
    }


def comm(kind, rows, graph):
    candidates = []
    for rank in range(2):
        data = read(ROOT / f"collective-reference-rank{rank}.json")["rows"]
        hit = next(
            r
            for r in data
            if r["kind"] == kind and r["rows"] == rows and r["graph"] == graph
        )
        assert len(hit["samples_us"]) == 3
        candidates.append(hit["median_us"])
    return max(candidates) / 1000


def ledger(k, context, markov_bytes=2):
    main = read(MAIN)["text_config"]
    draft = read(DRAFT)
    assert main["hidden_size"] == draft["hidden_size"] == 2048
    assert main["num_experts"] == 256 and main["num_experts_per_tok"] == 8
    assert main["num_hidden_layers"] == 40 and draft["num_hidden_layers"] == 5
    assert main["mamba_ssm_dtype"] == "float32"
    tp = 2
    h = main["hidden_size"]
    m = k + 1 if k else 1
    nl = main["layer_types"].count("linear_attention")
    nf = main["layer_types"].count("full_attention")
    hv = main["linear_num_value_heads"] // tp
    hk = main["linear_num_key_heads"] // tp
    kd = main["linear_key_head_dim"]
    vd = main["linear_value_head_dim"]
    hd = main["head_dim"]
    qh = main["num_attention_heads"] // tp
    kvh = main["num_key_value_heads"] // tp
    dq = draft["num_attention_heads"] // tp
    dkv = draft["num_key_value_heads"] // tp
    # TP head padding is verified in retained aten::mm traces.
    vocab = main["vocab_size"]
    padded = ((vocab + 127) // 128) * 128
    shard = padded // tp
    rows = []

    def add(
        name,
        flops,
        low,
        high=None,
        count=1,
        domain="matrix",
        proof="source/config",
        note="",
    ):
        rows.append(
            {
                "name": name,
                "count_per_round": count,
                "logical_flops_per_call": flops,
                "logical_bytes_per_call_low": low,
                "logical_bytes_per_call_high": low if high is None else high,
                "compute_domain": domain,
                "geometry_evidence": proof,
                "note": note,
            }
        )

    def mm(name, r, kk, nn, weight_bytes, count=1):
        add(
            name,
            2 * r * kk * nn,
            kk * nn * weight_bytes + 2 * (r * kk + r * nn),
            count=count,
            proof="exact retained mm / fp8_gemm_w8a16 correlation + current configured row count",
            note=f"[{r},{kk}] x [{kk},{nn}]; weights {weight_bytes} bytes/element; read once/call",
        )

    add("target.embedding", 0, 2 * m * h + 8 * m, domain="vector")
    # These six projection geometries and counts are independently recovered
    # from dspark_NONE_fixed rank0/rank1, not inferred from nearby CPU ops.
    mm("target.GDN.qkvz", m, h, (2 * hk * kd + 2 * hv * vd), 1, nl)
    mm("target.GDN.ba", m, h, 2 * hv, 1, nl)
    mm("target.attention.output", m, h, h, 1, nl + nf)
    mm("target.full_attention.qkv_gate", m, h, 2 * qh * hd + 2 * kvh * hd, 1, nf)
    mm("target.router", m, h, main["num_experts"], 2, 40)
    mm("target.shared_gate", m, h, 1, 2, 40)
    si = main["shared_expert_intermediate_size"] // tp
    mm("target.shared_expert.gate_up", m, h, 2 * si, 1, 40)
    mm("target.shared_expert.down", m, si, h, 1, 40)
    ei = main["moe_intermediate_size"] // tp
    topk = main["num_experts_per_tok"]
    experts = main["num_experts"]
    ulo = topk
    uhi = min(experts, m * topk)
    # Both scenarios execute the same useful expert arithmetic. Weight bytes
    # differ with overlap; routed intermediate tensors are materialized here.
    expert_activations = 2 * m * topk * (h + 2 * ei + ei + h)
    one_expert_weight = 3 * h * ei
    add(
        "target.routed_experts",
        6 * m * topk * h * ei,
        ulo * one_expert_weight + expert_activations,
        uhi * one_expert_weight + expert_activations,
        count=40,
        note=f"unique experts/layer in [{ulo},{uhi}]; logical FLOPs exclude DPAS padding, actual IDs unknown",
    )

    state = hv * kd * vd * 4
    gdn_aux = 2 * m * (2 * hk * kd + hv * vd + hv * vd) + 4 * m * 2 * hv
    add(
        "target.GDN.recurrence_and_rollback",
        m * (7 * hv * kd * vd + 2 * hv * vd + 6 * hv * kd),
        state * (m + 1) + gdn_aux,
        count=nl,
        domain="vector",
        note="one initial FP32 state read; m independent FP32 rollback-state writes; sequential positions",
    )
    conv = 2 * hk * kd + hv * vd
    width = main["linear_conv_kernel_dim"]
    add(
        "target.GDN.conv_and_state",
        2 * width * conv * m,
        2 * conv * ((width - 1) + m + m + (width - 1 + k) + width),
        count=nl,
        domain="vector",
        note="BF16 conv; reads initial window and new inputs, writes outputs and speculative window",
    )
    # Low: one logical shared KV-window read per layer. High: a read for
    # each query/head. This is a cache scenario envelope, not DRAM counters.
    kv_window = 2 * kvh * hd * context * 2
    full_aux = 2 * m * (qh * hd + qh * hd + 2 * kvh * hd)
    add(
        "target.full_attention.QK_PV",
        4 * m * qh * hd * context,
        kv_window + full_aux,
        kv_window * m * (qh // kvh) + full_aux,
        count=nf,
        domain="matrix",
        note="all queries share physical KV blocks; low shared-window / high query-head rescan scenario; softmax special-op latency idealized",
    )
    # Elementwise lower side allows fusion. Upper side counts distinct tensor
    # materializations; neither is asserted to equal measured DRAM traffic.
    scalar_width = 40 * 2 * h + nl * hv * vd + nf * (qh + kvh) * hd
    add(
        "target.norm_gate_rope_residual",
        8 * m * scalar_width,
        2 * (40 * 2 * h + nl * vd + nf * 2 * hd),
        2 * (40 * 2 * h + nl * vd + nf * 2 * hd) + 8 * m * scalar_width,
        domain="vector",
        note="nominal arithmetic; SFU latency and reductions not independently calibrated",
    )
    mm("target.vocab_projection", m, h, shard, 2)
    add(
        "target.greedy_rejection_and_logits",
        0,
        10 * m * vocab + 8 * m * (k + 2),
        domain="vector",
        note="BF16 logits -> FP32 logits -> greedy read; rejection/token indices; host time idealized",
    )
    if k:
        mm("draft.context_aux_projection", m, h * len(draft["target_layer_ids"]), h, 2)
        mm(
            "draft.context_KV_projection",
            m,
            h,
            2 * dkv * hd * draft["num_hidden_layers"],
            2,
        )
        ctx = 2 * m * (h + 2 * dkv * hd * draft["num_hidden_layers"])
        add(
            "draft.context_norm_rope_cache_write",
            8 * m * (h + dkv * hd * draft["num_hidden_layers"]),
            ctx,
            3 * ctx,
            domain="vector",
        )
        add("draft.embedding", 0, 2 * k * h + 8 * k, domain="vector")
        dl = draft["num_hidden_layers"]
        di = draft["intermediate_size"] // tp
        mm("draft.backbone.qkv", k, h, (dq + 2 * dkv) * hd, 2, dl)
        mm("draft.backbone.output", k, dq * hd, h, 2, dl)
        mm("draft.backbone.gate_up", k, h, 2 * di, 2, dl)
        mm("draft.backbone.down", k, di, h, 2, dl)
        dkv_window = 2 * dkv * hd * (context + k) * 2
        d_aux = 2 * k * ((dq + dkv) * hd + dq * hd)
        add(
            "draft.noncausal_attention.QK_PV",
            4 * k * dq * hd * (context + k),
            dkv_window + d_aux,
            dkv_window * k * (dq // dkv) + d_aux,
            count=dl,
            domain="matrix",
            note="query block shares context KV; noncausal query/query contribution included",
        )
        dw = dl * (2 * h + (dq + dkv) * hd + di) + 2 * h
        add(
            "draft.norm_rope_silu_residual",
            8 * k * dw,
            2 * (2 * dl * h + 2 * h),
            2 * (2 * dl * h + 2 * h) + 8 * k * dw,
            domain="vector",
        )
        mm("draft.vocab_projection", k, h, shard, 2)
        mm("draft.Markov.projection", 1, draft["markov_rank"], padded, markov_bytes, k)
        add(
            "draft.Markov.embed_bias_argmax",
            vocab,
            8 * vocab + 2 * draft["markov_rank"] + 8,
            count=k,
            domain="vector",
            note="sequential BF16 bias/base add and BF16 argmax; full draft vocabulary; identity remap",
        )
    return rows, {
        "verify_queries": m,
        "draft_queries": k,
        "context_tokens": context,
        "unique_expert_range_per_layer": [ulo, uhi],
        "target_layers": 40,
        "linear_attention_layers": nl,
        "full_attention_layers": nf,
        "draft_layers": draft["num_hidden_layers"] if k else 0,
        "padded_vocab": padded,
        "tp_vocab_shard": shard,
        "fp32_recurrent_state_bytes_per_layer_per_rank": state,
        "markov_weight_bytes_per_element": markov_bytes,
        "adaptive_confidence_head_executed": False,
    }


def modeled_round(
    k,
    graph,
    context,
    hw,
    bandwidth_key,
    markov_bytes=2,
    draft_vocab_bytes=2,
    routing=None,
):
    rows, geometry = ledger(k, context, markov_bytes)
    if draft_vocab_bytes == 1:
        node = next(r for r in rows if r["name"] == "draft.vocab_projection")
        saved = 2048 * geometry["tp_vocab_shard"]
        for edge in ["low", "high"]:
            node[f"logical_bytes_per_call_{edge}"] -= saved
        node["note"] += "; draft-only online FP8 weights, BF16 output"
    if routing:
        observed = read(routing)
        frames = observed["rounds"]
        assert len(frames) == 64 and all(
            f["m"] == geometry["verify_queries"] for f in frames
        )
        unique = [v for f in frames for v in f["unique_experts_by_router"]]
        assert len(unique) == 64 * 40
        mean_unique = statistics.mean(unique)
        node = next(r for r in rows if r["name"] == "target.routed_experts")
        one_weight = 3 * 2048 * 256
        activation = node["logical_bytes_per_call_low"] - 8 * one_weight
        for edge in ["low", "high"]:
            node[f"logical_bytes_per_call_{edge}"] = (
                mean_unique * one_weight + activation
            )
        node["note"] += (
            "; replace expert-range scenario with actual 64-round prefix mean; KV/cache uncertainty retained"
        )
        geometry["observed_routing"] = {
            "path": str(routing),
            "mean_unique_experts": mean_unique,
            "sampled_rounds": 64,
            "full_request_route_distribution_verified": False,
            "sha256": hashlib.sha256(routing.read_bytes()).hexdigest(),
        }
    bw = hw[bandwidth_key]
    totals = {"low": 0.0, "high": 0.0}
    for r in rows:
        p = (
            hw["bf16_matrix_reference_flop_per_second"]
            if r["compute_domain"] == "matrix"
            else hw["fp32_vector_published_flop_per_second"]
        )
        for edge in ["low", "high"]:
            r[f"{edge}_roof_ms"] = (
                max(
                    r["logical_flops_per_call"] / p,
                    r[f"logical_bytes_per_call_{edge}"] / bw,
                )
                * 1000
                * r["count_per_round"]
            )
            totals[edge] += r[f"{edge}_roof_ms"]
    m = k + 1 if k else 1
    communication = {
        "target_all_reduce_ms": 81 * comm("all_reduce", m, graph),
        "target_vocab_gather_ms": comm("vocab_all_gather", m, graph),
    }
    if k:
        communication.update(
            {
                "draft_all_reduce_ms": 11 * comm("all_reduce", k, graph),
                "draft_vocab_gather_ms": comm("vocab_all_gather", k, graph),
            }
        )
    communication["sum_ms"] = sum(communication.values())
    for edge in totals:
        totals[edge] += communication["sum_ms"]
    return {
        "geometry": geometry,
        "components": rows,
        "communication_reference": communication,
        "round_roof_ms": totals,
        "bandwidth_parameter": bandwidth_key,
        "aggregation": "serial dependent round; one per-rank stream, balanced communication references; no cross-rank sum",
    }


def evaluate(
    label,
    source,
    samples,
    k,
    graph,
    hw,
    markov_bytes=2,
    draft_vocab_bytes=2,
    routing=None,
):
    measured = [s for s in read(samples) if s["phase"] == "measured"]
    assert len(measured) == 3
    context = (
        measured[0]["usage"]["prompt_tokens"]
        + (measured[0]["usage"]["completion_tokens"] - 1) / 2
    )
    refs = []
    for bwkey in [
        "dram_peak_bytes_per_second",
        "dram_stream_copy_reference_bytes_per_second",
    ]:
        round_model = modeled_round(
            k, graph, context, hw, bwkey, markov_bytes, draft_vocab_bytes, routing
        )
        sample_models = []
        for s in measured:
            tokens = s["usage"]["completion_tokens"] - 1
            spec = (s.get("request_metrics") or {}).get("speculative_decoding")
            rounds = spec["num_spec_steps"] if spec else tokens
            # Scheduler observes speculative metrics before max_tokens clips
            # the final output burst. Use exact round count / emitted tokens,
            # not the reported acceptance length (which can include overshoot).
            overshoot = (
                rounds + spec["num_accepted_draft_tokens"] - tokens if spec else 0
            )
            assert 0 <= overshoot <= k
            assert (
                s["content_chunks"] == rounds + 1
                if spec
                else s["content_chunks"] == tokens + 1
            )
            observed_ms = s["tpot_ms"] * tokens
            model_ms = {e: v * rounds for e, v in round_model["round_roof_ms"].items()}
            sample_models.append(
                {
                    "index": s["index"],
                    "verify_rounds_in_decode_window": rounds,
                    "output_tokens_in_decode_window": tokens,
                    "effective_tokens_per_round": tokens / rounds,
                    "terminal_preclip_token_overshoot": overshoot,
                    "measured_decode_ms": observed_ms,
                    "measured_tpot_ms": s["tpot_ms"],
                    "modeled_decode_ms": model_ms,
                    "modeled_tpot_ms": {e: v / tokens for e, v in model_ms.items()},
                    "roofline_ratio": {e: v / observed_ms for e, v in model_ms.items()},
                }
            )
        refs.append(
            {
                "round_model": round_model,
                "samples": sample_models,
                "median_modeled_tpot_ms": {
                    e: statistics.median(s["modeled_tpot_ms"][e] for s in sample_models)
                    for e in ["low", "high"]
                },
                "median_roofline_ratio": {
                    e: statistics.median(s["roofline_ratio"][e] for s in sample_models)
                    for e in ["low", "high"]
                },
            }
        )
    return {
        "label": label,
        "measurement_source": str(samples.relative_to(PREP)),
        "measurement_date_group": source,
        "k": k,
        "graph": graph,
        "measured_tpot_median_ms": statistics.median(s["tpot_ms"] for s in measured),
        "evidence_class": "conditional_analytical_roofline_with_balanced_hardware_references",
        "models": refs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results.json")
    args = parser.parse_args()
    hw = hardware()
    profiles = []
    for label, k in [
        ("baseline_FULL", 0),
        ("k3_FULL", 3),
        ("k7_FULL", 7),
        ("m16_FULL", 15),
    ]:
        profiles.append(
            evaluate(
                label,
                "20261009",
                PREP / "dspark-iteration-20261009" / f"{label}-samples.json",
                k,
                True,
                hw,
            )
        )
    for label, k, graph in [
        ("baseline_NONE", 0, False),
        ("baseline_PIECEWISE_fresh", 0, True),
        ("baseline_FULL_retry", 0, True),
        ("dspark_NONE_fixed", 15, False),
        ("dspark_PIECEWISE", 15, True),
        ("dspark_FULL", 15, True),
    ]:
        profiles.append(
            evaluate(
                label,
                "20261008",
                PREP / "dspark-performance-20261008" / f"{label}-samples.json",
                k,
                graph,
                hw,
            )
        )
    inputs = [
        MAIN,
        DRAFT,
        ROOT / "compute-reference2.json",
        ROOT / "collective-reference-rank0.json",
        ROOT / "collective-reference-rank1.json",
        PREP / "dspark-performance-20261008/shape-analysis.json",
        Path(__file__),
    ]
    experiments = PREP / "dspark-optimization-20261009"
    for samples in sorted(experiments.glob("*-samples.json")):
        launch_path = experiments / (
            samples.name.replace("-samples.json", "-launch.json")
        )
        if not launch_path.exists():
            continue
        launch = read(launch_path)
        env = launch["environment"]
        spec = read(samples)[-1].get("request_metrics") or {}
        if not spec.get("speculative_decoding"):
            continue
        k = spec["speculative_decoding"]["num_spec_tokens"]
        label = launch["profile"]
        route = experiments / f"routing-{label}/rank0.json"
        profiles.append(
            evaluate(
                label,
                "roofline_directed_20261009",
                samples,
                k,
                True,
                hw,
                1 if env.get("MINICPM_MARKOV_FP8") == "1" else 2,
                1 if env.get("MINICPM_DRAFT_VOCAB_FP8") == "1" else 2,
                route if route.exists() else None,
            )
        )
        inputs.extend([samples, launch_path])
        if route.exists():
            inputs.append(route)
    result = {
        "evidence_class": "conditional_analytical_roofline_with_balanced_hardware_references",
        "analytical_component_model_implemented": True,
        "empirical_full_model_roofline_validated": False,
        "hardware": hw,
        "profiles": profiles,
        "ratio_definition": "modeled decode time / measured decode time per sample; no clipping above one",
        "hypotheses": [
            {
                "k": k,
                "bf16": modeled_round(
                    k, True, 2175.5, hw, "dram_peak_bytes_per_second", 2
                )["round_roof_ms"],
                "markov_fp8_only": modeled_round(
                    k, True, 2175.5, hw, "dram_peak_bytes_per_second", 1
                )["round_roof_ms"],
            }
            for k in [3, 7, 15]
        ],
        "limits": [
            "routing and DRAM/cache scenarios are model inputs; no fabricated actual expert IDs or memory counters",
            "low/high are scenario envelopes, not confidence intervals or guaranteed physical extrema",
            "logical useful FLOPs exclude padded DPAS work, SFU/reduction latency and kernel launch latency",
            "fusion reuses activation buffers; lower/upper materialization assumptions are explicit",
            "PIECEWISE uses balanced captured collective reference as an optimistic graph scenario",
            "network reference includes transfer and balanced submission; measured peer waits are not added",
            "context uses mean 2K/256 decode position; finite-window token/round counts are verified",
            "no model-derived effective bandwidth was fitted to the E2E data",
        ],
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in inputs
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for p in profiles:
        model = p["models"][0]
        print(
            p["label"],
            "TPOT",
            round(p["measured_tpot_median_ms"], 3),
            "roof TPOT",
            {e: round(v, 3) for e, v in model["median_modeled_tpot_ms"].items()},
            "ratio_pct",
            {e: round(v * 100, 2) for e, v in model["median_roofline_ratio"].items()},
        )


if __name__ == "__main__":
    main()
