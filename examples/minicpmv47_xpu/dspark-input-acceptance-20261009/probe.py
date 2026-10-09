"""Observe acceptance and source-continuation overlap; claim no timings."""

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:31065"


def post(endpoint, body):
    request = urllib.request.Request(
        BASE + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        assert response.status == 200
        return json.load(response)


def overlap(generated, reference):
    prefix = 0
    for a, b in zip(generated, reference, strict=False):
        if a != b:
            break
        prefix += 1
    previous = [0] * (len(reference) + 1)
    longest = 0
    for token in generated:
        current = [0] * (len(reference) + 1)
        for index, ref in enumerate(reference, start=1):
            if token == ref:
                current[index] = previous[index - 1] + 1
                longest = max(longest, current[index])
        previous = current
    return {
        "exact_prefix_tokens": prefix,
        "exact_prefix_fraction": prefix / len(generated),
        "longest_contiguous_source_overlap_tokens": longest,
        "reference_tokens": len(reference),
    }


def main():
    plan = json.loads((ROOT / "PLAN.json").read_text())
    assert (
        hashlib.sha256((ROOT / "cases.json").read_bytes()).hexdigest()
        == plan["cases_sha256"]
    )
    cases = json.loads((ROOT / "cases.json").read_text())
    assert not (ROOT / "results.json").exists()
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if (ROOT / "server.log").exists():
            log = (ROOT / "server.log").read_text()
            if (
                "Worker failed with error" in log
                or "Engine core initialization failed" in log
            ):
                raise RuntimeError("Server failed; retained server.log")
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as response:
                if response.status == 200:
                    break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    else:
        raise RuntimeError("Service readiness deadline exceeded")
    common = {
        "model": "MiniCPM-V-4.7-1008",
        "temperature": plan["temperature"],
        "seed": plan["seed"],
        "max_tokens": plan["max_tokens"],
        "ignore_eos": False,
        "return_token_ids": True,
    }
    warmup = post(
        "/v1/completions",
        {**common, "prompt": cases[0]["prompt_token_ids"], "max_tokens": 32},
    )
    (ROOT / "warmup.json").write_text(
        json.dumps(warmup, ensure_ascii=False, indent=2) + "\n"
    )
    results = []
    for mode in plan["modes"]:
        for case in cases:
            if mode == "raw_completion":
                endpoint = "/v1/completions"
                body = {**common, "prompt": case["prompt_token_ids"]}
            else:
                endpoint = "/v1/chat/completions"
                body = {
                    **common,
                    "messages": [
                        {
                            "role": "user",
                            "content": plan["chat_instruction"] + case["text"],
                        }
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            response = post(endpoint, body)
            choice = response["choices"][0]
            ids = choice["token_ids"]
            assert ids and len(ids) == response["usage"]["completion_tokens"]
            if mode == "raw_completion":
                assert response["usage"]["prompt_tokens"] == 2048
            metrics = response["metrics"]["speculative_decoding"]
            hist = metrics["acceptance_histogram"]
            assert len(hist) == 4 and sum(hist) == metrics["num_spec_steps"]
            assert (
                sum(i * n for i, n in enumerate(hist))
                == metrics["num_accepted_draft_tokens"]
            )
            accepted = metrics["per_step_accepted"]
            drafted = metrics["per_step_drafted"]
            assert accepted is not None and drafted is not None
            row = {
                "mode": mode,
                "case": case["label"],
                "source_token_start": case["source_token_start"],
                "source_token_end_exclusive": case["source_token_end_exclusive"],
                "source_window_tokens": 2048,
                "text_sha256": case["text_sha256"],
                "source_overlap": overlap(ids, case["source_next_token_ids"]),
                "terminal_preclip_overshoot": 1
                + metrics["num_spec_steps"]
                + metrics["num_accepted_draft_tokens"]
                - len(ids),
                "fully_accepted_round_fraction": hist[-1] / sum(hist),
                "response": response,
            }
            results.append(row)
            (ROOT / "results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n"
            )
            print(
                mode,
                case["label"],
                "acceptance",
                round(metrics["draft_acceptance_rate"], 4),
                "AL",
                round(metrics["mean_acceptance_length"], 4),
                "tokens",
                len(ids),
                "source prefix",
                row["source_overlap"]["exact_prefix_tokens"],
                flush=True,
            )
    summary = {
        "evidence_class": plan["evidence_class"],
        "num_requests": len(results),
        "profile": plan["profile"],
        "num_speculative_tokens": 3,
        "latency_claim": False,
        "formal_acceptance": False,
        "modes": {},
    }
    for mode in plan["modes"]:
        selected = [r for r in results if r["mode"] == mode]
        metrics = [r["response"]["metrics"]["speculative_decoding"] for r in selected]
        steps = sum(m["num_spec_steps"] for m in metrics)
        accepted = sum(m["num_accepted_draft_tokens"] for m in metrics)
        proposed = sum(m["num_draft_tokens"] for m in metrics)
        summary["modes"][mode] = {
            "requests": len(selected),
            "steps": steps,
            "accepted": accepted,
            "proposed": proposed,
            "draft_acceptance_rate": accepted / proposed,
            "mean_acceptance_length": 1 + accepted / steps,
            "fully_accepted_round_fraction": sum(
                m["acceptance_histogram"][-1] for m in metrics
            )
            / steps,
            "total_output_tokens": sum(
                r["response"]["usage"]["completion_tokens"] for r in selected
            ),
            "max_exact_source_prefix_tokens": max(
                r["source_overlap"]["exact_prefix_tokens"] for r in selected
            ),
            "length_capped_requests": sum(
                r["response"]["choices"][0]["finish_reason"] == "length"
                for r in selected
            ),
            "per_window_acceptance": [m["draft_acceptance_rate"] for m in metrics],
        }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
