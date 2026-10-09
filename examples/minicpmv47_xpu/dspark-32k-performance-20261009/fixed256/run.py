"""Run three isolated server lifecycles and release devices on every exit."""

import hashlib
import json
import re
import shlex
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parents[1]
NAME = "codex-minicpm47-32k-performance-20261009"
IMAGE = "sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566"
BASE = "http://127.0.0.1:31065"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inspect():
    return json.loads(subprocess.check_output(["docker", "inspect", NAME], text=True))[
        0
    ]


def processes(destination, prefix):
    queries = {}
    for rank in [0, 1]:
        helper = f"codex-minicpm47-root-ps-{rank}-20261009"
        meta = json.loads(
            subprocess.check_output(["docker", "inspect", helper], text=True)
        )[0]
        assert (
            meta["Config"]["Labels"]["codex.task"] == "minicpm-v47-main-prep-20261008"
        )
        assert meta["HostConfig"]["PidMode"] == "host"
        result = subprocess.run(
            ["docker", "start", "-a", helper],
            text=True,
            capture_output=True,
            check=True,
        )
        queries[str(rank)] = result.stdout + result.stderr
        (destination / f"{prefix}-root-ps-gpu{rank}.txt").write_text(queries[str(rank)])
    return queries


def create():
    original = json.loads((ROOT.parent / "container.json").read_text())[0]
    current = inspect()
    assert current["Id"] == original["Id"]
    assert current["Image"] == IMAGE
    assert current["Config"]["Labels"]["codex.task"] == "minicpm-v47-main-prep-20261008"
    assert current["Config"]["Labels"]["codex.stage"] == "32k-performance-20261009"
    assert not current["State"]["Running"]
    assert any(
        m["Destination"] == "/workspace/evidence"
        and m["Source"] == str(ROOT.parent.resolve())
        and m["RW"]
        for m in current["Mounts"]
    )
    write(ROOT / "container.json", [current])
    subprocess.run(["docker", "start", NAME], check=True, capture_output=True)
    identity = subprocess.check_output(
        [
            "docker",
            "exec",
            NAME,
            "/workspace/runtime/.venv/bin/python",
            "/workspace/preparation/dspark-optimization-20261009/check-device.py",
        ],
        text=True,
    )
    (ROOT / "device-identity.json").write_text(identity)
    props = json.loads(identity)
    plan = json.loads((ROOT / "PLAN.json").read_text())
    assert [d["sysman_uuid"] for d in props["devices"]] == plan["expected_sysman_uuid"]
    print(
        "Reused verified owned container and compiled cache", current["Id"], flush=True
    )


def stop(destination):
    current = inspect()
    original = json.loads((ROOT / "container.json").read_text())[0]
    assert current["Id"] == original["Id"]
    assert current["Config"]["Labels"]["codex.stage"] == "32k-performance-20261009"
    if current["State"]["Running"]:
        subprocess.run(
            ["docker", "stop", "--time", "10", current["Id"]],
            check=True,
            capture_output=True,
        )
    current = inspect()
    assert not current["State"]["Running"]
    queries = processes(destination, "final")
    ports = subprocess.check_output(["ss", "-ltn", "( sport = :31065 )"], text=True)
    result = {
        "released_utc": datetime.now(timezone.utc).isoformat(),
        "container_id": current["Id"],
        "container_name": current["Name"],
        "image_id": current["Image"],
        "running": False,
        "physical_gpus": [0, 1],
        "root_sysman_queries": queries,
        "port_31065_listening": len(ports.strip().splitlines()) > 1,
        "hardware_reset": False,
        "other_containers_stopped": False,
    }
    write(destination / "release.json", result)
    return result


def start(profile, output, plan):
    if not inspect()["State"]["Running"]:
        subprocess.run(["docker", "start", NAME], check=True, capture_output=True)
    previous = PREP / "dspark-optimization-20261009"
    template = json.loads(
        (
            PREP / "dspark-iteration-20261009/baseline_FULL-launch.json"
            if profile == "baseline_FULL"
            else previous / "k3_heads_fp8-launch.json"
        ).read_text()
    )
    args = template["args"][:]
    index = args.index("--profiler-config")
    del args[index : index + 2]
    args[args.index("--max-model-len") + 1] = str(plan["max_model_len"])
    env = template["environment"].copy()
    env.update(
        {
            "MINICPM_ROUTING_AUDIT": "0",
            "VLLM_NO_USAGE_STATS": "1",
            "MINICPM_MARKOV_FP8": "1" if profile == "k3_heads_fp8_FULL" else "0",
            "MINICPM_DRAFT_VOCAB_FP8": "1" if profile == "k3_heads_fp8_FULL" else "0",
        }
    )
    if profile == "k3_bf16_FULL":
        env["MINICPM_MARKOV_BACKEND"] = "generic"
    command = ["docker", "exec"]
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += [
        "-d",
        NAME,
        "/bin/bash",
        "-lc",
        "exec "
        + shlex.join(args)
        + " > "
        + shlex.quote(f"/workspace/evidence/fixed256/{profile}/server.log")
        + " 2>&1",
    ]
    write(
        output / "launch.json",
        {
            "profile": profile,
            "container": NAME,
            "args": args,
            "environment": env,
            "command": command,
            "profiler_enabled": False,
        },
    )
    subprocess.run(command, check=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        log = output / "server.log"
        if log.exists() and any(
            s in log.read_text()
            for s in ["Worker failed with error", "Engine core initialization failed"]
        ):
            raise RuntimeError(f"Server failed: {log}")
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as response:
                if response.status == 200:
                    print(profile, "ready", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError("Service readiness deadline exceeded")


def request(prompt, output_limit, plan):
    body = {
        "model": "MiniCPM-V-4.7-1008",
        "prompt": prompt,
        "temperature": 0,
        "seed": 7,
        "max_tokens": output_limit,
        "ignore_eos": True,
        "min_tokens": output_limit,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    began = time.perf_counter()
    first = first_text = done = None
    first_batch = 0
    usage = metrics = finish = None
    generated = []
    text = []
    chunks = []
    with urllib.request.urlopen(req, timeout=600) as response:
        assert response.status == 200
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            received = time.perf_counter()
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                done = received
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            metrics = event.get("metrics") or metrics
            for choice in event.get("choices", []):
                ids = choice.get("token_ids") or []
                content = choice.get("text") or ""
                if ids:
                    if first is None:
                        first = received
                        first_batch = len(ids)
                    generated.extend(ids)
                    chunks.append(
                        {
                            "elapsed_ms": (received - began) * 1000,
                            "token_batch_count": len(ids),
                            "cumulative_tokens": len(generated),
                        }
                    )
                if content:
                    first_text = received if first_text is None else first_text
                    text.append(content)
                finish = choice.get("finish_reason") or finish
    assert first is not None and done is not None and usage is not None
    assert usage["prompt_tokens"] == len(prompt)
    assert usage["completion_tokens"] == len(generated) == output_limit
    denominator = len(generated) - first_batch
    return {
        "http_status": 200,
        "usage": usage,
        "ttft_ms": (first - began) * 1000,
        "textual_ttft_ms": (first_text - began) * 1000
        if first_text is not None
        else None,
        "decode_ms": (done - first) * 1000,
        "tpot_ms": (done - first) * 1000 / denominator if denominator > 0 else None,
        "total_ms": (done - began) * 1000,
        "first_token_batch_count": first_batch,
        "decode_token_denominator": denominator,
        "finish_reason": finish,
        "generated_text": "".join(text),
        "generated_token_ids": generated,
        "chunks": chunks,
        "request_metrics": metrics,
    }


def benchmark(profile, output, plan, prompt):
    for index in range(plan["short_warmup_requests"]):
        row = request(prompt[: plan["short_warmup_prefix_tokens"]], 16, plan)
        write(output / f"short-warmup-{index}.json", row)
    samples = []
    for index in range(plan["full_warmup_requests"] + plan["measured_requests"]):
        phase = "warmup" if index < plan["full_warmup_requests"] else "measured"
        row = request(prompt, plan["max_output_tokens"], plan)
        row.update({"phase": phase, "index": index, "profile": profile})
        assert row["usage"]["prompt_tokens"] == 32768
        assert row["tpot_ms"] is not None
        metric = (row["request_metrics"] or {}).get("speculative_decoding")
        if profile != "baseline_FULL":
            assert metric and metric["num_spec_tokens"] == 3
            assert sum(metric["acceptance_histogram"]) == metric["num_spec_steps"]
            assert (
                sum(i * n for i, n in enumerate(metric["acceptance_histogram"]))
                == metric["num_accepted_draft_tokens"]
            )
        write(output / f"{phase}-{index}.json", row)
        samples.append(row)
        write(output / "samples.json", samples)
        print(
            profile,
            phase,
            index,
            "TTFT",
            round(row["ttft_ms"], 3),
            "TPOT",
            round(row["tpot_ms"], 3),
            "total",
            round(row["total_ms"], 3),
            "tokens",
            row["usage"]["completion_tokens"],
            "acceptance",
            metric["draft_acceptance_rate"] if metric else None,
            flush=True,
        )
    measured = [r for r in samples if r["phase"] == "measured"]
    summary = {
        "profile": profile,
        "measured_samples": len(measured),
        "input_tokens": 32768,
        "medians": {
            k: statistics.median(r[k] for r in measured)
            for k in ["ttft_ms", "textual_ttft_ms", "decode_ms", "tpot_ms", "total_ms"]
        },
        "completion_tokens": [r["usage"]["completion_tokens"] for r in measured],
        "finish_reasons": [r["finish_reason"] for r in measured],
        "first_token_batch_counts": [r["first_token_batch_count"] for r in measured],
        "evidence_class": plan["evidence_class"],
        "formal_acceptance": False,
    }
    if profile != "baseline_FULL":
        metrics = [r["request_metrics"]["speculative_decoding"] for r in measured]
        steps = sum(m["num_spec_steps"] for m in metrics)
        accepted = sum(m["num_accepted_draft_tokens"] for m in metrics)
        proposed = sum(m["num_draft_tokens"] for m in metrics)
        summary["speculative"] = {
            "steps": steps,
            "accepted": accepted,
            "proposed": proposed,
            "draft_acceptance_rate": accepted / proposed,
            "mean_acceptance_length": 1 + accepted / steps,
        }
    replay = [
        s
        for s in (output / "server.log").read_text().splitlines()
        if "MINICPM FULL GRAPH REPLAY" in s
    ]
    required = ["ModelCudaGraphManager"] + (
        [] if profile == "baseline_FULL" else ["DFlashCudaGraphManager"]
    )
    assert all(
        any(
            manager in line
            and re.search(rf"\brank {rank}\b", line)
            and "graph_type XPUGraph" in line
            for line in replay
        )
        for manager in required
        for rank in [0, 1]
    )
    summary["graph_replay_evidence"] = replay
    write(output / "summary.json", summary)
    return summary


def main():
    plan = json.loads((ROOT / "PLAN.json").read_text())
    prompt_bytes = (ROOT / "prompt-token-ids.json").read_bytes()
    assert hashlib.sha256(prompt_bytes).hexdigest() == plan["input"]["token_ids_sha256"]
    prompt = json.loads(prompt_bytes)
    source = json.loads(
        (PREP / "dspark-iteration-20261009/runtime-source-manifest.json").read_text()
    )
    assert all(
        hashlib.sha256((PREP / "vllm-functional" / f).read_bytes()).hexdigest() == h
        for f, h in source["source_files"].items()
    )
    write(
        ROOT / "source-identity.json",
        {
            "base_revision": source["base_revision"],
            "runtime_source_manifest_sha256": hashlib.sha256(
                (
                    PREP / "dspark-iteration-20261009/runtime-source-manifest.json"
                ).read_bytes()
            ).hexdigest(),
            "source_files_verified": len(source["source_files"]),
            "image_id": IMAGE,
            "script_sha256": {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in ROOT.glob("*.py")
            },
            "version_changed": False,
            "remote_write": False,
            "runtime_code_changed": False,
        },
    )
    created = False
    try:
        queries = processes(ROOT, "preflight")
        assert all("No data" in s for s in queries.values()), queries
        (ROOT / "preflight-discovery.json").write_text(
            subprocess.check_output(["xpu-smi", "discovery", "-j"], text=True)
        )
        create()
        created = True
        summaries = []
        for profile in plan["profiles"]:
            output = ROOT / profile
            output.mkdir(exist_ok=False)
            queries = processes(output, "preflight")
            assert all("No data" in s for s in queries.values()), queries
            print("START", profile, flush=True)
            try:
                start(profile, output, plan)
                summaries.append(benchmark(profile, output, plan, prompt))
                write(ROOT / "comparison-in-progress.json", summaries)
            finally:
                stop(output)
            print("COMPLETE", profile, "GPU context released", flush=True)
        baseline = summaries[0]["medians"]
        for row in summaries:
            row["change_vs_non_speculative_pct"] = {
                k: 100 * (baseline[k] - row["medians"][k]) / baseline[k]
                for k in ["ttft_ms", "tpot_ms", "total_ms"]
            }
        reference = summaries[1]["medians"]
        summaries[2]["change_vs_k3_bf16_pct"] = {
            k: 100 * (reference[k] - summaries[2]["medians"][k]) / reference[k]
            for k in ["ttft_ms", "tpot_ms", "total_ms"]
        }
        write(ROOT / "comparison.json", summaries)
        write(
            ROOT / "status.json",
            {
                "status": "complete",
                "measured_samples": 9,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print("ALL PERFORMANCE SAMPLES COMPLETE", flush=True)
    except Exception as error:
        write(
            ROOT / "status.json",
            {
                "status": "incomplete",
                "error": repr(error),
                "utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        if created or (ROOT / "container.json").exists():
            stop(ROOT)


if __name__ == "__main__":
    main()
