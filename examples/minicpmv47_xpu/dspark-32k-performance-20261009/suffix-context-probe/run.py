"""Keep the continuation endpoint fixed while shortening the visible history."""

import importlib.util
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parents[1]

spec = importlib.util.spec_from_file_location(
    "format_probe", ROOT.parent / "chat-template-probe/run.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.ROOT = ROOT
probe.helpers.ROOT = ROOT


def start():
    receipt = json.loads(
        (ROOT.parent / "fixed256/k3_bf16_FULL/launch.json").read_text()
    )
    args = receipt["args"][:]
    args[args.index("--per-request-spec-decode-metrics") + 1] = "detailed"
    command = ["docker", "exec"]
    for key, value in receipt["environment"].items():
        command += ["--env", f"{key}={value}"]
    command += [
        "-d",
        probe.helpers.NAME,
        "/bin/bash",
        "-lc",
        "exec "
        + shlex.join(args)
        + " > /workspace/evidence/suffix-context-probe/server.log 2>&1",
    ]
    probe.write(
        ROOT / "launch.json",
        {"args": args, "command": command, "environment": receipt["environment"]},
    )
    subprocess.run(command, check=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(probe.BASE + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError("Server readiness timeout")


def main():
    plan = json.loads((ROOT / "PLAN.json").read_text())
    original = json.loads((ROOT.parent / "prompt-token-ids.json").read_text())
    decode = """
import json, sys
from tokenizers import Tokenizer
p = json.load(sys.stdin)
t = Tokenizer.from_str(p['tokenizer'])
print(json.dumps({str(n):t.decode(p['ids'][-n:],skip_special_tokens=False) for n in [2048,8192]},ensure_ascii=False))
"""
    tokenizer = Path(
        "/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-1008/tokenizer.json"
    ).read_text()
    decoded = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "codex-minicpm47-bf16-build-20261008",
            "/opt/venv/bin/python",
            "-c",
            decode,
        ],
        input=json.dumps({"tokenizer": tokenizer, "ids": original}).encode(),
        capture_output=True,
        check=True,
    )
    texts = json.loads(decoded.stdout)
    probe.write(ROOT / "window-texts.json", texts)
    created = False
    try:
        queries = probe.helpers.processes(ROOT, "preflight")
        assert all("No data" in value for value in queries.values())
        probe.helpers.create()
        created = True
        start()
        summaries = {}
        for length in plan["context_windows"]:
            text = texts[str(length)]
            assert "\ufffd" not in text
            content = plan["long_text"]["instruction"] + text
            tokenized = probe.post(
                "/tokenize",
                {
                    "model": "MiniCPM-V-4.7-1008",
                    "messages": [{"role": "user", "content": content}],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            ids = tokenized["tokens"]
            probe.write(
                ROOT / f"input-{length}.json",
                {
                    "source_range": [32768 - length, 32768],
                    "body_tokens": length,
                    "input_tokens": len(ids),
                    "token_ids": ids,
                },
            )
            body = {
                "model": "MiniCPM-V-4.7-1008",
                "prompt": ids,
                "temperature": 0,
                "seed": 7,
                "ignore_eos": True,
                "min_tokens": 256,
                "max_tokens": 256,
                "return_token_ids": True,
            }
            probe.write(
                ROOT / f"warmup-{length}.json", probe.post("/v1/completions", body)
            )
            rows = []
            for index in range(3):
                response = probe.post("/v1/completions", body)
                assert response["usage"]["completion_tokens"] == 256
                rows.append(
                    {"context_window": length, "index": index, "response": response}
                )
                probe.write(ROOT / f"results-{length}.json", rows)
                m = response["metrics"]["speculative_decoding"]
                print(
                    "WINDOW",
                    length,
                    index,
                    "acceptance",
                    round(m["draft_acceptance_rate"], 4),
                    "AL",
                    round(m["mean_acceptance_length"], 4),
                    flush=True,
                )
            summaries[str(length)] = probe.summarize(rows)
            summaries[str(length)]["input_tokens"] = len(ids)
        previous = json.loads(
            (ROOT.parent / "chat-template-probe/summary.json").read_text()
        )
        summaries["32768"] = previous["long_text"]["chat_rendered"]
        probe.write(
            ROOT / "summary.json",
            {
                "windows": summaries,
                "evidence_class": plan["evidence_class"],
                "previous_32768_source": "../chat-template-probe/long-chat_rendered-results.json",
                "latency_claim": False,
            },
        )
        print("WINDOW DIAGNOSTIC COMPLETE", json.dumps(summaries), flush=True)
    finally:
        if created or (ROOT / "container.json").exists():
            probe.helpers.stop(ROOT)


if __name__ == "__main__":
    main()
