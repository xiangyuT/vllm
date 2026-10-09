"""Isolate prompt formatting on short GSM8K and the same 32K book body."""

import hashlib
import importlib.util
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parents[1]
BASE = "http://127.0.0.1:31065"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_module("owned_lifecycle", ROOT.parent / "fixed256/run.py")
helpers.ROOT = ROOT
helpers.PREP = PREP
score = load_module("gsm_score", PREP / "dspark-optimization-20261009/gsm8k/helpers.py")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def post(endpoint, body):
    req = urllib.request.Request(
        BASE + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        assert response.status == 200
        return json.load(response)


def start():
    receipt = json.loads(
        (ROOT.parent / "fixed256/k3_bf16_FULL/launch.json").read_text()
    )
    args = receipt["args"][:]
    args[args.index("--per-request-spec-decode-metrics") + 1] = "detailed"
    env = receipt["environment"]
    command = ["docker", "exec"]
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += [
        "-d",
        helpers.NAME,
        "/bin/bash",
        "-lc",
        "exec "
        + __import__("shlex").join(args)
        + " > /workspace/evidence/chat-template-probe/server.log 2>&1",
    ]
    write(ROOT / "launch.json", {"args": args, "environment": env, "command": command})
    subprocess.run(command, check=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        log = ROOT / "server.log"
        if log.exists() and any(
            s in log.read_text()
            for s in ["Worker failed with error", "Engine core initialization failed"]
        ):
            raise RuntimeError("Server initialization failed; log retained")
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as response:
                if response.status == 200:
                    print("Server ready", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError("Service readiness timeout")


def summarize(rows):
    metrics = [r["response"]["metrics"]["speculative_decoding"] for r in rows]
    steps = sum(m["num_spec_steps"] for m in metrics)
    accepted = sum(m["num_accepted_draft_tokens"] for m in metrics)
    proposed = sum(m["num_draft_tokens"] for m in metrics)
    histogram = [sum(m["acceptance_histogram"][i] for m in metrics) for i in range(4)]
    for metric in metrics:
        assert sum(metric["per_step_accepted"]) == metric["num_accepted_draft_tokens"]
        assert sum(metric["per_step_drafted"]) == metric["num_draft_tokens"]
        assert len(metric["per_step_accepted"]) == metric["num_spec_steps"]
    return {
        "requests": len(rows),
        "steps": steps,
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_rate": accepted / proposed if proposed else None,
        "mean_acceptance_length": 1 + accepted / steps if steps else None,
        "acceptance_histogram": histogram,
        "output_tokens": sum(r["response"]["usage"]["completion_tokens"] for r in rows),
        "length_truncations": sum(
            r["response"]["choices"][0]["finish_reason"] == "length" for r in rows
        ),
        "prefix_survival_rates": [sum(histogram[i:]) / steps for i in range(1, 4)]
        if steps
        else [],
    }


def gsm8k(plan):
    questions_bytes = (ROOT / "questions.json").read_bytes()
    assert (
        hashlib.sha256(questions_bytes).hexdigest() == plan["gsm8k"]["question_sha256"]
    )
    questions = json.loads(questions_bytes)
    summaries = {}
    common = {
        "model": "MiniCPM-V-4.7-1008",
        "temperature": 0,
        "seed": 7,
        "max_tokens": 1024,
        "ignore_eos": False,
        "return_token_ids": True,
        "stop": plan["gsm8k"]["stop"],
    }
    for arm in plan["gsm8k"]["arms"]:
        results = []
        for question in questions:
            if arm == "chat":
                endpoint = "/v1/chat/completions"
                body = {
                    **common,
                    "messages": [{"role": "user", "content": question["prompt"]}],
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            else:
                endpoint = "/v1/completions"
                body = {**common, "prompt": question["prompt"]}
            response = post(endpoint, body)
            write(ROOT / f"gsm-{arm}-{question['test_index']}-response.json", response)
            choice = response["choices"][0]
            text = choice["message"]["content"] if arm == "chat" else choice["text"]
            prediction = score.get_answer_value(text or "")
            row = {
                "arm": arm,
                "test_index": question["test_index"],
                "prompt_sha256": question["prompt_sha256"],
                "label": question["label"],
                "prediction": prediction,
                "correct": prediction == question["label"],
                "response": response,
            }
            results.append(row)
            write(ROOT / f"gsm-{arm}-results.json", results)
            metric = response["metrics"]["speculative_decoding"]
            print(
                "GSM",
                arm,
                question["test_index"],
                "correct",
                row["correct"],
                "acceptance",
                round(metric["draft_acceptance_rate"], 4),
                "tokens",
                response["usage"]["completion_tokens"],
                flush=True,
            )
        summary = summarize(results)
        summary.update(
            {
                "correct": sum(r["correct"] for r in results),
                "num_questions": 20,
                "meets_70_percent_target": summary["acceptance_rate"] >= 0.70,
                "latency_claim": False,
            }
        )
        summaries[arm] = summary
        write(ROOT / f"gsm-{arm}-summary.json", summary)
        print("GSM COMPLETE", arm, json.dumps(summary), flush=True)
    return summaries


def long_text(plan):
    text = (ROOT.parent / "prompt.txt").read_text()
    assert (
        hashlib.sha256(text.encode()).hexdigest()
        == plan["long_text"]["body_text_sha256"]
    )
    plain = plan["long_text"]["instruction"] + text
    messages = [{"role": "user", "content": plain}]
    body_ids = json.loads((ROOT.parent / "prompt-token-ids.json").read_text())
    raw = post(
        "/tokenize",
        {"model": "MiniCPM-V-4.7-1008", "prompt": plain, "add_special_tokens": False},
    )
    chat = post(
        "/tokenize",
        {
            "model": "MiniCPM-V-4.7-1008",
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    cases = {
        "raw_body": body_ids,
        "instruction_raw": raw["tokens"],
        "chat_rendered": chat["tokens"],
    }
    assert len(body_ids) == 32768
    write(ROOT / "long-prompt-token-ids.json", cases)
    write(
        ROOT / "tokenize-responses.json",
        {"instruction_raw": raw, "chat_rendered": chat},
    )
    summaries = {}
    common = {
        "model": "MiniCPM-V-4.7-1008",
        "temperature": 0,
        "seed": 7,
        "max_tokens": 256,
        "min_tokens": 256,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    for arm in plan["long_text"]["arms"]:
        prompt = cases[arm]
        warmup = post("/v1/completions", {**common, "prompt": prompt})
        write(ROOT / f"long-{arm}-warmup.json", warmup)
        results = []
        for index in range(3):
            response = post("/v1/completions", {**common, "prompt": prompt})
            write(ROOT / f"long-{arm}-{index}-response.json", response)
            assert response["usage"]["prompt_tokens"] == len(prompt)
            assert response["usage"]["completion_tokens"] == 256
            results.append({"arm": arm, "index": index, "response": response})
            write(ROOT / f"long-{arm}-results.json", results)
            m = response["metrics"]["speculative_decoding"]
            print(
                "LONG",
                arm,
                index,
                "input",
                len(prompt),
                "acceptance",
                round(m["draft_acceptance_rate"], 4),
                "AL",
                round(m["mean_acceptance_length"], 4),
                flush=True,
            )
        summary = summarize(results)
        summary.update({"input_tokens": len(prompt), "latency_claim": False})
        summaries[arm] = summary
        write(ROOT / f"long-{arm}-summary.json", summary)
    return summaries


def main():
    plan = json.loads((ROOT / "PLAN.json").read_text())
    created = False
    try:
        queries = helpers.processes(ROOT, "preflight")
        assert all("No data" in s for s in queries.values()), queries
        helpers.create()
        created = True
        start()
        for arm in ["chat", "raw"]:
            warm = {
                "model": "MiniCPM-V-4.7-1008",
                "temperature": 0,
                "seed": 7,
                "max_tokens": 32,
            }
            if arm == "chat":
                warm.update(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Calculate 1+1. Answer with one number.",
                            }
                        ],
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
                endpoint = "/v1/chat/completions"
            else:
                warm["prompt"] = "Question: Calculate 1+1.\nAnswer:"
                endpoint = "/v1/completions"
            write(ROOT / f"warmup-{arm}.json", post(endpoint, warm))
        result = {"gsm8k": gsm8k(plan)}
        write(ROOT / "summary-in-progress.json", result)
        result["long_text"] = long_text(plan)
        result.update(
            {
                "evidence_class": plan["evidence_class"],
                "latency_claim": False,
                "formal_acceptance": False,
            }
        )
        write(ROOT / "summary.json", result)
        print("ALL DIAGNOSTICS COMPLETE", json.dumps(result), flush=True)
    finally:
        if created or (ROOT / "container.json").exists():
            helpers.stop(ROOT)


if __name__ == "__main__":
    main()
