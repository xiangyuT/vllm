"""Prepare a short acceptance diagnostic from six continuous book windows."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parent
SOURCE = PREP / "benchmark-inputs/war-and-peace-2600"
MODEL = Path("/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-1008")


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    tokenizer = (MODEL / "tokenizer.json").read_bytes()
    assert (
        hashlib.sha256(tokenizer).hexdigest()
        == manifest["tokenization"]["tokenizer_sha256"]
    )
    body = (SOURCE / manifest["body"]["path"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == manifest["body"]["sha256"]
    code = """
import array, hashlib, json, sys
from pathlib import Path
from tokenizers import Tokenizer
t = Tokenizer.from_str(sys.stdin.read())
t.no_truncation(); t.no_padding()
ids = array.array('I')
ids.frombytes(Path('/workspace/preparation/benchmark-inputs/war-and-peace-2600/war-and-peace.tokens.u32le').read_bytes())
if sys.byteorder != 'little': ids.byteswap()
cases = []
for end in [2048, 65536, 131072, 262144, 524288, 700000]:
    start = end - 2048
    prompt_ids = list(ids[start:end])
    reference_ids = list(ids[end:end+512])
    text = t.decode(prompt_ids, skip_special_tokens=False)
    assert '\\ufffd' not in text, 'Window crosses a partial Unicode character'
    cases.append({'label': f'window_{start}_{end}', 'source_token_start': start,
                  'source_token_end_exclusive': end, 'source_window_tokens': 2048,
                  'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                  'prompt_token_ids': prompt_ids, 'source_next_token_ids': reference_ids,
                  'source_next_text': t.decode(reference_ids, skip_special_tokens=False)})
print(json.dumps(cases, ensure_ascii=False))
"""
    result = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "codex-minicpm47-bf16-build-20261008",
            "/opt/venv/bin/python",
            "-c",
            code,
        ],
        input=tokenizer,
        capture_output=True,
        check=True,
    )
    cases = json.loads(result.stdout)
    assert len(cases) == 6
    target = ROOT / "cases.json"
    assert not target.exists()
    target.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    plan = {
        "evidence_class": "exploratory_input_acceptance_diagnostic",
        "scope": "check whether this public novel saturates k3 acceptance; no performance comparison or long-context claim",
        "device_authorization": "existing user instruction: 使用GPU 01进行开发 完成迭代优化吧; reused for scoped DSpark input tests",
        "physical_gpus": [0, 1],
        "expected_sysman_uuid": [
            "00000000-0000-0018-0000-0000e2238086",
            "00000000-0000-0036-0000-0000e2238086",
        ],
        "expected_pci": ["0000:18:00.0", "0000:36:00.0"],
        "source_manifest": str(SOURCE / "manifest.json"),
        "body_sha256": manifest["body"]["sha256"],
        "tokenizer_sha256": manifest["tokenization"]["tokenizer_sha256"],
        "cases_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_cutoff_unit": "model tokenizer tokens, excluding chat template",
        "sampling": "six nonoverlapping 2048-token windows ending at specified source positions; diagnostic only, not a prefix long-context benchmark",
        "source_cutoff_positions": [c["source_token_end_exclusive"] for c in cases],
        "modes": ["raw_completion", "chat_continuation"],
        "chat_instruction": "Continue the following passage naturally from its endpoint. Output only the continuation, without an introduction or a summary.\n\n",
        "requests_per_mode": 6,
        "max_concurrency": 1,
        "temperature": 0,
        "seed": 7,
        "max_tokens": 256,
        "ignore_eos": False,
        "min_tokens": 0,
        "prefix_caching": False,
        "num_speculative_tokens": 3,
        "profile": "k3 FULL; main online FP8/BF16; draft BF16 backbone; Markov and draft vocab FP8/BF16 oneDNN",
        "metrics_mode": "detailed, including per-step accepted/drafted and acceptance histogram",
        "memory_diagnostic": "exact generated-token prefix match and longest contiguous overlap with the next 512 source tokens; neither is a correctness gate or proof of training membership",
        "interpretation": "acceptance near 100% is a saturation signal for this k3/profile/task; does not alone prove memorization or invalidate all uses of the corpus",
        "terminal_clip_limit": "speculative metrics are observed before max_tokens truncation; terminal overshoot retained and identified",
        "formal_acceptance": False,
        "latency_claim": False,
        "previous_results": "retain synthetic and GSM8K results; no response reuse",
        "future_prefix_test_parameters": "still pending user selection; these short diagnostic windows do not choose them",
    }
    (ROOT / "PLAN.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Prepared six source windows; both raw and chat continuation; CPU only",
        flush=True,
    )


if __name__ == "__main__":
    main()
