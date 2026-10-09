"""Freeze the real 32768-token prefix and the three-profile measurement plan."""

import array
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parent
SOURCE = PREP / "benchmark-inputs/war-and-peace-2600"


def main():
    source = json.loads((SOURCE / "manifest.json").read_text())
    raw_ids = (SOURCE / source["tokenization"]["token_ids_path"]).read_bytes()
    assert (
        hashlib.sha256(raw_ids).hexdigest()
        == source["tokenization"]["token_ids_sha256"]
    )
    ids = array.array("I")
    ids.frombytes(raw_ids)
    if sys.byteorder != "little":
        ids.byteswap()
    prefix = list(ids[:32768])
    assert len(prefix) == 32768
    tokenizer = Path(source["tokenization"]["tokenizer_source"]).read_bytes()
    assert (
        hashlib.sha256(tokenizer).hexdigest()
        == source["tokenization"]["tokenizer_sha256"]
    )
    code = """
import array, json, sys
from pathlib import Path
from tokenizers import Tokenizer
t = Tokenizer.from_str(sys.stdin.read())
t.no_truncation(); t.no_padding()
ids = array.array('I')
ids.frombytes(Path('/workspace/preparation/benchmark-inputs/war-and-peace-2600/war-and-peace.tokens.u32le').read_bytes())
if sys.byteorder != 'little': ids.byteswap()
text = t.decode(list(ids[:32768]), skip_special_tokens=False)
assert '\\ufffd' not in text
print(json.dumps({'text':text, 'retokenized_text_tokens':len(t.encode(text,add_special_tokens=False).ids)}, ensure_ascii=False))
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
    decoded = json.loads(result.stdout)
    for name in ["prompt-token-ids.json", "prompt.txt", "PLAN.json"]:
        assert not (ROOT / name).exists()
    (ROOT / "prompt-token-ids.json").write_text(json.dumps(prefix) + "\n")
    (ROOT / "prompt.txt").write_text(decoded["text"])
    plan = {
        "evidence_class": "development_real_text_end_to_end_performance",
        "input": {
            "title": "War and Peace",
            "source_manifest": str(SOURCE / "manifest.json"),
            "source_body_sha256": source["body"]["sha256"],
            "tokenizer_sha256": source["tokenization"]["tokenizer_sha256"],
            "cutoff_unit": "main model tokenizer token IDs, no special tokens or chat template",
            "source_token_range": [0, 32768],
            "token_ids_file": "prompt-token-ids.json",
            "token_ids_sha256": hashlib.sha256(
                (ROOT / "prompt-token-ids.json").read_bytes()
            ).hexdigest(),
            "text_file": "prompt.txt",
            "text_sha256": hashlib.sha256(
                (ROOT / "prompt.txt").read_bytes()
            ).hexdigest(),
            "retokenized_text_tokens": decoded["retokenized_text_tokens"],
            "transport": "raw /v1/completions with exact token-ID prefix; natural continuation",
            "prompt_tokens": 32768,
        },
        "profiles": ["baseline_FULL", "k3_bf16_FULL", "k3_heads_fp8_FULL"],
        "profile_meanings": {
            "baseline_FULL": "no speculative decoding, main online FP8/BF16",
            "k3_bf16_FULL": "DSpark k3, BF16 draft backbone/Markov/vocab",
            "k3_heads_fp8_FULL": "DSpark k3, BF16 backbone, FP8 Markov and draft vocab/BF16 activation/oneDNN",
        },
        "max_model_len": 33280,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.8,
        "physical_gpus": [0, 1],
        "device_authorization": "reuse user's GPU 01 authorization for ongoing DSpark work; release at completion",
        "expected_sysman_uuid": [
            "00000000-0000-0018-0000-0000e2238086",
            "00000000-0000-0036-0000-0000e2238086",
        ],
        "expected_pci": ["0000:18:00.0", "0000:36:00.0"],
        "tensor_parallel_size": 2,
        "runner": "V2",
        "activation_dtype": "bfloat16",
        "target_quantization": "online_fp8",
        "graph_mode": "FULL_DECODE_ONLY",
        "prefix_caching": False,
        "max_output_tokens": 256,
        "ignore_eos": False,
        "min_output_tokens": 0,
        "temperature": 0,
        "seed_schedule": [7, 7, 7, 7, 7],
        "short_warmup_requests": 4,
        "short_warmup_prefix_tokens": 256,
        "full_warmup_requests": 2,
        "measured_requests": 3,
        "max_concurrency": 1,
        "ttft_definition": "client request start to first SSE chunk with generated token IDs; textual TTFT also retained",
        "tpot_definition": "(DONE time - first generated-token SSE time)/(actual completion tokens - first SSE token batch count); an average client decode metric, not individual token intervals",
        "total_definition": "client request start to SSE DONE",
        "report": "all samples retained; medians over three measured samples; prefill and decode reported separately",
        "metric_mode": "summary speculative metrics only; no GPU profiler/snapshot/event timing in measured requests",
        "exclusions": [
            "model loading",
            "compilation",
            "graph capture",
            "short warmup",
            "two full warmups",
        ],
        "source_reuse": "unchanged exact runtime/native overlay identities; old synthetic and short-window timings not reused",
        "accuracy": "no runtime/model implementation change; existing GSM8K score provenance retained, no new score claim",
        "formal_acceptance": False,
    }
    (ROOT / "PLAN.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Prepared exact 32768-token real-text prefix; retokenized text",
        decoded["retokenized_text_tokens"],
        flush=True,
    )


if __name__ == "__main__":
    main()
