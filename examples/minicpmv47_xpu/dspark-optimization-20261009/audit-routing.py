"""One extra unmeasured request; retain full verify routing for both ranks."""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
label = sys.argv[1]
phase = len(sys.argv) > 2 and sys.argv[2] == "phases"
out = ROOT / (f"phases-{label}" if phase else f"routing-{label}")
assert not out.exists()
out.mkdir()
cases = json.loads(
    (ROOT.parent / "dspark-gpu01-20261008/correctness-cases.json").read_text()
)
prompt = next(r["text"] for r in cases if r.get("prompt_tokens") == 2048)
body = {
    "model": "MiniCPM-V-4.7-1008",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "seed": 7,
    "min_tokens": 256,
    "max_tokens": 256,
    "ignore_eos": True,
    "chat_template_kwargs": {"enable_thinking": False},
}
flag = ROOT / ("phase-events.enabled" if phase else "routing-audit.enabled")
assert not flag.exists()
flag.write_text(label + "\n")
try:
    time.sleep(0.7)
    request = urllib.request.Request(
        "http://127.0.0.1:31065/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    (out / "response.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
finally:
    flag.unlink()
for rank in [0, 1]:
    p = ROOT / (
        f"phase-events-rank{rank}.json" if phase else f"routing-rank{rank}.json"
    )
    assert p.exists(), f"No complete routing observation for rank{rank}"
    rows = json.loads(p.read_text())
    if phase:
        assert len(rows["events"]) == 128
        assert all(r["gpu_elapsed_ms"] > 0 for r in rows["events"])
    else:
        assert len(rows["rounds"]) == 64
        assert all(len(r["unique_experts_by_router"]) == 40 for r in rows["rounds"])
    target = out / f"rank{rank}.json"
    p.rename(target)
    print(
        rank,
        "rounds",
        len(rows["events"] if phase else rows["rounds"]),
        "sha256",
        hashlib.sha256(target.read_bytes()).hexdigest(),
    )
