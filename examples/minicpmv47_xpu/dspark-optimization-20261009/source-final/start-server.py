"""Launch baseline or scoped Markov candidates without changing old receipts."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
label = sys.argv[1]
assert label.replace("_", "").isalnum()
block = int(sys.argv[2])
assert block in (3, 7, 15)
fp8 = sys.argv[3] == "fp8"
backend = sys.argv[4] if len(sys.argv) > 4 else "generic"
vocab_fp8 = len(sys.argv) > 5 and sys.argv[5] == "vocab_fp8"
template = json.loads(
    (ROOT.parent / "dspark-iteration-20261009/k3_FULL-launch.json").read_text()
)
args = template["args"][:]
comp_idx = args.index("--compilation-config") + 1
config = json.loads(args[comp_idx])
config["cudagraph_capture_sizes"] = [1, block + 1]
config["max_cudagraph_capture_size"] = block + 1
if block == 15:
    config["pass_config"]["xpu_moe_shared_fusion_max_token_num"] = 16
    config["compile_ranges_endpoints"] = [8, 16]
args[comp_idx] = json.dumps(config)
spec_idx = args.index("--speculative-config") + 1
spec = json.loads(args[spec_idx])
spec["num_speculative_tokens"] = block
args[spec_idx] = json.dumps(spec)
prof_idx = args.index("--profiler-config") + 1
prof = json.loads(args[prof_idx])
prof["torch_profiler_dir"] = f"/workspace/runtime/dspark-opt-traces-{label}"
args[prof_idx] = json.dumps(prof)
env = dict(template["environment"])
env["PYTHONPATH"] = (
    "/workspace/preparation/dspark-optimization-20261009:/workspace/preparation/vllm-functional"
)
env["MINICPM_MARKOV_FP8"] = "1" if fp8 else "0"
env["MINICPM_ROUTING_AUDIT"] = "1"
env["MINICPM_MARKOV_BACKEND"] = backend
env["MINICPM_DRAFT_VOCAB_FP8"] = "1" if vocab_fp8 else "0"
env["VLLM_NO_USAGE_STATS"] = "1"
env["MINICPM_M16_MOE"] = "1" if block == 15 else "0"
container = json.loads((ROOT / (label + "-container.json")).read_text())[0][
    "Name"
].lstrip("/")
command = ["docker", "exec"]
for name, value in env.items():
    command += ["--env", f"{name}={value}"]
log = f"/workspace/evidence/performance-{label}.log"
command += [
    "-d",
    container,
    "/bin/bash",
    "-lc",
    "cd /workspace/runtime; exec "
    + shlex.join(args)
    + " > "
    + shlex.quote(log)
    + " 2>&1",
]
receipt = ROOT / (label + "-launch.json")
assert not receipt.exists()
receipt.write_text(
    json.dumps(
        {
            "profile": label,
            "container": container,
            "args": args,
            "environment": env,
            "command": command,
            "log": log,
        },
        indent=2,
    )
    + "\n"
)
subprocess.run(command, check=True)
print("Started", label, flush=True)
