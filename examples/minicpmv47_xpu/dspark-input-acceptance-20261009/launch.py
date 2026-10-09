"""Create one owned diagnostic container, verify affinity, then launch k3."""

import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parent
NAME = "codex-minicpm47-input-acceptance-20261009"
IMAGE = "sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566"


def main():
    assert (
        subprocess.check_output(
            ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True
        ).strip()
        == IMAGE
    )
    for rank in [0, 1]:
        assert "No data" in (ROOT / f"preflight-root-ps-gpu{rank}.txt").read_text()
    args = [
        "docker",
        "create",
        "--name",
        NAME,
        "--network",
        "host",
        "--shm-size",
        "8g",
        "--device",
        "/dev/dri",
        "--entrypoint",
        "/bin/bash",
        "--workdir",
        "/workspace/runtime",
        "--label",
        "codex.task=minicpm-v47-main-prep-20261008",
        "--label",
        "codex.stage=input-acceptance-20261009",
    ]
    for source, target, readonly in [
        ("/dev/dri/by-path", "/dev/dri/by-path", True),
        (str(PREP), "/workspace/preparation", True),
        (str(ROOT), "/workspace/evidence", False),
        ("/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-1008", "/minicpm", True),
        ("/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-DSpark-1008", "/dspark", True),
    ]:
        mount = f"type=bind,src={source},dst={target}" + (
            ",readonly" if readonly else ""
        )
        args += ["--mount", mount]
    args += [
        "--env",
        "ZE_AFFINITY_MASK=0,1",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        IMAGE,
        "-lc",
        "exec sleep infinity",
    ]
    subprocess.run(args, capture_output=True, check=True)
    (ROOT / "container.json").write_text(
        subprocess.check_output(["docker", "inspect", NAME], text=True)
    )
    subprocess.run(["docker", "start", NAME], check=True, capture_output=True)
    subprocess.run(
        [
            "docker",
            "exec",
            NAME,
            "uv",
            "venv",
            "--python",
            "/opt/venv/bin/python",
            "/workspace/runtime/.venv",
        ],
        check=True,
        capture_output=True,
    )
    setup = "from pathlib import Path; Path('/workspace/runtime/.venv/lib/python3.12/site-packages/base_image.pth').write_text('/opt/venv/lib/python3.12/site-packages\\n')"
    subprocess.run(
        ["docker", "exec", NAME, "/workspace/runtime/.venv/bin/python", "-c", setup],
        check=True,
    )
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
    template = json.loads(
        (PREP / "dspark-optimization-20261009/k3_heads_fp8-launch.json").read_text()
    )
    server_args = template["args"][:]
    index = server_args.index("--profiler-config")
    del server_args[index : index + 2]
    server_args[server_args.index("--per-request-spec-decode-metrics") + 1] = "detailed"
    env = template["environment"].copy()
    env["MINICPM_ROUTING_AUDIT"] = "0"
    command = ["docker", "exec"]
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += [
        "-d",
        NAME,
        "/bin/bash",
        "-lc",
        "exec " + shlex.join(server_args) + " > /workspace/evidence/server.log 2>&1",
    ]
    (ROOT / "launch.json").write_text(
        json.dumps(
            {
                "container": NAME,
                "image": IMAGE,
                "args": server_args,
                "environment": env,
                "command": command,
                "profiler_enabled": False,
            },
            indent=2,
        )
        + "\n"
    )
    subprocess.run(command, check=True)
    print("Launched", NAME, "with verified two-device affinity", flush=True)


if __name__ == "__main__":
    main()
