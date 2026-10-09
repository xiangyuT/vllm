"""Stop only this diagnostic container and record root-visible occupancy."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    original = json.loads((ROOT / "container.json").read_text())[0]
    identity = original["Id"]
    current = json.loads(
        subprocess.check_output(["docker", "inspect", identity], text=True)
    )[0]
    assert current["Name"] == "/codex-minicpm47-input-acceptance-20261009"
    assert current["Config"]["Labels"]["codex.task"] == "minicpm-v47-main-prep-20261008"
    assert current["Config"]["Labels"]["codex.stage"] == "input-acceptance-20261009"
    if current["State"]["Running"]:
        subprocess.run(
            ["docker", "stop", "--time", "10", identity],
            check=True,
            capture_output=True,
        )
    current = json.loads(
        subprocess.check_output(["docker", "inspect", identity], text=True)
    )[0]
    assert not current["State"]["Running"]
    queries = {}
    for rank in [0, 1]:
        name = f"codex-minicpm47-root-ps-{rank}-20261009"
        helper = json.loads(
            subprocess.check_output(["docker", "inspect", name], text=True)
        )[0]
        assert (
            helper["Config"]["Labels"]["codex.task"] == "minicpm-v47-main-prep-20261008"
        )
        assert helper["HostConfig"]["PidMode"] == "host"
        result = subprocess.run(
            ["docker", "start", "-a", name], check=True, text=True, capture_output=True
        )
        queries[str(rank)] = result.stdout + result.stderr
        (ROOT / f"final-root-ps-gpu{rank}.txt").write_text(queries[str(rank)])
    ports = subprocess.check_output(["ss", "-ltn", "( sport = :31065 )"], text=True)
    released = {
        "released_utc": datetime.now(timezone.utc).isoformat(),
        "container_id": current["Id"],
        "container_name": current["Name"],
        "image_id": current["Image"],
        "running": current["State"]["Running"],
        "physical_gpus": [0, 1],
        "root_sysman_queries": queries,
        "port_31065_listening": len(ports.strip().splitlines()) > 1,
        "hardware_reset": False,
        "other_containers_stopped": False,
    }
    (ROOT / "release.json").write_text(json.dumps(released, indent=2) + "\n")
    print(json.dumps(released), flush=True)


if __name__ == "__main__":
    main()
