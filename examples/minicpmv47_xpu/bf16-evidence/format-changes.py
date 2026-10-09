"""Generate a formatting patch for the modified source files."""

import difflib
from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[1] / sys.argv[1]
for name in sys.argv[2:]:
    original = (root / name).read_text()
    if name.endswith((".cpp", ".hpp")):
        command = ["/workspace/tools/.venv/bin/clang-format", "--style=file", str(root / name)]
        formatted = subprocess.check_output(command, text=True)
    else:
        command = ["/workspace/tools/.venv/bin/ruff", "format", "--no-cache", "--stdin-filename", name, "-"] if sys.argv[1] == "vllm-bf16" else ["/workspace/tools/.venv/bin/yapf"]
        formatted = subprocess.run(command, input=original, cwd=root, text=True, capture_output=True, check=True).stdout
    print("".join(difflib.unified_diff(original.splitlines(keepends=True), formatted.splitlines(keepends=True), fromfile="a/" + name, tofile="b/" + name)), end="")
