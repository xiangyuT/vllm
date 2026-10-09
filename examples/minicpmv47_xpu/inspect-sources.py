"""Record source identities and check syntax without importing project code."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess


root = Path(__file__).resolve().parent
entries = [
    ("vllm-main", "vllm-pr-review", "review/pr-60539", "vllm-pr-60539.patch"),
    (
        "vllm-xpu-kernels-main",
        "kernels-pr-review",
        "review/pr-622",
        "kernels-pr-622.patch",
    ),
    ("vllm-main", "minicpm-pr-review", "review/pr-58674", "minicpm-pr-58674.patch"),
]
result = {
    "evidence_class": "static_source_review",
    "gpu_execution": False,
    "sources": [],
}
for checkout, snapshot, ref, patch in entries:
    repo = root / checkout
    git_command = ["git", "-c", f"safe.directory={repo}", "-C", str(repo)]

    def git(*args):
        return subprocess.check_output(git_command + list(args), text=True).strip()

    files = git("diff", "--name-only", f"HEAD...{ref}").splitlines()
    checked = []
    for filename in files:
        if filename.endswith(".py"):
            path = root / snapshot / filename
            ast.parse(path.read_text(), filename=str(path))
            checked.append(filename)
    merge_base = git("merge-base", "HEAD", ref)
    patch_path = root / "patches" / patch
    check = subprocess.run(
        git_command + ["apply", "--check", str(patch_path)],
        text=True,
        capture_output=True,
    )
    result["sources"].append(
        {
            "checkout": checkout,
            "snapshot": snapshot,
            "base_revision": git("rev-parse", "HEAD"),
            "merge_base": merge_base,
            "pr_head": git("rev-parse", ref),
            "ordered_commits": git(
                "rev-list", "--reverse", f"{merge_base}..{ref}"
            ).splitlines(),
            "changed_files": files,
            "python_ast_checked": checked,
            "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            "patch_check_exit_code": check.returncode,
            "patch_check_output": check.stderr,
        }
    )
result["python_files_checked"] = sum(
    len(source["python_ast_checked"]) for source in result["sources"]
)
print(json.dumps(result, indent=2))
