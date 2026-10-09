import ast
import hashlib
import json
from pathlib import Path
import subprocess

folder = Path(__file__).resolve().parent
root = folder.parent / 'vllm-functional'
def git(*args, check=True):
    return subprocess.run(['git', '-C', str(root), *args], check=check, capture_output=True).stdout

tracked = git('diff', '--name-only', '-z').decode().split('\0')[:-1]
untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')[:-1]
names = sorted(set(tracked + untracked))
patch = git('diff', '--binary')
for name in untracked:
    patch += git('diff', '--no-index', '--binary', '--', '/dev/null', name, check=False)
target = folder / 'vllm-main-integrated.patch'
target.write_bytes(patch)
for name in names:
    if name.endswith('.py'):
        ast.parse((root / name).read_text(), filename=name)
data = {'path': str(root), 'branch': git('branch', '--show-current').decode().strip(),
        'base_revision': git('rev-parse', 'HEAD').decode().strip(),
        'integrated_pr_heads': {'60539': '8f3e461beebc37d665a2cc2ba805adea1617dc34',
                                '58674': '4f5977e4c23c55f1d10b12a687443bbb262a27eb'},
        'bf16_patch': '../bf16-evidence/vllm-bf16-final.patch',
        'changed_files': names,
        'file_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
        'patch': target.name, 'patch_sha256': hashlib.sha256(patch).hexdigest(),
        'committed': False, 'pushed': False}
(folder / 'source-manifest.json').write_text(json.dumps(data, indent=2) + '\n')
print(f'{len(names)} changed files, Python AST and diff checks complete')
