import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
label = sys.argv[1]
assert label.replace('_', '').isalnum()
container = 'codex-minicpm47-perf-' + label.lower().replace('_', '-') + '-20261008'
template = json.loads((root / 'fresh-container-created.json').read_text())[0]
image = template['Image']
assert subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                               text=True).strip() == image
args = ['docker', 'create', '--name', container, '--network', 'host', '--ipc', 'private',
        '--shm-size', '8g', '--device', '/dev/dri',
        '--label', 'codex.task=minicpm-v47-main-prep-20261008',
        '--label', 'codex.stage=' + label, '--workdir', '/workspace/runtime']
for mount in template['HostConfig']['Mounts']:
    arg = 'type=bind,src=' + mount['Source'] + ',dst=' + mount['Target']
    if mount.get('ReadOnly'):
        arg += ',readonly'
    args += ['--mount', arg]
for env in ['ZE_AFFINITY_MASK=0,1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1']:
    args += ['--env', env]
args += [image, '-lc', 'exec sleep infinity']
subprocess.run(args, check=True, capture_output=True)
receipt = root / (label + '-container.json')
assert not receipt.exists()
receipt.write_text(subprocess.check_output(['docker', 'inspect', container], text=True))
subprocess.run(['docker', 'start', container], check=True, capture_output=True)
setup = ('mkdir -p /workspace/runtime; '
         'uv venv --python /opt/venv/bin/python /workspace/runtime/.venv; '
         '/workspace/runtime/.venv/bin/python -c '
         '\'from pathlib import Path; '
         'p=Path("/workspace/runtime/.venv/lib/python3.12/site-packages/base_image.pth"); '
         'p.write_text("/opt/venv/lib/python3.12/site-packages\\n")\'')
subprocess.run(['docker', 'exec', container, '/bin/bash', '-lc', setup], check=True)
identity = subprocess.check_output(['docker', 'exec', container,
    '/workspace/runtime/.venv/bin/python', '/workspace/evidence/check-device.py'], text=True)
(root / (label + '-device-identity.json')).write_text(identity)
print(container, flush=True)
