import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parent
profile = sys.argv[1]
assert profile in ['NONE', 'PIECEWISE', 'FULL_DECODE_ONLY']
label = sys.argv[2] if len(sys.argv) > 2 else profile
assert label in ['NONE', 'PIECEWISE', 'FULL_DECODE_ONLY', 'NONE_RECHECK']
previous = json.loads((root.parent / 'performance-20261008/compile_bf16_overlay-launch.json').read_text())
args = previous['args'].copy()
index = args.index('--compilation-config') + 1
config = json.loads(args[index])
config['cudagraph_mode'] = profile
if profile != 'NONE':
    config['cudagraph_capture_sizes'] = [1]
    config['max_cudagraph_capture_size'] = 1
args[index] = json.dumps(config)
args += ['--cudagraph-metrics']
index = args.index('--profiler-config') + 1
profiler = json.loads(args[index])
profiler['torch_profiler_dir'] = f'/workspace/runtime/graph-experiment-traces-{profile}'
args[index] = json.dumps(profiler)
environment = previous['environment'].copy()
environment['VLLM_CACHE_ROOT'] = '/workspace/runtime/compile-cache-compile_bf16_overlay'
command = ['docker', 'exec']
for key, value in environment.items():
    command += ['--env', f'{key}={value}']
command += ['-d', previous['container'], '/bin/bash', '-lc',
            'cd /workspace/runtime; exec ' + shlex.join(args) +
            f' > /workspace/runtime/graph-experiment-{label}.log 2>&1']
(root / f'{label}-launch.json').write_text(json.dumps({
    'container': previous['container'], 'args': args, 'environment': environment,
    'docker_command': command}, indent=2) + '\n')
subprocess.run(command, check=True)
print('Started', label, flush=True)
