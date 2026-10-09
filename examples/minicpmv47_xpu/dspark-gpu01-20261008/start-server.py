import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parent
profile = sys.argv[1]
assert profile in ['baseline_eager', 'dspark_eager', 'dspark_native', 'baseline_recheck', 'dspark_native_recheck', 'baseline_gsm20', 'dspark_native_gsm20', 'dspark_native_gsm20_cap1024']
container = 'codex-minicpm47-dspark-gpu01-20261008'
args = ['/workspace/runtime/.venv/bin/python', '-m', 'vllm.entrypoints.cli.main',
        'serve', '/minicpm', '--served-model-name', 'MiniCPM-V-4.7-1008',
        '--trust-remote-code', '--model-impl', 'vllm', '--tensor-parallel-size', '2',
        '--dtype', 'bfloat16', '--quantization', 'fp8', '--max-model-len', '8704',
        '--max-num-batched-tokens', '4096', '--max-num-seqs', '1',
        '--gpu-memory-utilization', '0.80', '--no-enable-prefix-caching',
        '--limit-mm-per-prompt', '{"image":0,"video":0}', '--enforce-eager',
        '--compilation-config', '{"mode":0,"cudagraph_mode":"NONE"}',
        '--host', '127.0.0.1', '--port', '31065']
if profile.startswith('dspark'):
    args += ['--speculative-config', json.dumps({
        'method': 'dspark', 'model': '/dspark', 'num_speculative_tokens': 15,
        'draft_tensor_parallel_size': 2, 'enable_adaptive_verification': False})]
if profile.startswith('dspark_native'):
    args += ['--per-request-spec-decode-metrics','summary']
environment = {
    'PYTHONPATH': '/workspace/preparation/vllm-functional:/workspace/preparation/functional-20261008',
    'VLLM_USE_V2_MODEL_RUNNER': '1',
    'VLLM_CACHE_ROOT': '/workspace/runtime/cache',
    'VLLM_LOGGING_LEVEL': 'INFO',
}
if profile.startswith('dspark_native'):
    environment['PYTHONPATH'] = '/workspace/preparation/vllm-functional:/workspace/preparation/dspark-gpu01-20261008'
    environment['MINICPM_DSPARK_ATTENTION'] = '1'
command = ['docker', 'exec']
for name, value in environment.items():
    command += ['--env', f'{name}={value}']
command += ['-d', container, '/bin/bash', '-lc',
            'cd /workspace/runtime; exec ' + shlex.join(args) +
            f' > /workspace/evidence/server-{profile}.log 2>&1']
receipt = root / f'{profile}-launch.json'
assert not receipt.exists()
receipt.write_text(json.dumps({'container': container, 'args': args,
                              'environment': environment, 'command': command}, indent=2) + '\n')
subprocess.run(command, check=True)
print('Started', profile, flush=True)
