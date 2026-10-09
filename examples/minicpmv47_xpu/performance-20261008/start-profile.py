import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parent
profile = sys.argv[1]
assert profile in ['compile', 'compile_bf16_overlay']
container = 'codex-minicpm47-functional-gpu03-attempt2-20261008'
config = {'mode': 3, 'backend': 'inductor', 'cudagraph_mode': 'NONE'}
if profile == 'compile_bf16_overlay':
    config['pass_config'] = {
        'fuse_xpu_moe_shared': True, 'fuse_xpu_qkv_norm_rope': True,
        'fuse_xpu_fp8_gemm_pair': True, 'fuse_xpu_norm_fp8_gemm': True,
        'xpu_gdn_output_alloc': True, 'xpu_inplace_all_reduce': True}
args = ['.venv/bin/python', '-m', 'vllm.entrypoints.cli.main', 'serve', '/minicpm',
        '--served-model-name', 'MiniCPM-V-4.7-old', '--trust-remote-code',
        '--model-impl', 'vllm', '--tensor-parallel-size', '2', '--dtype', 'bfloat16',
        '--quantization', 'fp8', '--max-model-len', '33280',
        '--max-num-batched-tokens', '4096', '--max-num-seqs', '1',
        '--gpu-memory-utilization', '0.82', '--no-enable-prefix-caching',
        '--limit-mm-per-prompt', '{"image":0,"video":0}', '--optimization-level', '2',
        '--compilation-config', json.dumps(config), '--host', '127.0.0.1', '--port', '31063']
if profile == 'compile_bf16_overlay':
    args += ['--profiler-config', json.dumps({
        'profiler': 'torch', 'torch_profiler_dir': '/workspace/runtime/bf16-traces',
        'torch_profiler_activities': ['CPU', 'XPU'],
        'torch_profiler_with_stack': False, 'torch_profiler_record_shapes': True,
        'torch_profiler_dump_cuda_time_total': False, 'ignore_frontend': True})]
environment = {'PYTHONPATH': '/workspace/preparation/vllm-functional:/workspace/preparation/performance-20261008',
               'VLLM_USE_V2_MODEL_RUNNER': '0',
               'MINICPM_BF16_OVERLAY': '1' if profile == 'compile_bf16_overlay' else '0',
               'VLLM_LOGGING_LEVEL': 'INFO',
               'VLLM_CACHE_ROOT': f'/workspace/runtime/compile-cache-{profile}'}
if profile == 'compile_bf16_overlay':
    environment['VLLM_LOGGING_CONFIG_PATH'] = '/workspace/preparation/performance-20261008/logging.json'
command = ['docker', 'exec']
for name, value in environment.items():
    command += ['--env', f'{name}={value}']
command += ['-d', container, '/bin/bash', '-lc',
            'cd /workspace/runtime; exec ' + shlex.join(args) +
            f' > /workspace/runtime/perf-{profile}.log 2>&1']
(root / f'{profile}-launch.json').write_text(json.dumps({
    'container': container, 'environment': environment, 'args': args,
    'docker_command': command}, indent=2) + '\n')
subprocess.run(command, check=True)
print('Started', profile, flush=True)
