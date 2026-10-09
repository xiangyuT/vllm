import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

root=Path(__file__).resolve().parent
profile=sys.argv[1]
label=sys.argv[2] if len(sys.argv)>2 else profile
plan=json.loads((root/'PLAN.json').read_text())
assert profile in plan['initial_profiles']
assert label.replace('_','').isalnum()
draft=profile.startswith('dspark_')
block=int(os.environ.get('MINICPM_DRAFT_TOKENS','15'))
assert block in [3,7,15]
m16=os.environ.get('MINICPM_M16_MOE')=='1'
graph=profile.split('_',1)[1]
mode='FULL_DECODE_ONLY' if graph=='FULL' else graph
config={'mode':3,'backend':'inductor','cudagraph_mode':mode,
        'pass_config':{name:True for name in plan['xpu_passes']}}
if len(sys.argv)>3 and sys.argv[3]=='no_small_range':
    assert draft
    config['pass_config']['fuse_xpu_moe_shared']=False
if m16:
    config['pass_config']['xpu_moe_shared_fusion_max_token_num']=16
    config['compile_ranges_endpoints']=[8,16]
if mode!='NONE':
    config['cudagraph_capture_sizes']=([block+1] if mode=='PIECEWISE' else [1,block+1]) if draft else [1]
    config['max_cudagraph_capture_size']=block+1 if draft else 1
args=['/workspace/runtime/.venv/bin/python','-m','vllm.entrypoints.cli.main','serve','/minicpm',
      '--served-model-name','MiniCPM-V-4.7-1008','--trust-remote-code','--model-impl','vllm',
      '--tensor-parallel-size','2','--dtype','bfloat16','--quantization','fp8',
      '--max-model-len','8704','--max-num-batched-tokens','4096','--max-num-seqs','1',
      '--gpu-memory-utilization','0.80','--no-enable-prefix-caching',
      '--limit-mm-per-prompt','{"image":0,"video":0}','--optimization-level','2',
      '--compilation-config',json.dumps(config),'--cudagraph-metrics',
      '--profiler-config',json.dumps({'profiler':'torch',
       'torch_profiler_dir':f'/workspace/runtime/dspark-performance-traces-{label}',
       'torch_profiler_activities':['CPU','XPU'],'torch_profiler_record_shapes':True,
       'torch_profiler_with_stack':False,'torch_profiler_dump_cuda_time_total':False,
       'ignore_frontend':True}),
      '--host','127.0.0.1','--port','31065']
if draft:
    args+=['--speculative-config',json.dumps({'method':'dspark','model':'/dspark',
            'num_speculative_tokens':block,'draft_tensor_parallel_size':2,
            'enable_adaptive_verification':False}),
           '--per-request-spec-decode-metrics','summary']
env={'PYTHONPATH':'/workspace/preparation/vllm-functional:/workspace/preparation/dspark-iteration-20261009',
     'VLLM_USE_V2_MODEL_RUNNER':'1','MINICPM_BF16_OVERLAY':'1',
     'MINICPM_DSPARK_ATTENTION':'1' if draft else '0',
     'MINICPM_M16_MOE':'1' if m16 else '0',
     'MINICPM_CG_AUDIT':'1' if graph=='FULL' else '0',
     'VLLM_CACHE_ROOT':'/workspace/runtime/dspark-performance-cache',
     'VLLM_LOGGING_LEVEL':'INFO',
     'VLLM_LOGGING_CONFIG_PATH':'/workspace/preparation/performance-20261008/logging.json'}
container=os.environ.get('MINICPM_PERF_CONTAINER','codex-minicpm47-dspark-perf-gpu01-20261008')
command=['docker','exec']
for name,value in env.items():
    command+=['--env',f'{name}={value}']
log=f'/workspace/evidence/performance-{label}.log'
command+=['-d',container,'/bin/bash','-lc',
          'cd /workspace/runtime; exec '+shlex.join(args)+' > '+shlex.quote(log)+' 2>&1']
receipt=root/f'{label}-launch.json'
assert not receipt.exists()
receipt.write_text(json.dumps({'profile':profile,'container':container,'args':args,'environment':env,
                              'command':command,'log':log},indent=2)+'\n')
subprocess.run(command,check=True)
print('Started',label,flush=True)
