import importlib.util
import json
import os
from pathlib import Path
import torch

path=Path('/workspace/preparation/kernels-bf16/tests/fused_moe/test_router_gemv_topk_softmax.py')
spec=importlib.util.spec_from_file_location('router_checks',path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
torch.manual_seed(0)
dtype=torch.bfloat16
e,k,top_k=256,2048,8
w=torch.randn(e,k,device='xpu',dtype=dtype)*0.02
g=torch.randn(k,device='xpu',dtype=dtype)*0.1
cases=[]
for it in range(4):
 for m in [1,2,3,4,5,6,7,8,9,15,16]:
  x=torch.randn(m,k,device='xpu',dtype=dtype)
  r=torch.randn(m,k,device='xpu',dtype=dtype)
  cases.append((x,r,mod._outputs(m,e,top_k),mod._outputs(m,e,top_k),torch.empty_like(x),torch.empty_like(x)))
streams=[torch.xpu.current_stream(),torch.xpu.Stream()]
max_rows=16 if os.environ.get('MINICPM_M16_MOE')=='1' else 8
for i,(x,r,outs,nouts,normed,residual) in enumerate(cases):
 if x.shape[0]>max_rows:continue
 with torch.xpu.stream(streams[i%2]):
  mod.OPS.router_gemv_topk_softmax(x,w,*outs,True)
  mod.OPS.router_resadd_norm_gemv_topk_softmax(x,r,g,mod.EPS,w,*nouts,normed,residual,True)
torch.xpu.synchronize()
rows=[]
for index,(x,r,outs,nouts,normed,residual) in enumerate(cases):
 if x.shape[0]>max_rows:continue
 ref,_=mod._norm_reference(x,r,g)
 err=(normed.float()-ref.float()).abs()
 bad=err>0.001+0.001*ref.float().abs()
 if not bad.any():continue
 repeated=[]
 for _ in range(5):
  actual=torch.empty_like(x);res=torch.empty_like(x);bufs=mod._outputs(x.shape[0],e,top_k)
  mod.OPS.router_resadd_norm_gemv_topk_softmax(x,r,g,mod.EPS,w,*bufs,actual,res,True)
  repeated.append(bool(torch.equal(actual,normed)))
 rows.append({'case':index,'m':x.shape[0],'mismatches':int(bad.sum()),
 'max_absolute_error':float(err.max()),'serial_repeats_match_stream_result':repeated,
 'bf16_code_distance':int((normed.view(torch.int16).int()-ref.view(torch.int16).int()).abs().max())})
 assert all(repeated),rows[-1]
 assert rows[-1]['bf16_code_distance']<=1,rows[-1]
print(json.dumps({'max_rows':max_rows,'bf16_epsilon':torch.finfo(dtype).eps,'cases':rows,'counter_or_stream_instability_observed':False},indent=2))
