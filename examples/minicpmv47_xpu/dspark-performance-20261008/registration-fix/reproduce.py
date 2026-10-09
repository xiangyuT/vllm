import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.compilation.passes.fusion.xpu_fusion import XpuGatedQkvNormRopePattern, _RopeSpec
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

config=VllmConfig(device_config=DeviceConfig(device='cpu'))
matcher=PatternMatcherPass()
with FakeTensorMode(),set_current_vllm_config(config),config.kernel_config.ir_op_priority.set_priority():
    for heads in [4,8]:
        pattern=XpuGatedQkvNormRopePattern(num_heads=heads,num_kv_heads=1,eps=1e-6,
            rope=_RopeSpec(head_dim=256,rotary_dim=64,mrope_section=[11,11,10],mrope_interleaved=True),
            mrope_positions=False,dtype=torch.bfloat16,config=config)
        pattern.get_inputs=lambda heads=heads:[
            torch.empty(5,2*heads*256+2*256,dtype=torch.bfloat16),
            torch.empty(5,dtype=torch.int64),torch.empty(256,dtype=torch.bfloat16),
            torch.empty(256,dtype=torch.bfloat16),torch.empty(4096,64,dtype=torch.bfloat16)]
        pattern.register(matcher)
        print('registered',heads,flush=True)
