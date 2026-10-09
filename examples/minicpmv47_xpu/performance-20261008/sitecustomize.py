import importlib.util
import logging
import os
import sys

import vllm

vllm.__path__.append('/opt/venv/lib/python3.12/site-packages/vllm')
if os.environ.get('MINICPM_BF16_OVERLAY') == '1':
    import torch
    import vllm_xpu_kernels
    import vllm_xpu_kernels._xpu_C
    import vllm_xpu_kernels._moe_C

    torch.ops.load_library('/workspace/preparation/performance-20261008/bf16-performance.so')
    name = 'vllm_xpu_kernels.moe_shared_fused_interface'
    path = '/workspace/preparation/kernels-bf16/vllm_xpu_kernels/moe_shared_fused_interface.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    vllm_xpu_kernels.moe_shared_fused_interface = module
    print('MINICPM BF16 native performance overlay loaded', flush=True)
    logging.getLogger('vllm.compilation.passes.fusion.xpu_fusion').setLevel(logging.DEBUG)
    logging.getLogger('vllm.compilation.passes.utility.xpu_utility').setLevel(logging.DEBUG)
