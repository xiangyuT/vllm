from pathlib import Path

import torch
import vllm_xpu_kernels.flash_attn_interface  # noqa: F401

library = Path('/workspace/preparation/dspark-gpu01-20261008/_minicpm_dspark_fa2.abi3.so')
torch.ops.load_library(str(library))
original = torch.ops._vllm_fa2_C.varlen_fwd
native = torch.ops._minicpm_dspark_fa2.varlen_fwd
native_calls = 0
native_success_calls = 0
stock_calls = 0

def route(*args, **kwargs):
    global native_calls, native_success_calls, stock_calls
    if (not kwargs and len(args) == 29 and args[0].shape[-1] == 256
            and args[1].ndim == 4 and args[8] is not None
            and args[17] is None and not args[19]
            and args[20] == -1 and args[21] == -1 and not args[23]):
        if native_calls == 0:
            print('MINICPM DSpark native non-causal attention',
                  'q',tuple(args[0].shape),'kv',tuple(args[1].shape),flush=True)
        native_calls += 1
        result = native(*args)
        native_success_calls += 1
        return result
    stock_calls += 1
    return original(*args,**kwargs)

torch.ops._vllm_fa2_C.varlen_fwd = route
