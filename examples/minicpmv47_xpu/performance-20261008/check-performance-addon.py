import importlib.util
import json
from pathlib import Path

import torch

path = Path('/workspace/preparation/kernels-bf16/tests/test_qkv_split_norm_rope.py')
spec = importlib.util.spec_from_file_location('qkv_tests', path)
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
results = []
for m in [1, 4]:
    torch.manual_seed(7)
    x = torch.randn(m, 2048, device='xpu', dtype=torch.bfloat16)
    w = (torch.randn(32, 2048, device='xpu') * .05).to(torch.float8_e4m3fn)
    s = torch.tensor([.02], device='xpu')
    actual = torch.ops._xpu_C.fp8_gemm_w8a16(x, w.t(), s, None)
    expected = (x.float() @ (w.float() * s).t()).to(x.dtype)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    results.append({'op': 'single_gemv' if m == 1 else 'oneDNN_fallback', 'm': m, 'passed': True})
for m in [1, 64]:
    tests.test_qkv_split_norm_rope(m, 8, 1, 256, 64, True, 'interleaved', torch.bfloat16)
    results.append({'op': 'qkv_norm_mrope', 'm': m, 'passed': True})
torch.xpu.synchronize()
Path('/workspace/runtime/performance-addon-validation.json').write_text(json.dumps(results, indent=2) + '\n')
print('Performance addon: 4/4 checks passed', flush=True)
