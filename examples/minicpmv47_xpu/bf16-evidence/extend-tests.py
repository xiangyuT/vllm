"""Add BF16 cases to the existing kernel correctness suites."""

import difflib
from pathlib import Path


root = Path(__file__).resolve().parents[1] / "kernels-bf16"
changes = {}
dtype_cases = '@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])\n'


def parameterize(text, name):
    old = "def " + name + "("
    start = text.index(old)
    opening = text.index("(", start)
    closing = text.index("):", opening)
    args = text[opening + 1:closing].rstrip()
    return text[:start] + dtype_cases + text[start:opening + 1] + (args + ", dtype" if args else "dtype") + text[closing:]


name = "tests/test_fp8_gemv_dispatch.py"
s = (root / name).read_text()
for function in ("test_fp8_gemm_w8a16_small_m", "test_fp8_gemm_w8a16_strided_row", "test_fp8_gemm_w8a16_pair"):
    s = parameterize(s, function)
s = s.replace("dtype=torch.float16", "dtype=dtype")
s += '''

@pytest.mark.parametrize("value", [2.0**17, -(2.0**17), 2.0**-100])
def test_bf16_gemv_preserves_values_outside_fp16_range(value):
    """BF16 activations must not overflow or underflow through FP16."""
    x = torch.zeros(1, 64, device=DEVICE, dtype=torch.bfloat16)
    x[:, 0] = value
    w = torch.zeros(16, 64, device=DEVICE)
    w[:, 0] = 1
    w = w.to(torch.float8_e4m3fn)
    scale = torch.ones(1, device=DEVICE)
    out = torch.ops._xpu_C.fp8_gemm_w8a16(x, w.t(), scale, None)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, x[:, :1].expand(1, 16), atol=0, rtol=0)
'''
changes[name] = s

name = "tests/test_norm_fp8_gemv.py"
s = (root / name).read_text()
for function in ("test_gated_rmsnorm_fp8_gemm", "test_resadd_rmsnorm_fp8_gemm", "test_resadd_rmsnorm_fp8_gemm_pair", "test_unfused_norm_strided_inputs"):
    s = parameterize(s, function)
s = s.replace("dtype=torch.float16", "dtype=dtype")
s = s.replace("(w.float() * s).t()).half()", "(w.float() * s).t()).to(y.dtype)")
s = s.replace("torch.nn.functional.silu(z.float())).half()", "torch.nn.functional.silu(z.float())).to(dtype)")
s = s.replace("(nw.float() + 1)).half()", "(nw.float() + 1)).to(x.dtype)")
s = s.replace("return y, t.half()", "return y, t.to(x.dtype)")
s = s.replace("y.half().reshape", "y.to(dtype).reshape")
s += '''

@pytest.mark.parametrize("m", [1, 4])
def test_bf16_resadd_norm_keeps_large_residual(m):
    x = torch.full((m, 2048), 2.0**17, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    norm_weight = torch.zeros(2048, device=DEVICE, dtype=x.dtype)
    w = _w(32, 2048)
    scale = torch.tensor([0.02], device=DEVICE)
    out, residual_out = torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm(
        x, residual, norm_weight, EPS, w.t(), scale)
    y, expected_residual = _resadd_ref(x, residual, norm_weight)
    assert out.dtype == residual_out.dtype == torch.bfloat16
    torch.testing.assert_close(residual_out, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(out, _lin(y, w, scale), atol=3e-3, rtol=2e-2)
'''
changes[name] = s

name = "tests/fused_moe/test_router_gemv_topk_softmax.py"
s = (root / name).read_text()
for function in ("test_router_gemv_topk_softmax", "test_router_gemv_topk_softmax_strided_input", "test_router_resadd_norm_gemv_topk_softmax", "test_router_resadd_norm_strided", "test_router_back_to_back_and_streams"):
    s = parameterize(s, function)
s = s.replace("dtype=torch.float16", "dtype=dtype")
s = s.replace("return normed.to(torch.float16), s.to(torch.float16)", "return normed.to(x.dtype), s.to(x.dtype)")
# Comparing ids against the same rounded-logit reference keeps tie semantics.
s += '''

def test_bf16_router_does_not_round_logits_through_fp16():
    x = torch.zeros(1, 128, device=DEVICE, dtype=torch.bfloat16)
    x[0, 0] = 2.0**17
    w = torch.zeros(64, 128, device=DEVICE, dtype=x.dtype)
    w[:, 0] = torch.arange(1, 65, device=DEVICE).to(x.dtype)
    logits, weights, ids = _outputs(1, 64, 8)
    OPS.router_gemv_topk_softmax(x, w, logits, weights, ids, True)
    _check_topk(logits, weights, ids, _reference(x, w, 8, True))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_router_rejects_mixed_weight_dtype(dtype):
    x = torch.zeros(1, 128, device=DEVICE, dtype=dtype)
    other = torch.bfloat16 if dtype == torch.float16 else torch.float16
    w = torch.zeros(64, 128, device=DEVICE, dtype=other)
    logits, weights, ids = _outputs(1, 64, 8)
    with pytest.raises(RuntimeError, match="router_weight"):
        OPS.router_gemv_topk_softmax(x, w, logits, weights, ids, True)
'''
changes[name] = s

name = "tests/fused_moe/test_moe_shared_fused_decode.py"
s = (root / name).read_text()
s = s.replace("def make_weights(inter: int):", "def make_weights(inter: int, dtype=torch.float16):")
s = s.replace("gate = (torch.randn(1, HIDDEN, device=DEVICE) / 32).half()", "gate = (torch.randn(1, HIDDEN, device=DEVICE) / 32).to(dtype)")
s = s.replace("The SiLU-and-mul result is rounded to fp16 as the kernel stores it.", "Intermediate projections and activations are rounded to the input dtype.")
s = s.replace("h = xf[m] @ w13[e]", "h = (xf[m] @ w13[e]).to(x.dtype).float()")
s = s.replace("out[m] += topk_weights[m, k] * (act.half().float() @ w2[e])", "projection = (act.to(x.dtype).float() @ w2[e]).to(x.dtype).float()\n            out[m] += topk_weights[m, k] * projection")
s = s.replace("h = xf @ sw13", "h = (xf @ sw13).to(x.dtype).float()")
s = s.replace("out += torch.sigmoid(xf @ gate.float().t()) * (act.half().float() @ sw2)", "projection = (act.to(x.dtype).float() @ sw2).to(x.dtype).float()\n    out += torch.sigmoid(xf @ gate.float().t()) * projection")
for function in ("test_moe_shared_fused_decode", "test_moe_shared_fused_decode_int64_ids", "test_moe_shared_fused_decode_workspace_size"):
    s = parameterize(s, function)
    start = s.index("def " + function + "(")
    next_def = s.find("\ndef ", start + 1)
    end = len(s) if next_def < 0 else next_def
    block = s[start:end]
    block = block.replace("make_weights(inter)", "make_weights(inter, dtype=dtype)").replace("make_weights(256)", "make_weights(256, dtype=dtype)").replace("/ 4).half()", "/ 4).to(dtype)")
    s = s[:start] + block + s[end:]
s = s.replace("def test_moe_shared_fused_decode_rejects_non_fp16():", "def test_moe_shared_fused_decode_rejects_float32():")
s = s.replace("dtype=torch.bfloat16)", "dtype=torch.float32)")
s = s.replace('match="fp16 activations"', 'match="fp16/bf16 activations"')
s += '''

def test_moe_shared_fused_decode_rejects_mixed_shared_gate_dtype():
    seed_everything(7)
    weights = make_weights(256, dtype=torch.float16)
    op = make_op(weights)
    x = torch.randn(1, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    topk_weights, topk_ids = route(1)
    with pytest.raises(RuntimeError, match="shared_gate.*input dtype"):
        op.forward(torch.empty_like(x), x, topk_weights, topk_ids)
'''
changes[name] = s

for name, content in changes.items():
    original = (root / name).read_text()
    print("".join(difflib.unified_diff(original.splitlines(keepends=True), content.splitlines(keepends=True), fromfile="a/" + name, tofile="b/" + name)), end="")
