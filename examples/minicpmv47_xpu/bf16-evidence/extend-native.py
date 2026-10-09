"""Generate the native BF16 patch against the preserved PR worktree."""

import difflib
from pathlib import Path
import re


root = Path(__file__).resolve().parents[1] / "kernels-bf16"
changes = {}


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def save(name, text):
    changes[name] = text


def typed_entry(text, name, start_marker, arguments):
    start = text.index(start_marker)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    signature = text[start:opening].rstrip()
    implementation = text[start:end].replace(name + "(", name + "_impl(", 1)
    wrapper = (
        signature + " {\n"
        f"  if (x.scalar_type() == at::kHalf)\n"
        f"    return {name}_impl<sycl::half>({arguments});\n"
        f"  return {name}_impl<sycl::ext::oneapi::bfloat16>({arguments});\n"
        "}"
    )
    return text[:start] + "template <typename Scalar>\n" + implementation + "\n\n" + wrapper + text[end:]


name = "csrc/xpu/gemv/xe_2/fp8_gemv.hpp"
s = (root / name).read_text()
s = s.replace("#include <sycl/sycl.hpp>", "#include <sycl/sycl.hpp>\n#include <sycl/ext/oneapi/bfloat16.hpp>\n#include <type_traits>")
s = once(s, "template <int MP>\nstruct Dpas;", "template <int MP, typename Scalar = sycl::half>\nstruct Dpas;")
for mp in (1, 2, 4, 8):
    s = once(s, f"template <>\nstruct Dpas<{mp}>", f"template <typename Scalar>\nstruct Dpas<{mp}, Scalar>")
    s = once(s, f"    cute::XE_{mp}x16x16_F32F16F16F32_TT::fma(c, a, b, c);", f"    if constexpr (std::is_same_v<Scalar, sycl::half>)\n      cute::XE_{mp}x16x16_F32F16F16F32_TT::fma(c, a, b, c);\n    else\n      cute::XE_{mp}x16x16_F32BF16BF16F32_TT::fma(c, a, b, c);")
start = s.index("static inline uint32_t e4m3_even(")
end = s.index("// DPAS A operands", start)
s = s[:start] + """// Decode an E4M3 byte into BF16 bits at the same 2^-8 scale as FP16.
// Subnormals must be normalized explicitly: BF16 has a different bias.
static inline uint16_t e4m3_bf16(uint32_t byte) {
  const uint16_t sign = uint16_t((byte & 0x80u) << 8);
  const uint32_t magnitude = byte & 0x7fu;
  const uint32_t exponent = magnitude >> 3;
  const uint32_t mantissa = magnitude & 7u;
  if (magnitude == 0x7fu) return sign | 0x7fc0u;
  if (exponent != 0)
    return sign | uint16_t((exponent + 112u) << 7) | uint16_t(mantissa << 4);
  if (mantissa == 0) return sign;
  const uint32_t leading = mantissa >= 4 ? 2 : (mantissa >= 2 ? 1 : 0);
  return sign | uint16_t((110u + leading) << 7) |
         uint16_t((mantissa - (1u << leading)) << (7u - leading));
}

template <typename Scalar = sycl::half>
static inline uint32_t e4m3_even(uint32_t v) {
  if constexpr (std::is_same_v<Scalar, sycl::half>)
    return ((v & 0x00800080u) << 8) | ((v & 0x007F007Fu) << 7);
  else
    return uint32_t(e4m3_bf16(v & 0xffu)) |
           (uint32_t(e4m3_bf16((v >> 16) & 0xffu)) << 16);
}
template <typename Scalar = sycl::half>
static inline uint32_t e4m3_odd(uint32_t v) {
  if constexpr (std::is_same_v<Scalar, sycl::half>)
    return (v & 0x80008000u) | ((v & 0x7F007F00u) >> 1);
  else
    return uint32_t(e4m3_bf16((v >> 8) & 0xffu)) |
           (uint32_t(e4m3_bf16(v >> 24)) << 16);
}

""" + s[end:]
s = once(s, "template <int MP>\nstatic inline void load_a_global(", "template <int MP, typename Scalar>\nstatic inline void load_a_global(")
s = once(s, "    const sycl::half* a, int K, int M, int pitch_bytes, int kc, AOps<MP>& A)", "    const Scalar* a, int K, int M, int pitch_bytes, int kc, AOps<MP>& A)")
s = once(s, "template <int MP>\nstruct GlobalA {\n  const sycl::half* a;", "template <int MP, typename Scalar = sycl::half>\nstruct GlobalA {\n  const Scalar* a;")
s = once(s, "template <int MP>\nstatic inline void compute_chunk(", "template <int MP, typename Scalar>\nstatic inline void compute_chunk(")
s = s.replace("e4m3_even(wr[t][i])", "e4m3_even<Scalar>(wr[t][i])").replace("e4m3_odd(wr[t][i])", "e4m3_odd<Scalar>(wr[t][i])")
s = s.replace("Dpas<MP>::mma", "Dpas<MP, Scalar>::mma")
s = once(s, "struct Seg {", "template <typename Scalar = sycl::half>\nstruct Seg {")
s = once(s, "  sycl::half* out;", "  Scalar* out;")
s = once(s, "template <int MP, int KS, int RG, class ALoad, int PRE = 1, int CPS = 0>", "template <int MP, int KS, int RG, class ALoad, int PRE = 1, int CPS = 0,\n          typename Scalar = sycl::half>")
s = s.replace("const Seg& seg", "const Seg<Scalar>& seg")
s = once(s, "  using D = Dpas<MP>;", "  using D = Dpas<MP, Scalar>;")
s = once(s, "compute_chunk<MP>(wr, A, acc_e, acc_o);", "compute_chunk<MP, Scalar>(wr, A, acc_e, acc_o);")
s = s.replace("sycl::half(vget", "Scalar(vget").replace("sycl::half(v * s)", "Scalar(v * s)")
s = once(s, "        sycl::half* o = seg.out", "        Scalar* o = seg.out")
s = s.replace("sycl::vec<sycl::half, RG> hv;", "sycl::vec<uint16_t, RG> hv;")
s = s.replace("hv[u] = sycl::half(v[u]);", "hv[u] = sycl::bit_cast<uint16_t>(Scalar(v[u]));")
s = s.replace("reinterpret_cast<sycl::vec<sycl::half, RG>*>", "reinterpret_cast<sycl::vec<uint16_t, RG>*>")
s = s.replace("o[u] = sycl::half(v[u]);", "o[u] = Scalar(v[u]);")
s = once(s, "template <int PRE, int KS>\nstatic inline bool preload_w(", "template <int PRE, int KS, typename Scalar>\nstatic inline bool preload_w(")
s = once(s, "template <int MP>\nstatic inline void stage_a_slm(", "template <int MP, typename Scalar>\nstatic inline void stage_a_slm(")
s = s.replace("    const sycl::half* a,\n    int64_t pitch,", "    const Scalar* a,\n    int64_t pitch,")
s = s.replace("FP8-E4M3 x FP16", "FP8-E4M3 x FP16/BF16").replace("a is FP16.", "a is FP16 or BF16.")
save(name, s)

name = "csrc/xpu/gemv/xe_2/fp8_gemv_xe2.cpp"
s = (root / name).read_text()
s = re.sub(r"template <([^>]+)>\n(?=(?:struct Fp8GemvKernel|static void (?:launch_gemv|launch_mp|run_cfg|run_gemv)))", r"template <\1, typename Scalar>\n", s)
s = s.replace("sycl::half", "Scalar")
s = re.sub(r"\bSeg\b", "Seg<Scalar>", s)
s = s.replace("GlobalA<MP>", "GlobalA<MP, Scalar>").replace("Fp8GemvKernel<MP, KS, RG, NSEG>", "Fp8GemvKernel<MP, KS, RG, NSEG, Scalar>")
supported = "(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)"
s = s.replace("x.scalar_type() != at::kHalf", "!" + supported).replace("x.scalar_type() == at::kHalf", supported, 1)
# The first replacement also contains == kHalf: restore its nested spelling.
s = s.replace("((x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) || x.scalar_type() == at::kBFloat16)", supported)
s = once(s, 'TORCH_CHECK(x.scalar_type() == at::kHalf, "fp8_gemv: x must be float16");', f'TORCH_CHECK({supported}, "fp8_gemv: x must be float16 or bfloat16");')
s = typed_entry(s, "fp8_gemv_w8a16", "torch::Tensor fp8_gemv_w8a16(", "x, w, scale")
s = typed_entry(s, "fp8_gemv2_w8a16", "std::tuple<torch::Tensor, torch::Tensor> fp8_gemv2_w8a16(", "x, w1, scale1, w2, scale2")
save(name, s)

name = "csrc/xpu/gemv/xe_2/norm_fp8_gemv_xe2.cpp"
s = (root / name).read_text()
s = s.replace("sycl::half", "Scalar")
for helper in ("half_bits", "lo_f", "hi_f"):
    marker = "static inline " + ("uint16_t " if helper == "half_bits" else "float ") + helper
    s = once(s, marker, "template <typename Scalar>\n" + marker)
    s = s.replace(helper + "(", helper + "<Scalar>(")
    s = s.replace(marker + "<Scalar>(", marker + "(")
s = s.replace("half_bits", "scalar_bits")
typed_names = ("GatedNormA", "ResAddNormA", "GatedNormGemvKernel", "ResAddNormGemvKernel", "ResAddRegA", "GatedRegA", "ResAddNormGemvRegKernel", "GatedNormGemvRegKernel")
for typ in typed_names:
    s = re.sub(r"template <([^>]+)>\nstruct " + typ + r"\b", r"template <\1, typename Scalar>\nstruct " + typ, s)
    s = re.sub(r"\b" + typ + r"<([^>]+)>", typ + r"<\1, Scalar>", s)
for helper in ("load_chunks", "store_chunks", "launch_resadd"):
    s = re.sub(r"template <([^>]+)>\n(static (?:inline void|void) " + helper + r"\()", r"template <\1, typename Scalar>\n\2", s)
for typ in ("GatedNormKernel", "ResAddNormKernel"):
    s = once(s, "struct " + typ + " {", "template <typename Scalar>\nstruct " + typ + " {")
    s = s.replace(typ + "{", typ + "<Scalar>{")
s = re.sub(r"\bSeg\b", "Seg<Scalar>", s)
s = s.replace("t.scalar_type() != at::kHalf", "(t.scalar_type() != at::kHalf && t.scalar_type() != at::kBFloat16)")
s = s.replace("wn.scalar_type() == at::kHalf", "wn.scalar_type() == x.scalar_type()")
s = s.replace("r.scalar_type() == at::kHalf", "r.scalar_type() == x.scalar_type()")
s = s.replace("x.scalar_type() == at::kHalf", "(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)")
s = s.replace("wn.scalar_type() != at::kHalf", "wn.scalar_type() != x.scalar_type()")
s = s.replace("x.scalar_type() != at::kHalf", "(x.scalar_type() != at::kHalf && x.scalar_type() != at::kBFloat16)")
s = s.replace("r.scalar_type() != at::kHalf", "r.scalar_type() != x.scalar_type()")
s = once(s, "heads_ok(z, M, H, D, zs[0], zs[1]) && z.device() == x.device() &&", "heads_ok(z, M, H, D, zs[0], zs[1]) && z.scalar_type() == x.scalar_type() && z.device() == x.device() &&")
s = once(s, "t.device() != ref.device())", "(t.device() != ref.device() || t.scalar_type() != ref.scalar_type()))")
for fname, marker, args in (
    ("gated_rmsnorm_fp8_gemv", "torch::Tensor gated_rmsnorm_fp8_gemv(", "x, z, norm_weight, eps, w, scale"),
    ("resadd_rmsnorm_fp8_gemv", "std::tuple<torch::Tensor, torch::Tensor> resadd_rmsnorm_fp8_gemv(", "x, residual, norm_weight, eps, w, scale"),
    ("resadd_rmsnorm_fp8_gemv2", "std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>\nresadd_rmsnorm_fp8_gemv2(", "x, residual, norm_weight, eps, w1, scale1, w2, scale2"),
    ("gated_rmsnorm_kernel", "static std::optional<torch::Tensor> gated_rmsnorm_kernel(", "x, z, wn, eps"),
    ("resadd_rmsnorm_kernel", "static std::optional<std::tuple<torch::Tensor, torch::Tensor>>\nresadd_rmsnorm_kernel(", "x, r, wn, eps"),
):
    s = typed_entry(s, fname, marker, args)
s = s.replace("must be fp16", "must be matching fp16/bf16").replace('"fp16 [', '"fp16/bf16 [')
save(name, s)

name = "csrc/xpu/grouped_gemm/xe_2/moe_shared_fused_decode_xe2.cpp"
s = (root / name).read_text()
s = s.replace("using ElementA = cutlass::half_t;\nusing ElementD = cutlass::half_t;\n", "").replace("using ElementBI = cutlass::half_t;\n", "")
for typ in ("SpGatherName", "SpSiluMulName", "SpFinalizeName"):
    s = s.replace("class " + typ + ";", "template <typename ElementA>\nclass " + typ + ";")
    s = s.replace("parallel_for<" + typ + ">", "parallel_for<" + typ + "<ElementA>>")
s = once(s, "template <class Policy, bool SingleGroup>", "template <class Policy, bool SingleGroup, typename ElementA>")
s = s.replace("SpGemmName<Policy, false>", "SpGemmName<Policy, false, ElementA>")
for typ in ("SpGatherLauncher", "SpSiluMulLauncher", "SpFinalizeLauncher"):
    s = s.replace("class " + typ + " {", "template <typename ElementA>\nclass " + typ + " {")
    s = s.replace(typ + "::launch", typ + "<ElementA>::launch")
s = s.replace("ElementD", "ElementA").replace("ElementBI", "ElementA")
s = once(s, "template <class Policy, typename ElementB>", "template <class Policy, typename ElementB, typename ElementA>")
s = s.replace("cutlass::float_e4m3_t>(", "cutlass::float_e4m3_t, ElementA>(")
s = once(s, "void moe_shared_fused_decode_xe2(", "template <typename ElementA>\nvoid moe_shared_fused_decode_xe2_impl(")
s = once(s, "x.scalar_type() == at::kHalf && output.scalar_type() == at::kHalf", "(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) && output.scalar_type() == x.scalar_type()")
s = s.replace("shared_gate_t.scalar_type() == at::kHalf", "shared_gate_t.scalar_type() == x.scalar_type()")
s = s.replace("shared_gate->scalar_type() == at::kHalf", "shared_gate->scalar_type() == x.scalar_type()")
s = s.replace("only supports fp16 activations", "only supports fp16/bf16 activations").replace("x and output must be float16", "x and output must have the same float16 or bfloat16 dtype").replace("shared_gate must be float16", "shared_gate must have the input dtype")
original = (root / name).read_text()
signature = original[original.index("void moe_shared_fused_decode_xe2("):]
signature = signature[:signature.index("{")].rstrip()
args = "output, x, topk_ids, topk_weights, w13, w13_scale, w2, w2_scale, shared_w13_t, shared_w13_scale_t, shared_w2_t, shared_w2_scale_t, shared_gate_t, ws, tile_m"
s += "\n\n" + signature + " {\n  if (x.scalar_type() == at::kBFloat16)\n    return moe_shared_fused_decode_xe2_impl<cutlass::bfloat16_t>(" + args + ");\n  return moe_shared_fused_decode_xe2_impl<cutlass::half_t>(" + args + ");\n}\n"
save(name, s)

name = "csrc/moe/moe_router_decode.cpp"
s = (root / name).read_text()
s = s.replace("#include <sycl/sycl.hpp>", "#include <sycl/sycl.hpp>\n#include <sycl/ext/oneapi/bfloat16.hpp>")
s = s.replace("sycl::half", "Scalar")
s = once(s, "struct alignas(16) half8 {", "template <typename Scalar>\nstruct alignas(16) Scalar8 {")
s = re.sub(r"\bhalf8\b", "Scalar8<Scalar>", s)
for helper in ("sg_load8", "sg_store8", "router_key", "router_key_logit", "router_dot8", "router_sumsq8", "router_norm8", "router_add8"):
    s = re.sub(r"(^static inline [^;{]*\b" + helper + r"\()", r"template <typename Scalar>\n\1", s, count=1, flags=re.M)
for helper in ("router_key", "router_key_logit"):
    s = s.replace(helper + "(", helper + "<Scalar>(")
    s = re.sub(r"(^static inline (?:uint32_t|float) " + helper + r")<Scalar>", r"\1", s, flags=re.M)
s = once(s, "struct RouterArgs {", "template <typename Scalar>\nstruct RouterArgs {")
s = re.sub(r"\bRouterArgs\b(?!\s*\{)", "RouterArgs<Scalar>", s)
s = once(s, "template <int MR, int EPL, bool NORM>\nclass router_kernel", "template <int MR, int EPL, bool NORM, typename Scalar>\nclass router_kernel")
for helper in ("router_launch", "router_dispatch_e", "router_dispatch"):
    s = re.sub(r"template <([^>]+)>\n(static void " + helper + r"\()", r"template <\1, typename Scalar>\n\2", s)
s = s.replace("router_kernel<MR, EPL, NORM>(", "router_kernel<MR, EPL, NORM, Scalar>(")
s = once(s, "static RouterArgs<Scalar> router_make_args(", "template <typename Scalar>\nstatic RouterArgs<Scalar> router_make_args(")
s = s.replace("x.dim() == 2 && x.scalar_type() == at::kHalf", "x.dim() == 2 && (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)")
s = s.replace("router_weight.scalar_type() == at::kHalf", "router_weight.scalar_type() == x.scalar_type()")
s = s.replace("t.scalar_type() == at::kHalf", "t.scalar_type() == x.scalar_type()")
s = s.replace("norm_weight.scalar_type() == at::kHalf", "norm_weight.scalar_type() == x.scalar_type()")
s = once(s, "  const RouterArgs<Scalar> a = router_make_args(\n      x, router_weight, logits, topk_weights, topk_ids, renormalize, queue);\n  router_dispatch<false>(queue, a);", """  auto launch = [&](auto scalar) {
    using Scalar = decltype(scalar);
    const RouterArgs<Scalar> a = router_make_args<Scalar>(
        x, router_weight, logits, topk_weights, topk_ids, renormalize, queue);
    router_dispatch<false>(queue, a);
  };
  if (x.scalar_type() == at::kHalf)
    launch(sycl::half{});
  else
    launch(sycl::ext::oneapi::bfloat16{});""")
start = s.index("  RouterArgs<Scalar> a = router_make_args(")
end = s.index("  router_dispatch<true>(queue, a);", start) + len("  router_dispatch<true>(queue, a);")
block = s[start:end].replace("router_make_args(", "router_make_args<Scalar>(")
s = s[:start] + "  auto launch = [&](auto scalar) {\n    using Scalar = decltype(scalar);\n" + block + "\n  };\n  if (x.scalar_type() == at::kHalf)\n    launch(sycl::half{});\n  else\n    launch(sycl::ext::oneapi::bfloat16{});" + s[end:]
s = s.replace("must be fp16", "must be fp16/bf16").replace("contiguous fp16", "contiguous matching fp16/bf16").replace("aligned fp16", "aligned matching fp16/bf16")
save(name, s)

for name, content in changes.items():
    original = (root / name).read_text()
    print("".join(difflib.unified_diff(original.splitlines(keepends=True), content.splitlines(keepends=True), fromfile="a/" + name, tofile="b/" + name)), end="")
