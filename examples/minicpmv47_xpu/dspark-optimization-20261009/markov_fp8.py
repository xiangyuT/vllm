"""Opt-in experiment: per-tensor online FP8 Markov weights, BF16 activations."""

import os

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def install():
    original_load = DSparkSpeculator.load_draft_model
    original_bias = DSparkMarkovHead.bias
    original_logits = Qwen3DSparkForCausalLM.compute_draft_logits
    backend = os.environ.get("MINICPM_MARKOV_BACKEND", "generic")
    vocab_fp8 = os.environ.get("MINICPM_DRAFT_VOCAB_FP8") == "1"
    if backend == "onednn" or vocab_fp8:
        torch.ops.load_library(
            "/workspace/preparation/dspark-optimization-20261009/markov-flat.so"
        )

    def prepare(weight, valid):
        scale = weight[:valid].float().abs().amax().clamp_min(1e-12) / 448.0
        quantized = (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        quantized[valid:] = 0
        return quantized, scale.reshape(1)

    def load(self, *args, **kwargs):
        model = original_load(self, *args, **kwargs)
        head = model.model.markov_head
        weight = head.markov_w2.weight
        assert weight.device.type == "xpu" and weight.dtype == torch.bfloat16
        assert not head.markov_w2.is_w4a16_nvfp4
        assert self._draft_topk is None, "This experiment uses full-vocabulary Markov"
        with torch.no_grad():
            vocab = model.config.draft_vocab_size
            quantized, scale = prepare(weight, vocab)
            head.register_buffer("_markov_fp8_weight", quantized.t())
            head.register_buffer("_markov_fp8_scale", scale)
            if vocab_fp8:
                lm_weight = model.lm_head.weight
                assert lm_weight.dtype == torch.bfloat16
                rank = get_tensor_model_parallel_rank()
                valid = min(lm_weight.shape[0], vocab - rank * lm_weight.shape[0])
                lm_quantized, lm_scale = prepare(lm_weight, valid)
                # Keep buffers on the draft model, including when lm_head is
                # shared with the target. The target's module is never patched.
                model.register_buffer("_draft_vocab_fp8_weight", lm_quantized)
                model.register_buffer("_draft_vocab_fp8_scale", lm_scale)
                print(
                    "MINICPM DRAFT VOCAB FP8 prepared",
                    list(lm_weight.shape),
                    "valid_local_rows",
                    valid,
                    flush=True,
                )
        print(
            "MINICPM MARKOV FP8 prepared",
            list(weight.shape),
            "activation",
            weight.dtype,
            "weights",
            quantized.dtype,
            flush=True,
        )
        return model

    def bias(self, markov_embed, logits_processor):
        if not hasattr(self, "_markov_fp8_weight"):
            return original_bias(self, markov_embed, logits_processor)
        assert logits_processor.soft_cap is None and logits_processor.scale == 1.0
        if backend == "onednn":
            logits = torch.ops._minicpm_markov2_C.onednn(
                markov_embed, self._markov_fp8_weight.t(), self._markov_fp8_scale
            )
        else:
            logits = torch.ops._xpu_C.fp8_gemm_w8a16(
                markov_embed, self._markov_fp8_weight, self._markov_fp8_scale, None
            )
        return logits[..., : logits_processor.org_vocab_size]

    def logits(self, hidden):
        if not hasattr(self, "_draft_vocab_fp8_weight"):
            return original_logits(self, hidden)
        processor = self.logits_processor
        assert processor.soft_cap is None and processor.scale == 1.0
        local = torch.ops._minicpm_markov2_C.onednn(
            hidden, self._draft_vocab_fp8_weight, self._draft_vocab_fp8_scale
        )
        gathered = processor._gather_logits(local)
        return (
            gathered[..., : processor.org_vocab_size] if gathered is not None else None
        )

    DSparkSpeculator.load_draft_model = load
    DSparkMarkovHead.bias = bias
    Qwen3DSparkForCausalLM.compute_draft_logits = logits
