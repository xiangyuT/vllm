"""Opt-in experiment: per-tensor online FP8 Markov weights, BF16 activations."""

import torch
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def install():
    original_load = DSparkSpeculator.load_draft_model
    original_bias = DSparkMarkovHead.bias

    def load(self, *args, **kwargs):
        model = original_load(self, *args, **kwargs)
        head = model.model.markov_head
        weight = head.markov_w2.weight
        assert weight.device.type == "xpu" and weight.dtype == torch.bfloat16
        assert not head.markov_w2.is_w4a16_nvfp4
        assert self._draft_topk is None, "This experiment uses full-vocabulary Markov"
        with torch.no_grad():
            vocab = model.config.draft_vocab_size
            scale = weight[:vocab].float().abs().amax().clamp_min(1e-12) / 448.0
            quantized = (
                (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            )
            quantized[vocab:] = 0
            head.register_buffer("_markov_fp8_weight", quantized.t())
            head.register_buffer("_markov_fp8_scale", scale.reshape(1))
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
        logits = torch.ops._xpu_C.fp8_gemm_w8a16(
            markov_embed, self._markov_fp8_weight, self._markov_fp8_scale, None
        )
        return logits[..., : logits_processor.org_vocab_size]

    DSparkSpeculator.load_draft_model = load
    DSparkMarkovHead.bias = bias
