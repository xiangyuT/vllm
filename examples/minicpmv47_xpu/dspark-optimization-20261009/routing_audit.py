"""Observe persistent native router buffers only for a separate audit request."""

import json
import threading
import time
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensor
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm_xpu_kernels.moe_shared_fused_interface import XpuMoESharedFusedDecode

ROOT = Path("/workspace/evidence")
HELPERS = {}
STATE = {"active": False, "rounds": [], "phase_active": False, "phases": []}


def install():
    original_buffers = XpuMoESharedFusedDecode._routing_buffers
    original_replay = CudaGraphManager.run_fullgraph

    def buffers(self, m, device):
        result = original_buffers(self, m, device)
        weight = self.router_weight
        if weight is not None and not isinstance(weight, FakeTensor):
            HELPERS[(weight.data_ptr(), m)] = self
        return result

    def replay(self, desc):
        profile = STATE["phase_active"] and len(STATE["phases"]) < 128
        if profile:
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
        result = original_replay(self, desc)
        if profile:
            end.record()
            end.synchronize()
            STATE["phases"].append(
                {
                    "manager": type(self).__name__,
                    "query_tokens": desc.num_tokens,
                    "gpu_elapsed_ms": start.elapsed_time(end),
                    "timing_boundary": "events outside XPUGraph replay; synchronized audit request",
                }
            )
            (ROOT / f"phase-events-rank{torch.xpu.current_device()}.json").write_text(
                json.dumps(
                    {
                        "evidence_class": "separate_gpu_event_phase_trace",
                        "excluded_from_performance_samples": True,
                        "events": STATE["phases"],
                    },
                    indent=2,
                )
                + "\n"
            )
        if not STATE["active"] or len(STATE["rounds"]) >= 64:
            return result
        if type(self).__name__ != "ModelCudaGraphManager":
            return result
        m = desc.num_tokens
        if m not in (4, 8, 16):
            return result
        helpers = [v for (_, r), v in sorted(HELPERS.items()) if r == m]
        if len(helpers) != 40:
            (
                ROOT / f"routing-incomplete-rank{torch.xpu.current_device()}.json"
            ).write_text(
                json.dumps(
                    {"expected_layers": 40, "observed_helpers": len(helpers), "m": m}
                )
                + "\n"
            )
            return result
        ids = torch.stack([v._routing[m][2] for v in helpers]).cpu().tolist()
        for layer in ids:
            assert all(
                len(set(row)) == 8 and all(0 <= x < 256 for x in row) for row in layer
            )
        STATE["rounds"].append(
            {
                "m": m,
                "topk_ids_by_router": ids,
                "unique_experts_by_router": [
                    len({x for row in layer for x in row}) for layer in ids
                ],
            }
        )
        (ROOT / f"routing-rank{torch.xpu.current_device()}.json").write_text(
            json.dumps(
                {
                    "evidence_class": "separate_graph_replay_routing_audit",
                    "full_verification_positions_including_rejected": True,
                    "profiling_data_excluded_from_latency_samples": True,
                    "router_order": "weight storage pointer order; geometry equal for all 40 layers",
                    "rounds": STATE["rounds"],
                },
                indent=2,
            )
            + "\n"
        )
        return result

    def control():
        while True:
            STATE["active"] = (ROOT / "routing-audit.enabled").exists()
            STATE["phase_active"] = (ROOT / "phase-events.enabled").exists()
            time.sleep(0.5)

    XpuMoESharedFusedDecode._routing_buffers = buffers
    CudaGraphManager.run_fullgraph = replay
    threading.Thread(target=control, daemon=True).start()
