"""Run metadata-only fusion tests on a CPU platform without GPU fixtures."""

import os
from pathlib import Path
import sys

import pytest


root = Path(__file__).resolve().parents[1] / "vllm-bf16"
sys.path.insert(0, str(root))
os.chdir(root)

import vllm.platforms
from vllm.platforms.cpu import CpuPlatform

vllm.platforms._current_platform = CpuPlatform()


class CpuFixtures:
    @pytest.fixture
    def cleanup_fixture(self):
        # These cases use CPU/meta tensors and have no accelerator allocator.
        yield


selection = (
    "xpu_moe_shared or rewrites_every_layer or input_norm or wrong_dtype "
    "or base_case_supported or bf16_runner or each_precondition "
    "or fp8_per_tensor or range_gating or unsupported "
    "or router_in_not_hidden or input_ids_unchanged or extra_getitem "
    "or non_add_combine or kernel_rejects or no_shared_layer "
    "or experts_not_xpu or routed_scale_not"
)
raise SystemExit(
    pytest.main(
        [
            "--confcutdir=tests/compile",
            "tests/compile/passes/test_xpu_fusion.py",
            "tests/compile/test_config.py",
            "-k",
            selection,
            "-q",
            "-o",
            "cache_dir=/workspace/build/pytest-cache",
        ],
        plugins=[CpuFixtures()],
    )
)
