# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared deterministic KDA test cases and runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import ttnn
from models.experimental.kimi_delta_attention.config import KDAConfig, KDAProgramConfig
from models.experimental.kimi_delta_attention.checkpoint import load_kda_layer_state_dict
from models.experimental.kimi_delta_attention.kimi_k3_config import (
    KimiK3Config,
    kimi_k3_kda_config,
    kimi_k3_program_config,
)
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.experimental.kimi_delta_attention.tt.weights import KDAWeights
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


@dataclass(frozen=True)
class KimiK3TestCase:
    config: KDAConfig
    state_dict: dict[str, torch.Tensor]
    hidden: torch.Tensor
    checkpoint_dir: Path


def make_kimi_k3_test_case(checkpoint_dir: Path, *, sequence: int) -> KimiK3TestCase:
    """Load the pinned Kimi-K3 layer and deterministic input used by correctness and perf."""
    config = kimi_k3_kda_config()
    downloaded_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    assert KDAConfig.from_model_config(downloaded_config) == config
    state_dict = load_kda_layer_state_dict(checkpoint_dir, KimiK3Config.FIRST_KDA_LAYER, config)
    hidden = torch.randn(
        1,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed(1607),
        dtype=torch.bfloat16,
    )
    return KimiK3TestCase(config=config, state_dict=state_dict, hidden=hidden, checkpoint_dir=checkpoint_dir)


def make_kimi_k3_device_case(
    mesh_device: ttnn.MeshDevice,
    case: KimiK3TestCase,
    *,
    tensor_parallel_axis: int = 1,
    weights: KDAWeights | None = None,
) -> tuple[KimiDeltaAttention, ttnn.Tensor]:
    """Construct the real-weight layer and sequence-parallel device input."""
    sequence_parallel_axis = 1 - tensor_parallel_axis
    mesh_dims: list[int | None] = [None, None]
    mesh_dims[sequence_parallel_axis] = 1
    hidden = ttnn.from_torch(
        case.hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=tuple(mesh_dims),
            mesh_shape=tuple(mesh_device.shape),
        ),
    )
    layer = KimiDeltaAttention(
        mesh_device,
        case.config,
        case.state_dict if weights is None else None,
        tensor_cache_path=case.checkpoint_dir / "ttnn_cache",
        cache_name_prefix=f"layer_{KimiK3Config.FIRST_KDA_LAYER}.kda",
        weights=weights,
        tt_ccl=TT_CCL(mesh_device),
        tensor_parallel_axis=tensor_parallel_axis,
        program_config=kimi_k3_program_config(),
    )
    return layer, hidden


def run_profiled_forward(
    mesh_device: ttnn.MeshDevice,
    forward: Callable[[], ttnn.Tensor],
) -> tuple[ttnn.Tensor, list[dict[str, object]]]:
    """Run one correctness forward and require usable realtime-profiler records."""
    assert ttnn.device.IsProgramRealtimeProfilerActive(), "realtime profiler must be active for KDA correctness"
    output, records = profile_realtime_program(
        mesh_device,
        forward,
        collect_all=True,
        record_timeout_seconds=10.0,
    )
    non_sentinel_records = [record for record in records if int(record["runtime_id"]) != 0]
    assert non_sentinel_records, "realtime profiler returned no program records"
    return output, non_sentinel_records


def make_config() -> KDAConfig:
    return KDAConfig(
        hidden_size=64,
        num_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )


def make_program_config(*, recurrent_state_dtype: ttnn.DataType = ttnn.float32) -> KDAProgramConfig:
    return KDAProgramConfig(recurrent_state_dtype=recurrent_state_dtype)


def random_weights(config: KDAConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260723)

    def normal(*shape: int, scale: float = 0.05) -> torch.Tensor:
        return scale * torch.randn(*shape, generator=generator)

    hidden = config.hidden_size
    key_rank, value_rank = config.head_k_dim, config.head_v_dim
    weights = {
        "q_proj.weight": normal(config.q_dim, hidden),
        "k_proj.weight": normal(config.k_dim, hidden),
        "v_proj.weight": normal(config.v_dim, hidden),
        "q_conv1d.weight": normal(config.q_dim, 1, config.conv_kernel_size, scale=0.2),
        "k_conv1d.weight": normal(config.k_dim, 1, config.conv_kernel_size, scale=0.2),
        "v_conv1d.weight": normal(config.v_dim, 1, config.conv_kernel_size, scale=0.2),
        "A_log": torch.log(torch.linspace(1.0, 4.0, config.num_heads)).reshape(1, 1, config.num_heads, 1),
        "f_a_proj.weight": normal(key_rank, hidden),
        "f_b_proj.weight": normal(config.num_heads * key_rank, key_rank),
        "dt_bias": normal(config.num_heads * key_rank),
        "b_proj.weight": normal(config.num_heads, hidden),
        "o_norm.weight": 1.0 + normal(value_rank),
        "o_proj.weight": normal(hidden, config.num_heads * value_rank),
    }
    if config.use_full_rank_gate:
        weights["g_proj.weight"] = normal(config.v_dim, hidden)
    else:
        weights["g_a_proj.weight"] = normal(value_rank, hidden)
        weights["g_b_proj.weight"] = normal(config.num_heads * value_rank, value_rank)
    return weights
