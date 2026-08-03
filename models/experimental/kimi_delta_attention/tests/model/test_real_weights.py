# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 layer-1 KDA PCC with the pinned Hugging Face weights."""

from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.kimi_k3_config import KimiK3Config
from models.experimental.kimi_delta_attention.tests.utils import (
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
    run_profiled_forward,
)
from models.experimental.kimi_delta_attention.tt.weights import KDAWeights

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def test_kimi_k3_layer_1_real_weights_pcc(
    mesh_device: ttnn.MeshDevice,
    kimi_k3_checkpoint_dir: Path,
) -> None:
    case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=32)
    golden_output, golden_state = kda_forward_reference(case.hidden, case.state_dict, case.config)
    cache_path = case.checkpoint_dir / "ttnn_cache"
    cache_prefix = f"layer_{KimiK3Config.FIRST_KDA_LAYER}.kda"
    if not KDAWeights.check_cache_complete(cache_path, cache_prefix, case.config, mesh_device):
        KDAWeights.build_ttnn_cache(case.state_dict, cache_path, cache_prefix, mesh_device, case.config)
    assert KDAWeights.check_cache_complete(cache_path, cache_prefix, case.config, mesh_device)
    cached_weights = KDAWeights.from_cache(mesh_device, case.config, cache_path, cache_prefix)
    layer, hidden_tt = make_kimi_k3_device_case(mesh_device, case, weights=cached_weights)
    layer.reset_state(batch_size=1)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, records = run_profiled_forward(mesh_device, lambda: layer.forward(hidden_tt))
    print(f"Kimi-K3 layer 1 realtime program records: {len(records)}")

    actual_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    actual_recurrent = torch.cat(_host_shards(layer.recurrent_state), dim=1)
    convolution_shards = _host_shards(layer.convolution_state)
    local_heads = case.config.num_heads // 8
    local_key_width = local_heads * case.config.head_k_dim
    local_value_width = local_heads * case.config.head_v_dim
    actual_convolution = torch.cat(
        (
            torch.cat([shard[..., :local_key_width] for shard in convolution_shards], dim=-1),
            torch.cat([shard[..., local_key_width : 2 * local_key_width] for shard in convolution_shards], dim=-1),
            torch.cat(
                [
                    shard[..., 2 * local_key_width : 2 * local_key_width + local_value_width]
                    for shard in convolution_shards
                ],
                dim=-1,
            ),
        ),
        dim=-1,
    )
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )

    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, actual_recurrent),
        ("convolution state", golden_convolution, actual_convolution),
    ):
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        print(f"Kimi-K3 layer 1 {name}: PCC={pcc:.6f}")
        assert passed, f"Kimi-K3 layer 1 {name} PCC {pcc:.6f} < 0.98"
