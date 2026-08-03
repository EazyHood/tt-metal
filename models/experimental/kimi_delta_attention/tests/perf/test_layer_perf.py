# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight correctness and performance acceptance for Kimi-K3 KDA."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.experimental.kimi_delta_attention.tests.utils import (
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
    run_profiled_forward,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.parametrize(
        "device_params",
        [
            {
                "l1_small_size": 24576,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 256 * 1024 * 1024,
            }
        ],
        indirect=True,
    ),
]

_SEQUENCE = 5120
_REPETITIONS = 10
_PCC_THRESHOLD = 0.98


def _flatten_shards(tensor: ttnn.Tensor) -> torch.Tensor:
    return torch.cat([ttnn.to_torch(shard).float().reshape(-1) for shard in ttnn.get_device_tensors(tensor)])


def _capture(layer: KimiDeltaAttention, output: ttnn.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    return (
        _flatten_shards(output),
        _flatten_shards(layer.recurrent_state),
        _flatten_shards(layer.convolution_state),
    )


def _reset_external_state(layer: KimiDeltaAttention) -> None:
    layer.reset_state(batch_size=1)
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    layer.set_external_state(layer.recurrent_state, layer.convolution_state)


def _trace_wall_ms(
    mesh_device: ttnn.MeshDevice,
    layer: KimiDeltaAttention,
    hidden: ttnn.Tensor,
    repetitions: int,
) -> float:
    _reset_external_state(layer)
    warm_output = layer.forward(hidden)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_output)
    _reset_external_state(layer)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = layer.forward(hidden)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(repetitions):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(output)
    return elapsed * 1e3 / repetitions


def _assert_pcc(name: str, expected: torch.Tensor, actual: torch.Tensor) -> float:
    passed, value = comp_pcc(expected, actual, pcc=_PCC_THRESHOLD)
    print(f"{name}: PCC={value:.6f}")
    assert passed, f"{name} PCC {value:.6f} < {_PCC_THRESHOLD}"
    return value


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2"],
)
def test_kimi_k3_layer_1_perf(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_checkpoint_dir: Path,
) -> None:
    """Check profiled output/state reproducibility before timing the same case."""
    sequence = int(os.getenv("KIMI_K3_PERF_SEQ", str(_SEQUENCE)))
    case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=sequence)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    layer, hidden_tt = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=tensor_parallel_axis)

    _reset_external_state(layer)
    warm_output = layer.forward(hidden_tt)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_output)

    _reset_external_state(layer)
    start = time.perf_counter()
    expected_output = layer.forward(hidden_tt)
    ttnn.synchronize_device(mesh_device)
    unprofiled_ms = (time.perf_counter() - start) * 1e3
    expected = _capture(layer, expected_output)
    ttnn.deallocate(expected_output)

    _reset_external_state(layer)
    start = time.perf_counter()
    profiled_output, records = run_profiled_forward(mesh_device, lambda: layer.forward(hidden_tt))
    profiled_ms = (time.perf_counter() - start) * 1e3
    actual = _capture(layer, profiled_output)
    ttnn.deallocate(profiled_output)
    pcc = {
        name: _assert_pcc(name, expected_value, actual_value)
        for name, expected_value, actual_value in zip(
            ("output", "recurrent", "convolution"),
            expected,
            actual,
            strict=True,
        )
    }

    repetitions = int(os.getenv("PERF_REPS", str(_REPETITIONS)))
    wall_ms = _trace_wall_ms(mesh_device, layer, hidden_tt, repetitions)
    mesh_shape = tuple(mesh_device.shape)
    result = {
        "layout": f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}",
        "sequence": sequence,
        "repetitions": repetitions,
        "pcc": pcc,
        "pcc_reference": "identical restored-state device forward",
        "trace_wall_ms": wall_ms,
        "realtime_program_records": len(records),
        "unprofiled_forward_ms": unprofiled_ms,
        "profiled_forward_and_collection_ms": profiled_ms,
        "profile_collection_overhead_pct": 100.0 * (profiled_ms - unprofiled_ms) / unprofiled_ms,
    }
    print("KDA_LAYER_PERF=" + json.dumps(result, sort_keys=True))
