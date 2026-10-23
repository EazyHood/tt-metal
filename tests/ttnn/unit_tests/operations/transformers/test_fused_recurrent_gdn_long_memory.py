# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.transformer.fused_recurrent_gated_delta_rule must carry its state over many tokens without drift.

The state is decayed and updated on every token, so any rounding in that path compounds. These cases use gates close
to 1 (the long-memory channels, which are where a GDN layer keeps context) and enough tokens for a per-token bias of
~1e-3 to show, while a single read's operand rounding in the q . S and k . S matmuls (~1e-3, not compounded) stays
under the thresholds. The last test checks the update itself where it nearly cancels the decayed state.
"""

import zlib

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import (
    fla_naive_recurrent_gated_delta_rule,
    make_inputs,
)

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
K = V = 128


def _run_op(device, q, k, v, g, beta, initial_state):
    """The op on fp32 TILE tensors; returns (o, final state) as torch tensors."""

    def dev(x):
        return ttnn.from_torch(x.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    o, state = OP(dev(q), dev(k), dev(v), dev(g), dev(beta), initial_state=dev(initial_state), output_final_state=True)
    return ttnn.to_torch(o), ttnn.to_torch(state)


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def test_decay_only_state_is_the_product_of_the_decays(device):
    """beta = 0 removes the update, so after T tokens the state is S0 * exp(g_1) * ... * exp(g_T). With g = -1e-4
    (a 10k-token memory) and T = 128 the exact state keeps 98.7% of S0; a 1e-3 per-token loss would leave 87%."""
    B, T, H = 1, 128, 1
    x = make_inputs(B, T, H, H, K, V, seed=0)
    g = torch.full((B, T, H), -1e-4)
    beta = torch.zeros(B, T, H)
    s0 = torch.randn(B, H, K, V, generator=torch.Generator().manual_seed(1))

    _, state = _run_op(device, x["q"], x["k"], x["v"], g, beta, s0)

    expected = s0.double() * torch.exp(g.double()).prod(dim=1)[..., None, None]
    assert _rel(state, expected) < 1e-4


@pytest.mark.parametrize("H, HV", [(1, 1), (4, 12)])
def test_chained_single_token_decode_tracks_the_reference(device, H, HV):
    """The model's decode loop: one token per call, the state handed back in fp32 every time. Gates in [-0.01, 0]
    (decays 0.99 to 1); the reference is FLA's naive recurrence over the same 64 tokens."""
    B, T = 1, 64
    x = make_inputs(B, T, H, HV, K, V, seed=2)
    g = -0.01 * torch.rand(B, T, HV, generator=torch.Generator().manual_seed(3))

    state = x["initial_state"]
    for t in range(T):
        sl = slice(t, t + 1)
        _, state = _run_op(device, x["q"][:, sl], x["k"][:, sl], x["v"][:, sl], g[:, sl], x["beta"][:, sl], state)

    groups = HV // H
    _, expected = fla_naive_recurrent_gated_delta_rule(
        x["q"].repeat_interleave(groups, dim=2),
        x["k"].repeat_interleave(groups, dim=2),
        x["v"],
        x["beta"],
        g,
        initial_state=x["initial_state"],
        output_final_state=True,
    )
    assert _rel(state, expected) < 1e-2


@pytest.mark.parametrize(
    "key, B, T, H, HV, row, tokens",
    [("qwen27b_tp4-4-8", 4, 8, 4, 12, 2, 2), ("qwen35b_a3b_tp4-1-12", 1, 12, 4, 8, 0, 3)],
    ids=["qwen27b_tp4", "qwen35b_a3b_tp4"],
)
def test_update_that_cancels_the_decayed_state(device, key, B, T, H, HV, row, tokens):
    """S_new = sd + k (x) u where the two nearly cancel. make_inputs with these seeds (one batch row, the first
    tokens) holds such an element in the last token; an FPU accumulation of k (x) u onto the fp32 sd came out off by a
    power of two there (0.34 and 0.19 of max |S| on a Wormhole N150). Elsewhere the error is about 2e-3."""
    x = make_inputs(B, T, H, HV, K, V, seed=zlib.crc32(key.encode()))
    q, k, v, g, beta = (x[n][row : row + 1, :tokens] for n in ("q", "k", "v", "g", "beta"))
    s0 = x["initial_state"][row : row + 1]

    _, state = _run_op(device, q, k, v, g, beta, s0)

    groups = HV // H
    _, expected = fla_naive_recurrent_gated_delta_rule(
        q.repeat_interleave(groups, dim=2),
        k.repeat_interleave(groups, dim=2),
        v,
        beta,
        g,
        initial_state=s0,
        output_final_state=True,
    )
    err = (state.double() - expected.double()).abs().max() / expected.double().abs().max()
    assert err < 1e-2
