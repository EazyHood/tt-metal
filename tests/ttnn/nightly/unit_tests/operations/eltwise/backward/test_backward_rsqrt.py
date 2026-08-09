# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_rsqrt(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -201, 199, device, True)

    tt_output_tensor_on_device = ttnn.rsqrt_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.rsqrt_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_rsqrt_opt_output(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -201, 199, device, True)

    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.rsqrt_bw(grad_tensor, input_tensor, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.rsqrt_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("grad_value", [2.0, -2.0, 0.0])
def test_bw_rsqrt_at_zero(input_shapes, device, grad_value):
    # input == 0 is the one point the two tests above cannot reach: they draw continuous random
    # data over a range, which never lands on exactly 0.0. It is also the only place where the
    # result is an infinity whose *sign* carries the information, so the comparison has to be
    # exact rather than by magnitude or correlation.
    in_data = torch.zeros(input_shapes, dtype=torch.bfloat16, requires_grad=True)
    grad_data = torch.full(input_shapes, grad_value, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(in_data.detach(), layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn.to_torch(ttnn.rsqrt_bw(grad_tensor, input_tensor)[0])

    golden_function = ttnn.get_golden_function(ttnn.rsqrt_bw)
    golden = golden_function(grad_data, in_data)[0]

    # rtol=atol=0 so that +inf and -inf cannot compare equal; equal_nan so that the grad == 0
    # case, where the golden is NaN, still passes.
    torch.testing.assert_close(tt_output, golden, rtol=0, atol=0, equal_nan=True)
