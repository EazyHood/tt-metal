# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import ttnn
import torch
from loguru import logger

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)

DEFAULT_SHAPE = (32, 32)
SHAPES = [tuple([32] * i) for i in range(6)]


def check_standard_normal_distribution(data: torch.Tensor, tt_dtype) -> bool:
    n = data.numel()

    if n < 10000:
        logger.warning("[Warning] A meaningful analysis requires at least 10000 samples.")
        if n < 2:
            logger.warning("[Error] Cannot perform test with less than 2 data points.")
            return False

    mean = data.mean().item()
    std = data.std(unbiased=False).item()

    if tt_dtype == ttnn.float32:
        mean_tol = 0.05
        std_tol = 0.05
    else:
        mean_tol = 0.08
        std_tol = 0.10

    if abs(mean) > mean_tol:
        logger.warning("Mean too far from 0: mean={}, tol={}", mean, mean_tol)
        return False
    if abs(std - 1.0) > std_tol:
        logger.warning("Std too far from 1: std={}, tol={}", std, std_tol)
        return False

    absx = data.abs()
    within1 = (absx <= 1.0).to(torch.float32).mean().item()
    within2 = (absx <= 2.0).to(torch.float32).mean().item()
    within3 = (absx <= 3.0).to(torch.float32).mean().item()

    if not (0.60 <= within1 <= 0.75):
        logger.warning("within1 out of range: {}", within1)
        return False
    if not (0.90 <= within2 <= 0.99):
        logger.warning("within2 out of range: {}", within2)
        return False
    if not (0.98 <= within3 <= 1.00):
        logger.warning("within3 out of range: {}", within3)
        return False

    return True


def test_randn_defaults(device):
    tensor = ttnn.randn(DEFAULT_SHAPE, device=device)

    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert tensor.storage_type() == ttnn.StorageType.DEVICE
    assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


@pytest.mark.parametrize("shapes", SHAPES)
def test_randn_shapes(device, shapes):
    tensor = ttnn.randn(shapes, device=device)
    assert tuple(tensor.shape) == tuple(shapes)


@pytest.mark.parametrize("dim", [i for i in range(32)])
def test_randn_dims(dim, device):
    shape = (dim, dim)
    tensor = ttnn.randn(shape, device=device)
    assert tuple(tensor.shape) == tuple(shape)


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_randn_with_memory_config(device, mem_config):
    tensor = ttnn.randn(DEFAULT_SHAPE, device=device, memory_config=mem_config)
    assert tensor.memory_config() == mem_config
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_randn_dtype_layout_and_distribution(device, dtype, layout):
    shape = (1024, 1024)

    tensor = ttnn.randn(shape, device=device, dtype=dtype, layout=layout, seed=1234)

    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == tuple(shape)

    torch_tensor = ttnn.to_torch(tensor)
    assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"
    assert check_standard_normal_distribution(torch_tensor, dtype), "Random values do not look standard normal!"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_randn_dtype_layout_and_distribution_no_check(device, dtype, layout):
    shape = (1024, 1024)

    tensor = ttnn.randn(shape, device=device, dtype=dtype, layout=layout, seed=1234)

    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == tuple(shape)

    torch_tensor = ttnn.to_torch(tensor)
    assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"


def test_randn_invalid_args(device, expect_error):
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.randn(5, device=device)

    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.randn([2, -1], device=device)

    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.randn([2, 2], device=device, layout="ROW_MAJOR")

    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.randn([2, 2], device=device, memory_config="DRAM")

    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.randn([2, 2], device="WORMHOLE")

    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.randn([2, 2], device=device, dtype="ttnn.bfloat16")


@pytest.mark.parametrize("shape", [[2, 32, 32, 16]])
@pytest.mark.parametrize("seed", [1234])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_randn_callback(shape, seed, dtype, device):
    num_program_cache_entries_list = []
    for i in range(2):
        ttnn.randn(shape, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, seed=seed + i)
        ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "shard_strategy, core_grid",
    [
        (ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=4, x=8)),  # shard (32, 1024)
        (ttnn.ShardStrategy.WIDTH, ttnn.CoreGrid(y=8, x=4)),  # shard (1024, 32)
        (ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(y=8, x=8)),  # shard (128, 128)
    ],
)
def test_randn_sharded_l1(device, dtype, layout, shard_strategy, core_grid):
    shape = (1024, 1024)

    shard_mem_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=core_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    tensor = ttnn.randn(shape, device=device, dtype=dtype, layout=layout, memory_config=shard_mem_config, seed=1234)

    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == shape
    assert tensor.memory_config().is_sharded()

    torch_tensor = ttnn.to_torch(tensor)
    assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"
    assert check_standard_normal_distribution(torch_tensor, dtype), "Random values do not look standard normal!"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_randn_interleaved_distribution(device, dtype, layout, mem_config):
    shape = (1024, 1024)

    tensor = ttnn.randn(shape, device=device, dtype=dtype, layout=layout, memory_config=mem_config, seed=1234)

    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == shape
    assert tensor.memory_config() == mem_config

    torch_tensor = ttnn.to_torch(tensor)
    assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"
    assert check_standard_normal_distribution(torch_tensor, dtype), "Random values do not look standard normal!"


@pytest.mark.parametrize("shape", [[512, 512], [5, 8, 70, 40]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_randn_with_compute_kernel_options(shape, dtype, device, compute_kernel_options):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    if compute_kernel_config is None:
        pytest.skip("Kernel option is not available")

    tensor = ttnn.randn(
        shape,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        seed=1234,
        compute_kernel_config=compute_kernel_config,
    )

    assert list(tensor.shape) == list(shape)
    assert tensor.dtype == dtype

    torch_tensor = ttnn.to_torch(tensor)
    assert check_standard_normal_distribution(torch_tensor, dtype), "Random values do not look standard normal!"


def test_randn_unseeded_calls_differ(device):
    """Consecutive unseeded randn calls must draw independent samples.

    The unseeded path used to build a fresh std::mt19937 from std::time(nullptr) on every
    dispatch. std::time has one-second resolution, so every call made within the same
    wall-clock second was given an identical seed and returned a bitwise-identical tensor.
    """
    outputs = [ttnn.to_torch(ttnn.randn((256, 256), device=device, dtype=ttnn.bfloat16)) for _ in range(4)]

    duplicates = [
        (i, j) for i in range(len(outputs)) for j in range(i + 1, len(outputs)) if torch.equal(outputs[i], outputs[j])
    ]
    assert not duplicates, f"unseeded randn returned identical tensors for call pairs {duplicates}"


def test_randn_same_seed_is_reproducible(device):
    """An explicit seed must still give reproducible output."""
    a = ttnn.to_torch(ttnn.randn((256, 256), device=device, dtype=ttnn.bfloat16, seed=1234))
    b = ttnn.to_torch(ttnn.randn((256, 256), device=device, dtype=ttnn.bfloat16, seed=1234))
    assert torch.equal(a, b), "randn with the same explicit seed must be reproducible"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_randn_default_compute_config_is_accurate(device, dtype):
    """randn without a compute_kernel_config must give the same values as with math_approx_mode=False.

    In approximate mode sqrt_tile is only good to about 10 bits, which moves the Box-Muller radius of every pair.
    """
    accurate_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False
    )
    default = ttnn.to_torch(ttnn.randn((256, 256), device=device, dtype=dtype, seed=1234))
    accurate = ttnn.to_torch(
        ttnn.randn((256, 256), device=device, dtype=dtype, seed=1234, compute_kernel_config=accurate_config)
    )
    assert torch.equal(default, accurate), "randn without a compute_kernel_config must match math_approx_mode=False"


def test_randn_float32_box_muller_radius(device):
    """The radius of each Box-Muller pair must follow its exact distribution.

    randn draws U1 and U2, and writes Z1 = R cos(2 pi U2) and Z2 = R sin(2 pi U2), with R = sqrt(-2 ln U1), to two
    consecutive tiles on the same core. exp(-(Z1^2 + Z2^2) / 2) gives back U1, which must be uniform on (0, 1).
    A square root good to about 10 bits (math_approx_mode=True) leaves ripples in that histogram, which a chi-squared
    test over 2**20 pairs picks up.
    """
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    # An even number of tiles on every core keeps tiles 2i and 2i + 1 on the same pairs; at least 2**20 pairs in total.
    tiles_per_core = 2 * math.ceil(1024 / num_cores)
    shape = (tiles_per_core * num_cores * 32, 32)

    z = ttnn.to_torch(ttnn.randn(shape, device=device, dtype=ttnn.float32, seed=1234)).double()
    z = z.reshape(-1, 2, 32, 32)
    # Uniform for any two independent standard normals, so a different pairing cannot make this check fail.
    u1 = torch.exp(-(z[:, 0] ** 2 + z[:, 1] ** 2) / 2).flatten()

    bins = 100
    counts = torch.bincount((u1 * bins).long().clamp(max=bins - 1), minlength=bins).double()
    expected = u1.numel() / bins
    chi2 = ((counts - expected) ** 2 / expected).sum()
    # Chi-squared survival function with bins - 1 degrees of freedom.
    p_value = torch.special.gammaincc(torch.tensor((bins - 1) / 2, dtype=torch.float64), chi2 / 2).item()
    logger.info(f"{u1.numel()} pairs: chi2={chi2.item():.1f} with {bins - 1} degrees of freedom, p={p_value:.2e}")
    assert p_value > 1e-6, f"exp(-(Z1^2 + Z2^2) / 2) is not uniform: chi2={chi2.item():.1f}, p={p_value:.2e}"
