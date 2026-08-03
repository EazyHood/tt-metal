# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for KDA model configuration."""

import pytest

import ttnn
from models.experimental.kimi_delta_attention.config import KDAConfig, KDAProgramConfig
from models.experimental.kimi_delta_attention.kimi_k3_config import (
    KimiK3Config,
    kimi_k3_kda_config,
    kimi_k3_program_config,
)


def test_target_config_mapping() -> None:
    config = KDAConfig.from_model_config(
        {
            "hidden_size": 2304,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
            },
        }
    )

    assert config.hidden_size == 2304
    assert config.num_heads == 32
    assert config.head_k_dim == config.head_v_dim == 128
    assert config.q_dim == config.k_dim == config.v_dim == 4096
    assert config.conv_kernel_size == 4


def test_kimi_k3_config_mapping() -> None:
    config = kimi_k3_kda_config()

    assert config.hidden_size == KimiK3Config.HIDDEN_SIZE == 7168
    assert config.num_heads == KimiK3Config.KDA_NUM_HEADS == 96
    assert config.head_k_dim == config.head_v_dim == 128
    assert config.conv_kernel_size == 4
    assert config.use_full_rank_gate
    assert config.gate_lower_bound == -5.0


def test_program_config_p2p_links() -> None:
    assert KDAProgramConfig().p2p_num_links == 1
    assert kimi_k3_program_config().p2p_num_links == 2


def test_program_config_affine_summary_dtype() -> None:
    assert KDAProgramConfig().affine_summary_dtype == ttnn.float32
    assert kimi_k3_program_config().affine_summary_dtype == ttnn.bfloat16


def test_program_config_rejects_nonpositive_p2p_links(expect_error) -> None:
    with expect_error(ValueError, "p2p_num_links"):
        KDAProgramConfig(p2p_num_links=0)


@pytest.mark.parametrize("field", ["hidden_size", "num_heads", "head_k_dim", "head_v_dim"])
def test_config_rejects_nonpositive_dimensions(field: str, expect_error) -> None:
    values = {
        "hidden_size": 64,
        "num_heads": 2,
        "head_k_dim": 32,
        "head_v_dim": 32,
        "conv_kernel_size": 4,
        "norm_eps": 1e-5,
    }
    values[field] = 0
    with expect_error(ValueError, field):
        KDAConfig(**values)
