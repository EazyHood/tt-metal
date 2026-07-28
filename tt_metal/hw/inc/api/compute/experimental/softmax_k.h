// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"
#ifdef TRISC_MATH
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_math_unary_datacopy_softmax_k_api.h"
#include "experimental/llk_sfpu/llk_math_softmax_k_api.h"
#endif
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A.h"
#endif

namespace ckernel {

#ifdef ARCH_BLACKHOLE

ALWI void softmax_k_init(uint32_t icb) {
    UNPACK((llk_unpack_A_init<BroadcastType::SCALAR, false, EltwiseBinaryReuseDestType::NONE, false>(false, 1, icb)));
    MATH((llk_math_eltwise_unary_datacopy_softmax_k_init()));
    MATH((sfpu::llk_math_sfpu_softmax_k_init()));
}

template <int k = 8>
ALWI void softmax_k(uint32_t icb) {
    // Unpack row0 to srcB
    UNPACK((llk_unpack_A<BroadcastType::SCALAR, false, EltwiseBinaryReuseDestType::NONE, false>(icb, 0)));

    // B2D direct copy and scalar broadcast.
    MATH((llk_math_eltwise_unary_datacopy_softmax_k(0)));

    // Softmax
    MATH((sfpu::llk_math_sfpu_softmax_k<k>()));
}

#endif

}  // namespace ckernel
