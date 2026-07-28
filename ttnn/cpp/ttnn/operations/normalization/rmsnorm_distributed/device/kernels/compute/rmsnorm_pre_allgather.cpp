// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
 * For rmsnorm we compute E(x**2) and return it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/operations/normalization/kernel_util/compute/pre_add.h"

namespace pre_add = norm::kernel_util::compute::pre_add;

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    // True iff the factory configured UnpackToDestFp32 on input/scratch CBs.
    constexpr bool unpack_fp32_active = get_named_compile_time_arg_val("unpack_fp32_active") != 0;
    constexpr auto reduce_type = unpack_fp32_active ? PoolType::SUM : PoolType::AVG;
    constexpr auto fp32_mode = unpack_fp32_active ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce_id = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    constexpr uint32_t cb_x2_id = tt::CBIndex::c_6;   // x**2
    constexpr uint32_t cb_res_id = tt::CBIndex::c_5;  // residual b (unused when !FUSE_PRE_ADD)
    constexpr uint32_t cb_inp_id = FUSE_PRE_ADD ? tt::CBIndex::c_3 : cb_in0_id;  // fused a + b, or just a

    if constexpr (FUSE_PRE_ADD) {
        binary_op_init_common(cb_in0_id, cb_res_id, cb_inp_id);
    } else {
        binary_op_init_common(cb_inp_id, cb_reduce_id, cb_x2_id);
    }

    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_res(cb_res_id);
    CircularBuffer cb_inp(cb_inp_id);
    CircularBuffer cb_x2(cb_x2_id);
    CircularBuffer cb_reduce(cb_reduce_id);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        pre_add::one_row<FUSE_PRE_ADD, unpack_fp32_active>(cb_in0, cb_res, cb_inp, Wt, blk);

        /*
         * x**2
         */
        reconfig_data_format(cb_inp_id, cb_inp_id);
        pack_reconfig_data_format(cb_x2_id);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_inp.wait_front(wt + blk);  // cumulative wait
            cb_x2.reserve_back(blk);

            if constexpr (unpack_fp32_active) {
                copy_tile_to_dst_init_short(cb_inp_id);
                square_tile_init();
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    tile_regs_acquire();
                    copy_tile(cb_inp_id, wt + wtr, 0);
                    square_tile(0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_x2_id, wt + wtr);
                    tile_regs_release();
                }
            } else {
                mul_tiles_init(cb_inp_id, cb_inp_id);
                ACQ();
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    mul_tiles(cb_inp_id, cb_inp_id, wt + wtr, wt + wtr, wtr);
                    pack_tile(wtr, cb_x2_id, wt + wtr);
                }
                REL();
            }
            cb_x2.push_back(blk);
        }

        /*
         * sum(x**2)
         */
        compute_kernel_lib::reduce<
            reduce_type,
            ReduceDim::REDUCE_ROW,
            cb_x2_id,
            cb_reduce_id,
            cb_out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            fp32_mode>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        cb_inp.pop_front(Wt);
    }
    cb_reduce.pop_front(1);
}
