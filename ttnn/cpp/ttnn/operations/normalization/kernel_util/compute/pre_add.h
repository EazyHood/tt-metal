// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file pre_add.h
 * @brief Helpers for fused pre-add (cb_in0 + cb_res -> cb_inp) in layernorm/rmsnorm
 *        distributed pre-allgather compute kernels.
 */

#pragma once

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace norm::kernel_util::compute::pre_add {

/**
 * Perform fused pre-add for one H row: cb_inp = cb_in0 + cb_res for Wt tiles,
 * processed in blocks of blk tiles. Compile-time no-op when !fuse_pre_add.
 *
 * use_sfpu=true: copy_tile + add_binary_tile (full FP32 with UnpackToDestFp32).
 * use_sfpu=false: add_tiles (FPU / TF32).
 */
template <bool fuse_pre_add, bool use_sfpu = false>
ALWI void one_row(CircularBuffer& cb_in0, CircularBuffer& cb_res, CircularBuffer& cb_inp, uint32_t Wt, uint32_t blk) {
    if constexpr (!fuse_pre_add) {
        return;
    }
    reconfig_data_format(cb_in0.get_cb_id(), cb_res.get_cb_id());
    pack_reconfig_data_format(cb_inp.get_cb_id());
    for (uint32_t wt = 0; wt < Wt; wt += blk) {
        cb_in0.wait_front(blk);
        cb_res.wait_front(blk);
        cb_inp.reserve_back(blk);
        if constexpr (use_sfpu) {
            copy_tile_to_dst_init_short(cb_in0.get_cb_id());
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                tile_regs_acquire();
                copy_tile(cb_in0.get_cb_id(), wtr, 0);
                copy_tile_to_dst_init_short_with_dt(cb_in0.get_cb_id(), cb_res.get_cb_id());
                copy_tile(cb_res.get_cb_id(), wtr, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_inp.get_cb_id());
                tile_regs_release();
                copy_tile_to_dst_init_short_with_dt(cb_res.get_cb_id(), cb_in0.get_cb_id());
            }
        } else {
            add_tiles_init(cb_in0.get_cb_id(), cb_res.get_cb_id());
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                add_tiles(cb_in0.get_cb_id(), cb_res.get_cb_id(), wtr, wtr, wtr);
                pack_tile(wtr, cb_inp.get_cb_id());
            }
            tile_regs_commit();
            tile_regs_release();
        }
        cb_inp.push_back(blk);
        cb_in0.pop_front(blk);
        cb_res.pop_front(blk);
    }
}

}  // namespace norm::kernel_util::compute::pre_add
