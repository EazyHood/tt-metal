// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_input_tiles = get_arg(args::num_input_tiles);
    const auto num_output_tiles = get_arg(args::num_output_tiles);

    // dfb::in = input DFB (c_0), dfb::out = output DFB (c_3).
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    DataflowBuffer cb_in0_obj(dfb::in);
    DataflowBuffer cb_out0_obj(dfb::out);

    binary_op_init_common(dfb::in, dfb::in, dfb::out);
    pack_reconfig_data_format(dfb::out);

    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        // Each output tile is an independent reduction of num_input_tiles.
        // The running product lives in DEST.
        tile_regs_acquire();

        // Seed DEST with the first input tile of this reduction.
        cb_in0_obj.wait_front(onetile);
        copy_tile_to_dst_init_short(dfb::in);
        copy_tile(dfb::in, 0, dst0);
        cb_in0_obj.pop_front(onetile);

        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::in);
        for (uint32_t j = 1; j < num_input_tiles; ++j) {
            cb_in0_obj.wait_front(onetile);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                dfb::in, 0, dst0);
            cb_in0_obj.pop_front(onetile);
        }

        tile_regs_commit();

        cb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst0, dfb::out);
        cb_out0_obj.push_back(onetile);
        tile_regs_release();
    }
}
