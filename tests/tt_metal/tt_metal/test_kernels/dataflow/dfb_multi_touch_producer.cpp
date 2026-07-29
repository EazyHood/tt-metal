// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <typename DfbTag, bool ImplicitSync>
static inline void touch_one() {
    DataflowBuffer dfb(DfbTag{});
    (void)dfb.get_entry_size();
    if constexpr (ImplicitSync) {
        (void)dfb.get_id();
    } else {
        dfb.reserve_back(1);
        dfb.push_back(1);
    }
    dfb.finish();
}

void kernel_main() {
    constexpr uint32_t num_dfbs = get_arg(args::num_dfbs);
    constexpr bool implicit_sync = get_arg(args::implicit_sync);

    if constexpr (num_dfbs > 0) {
        touch_one<dfb::dfb_0, implicit_sync>();
    }
    if constexpr (num_dfbs > 1) {
        touch_one<dfb::dfb_1, implicit_sync>();
    }
    if constexpr (num_dfbs > 2) {
        touch_one<dfb::dfb_2, implicit_sync>();
    }
    if constexpr (num_dfbs > 3) {
        touch_one<dfb::dfb_3, implicit_sync>();
    }
    if constexpr (num_dfbs > 4) {
        touch_one<dfb::dfb_4, implicit_sync>();
    }
    if constexpr (num_dfbs > 5) {
        touch_one<dfb::dfb_5, implicit_sync>();
    }
    if constexpr (num_dfbs > 6) {
        touch_one<dfb::dfb_6, implicit_sync>();
    }
    if constexpr (num_dfbs > 7) {
        touch_one<dfb::dfb_7, implicit_sync>();
    }
    if constexpr (num_dfbs > 8) {
        touch_one<dfb::dfb_8, implicit_sync>();
    }
    if constexpr (num_dfbs > 9) {
        touch_one<dfb::dfb_9, implicit_sync>();
    }
    if constexpr (num_dfbs > 10) {
        touch_one<dfb::dfb_10, implicit_sync>();
    }
    if constexpr (num_dfbs > 11) {
        touch_one<dfb::dfb_11, implicit_sync>();
    }
    if constexpr (num_dfbs > 12) {
        touch_one<dfb::dfb_12, implicit_sync>();
    }
    if constexpr (num_dfbs > 13) {
        touch_one<dfb::dfb_13, implicit_sync>();
    }
    if constexpr (num_dfbs > 14) {
        touch_one<dfb::dfb_14, implicit_sync>();
    }
    if constexpr (num_dfbs > 15) {
        touch_one<dfb::dfb_15, implicit_sync>();
    }
}
