// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <typename DfbTag, bool ImplicitSync, uint32_t Index>
static inline void touch_one(volatile uint32_t* results, uint32_t touched_magic) {
    DataflowBuffer dfb(DfbTag{});
    if constexpr (!ImplicitSync) {
        dfb.wait_front(1);
        dfb.pop_front(1);
    }
    dfb.finish();
    results[Index * 3 + 0] = dfb.get_entry_size();
    results[Index * 3 + 1] = dfb.get_id();
    results[Index * 3 + 2] = touched_magic | Index;
}

void kernel_main() {
    constexpr uint32_t num_dfbs = get_arg(args::num_dfbs);
    constexpr bool implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t touched_magic = get_arg(args::touched_magic);
    volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(get_arg(args::result_l1_addr));

    if constexpr (num_dfbs > 0) {
        touch_one<dfb::dfb_0, implicit_sync, 0>(results, touched_magic);
    }
    if constexpr (num_dfbs > 1) {
        touch_one<dfb::dfb_1, implicit_sync, 1>(results, touched_magic);
    }
    if constexpr (num_dfbs > 2) {
        touch_one<dfb::dfb_2, implicit_sync, 2>(results, touched_magic);
    }
    if constexpr (num_dfbs > 3) {
        touch_one<dfb::dfb_3, implicit_sync, 3>(results, touched_magic);
    }
    if constexpr (num_dfbs > 4) {
        touch_one<dfb::dfb_4, implicit_sync, 4>(results, touched_magic);
    }
    if constexpr (num_dfbs > 5) {
        touch_one<dfb::dfb_5, implicit_sync, 5>(results, touched_magic);
    }
    if constexpr (num_dfbs > 6) {
        touch_one<dfb::dfb_6, implicit_sync, 6>(results, touched_magic);
    }
    if constexpr (num_dfbs > 7) {
        touch_one<dfb::dfb_7, implicit_sync, 7>(results, touched_magic);
    }
    if constexpr (num_dfbs > 8) {
        touch_one<dfb::dfb_8, implicit_sync, 8>(results, touched_magic);
    }
    if constexpr (num_dfbs > 9) {
        touch_one<dfb::dfb_9, implicit_sync, 9>(results, touched_magic);
    }
    if constexpr (num_dfbs > 10) {
        touch_one<dfb::dfb_10, implicit_sync, 10>(results, touched_magic);
    }
    if constexpr (num_dfbs > 11) {
        touch_one<dfb::dfb_11, implicit_sync, 11>(results, touched_magic);
    }
    if constexpr (num_dfbs > 12) {
        touch_one<dfb::dfb_12, implicit_sync, 12>(results, touched_magic);
    }
    if constexpr (num_dfbs > 13) {
        touch_one<dfb::dfb_13, implicit_sync, 13>(results, touched_magic);
    }
    if constexpr (num_dfbs > 14) {
        touch_one<dfb::dfb_14, implicit_sync, 14>(results, touched_magic);
    }
    if constexpr (num_dfbs > 15) {
        touch_one<dfb::dfb_15, implicit_sync, 15>(results, touched_magic);
    }
}
