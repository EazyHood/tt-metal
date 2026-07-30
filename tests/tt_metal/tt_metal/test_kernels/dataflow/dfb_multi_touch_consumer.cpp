// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <bool ImplicitSync>
static inline void touch_one(uint16_t dfb_id, uint32_t index, volatile uint32_t* results, uint32_t touched_magic) {
    DataflowBuffer dfb(dfb_id);
    if constexpr (!ImplicitSync) {
        dfb.wait_front(1);
        dfb.pop_front(1);
    }
    dfb.finish();
    results[index * 3 + 0] = dfb.get_entry_size();
    results[index * 3 + 1] = dfb.get_id();
    results[index * 3 + 2] = touched_magic | index;
}

void kernel_main() {
    constexpr uint32_t num_dfbs = get_arg(args::num_dfbs);
    constexpr bool implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t touched_magic = get_arg(args::touched_magic);
    volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(get_arg(args::result_l1_addr));

    for (uint32_t i = 0; i < num_dfbs; ++i) {
        touch_one<implicit_sync>(static_cast<uint16_t>(i), i, results, touched_magic);
    }
}
