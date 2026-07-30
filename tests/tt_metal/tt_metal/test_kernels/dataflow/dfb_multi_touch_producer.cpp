// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <bool ImplicitSync>
static inline void touch_one(uint16_t dfb_id) {
    DataflowBuffer dfb(dfb_id);
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

    for (uint32_t i = 0; i < num_dfbs; ++i) {
        touch_one<implicit_sync>(static_cast<uint16_t>(i));
    }
}
