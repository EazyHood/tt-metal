// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_interleaved_start_id.cpp. The legacy copy stays in place for
// its (~29) unmigrated co-borrowers; this fork carries the named-binding rewrite for ops already
// on Metal 2.0. Sunset the legacy copy once the last co-borrower migrates (keep the two in sync
// until then). See reduction/prod/METAL2_PORT_REPORT.md.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_pages = get_arg(args::num_pages);
    const auto start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb(dfb::out);

    // Get page size from the DFB (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb.get_entry_size();

#ifdef OUT_SHARDED
    dfb.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::output);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb.wait_front(onepage);
        noc.async_write(dfb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
