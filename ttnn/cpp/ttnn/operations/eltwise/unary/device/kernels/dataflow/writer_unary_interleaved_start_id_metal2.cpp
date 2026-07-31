// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_interleaved_start_id.cpp.
//
// Same logic, expressed against the Metal 2.0 named-binding APIs: the output CB index CTA became
// the `dfb::out` DFB binding, and the destination buffer-address RTA plus the
// TensorAccessorArgs<N>() CTA chain collapsed into `TensorAccessor(tensor::dst)`.
//
// This fork exists because the legacy original is bound by ~two dozen op directories that cannot all
// convert at once; it lives alongside the original rather than replacing it. Ops still on the legacy
// host API keep binding the original.
//
// Binding vocabulary a Metal 2.0 KernelSpec must supply for this source:
//   dfb::out      — the output DFB, bound CONSUMER
//   tensor::dst   — the destination tensor (omit when OUT_SHARDED is defined; unused there)
//   args::num_pages, args::start_id — runtime args
//   optional defines: OUT_SHARDED, BACKWARDS (same meaning as in the legacy original)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb(dfb::out);

    // Get page size from the DFB (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb.get_entry_size();

#ifdef OUT_SHARDED
    dfb.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::dst);

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
