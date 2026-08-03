// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "llk_defs.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reduce_rm_dataflow_common.hpp"

//
// Dense RM reduce writer (handles both W reduce and H reduce; branched on REDUCE_DIM).
//
// Compute packs one or more output tiles per work unit into dfb_id_tile (c_3). The writer extracts
// the meaningful datums from those tiles and emits them into the corresponding RM output pages.
//
// W reduce path (REDUCE_DIM == REDUCE_ROW):
//   Output: one scalar per reduced logical row, one RM page per scalar (page_id == logical row).
//   Each compute output tile carries up to TILE_HEIGHT row reductions in column 0 of the tile
//   (intra-tile via get_tilized_idx(r, 0)). Writer emits up to TILE_HEIGHT separate
//   datum_bytes-sized writes per popped tile, each to its own page.
//
// H reduce path (REDUCE_DIM == REDUCE_COL):
//   Compute produces wt_in_chunk output tiles per work unit, each tile holding the reduced row of
//   W datums in tile-row 0 (rows 1..TILE_HEIGHT-1 unused). Under an H-axis split the work units are
//   (nc, slice, wt) triples over a (N, C, num_h_slices, W) result; num_h_slices == 1 is the plain
//   (N, C, 1, W) reduce. Two output layouts:
//
//   ROW_MAJOR (tile_output == 0): one page per (n, c, slice), page width == W. Writer extracts
//   tile-row 0 face-by-face (1–2 wide writes per tile, each up to half_tile_width datums) into that
//   page at offset (w_base_col * datum_bytes), clamping the last W tile to W_logical so we don't
//   overflow the destination page.
//
//   TILE (tile_output != 0): one page per output tile. Work unit (nc, slice, wt) owns tile-row
//   (slice % TILE_HEIGHT) of the tile at (nc, slice / TILE_HEIGHT, wt), so the writer re-lands its
//   tile-row 0 at that row via get_tilized_idx(row_in_tile, face_col). Different slices of the same
//   output tile therefore hit disjoint, face-aligned byte ranges and may live on different cores —
//   no read-modify-write, and every offset is a multiple of half_tile_width datums (32 B at BF16,
//   64 B at FP32), which satisfies the DRAM write alignment. The trailing rows of the last
//   tile-row-block (num_h_slices % TILE_HEIGHT .. TILE_HEIGHT-1) belong to no work unit, so the
//   core owning the final slice also fills them with the reduction identity from the reader's clear
//   template — upholding the usual "TILE padding carries the reduction identity" invariant that a
//   downstream tile reduce relies on.
//

template <ckernel::ReduceDim DIM>
void reduce_rm_writer() {
    //
    // Runtime args. Slots shared between paths; semantics differ:
    //   W reduce: (dst_addr, num_rows, start_page)
    //   H reduce: (dst_addr, num_output_tiles_local, start_output_tile_id)
    //
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t rt_count = get_arg_val<uint32_t>(1);
    const uint32_t rt_start = get_arg_val<uint32_t>(2);

    //
    // Compile-time args. Slot 0 (datum_bytes) is shared. The H reduce path adds Wt, W_logical and
    // wt_tiles_per_chunk at slots 1-3, plus the output-layout/split geometry (tile_output,
    // num_h_slices, out_tile_rows) at slots 4-6, so the dst TensorAccessor args start at slot 1 (W)
    // or slot 7 (H).
    //
    constexpr uint32_t datum_bytes = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<(DIM == ckernel::ReduceDim::REDUCE_ROW) ? 1 : 7>();

    constexpr uint32_t dfb_id_tile = tt::CBIndex::c_3;
    constexpr uint32_t onetile = 1;

    const uint32_t tile_size_bytes = get_tile_size(dfb_id_tile);
    const auto dst_accessor = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb_tile(dfb_id_tile);

    if constexpr (DIM == ckernel::ReduceDim::REDUCE_ROW) {
        //
        // === W reduce path ===
        //
        // One scalar per logical row, one page per scalar. Compute emits one tile at a time
        // (single-shot path for MAX with wt_tiles_per_chunk == Wt, or chunked SUM that ends on
        // is_last_chunk for each ht). Either way the writer sees one tile per pop.
        //
        const uint32_t num_rows = rt_count;
        const uint32_t start_page = rt_start;

        uint32_t rows_written = 0;
        while (rows_written < num_rows) {
            dfb_tile.wait_front(onetile);
            const uint32_t rows_this_tile = ((num_rows - rows_written) < tt::constants::TILE_HEIGHT)
                                                ? (num_rows - rows_written)
                                                : tt::constants::TILE_HEIGHT;
            for (uint32_t r = 0; r < rows_this_tile; ++r) {
                const uint32_t tile_scalar_idx = get_tilized_idx(r, 0);
                noc.async_write(
                    dfb_tile,
                    dst_accessor,
                    datum_bytes,
                    {.offset_bytes = tile_scalar_idx * datum_bytes},
                    {.page_id = start_page + rows_written + r, .offset_bytes = 0});
            }
            noc.async_write_barrier();
            dfb_tile.pop_front(onetile);
            rows_written += rows_this_tile;
        }
    } else {
        //
        // === H reduce path ===
        //
        // Either layout emits 1–2 face-wise wide writes per output tile; they differ in where those
        // writes land (a (n, c, slice) row-major page vs tile row slice % TILE_HEIGHT of an output
        // tile) and in whether the trailing tile-padding rows need an identity fill. See the header.
        //
        // Wt, W_logical, wt_tiles_per_chunk and the output-layout/split geometry are only consumed
        // here, so they live in this branch. The indices embed DIM to make them value-dependent: a
        // literal `get_compile_time_arg_val(N)` is non-dependent and would be eagerly instantiated
        // even in this discarded branch, tripping the index range check for the W path (which never
        // passes these slots).
        constexpr uint32_t Wt = get_compile_time_arg_val((DIM == ckernel::ReduceDim::REDUCE_COL) ? 1 : 0);
        constexpr uint32_t W_logical = get_compile_time_arg_val((DIM == ckernel::ReduceDim::REDUCE_COL) ? 2 : 0);
        constexpr uint32_t wt_tiles_per_chunk =
            get_compile_time_arg_val((DIM == ckernel::ReduceDim::REDUCE_COL) ? 3 : 0);
        constexpr uint32_t tile_output = get_compile_time_arg_val((DIM == ckernel::ReduceDim::REDUCE_COL) ? 4 : 0);
        constexpr uint32_t num_h_slices = get_compile_time_arg_val((DIM == ckernel::ReduceDim::REDUCE_COL) ? 5 : 0);
        // Tile-rows in the output: div_up(num_h_slices, TILE_HEIGHT). 1 for every non-split reduce.
        constexpr uint32_t out_tile_rows = get_compile_time_arg_val((DIM == ckernel::ReduceDim::REDUCE_COL) ? 6 : 0);
        constexpr uint32_t face_w = tt::constants::TILE_WIDTH / 2;
        constexpr uint32_t face_h = tt::constants::TILE_HEIGHT / 2;
        constexpr uint32_t datums_per_face = face_h * face_w;
        const uint32_t num_output_tiles_local = rt_count;
        const uint32_t start_output_tile_id = rt_start;

        // Reduction-identity source for the TILE-output padding fill (see the header comment). The
        // reader builds this one-tile template in c_4 and never pops it, so this wait_front is a
        // one-time handshake rather than per-tile flow control. c_4 is at least one BF16 tile
        // (2048 B), which always covers our largest padding write (one full face: 256 datums =
        // 1024 B at FP32).
        DataflowBuffer dfb_clear(tt::CBIndex::c_4);
        if constexpr (tile_output != 0) {
            dfb_clear.wait_front(1);
        }

        uint32_t outputs_remaining = num_output_tiles_local;
        // Peeling off the W column leaves the flattened (nc, slice) index == nc * num_h_slices +
        // slice, matching the reader's decomposition of the same global tile id.
        uint32_t nc_slice = start_output_tile_id / Wt;
        uint32_t wt_in_nc = start_output_tile_id % Wt;

        while (outputs_remaining > 0) {
            // Pick the largest chunk that stays within one (nc, slice) group and within remaining work.
            uint32_t wt_in_chunk = wt_tiles_per_chunk;
            if (wt_in_chunk > Wt - wt_in_nc) {
                wt_in_chunk = Wt - wt_in_nc;
            }
            if (wt_in_chunk > outputs_remaining) {
                wt_in_chunk = outputs_remaining;
            }

            dfb_tile.wait_front(wt_in_chunk);

            for (uint32_t wt = 0; wt < wt_in_chunk; ++wt) {
                const uint32_t w_tile_col = wt_in_nc + wt;
                const uint32_t src_tile_offset = wt * tile_size_bytes;

                if constexpr (tile_output != 0) {
                    const uint32_t slice = nc_slice % num_h_slices;
                    const uint32_t nc = nc_slice / num_h_slices;
                    const uint32_t row_in_tile = slice % tt::constants::TILE_HEIGHT;
                    const uint32_t out_page_id =
                        (nc * out_tile_rows + slice / tt::constants::TILE_HEIGHT) * Wt + w_tile_col;

                    // Re-land tile-row 0 of the compute output at row_in_tile of the destination
                    // tile: one wide write per face along W. Padded columns past W_logical carry the
                    // reduction identity already (the reader pre-fills them), so unlike the ROW_MAJOR
                    // branch there is nothing to clamp — writing them keeps the output tile's own
                    // W padding well-formed.
                    for (uint32_t face_col = 0; face_col < tt::constants::TILE_WIDTH; face_col += face_w) {
                        noc.async_write(
                            dfb_tile,
                            dst_accessor,
                            face_w * datum_bytes,
                            {.offset_bytes = src_tile_offset + get_tilized_idx(0, face_col) * datum_bytes},
                            {.page_id = out_page_id,
                             .offset_bytes = get_tilized_idx(row_in_tile, face_col) * datum_bytes});
                    }

                    // Final slice: fill the rows past it in this tile-row-block with the identity.
                    // Rows [tail_row, TILE_HEIGHT) are contiguous within each face, so this is at
                    // most one write per face.
                    constexpr uint32_t tail_row = num_h_slices % tt::constants::TILE_HEIGHT;
                    if (tail_row != 0 && slice == num_h_slices - 1) {
                        for (uint32_t face = 0; face < 4; ++face) {
                            const uint32_t face_first_row = (face < 2) ? 0 : face_h;
                            const uint32_t start_row_in_face =
                                (tail_row > face_first_row) ? (tail_row - face_first_row) : 0;
                            if (start_row_in_face >= face_h) {
                                continue;
                            }
                            const uint32_t fill_datums = (face_h - start_row_in_face) * face_w;
                            noc.async_write(
                                dfb_clear,
                                dst_accessor,
                                fill_datums * datum_bytes,
                                {.offset_bytes = 0},
                                {.page_id = out_page_id,
                                 .offset_bytes = (face * datums_per_face + start_row_in_face * face_w) * datum_bytes});
                        }
                    }
                    continue;
                }

                const uint32_t w_base_col = w_tile_col * tt::constants::TILE_WIDTH;
                // Clamp the last W tile to W_logical so we don't write into padding.
                uint32_t valid_cols = tt::constants::TILE_WIDTH;
                if (w_base_col + valid_cols > W_logical) {
                    valid_cols = (w_base_col >= W_logical) ? 0 : (W_logical - w_base_col);
                }
                if (valid_cols == 0) {
                    continue;
                }

                // Emit at most 2 wide writes per tile (one per face along W), each up to face_w datums.
                for (uint32_t face_col = 0; face_col < valid_cols; face_col += face_w) {
                    const uint32_t face_valid = (valid_cols - face_col) < face_w ? (valid_cols - face_col) : face_w;
                    const uint32_t src_idx_in_tile = get_tilized_idx(0, face_col);
                    noc.async_write(
                        dfb_tile,
                        dst_accessor,
                        face_valid * datum_bytes,
                        {.offset_bytes = src_tile_offset + src_idx_in_tile * datum_bytes},
                        {.page_id = nc_slice, .offset_bytes = (w_base_col + face_col) * datum_bytes});
                }
            }

            noc.async_write_barrier();
            dfb_tile.pop_front(wt_in_chunk);

            wt_in_nc += wt_in_chunk;
            outputs_remaining -= wt_in_chunk;
            if (wt_in_nc == Wt) {
                wt_in_nc = 0;
                ++nc_slice;
            }
        }
    }
}

void kernel_main() { reduce_rm_writer<REDUCE_DIM>(); }
