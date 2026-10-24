// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: the sequential-over-token recurrence for one head. Carries the recurrent state
// S [K,V] on-core across T tokens. Derived from flash-linear-attention
// `naive_recurrent_gated_delta_rule`; the vLLM `fused_sigmoid_gating_delta_rule_update` is this
// same recurrence over the K+1 speculative tokens.
//
// Per token (q pre-scaled + L2-normed, k L2-normed, decay = exp(g_t), all done host-side):
//   sd     = S * decay            (decay the state BEFORE the read, matches FLA + chunk scan)
//   vread  = k . sd               ([1,V])
//   u      = beta * (v - vread)   ([1,V])
//   S_new  = sd + k^T (x) u       ([K,V] rank-1 update)
//   o      = q . S_new            (read from the POST-update state)
//
// Per-token tiles are [1, D] (token in row 0, rows 1..31 host-zero-padded), so the outer-product
// update k^T (x) u is a matmul with inner dim 1 whose 31 padding lanes are zero and vanish.
//
// Precision: the carried state never goes through srcA/srcB, which keep 19 bits of an fp32
// operand. The state CBs (S, s2, s3, sd), decay and beta are unpacked straight to the fp32 Dest
// (UnpackToDestFp32, set in the program factory); the decay is an SFPU fp32 multiply, and the
// rank-1 term k (x) u, still a matmul, is added with an SFPU fp32 add. A truncation there would
// compound every token: decay = 0.9999 (a 10k-token memory) would lose about 1.3e-3 per token
// instead of 1e-4. The two matmuls that read the state (k . sd and q . S_new) read copies
// (sd_mm, S_mm): a CB unpacked straight to Dest cannot feed srcA/srcB
// (tech_reports/op_kernel_dev/accuracy_tips, "Accumulation Precision"). Their operands are
// still truncated to 19 bits on each read, as on main. Through u, that error reaches the state
// once per token, but it does not compound: the carried state itself keeps all 32 bits.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5;
constexpr uint32_t cb_out = 6, cb_state = 7, cb_s2 = 8, cb_s3 = 9, cb_sd = 10;
constexpr uint32_t cb_vread = 11, cb_u = 12, cb_kcol = 13, cb_sd_mm = 14, cb_S_mm = 15;

// fp32 DST tiles available per tile_regs_acquire (half-sync).
constexpr uint32_t DST_TILES = 4;

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must already be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(Mt * Nt);
    matmul_init(a, b, tr ? 1 : 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                matmul_tiles(a, b, mi * Kt + ki, bi, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_o.push_back(Mt * Nt);
}

// S_new[Kt,Vt] = sd + kcol[Kt,1] (x) u[1,Vt], without the outer product round-tripping through
// L1. Per DST batch: the rank-1 matmul writes into DST[0..nn), sd is unpacked straight to the
// fp32 DST[nn..2nn), and an SFPU add sums them in fp32. The FPU does not accumulate onto the fp32
// sd: on a Wormhole N150 that add came out off by a power of two (-2^(e+4), e the exponent of
// both sd and the update) in the cases seen, all where the two nearly cancel. The result goes to
// o (the carried state), to o_mm (the copy the output matmul reads) and, when emit_state, to
// the state output.
template <uint32_t Kt, uint32_t Vt>
void rank1_update(uint32_t sd, uint32_t kcol, uint32_t u, uint32_t o, uint32_t o_mm, bool emit_state) {
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t per_batch = DST_TILES / 2;  // an update tile and its sd tile each
    CircularBuffer cb_o(o);
    CircularBuffer cb_o_mm(o_mm);
    CircularBuffer cb_st(cb_state);
    cb_o.reserve_back(kv);
    cb_o_mm.reserve_back(kv);
    if (emit_state) {
        cb_st.reserve_back(kv);
    }
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += per_batch) {
            const uint32_t nn = (Vt - n0 < per_batch) ? (Vt - n0) : per_batch;
            tile_regs_acquire();
            matmul_init(kcol, u, 0);
            for (uint32_t j = 0; j < nn; j++) {
                matmul_tiles(kcol, u, mi, n0 + j, j);
            }
            copy_init(sd);
            for (uint32_t j = 0; j < nn; j++) {
                copy_tile(sd, mi * Vt + n0 + j, nn + j);
            }
            add_binary_tile_init();
            for (uint32_t j = 0; j < nn; j++) {
                add_binary_tile(j, nn + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nn; j++) {
                pack_tile<true>(j, o, mi * Vt + n0 + j);
                pack_tile<true>(j, o_mm, mi * Vt + n0 + j);
                if (emit_state) {
                    pack_tile<true>(j, cb_state, mi * Vt + n0 + j);
                }
            }
            tile_regs_release();
        }
    }
    cb_o.push_back(kv);
    cb_o_mm.push_back(kv);
    if (emit_state) {
        cb_st.push_back(kv);
    }
}

// out = (A - B) * beta, n tiles, without spilling A - B to L1. Per DST batch: FPU sub into
// DST[0..nn), the beta tile into DST[nn], then an SFPU multiply of each by DST[nn]'s column 0
// in place. beta sits at [0,0] and only row 0 of A/B carries data (rows 1..31 are zero), so
// row 0 is scaled by beta and the zero padding rows stay zero.
void delta_rule_residual(uint32_t a, uint32_t b, uint32_t beta, uint32_t o, uint32_t n) {
    constexpr uint32_t per_batch = DST_TILES - 1;  // the last DST slot holds beta
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    for (uint32_t i0 = 0; i0 < n; i0 += per_batch) {
        const uint32_t nn = (n - i0 < per_batch) ? (n - i0) : per_batch;
        tile_regs_acquire();
        sub_init(a, b);
        for (uint32_t j = 0; j < nn; j++) {
            sub_tiles(a, b, i0 + j, i0 + j, j);
        }
        copy_init(beta);
        copy_tile(beta, 0, nn);
        sfpu_mul_bcast_col_init();
        for (uint32_t j = 0; j < nn; j++) {
            sfpu_mul_bcast_col(j, nn);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_o.push_back(n);
}

// out = S * decay, n tiles, in fp32: S and the decay tile are unpacked straight to Dest and the
// multiply is an SFPU column broadcast, the same construction as delta_rule_residual. The reader
// copies decay from [0,0] down column 0, so every row of S sees it. The result goes to o (carried
// on into the rank-1 update) and to o_mm (the copy the k . sd matmul reads).
void decay_state(uint32_t s, uint32_t decay, uint32_t o, uint32_t o_mm, uint32_t n) {
    constexpr uint32_t per_batch = DST_TILES - 1;  // the last DST slot holds decay
    CircularBuffer cb_o(o);
    CircularBuffer cb_o_mm(o_mm);
    cb_o.reserve_back(n);
    cb_o_mm.reserve_back(n);
    for (uint32_t i0 = 0; i0 < n; i0 += per_batch) {
        const uint32_t nn = (n - i0 < per_batch) ? (n - i0) : per_batch;
        tile_regs_acquire();
        copy_init(s);
        for (uint32_t j = 0; j < nn; j++) {
            copy_tile(s, i0 + j, j);
        }
        copy_init(decay);
        copy_tile(decay, 0, nn);
        sfpu_mul_bcast_col_init();
        for (uint32_t j = 0; j < nn; j++) {
            sfpu_mul_bcast_col(j, nn);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, i0 + j);
            pack_tile<true>(j, o_mm, i0 + j);
        }
        tile_regs_release();
    }
    cb_o.push_back(n);
    cb_o_mm.push_back(n);
}

// out[Kt,1] = transpose of in[1,Kt]: transpose each of the Kt tiles. (in must be available.)
void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t per_token = get_compile_time_arg_val(2);
    const uint32_t T = get_arg_val<uint32_t>(0);

    constexpr uint32_t kv = Kt * Vt;

    // Every CB is fp32 (see the program factory), so the unpack/pack formats configured here hold
    // for every op below; no per-op data-format reconfig is needed.
    compute_kernel_hw_startup(cb_q, cb_v, cb_out);

    CircularBuffer q(cb_q), k(cb_k), v(cb_v), decay(cb_decay), beta(cb_beta);
    CircularBuffer sd(cb_sd), sd_mm(cb_sd_mm), S_mm(cb_S_mm), vread(cb_vread), u(cb_u), kcol(cb_kcol);

    for (uint32_t t = 0; t < T; t++) {
        const uint32_t cur_S_id = (t == 0) ? cb_S : ((t & 1u) ? cb_s2 : cb_s3);
        const uint32_t nxt_S_id = (t & 1u) ? cb_s3 : cb_s2;
        CircularBuffer cur_S(cur_S_id), nxt_S(nxt_S_id);
        const bool last = (t == T - 1);

        // sd = cur_S * decay  (decay before read)
        decay.wait_front(1);
        cur_S.wait_front(kv);
        decay_state(cur_S_id, cb_decay, cb_sd, cb_sd_mm, kv);
        decay.pop_front(1);
        cur_S.pop_front(kv);
        sd.wait_front(kv);
        sd_mm.wait_front(kv);

        // vread = k . sd  ([1,V])
        k.wait_front(Kt);
        mm(cb_k, cb_sd_mm, cb_vread, 1, Kt, Vt, false);
        sd_mm.pop_front(kv);
        vread.wait_front(Vt);

        // u = beta * (v - vread)
        v.wait_front(Vt);
        beta.wait_front(1);
        delta_rule_residual(cb_v, cb_vread, cb_beta, cb_u, Vt);
        v.pop_front(Vt);
        vread.pop_front(Vt);
        beta.pop_front(1);
        u.wait_front(Vt);

        // kcol = transpose(k) ([K,1])
        transpose_block(cb_k, cb_kcol, Kt);
        k.pop_front(Kt);
        kcol.wait_front(Kt);

        // S_new = sd + kcol (x) u -> nxt_S and S_mm (and the state output, per token or last)
        rank1_update<Kt, Vt>(cb_sd, cb_kcol, cb_u, nxt_S_id, cb_S_mm, per_token || last);
        kcol.pop_front(Kt);
        u.pop_front(Vt);
        sd.pop_front(kv);
        nxt_S.wait_front(kv);
        S_mm.wait_front(kv);

        // o = q . S_new  (read from POST-update state)
        q.wait_front(Kt);
        mm(cb_q, cb_S_mm, cb_out, 1, Kt, Vt, false);
        q.pop_front(Kt);
        S_mm.pop_front(kv);

        // nxt_S is intentionally NOT popped: the next iteration reads it as cur_S.
    }
}
