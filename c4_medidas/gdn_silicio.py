"""GDN decode on silicon: ttnn.transformer.fused_recurrent_gated_delta_rule against FLA's recurrence in float64.

Run from the root of a tt-metal checkout (main or main + fix), with that checkout's python_env:
    python c4_medidas/gdn_silicio.py <label> [case ...]
Each case prints the relative state error ||S_dev - S_ref|| / ||S_ref|| at the end of the run, the same for a plain
fp32 recurrence (what a correct fp32 kernel can reach), and ||S_dev|| / ||S_ref||. Inputs follow the op's test
utilities (make_inputs: q, k L2-normalised, v ~ N(0,1), beta ~ U(0,1), state 0.05 N(0,1)); the cases only change the
gates g (log decay) and, for the decay-only cases, beta = 0 and an N(0,1) state.
"""

import sys
import time

import torch
import ttnn

from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import make_inputs

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
K = V = 128


def reference(q, k, v, beta, g, s0, dt):
    """FLA naive_recurrent_gated_delta_rule (the op's oracle) in dtype dt; returns the state after every token,
    [B, T, HV, K, V]. q, k are repeated to HV heads by the caller."""
    q, k, v, beta, g = (x.transpose(1, 2).to(dt) for x in (q, k, v, beta, g))
    q = q * K**-0.5
    h = s0.to(dt).clone()
    states = []
    for t in range(k.shape[2]):
        h = h * g[:, :, t].exp()[..., None, None]
        u = (v[:, :, t] - (h * k[:, :, t][..., None]).sum(-2)) * beta[:, :, t][..., None]
        h = h + k[:, :, t].unsqueeze(-1) * u.unsqueeze(-2)
        states.append(h.clone())
    return torch.stack(states, dim=1)


def to_dev(device, x):
    return ttnn.from_torch(x.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def run_op(device, q, k, v, g, beta, s0, tokens_per_call):
    """The op over the whole sequence, tokens_per_call tokens per call, the fp32 state handed back between calls
    (the model's decode loop is tokens_per_call = 1). Returns the final state."""
    T = q.shape[1]
    state = s0
    for t0 in range(0, T, tokens_per_call):
        sl = slice(t0, t0 + tokens_per_call)
        _, st = OP(
            to_dev(device, q[:, sl]),
            to_dev(device, k[:, sl]),
            to_dev(device, v[:, sl]),
            to_dev(device, g[:, sl]),
            to_dev(device, beta[:, sl]),
            initial_state=to_dev(device, state),
            output_final_state=True,
        )
        state = ttnn.to_torch(st).float()
    return state


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def case_inputs(name):
    """(description, B, T, H, HV, tokens_per_call, q, k, v, g, beta, s0)."""
    B = 1
    if name.startswith("decay"):
        T, H, HV = 2048, 1, 1
        x = make_inputs(B, T, H, HV, K, V, seed=0)
        g = torch.full((B, T, HV), -1e-4)
        beta = torch.zeros(B, T, HV)
        s0 = torch.randn(B, HV, K, V, generator=torch.Generator().manual_seed(1))
        tpc = T if name == "decay_1call" else 1
        return ("decay only (beta = 0), g = -1e-4 (decay 0.9999)", B, T, H, HV, tpc, x["q"], x["k"], x["v"], g, beta, s0)
    if name == "full_const4_1call":
        T, H, HV = 2048, 1, 1
        x = make_inputs(B, T, H, HV, K, V, seed=4)
        g = torch.full((B, T, HV), -1e-4)
        return ("full recurrence, g = -1e-4 (decay 0.9999)", B, T, H, HV, T, x["q"], x["k"], x["v"], g, x["beta"], x["initial_state"])
    if name in ("full_softplus_1call", "full_softplus_chained"):
        T = 2048 if name == "full_softplus_1call" else 256
        H = HV = 1
        x = make_inputs(B, T, H, HV, K, V, seed=5)
        gen = torch.Generator().manual_seed(6)
        g = -torch.nn.functional.softplus(torch.randn(B, T, HV, generator=gen) - 4.0)
        tpc = T if name == "full_softplus_1call" else 1
        return ("full recurrence, g = -softplus(N(0,1) - 4) (decays 0.5 to 0.999)", B, T, H, HV, tpc, x["q"], x["k"], x["v"], g, x["beta"], x["initial_state"])
    if name == "tt_inputs_chained64":
        T, H, HV = 64, 1, 1
        x = make_inputs(B, T, H, HV, K, V, seed=7)
        return ("make_inputs unchanged (g = -2 U(0,1), the inputs of #59526's chained check)", B, T, H, HV, 1, x["q"], x["k"], x["v"], x["g"], x["beta"], x["initial_state"])
    if name == "qwen27b_tp4_softplus":
        T, H, HV = 512, 4, 12
        x = make_inputs(B, T, H, HV, K, V, seed=8)
        gen = torch.Generator().manual_seed(9)
        g = -torch.nn.functional.softplus(torch.randn(B, T, HV, generator=gen) - 4.0)
        return ("Qwen3.6-27B TP4 heads (4 key, 12 value), g = -softplus(N(0,1) - 4)", B, T, H, HV, T, x["q"], x["k"], x["v"], g, x["beta"], x["initial_state"])
    raise SystemExit(f"unknown case {name}")


CASES = [
    "decay_1call",
    "decay_chained",
    "full_const4_1call",
    "full_softplus_1call",
    "full_softplus_chained",
    "tt_inputs_chained64",
    "qwen27b_tp4_softplus",
]


def main():
    label = sys.argv[1]
    cases = sys.argv[2:] or CASES
    device = ttnn.open_device(device_id=0)
    print(f"label={label} arch={device.arch()} cases={cases}", flush=True)
    try:
        for name in cases:
            desc, B, T, H, HV, tpc, q, k, v, g, beta, s0 = case_inputs(name)
            t0 = time.time()
            S = run_op(device, q, k, v, g, beta, s0, tpc)
            secs = time.time() - t0
            grp = HV // H
            qr, kr = q.repeat_interleave(grp, dim=2), k.repeat_interleave(grp, dim=2)
            S64 = reference(qr, kr, v, beta, g, s0, torch.float64)[:, -1]
            S32 = reference(qr, kr, v, beta, g, s0, torch.float32)[:, -1]
            calls = T // tpc
            print(f"CASE {name}: {desc}; T={T} as {calls} call(s) of {tpc}; heads H={H} HV={HV}; {secs:.1f} s")
            print(
                f"  state rel error vs float64: device {rel(S, S64):.3e} | plain fp32 {rel(S32, S64):.3e}"
                f" ; ||S_dev||/||S_ref|| {(S.double().norm() / S64.norm()).item():.6f}",
                flush=True,
            )
            if name.startswith("decay"):
                extra = -(torch.log(S.double().norm() / S64.norm()) / T).item()
                print(f"  extra decay per token beyond the gate: {extra:.3e}", flush=True)
    finally:
        ttnn.close_device(device)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
