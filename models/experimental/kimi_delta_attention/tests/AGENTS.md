# Kimi Delta Attention tests

Run every device test through `scripts/run_safe_pytest.sh`. A passing hardware run must end with `SAFE_PYTEST_RESULT: PASS`; a skip is not a pass.

## Policy

- Hermetic tests use deterministic synthetic weights. Tests never switch behavior based on local checkpoint availability.
- Real-weight tests require an explicit `KIMI_K3_CKPT` path, provided by `conftest.py`.
- Required performance acceptance is real Kimi-K3, B=1, T=5120 on SP1xTP8, SP2xTP4, and SP4xTP2.
- A perf case checks output, recurrent-state, and convolution-state PCC on its realtime-profiled forward before trace timing. Its device-repeat PCC is reproducibility evidence; `model/test_real_weights.py` supplies the independent Torch-reference PCC gate.
- Realtime capture is used once for correctness and program records, not inside timing loops. On SP1xTP8 it increased one warm forward from 10.803 ms to 112.108 ms because record collection is host-heavy.
- Use synchronized trace wall time for routine latency. Use Tracy only for overlap, unexplained regressions, or per-core/kernel attribution. MMRS requires Tracy because standalone program sums do not expose its critical-path overlap.

## Catalogue

| Test | Intent | Required? |
|---|---|---|
| `reference/test_affine.py` | CPU proof of affine composition, prefix equivalence, and FP32 stability | Yes while affine-prefix recurrence remains |
| `reference/test_config.py` | Model-config mapping and validation contracts | Yes |
| `reference/test_reference.py` | Independent Torch operation and layer identities | Yes |
| `checkpoint/test_checkpoint.py` | Indexed-shard loading, failure contracts, and padded K3 normalization | Yes; evolve with checkpoint API |
| `operations/test_chunk.py` | Direct `chunk_kda` PCC across layouts, shapes, fidelity, and grouped summaries | Yes |
| `operations/test_convolution.py` | Direct fused four-tap Q/K/V convolution PCC | Yes while fused convolution remains |
| `operations/test_halo.py` | 2D-mesh convolution-halo correctness on both TP axes | Yes while SP convolution remains |
| `operations/test_distributed_affine.py` | Distributed-prefix equivalence, cache reuse, and trace replay | Yes while SP affine prefix remains |
| `model/test_layer.py` | Small synthetic composed-layer PCC, validation, segmented state, and external state | Yes |
| `model/test_distributed_layer.py` | Synthetic SP layer PCC and segmented-prefill state continuity | Yes |
| `model/test_weights.py` | TP placement plus TP/2D composed-layer correctness | Yes; direct placement and integration contracts |
| `model/test_real_weights.py` | Independent Torch-reference PCC with pinned Kimi-K3 layer-1 weights; correctness forward emits realtime records | Yes; primary real-weight accuracy gate |
| `perf/test_layer_perf.py` | Real-K3 T=5120 profiled-result PCC, profiler overhead, records, and trace latency on the three target layouts | Yes; primary perf acceptance |
| `perf/test_fusion_ab.py` | Real-K3 fused/unfused MMRS, convolution, and gated-RMS PCC/perf experiments | Development evidence; keep while fusion choices are under review |
| `perf/test_operation_perf.py` | Exact-shape `chunk_kda` microprofile | Development probe; keep while public op is tuned |
| `perf/test_distributed_operation_perf.py` | Exact K3 SP4xTP2 distributed-prefix microprofile | Development probe; keep while SP prefix is tuned |
| `utils.py` | Deterministic synthetic config/weight builders | Support module, not a test |
| `conftest.py` | Explicit pinned Kimi-K3 checkpoint fixture | Support module, not a test |

## Commands

Hermetic and real-weight correctness, excluding perf:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 \
scripts/run_safe_pytest.sh --run-all \
  models/experimental/kimi_delta_attention/tests \
  --ignore=models/experimental/kimi_delta_attention/tests/perf -q -s
```

Independent real-weight PCC:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 \
scripts/run_safe_pytest.sh \
  models/experimental/kimi_delta_attention/tests/model/test_real_weights.py -q -s
```

Required performance matrix:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 PERF_REPS=10 \
scripts/run_safe_pytest.sh --run-all \
  models/experimental/kimi_delta_attention/tests/perf/test_layer_perf.py -q -s
```

Fusion A/B matrix:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 KDA_FUSION_AB_REPS=10 \
scripts/run_safe_pytest.sh --run-all \
  models/experimental/kimi_delta_attention/tests/perf/test_fusion_ab.py -q -s
```

Add `--profile` only for a specific Tracy investigation and use an exact node ID; `run_safe_pytest.sh --profile` does not preserve a spaced `-k` expression as one argument.
