# Metal 2.0 Port Report — `reduction/prod`

*Opened at the start of the port; entries captured as they occurred, polished at the end.*

## Outcome

`PORTED` — both factories (`ProdAllProgramFactory`, `ProdNcProgramFactory`) converted to
`MetalV2FactoryConcept`; confirmed against the baseline test set (see Verification below).

## Provenance

- **Recipe docs (this port):** `37f03926088 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `37f03926088 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Verification

Baseline captured pre-port and re-run post-port; results **identical** (no behavior change):

| Test | Pre-port | Post-port |
|---|---|---|
| `test_prod_all.py` | 18 passed | 18 passed |
| `test_prod_nc.py` | 32 passed | 32 passed |
| `test_reduction.py::test_prod` | 56 passed, 24 skipped | 56 passed, 24 skipped |

Build: `./build_metal.sh --build-tests` — clean (reduction unity TUs + `unit_tests_ttnn` relinked, no errors).
Confirmed baseline set with the invoker before relying on it (the two direct device-op tests +
`reduce/test_reduction.py::test_prod`). Kernels JIT-compiled at test runtime (the host build does not
compile them), so the green test run is the real kernel-side validation.

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for both factories — each `create_descriptor` replaced by a static
`create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`. No change
to the surrounding device-op class (`validate_*`, `compute_output_specs`, `create_output_tensors`).

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (neither op had one).
- Pybind entry points removed: **none** (`prod_nanobind.cpp` never bound a factory entry point).

### Open items
- **Relaxation candidates:** none applied. The `TensorParameter`s use strict `TensorSpec` matching
  (default). The op's kernels iterate tile-by-tile, so a `dynamic_tensor_shape` relaxation *might*
  be tolerable, but the legacy factory declared no `ArgConfig::Runtime*` shape flag (grep clean), so
  there is nothing to mirror — left strict per the "don't self-decide a relaxation" rule.
- No op-owned tensors, no `GlobalSemaphore`, no multi-program need — the base concept fits cleanly.

## Handoff points

- **Port capitulation:** none — both factories ported.
- **Boundary-rule assumption violations (`sem::`/`tensor::` crossing out-of-op):** none.
- **Kernel-lib / framework gaps:** none hit.
- **Removed pybind surface:** none — `prod_nanobind.cpp` never bound a factory entry point.

## Successes

- **[Shared-dataflow-kernel fork Caution](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)**
  fired exactly as written. The brief flagged the two `eltwise/unary` donor kernels as broadly-shared
  (~29 / ~12 co-borrowers); the fork-with-`_metal2`-suffix path let prod adopt Metal 2.0 bindings
  without touching the shared originals. `eltwise/unary`'s `GLOB_RECURSE device/kernels/*.cpp`
  auto-picked up the forks — no CMake edit needed.
- **[Hardware-config discipline](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration)**
  caught two silent-regression traps: `dst_full_sync_en=false` → `double_buffer_dest=true` (the
  inversion) and the FP32-only required `unpack_modes` entry (`prod_all_program_factory.cpp` /
  `prod_nc_program_factory.cpp`). Both would have compiled and passed tests while silently shifting
  precision/perf. The before/after value diff (Style B, direct `ComputeGen1Config`) is what surfaced them.
- **Named-arg model naturally shed the four dead legacy args** (audit "Misc anomalies") with no
  cleanup action — binding exactly the kernel's reads leaves them out by construction.

## Friction

- **Gap — the recommended reference port is API-stale.** The accumulation reference on
  `akertesz/porting-experiment-accumulation-jun10` (the recipe's "first worked end-to-end" shape
  reference) predates the current headers in several load-bearing ways: it uses
  `create_program_spec` (not `create_program_artifacts`), `#include "ttnn/metal2_artifacts.hpp"`
  (now `metal_v2_artifacts.hpp`), `DataMovementHardwareConfig{.role = RoleHint::READER}` (the
  `.role`/`RoleHint` field no longer exists — the current path is the `ttnn::create_reader/writer_datamovement_config(arch)`
  helpers), and the old flat `ComputeHardwareConfig{.math_fidelity, .fp32_dest_acc_en, .dst_full_sync_en,
  .math_approx_mode, .unpack_to_dest_mode}` struct (now `ComputeGen1Config{.fpu_math_fidelity,
  .sfpu_precision_mode, .enable_32_bit_dest, .double_buffer_dest, .unpack_modes}` with the
  bool→`Precision` and `dst_full_sync_en`→`double_buffer_dest` transforms). Its `ProgramRunArgs`
  RTA shape is also node-first-list, not the current name-first `Table`. The recipe already warns
  "don't lean on already-ported ops as templates"; this port confirms that warning is load-bearing
  even for the *designated* reference — I followed the current headers + migration guide over the
  reference wherever they disagreed. **Suggestion:** either refresh the accumulation reference to
  the current API, or have the recipe name a more recently-landed reference.
- **Confusion (minor) — `ProgramRunArgs` "must specify KernelRunArgs for ALL kernels" vs the recipe's
  "omit the entry if no RTAs."** `program_run_args.hpp` says a `KernelRunArgs` must exist for every
  kernel; the recipe says the entry "may be omitted entirely" when a kernel has no RTAs (prod_all
  compute). Resolved by including a values-less `KernelRunArgs{.kernel = COMPUTE}` — satisfies both
  readings (an entry exists; its empty schema means nothing to set). Tests pass, so an empty entry
  is accepted. Worth a one-line reconciliation in the recipe.
- **Confusion (minor) — DFB page-size getter units.** The CB→DFB whitelist maps
  `get_local_cb_interface(...).fifo_page_size` → §B `get_entry_size()`, and separately notes "TRISC
  size getters return bytes." For the DM writer/reader forks, `get_entry_size()` is the right
  bytes-valued replacement for the legacy `fifo_page_size` fed to `noc.async_write/read`; tests
  confirm. A one-line "DM `get_entry_size()` == legacy `fifo_page_size` (bytes)" note would remove
  the moment of doubt.

## Open items for downstream

### Cross-op kernel touches (forks)
- **`eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`**
  - Path taken: **fork** → `writer_unary_interleaved_start_id_metal2.cpp` (alongside original).
  - Consumed by: prod_all + prod_nc writer KernelSpecs.
  - Remaining unmigrated consumers of the **legacy** copy: ~29 op families (typecast, bcast,
    concat, copy, permute, reshape_on_device, slice, tilize, tilize_with_val_padding, transpose,
    embedding, …). Sunset the fork when the last co-borrower ports.
- **`eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`**
  - Path taken: **fork** → `reader_unary_interleaved_start_id_metal2.cpp` (alongside original).
  - Consumed by: prod_all reader KernelSpec.
  - Remaining unmigrated consumers of the **legacy** copy: ~12 op families. Sunset as above.

### Pre-existing anomalies left for the ops team (NOT acted on)
- prod_nc dead reader RTA `dim` (`prod_nc_program_factory.cpp` legacy RTA 7), dead writer RTA
  `is_dram` (legacy RTA 3), dead compute CTA `num_cols_per_core_group_*`; prod_all dead compute
  CTA `per_core_block_size` (CTA[1]). None are read by any kernel; they have no representation in
  the named-arg model, so the port neither carries nor "cleans up" the legacy factory lines —
  behavior is unchanged. The anomalies themselves remain for the ops team.
- Output CB at `c_3` (not the `c_16+` output convention). Cosmetic; invisible under DFB naming.
