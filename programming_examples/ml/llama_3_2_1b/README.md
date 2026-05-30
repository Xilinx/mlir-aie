# Llama 3.2 1B (INT8) on AI Engine (IRON) — WIP

End-to-end INT8 Llama 3.2 1B decode on the AMD Strix Point NPU2
(AIE2P), implemented in the high-level IRON Python API and structured
to mirror `programming_examples/ml/yolo26n/`. All 16 decoder layers
will run via a single reused worker set with per-layer INT8 weights
streamed from DRAM, plus on-device sampling. Validation target: bit-exact
against the `cautious-eureka` numpy reference oracle.

The dataflow design + per-channel INT8 quant recipe were developed and
simulation-validated in the [cautious-eureka](../../../../) repo
(`npu2/llama_layer_ref.py` is the oracle, `npu2/aie2_llama_iron.py` is
the design source-of-truth). This example is the hardware bring-up.

## Status

| Phase | State |
|---|---|
| 0. Scaffolding (`llama_spec.py`, `placement.py`, env)         | done |
| 1. **Dataflow stubs** — all topology validated on hardware     | **done** |
| 2. Real kernel: `gemm_int8_srs` (scalar then vec)              | pending |
| 3. Real glue: `rmsnorm_residual`, `rope`, `silu_mul`           | pending |
| 4. Real FlowKV: `flowkv_qk`, `flowkv_sv`                       | pending |
| 5. Single decoder layer bit-exact vs oracle                    | pending |
| 6. 16-layer decode chain via `build_decode_design`             | pending |
| 7. `sample` kernel + end-to-end generation                     | pending |
| 8. Prefill overlay                                             | deferred (follow-up PR) |

## Dataflow stubs (Phase 1) — done

Every placement pattern the real design needs has been validated
**bit-exact on hardware** by a small stub kernel that runs in the right
tile(s) and the right ObjectFifo topology. Each stub uses simple
math (passthrough / xor / int8 add) so the host can compute the
expected output explicitly. Building real kernels from here on is a
focused kernel-level task — the dataflow risk is retired.

| Target | Validates | CTs |
|---|---|---|
| `make run_gemm_pt`      | 1-CT packed weight-stream payload (act + weights blob) | 1 |
| `make run_gemm_pt_col`  | 2-CT col: activation broadcast + memtile split + memtile join | 2 |
| `make run_gemm_pt_proj` | Full decode-overlay projection: 16 CTs (rows 2–3 × cols 0–7), consolidated runtime args + TAP-based per-col dispatch | 16 |
| `make run_flowkv_pair`  | CT0→CT1 vertical neighbor stream (attn pair0: col 0, rows 4↔5) | 2 |
| `make run_glue`         | 2-input fanin convergence at one CT (rmsnorm+residual / silu+mul shape) | 1 |
| `make run_layer_pt`     | **Full 16-worker single-decoder-layer integration** at real dimensions (D=2048, QD=2048, KVD=512, HD=8192). 3-way h1 broadcast, residual hold on x_in + x1, FlowKV pair, 2-input adds — all together. | 16 |

Each stub kernel pairs with a Python test that bit-exact-compares the
NPU output against the traced-by-hand expected result. The
`layer_pt` test, for instance, expects `out == 6 * x_in (mod 256)`
(see [`test_layer_pt.py`](test_layer_pt.py) for the trace).

## Quickstart

```bash
# 1. Env (one-time)
source /opt/xilinx/xrt/setup.sh
source ../../../utils/quick_setup.sh       # creates ironenv via wheels
pip install --upgrade cmake                # ironenv cmake (needs >= 3.30)
pip install -r ../../../python/requirements_ml.txt

# 2. Point at weights (only needed for Phase 5+ layer-level tests
#    against the cautious-eureka oracle; the stubs need nothing here)
export LLAMA_3_2_1B_WEIGHTS=/path/to/llama-3.2-1b   # dir with model.safetensors

# 3. List targets
make help    # or just open the Makefile
```

## Files

**Design / placement (ports of cautious-eureka)**
- `llama_spec.py` — algorithm + shapes (one decoder layer parameterized by M).
  Self-validates param counts on import.
- `placement.py` — physical tile placement for both decode and prefill
  overlays (`DECODE_PLACEMENT`, `PREFILL_PLACEMENT`). The ONLY place tile
  coordinates appear. `render_diagram` prints the 28/32-tile decode layout.

**Dataflow stubs (Phase 1)**
- `aie2_gemm_int8_srs.py` / `test_gemm_int8_srs_pt.py` — 1-CT frame
- `aie2_gemm_int8_srs_col.py` / `test_gemm_int8_srs_pt_col.py` — 2-CT col
- `aie2_gemm_int8_srs_proj.py` / `test_gemm_int8_srs_pt_proj.py` — 16-CT projection
- `aie2_flowkv_pair.py` / `test_flowkv_pair_pt.py` — qk→sv neighbor stream
- `aie2_glue.py` / `test_glue_pt.py` — 2-input fanin
- `aie2_layer.py` / `test_layer_pt.py` — full single-layer integration

**Stub kernels** (`kernels/`)
- `llama_gemm_int8_srs_pt.cc` — act passthrough, ignores weight blob
- `llama_flowkv_pt.cc` — bitwise-inverts input (qk and sv share semantics)
- `llama_glue_pt.cc` — xor of two inputs
- `llama_layer_pt.cc` — shape-specific copy / tile / add / first-of-two-inputs
  symbols for every call site in the full-layer stub

## Hardware constraints learned during bring-up

These are real and apply to every new IRON design on AIE2P. Apply
preemptively when sketching new designs to skip a build cycle:

1. **Compute tile DMA budget: 2 in + 2 out per CT.** Hardware constraint.
   Surfaces as `tile requires N input/M output DMA channels, but only
   2 input/2 output available`. → Pack per-call constants (weights +
   bias + scale) into one `ObjectFifo` as a packed payload. This is the
   pattern cautious-eureka's `StaticWeightStream` already uses.

2. **`Worker(tile=Tile(col, row))` is required for multi-column designs.**
   Without it the auto-placer piles fifos onto col 0 and hits
   `'aie.tile' op number of output DMA channel exceeded!` on the shim.
   Per-col layouts (like the 8-col decode projection) must pin each
   worker to its column.

3. **`DefaultNPURuntime.run_test` segfaults past ~5 XRT kernel args.**
   Keep runtime args to ~3 big consolidated buffers (e.g. one act, one
   packed weights, one packed outputs); dispatch per-tile via
   `TensorAccessPattern(tensor_dims=[total], offset=c*slice,
   sizes=[1,1,1,slice], strides=[0,0,0,1])` in `rt.fill` / `rt.drain`.
   Matches the `whole_array_iron` consolidated-args style.

4. **One `Kernel` object per unique C symbol — not per call site.**
   `Kernel(symbol, ...)` emits a fresh MLIR func decl every time;
   constructing two `Kernel` objects for the same symbol (even with
   different arg types) collides at `Program.resolve_program()`. Share
   one `Kernel` across workers that have the same signature; use a
   different C symbol per shape when signatures differ.

## What still needs work

Phase 2 onwards. The 7 `.cc` kernels listed in
`cautious-eureka/npu2/aie2_llama_iron.py::KERNELS` — most have a
starting point in `../../../aie_kernels/aie2p/` (`mm.cc` for int8
matmul, `rms_norm.cc`, `rope.cc`, `softmax.cc`, `swiglu.cc`). FlowKV
sv and `sample` are write-from-scratch. yolo26n's 10-item kernel
optimization playbook applies directly once kernels are bit-exact.
