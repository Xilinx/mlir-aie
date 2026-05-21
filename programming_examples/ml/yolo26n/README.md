# YOLO26n-cls (INT8) on AI Engine (IRON)

End-to-end YOLO26n classification inference on the AMD Strix Point NPU2
(8-column × 6-row array: row 0 shim DMA, row 1 memtile, rows 2–5 the
4 compute-tile rows = 32 compute tiles total), implemented in the
high-level IRON Python API. All 11 blocks (`m0..m10`) run depth-first
across the array, keeping intermediate activations on-chip, and emit
classification probabilities **bit-exactly matching the Quark XINT8 ONNX
reference**.

The full int8 quantization recipe lives in [`notebooks/`](notebooks): a
FP32 YOLO26n-cls PyTorch checkpoint → FP32 ONNX → AMD Quark XINT8 PTQ →
the deployment ONNX shipped in [`models/`](models).

## Quickstart

```bash
# 1. Activate the standard mlir-aie environment.
source /opt/xilinx/xrt/setup.sh
source ../../../ironenv/bin/activate
source ../../../utils/env_setup.sh ../../../install

# 2. List available targets.
make help

# 3. Extract per-block weight / bias / SiLU LUT bins from the shipped XINT8 ONNX.
make data

# 4. Bit-exact-check a single block on the NPU.
make BLOCK=m8 run_ort

# 5. Bit-exact-check the full chain on the NPU.
make run_chain                          # N=1, single-sample latency mode
CHAIN_N_SAMPLES=15 make run_chain       # batched throughput mode

# 6. Time it.
make BLOCK=m8 time                      # per-block (median ms + fps)
make time_chain                         # chain N=1
CHAIN_N_SAMPLES=15 make time_chain      # chain N=15
```

`make data` is idempotent; the resulting `data/` tree is ~2 MB of
per-layer bins (gitignored).

## Performance (Strix HW, bit-exact vs ORT)

Latest measurements at the current commit.

**Chain end-to-end (m0..m10):**

| Mode | Wall per dispatch | Per sample | Throughput |
|---|---:|---:|---:|
| **N=1** (single-sample latency) | 397.4 ms | 397.4 ms | **2.52 fps** |
| **N=15** (batched) | 4 776.8 ms | 318.5 ms | **3.14 fps** |

**Per-block (m0..m8) standalone wall time on NPU**, median of n=20:

| Block | Topology | Median (ms) | fps |
|---:|---|---:|---:|
| m0  | conv_stride stem            | 55.4 | 18.1 |
| m1  | conv_stride                  | 45.1 | 22.2 |
| m2  | c3k2_small                   | 26.4 | 37.9 |
| m3  | conv_stride (chunked)        | 76.4 | 13.1 |
| m4  | c3k2_small                   | 22.0 | 45.5 |
| m5  | conv_stride (chunked)        | 73.6 | 13.6 |
| m6  | c3k2_heavy                   | 15.1 | 66.0 |
| m7  | conv_stride (chunked)        | 36.5 | 27.4 |
| m8  | c3k2_heavy (2-tile megakernel) | 28.2 | 35.4 |

m9 (PSA) and m10 (head) aren't supported by `test_block_ort.py` standalone
(PSA topology has no single intermediate tensor to compare; m10's flat
softmax output doesn't fit the per-block tester's reshape). Both are
exercised end-to-end via `run_chain`.

Per-block sum ≈ 379 ms; N=1 chain ≈ 397 ms → ~18 ms of fill/drain. m3 and
m5 (the chunked stride-2 convs) are the largest single contributors and
the most attractive next targets if you want to push fps higher.

## Make targets

Run `make help` for the canonical list. Highlights:

| Target | What it does |
|---|---|
| `make data` | ONNX → `data/{weights,bias,silu_lut}.bin` + manifest |
| `make BLOCK=mN` | Build per-block xclbin + insts |
| `make BLOCK=mN run_ort` | Build + run + bit-exact compare vs ORT |
| `make BLOCK=mN time` | Build + time (mean/median/min/max ms, fps) |
| `make chain` | Build chain xclbin + insts (defaults to N=1) |
| `make run_chain` | Build + run + bit-exact compare vs ORT (every sample) |
| `make time_chain` | Build + time (latency or per-sample throughput) |
| `make trace` | Build with HW packet trace, run 1 iter, parse summary |
| `make clean` | Wipe `build/` entirely |
| `make clean_chain` | Wipe only the chain xclbin/MLIR (cheap) |

**Environment knobs:**

- `BLOCK=mN` — selects the per-block target (default `m0`).
- `CHAIN_N_SAMPLES=N` — number of samples per chain dispatch. The Makefile
  tracks this in a stamp file and forces the chain MLIR to rebuild
  whenever the value changes, so `run_chain` and `time_chain` always
  match the xclbin's compiled-in `N`.
- `TIME_ARGS="--n-warmup 5 --n-iters 100"` — passed through to the timing
  harnesses.
- `TRACE_SIZE_PER_WORKER=N`, `TRACE_EVENTS=A,B,C` — bake AIE2P packet
  tracing into per-block MLIR. The `make trace` target sets these and
  handles the build/run/parse round-trip via `scripts/trace_m8.sh`.

**Note on incremental builds:** the Makefile correctly rebuilds on `.cc`
or `.py` source changes (mtime-tracked). If you edit the `Makefile` itself
(e.g., changing `PEANOWRAP2P_FLAGS`) or change a `TRACE_*` env var,
existing `.o` / `.mlir` files won't pick up the change automatically — run
`make clean` first. `CHAIN_N_SAMPLES` is handled automatically.

## Tracing

`scripts/trace_m8.sh` (via `make trace`) builds the m8 megakernel with
HW packet trace ops, runs one iteration, and parses the trace. Output:

- `build/trace_m8.txt` — raw trace packets (hex)
- `build/trace_m8.json` — parsed events
- stdout — per-tile cycle summary from `python/utils/trace/get_trace_summary.py`

Pick the events you want via `TRACE_EVENTS`:

```bash
# Default (8-event packed set: INSTR_EVENT_0/1 + stalls + vector + ...)
make trace

# Stall attribution — what's the core blocked on?
TRACE_EVENTS=INSTR_EVENT_0,INSTR_EVENT_1,LOCK_STALL,STREAM_STALL,MEMORY_STALL \
    make trace

# Vector activity — what fraction of cycles are vmac?
TRACE_EVENTS=INSTR_EVENT_0,INSTR_EVENT_1,ACTIVE,INSTR_VECTOR \
    make trace

# Larger trace BO (default 32 KB / worker)
TRACE_SIZE_PER_WORKER=65536 make trace
```

Event names come from `aie.utils.trace.events.CoreEventAIE2P`. The
chain-side equivalent uses `aie2_yolo_iron_partial.py`'s `TRACE_BLOCKS`
env to restrict tracing to specific blocks (the AIE2 trace packet-ID
space caps at 31 simultaneously-traced workers, so the full 26-tile chain
needs front-half/back-half splits).

## Design overview

The 11 blocks map to these kernel families:

| Block | Topology | Kernel family | Tiles |
|---|---|---|---:|
| m0 | conv_stride (stem) | `yolo_m0_conv2dk3_silu_bias` (OIYX raw, in_c=3 padded to 8) | 1 |
| m1 | conv_stride | `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8` (shared, runtime shapes) | 1 |
| m3, m5, m7 | conv_stride | `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked` (per-block, weight-streamed) | 1 each |
| m2, m4 | c3k2_small | `yolo_c3k2_small_{cv1_split, m0_cv1, m0_cv2_skip, cv2_concat3}` | 3 each |
| m6 | c3k2_heavy | `yolo_c3k2_heavy_{m_0_split, inner_pair_cv1, inner_pair_cv2_skip, cv3_concat2}` + `c3k2_small_{cv1_split, cv2_concat3}` | 5 |
| **m8** | **c3k2_heavy 2-tile megakernel** | `yolo_m8_front_cv1_split_fused` + `yolo_m8_back_cv3_cv2_fused` + `yolo_c3k2_heavy_inner_pair_{cv1,cv2_skip}_streamed` | **2** |
| m9 | PSA (attention + FFN) | 15 kernels: `yolo_m9_{cv1_split, qkv, qkv_pack, qk_pack, qk_row, attn_scale, softmax_row, v_pack, sv_row, sv_row_acc, pe_add_row, proj_skip_row, ffn_0_silu_row, ffn_1_skip_row, cv2_concat2_streamed}` | 7 |
| m10 | head | `yolo_m10_{conv2dk1_silu_xy_pool, linear_gemm, softmax}` (fused onto 1 tile) | 1 |

Total: **26 of 32** compute tiles used.

### m8 — 2-tile megakernel

The most compute-dense block. Implementation lives in
[`scripts/m8_megakernel_2tile.py`](scripts/m8_megakernel_2tile.py).

Two compute tiles, each running a single Worker that fuses three c3k2_heavy
sub-operations into one C kernel call per chunk:

- **Tile A (5,3)** — `k_m8_front` (cv1 + m_0_split) + `k_pair_cv1` + `k_pair_cv2` for pair0
- **Tile B (5,4)** — `k_pair_cv1` + `k_pair_cv2` for pair1 + `k_m8_back` (cv3 + cv2)

Cross-tile data lives in shared L1 ObjectFifos (no DMA hop). Weights for
the big convs (cv1, cv2, pair0, pair1) are streamed from memtiles via
`StaticWeightStream`; the small m_0_split and cv3 weights are static on
the compute tiles. Tile A reads ws_pair1's recv buffer via west-neighbor
shared L1 from (4,4).

### m9 — staged PSA

[`scripts/m9_stage.py`](scripts/m9_stage.py) implements the PSA block
across 5 compute tiles + 2 memtiles in 10 stages (each stage adds one
attention sub-kernel). The chain uses stage 10 = full m9. Lower stages
exist for chain-bisect debugging via `M9_CHAIN_STAGE=N`.

### Kernel symbol mangling

Several blocks share a kernel template (e.g. c3k2_small for m2/m4,
c3k2_heavy for m6, chunked conv_stride for m3/m5/m7) but with different
runtime shapes / weight sizes. To let them coexist in one MLIR module the
Makefile compiles each block's variant with `-DKERNEL_SUFFIX=_mN`, which
suffixes every exported symbol — so `yolo_c3k2_small_cv1_split_vec.cc`
produces `..._m2.o` and `..._m4.o` with non-colliding `func.func` names.
The IRON builder passes the matching suffixed name into `Kernel(...)`.

### Bit-exactness

The integration test of record is `test_chain_ort.py`: same seeded RGB
input fed to both (a) the NPU chain xclbin and (b) ONNX Runtime on the
XINT8 ONNX, with hard element-wise INT8 equality on the final softmax
probs. For `CHAIN_N_SAMPLES=N>1`, every output slice is compared
independently — all N samples must match the single reference.

The conv_stride and c3k2 blocks (m0..m8) each pass `test_block_ort.py
--block mN` bit-exact standalone (see the `run_ort` target). m9 (PSA)
and m10 (head) aren't supported standalone but are covered by the
end-to-end chain test.

[`run_strix_makefile.lit`](run_strix_makefile.lit) exercises both paths
on NPU2 hardware.

## The int8 recipe

Three stages: train float, quantize to int8, materialize per-block
kernel inputs.

### 1. Train (upstream)

YOLO26n-cls is a 1.5M-parameter classifier from Ultralytics. The FP32
checkpoint shipped here (`models/phase1_25k_acc0.9424.pt`, 90.0% test /
94.2% val accuracy on a person/no-person dataset) is the source. See
[`notebooks/yoloexploration.ipynb`](notebooks/yoloexploration.ipynb) for
loading, inference, and COCO validation.

### 2. Quantize (Quark XINT8 PTQ)

[`notebooks/quark_quantization.ipynb`](notebooks/quark_quantization.ipynb)
walks through:

1. Export FP32 PyTorch → FP32 ONNX at fixed `(batch=4, 3, 512, 512)` (Ryzen
   AI requires static shapes).
2. Sanity-check FP32 ONNX vs the PyTorch baseline — catches preprocessing
   bugs before quantization muddies the signal.
3. Build an `onnxruntime.quantization.CalibrationDataReader` from 500
   in-distribution images.
4. Run Quark static PTQ with the **XINT8** spec for both activations and
   weights, **AdaRound** for weight rounding refinement, and **exclude the
   final Softmax** from quantization (empirically: quantizing the final
   Softmax causes a sign-flip that drops accuracy to ~44%; keeping it
   FP32 recovers 89.5%).
5. Evaluate the XINT8 ONNX on the held-out test set, compare to FP32.

Output: `phase1_25k_xint8_acc0.8968.onnx` (1.7 MB, 89.68% test
accuracy, ~0.3 pp from FP32 — within typical static-PTQ overhead). Shipped
under [`models/`](models) and consumed by `gen_yolo_data.py` /
`gen_yolo_silu_luts.py`.

### 3. Extract per-block kernel inputs

[`gen_yolo_data.py`](gen_yolo_data.py) walks the Quark XINT8 ONNX and, for
every Conv / MatMul / Gemm node, emits:

- INT8 weight bins, reordered from ONNX `OIYX` to the `OIYXI8O8` layout
  the AIE2P kernels expect (m0 keeps `OIYX` because `in_c=3` is not
  8-aligned and gets host-side padded to 8 instead).
- INT32 bias bins, promoted from INT8 by `bias_pre_shift` so the runtime
  kernel can use them as the accumulator init when `weight_index == 0`
  (folds the bias-add into the conv's first MAC — no separate bias stage).
- A per-op `right_shift = log2(scale_in) + log2(scale_w) − log2(scale_out)`
  for the int32→int8 banker-rounded requant epilogue.
- `data/manifest.json` indexing it all by layer name.

[`gen_yolo_silu_luts.py`](gen_yolo_silu_luts.py) handles the SiLU
non-linearity: the Quark XINT8 graph encodes SiLU as
`Conv → QL → DQ → HardSigmoid → Mul(const) → QL → DQ → Mul(linear) → QL`.
Per Conv we extract the three QuantizeLinear scales + HardSigmoid params
+ constant multiplier, then sample the closed-form HardSiLU over all 256
INT8 inputs to build a 256-byte LUT (`data/model.N/.../silu_lut.bin`).
At runtime each kernel's epilogue is one LUT lookup:
`out_i8 = silu_lut[clamp(banker_srs(acc, rs)) + 128]`.

## Repo layout

```
yolo26n/
├── README.md                          # this file
├── Makefile                           # build + run + time + trace targets (make help)
├── aie2_yolo_iron_partial.py          # chain orchestrator (CHAIN_BLOCKS / CHAIN_N_SAMPLES env)
├── aie2_yolo_per_block.py             # per-block builders (the meat)
├── yolo_spec.py                       # declarative network spec (shapes, ops)
├── placement.py                       # PLACEMENT[block_name] → tile coords + design rules
├── tile_layout.py                     # ASCII tile-map + per-block sizing
├── lowlevel_dma.py                    # StaticWeightStream helper for chunked weight DMA
├── kernels/                           # 37 AIE2P .cc/.h kernel files (vectorized _vec.cc throughout)
├── scripts/
│   ├── m8_megakernel_2tile.py         # m8 2-tile megakernel (the only m8 path)
│   ├── m9_stage.py                    # m9 PSA staged builder (stages 1..10)
│   ├── time_block.py                  # per-block timing harness
│   ├── time_chain.py                  # chain timing harness
│   └── trace_m8.sh                    # m8 HW packet-trace driver
├── gen_yolo_data.py                   # XINT8 ONNX → data/manifest.json + per-Conv bins
├── gen_yolo_silu_luts.py              # XINT8 ONNX → per-Conv 256-byte SiLU LUTs
├── data/                              # gitignored; regenerated by `make data`
├── test_block_ort.py                  # per-block (m0..m8) NPU vs ORT bit-exact host driver
├── test_chain_ort.py                  # full chain NPU vs ORT (N=1 or N>1) bit-exact host driver
├── notebooks/
│   ├── quark_quantization.ipynb       # full int8 recipe (FP32 → XINT8)
│   └── yoloexploration.ipynb          # upstream YOLO26n context + COCO val
├── models/
│   ├── phase1_25k_acc0.9424.pt        # FP32 PyTorch source (3.1 MB)
│   └── phase1_25k_xint8_acc0.8968.onnx # Quark XINT8 deployment artifact (1.7 MB)
└── run_strix_makefile.lit             # lit test (m0 per-block + full chain end-to-end)
```

## Contributing & debugging

### Where to start if you're new

1. **Run something on HW first.** `make BLOCK=m6 run_ort` is the simplest
   end-to-end (small block, vectorized, bit-exact passes in under a minute).
   If that works, your environment is set up.
2. **Read `placement.py`** — the header lists the per-block tile budget,
   and the **"DESIGN RULES"** comment block (lines 33–~150) captures the
   hard-won lessons (deadlock patterns, via_DMA gotchas, kernel-arg
   stride bugs, etc.). Worth reading once cover-to-cover.
3. **Run `python3 tile_layout.py`** for an ASCII map of the 32 compute
   tiles + which block sits where.
4. **Pick a per-block builder in `aie2_yolo_per_block.py`** and read its
   preceding block-comment header — those describe each block's
   architecture, fan-out, and kernel inventory.

### Adding an optimization to an existing block

1. **Baseline.** `make BLOCK=mN time TIME_ARGS="--n-warmup 5 --n-iters 100"`
   and write down the median ms.
2. **Bit-exact gate.** `make BLOCK=mN run_ort` must pass after every
   step (m9/m10 only via `make run_chain`).
3. **Trace if needed.** `make trace` builds the m8 megakernel with HW
   packet trace ops and prints the per-tile cycle summary. For other
   blocks, copy `scripts/trace_m8.sh` and adapt; the per-block trace
   plumbing in `aie2_yolo_per_block.py` already honors
   `TRACE_SIZE_PER_WORKER` and `TRACE_EVENTS`.
4. **Iterate.** `make clean` is rarely needed for `.cc`/`.py` edits;
   Make's mtime tracking is correct for those. Run `make clean` if you
   edit the Makefile itself or change a `TRACE_*` env var.

### Adding a new kernel

1. Drop a `_vec.cc` under `kernels/` (vectorized AIE2P API style — see
   `yolo_m8_back_cv3_cv2_fused_vec.cc` for a non-trivial example with
   chunked-OC, fused requant, SiLU LUT, and skip-add).
2. Wire a build rule in the **kernel build rules** section of the
   Makefile (search for "c3k2_heavy streamed inner-pair kernels" for the
   `$(foreach blk, ... $(eval ...))` pattern). Add the `.o` to
   `KERNEL_OBJS`.
3. Declare it as a `Kernel(...)` in the relevant builder
   (`aie2_yolo_per_block.py` or one of `scripts/*.py`) with arg types
   matching the C signature exactly.
4. Call it from a `Worker` body. Pass `aie::int8_t*` args as IRON
   `B._i8(shape)` types; scalar ints as `np.int32` (or `np.uint32` if
   the kernel wants to divide / shift on them — see below).

### Common debugging patterns

- **Build hangs, no error**: check the build log for **L1 overflow**
  warnings (e.g. `Failed to allocate buffer "X" with size: NNNN bytes`,
  `Bank-aware allocation failed`). The xclbin may still build but you
  used too much L1; reduce per-tile buffer depth or stream weights from
  the memtile via `StaticWeightStream`.
- **Bit-exact failure with a partial set of mismatches**: usually a
  kernel signature mismatch (IRON Kernel arg types vs C function args),
  or a stride mistake in a memref index. The kernel signature must match
  positionally — IRON doesn't type-check across the C boundary.
- **`make` builds something stale**: if you've edited the Makefile or
  changed a `TRACE_*` env var, run `make clean`. `CHAIN_N_SAMPLES`
  changes are handled automatically by a stamp file.
- **Runtime timeout (`ERT_CMD_STATE_TIMEOUT`)**: a deadlock. Likely
  causes — see `placement.py` DESIGN RULES section. Most common is a
  cons-side `acquire(N)` with N>1 emitting an `AcquireGE,N` that blocks
  the chain ahead of it. Smallest acquire first, or bump fifo depths.
- **`__divsi3` in a kernel `.o`** (visible via `llvm-readobj -r build/foo.o`):
  Peano can't prove signed-int division by a power of 2 is shift-eligible.
  Cast the dividend to `(uint32_t)` and the divisor to `Nu` — `/8`
  becomes `>> 3`. (Cleanup-only, not a major perf lever.)
- **Looking at the compiled `.o` directly**: `llvm-objdump -d build/foo.o`
  to see instructions; `grep -E '\bv[a-z]+\b'` for vector ops; the
  presence of `vmac`, `vldb`, `vmov` means you got vectorization. If
  the kernel was supposed to vectorize and only has scalar `mac/mov/add`,
  check the surrounding loop for branches that might be inhibiting the
  vectorizer.

### Adding a new block builder

1. Add a `_build_mN(...)` function in `aie2_yolo_per_block.py` following
   the pattern of an existing block of similar topology (conv_stride →
   `_build_conv_stride_block`, c3k2 → `_build_c3k2_small/heavy`,
   attention → `_build_psa` or the staged variant in `scripts/m9_stage.py`).
2. Register it in the `_BUILDERS` dict near the bottom of
   `aie2_yolo_per_block.py`.
3. Add the block to `yolo_spec.NETWORK` so the layer shapes / kernel
   parameter inference works.
4. Add a `PLACEMENT[block_name]` entry (or delegate to a staged builder
   that hardcodes its own coords, like `m8_megakernel_2tile.py`).

## Known limitations and follow-ups

- **m3 / m5 are the new bottlenecks**: at 76 / 74 ms standalone each,
  they dominate the chain critical path now that m8 is fast (28 ms). The
  same kernel template (`yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc`)
  serves m3, m5, and m7 with different shape constants — measurable
  per-block via `make BLOCK=mN time`.
- **PSA m9 multi-sample**: end-to-end multi-sample dispatch already
  works via `CHAIN_N_SAMPLES=N` (HW BD-chain replays the per-sample
  transfer N times in one XRT call). What's *not* done: pushing the batch
  dimension into the PSA m9 kernel signatures so the attention matmuls
  compute on `(B, nh, S, d)` natively instead of N sequential
  `(nh, S, d)` tensors. Whether it would pay off depends on the current
  per-sample breakdown — m9 isn't the top contributor today.
- **Trace flow tile cap**: AIE2 trace packet IDs cap simultaneously-
  traced workers at 31 — the full chain (26 compute tiles) fits, but
  larger designs may need front-half / back-half traces (the
  `TRACE_BLOCKS=m0,m1,...` env in `aie2_yolo_iron_partial.py` handles
  this).
- **m8 megakernel scalar skip-row copy**: the worker body has a small
  IRON-emitted `for x: for kk: sk_r[x,0,kk] = sa_mid[x,0,kk]` to forward
  one row from split_a to pair0_skip (and same for pair1). The cost is
  small relative to the conv but folding it into `k_pair_cv1` as a second
  vectorized output would be cleaner. See the TODO in
  `scripts/m8_megakernel_2tile.py:_do_p0c1`.

## References

- [`programming_examples/ml/mobilenet/`](../mobilenet) — the reference
  IRON design this example parallels in structure
- [AMD Quark documentation](https://quark.docs.amd.com/) — the int8
  quantization library used by `notebooks/quark_quantization.ipynb`
- [Ryzen AI Software Stack](https://ryzenai.docs.amd.com/) — the Vitis
  AI Execution Provider path for running the same XINT8 ONNX on Ryzen AI
  through ONNX Runtime
- [Ultralytics YOLO26](https://docs.ultralytics.com/) — upstream model
