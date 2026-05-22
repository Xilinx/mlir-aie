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

# 1b. Install the ML extras (onnx + onnxruntime + torch). The standard
#     mlir-aie env install only pulls requirements.txt + requirements_dev.txt;
#     `make data` and the ORT bit-exact tests need this file too.
pip install -r ../../../python/requirements_ml.txt

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

All measurements taken with the NPU in turbo mode
(`sudo xrt-smi configure --pmode turbo`).

**Chain — partial (re-measured at current commit).** Full m0..m10 chain
numbers will land once the per-block re-validation is complete; the rows
below were collected by `CHAIN_BLOCKS=… make {run_chain,time_chain}`.

| Chain | N=1 wall | N=15 per-sample | N=15 fps |
|---|---:|---:|---:|
| m0           | 16.35 ms | 13.02 ms | **76.78** |
| m0..m1       | 14.66 ms | 13.01 ms | **76.85** |
| m0..m1..m2   | 32.17 ms | 30.94 ms | **32.32** |

**Per-block standalone wall time on NPU**, median of n=20 (turbo).
Rows marked ✓ have been re-measured at the current commit; the rest
are pre-validation snapshots pending re-measurement.

| Block | Topology | Median (ms) | fps | |
|---:|---|---:|---:|:--|
| m0  | conv_stride stem               | 13.10 | 76.3  | ✓ |
| m1  | conv_stride                    |  8.63 | 115.8 | ✓ |
| **m2**  | **c3k2_small**             | **31.33** | **31.9** | ✓ |
| m3  | conv_stride (chunked)          |  8.10 | 123.5 | |
| m4  | c3k2_small                     | 21.96 | 45.5  | |
| m5  | conv_stride (chunked)          |  6.97 | 143.5 | |
| m6  | c3k2_heavy                     | 15.10 | 66.2  | |
| m7  | conv_stride (chunked)          |  4.78 | 209.2 | |
| m8  | c3k2_heavy (2-tile megakernel) | 28.19 | 35.5  | |
| m9 stage 1 (cv1 only)            | PSA cv1 |  1.78 | 561.8 | |
| m9 stage 10 (full PSA block)     | PSA full | 26.25 | 38.1  | |

> **Stale — pre-fix snapshot, pending re-measurement.** Numbers below
> are from before the c3k2_small bot_fifo depth fix (4→6) and the
> per-block re-validation pass; they referenced a chain that sometimes
> deadlocked on m2 (since fixed) and the m2 contribution of 0.18 ms is
> not reproducible at the current commit.
>
> **Chain bottleneck after the m1 deep-opt is no longer a single block.**
> NOOP_BLOCK ablation (chain N=15, per-sample ms): noop'ing m2 alone
> drops chain by 0.18 ms, m4 by 0.04 ms — both are nearly fully
> overlapped with the pipeline. m8 contributes 8.35 ms, m9 contributes
> 8.41 ms, m2+m8+m9 together 13.79 ms. The chain floor with the 3
> biggest blocks noop'd is 22.48 ms — that's dataflow / DMA infrastructure
> plus per-tile pipeline overhead, not compute. Per-kernel deep-opt
> beyond this point yields per-kernel speedups but small chain deltas;
> chain wins come from architectural / fifo-depth work.

m9 is supported standalone via `M9_STAGE=N make BLOCK=m9 run_ort` —
stage 1 compares against `/model.9/cv1/act` post-SiLU; stage 10 (default)
compares against `/model.9/cv2/act` post-SiLU. Intermediate stages emit
PSA-internal tensors with no clean ORT analog; verify those via
`make run_chain`. m10 (head) still standalone-unsupported; covered by
the chain test.

**Chain bottleneck has shifted twice this session.** m9 used to
dominate (>80% of chain time → ~260 ms standalone); after the m9 vec
arc it dropped to 26 ms and **m0 became the ceiling at 55 ms**. The
subsequent m0 deep-opt commit took m0 to 13 ms, making **m1 (45 ms)
the new ceiling**.

**Latest optimization arc** — m9 PSA kernels rewritten with
`aie::mmul<4,8,8,int8,int8>` + vector-broadcast reductions + scalar
SRS/SiLU/skip-add tails. Includes one fusion (`qk_row + attn_scale +
softmax_row` → `yolo_m9_attn_score_fused_vec.cc`) and one symbol merge
(`sv_row + sv_row_acc` → single `_vec.cc` with two extern C entries):

| step | m9 stage 10 (ms) | chain N=15 (ms/sample) | chain N=15 fps |
|---|---:|---:|---:|
| pre-session (scalar m9)                                | 353  | 317.5 | 3.15 |
| + cv1 vec (chunked 256→256 + top/bot)                  | 271.6| 256   | 3.91 |
| + qkv vec (128→256, no act)                            | 199.4| 210   | 4.76 |
| + ffn.0 vec (chunked 128→256 + SiLU)                   | 194.7| 205   | 4.87 |
| + proj vec (128→128 + cross-scale add)                 | 192.5| 203   | 4.93 |
| + cv1 deep-opt (pre-pack + multi-acc)                  | 191.1| 202.8 | 4.93 |
| + ffn.1 vec (256→128 + same-scale add)                 | 186.5| 198.2 | 5.05 |
| **+ cv2 vec (chunked 256→256 + concat) — m9 critical path** | **63.1** | **74.9** | **13.35** |
| + qk_row vec (broadcast Q[k] × K[k,j..j+15])           | 61.6 | 73.4  | 13.63 |
| **+ sv_row vec (4-way c-fold) — m9 critical path**     | **40.9** | **58.8** | **17.00** |
| + sv_row_acc merge (same body, second extern C entry)  | 29.2 | 58.0  | 17.24 |
| **+ attn_score_fused (qk + scale + softmax in 1 kernel)** | **26.3** | **57.9** | **17.27** |

After the m9 arc m0 then m1 become successive chain ceilings. Both
deep-opt'd with the same toolbox (shape #defines + OCx2 / 2X×2OC fold
+ x-split + AIE_LOOP_RANGE hints):

| step | block standalone (ms) | chain N=15 (ms/sample) | chain N=15 fps |
|---|---:|---:|---:|
| naive vec (pre-deep-opt)                                | — | 57.9 | 17.27 |
| + m0 deep-opt (shape #defines + OCx2 + x-split)         | m0: 55.4 → 13.1 | 47.6 | 20.99 |
| **+ m1 deep-opt (same toolbox, 2X×2OC interior fold)**  | **m1: 45.1 → 8.65** | **36.3** | **27.57** |
| **total session (chain N=15)**                          | — | **3.15 → 27.57 fps** | **8.75×** |

Two takeaways from the arc:
1. **Standalone speedup only translates to chain delta when the kernel is
   on the critical path.** cv2 and sv_row (the two big chain jumps) were
   the m9 critical-path kernels; the others shaved m9 standalone but
   barely moved chain because they overlapped with neighbor tiles.
2. **Fusion's biggest win is structural, not direct cycles.** The
   qk+scale+softmax fusion saved ~3 ms on m9 standalone but barely moved
   chain — because m9 isn't the bottleneck anymore. Fusion also
   collapsed 3 source files into 1 and sets up future deep-opt to span
   the whole compute (e.g., vector `aie::exp2<bfloat16>` for softmax —
   see `aie_kernels/aie2p/softmax.cc` reference).

**Prior optimization arc** (chunked stride-2 conv, kernel
[`yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc`](kernels/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc)
shared by m3/m5/m7) — 7.7-10.6× standalone vs scalar baseline:

| stage | m3 | m5 | m7 |
|---|---:|---:|---:|
| baseline scalar gather, single acc | 76.4 ms | 73.6 ms | 36.5 ms |
| + `AIE_LOOP_RANGE` pragmas + 64-bit word-copy gather | 22.5 | 23.5 | 15.4 |
| + 3-way x_tile split (edges / interior / edge), edges inlined | 16.0 | 18.8 | 14.5 |
| + OC×2 fold (one A-operand gather feeds 2 weight banks → 2 accs) | 13.5 | 17.5 | 15.0 |
| + 2X×2OC (4 accs: 2 X positions × 2 OC banks, example.h-style) | 13.5 | 17.4 | 15.2 |
| **+ compile-time shape `#define`s + `INTERIOR_MODE` selector**   | **8.04** | **6.95** | **4.73** |
| total speedup | **9.5×** | **10.6×** | **7.7×** |

The compile-time-shapes step was the single biggest lever — it lets peano
fold `* in_c`, `* out_c`, `* (ic_tiles*kH*kW*64)` etc. into shifts /
immediates, dead-strip the unused interior variant, and emit tighter
pipelined inner loops. See [kernel design notes](#kernel-design-notes-chunked-stride-2-conv).

Distance to 60 fps target (16.67 ms/sample): chain N=15 currently at
36.3 ms = **~2.2× more needed**. **No single block is the chain ceiling
anymore** — NOOP_BLOCK ablation shows m2 / m4 / m6 are nearly fully
overlapped (chain contribution ≤ 0.2 ms despite 15-26 ms standalone),
m8 contributes 8.35 ms, m9 contributes 8.41 ms, and the chain floor
with m2+m8+m9 noop'd is 22.48 ms of dataflow/DMA infrastructure.

Remaining levers in priority order:
1. **m8 / m9 deep-opt + fusion** — only blocks NOT fully overlapped.
   m8 has 4 naive vec kernels (front, back, pair0/1 cv1/cv2 streamed);
   m9 has 4 naive vec (qkv, proj_skip_row, sv_row, attn_score_fused)
   + 2 partial deep-opt (ffn_1, cv2_concat2 — promote to full) + 4 scalar
   (pe_add_row + qkv/qk/v_pack data shuffles). Direct chain delta:
   ~5-13 ms if both can shed their non-overlap portions.
2. **Architectural / fifo / placement work** — to push the 22.48 ms
   infrastructure floor down. Tile-pair rebalancing, fifo depth tuning,
   OC-split parallelism for hottest blocks.
3. **m10 head** (`yolo_m10_linear_gemm.cc` + `yolo_m10_softmax.cc`)
   still scalar; small per-sample contribution but completes the
   "every kernel deep-opt'd" goal.
4. **Retroactive deep-opt of overlapped naive vec kernels** (m2 / m4
   c3k2_small; m6 c3k2_heavy; m9 qkv / ffn.0 / proj / ffn.1 / cv2;
   m8's 4 kernels) — per-kernel speedups but ~0 chain delta. Needed for
   the "deep-opt all kernels" code-quality goal.
5. **Softmax exp2** — replace the scalar fp32 LUT in
   `yolo_m9_attn_score_fused_vec.cc` with hardware
   `aie::exp2<bfloat16>` (see `aie_kernels/aie2p/softmax.cc`
   reference). Bit-exactness risk; tested separately.

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
- `M9_STAGE=N` — for `BLOCK=m9`, selects which staged build of m9 to
  produce (1 = cv1 only, 10 = full PSA block; default 10). Same stamp-
  file rebuild trigger as `CHAIN_N_SAMPLES`. `M9_CHAIN_STAGE` is
  accepted as a legacy alias. Only stages 1 and 10 are wired through
  `test_block_ort.py` (others lack a clean single-tensor ORT analog).
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
| m9 | PSA (attention + FFN) | 12 kernels: `yolo_m9_{cv1_split, qkv, qkv_pack, qk_pack, attn_score_fused, v_pack, sv_row, pe_add_row, proj_skip_row, ffn_0_silu_row, ffn_1_skip_row, cv2_concat2_streamed}` (qk_row + attn_scale + softmax_row collapsed into `attn_score_fused_vec.cc`; sv_row + sv_row_acc share the same `.o` with two extern C entries) | 7 |
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

### Kernel design notes (chunked stride-2 conv)

The shared
[`yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc`](kernels/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc)
that backs m3/m5/m7 demonstrates several patterns worth carrying to other
peano-compiled AIE2P kernels in this repo:

**Compile-time shape specialization.** The kernel requires
`-DYOLO_IN_W=… -DYOLO_IN_C=… -DYOLO_OUT_C=… -DYOLO_INTERIOR_MODE=…` per
block (set in the Makefile's `CHUNKED_SHAPE_$(blk)` lookup). These become
`constexpr int` constants at file scope; the runtime ABI args are kept
in the C signature but the function body reads the constexprs. This is
the single biggest perf lever (typically 1.5-3× on top of vector loops)
because peano can:
- fold address math (`* in_c`, `* out_c`, `* (ic_tiles*kH*kW*64)`) into
  shifts and immediates, eliminating runtime scalar muls,
- dead-strip the unused `if constexpr (...)` branch (in this kernel:
  the unused `INTERIOR_MODE` variant), and
- emit tighter pipelined loops with known trip counts.

**`AIE_LOOP_RANGE` everywhere.** Peano needs `[[clang::loop_min/max_iteration_count]]`
to pipeline (and unroll) loops with runtime-derived bounds. Wrap each
non-fully-unrolled loop with `AIE_LOOP_RANGE(min, max)`; use
`AIE_LOOP_UNROLL_FULL` on small fixed-bound loops (e.g. `kx` over 3).
Note that `AIE_PREPARE_FOR_PIPELINING` is a no-op under peano (it only
lowers for chess) — see `aie_kernels/aie_kernel_utils.h`.

**3-way split for edge vs interior.** The bounds-check branch needed for
left/right border columns blocks software pipelining if it sits inside
the hot loop. Splitting the `x_tile` loop into left-edge / interior /
right-edge nests (with the edges' bodies *inlined*, not factored into
lambdas — peano doesn't reliably inline them) lets the interior become
fully straight-line vector ops.

**OC×2 and 2X×2OC accumulator fold.** The mmul-throughput-hiding pattern
from `example.h`: carry multiple accumulators so each weight load is
reused across multiple X positions and each input gather is reused across
multiple OC banks. `YOLO_INTERIOR_MODE=2` (4 accs: 2 X × 2 OC) for blocks
with enough x_tiles to amortize the pair setup (m3, m5); `MODE=1` (2 accs:
single-X × 2 OC) for small-x_tiles blocks (m7) where the pair setup
overhead would dominate.

**64-bit word-copy gather instead of scalar byte loop.** The 4×8-byte
input gather for one mmul is built via 4 aligned `uint64_t` reads into a
stack `a_buf`, then a single `aie::load_v<32>`. Peano hoists this into
register-to-register `vpush.hi.64` in the steady-state pipelined inner
loop, eliminating the stack round-trip in the hot path.

### Bit-exactness

The integration test of record is `test_chain_ort.py`: same seeded RGB
input fed to both (a) the NPU chain xclbin and (b) ONNX Runtime on the
XINT8 ONNX, with hard element-wise INT8 equality on the final softmax
probs. For `CHAIN_N_SAMPLES=N>1`, every output slice is compared
independently — all N samples must match the single reference.

The conv_stride and c3k2 blocks (m0..m8) each pass `test_block_ort.py
--block mN` bit-exact standalone (see the `run_ort` target). m9 (PSA)
is supported in two stages: `M9_STAGE=1 make BLOCK=m9 run_ort` compares
NPU vs ORT for `/model.9/cv1/act` (cv1 only); `M9_STAGE=10 make BLOCK=m9
run_ort` (default) compares for `/model.9/cv2/act` (full m9). Other m9
stages and m10 (head) aren't supported standalone — covered by the
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
├── kernels/                           # 30 .cc + 2 .h AIE2P kernel files. Mostly `_vec.cc` (vectorized; some at "full deep-opt" level — pre-pack scratch + shape #defines + multi-acc fold + AIE_LOOP_RANGE — others at "naive vec" level). Remaining scalar holdouts: yolo_m9_{qkv,qk,v}_pack.cc + yolo_m9_pe_add_row.cc + yolo_m10_{linear_gemm,softmax}.cc
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

### Block ablation (bottleneck attribution)

When the chain is slower than the per-block sum suggests it should be (or
faster — both happen in pipelined chains), per-block standalone timing
can mislead. Use the `NOOP_BLOCK` Make env var to compile any subset of
blocks' kernels with `-DNOOP_KERNEL`. Their `extern "C"` entries
early-return, so the chain's DMA / lock / placement pattern is unchanged
but those blocks contribute ~0 ms of compute. Output is garbage —
bit-exactness fails by design; the signal is chain wall-clock.

```bash
# Ablation: how much does m9 contribute to the chain?
rm -f build/yolo_m9_*.o && make clean_chain
CHAIN_N_SAMPLES=15 NOOP_BLOCK=m9 make time_chain
# Historical (pre-m9-1x1-vec arc): 317 ms baseline → 57 ms noop'd ⇒ m9 = 260 ms / 82%
# Today (post-cv2 vec):              75 ms baseline → ? ms noop'd  ⇒ rerun to attribute the new remainder

# Combine: noop multiple blocks (space-separated, quoted)
rm -f build/yolo_m9_*.o build/yolo_m10_*.o && make clean_chain
CHAIN_N_SAMPLES=15 NOOP_BLOCK="m9 m10" make time_chain
```

Notes:

- All 11 blocks (m0..m10) are supported.
- The `-DNOOP_KERNEL` flag only affects compile of that block's kernel
  `.o` files. Make won't rebuild a `.o` when only `NOOP_BLOCK` changes
  (mtime tracking misses it), so manually `rm` the affected `.o` plus
  `make clean_chain` between runs. For unsure cases, just `make clean`.
- The harness adds zero overhead when `NOOP_BLOCK` is unset — the
  `#ifdef NOOP_KERNEL return;` guards compile out completely.
- Output garbage from noop'd blocks doesn't break the chain — downstream
  blocks just consume junk inputs. Chain DMA scheduling is unaffected.

This is how the m9-is-the-bottleneck finding was discovered: the
optimization arc that took m3/m5/m7 from 186 ms → 19 ms standalone moved
the chain by only ~14 ms, and ablating each block individually (m0
alone, m1 alone, the 6 conv_stride blocks together) showed each
contributed <1 ms. Only m9 ablation moved the chain meaningfully. The
same harness then surfaced cv2 as the m9 critical-path kernel during
the m9 1×1 vec arc.

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

- **m9 (PSA attention) is still the chain bottleneck, but reduced from
  ~260 ms → 63 ms standalone via the m9 1×1 vectorization arc.** Chain
  N=15 went 317 → 75 ms/sample (3.15 → 13.35 fps); m9 stage 10 standalone
  went 353 → 63.1 ms (5.6×). The cv2 vec commit (`af7f9eece`) was the
  single biggest lever — it was the m9 critical-path kernel that hadn't
  yet been touched. Remaining scalar m9 kernels (`qkv_pack`, `qk_pack`,
  `qk_row`, `attn_scale`, `softmax_row`, `v_pack`, `sv_row`,
  `sv_row_acc`, `pe_add_row`) likely contain the next critical path —
  the attention matmuls (`qk_row`, `sv_row`) are the obvious next
  targets since they're scalar i8 N×N reductions that map cleanly to
  `aie::mmul`.
- **m0 (55 ms) is approaching becoming the next ceiling.** Chain N=15
  per-sample (74.9 ms) is now only ~1.2× m9 stage 10 (63.1 ms); m0
  alone is 55.4 ms standalone. Once m9 is reduced below ~55 ms per-
  sample, m0 will become the bottleneck. m0 uses OIYX raw layout with
  `in_c=3` padded to 8 → 37.5% mmul utilization ceiling, but 512×512
  input means lots of compute to amortize.
- **Per-block standalone savings translate to chain savings only when
  applied to the critical path.** The earlier m9 1×1 vectorizations
  (cv1, qkv, ffn.0, proj, ffn.1) shaved 165 ms off m9 stage 10
  standalone but moved chain N=15 by only ~7 ms — they overlapped in
  m9's pipeline. The cv2 vec, by contrast, shaved 123 ms off stage 10
  and moved chain by 123 ms — it was the serial critical path. Use the
  `NOOP_BLOCK` ablation harness (see
  [Block ablation](#block-ablation-bottleneck-attribution)) and per-tile
  trace before picking the next kernel.
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
