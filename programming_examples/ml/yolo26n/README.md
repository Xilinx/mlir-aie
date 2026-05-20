# YOLO26n-cls (INT8) on AI Engine (IRON)

End-to-end YOLO26n classification head inference on the Strix Point NPU2
(8 columns × 6 rows of compute tiles + a row of mem tiles), implemented in
the high-level IRON Python API. The full INT8 network — 11 blocks
(`m0..m10`) — runs depth-first across the array, keeping intermediate
activations on-chip, and emits classification probabilities bit-exactly
matching the Quark XINT8 ONNX reference.

This example parallels [`mobilenet/`](../mobilenet) in structure (per-block
builders, full-chain orchestrator, per-block ORT-driven host tests) and
adds the full int8 quantization recipe — the Jupyter notebooks under
[`notebooks/`](notebooks) walk through training a YOLO26n-cls FP32
checkpoint and quantizing it with [AMD Quark](https://quark.docs.amd.com/)
to produce the deployment ONNX shipped under [`models/`](models).

## Where to start

| File | What it shows |
|---|---|
| [`yolo_spec.py`](yolo_spec.py) | The whole network in one file — block names, layer kinds, in/out shapes, Conv / MatMul / Gemm records |
| [`tile_layout.py`](tile_layout.py) | ASCII diagram of the 8×6 NPU2 tile grid with per-block placement and weight/activation sizing |
| [`placement.py`](placement.py) | `PLACEMENT[block_name]` — physical tile assignments + the deadlock-avoidance design rules learned during bring-up |
| [`aie2_yolo_per_block.py`](aie2_yolo_per_block.py) | The meat — one builder per block (`_build_m0` … `_build_m10`) using IRON `ObjectFifo` / `Worker` / `Runtime` |
| [`aie2_yolo_iron_partial.py`](aie2_yolo_iron_partial.py) | Chain orchestrator. `CHAIN_BLOCKS=m0,m1,...` env var selects the span; defaults to the full `m0..m10` chain |
| [`aie2_yolo_iron.py`](aie2_yolo_iron.py) | Full-network IRON design (m0 → m10 → softmax) for a one-shot non-incremental build |
| [`kernels/`](kernels) | 40 hand-written AIE2P `.cc` kernels for the four kernel families (conv2dk3_stride2, c3k2_small, c3k2_heavy, PSA m9, head m10) |
| [`notebooks/quark_quantization.ipynb`](notebooks/quark_quantization.ipynb) | Full int8 recipe: FP32 PyTorch → FP32 ONNX → calibration → Quark XINT8 PTQ → accuracy validation |
| [`notebooks/yoloexploration.ipynb`](notebooks/yoloexploration.ipynb) | Upstream context: loading YOLO26n, running inference, COCO validation |

## Quickstart

```bash
# 1. Activate the mlir-aie env (provides aiecc.py, PEANO_INSTALL_DIR, MLIR_AIE_INSTALL_DIR).
source ../../../utils/env_setup.sh ../../../install

# 2. Extract per-block weight binaries + SiLU LUTs from the shipped XINT8 ONNX.
make data

# 3. Build and run a single block (any of m0..m10):
make BLOCK=m0
make run_ort BLOCK=m0      # bit-exact NPU vs ORT
make BLOCK=m10
make run_ort BLOCK=m10

# 4. Build and run the full chain (m0..m10 in a single xclbin):
make chain
make run_chain             # bit-exact NPU chain vs ORT, end-to-end
```

`make data` is idempotent and gitignored; the resulting `data/` tree is
~2 MB of per-layer weight/bias/scale bins.

## The int8 recipe

The deployment story has three stages: train float, quantize to int8, and
materialize per-block kernel inputs.

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
   Softmax causes a sign-flip that drops accuracy to ~44%; keeping it FP32
   recovers 89.5%).
5. Evaluate the XINT8 ONNX on the held-out test set, compare to FP32.

The output is `phase1_25k_xint8_acc0.8968.onnx` (1.7 MB, 89.68% test
accuracy, ~3.2 pp loss from FP32 — typical for static PTQ). This is the
ONNX shipped under [`models/`](models) and consumed by `gen_yolo_data.py` /
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
- A per-op `right_shift = log2(scale_in) + log2(scale_w) - log2(scale_out)`
  for the int32→int8 banker-rounded requant epilogue.
- `data/manifest.json` indexing all of it by layer name.

[`gen_yolo_silu_luts.py`](gen_yolo_silu_luts.py) handles the SiLU
non-linearity: the Quark XINT8 graph encodes SiLU as
`Conv → QL → DQ → HardSigmoid → Mul(const) → QL → DQ → Mul(linear) → QL`.
Per Conv we extract the three QuantizeLinear scales + HardSigmoid params
+ constant multiplier, then sample the closed-form HardSiLU over all 256
INT8 inputs to build a 256-byte LUT (`data/model.N/.../silu_lut.bin`).
At runtime each kernel's epilogue is just one LUT lookup —
`out_i8 = silu_lut[clamp(banker_srs(acc, rs)) + 128]`.

## Design overview

The 11 blocks map to four kernel families:

| Block | Topology | Kernel family | Tiles |
|---|---|---|---:|
| m0 | conv_stride (stem) | `yolo_m0_conv2dk3_silu_bias` (OIYX raw, in_c=3 padded to 8) | 1 |
| m1 | conv_stride | `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8` (shared, runtime shapes) | 1 |
| m3, m5, m7 | conv_stride | `yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked` (per-block, weight-streamed for L1) | 1 each |
| m2, m4 | c3k2_small | `yolo_c3k2_small_{cv1_split, m0_cv1, m0_cv2_skip, cv2_concat3}` | 3 each |
| m6, m8 | c3k2_heavy | `yolo_c3k2_heavy_{m_0_split, inner_pair_cv1, inner_pair_cv2_skip, cv3_concat2}` (+ streamed variants on m8) | 5 each |
| m9 | PSA (attention + FFN) | `yolo_m9_{cv1_split, qkv, qk_row, attn_scale, softmax_row, sv_row, pe_add_row, proj_skip_row, ffn_0_silu_row, ffn_1_skip_row, cv2_concat2_streamed}` | 5 |
| m10 | head | `yolo_m10_{conv2dk1_silu_xy_pool, linear_gemm, softmax}` | 2 |

Total: **28 of 32** compute tiles used. See `tile_layout.py` for the
ASCII diagram and per-block weight + peak activation sizing.

### Kernel symbol mangling

Several blocks share a kernel template (e.g. c3k2_small for m2/m4,
c3k2_heavy for m6/m8, chunked conv_stride for m3/m5/m7) but with different
runtime shapes / weight sizes. To let them coexist in one MLIR module, the
Makefile compiles each block's variant with `-DKERNEL_SUFFIX=_mN`, which
suffixes every exported symbol — so `yolo_c3k2_small_cv1_split.cc`
produces `..._m2.o` and `..._m4.o` with non-colliding `func.func` names.
The IRON builder passes the matching suffixed name into `Kernel(...)`.

### Bit-exactness

The integration test of record is `test_chain_ort.py`: same seeded RGB
input fed to both (a) the NPU chain xclbin and (b) ONNX Runtime on the
XINT8 ONNX, with hard element-wise INT8 equality on the final softmax
probs. All 11 blocks individually pass `test_block_ort.py --block mN`
bit-exact, and the full m0..m10 chain passes `test_chain_ort.py`
bit-exact. The lit test [`run_strix_makefile.lit`](run_strix_makefile.lit)
exercises both paths end-to-end on NPU2 hardware.

## Build system

```
make BLOCK=mN              -> build/aie_mN.mlir + build/final_mN.xclbin + build/insts_mN.bin
make run_ort BLOCK=mN      -> compile (if needed) + run on NPU + bit-exact vs ORT
make chain                 -> build/aie_chain.mlir + build/final_chain.xclbin + build/insts_chain.bin
make run_chain             -> compile chain (if needed) + run on NPU + bit-exact vs ORT
make data                  -> regenerate data/manifest.json + per-Conv SiLU LUTs from the ONNX
make clean                 -> wipe build/
```

The Makefile builds **all** kernel `.o` files up front (small, fast) so any
`BLOCK=mN` or `chain` target can link without re-running the kernel build.
Kernel objects are AIE2P-targeted (`--target=aie2p-none-unknown-elf`), built
with the same flag set as `programming_examples/makefile-common`'s
`PEANOWRAP2P_FLAGS` (inlined locally so the example is self-contained).

## Known limitations and follow-ups

- **PSA batch-loop**: `aie2_yolo_iron.py` runs one sample per Runtime
  invocation, matching mobilenet's convention. The Quark XINT8 ONNX
  declares `batch=4` for the PSA `MatMul` shapes; per-sample-per-invocation
  inside `attn_core` sees `(nh, S, d)`. Throughput today comes from the host
  invoking the design B=4 times back-to-back filling the pipeline. A
  proper in-design batch loop for the PSA block is a follow-up.
- **Vectorization headroom**: all kernels are currently scalar AIE2P C
  loops. End-to-end wall time on a chain run is dominated by m1/m3/m5/m7
  conv compute, not by inter-block lock stalls (early `INSTR_EVENT_0/1`
  trace cycle readings were misleading — those cycle counts understate
  real kernel time by ~200× for the no-op portions of the kernel, so the
  trace-driven "lock-stall" hypothesis was wrong). Vectorizing the
  conv2dk3_stride2 + c3k2_inner kernels is the path to the 60 fps target.
- **Trace flow tile cap**: AIE2 trace packet IDs cap the simultaneously
  traced workers at 31 — blocks m8 stage 5 (32 workers) and the full
  chain (~32 workers) can't be traced wholesale. Per-block or
  per-sub-stage traces work fine.

## Repo layout

```
yolo26n/
├── README.md                          # this file
├── Makefile                           # per-block + chain build targets
├── aie2_yolo_iron.py                  # full-network IRON design
├── aie2_yolo_iron_partial.py          # chain orchestrator (CHAIN_BLOCKS env)
├── aie2_yolo_per_block.py             # per-block builders (the meat)
├── yolo_spec.py                       # declarative network spec (shapes, ops)
├── placement.py                       # PLACEMENT[block_name] -> tiles + design rules
├── tile_layout.py                     # ASCII tile-map + per-block sizing
├── lowlevel_dma.py                    # StaticWeightStream helper for chunked weight DMA
├── kernels/                           # 40 AIE2P .cc/.h kernels (4 families)
├── gen_yolo_data.py                   # XINT8 ONNX -> data/manifest.json + per-Conv bins
├── gen_yolo_silu_luts.py              # XINT8 ONNX -> per-Conv SiLU 256-byte LUT bins
├── data/                              # gitignored; regenerated by `make data`
├── test_block_ort.py                  # per-block NPU vs ORT bit-exact host driver
├── test_chain_ort.py                  # chain NPU vs ORT bit-exact host driver
├── notebooks/
│   ├── quark_quantization.ipynb       # full int8 recipe (FP32 -> XINT8)
│   └── yoloexploration.ipynb          # upstream YOLO26n context + COCO val
├── models/
│   ├── phase1_25k_acc0.9424.pt        # FP32 PyTorch source (3.1 MB)
│   └── phase1_25k_xint8_acc0.8968.onnx # Quark XINT8 deployment artifact (1.7 MB)
└── run_strix_makefile.lit             # lit test (m0 end-to-end)
```

## References

- [`programming_examples/ml/mobilenet/`](../mobilenet) — the reference IRON
  design this example parallels in structure
- [AMD Quark documentation](https://quark.docs.amd.com/) — the int8
  quantization library used by `notebooks/quark_quantization.ipynb`
- [Ryzen AI Software Stack](https://ryzenai.docs.amd.com/) — the Vitis AI
  Execution Provider path for running the same XINT8 ONNX on Ryzen AI
  through ONNX Runtime
- [Ultralytics YOLO26](https://docs.ultralytics.com/) — upstream model
