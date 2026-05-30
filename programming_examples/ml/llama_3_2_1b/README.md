# Llama 3.2 1B (INT8) on AI Engine (IRON) — WIP

End-to-end INT8 Llama 3.2 1B decode on the AMD Strix Point NPU2
(AIE2P), implemented in the high-level IRON Python API and structured
to mirror `programming_examples/ml/yolo26n/`. All 16 decoder layers
will run via a single reused worker set with per-layer INT8 weights
streamed from DRAM, and on-device sampling — bit-exact against the
`cautious-eureka` numpy reference oracle.

The dataflow design + per-channel INT8 quant recipe were developed and
simulation-validated in the `cautious-eureka` repo. This example is the
hardware bring-up.

## Status

| Phase | State |
|---|---|
| 0. Scaffolding (`llama_spec.py`, `placement.py`, env) | done |
| 1. `gemm_int8_srs` standalone bit-exact on NPU | in progress |
| 2. Glue kernels (rmsnorm_residual, rope, silu_mul)  | pending |
| 3. FlowKV qk/sv pair                                 | pending |
| 4. Single decoder layer bit-exact end-to-end         | pending |
| 5. 16-layer decode chain                             | pending |
| 6. `sample` kernel + end-to-end generation           | pending |
| 7. Prefill overlay                                   | deferred (follow-up PR) |

## Quickstart

```bash
# 1. Env (one-time)
source /opt/xilinx/xrt/setup.sh
source ../../../utils/quick_setup.sh        # creates ironenv via wheels
pip install --upgrade cmake                 # ironenv cmake (needs >= 3.30)
pip install -r ../../../python/requirements_ml.txt

# 2. Point at weights (only needed for layer-level tests; not for kernel-level)
export LLAMA_3_2_1B_WEIGHTS=/scratch/roesti/models/llama_3.2_1b

# 3. List targets
make help
```

## Design files

- `llama_spec.py` — algorithm/shapes (one decoder layer, parameterized by M).
  Mirrors yolo26n's `yolo_spec.py`.
- `placement.py` — physical tile placement (decode + prefill overlays).
  Mirrors yolo26n's `placement.py`.
- `aie2_*.py` — IRON designs (added phase-by-phase).
- `kernels/llama_*.cc` — AIE2P kernels (added phase-by-phase). Many
  start from `aie_kernels/aie2p/{mm,rms_norm,rope,softmax,swiglu}.cc`.

## Provenance

Design + simulation-validated bit-exactness against a numpy reference
live in the [cautious-eureka](../../../../) repo. The reference oracle
is `cautious-eureka/npu2/llama_layer_ref.py`. Per-channel INT8 quant
math: `cautious-eureka/npu2/measure_weight_compressibility.py`.
