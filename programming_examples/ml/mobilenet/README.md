# MobileNet V3 on AI Engine (IRON)

End-to-end MobileNet V3 inference on the Strix NPU2 (4×8 compute tiles + a row
of mem tiles), implemented in the high-level IRON Python API. The full network
runs depth-first across the array — `init → bn0..bn14 → avgpool → FC1 → FC2` —
keeping intermediate activations on-chip.

<p align="center">
 <picture>
  <source media="(prefers-color-scheme: light)" srcset="./mobilenet_dataflow.png">
  <img alt="dataflow" src="./mobilenet_dataflow.png">
 </picture>
</p>

## Where to start

Open in this order to grasp the design:

| File | What it shows |
|---|---|
| [`network_spec.py`](network_spec.py) | The whole network in one file — block names, layer kinds, in/out shapes, scale-factor keys |
| [`mobilenet_numpy.py`](mobilenet_numpy.py) | Pure-numpy reference; bit-exact int8 inference matching the AIE kernels (algorithm onramp) |
| [`aie2_mobilenet_iron.py`](aie2_mobilenet_iron.py) | Full IRON design — orchestrates init + bottlenecks + post-processing on a fixed PLACEMENT |
| [`bottleneck/{regular,pipeline,cascade}.py`](bottleneck/) | Three families of bottleneck builders, grouped by tile-mapping strategy |
| [`aie2_iron_per_block.py`](aie2_iron_per_block.py) | Build any single bottleneck standalone (debugging / profiling) |
| [`aie2_iron_chain.py`](aie2_iron_chain.py) | Build a chained subset (`pipeline` = bn10..12, `cascade` = bn13..14) |

## Build + run end-to-end

```bash
make run_py
```

This compiles `aie2_mobilenet_iron.py` to xclbin and runs `test_mobilenet.py`
against `data/golden_output.txt`. Per Xilinx/mlir-aie issue #3009 the AIE
design currently differs from brevitas by max=9 (atol=9 in the assertion).

## Lit-driven verification

| Lit test | Hardware? | What it covers |
|---|---|---|
| [`run_strix_makefile.lit`](run_strix_makefile.lit) | yes | Full mobilenet end-to-end (atol=9 per #3009) |
| [`run_e2e.lit`](run_e2e.lit) | yes | bn1/2/3/6/7/8 standalone (atol=0), pipeline (bn10..12) + cascade (bn13..14) chains (atol=0), plus the regular bn0..bn9 chain (atol=14, matches the original perf-benchmark tolerance — see #3009) |
| [`run_numpy_per_bn.lit`](run_numpy_per_bn.lit) | no | numpy reference vs brevitas — every kernel arithmetic verified bit-exact |

The bit-exact per-block / per-chain results prove the IRON wrapper + AIE
kernel object files are correct on hardware; the residual gap on the full
network comes from cross-block effects in `init` / `bn0` / `post_l1` / `post_l2`
(blocks without a standalone brevitas fixture).

## Regenerating brevitas fixtures

The `data/` files (scale factors, per-block weights, golden outputs) are
brevitas-quantized references — produced by [`gen_golden.py`](gen_golden.py),
NOT IRON-specific. To regenerate them:

```bash
pip install -r requirements_gen_golden.txt    # PyTorch + brevitas + onnx
python3 gen_golden.py                          # writes data/scale_factors_final.json
                                               # + data/{bn*,init,post,FC*}_chain.txt
                                               # + data/golden_output.txt
                                               # + data/before_ifm_mem_fmt_1x1.txt
```

The per-block / per-chain fixtures used by `run_e2e.lit` live under
`bottleneck_A/data/`, `bottleneck_B/data/`, `bottleneck_C/data/` — each
calibrated independently by the corresponding `gen_golden*.py` script in
that subdirectory. Calibration images are not in this repo.

## Repo layout

```
mobilenet/
├── aie2_mobilenet_iron.py         # full network (IRON)
├── aie2_iron_per_block.py         # standalone block builder
├── aie2_iron_chain.py             # standalone chain builder (pipeline | cascade)
├── network_spec.py                # declarative algorithm description
├── mobilenet_numpy.py             # bit-exact numpy reference
├── bottleneck/                    # IRON builders by tile-mapping strategy
│   ├── regular.py                 #   bn0..bn9 (single-tile + fused-pair)
│   ├── pipeline.py                #   bn10..bn12 (3-tile + bn12 2-tile)
│   ├── cascade.py                 #   bn13, bn14 (5-tile cascade-split)
│   └── _common.py                 #   shared helpers
├── data/                          # brevitas fixtures for the full network
├── gen_golden.py                  # brevitas reference generator
├── bottleneck_{A,B,C}/data/       # brevitas fixtures for per-block / per-chain tests
├── bottleneck_{A,B,C}/gen_golden*.py  # brevitas generators for those fixtures
├── test_mobilenet.py              # full-network host runtime harness
├── test_e2e.py                    # per-block / per-chain hardware harness
├── test_numpy_per_bn.py           # numpy bit-exactness driver
├── run_e2e.sh                     # build+compile+run shell driver for run_e2e.lit
└── run_*.lit                      # lit tests (see table above)
```
