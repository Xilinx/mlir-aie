<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# YOLO26n detection on AI Engine (IRON)

Stock Ultralytics YOLO26n object detection at 640x640 on the NPU2 (AIE2P),
written in IRON. All of the graph runs on the NPU, from a letterboxed uint8
RGB image to the end-to-end `(300, 6)` detections (x1, y1, x2, y2, score,
class): the input quantizer, every conv, C3k2, SPPF, C2PSA attention, the
neck, the Detect head and its top-k postprocess. The device is checked bit
for bit against an integer numpy oracle, and the oracle against ONNX Runtime
on the quantized model.

The integer recipe is AMD Quark XINT8 with power-of-two scales, accumulators
seeded with the bias, `conv_even` requantization, and a per-conv int8 SiLU
table.

## Weights

The checkpoint is AGPL-3.0, so nothing derived from it is kept in the
repository. `make export` downloads `yolo26n.pt` and the coco128 images,
checks both against sha256 digests pinned in `export_yolo26n.py`, exports
the model to ONNX and quantizes it, calibrating on coco128. Everything goes
to `$IRON_EXAMPLE_WEIGHTS_DIR/yolo26n`, so point that variable at a
directory you can write to (unset, it is `/srv`). `make data` then extracts
the integer parameters into `data/yolo26n_params.npz`.

The lit tests need no checkpoint: `--synthetic SEED` swaps in seeded weights
with the same layout, and random images.

## Quick start

```bash
pip install -r requirements.txt    # onnx, onnxruntime, opencv
pip install -r ../../../python/requirements_ml.txt  # torch, CPU build (export only)
pip install -r requirements_export.txt  # ultralytics, amd-quark (Python 3.11-3.13)
export IRON_EXAMPLE_WEIGHTS_DIR=$HOME/models
make export                 # checkpoint, FP32 and XINT8 ONNX, coco128
make data                   # data/yolo26n_params.npz
make verify_oracle N=128    # numpy oracle vs ONNX Runtime, tensor by tensor (see Accuracy)
make run_blocks_synthetic   # one range of each block kind on the NPU, no checkpoint
make run N=8                # the whole model on coco128 images
```

`make run_block BLOCK=57:73` runs a range of `Network.yolo26n().ops` that
starts and ends on fused-run boundaries on the NPU against the oracle
(`SYNTHETIC=7` for seeded weights, `OUTPUTS=a,b` to drain more than the last
op's output); `python bringup.py --span FIRST LAST` names the range by its
first and last ops instead, and `--batch N` pipelines N images through it.

## Design

- `yolo_spec.py` lays out the network as 168 integer operators, each named
  after the ONNX node it stands for. `yolo26n_numpy.py` is the oracle,
  `gen_yolo_data.py` binds parameters by those names.
- `yolo26n_design.py` maps the whole network onto NPU2's 32 tiles in one
  device configuration: 30 workers and two tiles that only store attention
  values. Activations stream one image row at a time as `[C/8][W][8]` int8
  through ObjectFifos, from the producing worker to each consumer. Any range
  of operators that starts and ends on fused-run boundaries builds the same
  way, for bring-up.
- Runs of operators (`FUSED`, `SegmentDesign.fuse`) run fused, one worker
  each, every operator at its own rate on one clock; the run's steps roll
  into one loop, with guards on the rows each step takes and makes.
- Where two paths rejoin (a residual add, a concat), a schedule over which
  upstream rows each output row waits for sizes the fifo depths so neither
  path stalls the other. Rows a consumer holds past its own window wait in
  chains of memtile holds, a stage ending at each reader's lag; rows between
  neighbouring workers go over shared memory.
- Each fused worker reads one stream of weight chunks: resident in a memtile
  where a step's chunks fit, else streamed from the host's weights buffer.
- C2PSA attention runs on one worker per block for every head, which also
  makes PSA's positional conv and add. Its keys stay in its own tile, its
  values in a neighbour that runs no core, and its queries wait in a memtile
  that walks the stream of qkv rows.
- The Detect postprocess (best class per anchor, top 300 anchors, top 300
  anchor-class pairs, box decode) runs on one worker, `kernels/yolo_detect.cc`,
  reading the Detect outputs back from a DDR scratch buffer.
- `yolo26n.py` runs the network in one dispatch. One build serves every
  batch of 1 to 64 images: each batch is an entry of the design, its own
  instruction stream beside the one xclbin, and `--batch` picks it.

## Accuracy

mAP50-95 on coco128, measured when the model was exported and not recomputed
by this example: FP32 0.474, XINT8 0.443. The XINT8 model is calibrated on
the same 128 images, so its number is optimistic.

On the NPU, the whole model's detections match the oracle bit for bit on
all 128 coco128 images, and ONNX Runtime's on 126. The oracle agrees with
ONNX Runtime on those 126 in all 169 int8 tensors the spec names (the
outputs of every operator but the float postprocess, and both attentions'
values). In the other 2, the oracle's
softmax and ONNX Runtime's float32 one round an attention probability (4 of
82 million) to opposite sides of a rounding boundary; neither changes a
detection above 0.25 confidence.

`verify_oracle.py` settles each such probability with exact arithmetic
(`SoftmaxArbiter`). An image passes when it is exact, or when ONNX Runtime
is the one off at every differing probability and the oracle, run again
with ONNX Runtime's probabilities, then matches it in every tensor. It
exits 0 only when every image passes.

## Performance

The nightly runs every batch in `BATCHES` from one compile, on the real
weights over coco128, and records per image the end-to-end and NPU time
and the frame rate, with mAP@0.5 against coco128's labels. The current
figures and their history are on the
[maintainer dashboard](https://xilinx.github.io/mlir-aie/dashboard/#view=performance).
To time it yourself:

```bash
python yolo26n.py --n 16 --batch B --time
```

Each time is the fastest of 20 calls after 10 warm-up calls, the host's wall
time of the whole `yolo26n(...)` call divided by the batch, beside the
median. A batch pipelines its images through the array, fills issued three
images ahead. Compiling takes several minutes, nearly all of it placing the
workers.
