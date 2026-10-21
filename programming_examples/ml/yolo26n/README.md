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
pip install -r ../../../python/requirements_ml.txt  # torch (CPU), onnx, onnxruntime, opencv
pip install -r requirements.txt     # export only: ultralytics, amd-quark (Python 3.11-3.13)
export IRON_EXAMPLE_WEIGHTS_DIR=$HOME/models
make export                 # checkpoint, FP32 and XINT8 ONNX, coco128
make data                   # data/yolo26n_params.npz
make verify_oracle N=128    # numpy oracle vs ONNX Runtime, tensor by tensor (see Accuracy)
make run_blocks_synthetic   # one range of each block kind on the NPU, no checkpoint
make run N=8                # the whole model on coco128 images
```

`make run_block BLOCK=52:59` runs any range of `Network.yolo26n().ops` on
the NPU against the oracle (`SYNTHETIC=7` for seeded weights, `OUTPUTS=a,b`
to drain more than the last op's output); `python bringup.py --span FIRST
LAST` names the range by its first and last ops instead.

## Design

- `yolo_spec.py` lays out the network as 168 integer operators, each named
  after the ONNX node it stands for. `yolo26n_numpy.py` is the oracle,
  `gen_yolo_data.py` binds parameters by those names.
- `yolo26n_design.py` lowers any contiguous range of operators onto one
  array. Activations stream one image row at a time as `[C/8][W][8]` int8
  through ObjectFifos, from the producing worker to each consumer.
- Where two paths rejoin (a residual add, a concat), a schedule over which
  upstream rows each output row waits for sizes the fifo depths so neither
  path stalls the other. Rows held longer than a compute tile can buffer are
  staged in a memtile, or in a ring buffer on chained workers when they need
  more buffer descriptors than a memtile channel has; rows a consumer holds
  for the whole map, as across an attention barrier, go to DDR and back.
- Runs of operators (the C3k blocks and bottlenecks, the PSA FFNs, the
  Detect branches) run fused, one worker each, their rows in rings in its
  tile (`FUSED`, `SegmentDesign.fuse`).
- A conv whose weights do not fit beside its rows streams them from the
  host's weights buffer one output-channel chunk at a time, each chunk
  followed by its bias, once an output row.
- C2PSA attention runs two workers per head, each answering every other row
  of queries; a memtile splits each qkv row into the heads' queries, keys
  and values, and another joins every worker's rows into the output.
- The Detect postprocess (best class per anchor, top 300 anchors, top 300
  anchor-class pairs, box decode) runs on one worker, `kernels/yolo_detect.cc`.
- `yolo26n.py` runs the network as four such segments (`SEGMENTS`), each its
  own device configuration in one full ELF, handing tensors on through a DDR
  scratch buffer, in one dispatch.

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

On Strix (NPU2), one 640x640 image takes 11.3 ms (88.5 fps) at batch 1 and
6.22 ms per image (161 fps) at batch 8:

```bash
python yolo26n.py --synthetic 7 --n 8 --time              # batch 1
python yolo26n.py --synthetic 7 --n 8 --batch 8 --time    # batch 8
```

Each number is the fastest of 20 calls after 10 warm-up calls, the host's
wall time of the whole `yolo26n(...)` call divided by the batch; the line
also prints the median and its MAD (medians 11.5 ms and 6.37 ms here). The
weights and images are synthetic, so no checkpoint is needed; real ones move
the same data, but the postprocess's work depends on the scores, so their
time can differ a little. The design runs as four device configurations of
one full ELF. A batch-1 dispatch configures every segment's workers again,
at roughly 60-110 us a worker, which a batch shares.
