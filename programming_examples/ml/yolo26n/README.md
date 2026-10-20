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

The integer recipe follows PR #3142, a fine-tuned yolo26n-cls at 512 by
@hunhoffe, and ports the decisions validated there to the detection model:
AMD Quark XINT8 with power-of-two scales, accumulators seeded with the bias,
`conv_even` requantization, and a per-conv int8 SiLU table.

## Weights

The checkpoint is AGPL-3.0, so nothing derived from it is kept in the
repository. `export_yolo26n.py` fetches `yolo26n.pt`, exports it to ONNX and
quantizes it into `$IRON_EXAMPLE_WEIGHTS_DIR/yolo26n` (default `/srv/yolo26n`),
calibrating on coco128, which it also downloads there. `gen_yolo_data.py`
extracts the integer parameters into `data/yolo26n_params.npz`. Exporting
needs the extras in `requirements.txt` (ultralytics and amd-quark) and
extracting needs onnx and onnxruntime. Running on real images needs
opencv-python, and onnxruntime to compare with the ONNX graph too.

The lit tests need no checkpoint: `--synthetic SEED` swaps in seeded weights
with the same layout, and random images.

## Quick start

```bash
pip install -r requirements.txt
make export                 # checkpoint, FP32 and XINT8 ONNX, coco128
make data                   # data/yolo26n_params.npz
make verify_oracle N=128    # numpy oracle vs ONNX Runtime, tensor by tensor
make run_blocks_synthetic   # one range of each block kind on the NPU, no checkpoint
make run N=8                # the whole model on coco128 images
```

`make run_block BLOCK=52:59` runs any range of `Network.yolo26n().ops` on
the NPU against the oracle (`SYNTHETIC=7` for seeded weights, `OUTPUTS=a,b`
to drain more than the last op's output).

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
  more buffer descriptors than a memtile channel has.
- A conv whose weights do not fit beside its rows streams them from a
  memtile one output-channel chunk at a time, each chunk followed by its
  bias.
- C2PSA attention runs one worker per head; a memtile splits each qkv row
  into the heads' queries, keys and values.
- The Detect postprocess (best class per anchor, top 300 anchors, top 300
  anchor-class pairs, box decode) runs on one worker, `kernels/yolo_detect.cc`.
- `yolo26n.py` runs the network as a sequence of such segments, each its own
  device configuration in one full ELF, handing tensors on through a DDR
  scratch buffer, in one dispatch.

## Accuracy

mAP50-95 on coco128: FP32 0.474, XINT8 0.443. The XINT8 model is calibrated
on the same 128 images, so its number is optimistic.

On the NPU, the whole model's detections match the oracle bit for bit on
all 128 coco128 images, and ONNX Runtime's on 126. The oracle agrees with
ONNX Runtime on those 126 in all 169 tensors. In the other 2, ONNX
Runtime's float32 softmax rounds an attention probability (4 of 82 million)
to the other side of a rounding boundary from exact arithmetic; neither
changes a detection above 0.25 confidence.

## Performance

On Strix Halo (NPU2), one image takes 22.1 ms (45 fps) at batch 1 and
12.6 ms per image (79 fps) at batch 8 (`python yolo26n.py --n 8 --batch
8 --time`), in nine device configurations of one full ELF. Each configuration costs about 1.1 ms to set up, which a batch
shares. Compiling takes about 10 s.
