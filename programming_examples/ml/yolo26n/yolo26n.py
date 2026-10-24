#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""YOLO26n on the NPU, from letterboxed images to their (300, 6) detections, up to
--capacity images a dispatch.

    python yolo26n.py --images DIR --n 8          # check images against the oracle
    python yolo26n.py --synthetic 7 --n 2         # seeded weights and images, no checkpoint
    python yolo26n.py --images DIR --n 8 --time   # and time it
    python yolo26n.py --n 8 --batch 8 --time      # eight images per dispatch

A build holds --capacity images' buffers (CompileTime), and each image's
transfers are gated by a ScratchpadParameter (image{i}), so a dispatch runs
any number of them, --batch, chosen when it is dispatched; --n images run as
dispatches of --batch, the last of the rest.

With real images, the device's detections are also scored against the
images' YOLO labels (``--labels``) as mAP@0.5.

The network runs as a sequence of segments, each a device configuration of
its own in one full ELF, handing tensors on through a DDR scratch buffer.
Every image's detections are compared with the oracle bit for bit; with
the quantized ONNX at hand, also with ONNX Runtime's.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import (
    CompileTime,
    DeviceConfiguration,
    In,
    InOut,
    Out,
    Program,
    Runtime,
    ScratchpadParameter,
)
from aie.iron.device import NPU2
from aie.utils.benchmark import print_benchmark, run_iters
from coco_map import Labels, MeanAP50
from yolo26n_design import SegmentDesign
from yolo26n_numpy import Oracle
from yolo_params import DATA_DIR, WEIGHTS_DIR, Params, synthetic_params
from yolo_spec import (
    IMAGE_SIZE,
    MAX_DETECTIONS,
    Add,
    Attention,
    Concat,
    Conv,
    DetectPost,
    InputQuant,
    MaxPool,
    Network,
    Op,
    Slice,
    Tensor,
    Upsample,
)

# Each run fits one array: a block, or a few, as they were brought up. Each
# is named by its first and last op, and together they cover the network.
SEGMENTS = (
    ("images", "/model.9/Add"),
    ("/model.10/cv1/conv/Conv", "/model.13/cv2/conv/Conv"),
    ("/model.14/Resize", "/model.22/cv2/conv/Conv"),
    ("/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv", "/model.23"),
)
IMAGE = Tensor("images", 3, IMAGE_SIZE, IMAGE_SIZE)
IMAGE_WORDS = IMAGE_SIZE * IMAGE_SIZE * 3 // 4
DETECTION_WORDS = 6 * MAX_DETECTIONS


class Plan:
    """Which tensors each segment reads and hands on, and where each waits in the scratch buffer.

    Attributes:
        spans: Each segment's `(start, stop)` range of `network.ops`.

    Raises:
        KeyError: If a segment names an op the network lacks.
        ValueError: If the segments do not cover the network, each starting
            where the one before it stops.
    """

    def __init__(self, network: Network, params: Params) -> None:
        self.network = network
        self.spans = [network.span(first, last) for first, last in SEGMENTS]
        starts = [0] + [stop for _, stop in self.spans]
        if [a for a, _ in self.spans] != starts[:-1] or starts[-1] != len(network.ops):
            raise ValueError(f"SEGMENTS {self.spans} do not cover the network's ops")
        self.producer = {op.dst.name: op for op in network.ops}
        self.inputs = [self.reads(network.ops[a:b]) for a, b in self.spans]
        detections = network.ops[-1].dst
        self.outputs = []
        for k, (a, b) in enumerate(self.spans):
            made = {op.dst.name for op in network.ops[a:b]}
            later = {t.name: t for inputs in self.inputs[k + 1 :] for t in inputs}
            outputs = [later[n] for n in later if n in made]
            if detections.name in made:
                outputs.append(detections)
            self.outputs.append(outputs)
        self.offset: dict[str, int] = {}
        self.words = 0
        for outputs in self.outputs:
            for t in outputs:
                if t is not detections:
                    self.offset[t.name] = self.words
                    self.words += t.row_bytes * t.height // 4
        self.weight_offset: dict[str, int] = {}
        weights: list[np.ndarray] = []
        for k, (a, b) in enumerate(self.spans):
            design = SegmentDesign(network, params, network.ops[a:b], self.outputs[k])
            for tensor in design.in_ddr:
                self.offset[tensor.name] = self.words
                self.words += tensor.row_bytes * tensor.height // 4
            for tensor, packed in design.loads:
                self.weight_offset[tensor.name] = sum(w.size for w in weights) // 4
                weights.append(packed)
        # Every segment's streamed weights, in the host buffer they are read from.
        self.weights = np.concatenate(weights) if weights else np.zeros(4, np.int8)

    def reads(self, ops: list[Op]) -> list[Tensor]:
        """The tensors a run of operators reads from outside it, as streams."""
        local = {op.dst.name for op in ops} | {
            op.values.name for op in ops if isinstance(op, Attention)
        }
        found: dict[str, Tensor] = {}

        def visit(tensor: Tensor) -> None:
            if tensor.name in local:
                return
            op = self.producer.get(tensor.name)
            if isinstance(op, (Slice, Upsample)):
                visit(op.src)
            elif isinstance(op, Concat):
                for src in op.srcs:
                    visit(src)
            else:
                found.setdefault(tensor.name, tensor)

        for op in ops:
            match op:
                case InputQuant():
                    found.setdefault(IMAGE.name, IMAGE)
                case Conv() | MaxPool() | Slice() | Upsample():
                    visit(op.src)
                case Add():
                    visit(op.a)
                    visit(op.b)
                case Concat():
                    for src in op.srcs:
                        visit(src)
                case Attention():
                    visit(op.qkv)
                case DetectPost():
                    for t in (*op.scores, *op.boxes):
                        visit(t)
        return list(found.values())


@iron.jit
def yolo26n(
    image: In,
    scratch: InOut,
    detections: Out,
    weights: In,
    *,
    capacity: CompileTime[int],
    params_path: CompileTime[str],
    params_digest: CompileTime[str],
):
    """Every segment as a device configuration, and an entry sequence running them in turn.

    Each configuration runs the dispatch's images before the next is
    configured: image i of the `capacity` the buffers hold when its
    ScratchpadParameter image{i} is 1.
    """
    del params_digest  # keys the compile cache on the parameter file's contents
    network = Network.yolo26n()
    params = Params.load(Path(params_path))
    plan = Plan(network, params)
    image_ty = np.ndarray[(capacity * IMAGE_WORDS,), np.dtype[np.int32]]
    scratch_ty = np.ndarray[(capacity * plan.words,), np.dtype[np.int32]]
    det_ty = np.ndarray[(capacity * DETECTION_WORDS,), np.dtype[np.int32]]
    images_on = [ScratchpadParameter(f"image{i}", np.int32) for i in range(capacity)]
    weights_ty = np.ndarray[(plan.weights.size // 4,), np.dtype[np.int32]]
    configurations, runtimes = [], []
    for k, (a, b) in enumerate(plan.spans):
        design = SegmentDesign(network, params, network.ops[a:b], plan.outputs[k])
        if {t.name for t in design.inputs} != {t.name for t in plan.inputs[k]}:
            raise ValueError(
                f"segment {k}: the design reads {design.inputs}, the plan {plan.inputs[k]}"
            )

        def sequence(img, scr, det, wts, *handles, design=design):
            for i in range(capacity):

                def locate(tensor, i=i):
                    if tensor.name == IMAGE.name:
                        return img, capacity * IMAGE_WORDS, i * IMAGE_WORDS
                    if tensor.name in plan.weight_offset:
                        return (
                            wts,
                            plan.weights.size // 4,
                            plan.weight_offset[tensor.name],
                        )
                    if tensor.name in plan.offset:
                        return (
                            scr,
                            capacity * plan.words,
                            i * plan.words + plan.offset[tensor.name],
                        )
                    if tensor.name == network.ops[-1].dst.name:
                        return det, capacity * DETECTION_WORDS, i * DETECTION_WORDS
                    raise ValueError(f"no host buffer holds {tensor.name}")

                design.transfer(handles, locate, gate=images_on[i])

        runtime = Runtime(
            sequence,
            [image_ty, scratch_ty, det_ty, weights_ty, *design.handles()],
            name=f"segment{k}",
        )
        configurations.append(
            DeviceConfiguration(
                f"segment{k}", NPU2(), workers=design.workers, runtimes=[runtime]
            )
        )
        runtimes.append(runtime)

    def run_all(img, scr, det, wts):
        for configuration, runtime in zip(configurations, runtimes):
            with configuration.configure():
                runtime.call(img, scr, det, wts)

    entry = Runtime(run_all, [image_ty, scratch_ty, det_ty, weights_ty])
    main = DeviceConfiguration("main", NPU2(), runtimes=[entry])
    return Program.compose([main, *configurations], entry=entry).resolve_program()


def onnx_detections(session, image: np.ndarray) -> np.ndarray:
    x = (image.transpose(2, 0, 1)[None] / np.float32(255)).astype(np.float32)
    return np.asarray(session.run(None, {"images": x})[0])[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--images", type=Path, default=WEIGHTS_DIR / "coco128" / "images" / "train2017"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="YOLO label files for --images (default: its labels/ sibling, "
        "images/train2017 -> labels/train2017); scored as mAP@0.5 when present",
    )
    parser.add_argument("--n", type=int, default=4, help="images to run")
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="images per dispatch, at most --capacity; the last dispatch runs the rest",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=8,
        help="images a dispatch can run: the build's buffers (CompileTime)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        help="parameter file (default data/yolo26n_params.npz, or with --synthetic "
        "data/yolo26n_params_synthetic<SEED>.npz, made if missing)",
    )
    parser.add_argument("--onnx", type=Path, default=WEIGHTS_DIR / "yolo26n_xint8.onnx")
    parser.add_argument(
        "--synthetic", type=int, metavar="SEED", help="seeded weights and random images"
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="also time the dispatch; the per-image time is the fastest call",
    )
    parser.add_argument("--n-iters", type=int, default=20, help="timed calls (>= 1)")
    # The NPU takes about ten calls after a load to reach its steady latency.
    parser.add_argument("--n-warmup", type=int, default=10, help="untimed calls first")
    parser.add_argument(
        "--compile-only", action="store_true", help="build the ELF and stop"
    )
    args = parser.parse_args()
    if not 1 <= args.batch <= args.capacity:
        parser.error(f"--batch {args.batch} is not 1..--capacity {args.capacity}")
    if args.n_iters < 1 or args.n_warmup < 0:
        parser.error("--n-iters must be at least 1 and --n-warmup at least 0")
    paths = sorted(args.images.glob("*.jpg"))[: args.n]
    if args.synthetic is None and not args.compile_only and len(paths) < args.n:
        parser.error(f"{args.images} holds {len(paths)} .jpg images, not --n {args.n}")

    network = Network.yolo26n()
    if args.synthetic is not None:
        params_path = synthetic_params(
            args.synthetic,
            args.params or DATA_DIR / f"yolo26n_params_synthetic{args.synthetic}.npz",
        )
    else:
        params_path = args.params or DATA_DIR / "yolo26n_params.npz"
    params = Params.load(params_path)
    plan = Plan(network, params)
    kwargs = dict(
        capacity=args.capacity,
        params_path=str(params_path),
        params_digest=hashlib.sha1(params_path.read_bytes()).hexdigest(),
    )
    if args.compile_only:
        yolo26n.specialize(
            image=np.ndarray[(args.capacity * IMAGE_WORDS,), np.dtype[np.int32]],
            scratch=np.ndarray[(args.capacity * plan.words,), np.dtype[np.int32]],
            detections=np.ndarray[
                (args.capacity * DETECTION_WORDS,), np.dtype[np.int32]
            ],
            weights=np.ndarray[(plan.weights.size // 4,), np.dtype[np.int32]],
            **kwargs,
        ).compile()
        return

    if args.synthetic is not None:
        rng = np.random.default_rng(args.synthetic)
        images = [
            (
                f"random{i}",
                rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                None,
            )
            for i in range(args.n)
        ]
    else:
        # Optional: only real images need OpenCV; a weight-free run goes without.
        from letterbox import letterbox

        labels = args.labels or args.images.parent.parent / "labels" / args.images.name
        images = []
        for path in paths:
            image, fit = letterbox(path)
            truth = (
                Labels.read(
                    labels / f"{path.stem}.txt",
                    fit.height,
                    fit.width,
                    fit.ratio,
                    fit.left,
                    fit.top,
                )
                if labels.is_dir()
                else None
            )
            images.append((path.name, image, truth))
    oracle = Oracle(params, network)
    scratch = iron.zeros(args.capacity * plan.words, dtype=np.int32)
    detections = iron.zeros(args.capacity * DETECTION_WORDS, dtype=np.int32)

    def dispatch_args(group):
        """A dispatch's image buffer and its images' parameters: the group's
        pixels in the first slots, the rest of the buffer unread."""
        pixels = np.zeros(args.capacity * IMAGE_WORDS, np.int32)
        for i, (_, image, _) in enumerate(group):
            pixels[i * IMAGE_WORDS : (i + 1) * IMAGE_WORDS] = image.reshape(-1).view(
                np.int32
            )
        on = {f"image{i}": int(i < len(group)) for i in range(args.capacity)}
        return iron.tensor(pixels), on

    weights = iron.tensor(plan.weights.view(np.int32))
    session = None
    if args.synthetic is None and args.onnx.exists():
        # Optional: ONNX Runtime only serves the comparison with the ONNX graph.
        from gen_yolo_data import ort_session

        session = ort_session(str(args.onnx))

    passed = matched = 0
    first: list[np.ndarray] = []
    scored = MeanAP50()
    for start in range(0, len(images), args.batch):
        group = images[start : start + args.batch]
        image_t, on = dispatch_args(group)
        yolo26n(image_t, scratch, detections, weights, parameters=on, **kwargs)
        found = detections.numpy().reshape(args.capacity, MAX_DETECTIONS, 6)
        for (name, image, truth), mine in zip(group, found):
            want = oracle.run(image)[network.ops[-1].dst.name].view(np.int32)
            if start == 0:
                first.append(want)
            bad = int((mine != want).sum())
            passed += bad == 0
            print(
                f"image {name}: {'PASS' if bad == 0 else 'FAIL'} ({bad}/{DETECTION_WORDS} differ)"
            )
            if session is not None:
                matched += np.array_equal(
                    mine, onnx_detections(session, image).view(np.int32)
                )
            if truth is not None:
                scored.add(mine.view(np.float32), truth)
    print(f"yolo26n: {passed}/{len(images)} images PASS")
    if session is not None:
        print(f"ort: {matched}/{len(images)} images match onnxruntime")
    if scored.images:
        print(
            f"map50: {scored.value():.4f} over {scored.images} images "
            f"({scored.objects} labelled objects)"
        )
    failed = passed != len(images)

    if args.time:
        tensor, on = dispatch_args(images[: args.batch])
        bench = run_iters(
            yolo26n,
            tensor,
            scratch,
            detections,
            weights,
            warmup=args.n_warmup,
            iters=args.n_iters,
            parameters=on,
            **kwargs,
        )
        print_benchmark(bench)
        e2e = bench.e2e
        us = e2e.min_us / args.batch
        print(
            f"per-image: {us:.1f} us ({1e6 / us:.2f} fps) at batch {args.batch}, "
            f"fastest of {e2e.n} after {args.n_warmup} warmups; "
            f"median {e2e.median_us / args.batch:.1f} us, "
            f"MAD {e2e.mad_us / args.batch:.1f} us"
        )
        # The same dispatch with no image on: its segments' configurations
        # and the dispatch itself, which every dispatch pays once.
        empty = run_iters(
            yolo26n,
            tensor,
            scratch,
            detections,
            weights,
            warmup=args.n_warmup,
            iters=args.n_iters,
            parameters={name: 0 for name in on},
            **kwargs,
        ).e2e
        print(
            f"empty dispatch ({len(plan.spans)} segment configurations, no image): "
            f"{empty.min_us:.1f} us, median {empty.median_us:.1f} us"
        )
        timed = detections.numpy().reshape(args.capacity, MAX_DETECTIONS, 6)
        same = all(np.array_equal(t, w) for t, w in zip(timed, first))
        print(f"timed dispatch: {'PASS' if same else 'FAIL'}")
        failed |= not same
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
