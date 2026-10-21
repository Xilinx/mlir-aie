#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""YOLO26n on the NPU, from a letterboxed image to its (300, 6) detections, in one dispatch.

    python yolo26n.py --images DIR --n 8          # check images against the oracle
    python yolo26n.py --synthetic 7 --n 2         # seeded weights and images, no checkpoint
    python yolo26n.py --images DIR --n 8 --time   # and time it
    python yolo26n.py --n 8 --batch 8 --time      # eight images per dispatch

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
from aie.iron import CompileTime, DeviceConfiguration, In, InOut, Out, Program, Runtime
from aie.iron.device import NPU2
from aie.utils.benchmark import print_benchmark, run_iters
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

# Each run fits one array: a block, or a few, as they were brought up.
SEGMENTS = (
    (0, 59),
    (59, 90),
    (90, 143),
    (143, 168),
)
IMAGE = Tensor("images", 3, IMAGE_SIZE, IMAGE_SIZE)
IMAGE_WORDS = IMAGE_SIZE * IMAGE_SIZE * 3 // 4
DETECTION_WORDS = 6 * MAX_DETECTIONS


class Plan:
    """Which tensors each segment reads and hands on, and where each waits in the scratch buffer."""

    def __init__(self, network: Network, params: Params) -> None:
        self.network = network
        self.producer = {op.dst.name: op for op in network.ops}
        self.inputs = [self.reads(network.ops[a:b]) for a, b in SEGMENTS]
        detections = network.ops[-1].dst
        self.outputs = []
        for k, (a, b) in enumerate(SEGMENTS):
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
        for k, (a, b) in enumerate(SEGMENTS):
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
    batch: CompileTime[int],
    params_path: CompileTime[str],
    params_digest: CompileTime[str],
):
    """Every segment as a device configuration, and an entry sequence running them in turn.

    Each configuration runs all `batch` images before the next is configured.
    """
    del params_digest  # keys the compile cache on the parameter file's contents
    network = Network.yolo26n()
    params = Params.load(Path(params_path))
    plan = Plan(network, params)
    image_ty = np.ndarray[(batch * IMAGE_WORDS,), np.dtype[np.int32]]
    scratch_ty = np.ndarray[(batch * plan.words,), np.dtype[np.int32]]
    det_ty = np.ndarray[(batch * DETECTION_WORDS,), np.dtype[np.int32]]
    weights_ty = np.ndarray[(plan.weights.size // 4,), np.dtype[np.int32]]
    configurations, runtimes = [], []
    for k, (a, b) in enumerate(SEGMENTS):
        design = SegmentDesign(network, params, network.ops[a:b], plan.outputs[k])
        if {t.name for t in design.inputs} != {t.name for t in plan.inputs[k]}:
            raise ValueError(
                f"segment {k}: the design reads {design.inputs}, the plan {plan.inputs[k]}"
            )

        def sequence(img, scr, det, wts, *handles, design=design):
            for i in range(batch):

                def locate(tensor, i=i):
                    if tensor.name == IMAGE.name:
                        return img, batch * IMAGE_WORDS, i * IMAGE_WORDS
                    if tensor.name in plan.weight_offset:
                        return (
                            wts,
                            plan.weights.size // 4,
                            plan.weight_offset[tensor.name],
                        )
                    if tensor.name in plan.offset:
                        return (
                            scr,
                            batch * plan.words,
                            i * plan.words + plan.offset[tensor.name],
                        )
                    if tensor.name == network.ops[-1].dst.name:
                        return det, batch * DETECTION_WORDS, i * DETECTION_WORDS
                    raise ValueError(f"no host buffer holds {tensor.name}")

                design.transfer(handles, locate)

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
    parser.add_argument("--n", type=int, default=4, help="images to run")
    parser.add_argument(
        "--batch", type=int, default=1, help="images per dispatch (divides --n)"
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
    if args.batch < 1 or not args.compile_only and args.n % args.batch:
        parser.error(f"--batch {args.batch} does not divide --n {args.n}")
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
        batch=args.batch,
        params_path=str(params_path),
        params_digest=hashlib.sha1(params_path.read_bytes()).hexdigest(),
    )
    if args.compile_only:
        yolo26n.specialize(
            image=np.ndarray[(args.batch * IMAGE_WORDS,), np.dtype[np.int32]],
            scratch=np.ndarray[(args.batch * plan.words,), np.dtype[np.int32]],
            detections=np.ndarray[(args.batch * DETECTION_WORDS,), np.dtype[np.int32]],
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
            )
            for i in range(args.n)
        ]
    else:
        # Optional: only real images need OpenCV; a weight-free run goes without.
        from letterbox import letterbox

        images = [(p.name, letterbox(p)) for p in paths]
    oracle = Oracle(params, network)
    scratch = iron.zeros(args.batch * plan.words, dtype=np.int32)
    detections = iron.zeros(args.batch * DETECTION_WORDS, dtype=np.int32)
    weights = iron.tensor(plan.weights.view(np.int32))
    session = None
    if args.synthetic is None and args.onnx.exists():
        # Optional: ONNX Runtime only serves the comparison with the ONNX graph.
        import onnxruntime as ort  # pyright: ignore[reportMissingImports]

        session = ort.InferenceSession(
            str(args.onnx), providers=["CPUExecutionProvider"]
        )

    passed = matched = 0
    first: list[np.ndarray] = []
    for start in range(0, len(images), args.batch):
        group = images[start : start + args.batch]
        pixels = np.concatenate([image.reshape(-1) for _, image in group])
        yolo26n(
            iron.tensor(pixels.view(np.int32)), scratch, detections, weights, **kwargs
        )
        found = detections.numpy().reshape(args.batch, MAX_DETECTIONS, 6)
        for (name, image), mine in zip(group, found):
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
    print(f"yolo26n: {passed}/{len(images)} images PASS")
    if session is not None:
        print(f"ort: {matched}/{len(images)} images match onnxruntime")
    failed = passed != len(images)

    if args.time:
        pixels = np.concatenate(
            [image.reshape(-1) for _, image in images[: args.batch]]
        )
        tensor = iron.tensor(pixels.view(np.int32))
        bench = run_iters(
            yolo26n,
            tensor,
            scratch,
            detections,
            weights,
            warmup=args.n_warmup,
            iters=args.n_iters,
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
        timed = detections.numpy().reshape(args.batch, MAX_DETECTIONS, 6)
        same = all(np.array_equal(t, w) for t, w in zip(timed, first))
        print(f"timed dispatch: {'PASS' if same else 'FAIL'}")
        failed |= not same
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
