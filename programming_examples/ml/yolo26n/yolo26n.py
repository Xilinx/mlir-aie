#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""YOLO26n on the NPU, from a letterboxed image to its (300, 6) detections, in one dispatch.

    python yolo26n.py --images DIR --n 8          # check images against the oracle
    python yolo26n.py --synthetic 7 --n 2         # seeded weights and images, no checkpoint
    python yolo26n.py --images DIR --n 8 --time   # and time it

The network runs as a sequence of segments, each a device configuration of
its own in one full ELF, handing tensors on through a DDR scratch buffer.
Every image's detections are compared with the oracle bit for bit; with
the quantized ONNX at hand, also with ONNX Runtime's.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
import onnxruntime as ort
from aie.iron import CompileTime, DeviceConfiguration, In, InOut, Out, Program, Runtime
from aie.iron.device import NPU2

from gen_yolo_data import DATA_DIR, WEIGHTS_DIR, Params, Synthesizer
from letterbox import letterbox
from yolo26n_design import SegmentDesign
from yolo26n_numpy import Oracle
from yolo_spec import (
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
    (0, 21),
    (21, 36),
    (36, 37),
    (37, 52),
    (52, 59),
    (59, 73),
    (73, 90),
    (90, 107),
    (107, 124),
    (124, 143),
    (143, 151),
    (151, 159),
    (159, 167),
    (167, 168),
)
IMAGE = Tensor("images", 3, 640, 640)


class Plan:
    """Which tensors each segment reads and hands on, and where each waits in the scratch buffer."""

    def __init__(self, network: Network) -> None:
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
    *,
    params_path: CompileTime[str],
    params_digest: CompileTime[str],
):
    """Every segment as a device configuration, and an entry sequence running them in turn."""
    del params_digest  # keys the compile cache on the parameter file's contents
    network = Network.yolo26n()
    params = Params.load(Path(params_path))
    plan = Plan(network)
    image_words = 640 * 640 * 3 // 4
    det_words = 6 * 300
    image_ty = np.ndarray[(image_words,), np.dtype[np.int32]]
    scratch_ty = np.ndarray[(plan.words,), np.dtype[np.int32]]
    det_ty = np.ndarray[(det_words,), np.dtype[np.int32]]
    configurations, runtimes = [], []
    for k, (a, b) in enumerate(SEGMENTS):
        design = SegmentDesign(network, params, network.ops[a:b], plan.outputs[k])
        if {t.name for t in design.inputs} != {t.name for t in plan.inputs[k]}:
            raise ValueError(
                f"segment {k}: the design reads {design.inputs}, the plan {plan.inputs[k]}"
            )

        def sequence(img, scr, det, *handles, design=design):
            def locate(tensor):
                if tensor.name == IMAGE.name:
                    return img, image_words, 0
                if tensor.name in plan.offset:
                    return scr, plan.words, plan.offset[tensor.name]
                return det, det_words, 0

            design.transfer(handles, locate)

        runtime = Runtime(
            sequence,
            [image_ty, scratch_ty, det_ty, *design.handles()],
            name=f"segment{k}",
        )
        configurations.append(
            DeviceConfiguration(
                f"segment{k}", NPU2(), workers=design.workers, runtimes=[runtime]
            )
        )
        runtimes.append(runtime)

    def run_all(img, scr, det):
        for configuration, runtime in zip(configurations, runtimes):
            with configuration.configure():
                runtime.call(img, scr, det)

    entry = Runtime(run_all, [image_ty, scratch_ty, det_ty])
    main = DeviceConfiguration("main", NPU2(), runtimes=[entry])
    return Program.compose([main, *configurations], entry=entry).resolve_program()


def onnx_detections(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    x = (image.transpose(2, 0, 1)[None] / np.float32(255)).astype(np.float32)
    return session.run(None, {"images": x})[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--images", type=Path, default=WEIGHTS_DIR / "coco128" / "images" / "train2017"
    )
    parser.add_argument("--n", type=int, default=4, help="images to run")
    parser.add_argument("--params", type=Path, default=DATA_DIR / "yolo26n_params.npz")
    parser.add_argument("--onnx", type=Path, default=WEIGHTS_DIR / "yolo26n_xint8.onnx")
    parser.add_argument(
        "--synthetic", type=int, metavar="SEED", help="seeded weights and random images"
    )
    parser.add_argument("--time", action="store_true", help="also time the dispatch")
    parser.add_argument("--n-iters", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument(
        "--compile-only", action="store_true", help="build the ELF and stop"
    )
    args = parser.parse_args()

    network = Network.yolo26n()
    if args.synthetic is not None:
        params_path = DATA_DIR / f"yolo26n_params_synthetic{args.synthetic}.npz"
        if not params_path.exists():
            params_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(params_path, **Synthesizer(args.synthetic).run(network))
        rng = np.random.default_rng(args.synthetic)
        images = [
            (f"random{i}", rng.integers(0, 256, (640, 640, 3), dtype=np.uint8))
            for i in range(args.n)
        ]
    else:
        params_path = args.params
        images = [
            (p.name, letterbox(p)) for p in sorted(args.images.glob("*.jpg"))[: args.n]
        ]
    params = Params.load(params_path)
    oracle = Oracle(params, network)
    plan = Plan(network)
    kwargs = dict(
        params_path=str(params_path),
        params_digest=hashlib.sha1(params_path.read_bytes()).hexdigest(),
    )
    scratch = iron.zeros(plan.words, dtype=np.int32)
    detections = iron.zeros(6 * 300, dtype=np.int32)
    session = (
        ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        if args.synthetic is None and args.onnx.exists()
        else None
    )
    if args.compile_only:
        yolo26n(
            iron.tensor(images[0][1].reshape(-1).view(np.int32)),
            scratch,
            detections,
            **kwargs,
        )
        return

    passed = matched = 0
    for name, image in images:
        yolo26n(
            iron.tensor(image.reshape(-1).view(np.int32)), scratch, detections, **kwargs
        )
        mine = detections.numpy().reshape(300, 6)
        want = oracle.run(image)[network.ops[-1].dst.name].view(np.int32)
        bad = int((mine != want).sum())
        passed += bad == 0
        print(f"image {name}: {'PASS' if bad == 0 else 'FAIL'} ({bad}/1800 differ)")
        if session is not None:
            matched += np.array_equal(
                mine, onnx_detections(session, image).view(np.int32)
            )
    print(f"yolo26n: {passed}/{len(images)} images PASS")
    if session is not None:
        print(f"ort: {matched}/{len(images)} images match onnxruntime")

    if args.time:
        tensor = iron.tensor(images[0][1].reshape(-1).view(np.int32))
        for _ in range(args.n_warmup):
            yolo26n(tensor, scratch, detections, **kwargs)
        samples = []
        for _ in range(args.n_iters):
            start = time.perf_counter()
            yolo26n(tensor, scratch, detections, **kwargs)
            samples.append(time.perf_counter() - start)
        us = statistics.median(samples) * 1e6
        print(f"per-image: {us:.1f} us ({1e6 / us:.2f} fps) at batch 1")
    sys.exit(0 if passed == len(images) else 1)


if __name__ == "__main__":
    main()
