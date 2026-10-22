#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""YOLO26n on the NPU, from a letterboxed image to its (300, 6) detections, in one dispatch.

    python yolo26n.py --images DIR --n 8          # check images against the oracle
    python yolo26n.py --synthetic 7 --n 2         # seeded weights and images, no checkpoint
    python yolo26n.py --images DIR --n 8 --time   # and time it
    python yolo26n.py --n 8 --batch 8 --time      # eight images per dispatch

The whole network is one device configuration: every operator's rows stream
from worker to worker on chip. Every image's detections are compared with
the oracle bit for bit; with the quantized ONNX at hand, also with ONNX
Runtime's.
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
from aie.iron import CompileTime, In, InOut, Out, Program, Runtime
from aie.iron.device import NPU2
from yolo26n_design import SegmentDesign
from yolo26n_numpy import Oracle
from yolo_params import DATA_DIR, WEIGHTS_DIR, Params, synthetic_params
from yolo_spec import Network

IMAGE_WORDS = 640 * 640 * 3 // 4
DETECTION_WORDS = 6 * 300


class Plan:
    """The design, and where its DDR traffic waits: the weights it streams and the rows it stages."""

    def __init__(self, network: Network, params: Params) -> None:
        self.design = SegmentDesign.planned(
            network, params, network.ops, [network.ops[-1].dst]
        )
        self.offset: dict[str, int] = {}
        self.words = 0
        for _, tensor in self.design.staged:
            self.offset[tensor.name] = self.words
            self.words += tensor.row_bytes * tensor.height // 4
        self.weight_offset: dict[str, int] = {}
        at = 0
        for tensor, packed in self.design.loads:
            self.weight_offset[tensor.name] = at
            at += packed.size // 4
        self.weights = (
            np.concatenate([packed for _, packed in self.design.loads])
            if self.design.loads
            else np.zeros(4, np.int8)
        )


@iron.jit(aiecc_flags=["--placer=sa_placer", "--sa-seed=1", "--sa-effort=3"])
def yolo26n(
    image: In,
    scratch: InOut,
    detections: Out,
    weights: In,
    *,
    batch: CompileTime[int],
    params_path: CompileTime[str],
    params_digest: CompileTime[str],
    ahead: CompileTime[int] = 3,
):
    """The network's workers, and a sequence running the `batch` images in turn, `ahead` images' fills ahead."""
    del params_digest  # keys the compile cache on the parameter file's contents
    plan = Plan(Network.yolo26n(), Params.load(Path(params_path)))
    design = plan.design
    image_ty = np.ndarray[(batch * IMAGE_WORDS,), np.dtype[np.int32]]
    scratch_ty = np.ndarray[(batch * plan.words,), np.dtype[np.int32]]
    det_ty = np.ndarray[(batch * DETECTION_WORDS,), np.dtype[np.int32]]
    weights_ty = np.ndarray[(plan.weights.size // 4,), np.dtype[np.int32]]

    def sequence(img, scr, det, wts, *handles):
        def locate_for(i):
            def locate(tensor):
                if tensor.name == "images":
                    return img, batch * IMAGE_WORDS, i * IMAGE_WORDS
                if tensor.name in plan.weight_offset:
                    return wts, plan.weights.size // 4, plan.weight_offset[tensor.name]
                if tensor.name in plan.offset:
                    return (
                        scr,
                        batch * plan.words,
                        i * plan.words + plan.offset[tensor.name],
                    )
                return det, batch * DETECTION_WORDS, i * DETECTION_WORDS

            return locate

        design.transfer(handles, [locate_for(i) for i in range(batch)], ahead)

    rt = Runtime(
        sequence, [image_ty, scratch_ty, det_ty, weights_ty, *design.handles()]
    )
    return Program(NPU2(), rt, workers=design.workers).resolve_program()


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
    parser.add_argument("--time", action="store_true", help="also time the dispatch")
    parser.add_argument("--n-iters", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument(
        "--ahead", type=int, default=3, help="images' fills issued ahead of a wait"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="build the ELF and stop"
    )
    args = parser.parse_args()
    if args.batch < 1 or not args.compile_only and args.n % args.batch:
        parser.error(f"--batch {args.batch} does not divide --n {args.n}")
    # The design targets NPU2 alone: a compile-only run builds for it without
    # asking the runtime, so it works on a host without the device.
    if args.compile_only:
        iron.set_current_device(NPU2())

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
        ahead=args.ahead,
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
            (f"random{i}", rng.integers(0, 256, (640, 640, 3), dtype=np.uint8))
            for i in range(args.n)
        ]
    else:
        # Optional: only real images need OpenCV; a weight-free run goes without.
        from letterbox import letterbox

        images = [
            (p.name, letterbox(p)) for p in sorted(args.images.glob("*.jpg"))[: args.n]
        ]
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
    for start in range(0, len(images), args.batch):
        group = images[start : start + args.batch]
        pixels = np.concatenate([image.reshape(-1) for _, image in group])
        yolo26n(
            iron.tensor(pixels.view(np.int32)), scratch, detections, weights, **kwargs
        )
        found = detections.numpy().reshape(args.batch, 300, 6)
        for (name, image), mine in zip(group, found):
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
        pixels = np.concatenate(
            [image.reshape(-1) for _, image in images[: args.batch]]
        )
        tensor = iron.tensor(pixels.view(np.int32))
        for _ in range(args.n_warmup):
            yolo26n(tensor, scratch, detections, weights, **kwargs)
        samples = []
        for _ in range(args.n_iters):
            start = time.perf_counter()
            yolo26n(tensor, scratch, detections, weights, **kwargs)
            samples.append(time.perf_counter() - start)
        us = statistics.median(samples) * 1e6 / args.batch
        print(f"per-image: {us:.1f} us ({1e6 / us:.2f} fps) at batch {args.batch}")
    sys.exit(0 if passed == len(images) else 1)


if __name__ == "__main__":
    main()
