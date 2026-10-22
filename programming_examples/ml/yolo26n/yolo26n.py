#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""YOLO26n on the NPU, from a letterboxed image to its (300, 6) detections, in one dispatch.

    python yolo26n.py --images DIR --n 8          # check images against the oracle
    python yolo26n.py --synthetic 7 --n 2         # seeded weights and images, no checkpoint
    python yolo26n.py --images DIR --n 8 --time   # and time it
    python yolo26n.py --n 8 --batch 8 --time      # eight images per dispatch

With real images, the device's detections are also scored against the
images' YOLO labels (``--labels``) as mAP@0.5.

The whole network is one device configuration: every operator's rows stream
from worker to worker on chip. Every image's detections are compared with
the oracle bit for bit; with the quantized ONNX at hand, also with ONNX
Runtime's.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, InOut, Out, Program, Runtime
from aie.iron.device import NPU2
from aie.utils.benchmark import print_benchmark, run_iters
from coco_map import Labels, MeanAP50
from yolo26n_design import SegmentDesign
from yolo26n_numpy import Oracle
from yolo_params import DATA_DIR, WEIGHTS_DIR, Params, synthetic_params
from yolo_spec import IMAGE_SIZE, MAX_DETECTIONS, Network

IMAGE_WORDS = IMAGE_SIZE * IMAGE_SIZE * 3 // 4
DETECTION_WORDS = 6 * MAX_DETECTIONS


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
    network = Network.yolo26n()
    plan = Plan(network, Params.load(Path(params_path)))
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
                if tensor.name == network.ops[-1].dst.name:
                    return det, batch * DETECTION_WORDS, i * DETECTION_WORDS
                raise ValueError(f"no host buffer holds {tensor.name}")

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
    parser.add_argument(
        "--labels",
        type=Path,
        help="YOLO label files for --images (default: its labels/ sibling, "
        "images/train2017 -> labels/train2017); scored as mAP@0.5 when present",
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
        "--ahead", type=int, default=3, help="images' fills issued ahead of a wait"
    )
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
    # The design targets NPU2 alone. Its host-side plan lays weights out as
    # the bound device's kernels load them, so bind it before planning; a
    # compile-only run then needs no device either.
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
    scratch = iron.zeros(args.batch * plan.words, dtype=np.int32)
    detections = iron.zeros(args.batch * DETECTION_WORDS, dtype=np.int32)
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
        pixels = np.concatenate([image.reshape(-1) for _, image, _ in group])
        yolo26n(
            iron.tensor(pixels.view(np.int32)), scratch, detections, weights, **kwargs
        )
        found = detections.numpy().reshape(args.batch, MAX_DETECTIONS, 6)
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
        pixels = np.concatenate(
            [image.reshape(-1) for _, image, _ in images[: args.batch]]
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
