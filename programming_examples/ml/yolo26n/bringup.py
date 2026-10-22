#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Run a contiguous run of yolo26n operators on the NPU and check it against the oracle.

    python bringup.py --ops 0:11                 # the stem and model.2's C3k2
    python bringup.py --ops 0:11 --synthetic 7   # the same on seeded synthetic weights

A run starts and ends on the boundaries of the fused runs (`FUSED`), as the
whole network is built from them.

The run's inputs come from the oracle on one letterboxed image (a seeded
random one with --synthetic, or --image), its outputs are drained and
compared with the oracle bit for bit. Prints one PASS/FAIL line per output.
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
from aie.iron import CompileTime, In, Out, Program, Runtime
from aie.iron.device import NPU2
from yolo26n_design import SegmentDesign
from yolo26n_numpy import Oracle
from yolo_params import DATA_DIR, Params, synthetic_params
from yolo_spec import Network, Tensor


def to_rows(tensor: Tensor, values: np.ndarray) -> np.ndarray:
    """An oracle `(C, H, W)` tensor as the NPU streams it: per row `[C8/8][W][8]` int8."""
    padded = np.zeros((tensor.channels8, tensor.height, tensor.width), np.int8)
    padded[: tensor.channels] = values
    grouped = padded.reshape(tensor.channels8 // 8, 8, tensor.height, tensor.width)
    return np.ascontiguousarray(grouped.transpose(2, 0, 3, 1)).reshape(-1)


def from_rows(tensor: Tensor, rows: np.ndarray) -> np.ndarray:
    """The inverse of `to_rows`, dropping the padding channels."""
    grouped = rows.view(np.int8).reshape(
        tensor.height, tensor.channels8 // 8, tensor.width, 8
    )
    return grouped.transpose(1, 3, 0, 2).reshape(
        tensor.channels8, tensor.height, tensor.width
    )[: tensor.channels]


class Layout:
    """Where a run's host transfers sit: each image's inputs then the shared weights in one buffer, each image's outputs and staged rows in another."""

    def __init__(self, design: SegmentDesign, batch: int) -> None:
        self.offsets, at = {}, 0
        for tensor in design.inputs:
            self.offsets[tensor.name] = at
            at += design.host_bytes(tensor) // 4
        self.image_in = at
        # The weights the run's cores read from DDR (SegmentDesign.weights), once.
        at *= batch
        self.weights_at = at
        for tensor, packed in design.loads:
            self.offsets[tensor.name] = at
            at += packed.size // 4
        self.total_in = at
        self.outputs_at, at = {}, 0
        for tensor in design.outputs:
            self.outputs_at[tensor.name] = at
            at += design.host_bytes(tensor) // 4
        # Rows the run reads back from DDR (SegmentDesign.staged), after its outputs.
        for _, tensor in design.staged:
            self.outputs_at[tensor.name] = at
            at += tensor.row_bytes * tensor.height // 4
        self.image_out = at
        self.total_out = at * batch
        self.weights = {tensor.name for tensor, _ in design.loads}


@iron.jit(aiecc_flags=["--placer=sa_placer", "--sa-seed=1", "--sa-effort=3"])
def segment(
    inputs: In,
    outputs: Out,
    *,
    first: CompileTime[int],
    last: CompileTime[int],
    params_path: CompileTime[str],
    params_digest: CompileTime[str],
    drained: CompileTime[str],
    batch: CompileTime[int] = 1,
    ahead: CompileTime[int] = 3,
):
    """The run `ops[first:last]` of yolo26n over `batch` images, inputs packed into `inputs`, outputs into `outputs`."""
    del params_digest  # keys the compile cache on the parameter file's contents
    network = Network.yolo26n()
    tensors = {op.dst.name: op.dst for op in network.ops}
    design = SegmentDesign.planned(
        network,
        Params.load(Path(params_path)),
        network.ops[first:last],
        [tensors[n] for n in drained.split(",")],
    )
    layout = Layout(design, batch)
    in_ty = np.ndarray[(layout.total_in,), np.dtype[np.int32]]
    out_ty = np.ndarray[(layout.total_out,), np.dtype[np.int32]]

    def sequence(inp, out, *handles):
        def locate_for(i):
            def locate(tensor):
                if tensor.name in layout.outputs_at:
                    return (
                        out,
                        layout.total_out,
                        i * layout.image_out + layout.outputs_at[tensor.name],
                    )
                if tensor.name in layout.weights:
                    return inp, layout.total_in, layout.offsets[tensor.name]
                return (
                    inp,
                    layout.total_in,
                    i * layout.image_in + layout.offsets[tensor.name],
                )

            return locate

        design.transfer(handles, [locate_for(i) for i in range(batch)], ahead)

    handles = design.handles()
    rt = Runtime(sequence, [in_ty, out_ty, *handles])
    return Program(NPU2(), rt, workers=design.workers).resolve_program()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ops", required=True, help="FIRST:STOP indexes into Network.yolo26n().ops"
    )
    parser.add_argument(
        "--params",
        type=Path,
        help="parameter file (default data/yolo26n_params.npz, or with --synthetic "
        "data/yolo26n_params_synthetic<SEED>.npz, made if missing)",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        metavar="SEED",
        help="seeded synthetic weights and image",
    )
    parser.add_argument(
        "--image", type=Path, help="a letterboxed image instead of a random one"
    )
    parser.add_argument(
        "--outputs", help="comma-separated tensors to drain (default: the last op's)"
    )
    parser.add_argument("--time", action="store_true", help="also time the dispatch")
    parser.add_argument("--n-iters", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="images per dispatch, pipelined as yolo26n.py runs them",
    )
    parser.add_argument(
        "--ahead", type=int, default=3, help="images' fills issued ahead of a wait"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="build the ELF and stop"
    )
    args = parser.parse_args()
    if args.batch < 1:
        parser.error(f"--batch {args.batch} is not positive")

    first, stop = (int(v) for v in args.ops.split(":"))
    network = Network.yolo26n()
    if args.synthetic is not None:
        params_path = synthetic_params(
            args.synthetic,
            args.params or DATA_DIR / f"yolo26n_params_synthetic{args.synthetic}.npz",
        )
    else:
        params_path = args.params or DATA_DIR / "yolo26n_params.npz"
    params = Params.load(params_path)
    ops = network.ops[first:stop]
    drained = args.outputs or ops[-1].dst.name
    kwargs = dict(
        first=first,
        last=stop,
        params_path=str(params_path),
        params_digest=hashlib.sha1(params_path.read_bytes()).hexdigest(),
        drained=drained,
        batch=args.batch,
        ahead=args.ahead,
    )
    # Build once on the host to learn the run's inputs, in the order the design
    # fills them, and its weights as the NPU2 kernels it compiles for load them.
    iron.set_current_device(NPU2())
    tensors = {op.dst.name: op.dst for op in network.ops} | {
        "images": Tensor("images", 3, 640, 640)
    }
    plan = SegmentDesign.planned(
        network, params, ops, [tensors[n] for n in drained.split(",")]
    )
    layout = Layout(plan, args.batch)
    if args.compile_only:
        segment.specialize(
            inputs=np.ndarray[(layout.total_in,), np.dtype[np.int32]],
            outputs=np.ndarray[(layout.total_out,), np.dtype[np.int32]],
            **kwargs,
        ).compile()
        return

    if args.image is not None:
        # Optional: only a real image needs OpenCV.
        from letterbox import letterbox

        images = [letterbox(args.image)] * args.batch
    else:
        rng = np.random.default_rng(args.synthetic or 0)
        images = [
            rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
            for _ in range(args.batch)
        ]
    oracle = Oracle(params, network)
    references = []
    for image in images:
        reference = oracle.run(image)
        reference["images"] = image
        references.append(reference)
    packed_in = np.concatenate(
        [
            (
                reference["images"].reshape(-1).view(np.int8)
                if t.name == "images"
                else to_rows(t, reference[t.name])
            )
            for reference in references
            for t in plan.inputs
        ]
        + [packed for _, packed in plan.loads]
    )
    inputs = iron.tensor(packed_in.view(np.int32))
    outputs = iron.zeros(layout.total_out, dtype=np.int32)
    segment(inputs, outputs, **kwargs)
    got = outputs.numpy().view(np.int8)
    failed = False
    for i, reference in enumerate(references):
        for tensor in plan.outputs:
            at = 4 * (i * layout.image_out + layout.outputs_at[tensor.name])
            size = plan.host_bytes(tensor)
            want = reference[tensor.name]
            if want.dtype == np.float32:
                mine = got[at : at + size].view(np.int32).reshape(want.shape)
                want = want.view(np.int32)
            else:
                mine = from_rows(tensor, got[at : at + size]).astype(np.int64)
            bad = int((mine != want).sum())
            print(
                f"image {i} {tensor.name}: {'PASS' if bad == 0 else 'FAIL'} "
                f"({bad}/{want.size} differ)"
            )
            failed |= bad != 0

    if args.time:
        for _ in range(args.n_warmup):
            segment(inputs, outputs, **kwargs)
        samples = []
        for _ in range(args.n_iters):
            start = time.perf_counter()
            segment(inputs, outputs, **kwargs)
            samples.append(time.perf_counter() - start)
        us = statistics.median(samples) * 1e6 / args.batch
        print(f"ops {first}:{stop}: {us:.1f} us per image at batch {args.batch}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
