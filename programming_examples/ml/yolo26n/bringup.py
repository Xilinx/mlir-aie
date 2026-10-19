#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Run a contiguous run of yolo26n operators on the NPU and check it against the oracle.

    python bringup.py --ops 0:2                  # the input quantizer and the stem conv
    python bringup.py --ops 0:2 --synthetic 7    # the same on seeded synthetic weights

The run's inputs come from the oracle on one letterboxed image (a seeded
random one with --synthetic, or --image), its outputs are drained and
compared with the oracle bit for bit. Prints one PASS/FAIL line per output.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, Out, Program, Runtime, TaskGroup
from aie.iron.device import NPU2

from gen_yolo_data import DATA_DIR, Params, Synthesizer
from letterbox import letterbox
from yolo26n_design import IMAGE_ROW_BYTES, SegmentDesign
from yolo26n_numpy import Oracle
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


@iron.jit
def segment(
    inputs: In,
    outputs: Out,
    *,
    first: CompileTime[int],
    last: CompileTime[int],
    params_path: CompileTime[str],
    params_digest: CompileTime[str],
    drained: CompileTime[str],
):
    """The run `ops[first:last]` of yolo26n, inputs packed into `inputs`, outputs into `outputs`."""
    del params_digest  # keys the compile cache on the parameter file's contents
    network = Network.yolo26n()
    tensors = {op.dst.name: op.dst for op in network.ops}
    design = SegmentDesign(
        network,
        Params.load(Path(params_path)),
        network.ops[first:last],
        [tensors[n] for n in drained.split(",")],
    )
    offsets, at = {}, 0
    for tensor in design.inputs:
        offsets[tensor.name] = at
        at += design.host_bytes(tensor) // 4
    total_in = at
    sizes_out = [design.host_bytes(t) // 4 for t in design.outputs]
    in_ty = np.ndarray[(total_in,), np.dtype[np.int32]]
    out_ty = np.ndarray[(sum(sizes_out),), np.dtype[np.int32]]
    steps = sorted({f.step for f in design.fills})

    def sequence(inp, out, *handles):
        fill_handles = dict(zip(design.in_fifos, handles))
        drains = TaskGroup()
        offset = 0
        for size, handle in zip(sizes_out, handles[len(design.in_fifos) :]):
            tap = TensorAccessPattern(
                (sum(sizes_out),), offset, [1, 1, 1, size], [0, 0, 0, 1]
            )
            handle.drain(out, tap, wait=True, group=drains)
            offset += size
        # Each step's fills finish before the next step's start.
        for step in steps:
            fills = TaskGroup()
            for fill in (f for f in design.fills if f.step == step):
                tap = fill.tap(total_in, offsets[fill.tensor.name])
                fill_handles[fill.fifo].fill(
                    inp, tap, wait=step != steps[-1], group=fills
                )
            if step != steps[-1]:
                fills.finish()
        drains.finish()
        fills.finish()

    handles = [fifo.prod() for fifo in design.in_fifos]
    handles += [design.out_fifos[t.name].cons() for t in design.outputs]
    rt = Runtime(sequence, [in_ty, out_ty, *handles])
    return Program(NPU2(), rt, workers=design.workers).resolve_program()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ops", required=True, help="FIRST:STOP indexes into Network.yolo26n().ops"
    )
    parser.add_argument("--params", type=Path, default=DATA_DIR / "yolo26n_params.npz")
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
    args = parser.parse_args()

    first, stop = (int(v) for v in args.ops.split(":"))
    network = Network.yolo26n()
    if args.synthetic is not None:
        params_path = DATA_DIR / f"yolo26n_params_synthetic{args.synthetic}.npz"
        if not params_path.exists():
            params_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(params_path, **Synthesizer(args.synthetic).run(network))
    else:
        params_path = args.params
    params = Params.load(params_path)
    if args.image is not None:
        image = letterbox(args.image)
    else:
        image = np.random.default_rng(args.synthetic or 0).integers(
            0, 256, (640, 640, 3), dtype=np.uint8
        )
    reference = Oracle(params, network).run(image)
    reference["images"] = image

    ops = network.ops[first:stop]
    drained = args.outputs or ops[-1].dst.name
    # Build once on the host to learn the run's inputs, in the order the design fills them.
    tensors = {op.dst.name: op.dst for op in network.ops} | {
        "images": Tensor("images", 3, 640, 640)
    }
    plan = SegmentDesign(network, params, ops, [tensors[n] for n in drained.split(",")])
    packed_in = np.concatenate(
        [
            (
                image.reshape(-1).view(np.int8)
                if t.name == "images"
                else to_rows(t, reference[t.name])
            )
            for t in plan.inputs
        ]
    )
    sizes = [plan.host_bytes(t) for t in plan.outputs]
    inputs = iron.tensor(packed_in.view(np.int32))
    outputs = iron.zeros(sum(sizes) // 4, dtype=np.int32)
    segment(
        inputs,
        outputs,
        first=first,
        last=stop,
        params_path=str(params_path),
        params_digest=hashlib.sha1(params_path.read_bytes()).hexdigest(),
        drained=drained,
    )
    got = outputs.numpy().view(np.int8)
    failed = False
    at = 0
    for tensor, size in zip(plan.outputs, sizes):
        want = reference[tensor.name]
        if want.dtype == np.float32:
            mine = got[at : at + size].view(np.int32).reshape(want.shape)
            want = want.view(np.int32)
        else:
            mine = from_rows(tensor, got[at : at + size]).astype(np.int64)
        bad = int((mine != want).sum())
        print(
            f"{tensor.name}: {'PASS' if bad == 0 else 'FAIL'} ({bad}/{want.size} differ)"
        )
        failed |= bad != 0
        at += size
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
