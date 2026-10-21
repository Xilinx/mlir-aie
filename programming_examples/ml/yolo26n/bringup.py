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
from aie.iron import CompileTime, In, Out, Program, Runtime
from aie.iron.device import NPU2
from aie.utils.benchmark import print_benchmark, run_iters
from yolo26n_design import SegmentDesign
from yolo26n_numpy import Oracle
from yolo_params import DATA_DIR, Params, synthetic_params
from yolo_spec import IMAGE_SIZE, Network, Tensor


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


def check(
    drained: list[Tensor],
    sizes: list[int],
    reference: dict[str, np.ndarray],
    words: np.ndarray,
) -> bool:
    """Compare each drained tensor with the oracle, printing a PASS/FAIL line for it.

    Args:
        drained: The tensors the run drains, in the order they are packed.
        sizes: Each one's packed size in bytes.
        reference: The oracle's tensors by name.
        words: The output buffer as the NPU wrote it.

    Returns:
        Whether every drained tensor matched bit for bit.
    """
    got = words.view(np.int8)
    passed = True
    at = 0
    for tensor, size in zip(drained, sizes):
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
        passed &= bad == 0
        at += size
    return passed


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
    # The weights the run's cores read from DDR (SegmentDesign.weights).
    for tensor, packed in design.loads:
        offsets[tensor.name] = at
        at += packed.size // 4
    total_in = at
    outputs_at, at = {}, 0
    for tensor in design.outputs:
        outputs_at[tensor.name] = at
        at += design.host_bytes(tensor) // 4
    for tensor in design.in_ddr:
        outputs_at[tensor.name] = at
        at += tensor.row_bytes * tensor.height // 4
    total_out = at
    in_ty = np.ndarray[(total_in,), np.dtype[np.int32]]
    out_ty = np.ndarray[(total_out,), np.dtype[np.int32]]

    def sequence(inp, out, *handles):
        def locate(tensor):
            if tensor.name in outputs_at:
                return out, total_out, outputs_at[tensor.name]
            return inp, total_in, offsets[tensor.name]

        design.transfer(handles, locate)

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
    parser.add_argument(
        "--time",
        action="store_true",
        help="also time the dispatch, report the fastest call and check its output",
    )
    parser.add_argument("--n-iters", type=int, default=20, help="timed calls (>= 1)")
    # The NPU takes about ten calls after a load to reach its steady latency.
    parser.add_argument("--n-warmup", type=int, default=10, help="untimed calls first")
    args = parser.parse_args()
    if args.n_iters < 1 or args.n_warmup < 0:
        parser.error("--n-iters must be at least 1 and --n-warmup at least 0")

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
    if args.image is not None:
        # Optional: only a real image needs OpenCV.
        from letterbox import letterbox

        image = letterbox(args.image)
    else:
        image = np.random.default_rng(args.synthetic or 0).integers(
            0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
        )
    reference = Oracle(params, network).run(image)
    reference["images"] = image

    ops = network.ops[first:stop]
    drained = args.outputs or ops[-1].dst.name
    # Build once on the host to learn the run's inputs, in the order the design fills them.
    tensors = {op.dst.name: op.dst for op in network.ops} | {
        "images": Tensor("images", 3, IMAGE_SIZE, IMAGE_SIZE)
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
        + [packed for _, packed in plan.loads]
    )
    sizes = [plan.host_bytes(t) for t in plan.outputs]
    spilled = sum(t.row_bytes * t.height for t in plan.in_ddr)
    inputs = iron.tensor(packed_in.view(np.int32))
    outputs = iron.zeros((sum(sizes) + spilled) // 4, dtype=np.int32)
    kwargs = dict(
        first=first,
        last=stop,
        params_path=str(params_path),
        params_digest=hashlib.sha1(params_path.read_bytes()).hexdigest(),
        drained=drained,
    )
    segment(inputs, outputs, **kwargs)
    passed = check(plan.outputs, sizes, reference, outputs.numpy())

    if args.time:
        bench = run_iters(
            segment,
            inputs,
            outputs,
            warmup=args.n_warmup,
            iters=args.n_iters,
            **kwargs,
        )
        print_benchmark(bench)
        e2e = bench.e2e
        print(
            f"ops {first}:{stop}: {e2e.min_us:.1f} us, fastest of {e2e.n} after "
            f"{args.n_warmup} warmups; median {e2e.median_us:.1f} us, "
            f"MAD {e2e.mad_us:.1f} us"
        )
        print("timed dispatch:")
        passed &= check(plan.outputs, sizes, reference, outputs.numpy())
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
