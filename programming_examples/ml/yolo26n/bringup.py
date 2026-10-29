#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Run a contiguous run of yolo26n operators on the NPU and check it against the oracle.

    python bringup.py --ops 0:11                 # the stem and model.2's C3k2
    python bringup.py --ops 0:11 --synthetic 7   # the same on seeded synthetic weights
    python bringup.py --span images /model.2/cv2/conv/Conv   # the same, by op names

A run starts and ends on the boundaries of the fused runs (`FUSED`), as the
whole network is built from them.

The run's inputs come from the oracle on one letterboxed image (a seeded
random one with --synthetic, or --image), its outputs are drained and
compared with the oracle bit for bit. Prints one PASS/FAIL line per output.
"""

from __future__ import annotations

import argparse
import functools
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
    plan: SegmentDesign,
    layout: Layout,
    references: list[dict[str, np.ndarray]],
    words: np.ndarray,
) -> bool:
    """Compare each image's drained tensors with the oracle, printing a PASS/FAIL line for each.

    Args:
        plan: The run's design, built on the host.
        layout: Where each image's outputs sit in the output buffer.
        references: Per image, the oracle's tensors by name.
        words: The output buffer as the NPU wrote it.

    Returns:
        Whether every drained tensor of every image matched bit for bit.
    """
    got = words.view(np.int8)
    passed = True
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
            passed &= bad == 0
    return passed


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
    run = parser.add_mutually_exclusive_group(required=True)
    run.add_argument("--ops", help="FIRST:STOP indexes into Network.yolo26n().ops")
    run.add_argument(
        "--span",
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="the names of the run's first and last ops",
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
    if args.ahead < 0:
        parser.error(f"--ahead {args.ahead} is negative")
    if args.n_iters < 1 or args.n_warmup < 0:
        parser.error("--n-iters must be at least 1 and --n-warmup at least 0")

    network = Network.yolo26n()
    if args.span:
        first, stop = network.span(*args.span)
    else:
        first, stop = (int(v) for v in args.ops.split(":"))
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
        "images": Tensor("images", 3, IMAGE_SIZE, IMAGE_SIZE)
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

        images = [letterbox(args.image)[0]] * args.batch
    else:
        rng = np.random.default_rng(args.synthetic or 0)
        images = [
            rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
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
    passed = check(plan, layout, references, outputs.numpy())

    if args.time:
        bench = run_iters(
            functools.partial(segment, **kwargs),
            inputs,
            outputs,
            warmup=args.n_warmup,
            iters=args.n_iters,
        )
        print_benchmark(bench)
        e2e = bench.e2e
        print(
            f"ops {first}:{stop}: {e2e.min_us / args.batch:.1f} us per image at "
            f"batch {args.batch}, fastest of {e2e.n} after {args.n_warmup} warmups; "
            f"median {e2e.median_us / args.batch:.1f} us, "
            f"MAD {e2e.mad_us / args.batch:.1f} us"
        )
        print("timed dispatch:")
        passed &= check(plan, layout, references, outputs.numpy())
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
