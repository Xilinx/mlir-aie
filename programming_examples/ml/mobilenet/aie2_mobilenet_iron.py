#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""MobileNet V3 on the IRON API.

Computation is organized by block family, each in its own sibling module
under `bottleneck/`:

  init.py      — 3x3 stride-2 input conv
  regular.py   — bn0–bn9  (single compute tile per block)
  pipeline.py  — bn10–bn12 (one tile per layer)
  cascade.py   — bn13–bn14 (split-channel cascade stream, 5 tiles per block)
  post_l1.py   — avg pool + expand 1x1 conv
  post_l2.py   — 4-tile FC1+FC2 (split output channels)

Scale factors are compile-time Python int constants loaded from
scale_factors_final.json and passed directly in Worker fn_args — no RTP
buffers or NpuWriteRTPOp calls are needed.

The design pins no tiles; aiecc's SA placer places it.

Run `python3 -m mobilenet.aie2_mobilenet_iron --help` from
programming_examples/ml for the options.
"""

import argparse
import os
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    DispatchTime,
    In,
    InOut,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    require,
)
from aie.iron.controlflow import else_, if_, range_, yield_
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime.argparse import (
    add_benchmark_args,
    add_compile_args,
    add_trace_arg,
    device_from_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.ml import DataShaper
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent
from aie.utils.verify import Tolerance, compare

from . import mb_utils, mobilenet_numpy
from .bottleneck._common import sa_placer_flags
from .bottleneck.cascade import cascade_bottlenecks
from .bottleneck.init import init_conv
from .bottleneck.pipeline import pipeline_bottlenecks
from .bottleneck.post_l1 import post_l1
from .bottleneck.post_l2 import post_l2
from .bottleneck.regular import regular_bottlenecks
from .network_spec import block as nsblock

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), "data") + "/"
scale_factor_file = "scale_factors_final.json"

sf = mb_utils.read_scale_factors(data_dir + scale_factor_file)


# ---------------------------------------------------------------------------
# Module-scope dims used by the top-level runtime arg types and the runtime
# sequence (block-local dims live inside the builder modules).
# ---------------------------------------------------------------------------
tensorInW, tensorInH, tensorInC = nsblock("init").layers[0].in_shape
post_L1_OutW, post_L1_OutH, _ = nsblock("post_l1").layers[0].out_shape
post_L2_InC = nsblock("post_l2").layers[0].in_shape[2]
post_L2_OutC = nsblock("post_l2").layers[-1].out_shape[2]

# Input fills queued ahead of the image being drained in a batched launch.
_PREFETCH = 3


# Runtime arg types: i32 element view over the underlying byte buffers.
#   arg0 (act_in):            100352 i32 = 401408 bytes per image
#   arg1 (post-L1/FC1 scratch): 1280 i32 =   5120 bytes
#   arg2 (final FC2 output):    640 i32 =   2560 bytes per image
in_sz_i32 = tensorInW * tensorInH * tensorInC // 4
out_sz_i32 = post_L1_OutW * post_L1_OutH * post_L2_OutC * 2 // 4


# ---------------------------------------------------------------------------
# Design top-level function
# ---------------------------------------------------------------------------
@iron.jit(aiecc_flags=sa_placer_flags())
def mobilenet_iron(
    inp: In,
    scratch: InOut,
    out: Out,
    *,
    batch: DispatchTime[np.int32],
    trace_config: CompileTime[TraceConfig | None] = None,
):
    """Build the full mobilenet IRON design and return the resolved Program.

    Runtime args (declared via In/Out so @iron.jit knows the design takes
    three host tensors): ``batch`` images of activations, the scratch the
    post-L1 / FC1 outputs round-trip through, and ``batch`` FC2 outputs.
    The runtime ``sequence(...)`` body's args match.

    Specialized, ``batch`` sizes the host tensors and the sequence unrolls.
    Left free, the design compiles once: the host tensors it is called with
    set how many images it can take, and each dispatch runs any ``batch``
    from 1 up to that many.

    Raises:
        ValueError: If a specialized ``batch`` is below 1, or a free
            ``batch`` has no host tensor types to size its buffers from.
    """
    static = isinstance(batch, (int, np.integer))
    if static:
        if batch < 1:
            raise ValueError(f"batch must be a positive integer, got {batch}")
        capacity = int(batch)
        in_ty = np.ndarray[(capacity * in_sz_i32,), np.dtype[np.int32]]
        out_ty = np.ndarray[(capacity * out_sz_i32,), np.dtype[np.int32]]
    else:
        if not isinstance(inp, In) or not isinstance(out, Out):
            raise ValueError(
                "a dispatch-time batch sizes its host buffers from the input "
                "and output tensor types; call the design with tensors or "
                "specialize inp and out"
            )
        in_ty, out_ty = inp.ndarray_type, out.ndarray_type
        (in_elems,) = inp.shape
        (out_elems,) = out.shape
        capacity = in_elems // in_sz_i32
        if capacity < 1 or in_elems != capacity * in_sz_i32:
            raise ValueError(
                f"input holds {in_elems} i32, not a whole number of "
                f"{in_sz_i32}-i32 images"
            )
        if out_elems != capacity * out_sz_i32:
            raise ValueError(
                f"output holds {out_elems} i32 but the input holds {capacity} "
                f"images, which need {capacity * out_sz_i32}"
            )

    # ------------------------------------------------------------------
    # Block chain — each builder owns its fifos / kernels / workers and
    # returns the activation handoff for the next stage.
    # ------------------------------------------------------------------
    init_workers, act_in, act_init_out = init_conv(sf, data_dir=data_dir)
    a_workers, act_bn9_out = regular_bottlenecks(act_init_out, sf, data_dir=data_dir)
    b_workers, act_bn12_out = pipeline_bottlenecks(act_bn9_out, sf, data_dir=data_dir)
    c_workers, act_bn14_out = cascade_bottlenecks(act_bn12_out, sf, data_dir=data_dir)
    l1_workers, act_out_post_avgpool_shim = post_l1(act_bn14_out, sf, data_dir=data_dir)

    # post_l1 drains to host scratch and post_l2 fills from host scratch; this
    # bridge fifo is the runtime-sequence-side handle to that round-trip.
    # Neither builder is the natural owner — the top file glues them together.
    act_out_post_shim_FC = ObjectFifo(
        np.ndarray[(post_L2_InC,), np.dtype[np.uint16]],
        depth=2,
    )

    l2_workers, act_out_of = post_l2(act_out_post_shim_FC, sf, data_dir=data_dir)

    # ------------------------------------------------------------------
    # Collect all workers
    # ------------------------------------------------------------------
    all_workers = (
        init_workers + a_workers + b_workers + c_workers + l1_workers + l2_workers
    )

    _post_l1_out_sz_i32 = post_L1_OutW * post_L1_OutH * post_L2_InC * 2 // 4
    _scratch_sz_i32 = 2 * _post_l1_out_sz_i32
    scratch_ty = np.ndarray[(_scratch_sz_i32,), np.dtype[np.int32]]

    # Round-trip avgpool output through L3 (shim 30/40 hop). Offsets and
    # sizes are i32 elements (4 bytes each):
    #   avgpool / FC1-input scratch:    i32 offset 0
    #   FC1-output / FC2-input scratch: i32 offset 640 (byte 2560)
    #   transfer length: 640 i32 = 2560 B = 1280 ui16
    post_l1_tap, post_fc_tap = TensorAccessPattern.full((_scratch_sz_i32,)).partition(2)

    # Use the gemm-style "one task_group at a time" pattern. Each task_group
    # holds a batch of fills + a wait=True drain, and finish_task_group()
    # awaits the drain AND frees every task in the group atomically. Required
    # so `dma_free_task` is not emitted for tasks that were never awaited
    # (act_in, FC fills) — otherwise their BD IDs would be deallocated while
    # their DMAs were potentially still in flight.
    def sequence(inp, scratch, out, batch, act_in_prod, avgpool_cons, fc_prod, fc_cons):
        in_taps = TensorAccessPattern.full((capacity * in_sz_i32,)).partition(capacity)
        out_taps = TensorAccessPattern.full((capacity * out_sz_i32,)).partition(
            capacity
        )

        def fill_image(i):
            group = TaskGroup()
            act_in_prod.fill(inp, in_taps[i], group=group)
            return group

        def finish_image(group, i):
            # Image i's input fill shares this group with its avgpool drain:
            # by the time that drain completes, init_conv has consumed all of
            # the input, so freeing the fill here is safe (causal closure).
            avgpool_cons.drain(scratch, tap=post_l1_tap, wait=True, group=group)
            group.finish()

            # FC1: reads the avgpool scratch, drains FC1 output to scratch.
            tg = TaskGroup()
            fc_prod.fill(scratch, tap=post_l1_tap, group=tg)
            fc_cons.drain(scratch, tap=post_fc_tap, wait=True, group=tg)
            tg.finish()

            # FC2: reads FC1 output, drains image i's result to the host.
            tg = TaskGroup()
            fc_prod.fill(scratch, tap=post_fc_tap, group=tg)
            fc_cons.drain(out, tap=out_taps[i], wait=True, group=tg)
            tg.finish()

        # Fills run _PREFETCH images ahead so the backbone never idles while
        # the previous image's FC round-trip runs.
        def unrolled(n):
            groups = [fill_image(i) for i in range(min(_PREFETCH, n))]
            for i in range(n):
                if i + _PREFETCH < n:
                    groups.append(fill_image(i + _PREFETCH))
                finish_image(groups[i], i)

        # The same schedule with the steady state rolled: the _PREFETCH fill
        # groups in flight ride the loop as iter_args.
        def rolled():
            groups = [fill_image(i) for i in range(_PREFETCH)]
            for i, carried, last in range_(batch - _PREFETCH, iter_args=groups):
                ahead = fill_image(i + _PREFETCH)
                finish_image(carried[0], i)
                yield_([*carried[1:], ahead])
            # range_ traces its body exactly once, so last is always bound.
            tail = last  # pyright: ignore[reportPossiblyUnboundVariable]
            for k, group in enumerate(tail):
                finish_image(group, batch - _PREFETCH + k)

        # Batches below _PREFETCH never reach a steady state; each gets its
        # unrolled schedule.
        def select(n):
            if n >= _PREFETCH:
                rolled()
            elif n == capacity:
                unrolled(n)
            else:
                with if_(batch == n) as branch:
                    unrolled(n)
                with else_(branch):
                    select(n + 1)

        if static:
            unrolled(capacity)
        else:
            require(batch >= 1, "batch must be at least 1")
            require(
                batch <= capacity,
                f"batch exceeds the {capacity} images the host tensors hold",
            )
            select(1)

    rt = Runtime(
        sequence,
        [
            in_ty,
            scratch_ty,
            out_ty,
            batch,
            act_in.prod(depth=1),
            act_out_post_avgpool_shim.cons(),
            act_out_post_shim_FC.prod(),
            act_out_of.cons(),
        ],
    )

    # ------------------------------------------------------------------
    # Generate MLIR
    # ------------------------------------------------------------------
    prog = Program(iron.get_current_device(), rt, workers=all_workers)
    if trace_config:
        # Trace packet ids are 5 bits, so 31 tiles at most; the last FC split
        # runs the same work as its three siblings.
        untraced = l2_workers[-1]
        prog.enable_trace(
            trace_config.trace_size,
            workers=[w for w in all_workers if w is not untraced],
            coretile_events=[
                CoreEvent.ACTIVE,
                CoreEvent.LOCK_STALL,
                CoreEvent.STREAM_STALL,
                CoreEvent.CASCADE_STALL,
                CoreEvent.MEMORY_STALL,
                CoreEvent.INSTR_CALL,
                CoreEvent.INSTR_RETURN,
            ],
        )
    return prog.resolve_program()


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="python3 -m mobilenet.aie2_mobilenet_iron",
        description="MobileNet V3 on the IRON API: compile, run and verify "
        "against the golden output (run from programming_examples/ml). The "
        "design pins no tiles; aiecc's SA placer places it.",
        epilog="examples:\n"
        "  %(prog)s                 # compile, run, verify\n"
        "  %(prog)s --emit-mlir     # print the MLIR\n"
        "  %(prog)s --sa-seed 7     # place with another SA seed\n"
        "  %(prog)s --sa-effort 0.25\n"
        "  %(prog)s --batch 16      # 16 images per launch\n"
        "  %(prog)s --dynamic 1 4 16 64  # one compile, each batch per launch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_compile_args(p, default_dev="npu2", with_emit_mlir=True)
    p.add_argument("--sa-seed", type=int, help="SA placer seed (default: 3)")
    batch = p.add_mutually_exclusive_group()
    batch.add_argument(
        "--batch", type=_positive_int, default=1, help="images per launch (>= 1)"
    )
    batch.add_argument(
        "--dynamic",
        type=_positive_int,
        nargs="+",
        metavar="BATCH",
        help="compile once with a dispatch-time batch and run each listed "
        "batch; the host tensors hold the largest",
    )
    p.add_argument(
        "--sa-effort",
        type=float,
        help="SA placer search budget scale (default: 0.05; 1.0 is the full "
        "schedule, which compiles slower for a placement no faster on the NPU)",
    )
    # The NPU takes 6-13 launches after load to reach its steady latency.
    add_benchmark_args(p, default_warmup=20, default_iters=5)
    add_trace_arg(p, with_short=False)
    return p


def _run_and_verify(design, opts):
    ds = DataShaper()
    chw = (
        np.loadtxt(
            data_dir + "before_ifm_mem_fmt_1x1.txt", delimiter=",", dtype=np.int32
        )
        .astype(np.int8)
        .reshape(tensorInC, tensorInH, tensorInW)
    )
    golden_image = np.loadtxt(
        data_dir + "golden_output.txt", delimiter=",", dtype=np.int32
    ).astype(np.int8)
    batches = opts.dynamic or [opts.batch]
    capacity = max(batches)
    # Each image of a batch differs, so an image answered with another's output fails.
    images = [np.roll(chw, (3 * k, 5 * k), axis=(1, 2)) for k in range(capacity)]
    goldens = np.concatenate(
        [
            np.asarray(
                mobilenet_numpy.run(im.transpose(1, 2, 0), data_dir, sf, verbose=False)
            )
            .reshape(post_L2_OutC)
            .astype(np.int32)
            for im in images
        ]
    )
    inp = iron.tensor(
        np.concatenate(
            [
                ds.reorder_mat(im, "YCXC8", "CYX").flatten().view(np.int32)
                for im in images
            ]
        ),
        dtype=np.int32,
    )
    scratch = iron.zeros((post_L2_OutC * 2 // 4 * 2,), dtype=np.int32)
    trace = TraceConfig(trace_size=opts.trace_size) if opts.trace_size else None

    for n in batches:
        # A fresh output per batch, so no image passes on an earlier batch's result.
        out = iron.zeros((capacity * post_L2_OutC * 2 // 4,), dtype=np.int32)
        dispatch = {}
        if opts.dynamic:
            print(f"batch={n}")
            dispatch = dict(batch=n)
        if trace:
            dispatch["trace_config"] = trace
        bench = run_iters(
            design,
            inp,
            scratch,
            out,
            **dispatch,
            warmup=opts.warmup,
            iters=1 if trace else opts.iters,
        )
        if trace:
            if trace.physical_mlir_path is None:
                sys.exit("the JIT runtime did not record the traced design's MLIR")
            print(
                "trace:",
                *trace.trace_to_json(
                    trace.physical_mlir_path, output_name=f"trace_mobilenet_b{n}.json"
                ),
            )

        actual = np.concatenate(
            [
                ds.reorder_mat(
                    o.reshape(1, post_L2_OutC // 8, 1, 8), "CDYX", "YCXD"
                ).reshape(post_L2_OutC)
                for o in out.numpy().view(np.uint16).reshape(capacity, -1)[:n]
            ]
        )
        print_benchmark(bench)
        verdict = compare(
            actual[:post_L2_OutC].astype(np.int32),
            golden_image.astype(np.int32),
            Tolerance.lsb(
                3,
                note="measured max 3 (959/1280 exact): the golden's average "
                "pool keeps one more fractional bit than the kernel's",
            ),
        )
        print(f"max_difference: {verdict.max_abs_err:g}")
        if not verdict.ok:
            sys.exit(f"FAIL: image 0 against the brevitas golden: {verdict.detail}")
        verdict = compare(
            actual.astype(np.int32),
            goldens[: n * post_L2_OutC],
            Tolerance.exact(note="mobilenet_numpy mirrors the kernels' arithmetic"),
        )
        if not verdict.ok:
            sys.exit(f"FAIL: {n} images against mobilenet_numpy: {verdict.detail}")
        print("PASS!")


def _compile_kwargs(opts):
    """Size a ``--dynamic`` compile's host buffers for its largest batch."""
    if not opts.dynamic:
        return {}
    capacity = max(opts.dynamic)
    return dict(
        inp=np.ndarray[(capacity * in_sz_i32,), np.dtype[np.int32]],
        out=np.ndarray[(capacity * out_sz_i32,), np.dtype[np.int32]],
    )


def main():
    opts = _make_argparser().parse_args()
    design = mobilenet_iron
    if opts.sa_seed is not None or opts.sa_effort is not None:
        overrides = {}
        if opts.sa_seed is not None:
            overrides["seed"] = opts.sa_seed
        if opts.sa_effort is not None:
            overrides["effort"] = opts.sa_effort
        design = design.specialize(aiecc_flags=sa_placer_flags(**overrides))
    if not opts.dynamic:
        design = design.specialize(batch=opts.batch)
    run_design_cli(
        design,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: device_from_args(o, n_cols=None),
        run_and_verify=lambda o: _run_and_verify(design, o),
    )


if __name__ == "__main__":
    main()
