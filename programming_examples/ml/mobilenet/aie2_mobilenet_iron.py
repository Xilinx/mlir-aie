#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""MobileNet V3 — IRON API rewrite.

Replaces the placed-dialect implementation in aie2_mobilenet.py with the
high-level IRON API.  Computation is organized by block family, each in
its own sibling module under `bottleneck/`:

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
    In,
    InOut,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
)
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime.argparse import (
    add_benchmark_args,
    add_compile_args,
    device_from_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.ml import DataShaper
from aie.utils.verify import Tolerance, compare

from . import mb_utils
from .bottleneck._common import sa_placer_flags
from .bottleneck.cascade import cascade_bottlenecks

# Sibling imports below resolve via the script's parent dir (auto-added
# to sys.path[0] when invoked as ``python3 .../aie2_mobilenet_iron.py``).
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


# ---------------------------------------------------------------------------
# Design top-level function
# ---------------------------------------------------------------------------
@iron.jit(aiecc_flags=sa_placer_flags())
def mobilenet_iron(inp: In, scratch: InOut, out: Out, *, batch: CompileTime[int] = 1):
    """Build the full mobilenet IRON design and return the resolved Program.

    Runtime args (declared via In/Out so @iron.jit knows the design takes
    three host tensors): ``batch`` images of activations, the scratch the
    post-L1 / FC1 outputs round-trip through, and ``batch`` FC2 outputs.
    The runtime ``sequence(...)`` body's args match.
    """

    # Runtime arg types: i32 element view over the underlying byte buffers.
    #   arg0 (act_in):            100352 i32 = 401408 bytes per image
    #   arg1 (post-L1/FC1 scratch): 1280 i32 =   5120 bytes
    #   arg2 (final FC2 output):    640 i32 =   2560 bytes per image
    in_sz_i32 = tensorInW * tensorInH * tensorInC // 4
    out_sz_i32 = post_L1_OutW * post_L1_OutH * post_L2_OutC * 2 // 4
    in_ty = np.ndarray[(batch * in_sz_i32,), np.dtype[np.int32]]
    out_ty = np.ndarray[(batch * out_sz_i32,), np.dtype[np.int32]]

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

    def _tap(total, offset, size):
        return TensorAccessPattern(
            (total,), offset=offset, sizes=[1, 1, 1, size], strides=[0, 0, 0, 1]
        )

    # Round-trip avgpool output through L3 (shim 30/40 hop). Offsets and
    # sizes are i32 elements (4 bytes each):
    #   avgpool / FC1-input scratch:    i32 offset 0
    #   FC1-output / FC2-input scratch: i32 offset 640 (byte 2560)
    #   transfer length: 640 i32 = 2560 B = 1280 ui16
    post_l1_tap = _tap(_scratch_sz_i32, 0, _post_l1_out_sz_i32)
    post_fc_tap = _tap(_scratch_sz_i32, _post_l1_out_sz_i32, _post_l1_out_sz_i32)

    # Use the gemm-style "one task_group at a time" pattern. Each task_group
    # holds a batch of fills + a wait=True drain, and finish_task_group()
    # awaits the drain AND frees every task in the group atomically. Required
    # so `dma_free_task` is not emitted for tasks that were never awaited
    # (act_in, FC fills) — otherwise their BD IDs would be deallocated while
    # their DMAs were potentially still in flight.
    def sequence(inp, scratch, out, act_in_prod, avgpool_cons, fc_prod, fc_cons):
        # Image i's input fill joins image i's avgpool-drain group. By the
        # time that drain completes, init_conv has consumed all of it, so
        # freeing it at finish_task_group() is safe (causal closure). Fills
        # run _PREFETCH images ahead so the backbone never idles while the
        # previous image's FC round-trip runs.
        groups = [TaskGroup() for _ in range(batch)]

        def _fill(i):
            act_in_prod.fill(
                inp, _tap(batch * in_sz_i32, i * in_sz_i32, in_sz_i32), group=groups[i]
            )

        for i in range(min(_PREFETCH, batch)):
            _fill(i)
        for i in range(batch):
            if i + _PREFETCH < batch:
                _fill(i + _PREFETCH)
            avgpool_cons.drain(scratch, tap=post_l1_tap, wait=True, group=groups[i])
            groups[i].finish()

            # FC1: reads the avgpool scratch, drains FC1 output to scratch.
            tg = TaskGroup()
            fc_prod.fill(scratch, tap=post_l1_tap, group=tg)
            fc_cons.drain(scratch, tap=post_fc_tap, wait=True, group=tg)
            tg.finish()

            # FC2: reads FC1 output, drains image i's result to the host.
            tg = TaskGroup()
            fc_prod.fill(scratch, tap=post_fc_tap, group=tg)
            fc_cons.drain(
                out,
                tap=_tap(batch * out_sz_i32, i * out_sz_i32, out_sz_i32),
                wait=True,
                group=tg,
            )
            tg.finish()

    rt = Runtime(
        sequence,
        [
            in_ty,
            scratch_ty,
            out_ty,
            act_in.prod(depth=1),
            act_out_post_avgpool_shim.cons(),
            act_out_post_shim_FC.prod(),
            act_out_of.cons(),
        ],
    )

    # ------------------------------------------------------------------
    # Generate MLIR
    # ------------------------------------------------------------------
    return Program(iron.get_current_device(), rt, workers=all_workers).resolve_program()


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
        "  %(prog)s --batch 16      # 16 images per launch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_compile_args(p, default_dev="npu2", with_emit_mlir=True)
    p.add_argument("--sa-seed", type=int, help="SA placer seed (default: 3)")
    p.add_argument("--batch", type=int, default=1, help="images per launch")
    p.add_argument(
        "--sa-effort",
        type=float,
        help="SA placer search budget scale (default: 1.0; lower trades "
        "placement cost for compile time)",
    )
    # The NPU takes 6-13 launches after load to reach its steady latency.
    add_benchmark_args(p, default_warmup=20, default_iters=5)
    return p


def _loadtxt_i8(name):
    return np.loadtxt(data_dir + name, delimiter=",", dtype=np.int32).astype(np.int8)


def _run_and_verify(design, opts):
    ds = DataShaper()
    chw = _loadtxt_i8("before_ifm_mem_fmt_1x1.txt").reshape(
        tensorInC, tensorInH, tensorInW
    )
    n = opts.batch
    inp = iron.tensor(
        np.tile(ds.reorder_mat(chw, "YCXC8", "CYX").flatten().view(np.int32), n),
        dtype=np.int32,
    )
    scratch = iron.zeros((post_L2_OutC * 2 // 4 * 2,), dtype=np.int32)
    out = iron.zeros((n * post_L2_OutC * 2 // 4,), dtype=np.int32)

    bench = run_iters(
        design,
        inp,
        scratch,
        out,
        warmup=opts.warmup,
        iters=opts.iters,
    )

    actual = np.concatenate(
        [
            ds.reorder_mat(
                o.reshape(1, post_L2_OutC // 8, 1, 8), "CDYX", "YCXD"
            ).reshape(post_L2_OutC)
            for o in out.numpy().view(np.uint16).reshape(n, -1)
        ]
    )
    golden = np.tile(_loadtxt_i8("golden_output.txt"), n)
    verdict = compare(
        actual.astype(np.int32),
        golden.astype(np.int32),
        Tolerance.lsb(9, note="should be 1; #3009"),
    )
    print_benchmark(bench)
    print(f"max_difference: {verdict.max_abs_err:g}")
    if not verdict.ok:
        sys.exit(f"FAIL: {verdict.detail}")
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    design = mobilenet_iron
    if opts.sa_seed is not None or opts.sa_effort is not None:
        design = design.specialize(
            aiecc_flags=sa_placer_flags(
                opts.sa_seed if opts.sa_seed is not None else 3,
                opts.sa_effort if opts.sa_effort is not None else 1.0,
            )
        )
    if opts.batch > 1:
        design = design.specialize(batch=opts.batch)
    run_design_cli(
        design,
        opts,
        compile_kwargs={},
        device=lambda o: device_from_args(o, n_cols=None),
        run_and_verify=lambda o: _run_and_verify(design, o),
    )


if __name__ == "__main__":
    main()
