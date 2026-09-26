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

The design pins no tiles; aiecc's SA placer places it (--sa-seed picks the
seed, --sa-effort trades placement cost for compile time).

Usage (from programming_examples/ml):
    python3 -m mobilenet.aie2_mobilenet_iron               # compile, run, verify
    python3 -m mobilenet.aie2_mobilenet_iron --emit-mlir   # print the MLIR
    python3 -m mobilenet.aie2_mobilenet_iron --sa-seed 7
    python3 -m mobilenet.aie2_mobilenet_iron --sa-effort 0.25
"""

import argparse
import os
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, InOut, ObjectFifo, Out, Program, Runtime, TaskGroup
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


# ---------------------------------------------------------------------------
# Design top-level function
# ---------------------------------------------------------------------------
@iron.jit(aiecc_flags=sa_placer_flags())
def mobilenet_iron(inp: In, cascade_wts: In, scratch: InOut, out: Out):
    """Build the full mobilenet IRON design and return the resolved Program.

    Runtime args (declared via In/Out so @iron.jit knows the design takes
    four host tensors): activations, cascade weights, the scratch the
    post-L1 / FC1 outputs round-trip through, final FC2 output.  The
    runtime ``sequence(...)`` body's args match.
    """

    # Runtime arg types: i32 element view over the underlying byte buffers.
    #   arg0 (act_in):            100352 i32 = 401408 bytes
    #   arg2 (post-L1/FC1 scratch): 1280 i32 =   5120 bytes
    #   arg3 (final FC2 output):    640 i32 =   2560 bytes
    in_ty = np.ndarray[(tensorInW * tensorInH * tensorInC // 4,), np.dtype[np.int32]]
    out_ty = np.ndarray[
        (post_L1_OutW * post_L1_OutH * post_L2_OutC * 2 // 4,),
        np.dtype[np.int32],
    ]

    # ------------------------------------------------------------------
    # Block chain — each builder owns its fifos / kernels / workers and
    # returns the activation handoff for the next stage.
    # ------------------------------------------------------------------
    init_workers, act_in, act_init_out = init_conv(sf, data_dir=data_dir)
    a_workers, act_bn9_out = regular_bottlenecks(act_init_out, sf, data_dir=data_dir)
    b_workers, act_bn12_out = pipeline_bottlenecks(act_bn9_out, sf, data_dir=data_dir)
    c_workers, act_bn14_out, wts_fifos = cascade_bottlenecks(
        act_bn12_out, sf, data_dir=data_dir
    )
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

    # Combined cascade weight tensor — _run_and_verify concatenates 4 chunks
    # into a single buffer in this exact order:
    #   bn13_L1(76800) | bn13_L3(76800) | bn14_L1(76800) | bn14_L3(76800)
    _BN_L1_SZ = 80 * 960  # 76800 bytes per L1 weight chunk
    _BN_L3_SZ = 480 * 80 * 2  # 76800 bytes per L3 weight chunk (put+get)
    _CASCADE_OFFSETS = [0, _BN_L1_SZ, 2 * _BN_L1_SZ, 3 * _BN_L1_SZ]
    _CASCADE_SIZES = [_BN_L1_SZ, _BN_L3_SZ, _BN_L1_SZ, _BN_L3_SZ]
    _cascade_wts_sz_i32 = sum(_CASCADE_SIZES) // 4  # 76800 i32 = 307200 bytes
    cascade_wts_ty = np.ndarray[(_cascade_wts_sz_i32,), np.dtype[np.int32]]
    _post_l1_out_sz_i32 = post_L1_OutW * post_L1_OutH * post_L2_InC * 2 // 4
    _scratch_sz_i32 = 2 * _post_l1_out_sz_i32
    scratch_ty = np.ndarray[(_scratch_sz_i32,), np.dtype[np.int32]]

    def _wts_tap(byte_offset, byte_size):
        return TensorAccessPattern(
            (_cascade_wts_sz_i32,),
            offset=byte_offset // 4,
            sizes=[1, 1, 1, byte_size // 4],
            strides=[0, 0, 0, 1],
        )

    # Use the gemm-style "one task_group at a time" pattern. Each task_group
    # holds a batch of fills + a wait=True drain, and finish_task_group()
    # awaits the drain AND frees every task in the group atomically. Required
    # so `dma_free_task` is not emitted for tasks that were never awaited
    # (act_in, weights, FC fills) — otherwise their BD IDs would be
    # deallocated while their DMAs were potentially still in flight.
    def sequence(
        inp,
        cascade_wts,
        scratch,
        out,
        act_in_prod,
        wts_prods,
        avgpool_cons,
        fc_prod,
        fc_cons,
    ):
        # ---- Group 1: input + weights + avgpool drain ----
        # All upstream fills + the first sync drain in the same group. By the
        # time the avgpool drain completes, init_conv has consumed all of
        # act_in and bn13/14 have consumed their weights, so freeing all of
        # those at finish_task_group() is safe (causal closure).
        tg1 = TaskGroup()
        act_in_prod.fill(inp, group=tg1)
        # bn13/14 L1+L3 weight chunks from the combined cascade buffer
        for wts_prod, off, sz in zip(wts_prods, _CASCADE_OFFSETS, _CASCADE_SIZES):
            wts_prod.fill(cascade_wts, _wts_tap(off, sz), group=tg1)
        # Round-trip avgpool output through L3 (shim 30/40 hop). Offsets
        # and sizes are i32 elements (4 bytes each):
        #   avgpool / FC1-input scratch:    i32 offset 0
        #   FC1-output / FC2-input scratch: i32 offset 640 (byte 2560)
        #   transfer length: 640 i32 = 2560 B = 1280 ui16
        _post_l1_scratch_tap = TensorAccessPattern(
            (_scratch_sz_i32,),
            offset=0,
            sizes=[1, 1, 1, _post_l1_out_sz_i32],
            strides=[0, 0, 0, 1],
        )
        avgpool_cons.drain(
            scratch,
            tap=_post_l1_scratch_tap,
            wait=True,
            group=tg1,
        )
        tg1.finish()

        # ---- Group 2: FC1 fill + FC1 drain ----
        # FC1 fill reads the avgpool scratch (drained above). FC1 drain
        # waits for FC compute to consume the fill, then drains FC1 output
        # to L3. By finish_task_group, both FC1 fill and FC1 drain have
        # completed.
        tg2 = TaskGroup()
        fc_prod.fill(
            scratch,
            tap=_post_l1_scratch_tap,
            group=tg2,
        )
        _post_fc_out_tap = TensorAccessPattern(
            (_scratch_sz_i32,),
            offset=_post_l1_out_sz_i32,
            sizes=[1, 1, 1, _post_l1_out_sz_i32],
            strides=[0, 0, 0, 1],
        )
        fc_cons.drain(
            scratch,
            tap=_post_fc_out_tap,
            wait=True,
            group=tg2,
        )
        tg2.finish()

        # ---- Group 3: FC2 fill + FC2 final drain to host ----
        tg3 = TaskGroup()
        fc_prod.fill(
            scratch,
            tap=_post_fc_out_tap,
            group=tg3,
        )
        fc_cons.drain(
            out,
            wait=True,
            group=tg3,
        )
        tg3.finish()

    rt = Runtime(
        sequence,
        [
            in_ty,
            cascade_wts_ty,
            scratch_ty,
            out_ty,
            act_in.prod(depth=1),
            [fifo.prod() for fifo in wts_fifos],
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
    p = argparse.ArgumentParser(prog="MobileNet V3 — IRON API design")
    add_compile_args(p, default_dev="npu2", with_emit_mlir=True)
    p.add_argument("--sa-seed", type=int, help="SA placer seed (default: 3)")
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
    inp = iron.tensor(
        ds.reorder_mat(chw, "YCXC8", "CYX").flatten().view(np.int32), dtype=np.int32
    )
    wts = np.concatenate(
        [
            _loadtxt_i8(f"{bn}_{part}_chain.txt")
            for bn in ("bn13", "bn14")
            for part in ("1", "3_put", "3_get")
        ]
    )
    cascade_wts = iron.tensor(wts.view(np.int32), dtype=np.int32)
    scratch = iron.zeros((post_L2_OutC * 2 // 4 * 2,), dtype=np.int32)
    out = iron.zeros((post_L2_OutC * 2 // 4,), dtype=np.int32)

    bench = run_iters(
        design,
        inp,
        cascade_wts,
        scratch,
        out,
        warmup=opts.warmup,
        iters=opts.iters,
    )

    actual = ds.reorder_mat(
        out.numpy().view(np.uint16).reshape(1, post_L2_OutC // 8, 1, 8),
        "CDYX",
        "YCXD",
    ).reshape(post_L2_OutC)
    golden = _loadtxt_i8("golden_output.txt")
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
    run_design_cli(
        design,
        opts,
        compile_kwargs={},
        device=lambda o: device_from_args(o, n_cols=None),
        run_and_verify=lambda o: _run_and_verify(design, o),
    )


if __name__ == "__main__":
    main()
