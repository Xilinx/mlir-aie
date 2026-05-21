# vision/edge_detect/edge_detect.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 AMD Inc.
"""Vision edge-detect pipeline -- ``@iron.jit`` design.

A 4-stage line-based pipeline on a single column:

  shim --> rgba2gray --> filter2d (3x3 Laplacian) --> threshold -->
                                                     gray2rgba+addWeighted --> shim

All five kernels are pulled from ``aie.iron.kernels.vision``.  The filter2d
worker reuses an inner Buffer holding the constant Laplacian kernel
(cross-stencil with -16384 center / 4096 edges).  The gray2rgba+addWeighted
worker combines the thresholded edge map with the original RGBA input
(forwarded down via ``inOF_L2L1``).

``aiecc_flags=["--alloc-scheme=basic-sequential"]`` matches the pre-merge
Makefile's aiecc invocation; the default 4-bank allocator would shift
buffer addresses for no perf gain on this single-column line-based design.

Two invocation modes:

  * standalone:   ``python3 edge_detect.py``  (JIT-compile + run on random
                  input; output not verified -- use the C++/OpenCV host for
                  pixel-level checks).
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile).
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import (
    Buffer,
    Compile,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2
from aie.utils.hostruntime import set_current_device


def _device_for(dev_str):
    return NPU1Col1() if dev_str == "npu" else NPU2()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def edge_detect(
    in_tensor: In,
    _b_unused: In,
    out_tensor: Out,
    *,
    width: Compile[int] = 1920,
    height: Compile[int] = 1080,
):
    height_minus_1 = height - 1
    line_width = width
    line_width_in_bytes = width * 4  # 4 channels (RGBA)

    line_bytes_ty = np.ndarray[(line_width_in_bytes,), np.dtype[np.uint8]]
    line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]

    rgba2gray_line_kernel = kernels.rgba2gray(line_width=line_width)
    filter2d_line_kernel = kernels.filter2d(line_width=line_width)
    threshold_line_kernel = kernels.threshold(line_width=line_width, dtype=np.uint8)
    gray2rgba_line_kernel = kernels.gray2rgba(line_width=line_width)
    # add_weighted operates byte-wise over the flattened RGBA buffer, so its
    # "line width" is the full RGBA stride in bytes.
    add_weighted_line_kernel = kernels.add_weighted(
        line_width=line_width_in_bytes, dtype=np.uint8
    )

    # Dataflow
    in_of_l3l2 = ObjectFifo(line_bytes_ty, name="inOF_L3L2")
    in_of_l2l1 = in_of_l3l2.cons(7).forward(depth=7, name="inOF_L2L1")
    out_of_l1l2 = ObjectFifo(line_bytes_ty, name="outOF_L1L2")
    out_of_l2l3 = out_of_l1l2.cons().forward(name="outOF_L2L3")

    intermediate_depths = [4, 2, 2]
    of_intermediates = [
        ObjectFifo(line_ty, depth=intermediate_depths[i], name=f"OF_{i + 2}to{i + 3}")
        for i in range(3)
    ]
    of_local = ObjectFifo(line_bytes_ty, depth=1, name="OF_local")

    # Laplacian edge-detect kernel: cross stencil with -16384 center, 4096 edges.
    v0, v1, v_minus4 = 0, 4096, -16384
    filter_kernel_buff = Buffer(
        np.ndarray[(3, 3), np.dtype[np.int16]],
        name="kernel",
        initial_value=np.array(
            [[v0, v1, v0], [v1, v_minus4, v1], [v0, v1, v0]], dtype=np.int16
        ),
    )

    workers = []

    def rgba2gray_fn(of_in, of_out, rgba2gray_line):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        rgba2gray_line(elem_in, elem_out, line_width)
        of_in.release(1)
        of_out.release(1)

    workers.append(
        Worker(
            rgba2gray_fn,
            [in_of_l3l2.cons(), of_intermediates[0].prod(), rgba2gray_line_kernel],
        )
    )

    def filter_fn(of_in, of_out, filter_kernel, filter2d_line):
        # 3-line stencil over height rows.  Top/bottom borders duplicate the
        # adjacent row; the steady-state middle uses real (i-1, i, i+1).
        for _ in range_(sys.maxsize):
            # Top border
            elems_in_pre = of_in.acquire(2)
            elem_pre_out = of_out.acquire(1)
            filter2d_line(
                elems_in_pre[0],
                elems_in_pre[0],
                elems_in_pre[1],
                elem_pre_out,
                line_width,
                filter_kernel,
            )
            of_out.release(1)

            # Steady-state
            for _ in range_(1, height_minus_1):
                elems_in = of_in.acquire(3)
                elem_out = of_out.acquire(1)
                filter2d_line(
                    elems_in[0],
                    elems_in[1],
                    elems_in[2],
                    elem_out,
                    line_width,
                    filter_kernel,
                )
                of_in.release(1)
                of_out.release(1)

            # Bottom border
            elems_in_post = of_in.acquire(2)
            elem_post_out = of_out.acquire(1)
            filter2d_line(
                elems_in_post[0],
                elems_in_post[1],
                elems_in_post[1],
                elem_post_out,
                line_width,
                filter_kernel,
            )
            of_in.release(2)
            of_out.release(1)

    workers.append(
        Worker(
            filter_fn,
            [
                of_intermediates[0].cons(),
                of_intermediates[1].prod(),
                filter_kernel_buff,
                filter2d_line_kernel,
            ],
            while_true=False,
        )
    )

    def threshold_fn(of_in, of_out, threshold_line):
        v_thr, v_max, v_typ = 10, 255, 0
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        threshold_line(elem_in, elem_out, line_width, v_thr, v_max, v_typ)
        of_in.release(1)
        of_out.release(1)

    workers.append(
        Worker(
            threshold_fn,
            [
                of_intermediates[1].cons(),
                of_intermediates[2].prod(),
                threshold_line_kernel,
            ],
        )
    )

    def gray2rgba_add_weight_fn(
        of_in,
        of_in2,
        of_out_self,
        of_in_self,
        of_out,
        gray2rgba_line,
        add_weighted_line,
    ):
        elem_in = of_in.acquire(1)
        elem_out = of_out_self.acquire(1)
        gray2rgba_line(elem_in, elem_out, line_width)
        of_in.release(1)
        of_out_self.release(1)

        elem_in1 = of_in_self.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out2 = of_out.acquire(1)

        alpha, beta, gamma = 16384, 16384, 0
        add_weighted_line(
            elem_in1,
            elem_in2,
            elem_out2,
            line_width_in_bytes,
            alpha,
            beta,
            gamma,
        )
        of_in_self.release(1)
        of_in2.release(1)
        of_out.release(1)

    workers.append(
        Worker(
            gray2rgba_add_weight_fn,
            [
                of_intermediates[2].cons(),
                in_of_l2l1.cons(),
                of_local.prod(),
                of_local.cons(),
                out_of_l1l2.prod(),
                gray2rgba_line_kernel,
                add_weighted_line_kernel,
            ],
        )
    )

    tensor_size = width * height * 4
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int8]]
    tensor_16x16_ty = np.ndarray[(16, 16), np.dtype[np.int32]]

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_16x16_ty, tensor_ty) as (i_in, _b, o_out):
        rt.start(*workers)
        rt.fill(in_of_l3l2.prod(), i_in)
        rt.drain(out_of_l2l3.cons(), o_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Edge Detect")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-W", "--width", type=int, default=1920)
    p.add_argument("-H", "--height", type=int, default=1080)
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    return p


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = edge_detect.specialize(width=opts.width, height=opts.height)
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _run_no_verify(opts):
    """Standalone JIT + run.  Output is not verified in Python --
    edge detection is hard to reference in numpy without re-implementing the
    whole pipeline.  Use the C++/OpenCV host (make run) for pixel-level
    checks; this mode just confirms the design compiles and executes.
    """
    tensor_size = opts.width * opts.height * 4
    rng = np.random.default_rng(0)
    in_np = rng.integers(-128, 127, size=(tensor_size,), dtype=np.int8)
    b_np = np.zeros((16 * 16,), dtype=np.int32)
    out_np = np.zeros((tensor_size,), dtype=np.int8)

    in_t = iron.tensor(in_np, dtype=np.int8, device="npu")
    b_t = iron.tensor(b_np, dtype=np.int32, device="npu")
    out_t = iron.tensor(out_np, dtype=np.int8, device="npu")

    edge_detect(in_t, b_t, out_t, width=opts.width, height=opts.height)
    print("PASS! (output not verified; use 'make run' for pixel-level checks)")


def main():
    opts = _make_argparser().parse_args()
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_no_verify(opts)


if __name__ == "__main__":
    main()
