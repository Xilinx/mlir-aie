# vision/edge_detect/edge_detect.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 AMD Inc.
"""Vision edge-detect pipeline -- ``@iron.jit`` design.

  shim --> rgba2gray --> filter2d (3x3 Laplacian) --> threshold -->
                                                     gray2rgba+addWeighted --> shim

The filter2d worker reuses an inner Buffer holding the constant Laplacian
kernel.  The gray2rgba+addWeighted worker combines the thresholded edge
map with the original RGBA input (forwarded via ``inOF_L2L1``).
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
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


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
    add_compile_args(p)
    p.add_argument("-W", "--width", type=int, default=1920)
    p.add_argument("-H", "--height", type=int, default=1080)
    return p


def _compile_kwargs(opts):
    return dict(width=opts.width, height=opts.height)


def _rgba2gray_ref(rgba_uint8, height, width):
    """Numpy port of ``rgba2gray_aie`` (SRS_SHIFT=15)."""
    rgba = rgba_uint8.reshape(height, width, 4)
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    wt_r = int(round(0.299 * (1 << 15)))  # 9798
    wt_g = int(round(0.587 * (1 << 15)))  # 19235
    wt_b = int(round(0.114 * (1 << 15)))  # 3736
    y = (wt_r * r + wt_g * g + wt_b * b + (1 << 14)) >> 15
    return np.clip(y, 0, 255).astype(np.uint8)


def _filter2d_cv_ref(gray_uint8, height, width):
    """Numpy equivalent of cv::filter2D with the unscaled Laplacian
    ``[[0,1,0],[1,-4,1],[0,1,0]]`` + BORDER_REPLICATE."""
    img = gray_uint8.reshape(height, width).astype(np.float32)
    padded = np.pad(img, 1, mode="edge")
    out = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def _threshold_binary_ref(arr_uint8, thresh, max_val):
    """cv::threshold with THRESH_BINARY: out = (in > thresh) ? max : 0."""
    return np.where(arr_uint8 > thresh, np.uint8(max_val), np.uint8(0))


def _gray2rgba_ref(gray_uint8):
    """Replicate gray to R/G/B with alpha=255 (matches ``gray2rgba_aie``)."""
    flat = gray_uint8.reshape(-1)
    out = np.zeros((flat.size, 4), dtype=np.uint8)
    out[:, 0] = flat
    out[:, 1] = flat
    out[:, 2] = flat
    out[:, 3] = 255
    return out.reshape(-1)


def _add_weighted_cv_ref(a_uint8, b_uint8, alpha, beta, gamma):
    """Numpy equivalent of cv::addWeighted: ``saturate(alpha*a + beta*b + gamma)``.
    test.cpp passes alpha=beta=1.0, gamma=0.0; the AIE kernel computes
    ``(a+b)/2`` in fixed-point — the diffs land under ``_EPSILON``.
    """
    tmp = a_uint8.astype(np.int32) * alpha + b_uint8.astype(np.int32) * beta + gamma
    return np.clip(tmp, 0, 255).astype(np.uint8)


def _edge_detect_ref(rgba_uint8, height, width):
    """End-to-end reference mirroring test.cpp's edgeDetect() OpenCV pipeline."""
    gray = _rgba2gray_ref(rgba_uint8, height, width)
    edges = _filter2d_cv_ref(gray, height, width)
    thresholded = _threshold_binary_ref(edges, 10, 255)
    mask_rgba = _gray2rgba_ref(thresholded)
    return _add_weighted_cv_ref(rgba_uint8, mask_rgba, 1, 1, 0)


# Matches the C++ test.cpp's ``epsilon = 2.0`` tolerance on
# ``error_per_pixel = sum(abs(actual - golden)) / num_pixels``.
_EPSILON = 2.0


def _run_and_verify(opts):
    tensor_size = opts.width * opts.height * 4
    rng = np.random.default_rng(0)
    in_np = rng.integers(-128, 127, size=(tensor_size,), dtype=np.int8)
    in_t = iron.tensor(in_np, dtype=np.int8, device="npu")
    b_t = iron.zeros(16 * 16, dtype=np.int32, device="npu")
    out_t = iron.zeros(tensor_size, dtype=np.int8, device="npu")

    edge_detect(in_t, b_t, out_t, **_compile_kwargs(opts))

    in_uint8 = in_np.view(np.uint8)
    expected_uint8 = _edge_detect_ref(in_uint8, opts.height, opts.width)
    actual = out_t.numpy().view(np.uint8)

    n_diff = int(np.sum(actual != expected_uint8))
    error_per_pixel = float(
        np.sum(np.abs(actual.astype(np.int32) - expected_uint8.astype(np.int32)))
    ) / (opts.width * opts.height)
    print(f"Number of differences: {n_diff}, average L1 error: {error_per_pixel:.6f}")

    assert_pass(
        error_per_pixel < _EPSILON,
        True,
        fail_msg=f"error_per_pixel {error_per_pixel:.6f} >= epsilon {_EPSILON}",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        edge_detect,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
