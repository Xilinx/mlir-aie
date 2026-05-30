# vision/color_detect/color_detect.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 AMD Inc.
"""Color detect -- ``@iron.jit`` HSV-hue-mask + bitwise-blend pipeline.

A 4-worker line-based pipeline on a single column:

  shim --> rgba2hue --> (threshold-upper, threshold-lower in parallel) -->
           bitwiseOR --> gray2rgba --> bitwiseAND(original_rgba, mask) --> shim

The two threshold workers run in parallel on independent tiles with
different thresholds; the OR of their outputs is then expanded to RGBA
and AND-ed pixel-wise with the original input (carried forward via
``inOF_L2L1``) to produce the color-detected output.  All 5 kernels are
pulled from ``iron.kernels.vision``.

``aiecc_flags=["--alloc-scheme=basic-sequential"]`` matches the pre-merge
Makefile's aiecc invocation.

Two invocation modes:

  * standalone:   ``python3 color_detect.py``  (JIT-compile + run + verify
                  against a per-stage numpy reference mirroring the scalar
                  ``rgba2hue_aie_scalar`` formula plus threshold / OR /
                  gray2rgba / AND).
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile).
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.device import device_from_args
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def color_detect(
    in_tensor: In,
    _b_unused: In,
    out_tensor: Out,
    *,
    width: Compile[int] = 1920,
    height: Compile[int] = 1080,
):
    line_width = width
    line_width_in_bytes = width * 4  # 4 channels (RGBA)
    tensor_size = width * height * 4

    line_bytes_ty = np.ndarray[(line_width_in_bytes,), np.dtype[np.uint8]]
    line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int8]]
    tensor_16x16_ty = np.ndarray[(16, 16), np.dtype[np.int32]]

    rgba2hue_line = kernels.rgba2hue(line_width=line_width)
    threshold_line = kernels.threshold(line_width=line_width, dtype=np.uint8)
    bitwise_or_line = kernels.bitwise_or(line_width=line_width, dtype=np.uint8)
    gray2rgba_line = kernels.gray2rgba(line_width=line_width)
    bitwise_and_line = kernels.bitwise_and(
        line_width=line_width_in_bytes, dtype=np.uint8
    )

    in_of_l3l2 = ObjectFifo(line_bytes_ty, name="inOF_L3L2")
    in_of_l2l1 = in_of_l3l2.cons(6).forward(depth=6, name="inOF_L2L1")
    out_of_l1l2 = ObjectFifo(line_bytes_ty, name="outOF_L1L2")
    out_of_l2l3 = out_of_l1l2.cons().forward(name="outOF_L2L3")

    of_2to34 = ObjectFifo(line_ty, name="OF_2to34")
    of_3to3 = ObjectFifo(line_ty, name="OF_3to3", depth=1)
    of_3to5 = ObjectFifo(line_ty, name="OF_3to5")
    of_4to4 = ObjectFifo(line_ty, name="OF_4to4", depth=1)
    of_4to5 = ObjectFifo(line_ty, name="OF_4to5")
    of_5to5a = ObjectFifo(line_ty, name="OF_5to5a", depth=1)
    of_5to5b = ObjectFifo(line_bytes_ty, name="OF_5to5b", depth=1)

    def rgba2hue_fn(of_in, of_out, rgba2hue_kernel):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        rgba2hue_kernel(elem_in, elem_out, line_width)
        of_in.release(1)
        of_out.release(1)

    worker2 = Worker(rgba2hue_fn, [in_of_l3l2.cons(), of_2to34.prod(), rgba2hue_line])

    def threshold_fn(of_in, of_in3, of_out3, of_out5, threshold_kernel, is_first=True):
        if is_first:
            threshold_value_upper1 = 40
            threshold_value_lower1 = 30
        else:
            threshold_value_upper1 = 160
            threshold_value_lower1 = 90
        threshold_maxvalue = 255
        threshold_mode_to_zero_inv = 4
        threshold_mode_binary = 0

        elem_in = of_in.acquire(1)
        elem_out_tmp = of_in3.acquire(1)
        threshold_kernel(
            elem_in,
            elem_out_tmp,
            line_width,
            threshold_value_upper1,
            threshold_maxvalue,
            threshold_mode_to_zero_inv,
        )
        of_in.release(1)
        of_in3.release(1)
        elem_in_tmp = of_out3.acquire(1)
        elem_out = of_out5.acquire(1)
        threshold_kernel(
            elem_in_tmp,
            elem_out,
            line_width,
            threshold_value_lower1,
            threshold_maxvalue,
            threshold_mode_binary,
        )
        of_out3.release(1)
        of_out5.release(1)

    worker3 = Worker(
        threshold_fn,
        [
            of_2to34.cons(),
            of_3to3.prod(),
            of_3to3.cons(),
            of_3to5.prod(),
            threshold_line,
            True,
        ],
    )

    worker4 = Worker(
        threshold_fn,
        [
            of_2to34.cons(),
            of_4to4.prod(),
            of_4to4.cons(),
            of_4to5.prod(),
            threshold_line,
            False,
        ],
    )

    def or_gray2rgba_and_fn(
        of_in,
        of_in2,
        of_in_self,
        of_out_self,
        of_in_self2,
        of_out_self2,
        of_in3,
        of_out,
        bitwise_or_kernel,
        gray2rgba_kernel,
        bitwise_and_kernel,
    ):
        elem_in1 = of_in.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out_tmp_a = of_in_self.acquire(1)
        bitwise_or_kernel(elem_in1, elem_in2, elem_out_tmp_a, line_width)
        of_in.release(1)
        of_in2.release(1)
        of_in_self.release(1)

        elem_in_tmp_a = of_out_self.acquire(1)
        elem_out_tmp_b = of_in_self2.acquire(1)
        gray2rgba_kernel(elem_in_tmp_a, elem_out_tmp_b, line_width)
        of_out_self.release(1)
        of_in_self2.release(1)

        elem_in_tmp_b1 = of_out_self2.acquire(1)
        elem_in_tmp_b2 = of_in3.acquire(1)
        elem_out = of_out.acquire(1)
        bitwise_and_kernel(
            elem_in_tmp_b1, elem_in_tmp_b2, elem_out, line_width_in_bytes
        )
        of_out_self2.release(1)
        of_in3.release(1)
        of_out.release(1)

    worker5 = Worker(
        or_gray2rgba_and_fn,
        [
            of_3to5.cons(),
            of_4to5.cons(),
            of_5to5a.prod(),
            of_5to5a.cons(),
            of_5to5b.prod(),
            of_5to5b.cons(),
            in_of_l2l1.cons(),
            out_of_l1l2.prod(),
            bitwise_or_line,
            gray2rgba_line,
            bitwise_and_line,
        ],
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_16x16_ty, tensor_ty) as (i_in, _b, o_out):
        rt.start(worker2, worker3, worker4, worker5)
        rt.fill(in_of_l3l2.prod(), i_in)
        rt.drain(out_of_l2l3.cons(), o_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Color Detect")
    add_compile_args(p)
    p.add_argument("-W", "--width", type=int, default=1920)
    p.add_argument("-H", "--height", type=int, default=1080)
    return p


def _compile_kwargs(opts):
    return dict(width=opts.width, height=opts.height)


def _rgba2hue_ref(rgba_uint8):
    """Numpy port of ``rgba2hue_aie_scalar`` from aie_kernels/aie2/rgba2hue.cc.

    The vectorized variant runs only on AIE2 (NPU1); AIE2P (NPU2) takes the
    scalar fallback via ``#ifdef __AIE2__`` in the kernel.  This is the
    scalar formula.
    """
    rgba = rgba_uint8.reshape(-1, 4)
    r = rgba[:, 0].astype(np.int32)
    g = rgba[:, 1].astype(np.int32)
    b = rgba[:, 2].astype(np.int32)
    rgb_max = np.maximum.reduce([r, g, b])
    rgb_min = np.minimum.reduce([r, g, b])
    rng = rgb_max - rgb_min
    # Guard div-by-zero; kernel sets h=0 when rgb_max==0 or rgb_max==rgb_min.
    safe_rng = np.where(rng == 0, 1, rng)
    h = np.zeros_like(rgb_max)
    is_r = (rgb_max == r) & (rng != 0)
    is_g = (rgb_max == g) & (rng != 0) & ~is_r
    is_b = (rgb_max == b) & (rng != 0) & ~is_r & ~is_g

    # C++ ``int / int`` truncates toward zero; Python ``//`` floors.  Use
    # np.trunc-via-float so negative dividends behave like the kernel.
    def _trunc_div(num, den):
        return np.trunc(num.astype(np.float64) / den.astype(np.float64)).astype(
            np.int32
        )

    np.copyto(h, 0 + _trunc_div(85 * (g - b), safe_rng), where=is_r)
    np.copyto(h, 85 * 2 + _trunc_div(85 * (b - r), safe_rng), where=is_g)
    np.copyto(h, 170 * 2 + _trunc_div(85 * (r - g), safe_rng), where=is_b)
    np.copyto(h, 0, where=(rgb_max == 0) | (rgb_max == rgb_min))
    # Kernel does ``h = (h + 1) >> 1`` before storing -- halve with rounding.
    h = (h + 1) >> 1
    return h.astype(np.uint8)


def _threshold_ref(arr_uint8, thresh, max_val, mode):
    """threshold.cc reference (BIT_WIDTH=8 variants).  Modes used here:

    * mode 0 (BINARY):    ``out = (in > thresh) ? max : 0``
    * mode 4 (TOZERO_INV): ``out = (in > thresh) ? 0 : in``
    """
    if mode == 0:
        return np.where(arr_uint8 > thresh, np.uint8(max_val), np.uint8(0))
    if mode == 4:
        return np.where(arr_uint8 > thresh, np.uint8(0), arr_uint8)
    raise ValueError(f"threshold mode {mode} not modeled in reference")


def _gray2rgba_ref(gray_uint8):
    """Replicate gray to R, G, B with alpha = 255 (matches gray2rgba_aie)."""
    out = np.zeros((gray_uint8.size, 4), dtype=np.uint8)
    out[:, 0] = gray_uint8
    out[:, 1] = gray_uint8
    out[:, 2] = gray_uint8
    out[:, 3] = 255
    return out.reshape(-1)


def _color_detect_ref(rgba_uint8):
    """End-to-end pipeline reference matching the @iron.jit design."""
    hue = _rgba2hue_ref(rgba_uint8)
    t1a = _threshold_ref(hue, 40, 255, 4)
    t1b = _threshold_ref(t1a, 30, 255, 0)
    t2a = _threshold_ref(hue, 160, 255, 4)
    t2b = _threshold_ref(t2a, 90, 255, 0)
    mask = np.bitwise_or(t1b, t2b)
    mask_rgba = _gray2rgba_ref(mask)
    return np.bitwise_and(mask_rgba, rgba_uint8)


def _run_and_verify(opts):
    """JIT-compile + run + verify against the per-stage numpy reference."""
    tensor_size = opts.width * opts.height * 4
    rng = np.random.default_rng(0)
    in_np = rng.integers(-128, 127, size=(tensor_size,), dtype=np.int8)
    b_np = np.zeros((16 * 16,), dtype=np.int32)
    out_np = np.zeros((tensor_size,), dtype=np.int8)

    in_t = iron.tensor(in_np, dtype=np.int8, device="npu")
    b_t = iron.tensor(b_np, dtype=np.int32, device="npu")
    out_t = iron.tensor(out_np, dtype=np.int8, device="npu")

    color_detect(in_t, b_t, out_t, **_compile_kwargs(opts))

    in_uint8 = in_np.view(np.uint8)
    expected_uint8 = _color_detect_ref(in_uint8)
    actual = out_t.numpy().view(np.uint8)
    n_mismatch = int(np.sum(actual != expected_uint8))
    assert_pass(
        actual,
        expected_uint8,
        fail_msg=f"{n_mismatch} byte(s) mismatch the per-stage color-detect reference",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        color_detect,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
