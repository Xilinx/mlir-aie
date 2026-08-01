# dwconv1d/dwconv1d.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Depthwise conv1d, 'same' padding, stride 1, bf16, IRON API + ``@iron.jit``.

NPU2-only: ``aie_kernels/aie2p/dwconv1d.cc`` has no aie2 counterpart.

``n_cores`` cores each process ``channels // n_cores`` channels; one channel
is one length-``seq_len`` time series with its own ``kernel_size`` taps (+ an
optional bias). Per channel:

    out[t] = bias + sum_{p=0..K-1} w[p] * x[t + p],   t = 0 .. seq_len-1

which is a cross-correlation (no kernel flip) over ``x`` zero-padded by
``(kernel_size - 1) // 2`` on each side, matching ``torch.nn.Conv1d`` /
most framework "same" depthwise convs.

``x_in`` is passed to this design ALREADY padded: shape ``(channels,
seq_len + 16)``. The extra 16 (fixed, independent of ``kernel_size``) is
scratch room the vectorized kernel's aligned loads need past the halo; see
``dwconv1d.cc`` and ``_pad_input`` below, which is both the reference
construction and this file's own test input.
"""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D
from aie.utils import config
from aie.utils.hostruntime.argparse import device_from_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_KERNEL_SRC = Path(__file__).resolve().parents[3] / "aie_kernels/aie2p/dwconv1d.cc"
_TAIL_SLACK = 16  # matches the kernel's fixed aligned-load window, see dwconv1d.cc


def _dwconv1d_extern(chunk_in_ty, w_ty, chunk_out_ty, kernel_size, bias):
    return ExternalFunction(
        "dwconv1d_bf16",
        source_file=str(_KERNEL_SRC),
        arg_types=[chunk_in_ty, w_ty, chunk_out_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DDWCONV_K={kernel_size}",
            f"-DDWCONV_BIAS={int(bias)}",
        ],
    )


@iron.jit
def dwconv1d(
    x_in: In,
    w_in: In,
    y_out: Out,
    *,
    channels: CompileTime[int] = 64,
    seq_len: CompileTime[int] = 128,
    kernel_size: CompileTime[int] = 9,
    bias: CompileTime[bool] = True,
    n_cores: CompileTime[int] = 8,
):
    if channels % n_cores != 0:
        raise ValueError(f"channels ({channels}) must be a multiple of n_cores ({n_cores})")
    if seq_len % 16 != 0:
        # The kernel processes full 16-lane output chunks with no scalar
        # tail (see dwconv1d.cc), so a non-multiple would silently drop the
        # last columns.
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of 16")
    if kernel_size < 1 or kernel_size > 17 or kernel_size % 2 == 0:
        # Odd so 'same' padding is symmetric; <= 17 so a tap window (16 +
        # kernel_size - 1) fits the kernel's 32-lane window.
        raise ValueError(f"kernel_size ({kernel_size}) must be odd and in [1, 17]")

    device = iron.get_current_device()
    channels_per_core = channels // n_cores
    in_row = seq_len + _TAIL_SLACK
    # w_row is always kernel_size + 1, even when bias=False: kernel_size is
    # required odd (see the check above), so kernel_size alone gives an odd
    # bf16 row width, i.e. 2*kernel_size bytes, not a multiple of 4, which
    # aie.dma_bd's transfer-length check rejects. kernel_size + 1 is always
    # even, so 2*(kernel_size + 1) bytes is 4-byte aligned. The extra column
    # is unused padding when bias=False (dwconv1d.cc never reads w[K] unless
    # DWCONV_BIAS=1); callers should zero it, see _run_and_verify below.
    w_row = kernel_size + 1

    dtype = bfloat16
    x_tensor_ty = np.ndarray[(channels, in_row), np.dtype[dtype]]
    w_tensor_ty = np.ndarray[(channels, w_row), np.dtype[dtype]]
    y_tensor_ty = np.ndarray[(channels, seq_len), np.dtype[dtype]]

    # One ObjectFifo tile == one channel's row (matches the taps-differ-per-
    # channel shape of the op); each core loops channels_per_core times,
    # acquiring/releasing one channel at a time.
    x_row_ty = np.ndarray[(in_row,), np.dtype[dtype]]
    w_row_ty = np.ndarray[(w_row,), np.dtype[dtype]]
    y_row_ty = np.ndarray[(seq_len,), np.dtype[dtype]]

    of_ins = [ObjectFifo(x_row_ty, name=f"x_{i}") for i in range(n_cores)]
    of_ws = [ObjectFifo(w_row_ty, name=f"w_{i}") for i in range(n_cores)]
    of_outs = [ObjectFifo(y_row_ty, name=f"y_{i}") for i in range(n_cores)]

    dwconv_fn = _dwconv1d_extern(x_row_ty, w_row_ty, y_row_ty, kernel_size, bias)

    def core_fn(of_x, of_w, of_y, kernel):
        for _ in range_(channels_per_core):
            elem_x = of_x.acquire(1)
            elem_w = of_w.acquire(1)
            elem_y = of_y.acquire(1)
            kernel(elem_x, elem_w, elem_y, seq_len)
            of_x.release(1)
            of_w.release(1)
            of_y.release(1)

    workers = [
        Worker(core_fn, [of_ins[i].cons(), of_ws[i].cons(), of_outs[i].prod(), dwconv_fn])
        for i in range(n_cores)
    ]

    # Each core's block is channels_per_core contiguous channels; the
    # (channels_per_core, *) block streams through the core's per-channel
    # ObjectFifo as channels_per_core row-tiles (same fill-a-block /
    # acquire-a-row-at-a-time split the row_ty ObjectFifos above assume).
    x_taps = TensorTiler2D.simple_tiler((channels, in_row), (channels_per_core, in_row))
    w_taps = TensorTiler2D.simple_tiler((channels, w_row), (channels_per_core, w_row))
    y_taps = TensorTiler2D.simple_tiler((channels, seq_len), (channels_per_core, seq_len))

    def sequence(X, W, Y, in_prods, w_prods, out_conses):
        for i in range(n_cores):
            in_prods[i].fill(X, x_taps[i])
            w_prods[i].fill(W, w_taps[i])
        for i in range(n_cores):
            out_conses[i].drain(Y, y_taps[i], wait=True)

    rt = Runtime(
        sequence,
        [
            x_tensor_ty,
            w_tensor_ty,
            y_tensor_ty,
            [of_ins[i].prod() for i in range(n_cores)],
            [of_ws[i].prod() for i in range(n_cores)],
            [of_outs[i].cons() for i in range(n_cores)],
        ],
    )

    return Program(device, rt, workers=workers).resolve_program()


def _pad_input(x_np, kernel_size):
    """(channels, seq_len) -> (channels, seq_len + 16): zero-pad by
    (kernel_size - 1) // 2 on each side (the 'same' halo) plus the fixed
    16-element don't-care tail dwconv1d.cc's aligned loads need. This is the
    reference construction any caller of this design must follow, see the
    file docstring and dwconv1d.cc's header comment for why."""
    channels, seq_len = x_np.shape
    pad = (kernel_size - 1) // 2
    out = np.zeros((channels, seq_len + _TAIL_SLACK), dtype=x_np.dtype)
    out[:, pad : pad + seq_len] = x_np
    return out


def _dwconv1d_reference(x_np, w_np, kernel_size, bias):
    """Plain f32 'same' depthwise cross-correlation, no padding trick.
    This is the ground truth this design (and dwconv1d.cc) is computing."""
    channels, seq_len = x_np.shape
    pad = (kernel_size - 1) // 2
    x32 = x_np.astype(np.float32)
    xp = np.zeros((channels, seq_len + 2 * pad), dtype=np.float32)
    xp[:, pad : pad + seq_len] = x32
    w32 = w_np.astype(np.float32)
    out = np.zeros((channels, seq_len), dtype=np.float32)
    for p in range(kernel_size):
        out += w32[:, p : p + 1] * xp[:, p : p + seq_len]
    if bias:
        out += w32[:, kernel_size : kernel_size + 1]
    return out.astype(bfloat16)


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Depthwise Conv1d")
    add_compile_args(p, with_elf=True)
    p.add_argument("-c", "--channels", type=int, default=64)
    p.add_argument("-s", "--seq_len", type=int, default=128)
    p.add_argument("-k", "--kernel_size", type=int, default=9)
    p.add_argument("--no_bias", action="store_true")
    p.add_argument("-n", "--n_cores", type=int, default=8)
    return p


def _compile_kwargs(opts):
    return dict(
        channels=opts.channels,
        seq_len=opts.seq_len,
        kernel_size=opts.kernel_size,
        bias=not opts.no_bias,
        n_cores=opts.n_cores,
    )


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    channels, seq_len, K, bias = opts.channels, opts.seq_len, opts.kernel_size, not opts.no_bias
    w_row = K + 1  # always K+1, see the w_row comment in dwconv1d() above

    x_np = rng.uniform(-2.0, 2.0, size=(channels, seq_len)).astype(bfloat16)
    w_np = rng.uniform(-1.0, 1.0, size=(channels, w_row)).astype(bfloat16)
    if not bias:
        w_np[:, K] = bfloat16(0.0)  # padding column dwconv1d.cc never reads

    x_padded = _pad_input(x_np, K)

    x_t = iron.tensor(x_padded, dtype=bfloat16, device="npu")
    w_t = iron.tensor(w_np, dtype=bfloat16, device="npu")
    y_t = iron.tensor(np.zeros((channels, seq_len), dtype=bfloat16), dtype=bfloat16, device="npu")

    dwconv1d(x_t, w_t, y_t, **_compile_kwargs(opts))

    out = y_t.numpy().reshape(channels, seq_len)
    ref = _dwconv1d_reference(x_np, w_np, K, bias)
    assert_pass(out, ref, atol=0.05, fail_msg="dwconv1d output mismatch")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        dwconv1d,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
