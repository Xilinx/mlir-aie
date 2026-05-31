# conv2d_14x14/conv2dk14.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""14x14 Conv2D — Iron API designs (single-core + 32-core) with @iron.jit.

Two parallelism modes share the same conv2dk14 sub-kernel:

  * single-core (default): one shim DMA in/out pair, one worker; processes
    896x896x4 -> 1152 channels in ~20ms.
  * 32-core (``--multi``):  8 cols x 4 rows, activations row-broadcast
    across cols, weights col-broadcast across rows, output joined per
    column; ~5ms.

The library's ``kernels.conv2dk14`` sizes per-call output as
``output_channels * tiles * 8`` (acc-byte layout), but both designs here
feed the kernel ``sub_tiles x sub_out_channels = 256`` bytes per call.
Wired via ``ExternalFunction`` with the design's actual per-call sizing.

Compile-only entrypoint:
  ``python3 conv2dk14.py -d npu2 [--multi]
            --xclbin-path build/final.xclbin --insts-path build/insts.bin``
End-to-end verification lives in ``test.py``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import aie.iron as iron
from aie.utils.ml import DataShaper
from aie.iron import Compile, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, device_from_args, from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

# Sub-tile sizing baked into the conv2dk14 kernel.
_SUB_OUT_CHANNELS = 16
_SUB_TILES = 16
_KERNEL_SIZE = 14
_IN_CHANNELS = 4
_OUT_CHANNELS = 1152
_X_BLOCKS = 4
_KERNEL_SRC = Path(__file__).resolve().parents[3] / "aie_kernels/aie2p/conv2dk14.cc"


def _conv2dk14_extern(act_in_ty, weights_ty, out_ty):
    return ExternalFunction(
        "conv2dk14_i8",
        source_file=str(_KERNEL_SRC),
        arg_types=[
            act_in_ty,
            weights_ty,
            out_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
        include_dirs=[config.cxx_header_path()],
        compile_flags=["-DUINT8_ACT"],
    )


@iron.jit
def conv2dk14(
    a_in: In,
    w_in: In,
    b_out: Out,
    *,
    width: Compile[int] = 896,
    height: Compile[int] = 896,
    scale: Compile[int] = 14,
):
    if width % 8 != 0 or width < 8:
        raise ValueError("width must be a multiple of 8 and >= 8")
    if height % 8 != 0 or height < 8:
        raise ValueError("height must be a multiple of 8 and >= 8")

    device = iron.get_current_device()

    act_in = _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS * _SUB_TILES
    weights = _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS * _SUB_OUT_CHANNELS
    act_out = _SUB_TILES * _SUB_OUT_CHANNELS

    out_channels_group = _OUT_CHANNELS // _SUB_OUT_CHANNELS
    width_out = width // _KERNEL_SIZE
    height_out = height // _KERNEL_SIZE

    tensor_in_size = width * height * _IN_CHANNELS * out_channels_group
    tensor_wts_size = weights * out_channels_group
    tensor_out_size = width_out * height_out * _SUB_OUT_CHANNELS * out_channels_group

    buf_in = _KERNEL_SIZE * width * _IN_CHANNELS
    buf_out = _SUB_OUT_CHANNELS * width_out * height_out

    act_in_ty = np.ndarray[(act_in,), np.dtype[np.uint8]]
    buf_in_ty = np.ndarray[(buf_in,), np.dtype[np.uint8]]
    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]
    out_ty = np.ndarray[(act_out,), np.dtype[np.int8]]
    buf_out_ty = np.ndarray[(buf_out,), np.dtype[np.int8]]
    tensor_in_ty = np.ndarray[(tensor_in_size,), np.dtype[np.uint8]]
    tensor_wts_ty = np.ndarray[(tensor_wts_size,), np.dtype[np.int8]]
    tensor_out_ty = np.ndarray[(tensor_out_size,), np.dtype[np.int8]]

    conv_fn = _conv2dk14_extern(act_in_ty, weights_ty, out_ty)

    of_act_l3l2 = ObjectFifo(
        buf_in_ty,
        name="inOF_act_L3L2",
        dims_from_stream_per_cons=[
            (_KERNEL_SIZE, _KERNEL_SIZE * _IN_CHANNELS),
            (64, _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS),
            (_KERNEL_SIZE * _IN_CHANNELS, 1),
        ],
    )
    of_act_l2 = of_act_l3l2.cons().forward(
        obj_type=act_in_ty,
        name="act_L2_02",
        dims_to_stream=[
            (2, _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS * 8),
            (_KERNEL_SIZE * _KERNEL_SIZE // 2, 2 * _IN_CHANNELS),
            (8, _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS),
            (2 * _IN_CHANNELS, 1),
        ],
    )

    of_wts_l3l2 = ObjectFifo(weights_ty, depth=1, name="inOF_wts_0_L3L2")

    of_out_l2 = ObjectFifo(out_ty, name="out_02_L2")
    of_out_l3 = of_out_l2.cons().forward(
        obj_type=buf_out_ty,
        name="outOFL2L3",
        dims_to_stream=[(256, 256), (16, 8), (2, 128), (8, 1)],
    )

    def core_fn(of_wts, of_act, of_out, kernel):
        x_dim = width // _X_BLOCKS
        elem_wts = of_wts.acquire(1)
        for _ in range_(height // _KERNEL_SIZE):
            for _ in range_(_X_BLOCKS):
                elem_in = of_act.acquire(1)
                elem_out = of_out.acquire(1)
                kernel(
                    elem_in,
                    elem_wts,
                    elem_out,
                    x_dim,
                    _IN_CHANNELS,
                    _SUB_OUT_CHANNELS,
                    _KERNEL_SIZE,
                    scale,
                )
                of_act.release(1)
                of_out.release(1)
        of_wts.release(1)

    worker = Worker(
        core_fn,
        [of_wts_l3l2.cons(), of_act_l2.cons(), of_out_l2.prod(), conv_fn],
        stack_size=0x600,
    )

    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_wts_ty, tensor_out_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_act_l3l2.prod(), I)
        rt.fill(of_wts_l3l2.prod(), W)
        rt.drain(of_out_l3.cons(), O, wait=True)

    return Program(device, rt).resolve_program()


@iron.jit
def conv2dk14_multi(
    a_in: In,
    w_in: In,
    b_out: Out,
    *,
    width: Compile[int] = 896,
    height: Compile[int] = 896,
    scale: Compile[int] = 14,
):
    """32-core (8 cols x 4 rows) variant.

    Activations are split across 4 rows (one shim DMA per row), with each
    row's memtile broadcasting to all 8 cores in that row. Weights are
    split across 8 cols (one shim DMA per col, no memtile), broadcasting
    directly to the 4 cores in each col. Outputs are joined per column:
    each col's 4 row-cores feed into its memtile, then out to its shim
    DMA. Each core does 1/4 of the height work.
    """
    if width % 8 != 0 or width < 8:
        raise ValueError("width must be a multiple of 8 and >= 8")
    if height % 8 != 0 or height < 8:
        raise ValueError("height must be a multiple of 8 and >= 8")

    device = iron.get_current_device()
    n_cols, n_rows = 8, 4
    if device.cols < n_cols:
        raise ValueError(
            f"multi-core variant needs {n_cols} columns; device has {device.cols}"
        )

    act_in = _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS * _SUB_TILES
    weights = _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS * _SUB_OUT_CHANNELS
    act_out = _SUB_TILES * _SUB_OUT_CHANNELS

    act_repeat = (_OUT_CHANNELS // _SUB_OUT_CHANNELS) // n_cols  # 9

    out_channels_group = _OUT_CHANNELS // _SUB_OUT_CHANNELS  # 72
    width_out = width // _KERNEL_SIZE
    height_out = height // _KERNEL_SIZE

    tensor_in_size = (
        width * height * _IN_CHANNELS
    )  # one image, replayed via shim repeat
    tensor_wts_size = weights * out_channels_group
    tensor_out_size = width_out * height_out * _SUB_OUT_CHANNELS * out_channels_group

    buf_in_row = _KERNEL_SIZE * width * _IN_CHANNELS  # one tile-row of pixels

    act_in_ty = np.ndarray[(act_in,), np.dtype[np.uint8]]
    buf_in_ty = np.ndarray[(buf_in_row,), np.dtype[np.uint8]]
    weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]
    out_ty = np.ndarray[(act_out,), np.dtype[np.int8]]
    out_mem_ty = np.ndarray[(act_out * 4 * 16 * n_rows,), np.dtype[np.int8]]
    tensor_in_ty = np.ndarray[(tensor_in_size,), np.dtype[np.uint8]]
    tensor_wts_ty = np.ndarray[(tensor_wts_size,), np.dtype[np.int8]]
    tensor_out_ty = np.ndarray[(tensor_out_size,), np.dtype[np.int8]]

    conv_fn = _conv2dk14_extern(act_in_ty, weights_ty, out_ty)

    # Activations: per-row shim -> per-row memtile -> broadcast to 8 cores
    of_act_l3l2 = [None] * n_rows
    of_act_l2l1 = [None] * n_rows
    for j in range(n_rows):
        of_act_l3l2[j] = ObjectFifo(
            buf_in_ty,
            name=f"of_act_L3L2_{j}",
            dims_from_stream_per_cons=[
                (_KERNEL_SIZE, _KERNEL_SIZE * _IN_CHANNELS),
                (64, _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS),
                (_KERNEL_SIZE * _IN_CHANNELS, 1),
            ],
        )
        of_act_l2l1[j] = (
            of_act_l3l2[j]
            .cons()
            .forward(
                obj_type=act_in_ty,
                name=f"of_act_L2L1_{j}",
                dims_to_stream=[
                    (2, _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS * 8),
                    (_KERNEL_SIZE * _KERNEL_SIZE // 2, 2 * _IN_CHANNELS),
                    (8, _KERNEL_SIZE * _KERNEL_SIZE * _IN_CHANNELS),
                    (2 * _IN_CHANNELS, 1),
                ],
                tile=Tile(j, 1),
            )
        )

    # Weights: per-col shim -> broadcast directly to 4 row-cores in that col (no memtile)
    of_wts = [ObjectFifo(weights_ty, name=f"of_wts_L3L1_{i}") for i in range(n_cols)]

    # Outputs: per-col join across 4 rows in memtile, then to per-col shim
    of_out_l2l3 = [None] * n_cols
    of_out_l1l2 = [[None] * n_cols for _ in range(n_rows)]
    out_offsets = [act_out * 4 * 16 * j for j in range(n_rows)]
    for i in range(n_cols):
        of_out_l2l3[i] = ObjectFifo(
            out_mem_ty,
            name=f"of_out_L2L3_{i}",
            dims_to_stream=[(64, 256), (16, 8), (2, 128), (8, 1)],
        )
        col_fifos = (
            of_out_l2l3[i]
            .prod()
            .join(
                out_offsets,
                obj_types=[out_ty] * n_rows,
                names=[f"of_out_L1L2_{j}_{i}" for j in range(n_rows)],
                tile=Tile(i, 1),
            )
        )
        for j in range(n_rows):
            of_out_l1l2[j][i] = col_fifos[j]

    def core_fn(of_wts_in, of_act, of_out, kernel):
        y_dim = height // (_KERNEL_SIZE * n_rows)
        x_dim = width // _X_BLOCKS
        elem_wts = of_wts_in.acquire(1)
        for _ in range_(y_dim):
            for _ in range_(_X_BLOCKS):
                elem_in = of_act.acquire(1)
                elem_out = of_out.acquire(1)
                kernel(
                    elem_in,
                    elem_wts,
                    elem_out,
                    x_dim,
                    _IN_CHANNELS,
                    _SUB_OUT_CHANNELS,
                    _KERNEL_SIZE,
                    scale,
                )
                of_act.release(1)
                of_out.release(1)
        of_wts_in.release(1)

    workers = Worker.grid(
        n_rows,
        n_cols,
        lambda j, i: Worker(
            core_fn,
            [
                of_wts[i].cons(),
                of_act_l2l1[j].cons(),
                of_out_l1l2[j][i].prod(),
                conv_fn,
            ],
            tile=Tile(i, 2 + j),
            stack_size=0xC00,
        ),
    )

    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_wts_ty, tensor_out_ty) as (I, W, O):
        rt.start(*[w for row in workers for w in row])
        row_chunk = tensor_in_size // n_rows
        wts_chunk = tensor_wts_size // n_cols
        out_chunk = tensor_out_size // n_cols
        for j in range(n_rows):
            tap = TensorAccessPattern(
                (1, tensor_in_size),
                row_chunk * j,
                [act_repeat, 1, 1, row_chunk],
                [0, 0, 0, 1],
            )
            rt.fill(of_act_l3l2[j].prod(), I, tap)
        for i in range(n_cols):
            wts_tap = TensorAccessPattern(
                (1, tensor_wts_size),
                wts_chunk * i,
                [1, 1, 1, wts_chunk],
                [0, 0, 0, 1],
            )
            out_tap = TensorAccessPattern(
                (1, tensor_out_size),
                out_chunk * i,
                [1, 1, 1, out_chunk],
                [0, 0, 0, 1],
            )
            rt.fill(of_wts[i].prod(), W, wts_tap)
            rt.drain(of_out_l2l3[i].cons(), O, out_tap, wait=True)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Conv2D 14x14 (aie2p)")
    add_compile_args(p)
    p.add_argument("-wd", "--width", type=int, default=896)
    p.add_argument("-ht", "--height", type=int, default=896)
    p.add_argument("--scale", type=int, default=14)
    p.add_argument("--multi", action="store_true", help="32-core (8x4) variant")
    return p


def _compile_kwargs(opts):
    return dict(width=opts.width, height=opts.height, scale=opts.scale)


# Quantization scales used by the per-channel int8 conv reference.  Lifted
# from test.py (kept in sync so this standalone path matches that harness).
_CONV_SCALE = 1.9073e-06
_INT8_SCALE = 0.03125


def _run_and_verify(opts):
    """Compile, run on NPU, and check against a torch Conv2d golden."""
    design = conv2dk14_multi if opts.multi else conv2dk14
    width, height, ksz = int(opts.width), int(opts.height), int(opts.scale)
    ci, co = _IN_CHANNELS, _OUT_CHANNELS
    height_out, width_out = height // ksz, width // ksz
    co_group = co // _SUB_OUT_CHANNELS

    # Single-core variant replicates the image co_group times in the input
    # tensor; multi-core variant uses shim repeat and ships one image.
    num_act = 1 if opts.multi else co_group

    torch.manual_seed(0)
    int_inp = torch.randint(0, 255, (1, ci, height, width)).float()
    int_weight = torch.randint(2, 20, (co, ci, ksz, ksz)).float()

    model = nn.Conv2d(ci, co, kernel_size=ksz, stride=ksz, padding=0, bias=False)
    with torch.no_grad():
        model.weight.copy_(int_weight)
        out_int = model(int_inp)
    out_quant = out_int * _CONV_SCALE
    golden = (
        (_INT8_SCALE * torch.clamp(torch.round(out_quant / _INT8_SCALE), -128, 127))
        .squeeze(0)
        .numpy()
    )

    ds = DataShaper()
    int_inp_np = int_inp.squeeze(0).numpy().astype(np.uint8)
    ifm_aie = ds.reorder_mat(int_inp_np, "YXC", "CYX")
    ifm_flat = np.tile(ifm_aie, num_act).flatten()

    int_weight_np = int_weight.numpy().astype(np.int8)
    wts_aie = ds.reorder_mat(int_weight_np, "OYXIO8", "OIYX")
    wts_flat = np.concatenate(wts_aie, axis=None).flatten()

    in_t = iron.tensor(ifm_flat, dtype=np.uint8)
    w_t = iron.tensor(wts_flat, dtype=np.int8)
    out_size = co_group * height_out * width_out * _SUB_OUT_CHANNELS
    o_t = iron.zeros(out_size, dtype=np.int8)

    design(in_t, w_t, o_t, width=width, height=height, scale=ksz)

    aie_raw = o_t.numpy().reshape(co_group, height_out, width_out, _SUB_OUT_CHANNELS)
    aie_cyxd = ds.reorder_mat(aie_raw, "CDYX", "CYXD")
    aie_out = aie_cyxd.reshape(co, height_out, width_out) * _INT8_SCALE

    max_diff = float(np.max(np.abs(aie_out - golden)))
    if np.allclose(aie_out, golden, rtol=0, atol=2 * _INT8_SCALE):
        print(f"\nPASS! (max_abs_diff={max_diff:.4f})\n")
    else:
        print(f"\nFAILED (max_abs_diff={max_diff:.4f}, tol={2 * _INT8_SCALE:.4f})\n")
        sys.exit(-1)


def main():
    opts = _make_argparser().parse_args()
    design = conv2dk14_multi if opts.multi else conv2dk14
    run_design_cli(
        design,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=8 if opts.multi else 1),
    )


if __name__ == "__main__":
    main()
