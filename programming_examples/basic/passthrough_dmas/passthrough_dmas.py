# passthrough_dmas/passthrough_dmas.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Passthrough DMAs — Iron API design with ``@iron.jit`` compilation.

No compute tile (in the default mode): data flows shim → memtile → shim
via ``ObjectFifo.forward()``, exercising the implicit-copy DMA path.

Three target/topology modes:

  * ``--plio none`` (default):  shim → memtile → shim, NPU or VCK5000.
  * ``--plio input``:           PLIO ObjectFifo on the input side, forwarded
                                through a compute tile to a non-PLIO shim.
                                VCK5000 only.
  * ``--plio output``:          mirror of above — non-PLIO input, PLIO output.
                                VCK5000 only.

Three invocation modes (mirrors matrix_scalar_add):

  * standalone:   ``python3 passthrough_dmas.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``       (NPU)
  * emit-MLIR:    ``... -d xcvc1902 --emit-mlir [--plio input|output]`` (vck5000)
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import AnyShimTile, Tile, from_name
from aie.dialects._aie_enum_gen import AIETileType
from aie.utils.hostruntime import set_current_device
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.verify import assert_pass

LINE_SIZE = 1024  # transfer chunk; N must be a multiple of this

# VCK5000 PLIO requires the connection to land at one of the PLIO-wired
# shim columns; the matching compute tile pulls data through.  These
# coordinates are VCK5000-specific (xcvc1902 floorplan): col 30 is a
# ShimPLTile (PLIO-wired), col 26 is a regular ShimNOCTile.
_PLIO_SHIM_COL = 30
_NON_PLIO_SHIM_COL = 26
_PLIO_COMPUTE_TILE = Tile(col=_PLIO_SHIM_COL, row=2, tile_type=AIETileType.CoreTile)
_PLIO_SHIM_TILE = Tile(col=_PLIO_SHIM_COL, row=0, tile_type=AIETileType.ShimPLTile)
_NON_PLIO_SHIM_TILE = Tile(col=_NON_PLIO_SHIM_COL, row=0)


@iron.jit
def passthrough_dmas(
    a_in: In,
    _b_unused: In,
    c_out: Out,
    *,
    n: Compile[int] = 4096,
    plio_mode: Compile[str] = "none",
):
    if plio_mode not in ("none", "input", "output"):
        raise ValueError(
            f"plio_mode must be one of 'none', 'input', 'output'; got {plio_mode!r}"
        )

    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]
    line_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]

    if plio_mode == "input":
        # PLIO on the input side; forward through compute tile to non-PLIO
        # shim on the way out.
        of_in = ObjectFifo(line_ty, name="in", plio=True)
        of_out = of_in.cons().forward(tile=_PLIO_COMPUTE_TILE)
        fill_tile = _PLIO_SHIM_TILE
        drain_tile = _NON_PLIO_SHIM_TILE
    elif plio_mode == "output":
        # Mirror: non-PLIO input forwarded through compute tile to PLIO shim.
        of_in = ObjectFifo(line_ty, name="in")
        of_out = of_in.cons().forward(tile=_PLIO_COMPUTE_TILE, plio=True)
        fill_tile = _NON_PLIO_SHIM_TILE
        drain_tile = _PLIO_SHIM_TILE
    else:
        of_in = ObjectFifo(line_ty, name="in")
        of_out = of_in.cons().forward()
        fill_tile = AnyShimTile
        drain_tile = AnyShimTile

    rt = Runtime()
    with rt.sequence(vector_ty, vector_ty, vector_ty) as (a, _, c):
        rt.fill(of_in.prod(), a, tile=fill_tile)
        rt.drain(of_out.cons(), c, tile=drain_tile, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Passthrough DMAs")
    add_compile_args(
        p,
        dev_choices=("npu", "npu2", "xcvc1902"),
        with_emit_mlir=True,
        with_elf=True,
    )
    p.add_argument("-n", "--length", type=int, default=4096, help="elements")
    p.add_argument(
        "--plio",
        type=str,
        choices=["none", "input", "output"],
        default="none",
        help="PLIO topology — only valid with -d xcvc1902 (VCK5000)",
    )
    return p


def _validate(opts):
    if opts.length % LINE_SIZE != 0:
        sys.exit(f"--length ({opts.length}) must be a multiple of {LINE_SIZE}")
    if opts.plio != "none" and opts.dev != "xcvc1902":
        sys.exit(
            f"--plio {opts.plio} requires -d xcvc1902 (VCK5000); got -d {opts.dev}"
        )


def _compile_kwargs(opts):
    return dict(n=opts.length, plio_mode=opts.plio)


def _emit_mlir(opts):
    set_current_device(from_name(opts.dev, n_cols=None if opts.dev == "npu2" else 1))
    print(passthrough_dmas.as_mlir(None, None, None, **_compile_kwargs(opts)))


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(from_name(opts.dev, n_cols=None if opts.dev == "npu2" else 1))
    spec = passthrough_dmas.specialize(**_compile_kwargs(opts))
    spec.compile(
        xclbin_path=opts.xclbin_path,
        inst_path=opts.insts_path,
        elf_path=opts.elf_path,
    )


def _run_and_verify(opts):
    in_np = np.arange(1, opts.length + 1, dtype=np.int32)
    b_np = np.zeros_like(in_np)  # unused 2nd buffer
    out_np = np.zeros_like(in_np)

    a_t = iron.tensor(in_np, dtype=np.int32, device="npu")
    b_t = iron.tensor(b_np, dtype=np.int32, device="npu")
    c_t = iron.tensor(out_np, dtype=np.int32, device="npu")

    passthrough_dmas(a_t, b_t, c_t, **_compile_kwargs(opts))

    assert_pass(c_t.numpy(), in_np, fail_msg="output does not match input")


def main():
    opts = _make_argparser().parse_args()
    _validate(opts)
    if opts.emit_mlir:
        _emit_mlir(opts)
        return
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
