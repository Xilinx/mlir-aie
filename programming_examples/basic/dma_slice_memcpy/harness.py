# dma_slice_memcpy/harness.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Shared infra for the dma_slice_memcpy designs.

Each design lives in its own file and defines only what differs. The slice
itself, the geometry derived from it, and the run/verify/CLI path live here --
so the two runnable designs are checked against the same reference by the same
code, and a comparison between them is a comparison of the API and nothing else.
"""

import argparse
import math

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

DEVMEM_SHAPE = (16, 16, 4096)
DEVMEM_SLICE = np.s_[0::2, 1::2, ...]
TILE_ELEMS = 4096

# How much the slice covers, which the designs need at device scope to size
# their BD chains. The sequence bodies say the slice again, in slice notation,
# where it reads best.
IN_TAP = TensorAccessPattern.from_slice(DEVMEM_SHAPE, DEVMEM_SLICE)
SLICE_SHAPE = tuple(IN_TAP.sizes)
CHUNKS = math.prod(SLICE_SHAPE) // TILE_ELEMS

DEVMEM_TY = np.ndarray[DEVMEM_SHAPE, np.dtype[np.int8]]
SLICE_TY = np.ndarray[SLICE_SHAPE, np.dtype[np.int8]]


def add_col_arg(p):
    p.add_argument("-c", "--col", type=int, default=0)


def main(prog, design):
    """Compile, run and verify ``design`` on the NPU.

    The returned bytes are checked against the numpy slice, so a transfer that
    walked DDR incorrectly produces a wrong answer rather than a silent pass.
    """

    def compile_kwargs(opts):
        return dict(col=opts.col)

    def run_and_verify(opts):
        devmem = iron.randint(-128, 128, DEVMEM_SHAPE, dtype=np.int8, device="npu")
        out = iron.zeros(SLICE_SHAPE, dtype=np.int8, device="npu")
        design(devmem, out, **compile_kwargs(opts))
        assert_pass(
            out.numpy(),
            devmem.numpy()[DEVMEM_SLICE],
            fail_msg="sliced DDR read did not match numpy",
        )

    p = argparse.ArgumentParser(prog=prog)
    add_compile_args(p, dev_choices=("npu2",), default_dev="npu2", with_emit_mlir=True)
    add_col_arg(p)
    opts = p.parse_args()
    run_design_cli(
        design,
        opts,
        compile_kwargs=compile_kwargs,
        run_and_verify=run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )
