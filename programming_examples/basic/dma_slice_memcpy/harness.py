# dma_slice_memcpy/harness.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Shared run/verify path for the dma_slice_memcpy designs that dispatch.

Each design states its own geometry and passes it here, so the reference the
host checks against is built from the same slice the design moves. Only the
running and checking is shared -- which is what makes a comparison between two
designs a comparison of the API and nothing else.

static_dma.py and copy_buffer.py do not use this: they address DDR themselves,
so there is no host buffer to hand them and nothing to verify.
"""

import argparse

import aie.iron as iron
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


def main(prog, design, *, shape, dtype, key):
    """Compile, run and verify ``design`` on the NPU.

    ``key`` is the slice of a ``shape`` buffer the design moves. The bytes that
    come back are checked against it, so a transfer that walked DDR incorrectly
    produces a wrong answer rather than a silent pass.
    """

    def compile_kwargs(opts):
        return dict(col=opts.col)

    def run_and_verify(opts):
        devmem = iron.randint(-128, 128, shape, dtype=dtype, device="npu")
        out = iron.zeros(devmem.numpy()[key].shape, dtype=dtype, device="npu")
        design(devmem, out, **compile_kwargs(opts))
        assert_pass(
            out.numpy(),
            devmem.numpy()[key],
            fail_msg="sliced DDR read did not match numpy",
        )

    p = argparse.ArgumentParser(prog=prog)
    add_compile_args(p, dev_choices=("npu2",), default_dev="npu2", with_emit_mlir=True)
    p.add_argument("-c", "--col", type=int, default=0)
    opts = p.parse_args()
    run_design_cli(
        design,
        opts,
        compile_kwargs=compile_kwargs,
        run_and_verify=run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )
