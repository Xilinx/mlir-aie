# test_hostruntime_cli_compile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# REQUIRES: peano
# RUN: %pytest %s

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import DispatchTime, In, ObjectFifo, Out, Program, Runtime
from aie.utils import get_current_device, set_current_device
from aie.utils.hostruntime import cli


@pytest.fixture
def keep_device():
    previous = get_current_device(probe_runtime=False)
    try:
        yield
    finally:
        set_current_device(previous)


def _forwarded_copy():
    @iron.jit
    def copy(a: In, b: Out, *, n: DispatchTime[np.int32] = 1):
        ty = np.ndarray[(1024,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(256,), np.dtype[np.int32]]
        of_in = ObjectFifo(tile_ty)
        of_out = of_in.cons().forward()

        def sequence(a_h, b_h, n, a_in, b_out):
            tiles = TensorAccessPattern.full((1024,)).split(0, 256)
            a_in.fill(a_h, tap=tiles[:n])
            b_out.drain(b_h, tap=tiles[:n], wait=True)

        rt = Runtime(sequence, [ty, ty, n, of_in.prod(), of_out.cons()])
        return Program(iron.get_current_device(), rt).resolve_program()

    return copy


@pytest.mark.skipif(shutil.which("xclbinutil") is None, reason="xclbinutil")
def test_compile_only_builds_a_dispatch_time_design_from_xclbin_path_alone(
    tmp_path, keep_device
):
    xclbin = tmp_path / "copy.xclbin"
    opts = SimpleNamespace(
        dev="npu2", emit_mlir=False, xclbin_path=str(xclbin), insts_path=None
    )

    cli.run_design_cli(_forwarded_copy(), opts, compile_kwargs={})

    assert xclbin.stat().st_size
