# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s
# REQUIRES: peano

import os
import tempfile
import traceback

import numpy as np
from aie.helpers.errors import attach_tool_location
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.utils.compile.utils import compile_mlir_module

THIS_FILE = os.path.abspath(__file__)
SOURCE = open(THIS_FILE).read().splitlines()

too_big = np.ndarray[(12288,), np.dtype[np.int32]]


def design():
    of_in = ObjectFifo(too_big, depth=2, name="in")  # two 48 KiB buffers
    of_out = ObjectFifo(too_big, depth=2, name="out")

    def core_fn(a, b):
        elem_out = b.acquire(1)
        elem_in = a.acquire(1)
        elem_out[0] = elem_in[0]
        a.release(1)
        b.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def sequence(a_in, b_out, in_handle, out_handle):
        in_handle.fill(a_in)
        out_handle.drain(b_out, wait=True)

    rt = Runtime(sequence, [too_big, too_big, of_in.prod(), of_out.cons()])
    return Program(NPU2(), rt, workers=[worker]).resolve_program()


def check_aiecc_failure_points_at_design():
    module = design()
    with tempfile.TemporaryDirectory() as work_dir:
        try:
            compile_mlir_module(
                module,
                insts_path=os.path.join(work_dir, "insts.bin"),
                work_dir=work_dir,
            )
        except RuntimeError as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            message = str(exc)
        else:
            raise AssertionError("expected buffer allocation to fail")

    assert "could not be placed" in message, message
    innermost = frames[-1]
    assert os.path.abspath(innermost.filename) == THIS_FILE, frames
    assert "# two 48 KiB buffers" in SOURCE[innermost.lineno - 1], innermost
    assert "# two 48 KiB buffers" in (innermost.line or ""), innermost


def check_unreadable_locations_are_skipped():
    for output in (
        "/nonexistent/design.py:1:1: error: op rejected",
        f"{THIS_FILE}:{len(SOURCE) + 10}:1: error: op rejected",
        f"{THIS_FILE}:1:1: warning: only a warning",
    ):
        exc = RuntimeError(output)
        assert not attach_tool_location(exc, output), output
        assert exc.__traceback__ is None


check_unreadable_locations_are_skipped()
check_aiecc_failure_points_at_design()
print("PASS: aiecc failures point at the design")
