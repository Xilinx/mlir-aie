# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

import os
import traceback

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1
from aie.helpers.sourceloc import is_internal_file
from aie.ir import MLIRError

THIS_FILE = os.path.abspath(__file__)

line_type = np.ndarray[(256,), np.dtype[np.uint8]]
vector_type = np.ndarray[(1024,), np.dtype[np.uint8]]

MISTAKE = "elem_out[0] = elem_in[0] + elem_in[1]  # unsupported on uint8"
BAD_TRANSFER = "out_handle.fill(b_out)  # a consumer handle cannot fill"


def verifier_failure():
    """uint8 addition inside a core body -- rejected by the MLIR verifier."""
    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")

    def core_fn(a, b):
        elem_out = b.acquire(1)
        elem_in = a.acquire(1)
        elem_out[0] = elem_in[0] + elem_in[1]  # unsupported on uint8
        a.release(1)
        b.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def sequence(a_in, b_out, in_handle, out_handle):
        in_handle.fill(a_in)
        out_handle.drain(b_out, wait=True)

    rt = Runtime(sequence, [vector_type, vector_type, of_in.prod(), of_out.cons()])
    return Program(NPU1Col1(), rt, workers=[worker]).resolve_program()


def guard_failure():
    """A fill on a consumer handle -- rejected by an IRON guard."""
    of_in = ObjectFifo(line_type, name="in2")
    of_out = ObjectFifo(line_type, name="out2")

    def core_fn(a, b):
        elem_out = b.acquire(1)
        elem_in = a.acquire(1)
        elem_out[0] = elem_in[0]
        a.release(1)
        b.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def sequence(a_in, b_out, in_handle, out_handle):
        in_handle.fill(a_in)
        out_handle.fill(b_out)  # a consumer handle cannot fill

    rt = Runtime(sequence, [vector_type, vector_type, of_in.prod(), of_out.cons()])
    return Program(NPU1Col1(), rt, workers=[worker]).resolve_program()


def frames_of(exc):
    return [
        (os.path.abspath(f.filename), f.lineno, f.name, f.line or "")
        for f in traceback.extract_tb(exc.__traceback__)
    ]


def check_verifier_failure():
    try:
        verifier_failure()
    except MLIRError as exc:
        frames = frames_of(exc)
        message = str(exc)
    else:
        raise AssertionError("expected the uint8 addition to be rejected")

    assert "arith.addi" in message, message
    assert "unknown" not in message, message

    user_frames = [f for f in frames if f[0] == THIS_FILE]
    assert user_frames, f"no frame in {THIS_FILE}:\n{frames}"

    filename, lineno, name, text = user_frames[-1]
    assert text == MISTAKE, f"innermost frame is line {lineno}: {text!r}"
    assert name == "core_fn", f"frame should name the core body, got {name!r}"

    return f"{filename}:{lineno} in {name}"


def check_guard_failure():
    try:
        guard_failure()
    except ValueError as exc:
        frames = frames_of(exc)
    else:
        raise AssertionError("expected the consumer fill to be rejected")

    internal = [f for f in frames if is_internal_file(f[0])]
    # resolve_program re-raises, so its own frame is the one that remains.
    assert len(internal) <= 1, "IRON frames not filtered:\n" + "\n".join(
        f"  {f[0]}:{f[1]} in {f[2]}" for f in internal
    )

    user_frames = [f for f in frames if f[0] == THIS_FILE]
    assert user_frames, f"no frame in {THIS_FILE}:\n{frames}"
    _, lineno, _, text = user_frames[-1]
    assert text == BAD_TRANSFER, f"innermost frame is line {lineno}: {text!r}"
    return f"{len(frames)} frames, {len(internal)} internal"


def main():
    where = check_verifier_failure()
    guard = check_guard_failure()
    print(f"PASS: verifier failure reported at {where}")
    print(f"PASS: guard failure filtered to {guard}")


main()
