# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# An error IRON raises about an object the design declared ends its traceback
# at the statement that declared it, named after the object, with IRON's own
# frames hidden and a note saying so, whether the design is resolved directly
# or compiled.

import os
import tempfile
import traceback

import numpy as np
from aie.helpers.sourceloc import is_internal_file
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1
from aie.utils.compile.jit.compilabledesign import CompilableDesign

THIS_FILE = os.path.abspath(__file__)
SOURCE = open(THIS_FILE).read().splitlines()
DECLARATION = "dangling = ObjectFifo(line_type)"
line_type = np.ndarray[(16,), np.dtype[np.int32]]


def body(of):
    pass


def sequence(a):
    pass


def generator():
    dangling = ObjectFifo(line_type)
    worker = Worker(body, [dangling.prod()])
    rt = Runtime(sequence, [line_type])
    return Program(NPU1Col1(), rt, workers=[worker]).resolve_program()


def compile_generator():
    with tempfile.TemporaryDirectory() as work:
        CompilableDesign(generator, use_cache=False).compile(
            xclbin_path=f"{work}/design.xclbin", inst_path=f"{work}/design.bin"
        )


def check(build):
    try:
        build()
    except ValueError as exc:
        error = exc
    else:
        raise AssertionError("expected the fifo with no consumer to be rejected")
    frames = traceback.extract_tb(error.__traceback__)
    innermost = frames[-1]
    assert os.path.abspath(innermost.filename) == THIS_FILE, frames
    assert innermost.line == DECLARATION, innermost
    assert innermost.name == "dangling", innermost.name
    assert str(error).startswith("ObjectFifo 'dangling' has no cons() handle"), error
    # Only the lines that re-raise it remain of IRON's frames.
    internal = [f for f in frames if is_internal_file(f.filename)]
    assert all("raise" in (f.line or "") for f in internal), internal
    assert error.__notes__ == [
        "IRON's own frames are hidden; IRON_FULL_TRACEBACK=1 shows them."
    ], error.__notes__
    return len(frames)


resolved = check(generator)
compiled = check(compile_generator)
print(f"PASS: resolved directly, {resolved} frames end at the declaration")
print(f"PASS: through compile(), {compiled} frames end at the declaration")
