# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

"""A compile error on a real IRON design must read like a Python error.

Source locations are plumbing; this is what users actually experience. Before
this work the design below reported

    error: unknown: 'arith.addi' op operand #0 must be signless-...

which names neither the file nor the line, and is the same message for every
mistake in every design. It should instead arrive as an exception whose
traceback walks the user's own code -- the Worker declaration, then the
statement that failed -- with the source quoted, exactly as a TypeError would.

Both error kinds are covered: an MLIR verifier failure, which has to be rebuilt
from a diagnostic string, and an IRON guard, which is already an exception but
arrives buried under IRON's own frames.

The mistakes are genuine (uint8 arithmetic is unsupported; a shim BD has four
dimensions), so this keeps exercising real diagnostic paths rather than
synthetic ones.
"""

import os
import traceback

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1
from aie.helpers.errors import IronCompileError

THIS_FILE = os.path.abspath(__file__)
SOURCE = open(THIS_FILE).read().splitlines()

line_type = np.ndarray[(256,), np.dtype[np.uint8]]
vector_type = np.ndarray[(1024,), np.dtype[np.uint8]]

MISTAKE = "elem_out[0] = elem_in[0] + elem_in[1]  # unsupported on uint8"
BAD_TRANSFER = "in_handle.fill(a_in, sizes=[2, 2, 2, 2, 2], strides=[1, 1, 1, 1, 1])"


def _line_of(fragment):
    # Match the statement, not the constant above that spells it out: the
    # declaration reads `NAME = "..."` and so never starts with the fragment.
    return next(
        i + 1 for i, text in enumerate(SOURCE) if text.lstrip().startswith(fragment)
    )


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
    """A five-dimensional shim transfer -- rejected by an IRON guard."""
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
        in_handle.fill(a_in, sizes=[2, 2, 2, 2, 2], strides=[1, 1, 1, 1, 1])
        out_handle.drain(b_out, wait=True)

    rt = Runtime(sequence, [vector_type, vector_type, of_in.prod(), of_out.cons()])
    return Program(NPU1Col1(), rt, workers=[worker]).resolve_program()


def constructor_failure():
    """Two Workers sharing one consumer handle -- rejected during construction."""
    of_in = ObjectFifo(line_type, name="in3")
    of_out = ObjectFifo(line_type, name="out3")
    shared_consumer = of_in.cons()

    def core_fn(a, b):
        elem_out = b.acquire(1)
        elem_in = a.acquire(1)
        elem_out[0] = elem_in[0]
        a.release(1)
        b.release(1)

    Worker(core_fn, [shared_consumer, of_out.prod()])
    Worker(core_fn, [shared_consumer, of_out.prod()])


def frames_of(exc):
    return [
        (os.path.abspath(f.filename), f.lineno, f.name, f.line or "")
        for f in traceback.extract_tb(exc.__traceback__)
    ]


def check_verifier_failure():
    try:
        verifier_failure()
    except IronCompileError as exc:
        frames = frames_of(exc)
        message = str(exc)
    else:
        raise AssertionError("expected the uint8 addition to be rejected")

    assert "arith.addi" in message, message
    # The location moved out of the message and into the traceback, which is
    # what makes it print the offending source rather than just address it.
    assert "unknown" not in message, message

    user_frames = [f for f in frames if f[0] == THIS_FILE]
    assert user_frames, f"no frame in {THIS_FILE}:\n{frames}"

    # Innermost frame: the statement that failed, quoted from this file.
    filename, lineno, name, text = user_frames[-1]
    assert lineno == _line_of(MISTAKE), (
        f"innermost frame is line {lineno} ({text!r}), expected the line "
        f"holding {MISTAKE!r}"
    )
    assert name == "core_fn", f"frame should name the core body, got {name!r}"
    assert "elem_in[0] + elem_in[1]" in text, f"source not quoted: {text!r}"

    # Outer frame: the declaration that put that body on a core.
    declaration = [f for f in user_frames if "Worker(core_fn" in f[3]]
    assert declaration, f"Worker declaration missing from traceback:\n{user_frames}"

    return f"{filename}:{lineno} in {name}"


def check_guard_failure():
    try:
        guard_failure()
    except ValueError as exc:
        frames = frames_of(exc)
    else:
        raise AssertionError("expected the 5-dimensional transfer to be rejected")

    internal = [
        f
        for f in frames
        if any(part in f[0] for part in ("aie/iron", "aie/dialects", "aie/helpers"))
    ]
    # One frame survives filtering: Python appends the raising frame, and the
    # re-raise happens inside IRON. More than that means filtering regressed.
    assert len(internal) <= 1, "IRON frames not filtered:\n" + "\n".join(
        f"  {f[0]}:{f[1]} in {f[2]}" for f in internal
    )

    user_frames = [f for f in frames if f[0] == THIS_FILE]
    assert user_frames, f"no frame in {THIS_FILE}:\n{frames}"
    _, lineno, _, text = user_frames[-1]
    assert lineno == _line_of(BAD_TRANSFER), (
        f"innermost frame is line {lineno} ({text!r}), expected the line "
        f"holding {BAD_TRANSFER!r}"
    )
    return f"{len(frames)} frames, {len(internal)} internal"


def check_constructor_failure():
    """A constructor rejects a design before resolve_program can wrap it.

    Its traceback is internal end to end when caught, so the frames have to be
    dropped rather than filtered -- the declaration at fault is only appended
    on re-raise. Getting that wrong leaves the user staring at IRON's guts.
    """
    try:
        constructor_failure()
    except ValueError as exc:
        frames = frames_of(exc)
        message = str(exc)
    else:
        raise AssertionError("expected the shared consumer handle to be rejected")

    internal = [
        f
        for f in frames
        if any(part in f[0] for part in ("aie/iron", "aie/dialects", "aie/helpers"))
    ]
    assert len(internal) <= 1, "IRON frames not filtered:\n" + "\n".join(
        f"  {f[0]}:{f[1]} in {f[2]}" for f in internal
    )

    user_frames = [f for f in frames if f[0] == THIS_FILE]
    assert user_frames, f"no frame in {THIS_FILE}:\n{frames}"
    assert "Worker(core_fn" in user_frames[-1][3], (
        f"innermost frame should be the Worker declaration, got "
        f"{user_frames[-1][3]!r}"
    )

    # The Workers must be identifiable: the default object repr names neither.
    assert message.count(THIS_FILE) == 2, (
        "both Workers should be named by where they were declared:\n" + message
    )
    return f"{len(frames)} frames, {len(internal)} internal"


def main():
    where = check_verifier_failure()
    guard = check_guard_failure()
    ctor = check_constructor_failure()
    print(f"PASS: verifier failure reported at {where}")
    print(f"PASS: guard failure filtered to {guard}")
    print(f"PASS: constructor failure filtered to {ctor}")


main()
