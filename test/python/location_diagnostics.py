# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

"""A compile error on a real IRON design must name the user's file and line.

Source locations are plumbing; this is the thing users actually experience.
MLIR prints an op's location as the prefix of every diagnostic, so the payoff
for attribution is automatic -- and so is the regression if attribution breaks.
Before locations reached IRON, the design below reported:

    error: unknown: 'arith.addi' op operand #0 must be signless-...

which names neither the file nor the line and is the same message for every
mistake in every design.

This test is deliberately built on a genuine user error (`uint8` arithmetic is
not supported) rather than a synthetic one, so it keeps testing a real
diagnostic path.
"""

import os

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

THIS_FILE = os.path.abspath(__file__)
SOURCE = open(THIS_FILE).read().splitlines()

line_type = np.ndarray[(256,), np.dtype[np.uint8]]
vector_type = np.ndarray[(1024,), np.dtype[np.uint8]]

MISTAKE = "elem_out[0] = elem_in[0] + elem_in[1]  # unsupported on uint8"


def build():
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


def main():
    try:
        build()
    except Exception as exc:  # MLIRError, but keep the assert independent of it
        message = str(exc)
    else:
        raise AssertionError("expected the uint8 addition to be rejected")

    assert "arith.addi" in message, f"unexpected diagnostic:\n{message}"

    # The whole point: the diagnostic must not say "unknown".
    assert "unknown:" not in message, (
        "diagnostic still reports an unknown location:\n" + message
    )

    # It must name this file...
    assert THIS_FILE in message, f"diagnostic does not name {THIS_FILE}:\n{message}"

    # ...and point inside the core body that contains the mistake. The op is
    # built by running core_fn, so it is attributed to that function rather
    # than to the individual statement; assert the cited line is the enclosing
    # `def`, which is what the current mechanism can promise.
    def_line = next(
        i + 1 for i, text in enumerate(SOURCE) if text.strip().startswith("def core_fn")
    )
    expected = f'"{THIS_FILE}":{def_line}'
    assert expected in message, (
        f"diagnostic should cite {expected} (the body holding "
        f"{MISTAKE!r}), got:\n{message}"
    )

    print(f"PASS: diagnostic cites {THIS_FILE}:{def_line}")


main()
