# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

import re

import numpy as np
import pytest

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1

TY = np.ndarray[(16,), np.dtype[np.int32]]


def lowered(fill_burst, drain_burst):
    into = ObjectFifo(TY, name="into")
    back = ObjectFifo(TY, name="back")

    def forward(i, o):
        i.acquire(1)
        o.acquire(1)
        i.release(1)
        o.release(1)

    worker = Worker(forward, [into.cons(), back.prod()])

    def sequence(a, c, fill, drain):
        fill.fill(a, burst_length=fill_burst)
        drain.drain(c, wait=True, burst_length=drain_burst)

    rt = Runtime(sequence, [TY, TY, into.prod(), back.cons()])
    return str(Program(NPU2Col1(), rt, workers=[worker]).resolve_program())


@pytest.mark.parametrize("fill_burst,drain_burst", [(256, 64), (512, 128)])
def test_fill_and_drain_carry_their_burst_length(fill_burst, drain_burst):
    text = lowered(fill_burst, drain_burst)
    assert re.findall(r"burst_length = (\d+) : i32", text) == [
        str(fill_burst),
        str(drain_burst),
    ]


def test_an_unset_burst_length_emits_as_before():
    text = lowered(0, 0)
    assert set(re.findall(r"burst_length = (\d+) : i32", text)) <= {"0"}
