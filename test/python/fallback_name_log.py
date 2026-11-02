# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# An object that is not assigned to a variable where it is built is named after
# its kind, and a DEBUG log line says why.

import inspect
import logging

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

line_type = np.ndarray[(16,), np.dtype[np.int32]]


class Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def passthrough(a, b):
    a.acquire(1)
    b.acquire(1)
    a.release(1)
    b.release(1)


def sequence(x, y, a, b):
    a.fill(x)
    b.drain(y, wait=True)


line = inspect.currentframe().f_lineno + 1
fifos = dict(into=ObjectFifo(line_type))
act_out = ObjectFifo(line_type)
worker = Worker(passthrough, [fifos["into"].cons(), act_out.prod()])
rt = Runtime(sequence, [line_type] * 2 + [fifos["into"].prod(), act_out.cons()])

collect = Collect()
log = logging.getLogger("aie.iron.configuration")
log.addHandler(collect)
log.setLevel(logging.DEBUG)
Program(NPU1Col1(), rt, workers=[worker]).resolve_program()

fallbacks = [m for m in collect.messages if "is not assigned" in m]
expected = (
    f"ObjectFifo at fallback_name_log.py:{line} is not assigned to a variable "
    "where it is built; named of0"
)
assert fallbacks == [expected], collect.messages
assert fifos["into"].name == "of0" and act_out.name == "act_out"
print("PASS: fallback name logged")
