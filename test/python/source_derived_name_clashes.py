# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# A variable that shares its name with a symbol IRON declares for something
# else, here a kernel, does not take that symbol: the object is numbered past
# it, and the design still resolves.

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

line_type = np.ndarray[(16,), np.dtype[np.int32]]

passthrough_kernel = Kernel("passthrough", "passthrough.o", [line_type, line_type])
passthrough = ObjectFifo(line_type)
out = ObjectFifo(line_type)


def core_fn(a, b, kernel):
    x = a.acquire(1)
    y = b.acquire(1)
    kernel(x, y)
    a.release(1)
    b.release(1)


def body(A, C, a, c):
    a.fill(A)
    c.drain(C, wait=True)


worker = Worker(core_fn, [passthrough.cons(), out.prod(), passthrough_kernel])
rt = Runtime(body, [line_type, line_type, passthrough.prod(), out.cons()])
module = Program(NPU1Col1(), rt, workers=[worker]).resolve_program()
module.operation.verify()

text = str(module)
assert "func.func private @passthrough(" in text, text
assert "aie.objectfifo @passthrough_1(" in text, text
print("PASS: source names clear of kernel symbols")
