# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Worker(wait_for_parameters=True) waits for the current dispatch's parameters
# at the top of every iteration, before core_fn reads them. Without a
# ScratchpadParameter in fn_args there is nothing to wait for, so it raises.

import numpy as np

from aie.iron import Program, Runtime, ScratchpadParameter, Worker
from aie.iron.device import NPU2Col1


def task(n):
    n.read()


n = ScratchpadParameter("n", np.int32)
worker = Worker(task, fn_args=[n], wait_for_parameters=True)
rt = Runtime(lambda x: None, [np.ndarray[(16,), np.dtype[np.int32]]])

# CHECK-LABEL: waits
# CHECK:         aie.core
# CHECK:           scf.while : () -> () {
# CHECK:           } do {
# CHECK-NEXT:        aiex.wait_scratchpad_parameters
# CHECK-NEXT:        aiex.read_scratchpad_parameter @n : i32
# CHECK:           aie.end
print("waits")
print(Program(NPU2Col1(), rt, workers=[worker]).resolve_program())

# CHECK-LABEL: no_parameter
# CHECK: Worker(wait_for_parameters=True) needs a ScratchpadParameter in fn_args
print("no_parameter")
try:
    Worker(lambda: None, wait_for_parameters=True)
    raise AssertionError("expected ValueError for a Worker with no parameters")
except ValueError as e:
    print(e)
