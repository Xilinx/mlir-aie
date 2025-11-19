# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.


import sys
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1


# RUN: %python %s | FileCheck %s --check-prefix=NORMAL
# NORMAL: aie.objectfifo @shim_to_mem(%{{.*}}, {%{{.*}}}, {{.*}}) {iter_count = 1 : i32} : !aie.objectfifo<memref<2048xi32>>
# NORMAL: aie.objectfifo @shim_to_mem_fwd(%{{.*}}, {%{{.*}}}, {{.*}}) {iter_count = 3 : i32} : !aie.objectfifo<memref<1024xi32>>
def test_objectfifo_bd_chain_scenarios():

    dev = NPU1Col1()
    line_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    mem_ty = np.ndarray[(2048,), np.dtype[np.int32]]

    of_shim_to_mem = ObjectFifo(mem_ty, name="shim_to_mem")
    of_shim_to_mem.set_iter_count(1)

    of_mem_to_compute = of_shim_to_mem.cons().forward(obj_type=line_ty)
    of_mem_to_compute.set_iter_count(3)

    rt = Runtime()
    vector_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    with rt.sequence(vector_ty, vector_ty, vector_ty) as (a_in, _, c_out):
        rt.fill(of_shim_to_mem.prod(), a_in)
        rt.drain(of_mem_to_compute.cons(), c_out, wait=True)

    my_program = Program(dev, rt)
    module = my_program.resolve_program(SequentialPlacer())

    print(module)


# RUN: not %python %s zero 2>&1 | FileCheck %s --check-prefix=ERROR_ZERO
# ERROR_ZERO: ValueError: Iter count must be in [1, 256] range.
def test_objectfifo_bd_chain_error_zero():
    line_ty = np.ndarray[(1024,), np.dtype[np.int32]]

    of_test = ObjectFifo(line_ty, name="test_zero")
    of_test.set_iter_count(0)


# RUN: not %python %s high 2>&1 | FileCheck %s --check-prefix=ERROR_HIGH
# ERROR_HIGH: ValueError: Iter count must be in [1, 256] range.
def test_objectfifo_bd_chain_error_high():
    line_ty = np.ndarray[(1024,), np.dtype[np.int32]]

    of_test = ObjectFifo(line_ty, name="test_high")
    of_test.set_iter_count(257)


# RUN: not %python %s args 2>&1 | FileCheck %s --check-prefix=ERROR_ARGS
# ERROR_ARGS: ValueError: iter_count is required. Provide a value between 1 and 256.
def test_objectfifo_bd_chain_error_args():
    line_ty = np.ndarray[(1024,), np.dtype[np.int32]]

    of_test = ObjectFifo(line_ty, name="test_args")
    of_test.set_iter_count()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "zero":
            test_objectfifo_bd_chain_error_zero()
        elif test_type == "high":
            test_objectfifo_bd_chain_error_high()
        elif test_type == "args":
            test_objectfifo_bd_chain_error_args()
    else:
        test_objectfifo_bd_chain_scenarios()
