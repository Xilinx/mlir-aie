# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, AnyComputeTile
from aie.helpers.util import np_ndarray_type_get_shape
from util import construct_and_print_module

# RUN: %python %s | FileCheck %s

# CHECK-LABEL: TEST: shim_three_in
# CHECK: %[[shim_noc_tile_0_0:.+]] = aie.tile
# CHECK: %[[shim_noc_tile_1_0:.+]] = aie.tile
@construct_and_print_module
def shim_three_in(module):
    N = 4096
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    n_inputs = 3
    of_ins = []
    for i in range(n_inputs):
        of_ins.append(ObjectFifo(n_ty, name=f"in_{i}"))

    def core_fn(of_in):
        pass

    workers = []
    for i in range(n_inputs):
        workers.append(Worker(core_fn, [of_ins[i].cons()]))

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(*workers)
        rt.fill(of_ins[0].prod(), A)
        rt.fill(of_ins[1].prod(), B)
        rt.fill(of_ins[2].prod(), C)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    return module


# CHECK-LABEL: TEST: shim_two_in_one_out
# CHECK: %[[shim_noc_tile_0_0:.+]] = aie.tile
# CHECK-NOT: %[[shim_noc_tile_1_0:.+]] = aie.tile
@construct_and_print_module
def shim_two_in_one_out(module):
    N = 4096
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_in_A = ObjectFifo(n_ty, name="in_A")
    of_in_B = ObjectFifo(n_ty, name="in_B")
    of_out_C = ObjectFifo(n_ty, name="out_C")

    def core_fn(in_A, in_B, out_C):
        pass

    my_worker = Worker(core_fn, [of_in_A.cons(), of_in_B.cons(), of_out_C.prod()])

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(my_worker)
        rt.fill(of_in_A.prod(), A)
        rt.fill(of_in_B.prod(), B)
        rt.drain(of_out_C.cons(), C, wait=True)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    return module

# CHECK-LABEL: TEST: mem_eight_in_three_out
# CHECK: %[[mem_tile_0_1:.+]] = aie.tile
# CHECK: %[[shim_noc_tile_0_0:.+]] = aie.tile
# CHECK: %[[mem_tile_1_1:.+]] = aie.tile
# CHECK: %[[shim_noc_tile_1_0:.+]] = aie.tile
@construct_and_print_module
def mem_eight_in_three_out(module):
    N = 6000
    n = N // 6

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]
    N_ty = np.ndarray[(N,), np.dtype[np.int32]]

    n_join_inputs = 6
    of_offsets = [
        np.prod((np_ndarray_type_get_shape(n_ty))) * i for i in range(n_join_inputs)
    ]

    

    of_out_A = ObjectFifo(N_ty, name="out_A")
    of_joins = of_out_A.prod().join(
        of_offsets,
        obj_types=[n_ty] * n_join_inputs,
        names=[f"of_mem_in_{i}" for i in range(n_join_inputs)],
    )
    of_mem_in_6 = ObjectFifo(n_ty, name="of_mem_in_6")
    of_mem_in_7 = ObjectFifo(n_ty, name="of_mem_in_7")
    of_out_B = of_mem_in_6.cons().forward(obj_type=n_ty, name="out_B")
    of_out_C = of_mem_in_7.cons().forward(obj_type=n_ty, name="out_C")

    def core_fn(of_out):
        pass

    workers = []
    for i in range(n_join_inputs):
        workers.append(Worker(core_fn, [of_joins[i].prod()]))
    workers.append(Worker(core_fn, [of_mem_in_6.prod()]))
    workers.append(Worker(core_fn, [of_mem_in_7.prod()]))

    rt = Runtime()
    with rt.sequence(N_ty, n_ty, n_ty) as (A, B, C):
        rt.start(*workers)
        rt.drain(of_out_A.cons(), A, wait=True)
        rt.drain(of_out_B.cons(), B, wait=True)
        rt.drain(of_out_C.cons(), C, wait=True)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    return module
