# Copyright (C) 20252026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.helpers.util import np_ndarray_type_get_shape
from util import construct_and_print_module

# RUN: %python %s | FileCheck %s

# This test checks that IRON emits aie.logical_tile operations correctly.
# Tile placement is handled by the MLIR -aie-place-tiles pass.


# CHECK-LABEL: TEST: objectfifo_order
# CHECK: %[[WORKER:.*]] = aie.logical_tile<CoreTile>
# CHECK: %[[SHIM_A:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM_B:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM_C:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: aie.objectfifo @in_A(%[[SHIM_A]], {%[[WORKER]]}, {{.*}})
# CHECK: aie.objectfifo @in_B(%[[SHIM_B]], {%[[WORKER]]}, {{.*}})
# CHECK: aie.objectfifo @out_C(%[[WORKER]], {%[[SHIM_C]]}, {{.*}})
@construct_and_print_module
def objectfifo_order(module):
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

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: shim_three_in
# CHECK: %[[WORKER1:.*]] = aie.logical_tile<CoreTile>
# CHECK: %[[WORKER2:.*]] = aie.logical_tile<CoreTile>
# CHECK: %[[WORKER3:.*]] = aie.logical_tile<CoreTile>
# CHECK: %[[SHIM1:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM2:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM3:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: aie.objectfifo @in_0(%[[SHIM1]], {%[[WORKER1]]}, {{.*}})
# CHECK: aie.objectfifo @in_1(%[[SHIM2]], {%[[WORKER2]]}, {{.*}})
# CHECK: aie.objectfifo @in_2(%[[SHIM3]], {%[[WORKER3]]}, {{.*}})
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

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: shim_two_in_one_out
# CHECK: %[[WORKER:.*]] = aie.logical_tile<CoreTile>
# CHECK: %[[SHIM_A:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM_B:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM_C:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: aie.objectfifo @in_A(%[[SHIM_A]], {%[[WORKER]]}, {{.*}})
# CHECK: aie.objectfifo @in_B(%[[SHIM_B]], {%[[WORKER]]}, {{.*}})
# CHECK: aie.objectfifo @out_C(%[[WORKER]], {%[[SHIM_C]]}, {{.*}})
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

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: compute_three_in
# CHECK: %[[WORKER:.*]] = aie.logical_tile<CoreTile>
# CHECK: %{{.*}} = aie.logical_tile<ShimNOCTile>
# CHECK: aie.objectfifo @iof2
# CHECK: aie.objectfifo @of0
# CHECK: aie.objectfifo @of1
@construct_and_print_module
def compute_three_in(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of_0, of_1, of_2):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.cons(), of_2.cons()])

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_0.prod(), A)
        rt.fill(of_1.prod(), B)
        rt.fill(of_2.prod(), C)

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: compute_one_in_two_links
# CHECK: %[[WORKER:.*]] = aie.logical_tile<CoreTile>
# CHECK: %[[SHIM_IN1:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[MEM1:.*]] = aie.logical_tile<MemTile>
# CHECK: %[[SHIM_IN2:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[MEM2:.*]] = aie.logical_tile<MemTile>
# CHECK: %[[SHIM_OF0:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM_OUT1:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM_OUT2:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: aie.objectfifo @in1(%[[SHIM_IN1]], {%[[MEM1]]}, {{.*}})
# CHECK: aie.objectfifo @out1(%[[MEM1]], {%[[SHIM_OUT1]]}, {{.*}})
# CHECK: aie.objectfifo @in2(%[[SHIM_IN2]], {%[[MEM2]]}, {{.*}})
# CHECK: aie.objectfifo @out_2(%[[MEM2]], {%[[SHIM_OUT2]]}, {{.*}})
# CHECK: aie.objectfifo @of0(%[[SHIM_OF0]], {%[[WORKER]]}, {{.*}})
@construct_and_print_module
def compute_one_in_two_links(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_in1 = ObjectFifo(n_ty, name="in1")
    of_in2 = ObjectFifo(n_ty, name="in2")
    # Use default MemTile placement for forward()
    of_out1 = of_in1.cons().forward(obj_type=n_ty, name="out1")
    of_out2 = of_in2.cons().forward(obj_type=n_ty, name="out_2")

    def core_fn(of_in0):
        pass

    worker = Worker(core_fn, [of_0.cons()])

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty, n_ty, n_ty) as (A, B, C, D, E):
        rt.start(worker)
        rt.fill(of_0.prod(), A)
        rt.fill(of_in1.prod(), B)
        rt.fill(of_in2.prod(), C)
        rt.drain(of_out1.cons(), D, wait=True)
        rt.drain(of_out2.cons(), E, wait=True)

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: compute_partial_placement
# CHECK: %[[WORKER:.*]] = aie.logical_tile<CoreTile>(0, 4)
# CHECK: %[[SHIM_IN1:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[MEM1:.*]] = aie.logical_tile<MemTile>
# CHECK: %[[SHIM_IN2:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[MEM2:.*]] = aie.logical_tile<MemTile>
@construct_and_print_module
def compute_partial_placement(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_in1 = ObjectFifo(n_ty, name="in1")
    of_in2 = ObjectFifo(n_ty, name="in2")
    # Use default MemTile placement for forward()
    of_out1 = of_in1.cons().forward(obj_type=n_ty, name="out1")
    of_out2 = of_in2.cons().forward(obj_type=n_ty, name="out_2")

    def core_fn(of_in0):
        pass

    worker = Worker(core_fn, [of_0.cons()], placement=Tile(0, 4))

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty, n_ty, n_ty) as (A, B, C, D, E):
        rt.start(worker)
        rt.fill(of_0.prod(), A)
        rt.fill(of_in1.prod(), B)
        rt.fill(of_in2.prod(), C)
        rt.drain(of_out1.cons(), D, wait=True)
        rt.drain(of_out2.cons(), E, wait=True)

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: mem_eight_in_three_out
# CHECK-DAG: aie.logical_tile<CoreTile>
# CHECK-DAG: aie.logical_tile<MemTile>
# CHECK-DAG: aie.logical_tile<ShimNOCTile>
# CHECK-DAG: aie.objectfifo @of_mem_in_0
# CHECK-DAG: aie.objectfifo @of_mem_in_6
# CHECK-DAG: aie.objectfifo @out_A
# CHECK-DAG: aie.objectfifo @out_B
# CHECK-DAG: aie.objectfifo @out_C
# CHECK: aie.core
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

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: compute_three_in_col_lim
# CHECK: %[[WORKER1:.*]] = aie.logical_tile<CoreTile>
# CHECK: %[[WORKER2:.*]] = aie.logical_tile<CoreTile>
# CHECK: %[[WORKER3:.*]] = aie.logical_tile<CoreTile>
# CHECK: %[[SHIM1:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM2:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK: %[[SHIM3:.*]] = aie.logical_tile<ShimNOCTile>
@construct_and_print_module
def compute_three_in_col_lim(module):
    n = 1024
    cores_per_col = 2

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of):
        pass

    workers = [
        Worker(core_fn, [of_0.cons()]),
        Worker(core_fn, [of_1.cons()]),
        Worker(core_fn, [of_2.cons()]),
    ]

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(*workers)
        rt.fill(of_0.prod(), A)
        rt.fill(of_1.prod(), B)
        rt.fill(of_2.prod(), C)

    # NOTE: cores_per_col parameter will be supported by MLIR placement pass
    module = Program(NPU2(), rt).resolve_program()
    return module
