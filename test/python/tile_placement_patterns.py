# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""
Test tile placement patterns - partial coordinates, full coordinates, and mixed placement.
"""

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from util import construct_and_print_module


# CHECK-LABEL: TEST: partial_coordinates_col_only
# CHECK: aie.device(npu2) {
# CHECK:   %[[WORKER:.*]] = aie.logical_tile<CoreTile>(2, ?)
# CHECK:   %[[SHIM_IN:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   %[[SHIM_OUT:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   aie.objectfifo @in(%[[SHIM_IN]], {%[[WORKER]]}, {{.*}})
# CHECK:   aie.objectfifo @out(%[[WORKER]], {%[[SHIM_OUT]]}, {{.*}})
# CHECK:   aie.core(%[[WORKER]])
@construct_and_print_module
def partial_coordinates_col_only(module):
    """Test partial coordinates with ObjectFifos referencing the LogicalTileOp."""
    n = 1024
    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_in = ObjectFifo(n_ty, name="in")
    of_out = ObjectFifo(n_ty, name="out")

    def core_fn(of_in, of_out):
        pass

    # Partial placement - column constrained, row unconstrained
    worker = Worker(core_fn, [of_in.cons(), of_out.prod()], placement=Tile(col=2))

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), C, wait=True)

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: multiple_workers_mixed_placement
# CHECK: aie.device(npu2) {
# CHECK-DAG:   %[[W0:.*]] = aie.logical_tile<CoreTile>(?, ?)
# CHECK-DAG:   %[[W1:.*]] = aie.logical_tile<CoreTile>(1, ?)
# CHECK-DAG:   %[[W2:.*]] = aie.logical_tile<CoreTile>(0, 2)
# CHECK-DAG:   aie.objectfifo @of0({{.*}}, {%[[W0]]}, {{.*}})
# CHECK-DAG:   aie.objectfifo @of1({{.*}}, {%[[W1]]}, {{.*}})
# CHECK-DAG:   aie.objectfifo @of2({{.*}}, {%[[W2]]}, {{.*}})
# CHECK-DAG:   aie.core(%[[W0]])
# CHECK-DAG:   aie.core(%[[W1]])
# CHECK-DAG:   aie.core(%[[W2]])
@construct_and_print_module
def multiple_workers_mixed_placement(module):
    """Test mixed placement types - ObjectFifos reference correct LogicalTileOps."""
    n = 1024
    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="of2")

    def core_fn(of_in):
        pass

    # Mix of placement strategies
    worker0 = Worker(core_fn, [of_0.cons()])  # Unconstrained
    worker1 = Worker(core_fn, [of_1.cons()], placement=Tile(col=1))  # Partial
    worker2 = Worker(core_fn, [of_2.cons()], placement=Tile(0, 2))  # Full

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(worker0, worker1, worker2)
        rt.fill(of_0.prod(), A)
        rt.fill(of_1.prod(), B)
        rt.fill(of_2.prod(), C)

    module = Program(NPU2(), rt).resolve_program()
    return module
