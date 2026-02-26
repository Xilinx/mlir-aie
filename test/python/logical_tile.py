# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, NPU1Col1, Tile
from util import construct_and_print_module


# CHECK-LABEL: TEST: logical_tile_worker_unconstrained
# CHECK: aie.device(npu2) {
# CHECK:   %[[WORKER:.*]] = aie.logical_tile<CoreTile>(?, ?)
# CHECK:   %[[SHIM_IN:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   %[[SHIM_OUT:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   aie.objectfifo @in(%[[SHIM_IN]], {%[[WORKER]]}, {{.*}})
# CHECK:   aie.objectfifo @out(%[[WORKER]], {%[[SHIM_OUT]]}, {{.*}})
# CHECK:   aie.core(%[[WORKER]])
@construct_and_print_module
def logical_tile_worker_unconstrained(module):
    """Test unconstrained Worker - ObjectFifos and core consume LogicalTileOp."""
    n = 1024
    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_in = ObjectFifo(n_ty, name="in")
    of_out = ObjectFifo(n_ty, name="out")

    def core_fn(of_in, of_out):
        pass

    # Worker with default placement (AnyComputeTile)
    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), C, wait=True)

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: logical_tile_worker_constrained
# CHECK: aie.device(npu2) {
# CHECK:   %[[WORKER:.*]] = aie.logical_tile<CoreTile>(0, 2)
# CHECK:   %[[SHIM_IN:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   %[[SHIM_OUT:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   aie.objectfifo @in(%[[SHIM_IN]], {%[[WORKER]]}, {{.*}})
# CHECK:   aie.objectfifo @out(%[[WORKER]], {%[[SHIM_OUT]]}, {{.*}})
# CHECK:   aie.core(%[[WORKER]])
@construct_and_print_module
def logical_tile_worker_constrained(module):
    """Test constrained Worker - ObjectFifos and core consume same LogicalTileOp."""
    n = 1024
    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_in = ObjectFifo(n_ty, name="in")
    of_out = ObjectFifo(n_ty, name="out")

    def core_fn(of_in, of_out):
        pass

    # Worker with explicit tile placement
    worker = Worker(core_fn, [of_in.cons(), of_out.prod()], placement=Tile(0, 2))

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), C, wait=True)

    module = Program(NPU2(), rt).resolve_program()
    return module
