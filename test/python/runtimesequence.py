# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, AnyComputeTile, Tile
from aie.helpers.util import np_ndarray_type_get_shape
from util import construct_and_print_module

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: TEST: task_group_drain_sequence
# CHECK: aiex.dma_start_task(%0)
# CHECK: aiex.dma_start_task(%1)
# CHECK: aiex.dma_start_task(%2)
# CHECK: aiex.dma_await_task(%1)
# CHECK: aiex.dma_await_task(%2)
# CHECK: aiex.dma_free_task(%0)
@construct_and_print_module
def task_group_drain_sequence(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of_0, of_1, of_2):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.prod(), of_2.prod()])

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(worker)

        tg = rt.task_group()
        rt.fill(of_0.prod(), A, task_group=tg)
        rt.drain(of_1.cons(), B, task_group=tg, wait=True)
        rt.drain(of_2.cons(), C, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    return module


# CHECK-LABEL: TEST: default_rt_drain_sequence
# CHECK: aiex.dma_start_task(%0)
# CHECK: aiex.dma_start_task(%1)
# CHECK: aiex.dma_start_task(%2)
# CHECK: aiex.dma_await_task(%1)
# CHECK: aiex.dma_await_task(%2)
# CHECK: aiex.dma_free_task(%0)
@construct_and_print_module
def default_rt_drain_sequence(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of_0, of_1, of_2):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.prod(), of_2.prod()])

    rt = Runtime()
    with rt.sequence(n_ty, n_ty, n_ty) as (A, B, C):
        rt.start(worker)

        rt.fill(of_0.prod(), A)
        rt.drain(of_1.cons(), B, wait=True)
        rt.drain(of_2.cons(), C, wait=True)

    module = Program(NPU2(), rt).resolve_program(SequentialPlacer())
    return module
