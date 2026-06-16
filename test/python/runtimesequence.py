# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
from aie.iron import TaskGroup, ObjectFifo, Program, Runtime, Worker

from aie.iron.device import NPU2
from util import construct_and_print_module

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: TEST: task_group_drain_sequence
# CHECK: aiex.dma_start_task([[T0:%.*]])
# CHECK: aiex.dma_start_task([[T1:%.*]])
# CHECK: aiex.dma_start_task([[T2:%.*]])
# CHECK: aiex.dma_await_task([[T1]])
# CHECK: aiex.dma_await_task([[T2]])
# CHECK: aiex.dma_free_task([[T0]])
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

    def sequence(A, B, C):

        tg = TaskGroup()
        of_0.prod().fill(A, group=tg)
        of_1.cons().drain(B, group=tg, wait=True)
        of_2.cons().drain(C, group=tg, wait=True)
        tg.resolve()

    rt.sequence(sequence, [n_ty, n_ty, n_ty])

    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    return module


# CHECK-LABEL: TEST: default_rt_drain_sequence
# CHECK: aiex.dma_start_task([[T0:%.*]])
# CHECK: aiex.dma_start_task([[T1:%.*]])
# CHECK: aiex.dma_start_task([[T2:%.*]])
# CHECK: aiex.dma_await_task([[T1]])
# CHECK: aiex.dma_await_task([[T2]])
# CHECK: aiex.dma_free_task([[T0]])
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

    def sequence(A, B, C):

        of_0.prod().fill(A)
        of_1.cons().drain(B, wait=True)
        of_2.cons().drain(C, wait=True)

    rt.sequence(sequence, [n_ty, n_ty, n_ty])

    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    return module


# CHECK-LABEL: TEST: default_rt_basic_sequence
# CHECK: aiex.dma_start_task([[T0:%.*]])
# CHECK: aiex.dma_start_task([[T1:%.*]])
# CHECK: aiex.dma_await_task([[T1]])
# CHECK: aiex.dma_free_task([[T0]])
@construct_and_print_module
def default_rt_basic_sequence(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")

    def core_fn(of_0, of_1):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.prod()])

    rt = Runtime()

    def sequence(A, B, C):

        of_0.prod().fill(A)
        of_1.cons().drain(B, wait=True)

    rt.sequence(sequence, [n_ty, n_ty, n_ty])

    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    return module


# CHECK-LABEL: TEST: default_rt_fill_sequence
# CHECK: aiex.dma_start_task([[T0:%.*]])
# CHECK: aiex.dma_start_task([[T1:%.*]])
# CHECK: aiex.dma_start_task([[T2:%.*]])
# CHECK: aiex.dma_await_task([[T2]])
# CHECK: aiex.dma_free_task([[T0]])
# CHECK: aiex.dma_free_task([[T1]])
@construct_and_print_module
def default_rt_fill_sequence(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of_0, of_1, of_2):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.cons(), of_2.prod()])

    rt = Runtime()

    def sequence(A, B, C):

        of_0.prod().fill(A)
        of_1.prod().fill(B)
        of_2.cons().drain(C, wait=True)

    rt.sequence(sequence, [n_ty, n_ty, n_ty])

    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    return module


# CHECK-LABEL: TEST: rt_drain_then_fill_sequence
# CHECK: aiex.dma_start_task([[T0:%.*]])
# CHECK: aiex.dma_start_task([[T1:%.*]])
# CHECK: aiex.dma_start_task([[T2:%.*]])
# CHECK: aiex.dma_await_task([[T0]])
# CHECK: aiex.dma_free_task([[T1]])
# CHECK: aiex.dma_free_task([[T2]])
@construct_and_print_module
def rt_drain_then_fill_sequence(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of_0, of_1, of_2):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.cons(), of_2.prod()])

    rt = Runtime()

    def sequence(A, B, C):

        of_2.cons().drain(C, wait=True)
        of_0.prod().fill(A)
        of_1.prod().fill(B)

    rt.sequence(sequence, [n_ty, n_ty, n_ty])

    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    return module


# A TaskGroup may be mixed with ungrouped transfers (the ungrouped one joins the
# implicit default group, resolved at end-of-sequence). The eager runtime has no
# default-vs-explicit hazard, so this is always allowed.
# CHECK-LABEL: TEST: rt_mixed_group_and_default_sequence
# CHECK: aiex.dma_start_task([[T0:%.*]])
# CHECK: aiex.dma_start_task([[T1:%.*]])
# CHECK: aiex.dma_start_task([[T2:%.*]])
# CHECK: aiex.dma_await_task([[T0]])
# CHECK: aiex.dma_free_task([[T1]])
# CHECK: aiex.dma_free_task([[T2]])
@construct_and_print_module
def rt_mixed_group_and_default_sequence(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of_0, of_1, of_2):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.cons(), of_2.prod()])

    rt = Runtime()

    def sequence(A, B, C):
        tg = TaskGroup()
        of_2.cons().drain(C, wait=True, group=tg)
        of_0.prod().fill(A, group=tg)
        of_1.prod().fill(B)
        tg.resolve()

    rt.sequence(sequence, [n_ty, n_ty, n_ty])

    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    return module


# CHECK-LABEL: TEST: rt_two_task_group_sequence
# CHECK: aiex.dma_start_task([[T0:%.*]])
# CHECK: aiex.dma_start_task([[T1:%.*]])
# CHECK: aiex.dma_start_task([[T2:%.*]])
# CHECK: aiex.dma_await_task([[T0]])
# CHECK: aiex.dma_free_task([[T1]])
# CHECK: aiex.dma_free_task([[T2]])
@construct_and_print_module
def rt_two_task_group_sequence(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of_0, of_1, of_2):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.cons(), of_2.prod()])

    rt = Runtime()

    def sequence(A, B, C):

        tg = TaskGroup()
        tg2 = TaskGroup()
        of_2.cons().drain(C, wait=True, group=tg)
        of_0.prod().fill(A, group=tg)
        of_1.prod().fill(B, group=tg2)
        tg.resolve()
        tg2.resolve()

    rt.sequence(sequence, [n_ty, n_ty, n_ty])

    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    return module
