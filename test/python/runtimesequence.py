# Copyright (C) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker

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

    def sequence(A, B, C, in_0, out_1, out_2):
        tg = TaskGroup()
        in_0.fill(A, group=tg)
        out_1.drain(B, group=tg, wait=True)
        out_2.drain(C, group=tg, wait=True)
        tg.finish()

    rt = Runtime(
        sequence,
        [n_ty, n_ty, n_ty],
        fn_args=[of_0.prod(), of_1.cons(), of_2.cons()],
    )
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

    def sequence(A, B, C, in_0, out_1, out_2):
        in_0.fill(A)
        out_1.drain(B, wait=True)
        out_2.drain(C, wait=True)

    rt = Runtime(
        sequence,
        [n_ty, n_ty, n_ty],
        fn_args=[of_0.prod(), of_1.cons(), of_2.cons()],
    )
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

    def sequence(A, B, C, in_0, out_1):
        in_0.fill(A)
        out_1.drain(B, wait=True)

    rt = Runtime(
        sequence,
        [n_ty, n_ty, n_ty],
        fn_args=[of_0.prod(), of_1.cons()],
    )
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

    def sequence(A, B, C, in_0, in_1, out_2):
        in_0.fill(A)
        in_1.fill(B)
        out_2.drain(C, wait=True)

    rt = Runtime(
        sequence,
        [n_ty, n_ty, n_ty],
        fn_args=[of_0.prod(), of_1.prod(), of_2.cons()],
    )
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

    def sequence(A, B, C, in_0, in_1, out_2):
        out_2.drain(C, wait=True)
        in_0.fill(A)
        in_1.fill(B)

    rt = Runtime(
        sequence,
        [n_ty, n_ty, n_ty],
        fn_args=[of_0.prod(), of_1.prod(), of_2.cons()],
    )
    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    return module


# CHECK-LABEL: TEST: rt_strict_mixed_sequence
# CHECK: success!
@construct_and_print_module
def rt_strict_mixed_sequence(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of_0, of_1, of_2):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.cons(), of_2.prod()])

    def sequence(A, B, C, in_0, in_1, out_2):
        tg = TaskGroup()
        out_2.drain(C, wait=True, group=tg)
        in_0.fill(A, group=tg)
        in_1.fill(B)
        tg.finish()

    rt = Runtime(
        sequence,
        [n_ty, n_ty, n_ty],
        fn_args=[of_0.prod(), of_1.prod(), of_2.cons()],
    )
    try:
        Program(NPU2(), rt, workers=[worker]).resolve_program()
    except Exception as e:
        print("success!")
    return module


# CHECK-LABEL: TEST: rt_not_strict_mixed_sequence
# CHECK: aiex.dma_start_task([[T0:%.*]])
# CHECK: aiex.dma_start_task([[T1:%.*]])
# CHECK: aiex.dma_start_task([[T2:%.*]])
# CHECK: aiex.dma_await_task([[T0]])
# CHECK: aiex.dma_free_task([[T1]])
# CHECK: aiex.dma_free_task([[T2]])
@construct_and_print_module
def rt_not_strict_mixed_sequence(module):
    n = 1024

    n_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_0 = ObjectFifo(n_ty, name="of0")
    of_1 = ObjectFifo(n_ty, name="of1")
    of_2 = ObjectFifo(n_ty, name="iof2")

    def core_fn(of_0, of_1, of_2):
        pass

    worker = Worker(core_fn, [of_0.cons(), of_1.cons(), of_2.prod()])

    def sequence(A, B, C, in_0, in_1, out_2):
        tg = TaskGroup()
        out_2.drain(C, wait=True, group=tg)
        in_0.fill(A, group=tg)
        in_1.fill(B)
        tg.finish()

    rt = Runtime(
        sequence,
        [n_ty, n_ty, n_ty],
        fn_args=[of_0.prod(), of_1.prod(), of_2.cons()],
        strict_task_groups=False,
    )
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

    def sequence(A, B, C, in_0, in_1, out_2):
        tg = TaskGroup()
        tg2 = TaskGroup()
        out_2.drain(C, wait=True, group=tg)
        in_0.fill(A, group=tg)
        in_1.fill(B, group=tg2)
        tg.finish()
        tg2.finish()

    rt = Runtime(
        sequence,
        [n_ty, n_ty, n_ty],
        fn_args=[of_0.prod(), of_1.prod(), of_2.cons()],
    )
    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    return module
