# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from aie.dialects.aie import AIEDevice, BdIteration, device, object_fifo, tile
from aie.dialects.aiex import *
from util import construct_and_print_module

# RUN: %python %s | FileCheck %s


# The shim helpers forward iteration= to aie.dma_bd as an #aie.bd_iteration attr.


# CHECK-LABEL: shim_bd_forwards_iteration
# CHECK: aie.dma_bd({{.*}} sizes = [4, 8, 1] strides = [512, 32, 1])
# CHECK-SAME: iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2>
@construct_and_print_module
def shim_bd_forwards_iteration(module):
    N = 64

    @device(AIEDevice.npu1)
    def device_body():
        N_ty = np.ndarray[(N,), np.dtype[np.int32]]
        S = tile(0, 0)
        M = tile(0, 2)
        of_out = object_fifo("out", M, S, 2, N_ty)

        @runtime_sequence(N_ty)
        def sequence(C):
            task = dma_configure_task_for(of_out, issue_token=True)
            with bds(task) as bd:
                with bd[0]:
                    shim_dma_bd(
                        C,
                        sizes=[4, 8, 1],
                        strides=[512, 32, 1],
                        iteration=BdIteration(size=4, stride=16, current=2),
                    )
                    EndOp()
            dma_start_task(task)

    return module


# With iteration= set the helper skips its usual pad-to-4, so the BD stays 3-dim;
# emitted sizes = [4, 8, 1] (not padded to [1, 4, 8, 1]) proves the skip.


# CHECK-LABEL: shim_task_forwards_iteration
# CHECK: aie.dma_bd({{.*}} sizes = [4, 8, 1] strides = [512, 32, 1])
# CHECK-SAME: iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2>
@construct_and_print_module
def shim_task_forwards_iteration(module):
    N = 64

    @device(AIEDevice.npu1)
    def device_body():
        N_ty = np.ndarray[(N,), np.dtype[np.int32]]
        S = tile(0, 0)
        M = tile(0, 2)
        of_out = object_fifo("out", M, S, 2, N_ty)

        @runtime_sequence(N_ty)
        def sequence(C):
            task = shim_dma_single_bd_task(
                of_out,
                C,
                sizes=[4, 8, 1],
                strides=[512, 32, 1],
                iteration=BdIteration(size=4, stride=16, current=2),
                issue_token=True,
            )
            dma_start_task(task)

    return module


# >3 access dims with iteration= must raise.
# CHECK-LABEL: shim_task_rejects_extra_dims
# CHECK: ValueError: shim_dma_single_bd_task: iteration= supports at most 3 access dimensions
print("\nTEST: shim_task_rejects_extra_dims")


def _bad_iteration_dims(module):
    N = 64

    @device(AIEDevice.npu1)
    def device_body():
        N_ty = np.ndarray[(N,), np.dtype[np.int32]]
        S = tile(0, 0)
        M = tile(0, 2)
        of_out = object_fifo("out", M, S, 2, N_ty)

        @runtime_sequence(N_ty)
        def sequence(C):
            shim_dma_single_bd_task(
                of_out,
                C,
                sizes=[2, 4, 8, 1],
                strides=[4096, 512, 32, 1],
                iteration=BdIteration(size=4, stride=16, current=2),
            )

    return module


try:
    construct_and_print_module(_bad_iteration_dims)
except ValueError as e:
    print(f"ValueError: {e}")
