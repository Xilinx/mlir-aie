# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from aie.dialects.aie import (
    AIEDevice,
    ObjectFifoPort,
    core,
    device,
    external_func,
    object_fifo,
    tile,
)
from aie.dialects.aiex import *
from util import construct_and_print_module

# RUN: %python %s | FileCheck %s


@construct_and_print_module
def my_vector_scalar_memcpy(module):
    N = 4096
    n = 1024

    buffer_depth = 2

    @device(AIEDevice.npu1)
    def device_body():
        n_ty = np.ndarray[(n,), np.dtype[np.int32]]
        N_ty = np.ndarray[(N,), np.dtype[np.int32]]

        S = tile(0, 0)
        M = tile(0, 2)

        of_in = object_fifo("in", S, M, buffer_depth, n_ty)
        of_out = object_fifo("out", M, S, buffer_depth, n_ty)

        @runtime_sequence(N_ty, N_ty, N_ty)
        def sequence(A, B, C):
            # CHECK: burst_length = 64 : i64
            npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, burst_length=64)
            # CHECK: burst_length = 256 : i64
            npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, burst_length=256)
            dma_wait(of_out)

    return module


@construct_and_print_module
def my_vector_scalar_tasks(module):
    N = 4096
    n = 1024

    buffer_depth = 2

    @device(AIEDevice.npu1)
    def device_body():
        n_ty = np.ndarray[(n,), np.dtype[np.int32]]
        N_ty = np.ndarray[(N,), np.dtype[np.int32]]

        S = tile(0, 0)
        M = tile(0, 2)

        of_in = object_fifo("in", S, M, buffer_depth, n_ty)
        of_out = object_fifo("out", M, S, buffer_depth, n_ty)

        @runtime_sequence(N_ty, N_ty, N_ty)
        def sequence(A, B, C):
            # CHECK: burst_length = 64 : i32
            shim_dma_single_bd_task(of_in, A, burst_length=64)
            # CHECK: burst_length = 256 : i32
            shim_dma_single_bd_task(of_out, B, burst_length=256)
            dma_wait(of_out)

    return module
