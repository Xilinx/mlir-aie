# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A strided fill and a strided drain too large for one BD each: PAIRS pairs of
# elements at stride 3, PAIRS prime so that no factoring fits the pattern into
# the BD's fields. Each splits into 18 slices, more than the channel queues (4)
# and, fill and drain together, more than the shim tile's 16 BDs. So each slice
# becomes a task of its own, and the BD-id pass takes back the BDs of finished
# ones. The mem tile holds only two pairs, so the fill cannot get ahead of the
# drain: its slices have to be issued in step with the drain's. Every slice
# must move its own pairs: a wrong offset, a lost slice or an early await shows
# up as a mismatch, and a bad issue order as a hang.

# REQUIRES: ryzen_ai_npu2, peano
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: aie-opt --aie-objectFifo-stateful-transform \
# RUN:   --aie-substitute-shim-dma-allocations \
# RUN:   --aie-decompose-large-dma-bd \
# RUN:   --aie-assign-runtime-sequence-bd-ids ./aie2.mlir \
# RUN:   | FileCheck %s --check-prefix=MLIR
# RUN: %aiecc --get-xclbin --get-npu-insts --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: %host_clang %S/test.cpp -o test.exe -std=c++17 -Wall -Wextra %xrt_flags %host_link_flags %test_utils_flags
# RUN: %run_on_npu2% ./test.exe | FileCheck %s --check-prefix=DEVICE
# DEVICE: PASS!

# 36 single-BD tasks, fill and drain slices alternating; the 17th takes back
# BD 0.
# MLIR-COUNT-16: {bd_id = {{[0-9]+}} : i32}
# MLIR:          {bd_id = 0 : i32}

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

PAIRS = 17393
STRIDE = 3
LEN = 65536


def design():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            buff_ty = np.ndarray[(LEN,), np.dtype[np.int32]]
            obj_ty = np.ndarray[(2,), np.dtype[np.int32]]

            shim = tile(0, 0)
            mem = tile(0, 1)

            of_in = object_fifo("in", shim, mem, 2, obj_ty)
            of_out = object_fifo("out", mem, shim, 2, obj_ty)
            object_fifo_link(of_in, of_out)

            @runtime_sequence(buff_ty, buff_ty)
            def sequence(A, B):
                pattern = dict(sizes=[1, 1, PAIRS, 2], strides=[0, 0, STRIDE, 1])
                fill = dma_configure_task_for(of_in)
                with bds(fill) as bd:
                    with bd[0]:
                        shim_dma_bd(A, **pattern)
                        EndOp()
                dma_start_task(fill)
                drain = dma_configure_task_for(of_out, issue_token=True)
                with bds(drain) as bd:
                    with bd[0]:
                        shim_dma_bd(B, **pattern)
                        EndOp()
                dma_start_task(drain)
                dma_await_task(drain)

    print(ctx.module)


design()
