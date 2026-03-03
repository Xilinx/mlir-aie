#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

# Edge detection pipeline using the C ObjectFIFO API.
# Same algorithm as edge_detect but all core body logic (loops, acquire/release,
# buffer rotation) is managed by C kernels using objectfifo_t.
#
# All OFs with depth >= 2 use 2-buffer ping-pong (except OF_2to3 consumer
# side in filter2d which needs 4 buffers for the sliding window).
# OF_local has depth 1, so it uses a single buffer.

import numpy as np
import sys
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx


def edge_detect_c_objfifo(dev, width, height):
    heightMinus1 = height - 1
    lineWidth = width
    lineWidthInBytes = width * 4
    tensorSize = width * height * 4

    line_bytes_ty = np.ndarray[(lineWidthInBytes,), np.dtype[np.uint8]]
    line_ty = np.ndarray[(lineWidth,), np.dtype[np.uint8]]
    tensor_3x3_ty = np.ndarray[(3, 3), np.dtype[np.int16]]
    tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]
    tensor_16x16_ty = np.ndarray[(16, 16), np.dtype[np.int32]]

    @device(dev)
    def device_body():
        # Tiles
        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        tile2 = tile(0, 2)  # rgba2gray
        tile3 = tile(0, 3)  # filter2d
        tile4 = tile(0, 4)  # threshold
        tile5 = tile(0, 5)  # gray2rgba + addWeighted

        # ObjectFIFOs - input path
        # Depths: producer=2, mem_tile consumer=7 (for forwarding), tile2 consumer=2
        inOF_L3L2 = object_fifo("inOF_L3L2", shim_tile, [mem_tile, tile2],
                                [2, 7, 2], line_bytes_ty)
        # Asymmetric depth: mem_tile producer=7 (pipelining), tile5 consumer=4
        inOF_L2L1 = object_fifo("inOF_L2L1", mem_tile, tile5, [7, 4], line_bytes_ty)
        object_fifo_link(inOF_L3L2, inOF_L2L1, [], [0])

        # ObjectFIFOs - output path
        outOF_L1L2 = object_fifo("outOF_L1L2", tile5, mem_tile, 2, line_bytes_ty)
        outOF_L2L3 = object_fifo("outOF_L2L3", mem_tile, shim_tile, 2, line_bytes_ty)
        object_fifo_link(outOF_L1L2, outOF_L2L3, [], [0])

        # ObjectFIFOs - intermediate pipeline
        OF_2to3 = object_fifo("OF_2to3", tile2, tile3, 4, line_ty)
        OF_3to4 = object_fifo("OF_3to4", tile3, tile4, 2, line_ty)
        OF_4to5 = object_fifo("OF_4to5", tile4, tile5, 2, line_ty)

        # ObjectFIFO - local feedback on tile5
        OF_local = object_fifo("OF_local", tile5, tile5, 1, line_bytes_ty)

        # Sobel kernel coefficients
        v0 = 0
        v1 = 4096
        v_minus4 = -16384
        initial_value = np.array(
            [[v0, v1, v0], [v1, v_minus4, v1], [v0, v1, v0]], dtype=np.int16
        )
        kernel_buf = buffer(tile3, tensor_3x3_ty, "kernel",
                           initial_value=initial_value)

        # ---- C wrapper kernel declarations ----

        # rgba2gray_core: 2 in bufs (ping-pong), 4 out bufs (OF_2to3 depth=4)
        # Producer must match full OF depth since DMA BD chain cycles all bufs
        rgba2gray_fn = external_func(
            "rgba2gray_core",
            inputs=[
                line_bytes_ty,                             # in buf 0
                line_bytes_ty,                             # in buf 1
                line_ty,                                   # out buf 0
                line_ty,                                   # out buf 1
                line_ty,                                   # out buf 2
                line_ty,                                   # out buf 3
                T.index(), T.index(),                      # in locks
                T.index(), T.index(),                      # out locks
                T.i32(),                                   # lineWidth
            ],
        )

        # filter2d_core: in needs 4 buffers (sliding window of 3, depth 4)
        # out needs 2 buffers (ping-pong, depth 2)
        filter2d_fn = external_func(
            "filter2d_core",
            inputs=[
                line_ty, line_ty, line_ty, line_ty,        # in bufs (4)
                line_ty, line_ty,                          # out bufs (2)
                T.index(), T.index(),                      # in locks
                T.index(), T.index(),                      # out locks
                tensor_3x3_ty,                             # kernel coeffs
                T.i32(),                                   # lineWidth
                T.i32(),                                   # height
            ],
        )

        # threshold_core: 2 in bufs (ping-pong), 2 out bufs (ping-pong)
        threshold_fn = external_func(
            "threshold_core",
            inputs=[
                line_ty, line_ty,                          # in bufs (2)
                line_ty, line_ty,                          # out bufs (2)
                T.index(), T.index(),                      # in locks
                T.index(), T.index(),                      # out locks
                T.i32(),                                   # lineWidth
            ],
        )

        # gray2rgba_addweighted_core:
        #   OF_4to5 (in): 2 bufs ping-pong
        #   inOF_L2L1 (in2): 4 bufs (consumer depth matches DMA BD chain)
        #   OF_local (local): 1 buf (depth 1)
        #   outOF_L1L2 (out): 2 bufs ping-pong
        gray2rgba_addweighted_fn = external_func(
            "gray2rgba_addweighted_core",
            inputs=[
                line_ty, line_ty,                                          # in bufs (OF_4to5, 2)
                line_bytes_ty, line_bytes_ty,                              # in2 bufs (inOF_L2L1, 4)
                line_bytes_ty, line_bytes_ty,
                line_bytes_ty,                                             # local buf (depth 1)
                line_bytes_ty, line_bytes_ty,                              # out bufs (outOF_L1L2, 2)
                T.index(), T.index(),                                      # in locks (OF_4to5)
                T.index(), T.index(),                                      # in2 locks (inOF_L2L1)
                T.index(), T.index(),                                      # local prod locks
                T.index(), T.index(),                                      # local cons locks
                T.index(), T.index(),                                      # out locks (outOF_L1L2)
                T.i32(),                                                   # lineWidth
                T.i32(),                                                   # lineWidthInBytes
            ],
        )

        # ---- Core bodies ----

        @core(tile2, "rgba2gray_wrapper.a")
        def core2():
            in_b0 = inOF_L3L2.get_buffer(0)
            in_b1 = inOF_L3L2.get_buffer(1)
            in_acq, in_rel = inOF_L3L2.get_lock(ObjectFifoPort.Consume)

            # Producer must get all 4 buffers to match OF_2to3 depth
            out_b0 = OF_2to3.get_buffer(0)
            out_b1 = OF_2to3.get_buffer(1)
            out_b2 = OF_2to3.get_buffer(2)
            out_b3 = OF_2to3.get_buffer(3)
            out_acq, out_rel = OF_2to3.get_lock(ObjectFifoPort.Produce)

            rgba2gray_fn(in_b0, in_b1, out_b0, out_b1, out_b2, out_b3,
                         in_acq, in_rel, out_acq, out_rel,
                         lineWidth)

        @core(tile3, "filter2d_wrapper.a")
        def core3():
            in_b0 = OF_2to3.get_buffer(0)
            in_b1 = OF_2to3.get_buffer(1)
            in_b2 = OF_2to3.get_buffer(2)
            in_b3 = OF_2to3.get_buffer(3)
            in_acq, in_rel = OF_2to3.get_lock(ObjectFifoPort.Consume)

            out_b0 = OF_3to4.get_buffer(0)
            out_b1 = OF_3to4.get_buffer(1)
            out_acq, out_rel = OF_3to4.get_lock(ObjectFifoPort.Produce)

            filter2d_fn(in_b0, in_b1, in_b2, in_b3, out_b0, out_b1,
                        in_acq, in_rel, out_acq, out_rel,
                        kernel_buf, lineWidth, height)

        @core(tile4, "threshold_wrapper.a")
        def core4():
            in_b0 = OF_3to4.get_buffer(0)
            in_b1 = OF_3to4.get_buffer(1)
            in_acq, in_rel = OF_3to4.get_lock(ObjectFifoPort.Consume)

            out_b0 = OF_4to5.get_buffer(0)
            out_b1 = OF_4to5.get_buffer(1)
            out_acq, out_rel = OF_4to5.get_lock(ObjectFifoPort.Produce)

            threshold_fn(in_b0, in_b1, out_b0, out_b1,
                         in_acq, in_rel, out_acq, out_rel,
                         lineWidth)

        @core(tile5, "gray2rgba_addweighted_wrapper.a")
        def core5():
            # OF_4to5: depth 2, 2 bufs
            t5_in_b0 = OF_4to5.get_buffer(0)
            t5_in_b1 = OF_4to5.get_buffer(1)
            t5_in_acq, t5_in_rel = OF_4to5.get_lock(ObjectFifoPort.Consume)

            # inOF_L2L1: consumer depth 4, all 4 bufs (must match DMA BD chain)
            in2_b0 = inOF_L2L1.get_buffer(0)
            in2_b1 = inOF_L2L1.get_buffer(1)
            in2_b2 = inOF_L2L1.get_buffer(2)
            in2_b3 = inOF_L2L1.get_buffer(3)
            in2_acq, in2_rel = inOF_L2L1.get_lock(ObjectFifoPort.Consume)

            # OF_local: depth 1, 1 buf
            local_b0 = OF_local.get_buffer(0)
            local_p_acq, local_p_rel = OF_local.get_lock(ObjectFifoPort.Produce)
            local_c_acq, local_c_rel = OF_local.get_lock(ObjectFifoPort.Consume)

            # outOF_L1L2: depth 2, 2 bufs
            out_b0 = outOF_L1L2.get_buffer(0)
            out_b1 = outOF_L1L2.get_buffer(1)
            out_acq, out_rel = outOF_L1L2.get_lock(ObjectFifoPort.Produce)

            gray2rgba_addweighted_fn(
                t5_in_b0, t5_in_b1,
                in2_b0, in2_b1, in2_b2, in2_b3,
                local_b0,
                out_b0, out_b1,
                t5_in_acq, t5_in_rel,
                in2_acq, in2_rel,
                local_p_acq, local_p_rel,
                local_c_acq, local_c_rel,
                out_acq, out_rel,
                lineWidth, lineWidthInBytes,
            )

        # Runtime sequence
        @runtime_sequence(tensor_ty, tensor_16x16_ty, tensor_ty)
        def sequence(inTensor, _unused, outTensor):
            in_task = shim_dma_single_bd_task(
                inOF_L3L2, inTensor,
                sizes=[1, 1, 1, tensorSize],
            )
            out_task = shim_dma_single_bd_task(
                outOF_L2L3, outTensor,
                sizes=[1, 1, 1, tensorSize],
                issue_token=True,
            )
            dma_start_task(in_task)
            dma_start_task(out_task)
            dma_await_task(out_task)
            dma_free_task(in_task)


if len(sys.argv) < 2:
    raise ValueError("[ERROR] Need device argument (npu or npu2)")

p = argparse.ArgumentParser()
p.add_argument("device")
p.add_argument("width", type=int, nargs="?", default=1920)
p.add_argument("height", type=int, nargs="?", default=1080)
opts = p.parse_args()

if opts.device == "npu":
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError(f"Unknown device: {opts.device}")

with mlir_mod_ctx() as ctx:
    edge_detect_c_objfifo(dev, opts.width, opts.height)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
