# section-4/section-4b/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

import aie.utils.trace as trace_utils


def my_vector_scalar(opts):

    enableTrace = opts.trace_size > 0
    trace_size = opts.trace_size

    @device(AIEDevice.npu1_1col)
    def device_body():
        tensor_ty = np.ndarray[(4096,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(1024,), np.dtype[np.int32]]
        scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

        # AIE Core Function declarations
        scale_scalar = external_func(
            "vector_scalar_mul_aie_scalar",
            inputs=[tile_ty, tile_ty, scalar_ty, np.int32],
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tile_ty)
        of_factor = object_fifo("infactor", ShimTile, ComputeTile2, 2, scalar_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tile_ty)

        # Set up compute tiles
        # Compute tile 2
        @core(ComputeTile2, "scale.o")
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                elem_factor = of_factor.acquire(ObjectFifoPort.Consume, 1)
                # Number of sub-vector "tile" iterations
                for _ in range_(4):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    scale_scalar(elem_in, elem_out, elem_factor, 1024)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                of_factor.release(ObjectFifoPort.Consume, 1)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [ComputeTile2]
        if enableTrace:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
        def sequence(A, F, C):
            if enableTrace:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace, ShimTile, opts.trace_size, 4096 * 4
                )

            npu_dma_memcpy_nd(
                metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, 4096], issue_token=True
            )
            npu_dma_memcpy_nd(
                metadata=of_factor,
                bd_id=2,
                mem=F,
                sizes=[1, 1, 1, 1],
                issue_token=True,
            )
            npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, 4096])
            dma_wait(of_in, of_factor, of_out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-t",
        "--trace_sz",
        dest="trace_size",
        default=0,
        type=int,
        help="trace size in bytes",
    )
    opts = p.parse_args(sys.argv[1:])
    with mlir_mod_ctx() as ctx:
        my_vector_scalar(opts)
        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)
