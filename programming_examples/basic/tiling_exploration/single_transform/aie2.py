# tiling_exploration/aie2.py -*- Python -*-
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
from aie.dialects import arith
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D


def generate_module(tensor_height, tensor_width, tile_height, tile_width):
    @device(AIEDevice.npu1_1col)
    def device_body():
        # define types
        tensor_size = tensor_height * tensor_width
        flattened_tensor = np.ndarray[(tensor_size,), np.dtype[TensorTiler2D.DTYPE]]

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, flattened_tensor)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            for _ in range_(sys.maxsize):
                elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(tensor_size):
                    # TODO: fix need for cast here.
                    elemOut[i] = arith.index_cast(T.i32(), i)
                of_out.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(flattened_tensor)
        def sequence(access_count):
            t = TensorTiler2D(
                tensor_height, tensor_width, tile_height, tile_width
            ).as_tile()
            npu_dma_memcpy_nd(
                metadata=of_out,
                bd_id=1,
                mem=access_count,
                tensor_tile=t,
            )
            dma_wait(of_out)


def main(opts):
    with mlir_mod_ctx() as ctx:
        generate_module(
            opts.tensor_height, opts.tensor_width, opts.tile_height, opts.tile_width
        )
        print(ctx.module)


def get_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--tensor-height", required=True, help="Tensor height", type=int)
    p.add_argument("--tensor-width", required=True, help="Tensor width", type=int)
    p.add_argument("--tile-height", required=True, help="Tile height", type=int)
    p.add_argument("--tile-width", required=True, help="Tile width", type=int)
    return p


if __name__ == "__main__":
    p = get_arg_parser()
    opts = p.parse_args()
    main(opts)