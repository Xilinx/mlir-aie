# tiling_exploration/per_tile/per_tile.py-*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.iron import LocalBuffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def generate_module(
    tensor_height, tensor_width, tile_height, tile_width, generate_access_map=False
):
    tensor_size = tensor_height * tensor_width
    tile_size = tile_height * tile_width

    # define data types and tensor types
    dtype = np.int32
    flattened_tensor = np.ndarray[(tensor_size,), np.dtype[dtype]]
    flattened_tile = np.ndarray[(tile_size,), np.dtype[dtype]]

    # Define tensor access pattern on the input/output tensor (tiling)
    tiler = TensorTiler2D.simple_tiler(
        (tensor_height, tensor_width), (tile_height, tile_width)
    )

    # Generate a graph from the tensor access pattern
    if generate_access_map:
        tiler.visualize(file_path="per_tile.png")
        return

    # Use an ObjectFifo for dataflow
    of_out = ObjectFifo(flattened_tile)

    # The task a core will run
    def access_order(of_out):
        access_counter = LocalBuffer(initial_value=np.array([0], dtype=dtype))

        for _ in range_(sys.maxsize):
            elemOut = of_out.acquire(1)
            for i in range_(tile_size):
                elemOut[i] = access_counter[0]
                access_counter[0] += 1
            of_out.release(1)
        pass

    # Create a worker (which will be placed on a core) to run the task
    worker = Worker(access_order, [of_out.prod()], while_true=False)

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(flattened_tensor) as tensor_out:
        rt.start(worker)
        for t in tiler:
            rt.drain(of_out.cons(), tensor_out, t, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(NPU1Col1(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    return my_program.resolve_program(SequentialPlacer())


def main(opts):
    module = generate_module(
        opts.tensor_height,
        opts.tensor_width,
        opts.tile_height,
        opts.tile_width,
        opts.generate_access_map,
    )
    if not opts.generate_access_map:
        print(module)


def get_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--tensor-height", required=True, help="Tensor height", type=int)
    p.add_argument("--tensor-width", required=True, help="Tensor width", type=int)
    p.add_argument("--tile-height", required=True, help="Tile height", type=int)
    p.add_argument("--tile-width", required=True, help="Tile width", type=int)
    p.add_argument(
        "--generate-access-map",
        action="store_true",
        help="Produce a file showing data access order",
    )
    return p


if __name__ == "__main__":
    p = get_arg_parser()
    opts = p.parse_args()
    main(opts)
