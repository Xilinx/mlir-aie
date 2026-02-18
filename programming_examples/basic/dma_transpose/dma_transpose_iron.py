# dma_transpose/dma_transpose_iron.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU1Col1, NPU2Col1, AnyComputeTile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib import TensorAccessPattern

if len(sys.argv) > 3:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))


def my_passthrough(M, K, generate_acccess_map=False):

    # Define types
    tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]

    # Define tensor access pattern
    tap_in = TensorAccessPattern((M, K)).tile_sequence((M, K), tile_dim_order=[1, 0])[0]

    # Use tensor access pattern to create a graph
    if generate_acccess_map:
        tap_in.visualize(file_path="iron_transpose_data.png", show_tile=False)
        return

    # Dataflow with ObjectFifos
    of_in = ObjectFifo(tensor_ty)
    of_out = of_in.cons().forward(AnyComputeTile)

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, _, c_out):
        rt.fill(of_in.prod(), a_in, tap_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(dev, rt)

    # Place program components (assign the resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())

    # Print the generated MLIR
    print(module)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("device_name", help="Device name (npu or npu2)", type=str)
    p.add_argument("dims", help="M K", type=int, nargs="*", default=[64, 64])
    p.add_argument(
        "--generate-access-map",
        action="store_true",
        help="Produce a file showing data access order",
    )
    args = p.parse_args()

    if len(args.dims) != 2:
        print(
            "ERROR: Must provide either no dimensions or both M and K", file=sys.stderr
        )
        exit(-1)
    my_passthrough(
        M=args.dims[0],
        K=args.dims[1],
        generate_acccess_map=args.generate_access_map,
    )
