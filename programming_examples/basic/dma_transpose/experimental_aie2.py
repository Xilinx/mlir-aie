# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.api.io.iocoordinator import IOCoordinator
from aie.api.dataflow.objectfifo import ObjectFifo
from aie.api.placers import SequentialPlacer
from aie.api.program import Program
from aie.api.phys.device import NPU1Col1
from aie.api.phys.tile import AnyComputeTile
from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D


def my_passthrough(M, K, N, generate_acccess_map=False):
    tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]

    tiler_in = TensorTiler2D(M, K, tensor_col_major=True)
    tiler_out = TensorTiler2D(K, M)

    if generate_acccess_map:
        tiler_in.visualize(file_path="experimental_transpose_data.png", show_tile=False)
        return

    of_in = ObjectFifo(2, tensor_ty)
    of_out = of_in.second.forward(AnyComputeTile)

    io = IOCoordinator()
    with io.build_sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, _, c_out):
        for t_in, t_out in io.tile_loop(tiler_in.tile_iter(), tiler_out.tile_iter()):
            io.fill(of_in.first, t_in, a_in)
            io.drain(of_out.second, t_out, c_out, wait=True)

    my_program = Program(NPU1Col1(), io)
    my_program.resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
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
        N=args.dims[0] * args.dims[1],
        generate_acccess_map=args.generate_access_map,
    )
