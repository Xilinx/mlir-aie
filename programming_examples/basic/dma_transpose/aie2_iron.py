# passthrough_kernel/aie2_iron.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.iron.runtime import Runtime
from aie.iron.dataflow import ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.program import Program
from aie.iron.phys.device import NPU1Col1
from aie.iron.phys.tile import AnyComputeTile
from aie.helpers.taplib import TensorTiler2D


def my_passthrough(M, K, generate_acccess_map=False):
    tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]

    tap_in = TensorTiler2D.simple_tiler((M, K), tile_col_major=True)[0]

    if generate_acccess_map:
        tap_in.visualize(file_path="iron_transpose_data.png", show_tile=False)
        return

    of_in = ObjectFifo(2, tensor_ty)
    of_out = of_in.cons.forward(AnyComputeTile)

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, _, c_out):
        rt.fill(of_in.prod, a_in, tap_in)
        rt.drain(of_out.cons, c_out, wait=True)

    my_program = Program(NPU1Col1(), rt)
    module = my_program.resolve_program(SequentialPlacer())
    print(module)


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
        generate_acccess_map=args.generate_access_map,
    )
