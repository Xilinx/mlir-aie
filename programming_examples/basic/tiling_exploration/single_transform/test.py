# tiling_exploration/test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D
from aie.utils.xrt import setup_aie, execute as execute_on_aie


def main(opts):
    print("Running...\n")

    dtype = TensorTiler2D.DTYPE
    data_size = opts.tensor_height * opts.tensor_width

    reference_tiler = TensorTiler2D(
        opts.tensor_height, opts.tensor_width, opts.tile_height, opts.tile_width
    )
    reference_access_order = reference_tiler.access_order()

    app = setup_aie(
        opts.xclbin,
        opts.instr,
        None,
        None,
        None,
        None,
        data_size,
        dtype,
    )
    aie_output = execute_on_aie(app)
    aie_output = aie_output.reshape((opts.tensor_height, opts.tensor_width))

    # Copy output results and verify they are correct
    errors = 0
    if opts.verbosity >= 1:
        print("Verifying results ...")
    e = np.equal(reference_access_order, aie_output)
    errors = np.size(e) - np.count_nonzero(e)

    if not errors:
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        exit(-1)


def get_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-x", "--xclbin", default="final.xclbin", dest="xclbin", help="the xclbin path"
    )
    p.add_argument(
        "-k",
        "--kernel",
        dest="kernel",
        default="MLIR_AIE",
        help="the kernel name in the XCLBIN (for instance MLIR_AIE)",
    )
    p.add_argument(
        "-v", "--verbosity", default=0, type=int, help="the verbosity of the output"
    )
    p.add_argument(
        "-i",
        "--instr",
        dest="instr",
        default="instr.txt",
        help="path of file containing userspace instructions sent to the NPU",
    )
    p.add_argument("--tensor-height", required=True, help="Tensor height", type=int)
    p.add_argument("--tensor-width", required=True, help="Tensor width", type=int)
    p.add_argument("--tile-height", required=True, help="Tile height", type=int)
    p.add_argument("--tile-width", required=True, help="Tile width", type=int)
    return p


if __name__ == "__main__":
    p = get_arg_parser()
    opts = p.parse_args()
    main(opts)
