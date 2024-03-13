# test_utils.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse

# options
def parse_args(args):
    p = argparse.ArgumentParser()
    p.add_argument(
        "-x", "--xclbin", required=True, dest="xclbin", help="the input xclbin path"
    )
    p.add_argument(
        "-k",
        "--kernel",
        required=True,
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
    p.add_argument(
        "--verify",
        dest="verify",
        default=True,
        help="whether to verify the AIE computed output",
    )
    p.add_argument(
        "--iters",
        dest="iters",
        default=1,
        type=int,
        help="number of benchmark iterations",
    )
    p.add_argument(
        "--warmup",
        dest="warmup_iters",
        default=0,
        type=int,
        help="number of warmup iterations",
    )
    p.add_argument(
        "-t",
        "--trace_sz",
        dest="trace_size",
        default=0,
        type=int,
        help="trace size in bytes",
    )
    p.add_argument(
        "--trace_file",
        dest="trace_file",
        default="trace.txt",
        help="where to store trace output",
    )
    return p.parse_args(args)
