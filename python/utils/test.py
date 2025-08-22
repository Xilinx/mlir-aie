# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import argparse
import pyxrt as xrt


# Add default args to standard parser object
def create_default_argparser():
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
    p.add_argument(
        "-i1s",
        "--in1_size",
        dest="in1_size",
        default=0,
        help="Input 1 buffer size in bytes",
    )
    p.add_argument(
        "-i2s",
        "--in2_size",
        dest="in2_size",
        default=0,
        help="Input 2 buffer size in bytes",
    )
    p.add_argument(
        "-os",
        "--out_size",
        dest="out_size",
        default=0,
        help="Output buffer size in bytes",
    )
    return p


# options
def parse_args(args):
    p = create_default_argparser()
    return p.parse_args(args)


#
# Create new device and kernel based on xclbin
#
# If you want to setup XRT buffers as well, look at xrt.py/setup_aie
# to setup your environment
#
def init_xrt_load_kernel(opts):
    # Get a device handle
    device = xrt.device(0)

    # Load the xclbin
    xclbin = xrt.xclbin(opts.xclbin)

    # Load the kernel
    kernels = xclbin.get_kernels()
    try:
        xkernel = [k for k in kernels if opts.kernel in k.get_name()][0]
    except:
        print(f"Kernel '{opts.kernel}' not found in '{opts.xclbin}'")
        exit(-1)

    # Register xclbin
    device.register_xclbin(xclbin)

    # Get a hardware context
    context = xrt.hw_context(device, xclbin.get_uuid())

    # get a kernel handle
    kernel = xrt.kernel(context, xkernel.get_name())

    return (device, kernel)
