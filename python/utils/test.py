# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import argparse
from aie.iron.hostruntime import TraceConfig


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
        "--trace-sz",
        dest="trace_size",
        default=0,
        type=int,
        help="trace size in bytes",
    )
    p.add_argument(
        "--trace-file",
        dest="trace_file",
        default="trace.txt",
        help="where to store trace output",
    )
    p.add_argument(
        "-i1s",
        "--in1-size",
        dest="in1_size",
        default=0,
        help="Input 1 buffer size in bytes",
    )
    p.add_argument(
        "-i2s",
        "--in2-size",
        dest="in2_size",
        default=0,
        help="Input 2 buffer size in bytes",
    )
    p.add_argument(
        "-os",
        "--out-size",
        dest="out_size",
        default=0,
        help="Output buffer size in bytes",
    )
    p.add_argument(
        "--trace-after-output",
        dest="trace_after_output",
        action="store_true",
        help="Trace after output",
    )
    p.add_argument(
        "--enable-ctrl-pkts",
        dest="enable_ctrl_pkts",
        action="store_true",
        help="Enable control packets",
    )
    return p


def parse_trace_config(opts):
    trace_config = None
    trace_size = getattr(opts, "trace_size", 0)
    if trace_size > 0:
        trace_config = TraceConfig(
            trace_size=trace_size,
            trace_file=getattr(opts, "trace_file", "trace.txt"),
            trace_after_last_tensor=getattr(opts, "trace_after_output", False),
            enable_ctrl_pkts=getattr(opts, "enable_ctrl_pkts", False),
        )
    opts.trace_config = trace_config
    return opts


# options
def parse_args(args):
    p = create_default_argparser()
    opts = p.parse_args(args)
    return parse_trace_config(opts)
