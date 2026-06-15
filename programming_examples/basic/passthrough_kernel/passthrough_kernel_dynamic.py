# passthrough_kernel/passthrough_kernel_dynamic.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
#
"""Dynamic passthrough TXN generation using the same ``@iron.jit`` design.

This emits the MLIR for a runtime-parameterized passthrough (the transfer
length is an SSA argument of the runtime sequence rather than a compile-time
constant) and prints it to stdout.  The Makefile feeds that MLIR to a single
``aiecc`` invocation that produces both the XCLBIN and the C++ TXN header, so
one compiled design serves any transfer size up to the compiled-in maximum.

Because this path only needs MLIR (never on-device execution), it uses the
compile-only ``as_mlir`` entry point of ``@iron.jit`` rather than calling the
design, which would compile and run on the NPU.
"""

import argparse

import aie.utils as utils
from aie.iron.device import NPU1Col1, NPU2

from passthrough_kernel import my_passthrough_kernel

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--device", choices=["npu", "npu2"], default="npu2")
    p.add_argument("-i1s", "--in1_size", type=int, default=4096)
    args = p.parse_args()

    # Match the device selection used by passthrough_kernel.main: a single
    # column for "npu" (this is a one-column passthrough design).  The design
    # body reads the device via iron.get_current_device(), so set it here.
    dev = NPU2() if args.device == "npu2" else NPU1Col1()
    utils.set_current_device(dev)

    n_elems = args.in1_size  # uint8 kernel: byte size == element count
    print(my_passthrough_kernel.as_mlir(None, None, n=n_elems, dynamic_txn=True))
