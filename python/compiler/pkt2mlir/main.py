#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

from aie.ir import *
from aie.dialects.aie import *

import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Convert AIE control packet binaries to MLIR"
    )
    parser.add_argument(
        "-file",
        "-f",
        type=argparse.FileType("rb"),
        required=True,
        help="Input control packet binary file",
    )
    parser.add_argument(
        "-device",
        "-d",
        type=str,
        default="npu1",
        help="Target AIE device type (default: npu1)",
    )

    args = parser.parse_args()

    # Read the data from the file
    data = args.file.read()

    # Get the device enum value
    device_value = getattr(AIEDevice, args.device)

    with Context() as ctx:
        module = control_packets_binary_to_mlir(ctx, data, device_value)

    print(str(module))


if __name__ == "__main__":
    main()
