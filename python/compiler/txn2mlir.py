#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

import aie
from aie.ir import *
from aie.dialects.aie import *

import argparse

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "-f", type=argparse.FileType("rb"))

    args = parser.parse_args()

    # Read the data from the file
    data = args.file.read()

    with Context() as ctx:
        module = transaction_binary_to_mlir(ctx, data)

    print(str(module))
