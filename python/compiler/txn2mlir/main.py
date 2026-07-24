#!/usr/bin/env python3
#
# Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import argparse

from aie.dialects.aie import (
    transaction_binary_to_mlir,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.ir import (  # pyright: ignore[reportMissingImports]
    Context,
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "-f", type=argparse.FileType("rb"))

    args = parser.parse_args()

    # Read the data from the file
    data = args.file.read()

    with Context() as ctx:
        module = transaction_binary_to_mlir(ctx, data)

    print(str(module))
