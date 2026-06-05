# passthrough_kernel/passthrough_kernel_dynamic.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
#
"""Dynamic passthrough TXN generation using the same IRON API as the static example."""

import argparse

from aie.iron.device import NPU1, NPU2

from passthrough_kernel import my_passthrough_kernel

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--device", choices=["npu", "npu2"], default="npu2")
    p.add_argument("-i1s", "--in1_size", type=int, default=4096)
    p.add_argument("-os", "--out_size", type=int, default=4096)
    args = p.parse_args()

    dev = NPU2() if args.device == "npu2" else NPU1()
    print(
        my_passthrough_kernel(
            dev,
            args.in1_size,
            args.out_size,
            0,
            dynamic_txn=True,
        )
    )
