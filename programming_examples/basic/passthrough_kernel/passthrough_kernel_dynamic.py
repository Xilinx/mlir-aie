# passthrough_kernel/passthrough_kernel_dynamic.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
#
"""Dynamic passthrough TXN generation using the same IRON API as the static example."""

from aie.iron.device import NPU2

from passthrough_kernel import my_passthrough_kernel


if __name__ == "__main__":
    print(
        my_passthrough_kernel(
            NPU2(),
            4096,
            4096,
            0,
            dynamic_txn=True,
        )
    )
