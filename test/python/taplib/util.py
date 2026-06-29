# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run test
def construct_test(f):
    print("\nTEST:", f.__name__)
    f()
