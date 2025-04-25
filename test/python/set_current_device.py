# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %python %s | FileCheck %s

from aie.iron.device import NPU2

import aie.iron as iron

# CHECK: NPU2
def main():
    iron.set_current_device(NPU2())
    device = iron.get_current_device()
    print(device)

if __name__ == "__main__":
    main()
  
