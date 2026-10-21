#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import sys

from aie.compiler.hsaco.dump import main  # pyright: ignore[reportMissingImports]

if __name__ == "__main__":
    sys.exit(main())
