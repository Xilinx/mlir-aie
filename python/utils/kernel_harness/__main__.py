# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""``python -m aie.utils.kernel_harness``: benchmark kernels; see ``bench.py``."""

import sys

from .bench import main

sys.exit(main())
