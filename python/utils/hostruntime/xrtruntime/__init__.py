# __init__.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

try:
    import pyxrt as xrt  # pyright: ignore[reportMissingImports]
except Exception as e:
    raise ImportError(f"Cannot import pyxrt (err={e})... is XRT installed?")
