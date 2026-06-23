# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Backwards-compatible re-export from aie.utils."""

from aie.utils.callabledesign import CallableDesign
from aie.utils.jit import jit

__all__ = ["CallableDesign", "jit"]
