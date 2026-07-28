# kernel.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Backwards-compatible re-export from aie.utils.kernel.

BaseKernel/Kernel/ExternalFunction have no iron-specific dependencies -- they
live in aie.utils so that aie.utils.compile (which needs ExternalFunction to
drive its compile pipeline) doesn't have to import through aie.iron and risk
a circular import.
"""

from aie.utils.kernel import BaseKernel, ExternalFunction, Kernel

__all__ = ["BaseKernel", "ExternalFunction", "Kernel"]
