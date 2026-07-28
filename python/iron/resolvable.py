# resolvable.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Backwards-compatible re-export from aie.utils.resolvable.

Resolvable has no iron-specific dependencies -- it lives in aie.utils so that
aie.utils.compile (which needs the same structural protocol) doesn't have to
import through aie.iron and risk a circular import.
"""

from aie.utils.resolvable import NotResolvedError, Resolvable

__all__ = ["NotResolvedError", "Resolvable"]
