# placers.py - DEPRECATED AND REMOVED
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.

"""
DEPRECATED: Python-based tile placement has been removed.

Tile placement is now handled by the MLIR compiler pass -aie-place-tiles.
IRON programs emit aie.logical_tile operations which are converted to
physical aie.tile operations during compilation.

Migration:
    OLD: module = program.resolve_program(SequentialPlacer())
    NEW: module = program.resolve_program()
"""

raise ImportError(
    "aie.iron.placers module has been removed. "
    "Tile placement is now handled by the MLIR -aie-place-tiles compiler pass. "
    "Remove 'from aie.iron.placers import SequentialPlacer' from your code."
)
