# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""High-level algorithm templates built on IRON (transform, for_each, etc.)."""

from .for_each import for_each, for_each_typed
from .transform import (
    transform,
    transform_binary,
    transform_binary_typed,
    transform_parallel,
    transform_parallel_binary,
    transform_parallel_binary_typed,
    transform_parallel_typed,
    transform_typed,
)
