# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

try:
    import pyxrt as xrt
except Exception as e:
    raise ImportError(f"Cannot import pyxrt (err={e})... is XRT installed?")
