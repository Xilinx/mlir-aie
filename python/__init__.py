# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.

# Makes `aie` a regular package, not a PEP 420 namespace package. A namespace
# portion never wins over a regular-package `aie` found later on sys.path, so
# without this the published mlir_aie wheel silently shadows the locally built
# tree even when the build is first on PYTHONPATH. The wheel build appends a
# version block here, so keep this import-safe and side-effect free.
