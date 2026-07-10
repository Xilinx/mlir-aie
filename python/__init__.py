# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Makes `aie` a regular package, not a PEP 420 namespace package. A namespace
# portion never wins over a regular-package `aie` found later on sys.path, so
# without this the published mlir_aie wheel silently shadows the locally built
# tree even when the build is first on PYTHONPATH. The wheel build appends a
# version block here, so keep this import-safe and side-effect free.
