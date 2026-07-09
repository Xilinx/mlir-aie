#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Guards against the `aie` package becoming a PEP 420 namespace package again.
# As a namespace portion it never wins over a regular-package `aie` found later
# on sys.path, so a stale installed wheel can silently shadow the built tree
# (see python/__init__.py). A regular package has a non-None __file__ ending in
# __init__.py.

import aie

assert aie.__file__ is not None, "aie is a namespace package (no __init__.py)"
assert aie.__file__.endswith("__init__.py"), f"unexpected aie.__file__: {aie.__file__}"

# CHECK: aie is a regular package
print("aie is a regular package")
