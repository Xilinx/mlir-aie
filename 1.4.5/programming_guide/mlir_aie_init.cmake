# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# The programming-guide sections need exactly the same pre-project() preamble as
# the programming examples: host-compiler selection, the Windows output
# directory, and ProjectName/currentTarget. This file used to be a byte-for-byte
# copy of programming_examples/mlir_aie_init.cmake, which meant a fix landed on
# one side only -- and the guide sections are built by the same Windows CI job
# (check-programming-guide), so they hit the same llvm-aie clang++ selection bug.
#
# Forward instead, so the two cannot drift. A macro defined by a nested
# include() is global, so mlir_aie_init_example() reaches the caller exactly as
# if it were defined here. CMAKE_CURRENT_LIST_DIR is this file's directory, not
# the includer's, so the three call sites (section-3, section-4/section-4a,
# section-4/section-4b) keep working at their differing relative depths.
include("${CMAKE_CURRENT_LIST_DIR}/../programming_examples/mlir_aie_init.cmake")
