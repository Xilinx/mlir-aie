#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

configure_file(version.h.in version.h)
if (CLANGIR_MLIR_FRONTEND)
  add_subdirectory(CIR)
endif()
add_subdirectory(Conversion)
add_subdirectory(Dialect)
