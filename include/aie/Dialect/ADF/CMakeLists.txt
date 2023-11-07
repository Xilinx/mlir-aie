#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_mlir_dialect(ADF ADF)
add_mlir_doc(ADF ADFDialect ./ -gen-dialect-doc)
set(LLVM_TARGET_DEFINITIONS ADF.td)
