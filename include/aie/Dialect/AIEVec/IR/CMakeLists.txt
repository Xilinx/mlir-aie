#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2022 Xilinx Inc.

add_mlir_dialect(AIEVecOps aievec)
add_mlir_doc(AIEVecOps AIEVecDialect ./ -gen-dialect-doc -dialect=aievec)

