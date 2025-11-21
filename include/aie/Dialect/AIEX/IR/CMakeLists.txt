#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2022 Xilinx Inc.

add_mlir_dialect(AIEX aiex)
add_mlir_doc(AIEX AIEXDialect ./ -gen-dialect-doc -dialect=aiex)

# AIEX dialect depends on AIE event enums being generated first
# because AIEX.td includes AIEAttrs.td which includes the generated files
add_dependencies(MLIRAIEXIncGen GenerateAIEEventsTD)
