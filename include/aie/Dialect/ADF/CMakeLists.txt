#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# the default add_mlir_dialect module will use table_gen to generate
# op_decls/defs, typedef_decls/defs, and dialect_decls/defs at 
# add_mlir_dialect()
add_mlir_dialect(ADF ADF)
add_mlir_doc(ADF ADFDialect ./ -gen-dialect-doc)
set(LLVM_TARGET_DEFINITIONS ADF.td)

# note: 
# you will need to add these tablegen statements in cmake if you 
# have enum, interfaces, and passes. Reference AIE when needed

set(LLVM_TARGET_DEFINITIONS ADFPasses.td)
mlir_tablegen(ADFPasses.h.inc -gen-pass-decls -name ADF)
add_public_tablegen_target(MLIRADFPassIncGen)

add_mlir_doc(ADFPasses ADFPasses ./ -gen-pass-doc)
