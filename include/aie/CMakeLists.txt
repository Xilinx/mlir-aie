#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_mlir_dialect(AIE AIE)
add_mlir_doc(AIE AIEDialect ./ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AIE.td)
mlir_tablegen(AIEEnums.h.inc -gen-enum-decls)
mlir_tablegen(AIEEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIEEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS AIEInterfaces.td)
mlir_tablegen(AIEInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AIEInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAIEInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS AIEPasses.td)
mlir_tablegen(AIEPasses.h.inc -gen-pass-decls -name AIE)
add_public_tablegen_target(MLIRAIEPassIncGen)

add_mlir_doc(AIEPasses AIEPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS AIENormalizeAddressSpaces.td)
mlir_tablegen(AIENormalizeAddressSpaces.inc -gen-rewriters)
add_public_tablegen_target(MLIRAIENormalizeAddressSpacesIncGen)

# set(LLVM_TARGET_DEFINITIONS AIE.td)
# mlir_tablegen(LLVMConversions.inc -gen-llvmir-conversions)
# mlir_tablegen(LLVMConversionEnumsToLLVM.inc -gen-enum-to-llvmir-conversions)
# mlir_tablegen(LLVMConversionEnumsFromLLVM.inc -gen-enum-from-llvmir-conversions)
# add_public_tablegen_target(MLIRLLVMConversionsIncGen)
