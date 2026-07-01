#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect(XLLVM xllvm)
add_mlir_doc(XLLVMOps XLLVMOps ./ -gen-dialect-doc -dialect=xllvm)

# Add XLLVM operations
set(LLVM_TARGET_DEFINITIONS XLLVMOps.td)
mlir_tablegen(XLLVMOps.h.inc -gen-op-decls)
mlir_tablegen(XLLVMOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRXLLVMOpsIncGen)
mlir_tablegen(XLLVMConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRXLLVMConversionsIncGen)
