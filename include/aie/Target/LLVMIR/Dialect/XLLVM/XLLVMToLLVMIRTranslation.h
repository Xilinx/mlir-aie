//===- XLLVMToLLVMIRTranslation.h - XLLVM to LLVM IR translate -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for XLLVM intrinsics to LLVM IR
// translation.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGET_LLVMIR_DIALECT_XLLVM_H
#define AIE_TARGET_LLVMIR_DIALECT_XLLVM_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace xilinx::xllvm {

/// Register the AIEVec dialect and the translation from it to the LLVM dialect
/// in the given registry.
void registerXLLVMDialectTranslation(mlir::DialectRegistry &registry);

/// Register the AIEVec dialect and the translation from it to the LLVM dialect
/// in the registry associated with the given context.
void registerXLLVMDialectTranslation(mlir::MLIRContext &context);

} // namespace xilinx::xllvm

#endif // AIE_TARGET_LLVMIR_DIALECT_AIEVEC_H
