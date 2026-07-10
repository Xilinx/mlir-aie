//===- All.h - MLIR-AIE to LLVM dialect conversion --------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to register the translations of all suitable
// AIE dialects to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGET_LLVMIR_DIALECT_ALL_H
#define AIE_TARGET_LLVMIR_DIALECT_ALL_H

#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace xilinx {
static inline void
registerAllAIEToLLVMIRTranslations(mlir::DialectRegistry &registry) {
  xllvm::registerXLLVMDialectTranslation(registry);
}
} // namespace xilinx

#endif // AIE_TARGET_LLVMIR_DIALECT_ALL_H
