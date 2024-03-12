//===- All.h - MLIR-AIE to LLVM dialect conversion --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
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
