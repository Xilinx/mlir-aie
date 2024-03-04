//===- AIEVecToLLVM.h - AIEVec to LLVM dialect conversion -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_AIEVECTOLLVM_AIEVECTOLLVM_H
#define AIE_CONVERSION_AIEVECTOLLVM_AIEVECTOLLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;
} // namespace mlir

namespace xilinx {
namespace aievec {
void populateAIEVecToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                            mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEVecToLLVMPass();
} // namespace aievec
} // namespace xilinx

#endif // AIE_CONVERSION_AIEVECTOLLVM_AIEVECTOLLVM_H
