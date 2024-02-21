//======- AIEVecToLLVMIRTranslation.cpp - Translate AIEVec to LLVM IR -=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the AIEVec dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "aie/Target/LLVMIR/Dialect/AIEVec/AIEVecToLLVMIRTranslation.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Utils/Utils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace xilinx;
using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the AIEVec dialect to LLVM IR.
class AIEVecDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "aie/Dialect/AIEVec/IR/AIEVecConversions.inc"

    return failure();
  }
};
} // namespace

void xilinx::aievec::registerAIEVecDialectTranslation(
    DialectRegistry &registry) {
  registry.insert<aievec::AIEVecDialect>();
  registry.addExtension(+[](MLIRContext *ctx, aievec::AIEVecDialect *dialect) {
    dialect->addInterfaces<AIEVecDialectLLVMIRTranslationInterface>();
  });
}

void xilinx::aievec::registerAIEVecDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerAIEVecDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
