//===---- XLLVMOps.cpp - XLLVM Dialect Operations ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// External LLVM (XLLVM) Dialect implementation.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/XLLVM/XLLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/IR/IRBuilder.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::xllvm;

#include "aie/Dialect/XLLVM/IR/XLLVMDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// XLLVMDialect
//===----------------------------------------------------------------------===//

void xllvm::XLLVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/XLLVM/IR/XLLVMOps.cpp.inc"
      >();
}

namespace xilinx::xllvm {

static llvm::Function *
getNamedIntrinsicDeclaration(llvm::Module *M, llvm::StringRef fullName,
                             llvm::Type *resTy, ArrayRef<llvm::Type *> argsTy) {
  auto *FT = llvm::FunctionType::get(resTy, argsTy, /*isVarArg=*/false);
  return cast<llvm::Function>(M->getOrInsertFunction(fullName, FT).getCallee());
}

llvm::CallInst *createExternalLLVMIntrinsicCall(
    llvm::IRBuilderBase &builder, LLVM::ModuleTranslation &moduleTranslation,
    Operation *intrOp, llvm::StringRef intrinsicName) {
  llvm::Type *resTy = nullptr;
  unsigned numResults = intrOp->getNumResults();
  if (numResults == 1)
    resTy = moduleTranslation.convertType(*(intrOp->getResultTypes().begin()));
  else if (numResults > 1) {
    SmallVector<llvm::Type *> resTys;
    for (auto ty : intrOp->getResultTypes())
      resTys.push_back(moduleTranslation.convertType(ty));
    resTy = llvm::StructType::get(builder.getContext(), resTys);
  }
  auto operands = moduleTranslation.lookupValues(intrOp->getOperands());
  SmallVector<llvm::Type *> types;
  for (auto op : operands)
    types.push_back(op->getType());
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *llvmIntr =
      getNamedIntrinsicDeclaration(module, intrinsicName, resTy, types);
  return builder.CreateCall(llvmIntr, operands);
}

} // namespace xilinx::xllvm

#define GET_OP_CLASSES
#include "aie/Dialect/XLLVM/IR/XLLVMOps.cpp.inc"
