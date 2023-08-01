//===- TranslateLinalgToADF.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "AIETargets.h"
#include "aie/Analysis/BufferDescriptorAnalysis.h"
#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/ADF/ADFOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include <iostream>

using namespace mlir;
using namespace xilinx;

namespace xilinx {
namespace AIE {

mlir::LogicalResult TranslateLinalgToADF(ModuleOp module, raw_ostream &output) {
  module.walk([&](func::FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        BufferDescriptorState bd;
        BufferDescriptorAnalysis::visitOperandCopy(copyOp, bd);
        bd.print(llvm::errs());
        bd.printInt(llvm::errs());
      } else if (auto tensorStoreOp = dyn_cast<memref::TensorStoreOp>(op)) {
        BufferDescriptorState bd;
        BufferDescriptorAnalysis::visitOperandTensorStore(tensorStoreOp, bd);
        bd.print(llvm::errs());
        bd.printInt(llvm::errs());
      }
    });
  });

  return mlir::success();
}

} // namespace AIE
} // namespace xilinx
