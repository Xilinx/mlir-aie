//===- AIESetELFforCore.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "aie-set-elf-for-core"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIESetELFforCorePass : AIESetELFforCoreBase<AIESetELFforCorePass> {
  void runOnOperation() override {
    DeviceOp deviceOp = getOperation();
    if (!deviceOp.getSymName().empty() && !clDevice.empty() &&
        0 != deviceOp.getSymName().compare(clDevice)) {
      return;
    }
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);
    for (CoreOp coreOp : deviceOp.getOps<CoreOp>()) {
      if ((unsigned)coreOp.colIndex() != clTileCol ||
          (unsigned)coreOp.rowIndex() != clTileRow) {
        continue; // not the right tile
      }
      coreOp.setElfFile(clElfFile);
      coreOp.getBody().dropAllReferences();
      coreOp.getBody().getBlocks().clear();
      rewriter.createBlock(&coreOp.getBody());
      rewriter.create<EndOp>(coreOp.getLoc());
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIESetELFforCorePass() {
  return std::make_unique<AIESetELFforCorePass>();
}
