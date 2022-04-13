//===- AIENormalizeAddressSpaces.cpp ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-normalize-address-spaces"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

Type memRefToDefaultAddressSpace(Type t) {
  auto memRefType = t.dyn_cast<MemRefType>();
  if (memRefType && memRefType.getMemorySpace() != 0)
    return MemRefType::get(memRefType.getShape(),
                            memRefType.getElementType(),
                            memRefType.getLayout(),
                            0 /* Address Space */);
  else 
    return t;
}

#include "aie/AIENormalizeAddressSpaces.inc"

struct AIENormalizeAddressSpacesPass
    : public AIENormalizeAddressSpacesBase<AIENormalizeAddressSpacesPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<memref::GlobalOp>([](memref::GlobalOp op) {
      return op.type().cast<MemRefType>().getMemorySpace() == 0;
    });
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();

    // Convert any output types to have the default address space
    m.walk([&](mlir::Operation *op) {
      for (Value r : op->getResults())
        r.setType(memRefToDefaultAddressSpace(r.getType()));
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIENormalizeAddressSpacesPass() {
  return std::make_unique<AIENormalizeAddressSpacesPass>();
}
