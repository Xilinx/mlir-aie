//===- PhyToAie.cpp -------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Conversion/PhysicalToAie.h"
#include "phy/Conversion/Passes.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Transform/AIE/LoweringPatterns.h"

#include "aie/AIEDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-physical-to-aie"

using namespace mlir;
using namespace xilinx::phy;

namespace {

// Run a pre pipeline of cleanup passes.
static void preCanonicalizeIR(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createInlinerPass());
  assert(!failed(pm.run(module)));
}

// Run a post pipeline of cleanup and optimization passes.
static void postCanonicalizeIR(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createSymbolDCEPass());
  assert(!failed(pm.run(module)));
}

struct PhysicalToAie : public PhysicalToAieBase<PhysicalToAie> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    preCanonicalizeIR(module);

    transform::aie::AIELoweringPatternSets pattern_sets(module);
    auto pattern_set_list = pattern_sets.getPatternSets();

    for (auto &pattern_set : pattern_set_list) {
      mlir::ConversionTarget target(getContext());
      target.addLegalDialect<xilinx::AIE::AIEDialect>();

      mlir::RewritePatternSet patterns(&getContext());
      for (auto &pattern : pattern_set) {
        pattern->populatePatternSet(patterns);
        pattern->populateTarget(target);
      }

      if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                    std::move(patterns)))) {
        signalPassFailure();
        break;
      }
    }

    postCanonicalizeIR(module);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> xilinx::phy::createPhysicalToAie() {
  return std::make_unique<PhysicalToAie>();
}
