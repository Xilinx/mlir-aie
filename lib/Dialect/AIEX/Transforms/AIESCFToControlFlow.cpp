//===- AIESCFToControlFlow.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Lower scf to cf everywhere in the module EXCEPT inside aie.runtime_sequence
// bodies.
//
// A runtime sequence is NoTerminator and is translated to a flat NPU
// instruction stream with no concept of branches. Its loops are lowered by the
// dedicated runtime-sequence path (unrolling for compile-time-constant trip
// counts; the dynamic EmitC path otherwise), not by generic scf->cf. Applying
// convert-scf-to-cf to a runtime sequence would (1) break the NoTerminator
// block invariant -- the ops after a lowered loop land in a terminator-less
// block -- and (2) emit cf.br / cf.cond_br that the flat NPU emitter silently
// drops, a miscompile. So we mark scf inside a runtime sequence as legal
// (leave it alone) and convert everything else.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIESCFTOCONTROLFLOW
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIESCFToControlFlowPass
    : xilinx::AIEX::impl::AIESCFToControlFlowBase<AIESCFToControlFlowPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    // Everything is legal by default; only scf outside a runtime sequence is
    // illegal. (Matching upstream convert-scf-to-cf, which marks every other
    // op legal so the structural block-splitting can relocate them freely.)
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addLegalDialect<cf::ControlFlowDialect>();
    // scf is illegal (must be converted) UNLESS it lives inside a runtime
    // sequence, whose control flow has its own lowering path.
    target.addDynamicallyLegalDialect<scf::SCFDialect>([](Operation *op) {
      return op->getParentOfType<AIE::RuntimeSequenceOp>() != nullptr;
    });

    RewritePatternSet patterns(&getContext());
    populateSCFToControlFlowConversionPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIESCFToControlFlowPass() {
  return std::make_unique<AIESCFToControlFlowPass>();
}
