//===- AIETargetCppTxn.cpp - Generate C++ TXN builder ----------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aie-translate target that lowers an aie.runtime_sequence's npu transaction
// ops into a standalone C++ function (via convert-aiex-to-emitc +
// translateToCpp). The input is expected to be already lowered to npu ops, the
// same precondition as the aie-npu-to-binary target; this is its
// runtime-parameterizable C++ counterpart.
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Conversion/AIEXToEmitC/AIEXToEmitC.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

LogicalResult xilinx::AIE::AIETranslateNpuToCpp(ModuleOp module,
                                                raw_ostream &output,
                                                bool foldDDRAddrOffset,
                                                bool emitDispatchShim) {
  if (failed(inlineWriteConfigs(module)))
    return failure();

  // A staged sequence repeats the same guard arithmetic at every use of a
  // scalar (each tap re-derives and re-checks its shape). CSE merges the
  // duplicated arithmetic and folding drops the guards that became constant.
  // Scoped to the sequences: the rest of the device is not this target's to
  // rewrite.
  MLIRContext *ctx = module.getContext();
  RewritePatternSet patterns(ctx);
  cf::AssertOp::getCanonicalizationPatterns(patterns, ctx);
  FrozenRewritePatternSet frozen(std::move(patterns));
  IRRewriter rewriter(ctx);
  DominanceInfo domInfo;
  module.walk([&](AIE::RuntimeSequenceOp seq) {
    eliminateCommonSubExpressions(rewriter, domInfo, seq.getBody());
    SmallVector<Operation *> ops;
    seq.getBody().walk([&](Operation *op) { ops.push_back(op); });
    (void)applyOpPatternsGreedily(ops, frozen);
  });

  PassManager pm(ctx);
  pm.addPass(xilinx::createConvertAIEXToEmitCPass(foldDDRAddrOffset,
                                                  emitDispatchShim));
  if (failed(pm.run(module)))
    return failure();
  return emitc::translateToCpp(module, output);
}
