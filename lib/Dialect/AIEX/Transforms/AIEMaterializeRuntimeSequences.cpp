//===- AIEMaterializeRuntimeSequence.cpp -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Targets/AIERT.h"
#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define DEBUG_TYPE "aie-materialize-runtime-sequence"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct RuntimeCallGraphCyclicityAnalysis {
  AnalysisManager &analysisManager;

  // if invalid, analysis failed and results should not be considered
  bool isValid = false;

  // Call graph is cyclic
  bool isCyclic = false;

  RuntimeCallGraphCyclicityAnalysis(Operation *op, AnalysisManager &am) 
  : analysisManager(am) 
  {
    RuntimeSequenceOp runtimeSequenceOp = llvm::dyn_cast<RuntimeSequenceOp>(op);
    if (!runtimeSequenceOp) {
      op->emitError("RuntimeCallGraphCyclicityAnalysis can only be called on aiex.runtime_sequence operations.");
      return;
    }
    llvm::SetVector<RuntimeSequenceOp> visited = {};
    llvm::SetVector<RunOp> todo;
    for (RunOp runOp : runtimeSequenceOp.getOps<RunOp>()) {
      todo.insert(runOp);
    }
    while(!todo.empty()) {
      RunOp curOp = todo.pop_back_val();
      if (RuntimeSequenceOp calleeRuntimeSequence = curOp.getCalleeRuntimeSequenceOp()) {
        if (visited.contains(calleeRuntimeSequence)) {
          continue;
        }
        visited.insert(calleeRuntimeSequence);
        for (RunOp runOp : calleeRuntimeSequence.getOps<RunOp>()) {
          todo.insert(runOp);
        }
      }
    }
    isCyclic = visited.contains(runtimeSequenceOp);
    isValid = true;
  }
};

// Turn aie.configure @device into aie.run %.. @configure
// TODO: add check that liveness of two aie.configures do not overlap
// (i.e., when we configure A, then configure B, cannot call runtime sequence of A after configuring B)
// TODO: add code to remove repeated @configure ops
struct InsertLoadPdiForConfigurePattern : RewritePattern {

  InsertLoadPdiForConfigurePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(ConfigureOp::getOperationName(), benefit, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, 
                  PatternRewriter &rewriter) const override {
    ConfigureOp configureOp = llvm::dyn_cast<ConfigureOp>(op);
    if (!configureOp) {
      return failure();
    }

    // LoadPDI resets the whole device, hence cannot do partial reconfiguration;
    // therefore, this only supports top-level configure ops
    if (!llvm::isa<RuntimeSequenceOp>(configureOp->getParentOp())) {
      return failure();
    }

    AIE::DeviceOp referencedDevice = configureOp.getReferencedDeviceOp();
    if (!referencedDevice) {
      configureOp.emitError("Referenced symbol is not a device");
      return failure();
    }

    Block &configureBlock = configureOp.getBody().front();
    rewriter.setInsertionPointToStart(&configureBlock);
    rewriter.create<AIEX::NpuLoadPdiOp>(
      configureOp.getLoc(), 
      FlatSymbolRefAttr::get(referencedDevice.getSymNameAttr())
    );

    return success();
  }
};


// Inlines the definitions of all symbols referenced in the given operation 
// at the current insertion point in the given rewriter, unless the symbol
// definition is in the "previouslyInlinedSymbolMap" map. While inlining,
// symbols will be renamed to have a unique name.
// The callback function is called for each symbol definition found. If it
// returns false, inlining is aborted and an error is emitted.
LogicalResult inlineReferencedSymbolDefinitions(
    PatternRewriter &rewriter, 
    Operation *op, 
    Operation *lookupFrom,
    IRMapping argMap,
    llvm::DenseMap<SymbolRefAttr, SymbolRefAttr> &previouslyInlinedSymbolMap,
    llvm::function_ref<bool(Operation *)> symbolDefValidator) {
  MLIRContext *ctx = op->getContext();
  for (NamedAttribute namedAttr : op->getAttrs()) {
    Attribute attr = namedAttr.getValue();
    auto newAttr = attr.replace(
      [&](SymbolRefAttr oldSymbolRef) {
        SymbolRefAttr newSymbolRef;
        if (!previouslyInlinedSymbolMap.count(oldSymbolRef)) {
          llvm::StringRef oldName = oldSymbolRef.getRootReference().getValue();
          std::string uniqueName = oldName.str();
          unsigned uniquingCounter = 0;
          while (SymbolTable::lookupNearestSymbolFrom(op, StringAttr::get(ctx, uniqueName))) {
            uniqueName = oldName.str() + "_" + std::to_string(uniquingCounter);
            uniquingCounter++;
          }
          newSymbolRef = SymbolRefAttr::get(ctx, uniqueName);
          previouslyInlinedSymbolMap[oldSymbolRef] = newSymbolRef;

          // Add the new symbol definition
          Operation *symbolDefOp = SymbolTable::lookupNearestSymbolFrom(lookupFrom, oldSymbolRef);
          if (!symbolDefValidator(symbolDefOp)) {
            return std::make_pair(newSymbolRef, WalkResult::interrupt());
          }
          Operation *clonedSymbolDefOp = rewriter.clone(*symbolDefOp, argMap);
          clonedSymbolDefOp->setAttr(SymbolTable::getSymbolAttrName(), StringAttr::get(ctx, uniqueName));
        } else {
          newSymbolRef = previouslyInlinedSymbolMap[oldSymbolRef];
        }
        return std::make_pair(newSymbolRef, WalkResult::advance());
      });
    if (!newAttr) {
      return failure();
    }
    op->setAttr(namedAttr.getName(), newAttr);
  }
  return success();
}

struct InlineRuntimeCallsPattern : RewritePattern {

  InlineRuntimeCallsPattern(MLIRContext *ctx)
      : RewritePattern(RunOp::getOperationName(), PatternBenefit(1),
                       ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // matching logic
    RunOp runOp = llvm::dyn_cast<RunOp>(op);
    if (!runOp) {
      return failure();
    }
    AIE::DeviceOp calleeDevice = runOp.getCalleeDeviceOp();
    RuntimeSequenceOp calleeRuntimeSequence = runOp.getCalleeRuntimeSequenceOp();
    if (!calleeDevice || !calleeRuntimeSequence) {
      return failure();
    }
    if (!calleeRuntimeSequence.getOps<RunOp>().empty()) {
      return failure();
    }

    // rewrite logic
    Region &calleeBody = calleeRuntimeSequence.getBody();
    AIE::DeviceOp callerDevice = op->getParentOfType<AIE::DeviceOp>();
    if (!callerDevice) {
      runOp.emitError() << "needs to be in a DeviceOp";
      return failure();
    }
    Region &callerDeviceBody = callerDevice.getBodyRegion();
    IRMapping argMap;
    ValueRange values = runOp.getArgs();
    for (unsigned i = 0, n = calleeBody.getNumArguments(); i < n; i++) {
      BlockArgument arg = calleeBody.getArgument(i);
      Value val = values[i];
      if(arg.getType() != val.getType()) {
        return runOp.emitOpError() << "argument " << i << " type mismatch: "
                                   << " expected " << arg.getType()
                                   << " but got " << val.getType();
      }
      argMap.map(arg, val);
    }
    llvm::DenseMap<SymbolRefAttr, SymbolRefAttr> previouslyInlinedSymbolMap;
    rewriter.setInsertionPoint(runOp);
    mlir::OpBuilder::InsertPoint clonedOpInsertionPoint = rewriter.saveInsertionPoint();
    mlir::Block &callerDeviceBodyFirstBlock = callerDeviceBody.front();
    mlir::OpBuilder::InsertPoint clonedSymbolInsertionPoint(&callerDeviceBodyFirstBlock, callerDeviceBodyFirstBlock.begin());
    for (Operation &op : calleeBody.getOps()) {
      rewriter.restoreInsertionPoint(clonedOpInsertionPoint);
      Operation *clonedOp = rewriter.clone(op, argMap);
      clonedOpInsertionPoint = rewriter.saveInsertionPoint();

      rewriter.restoreInsertionPoint(clonedSymbolInsertionPoint);
      if (failed(inlineReferencedSymbolDefinitions(
          rewriter, clonedOp, calleeRuntimeSequence.getOperation(), argMap, previouslyInlinedSymbolMap,
          [&](Operation *symbolDefOp) {
            if (!llvm::isa<AIE::ShimDMAAllocationOp>(symbolDefOp)) {
              runOp.emitError() << "referenced symbol '" 
                                << symbolDefOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()).getValue()
                                << "' must be a ShimDMAAllocationOp, but got: "
                                << symbolDefOp->getName().getStringRef();
              return false;
            }
            return true;
          }))) {
        return failure();
      }
      clonedSymbolInsertionPoint = rewriter.saveInsertionPoint();
    }

    rewriter.eraseOp(runOp);

    return success();
  }
};

struct AIEMaterializeRuntimeSequencesPass
    : AIEMaterializeRuntimeSequencesBase<AIEMaterializeRuntimeSequencesPass> {
  void runOnOperation() override {
    AIE::DeviceOp deviceOp = getOperation();

    // Turn aie.configure to aie.run @configure
    for (RuntimeSequenceOp runtimeSequenceOp : deviceOp.getOps<RuntimeSequenceOp>()) {
      AnalysisManager am = getAnalysisManager().nest(runtimeSequenceOp);
      RuntimeCallGraphCyclicityAnalysis cyclicity = am.getAnalysis<RuntimeCallGraphCyclicityAnalysis>();
      if (!cyclicity.isValid) {
        return signalPassFailure();
      }
      if (cyclicity.isCyclic) {
        runtimeSequenceOp.emitError("Runtime sequence contains a cycle");
        return signalPassFailure();
      }
      RewritePatternSet patterns(&getContext());
      patterns.insert<InsertLoadPdiForConfigurePattern>(&getContext());
      walkAndApplyPatterns(runtimeSequenceOp, std::move(patterns));
    }

    // Greedily inline all runtime sequences that can be inlined;
    // this will start with runtime sequences that do not call other runtime
    // sequences (leaves); once their callers inline them, the callers can
    // be inlined as well, and so on
    MLIRContext *ctx = &getContext();
    AIE::DeviceOp device = getOperation();
    GreedyRewriteConfig rewriter_config = GreedyRewriteConfig();
    rewriter_config.setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Disabled);

    RewritePatternSet patterns(ctx);
    patterns.insert<InlineRuntimeCallsPattern>(ctx);
    if (failed(applyPatternsGreedily(device, std::move(patterns), rewriter_config))) {
      signalPassFailure();
    }

    // Flatten the IR: hoist all operations inside aiex.configure to be direct
    // children of the runtime sequence, preserving order
    for (RuntimeSequenceOp runtimeSequenceOp : deviceOp.getOps<RuntimeSequenceOp>()) {
      SmallVector<ConfigureOp> configureOps;
      for (ConfigureOp configureOp : runtimeSequenceOp.getOps<ConfigureOp>()) {
        configureOps.push_back(configureOp);
      }
      
      IRRewriter rewriter(ctx);
      for (ConfigureOp configureOp : configureOps) {
        Block &configureBlock = configureOp.getBody().front();
        
        // Collect all operations in the configure block
        SmallVector<Operation *> opsToHoist;
        for (Operation &op : configureBlock) {
          opsToHoist.push_back(&op);
        }
        
        // Hoist operations to be right before the configure op
        rewriter.setInsertionPoint(configureOp);
        for (Operation *op : opsToHoist) {
          op->moveBefore(configureOp);
        }
        
        // Erase the now-empty configure op
        rewriter.eraseOp(configureOp);
      }
    }

  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEMaterializeRuntimeSequencesPass() {
  return std::make_unique<AIEMaterializeRuntimeSequencesPass>();
}
