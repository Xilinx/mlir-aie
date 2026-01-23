//===- AIEMaterializeRuntimeSequences.cpp -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Targets/AIERT.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

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
      : analysisManager(am) {
    AIE::RuntimeSequenceOp runtimeSequenceOp =
        llvm::dyn_cast<AIE::RuntimeSequenceOp>(op);
    if (!runtimeSequenceOp) {
      op->emitError("RuntimeCallGraphCyclicityAnalysis can only be called on "
                    "aiex.runtime_sequence operations.");
      return;
    }

    // Use DFS with a stack to detect cycles
    // A cycle exists if we encounter a sequence already on the current path
    llvm::DenseSet<AIE::RuntimeSequenceOp> callStack;
    llvm::DenseSet<AIE::RuntimeSequenceOp> visited;

    std::function<bool(AIE::RuntimeSequenceOp)> hasCycle =
        [&](AIE::RuntimeSequenceOp seq) -> bool {
      if (callStack.contains(seq)) {
        return true; // Found a cycle
      }
      if (visited.contains(seq)) {
        return false; // Already checked this sequence
      }

      callStack.insert(seq);
      visited.insert(seq);

      // Check all sequences called by this one
      bool foundCycle = false;
      seq.walk([&](RunOp runOp) {
        if (AIE::RuntimeSequenceOp callee =
                runOp.getCalleeRuntimeSequenceOp()) {
          if (hasCycle(callee)) {
            foundCycle = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

      callStack.erase(seq);
      return foundCycle;
    };

    if (hasCycle(runtimeSequenceOp)) {
      isCyclic = true;
      isValid = true;
      return;
    }
    isCyclic = false;
    isValid = true;
  }
};

// Turn aie.configure @device into aie.run %.. @configure
// TODO: add check that liveness of two aie.configures do not overlap
// (i.e., when we configure A, then configure B, cannot call runtime sequence of
// A after configuring B)
// TODO: add code to remove repeated @configure ops
struct InsertLoadPdiForConfigurePattern : RewritePattern {

  InsertLoadPdiForConfigurePattern(MLIRContext *context,
                                   PatternBenefit benefit = 1)
      : RewritePattern(ConfigureOp::getOperationName(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ConfigureOp configureOp = llvm::dyn_cast<ConfigureOp>(op);
    if (!configureOp) {
      return failure();
    }

    // LoadPDI resets the whole device, hence cannot do partial reconfiguration;
    // therefore, this only supports top-level configure ops
    if (!llvm::isa<AIE::RuntimeSequenceOp>(configureOp->getParentOp())) {
      return failure();
    }

    AIE::DeviceOp referencedDevice = configureOp.getReferencedDeviceOp();
    if (!referencedDevice) {
      configureOp.emitError("Referenced symbol is not a device");
      return failure();
    }

    Block *configureBlock;
    if (configureOp.getBody().empty()) {
      configureBlock = rewriter.createBlock(&configureOp.getBody());
    } else {
      configureBlock = &configureOp.getBody().front();
    }
    rewriter.setInsertionPointToStart(configureBlock);
    AIEX::NpuLoadPdiOp::create(
        rewriter, configureOp.getLoc(),
        FlatSymbolRefAttr::get(referencedDevice.getSymNameAttr()));

    return success();
  }
};

// Collects all external SSA values referenced by an operation (and its nested
// operations).
// 1. Collects SSA values from the operation's operands.
// 2. Recursively walks through all operations in the operation's regions.
// 3. For each nested operation, collects SSA values from its operands.
// 4. Skips values that are already in argMap or defined within the operation.
// 5. For memref.subview operations, traces to the root block argument
static void
collectReferencedSSAValues(Operation *op, const IRMapping &argMap,
                           llvm::SetVector<Value> &referencedValues) {

  auto processValue = [&](Value operand) {
    if (argMap.contains(operand)) {
      return;
    }

    // If this is a subview, trace to the root block argument
    if (auto traceResult = traceSubviewToBlockArgument(operand)) {
      // Check if the root argument is already mapped
      if (!argMap.contains(traceResult->rootArg)) {
        referencedValues.insert(traceResult->rootArg);
      }
      return;
    }

    // Not a subview chain leading to block arg, add as-is
    referencedValues.insert(operand);
  };

  // Collect SSA values from the operation's direct operands.
  for (Value operand : op->getOperands()) {
    processValue(operand);
  }

  // Recursively collect SSA values from nested operations in all regions.
  for (Region &region : op->getRegions()) {
    region.walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (argMap.contains(operand)) {
          return;
        }

        // Check if defined within the parent operation
        Operation *defOp = operand.getDefiningOp();
        if (defOp && op->isProperAncestor(defOp)) {
          return;
        }

        processValue(operand);
      }
    });
  }
}

// Copies SSA value definitions into the caller device.
// Currently, only `aie.tile` operations are supported.
// Updates argMap to map old values to new/existing values.
static LogicalResult
copyReferencedSSAValues(PatternRewriter &rewriter,
                        const llvm::SetVector<Value> &referencedValues,
                        AIE::DeviceOp callerDevice, IRMapping &argMap,
                        mlir::OpBuilder::InsertPoint &clonedSSAInsertionPoint,
                        Operation *errorReportOp) {

  for (Value referencedValue : referencedValues) {
    Operation *definingOp = referencedValue.getDefiningOp();
    if (!definingOp) {
      return errorReportOp->emitError()
             << "Referenced value is not defined by an operation";
    }

    if (auto tileOp = llvm::dyn_cast<AIE::TileOp>(definingOp)) {
      int col = tileOp.getCol();
      int row = tileOp.getRow();

      // Check if a tile with matching col/row already exists in the caller
      // device
      AIE::TileOp existingTile = nullptr;
      for (AIE::TileOp tile : callerDevice.getOps<AIE::TileOp>()) {
        if (tile.getCol() == col && tile.getRow() == row) {
          existingTile = tile;
          break;
        }
      }

      if (existingTile) {
        // Verify that all attributes match
        if (tileOp->getAttrDictionary() != existingTile->getAttrDictionary()) {
          // Filter out result type attributes and symbol attributes for
          // comparison
          auto filterAttrs = [](DictionaryAttr dict) -> DictionaryAttr {
            SmallVector<NamedAttribute> filteredAttrs;
            for (auto namedAttr : dict) {
              StringRef name = namedAttr.getName().getValue();
              if (name != "col" && name != "row") {
                filteredAttrs.push_back(namedAttr);
              }
            }
            return DictionaryAttr::get(dict.getContext(), filteredAttrs);
          };

          DictionaryAttr tileAttrs = filterAttrs(tileOp->getAttrDictionary());
          DictionaryAttr existingAttrs =
              filterAttrs(existingTile->getAttrDictionary());

          if (tileAttrs != existingAttrs) {
            return errorReportOp->emitError()
                   << "aie.tile(" << col << ", " << row
                   << ") already exists in the device with different "
                      "attributes";
          }
        }

        // Use the existing tile
        rewriter.restoreInsertionPoint(clonedSSAInsertionPoint);
        if (!existingTile->isBeforeInBlock(&*rewriter.getInsertionPoint())) {
          rewriter.moveOpBefore(existingTile,
                                clonedSSAInsertionPoint.getBlock(),
                                clonedSSAInsertionPoint.getPoint());
          rewriter.setInsertionPointAfter(existingTile);
          clonedSSAInsertionPoint = rewriter.saveInsertionPoint();
        }
        argMap.map(referencedValue, existingTile.getResult());
      } else {
        // Clone the tile operation into the caller device
        rewriter.restoreInsertionPoint(clonedSSAInsertionPoint);
        Operation *clonedTile = rewriter.clone(*tileOp);
        argMap.map(referencedValue, clonedTile->getResult(0));
        clonedSSAInsertionPoint = rewriter.saveInsertionPoint();
      }

    } else if(auto lockOp = llvm::dyn_cast<AIE::LockOp>(definingOp)) {

      // First ensure the tile referenced by the lock is available
      Value lockTile = lockOp.getTile();
      if (!argMap.contains(lockTile)) {
        Operation *lockTileDefOp = lockTile.getDefiningOp();
        if (auto lockTileOp = llvm::dyn_cast<AIE::TileOp>(lockTileDefOp)) {
          int col = lockTileOp.getCol();
          int row = lockTileOp.getRow();

          // Check if a tile with matching col/row already exists in the caller device
          AIE::TileOp existingTile = nullptr;
          for (AIE::TileOp tile : callerDevice.getOps<AIE::TileOp>()) {
            if (tile.getCol() == col && tile.getRow() == row) {
              existingTile = tile;
              break;
            }
          }

          if (existingTile) {
            // Use the existing tile - ensure it's before our insertion point
            if (!existingTile->isBeforeInBlock(&*rewriter.getInsertionPoint())) {
              rewriter.restoreInsertionPoint(clonedSSAInsertionPoint);
              rewriter.moveOpBefore(existingTile,
                                    clonedSSAInsertionPoint.getBlock(),
                                    clonedSSAInsertionPoint.getPoint());
              rewriter.setInsertionPointAfter(existingTile);
              clonedSSAInsertionPoint = rewriter.saveInsertionPoint();
            }
            argMap.map(lockTile, existingTile.getResult());
          } else {
            // Clone the tile operation into the caller device
            rewriter.restoreInsertionPoint(clonedSSAInsertionPoint);
            Operation *clonedTile = rewriter.clone(*lockTileOp);
            argMap.map(lockTile, clonedTile->getResult(0));
            clonedSSAInsertionPoint = rewriter.saveInsertionPoint();
          }
        } else {
          return errorReportOp->emitError()
                 << "Lock references a tile that is not defined by aie.tile operation";
        }
      }

      // Now clone the lock operation into the caller device
      rewriter.restoreInsertionPoint(clonedSSAInsertionPoint);
      Operation *clonedLock = rewriter.clone(*lockOp, argMap);
      argMap.map(referencedValue, clonedLock->getResult(0));
      clonedSSAInsertionPoint = rewriter.saveInsertionPoint();

    } else {
      return errorReportOp->emitError()
             << "Referenced SSA value defined by unsupported operation type: "
             << definingOp->getName().getStringRef()
             << ". Currently only aie.tile operations are supported.";
    }
  }

  return success();
}

// Inlines the definitions of all symbols referenced in the given operation
// at the current insertion point in the given rewriter, unless the symbol
// definition is in the "previouslyInlinedSymbolMap" map. While inlining,
// symbols will be renamed to have a unique name.
// Also copies in SSA values referenced by the inlined symbol definitions.
static LogicalResult inlineReferencedSymbolDefinitions(
    PatternRewriter &rewriter, Operation *op, Operation *lookupFrom,
    IRMapping argMap,
    llvm::DenseMap<SymbolRefAttr, SymbolRefAttr> &previouslyInlinedSymbolMap,
    AIE::DeviceOp callerDevice,
    mlir::OpBuilder::InsertPoint &clonedDefOpsInsertionPoint) {
  MLIRContext *ctx = op->getContext();
  for (NamedAttribute namedAttr : op->getAttrs()) {
    Attribute attr = namedAttr.getValue();
    auto newAttr = attr.replace([&](SymbolRefAttr oldSymbolRef) {
      SymbolRefAttr newSymbolRef;
      if (!previouslyInlinedSymbolMap.count(oldSymbolRef)) {
        llvm::StringRef oldName = oldSymbolRef.getRootReference().getValue();
        std::string uniqueName = oldName.str();
        unsigned uniquingCounter = 0;
        while (SymbolTable::lookupNearestSymbolFrom(
            op, StringAttr::get(ctx, uniqueName))) {
          uniqueName = oldName.str() + "_" + std::to_string(uniquingCounter);
          uniquingCounter++;
        }
        newSymbolRef = SymbolRefAttr::get(ctx, uniqueName);
        previouslyInlinedSymbolMap[oldSymbolRef] = newSymbolRef;

        // Add the new symbol definition
        Operation *symbolDefOp =
            SymbolTable::lookupNearestSymbolFrom(lookupFrom, oldSymbolRef);
        if (!symbolDefOp) {
          return std::make_pair(newSymbolRef, WalkResult::interrupt());
        }

        // Collect SSA values referenced by the symbol definition operation
        llvm::SetVector<Value> symbolReferencedValues;
        collectReferencedSSAValues(symbolDefOp, argMap, symbolReferencedValues);

        // Copy SSA values referenced by the symbol definition
        // This updates clonedDefOpsInsertionPoint to be after the copied SSA values
        if (failed(copyReferencedSSAValues(rewriter, symbolReferencedValues,
                                           callerDevice, argMap,
                                           clonedDefOpsInsertionPoint, op))) {
          return std::make_pair(newSymbolRef, WalkResult::interrupt());
        }

        // Insert the cloned symbol at the device level, after its SSA dependencies
        rewriter.restoreInsertionPoint(clonedDefOpsInsertionPoint);
        Operation *clonedSymbolDefOp = rewriter.clone(*symbolDefOp, argMap);
        clonedSymbolDefOp->setAttr(SymbolTable::getSymbolAttrName(),
                                   StringAttr::get(ctx, uniqueName));
        clonedDefOpsInsertionPoint = rewriter.saveInsertionPoint();
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
      : RewritePattern(RunOp::getOperationName(), PatternBenefit(1), ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // matching logic
    RunOp runOp = llvm::dyn_cast<RunOp>(op);
    if (!runOp) {
      return failure();
    }
    AIE::DeviceOp calleeDevice = runOp.getCalleeDeviceOp();
    AIE::RuntimeSequenceOp calleeRuntimeSequence =
        runOp.getCalleeRuntimeSequenceOp();
    if (!calleeDevice || !calleeRuntimeSequence) {
      return failure();
    }
    if (!calleeRuntimeSequence.getOps<RunOp>().empty()) {
      return failure();
    }

    // rewrite logic

    // Get caller and callee bodies. The callee body will be inlined into the
    // caller body at the point of the RunOp.
    Region &calleeBody = calleeRuntimeSequence.getBody();
    AIE::DeviceOp callerDevice = op->getParentOfType<AIE::DeviceOp>();
    if (!callerDevice) {
      runOp.emitError() << "needs to be in a DeviceOp";
      return failure();
    }
    Region &callerDeviceBody = callerDevice.getBodyRegion();

    // The argMap maps callee arguments to caller SSA values.
    IRMapping argMap;
    ValueRange values = runOp.getArgs();
    for (unsigned i = 0, n = calleeBody.getNumArguments(); i < n; i++) {
      BlockArgument arg = calleeBody.getArgument(i);
      Value val = values[i];
      argMap.map(arg, val);
    }

    // The callee body may reference SSA values and symbols that are defined
    // in the callee device (outside the callee runtime sequence). We will
    // inline a supported set of these and error otherwise.

    // Collect SSA values referenced in the callee not defined by the callee and
    // not in the argMap.
    llvm::SetVector<Value> referencedValues;
    for (Operation &op : calleeBody.getOps()) {
      collectReferencedSSAValues(&op, argMap, referencedValues);
    }
    llvm::SetVector<Value> filteredValues;
    for (Value val : referencedValues) {
      if (val.getParentRegion() != &calleeBody) {
        filteredValues.insert(val);
      }
    }
    referencedValues = std::move(filteredValues);

    // Copy the operations that define these SSA values into the caller device
    mlir::Block &callerDeviceBodyFirstBlock = callerDeviceBody.front();
    mlir::OpBuilder::InsertPoint clonedDefOpsInsertPoint(
        &callerDeviceBodyFirstBlock, callerDeviceBodyFirstBlock.begin());
    if (failed(copyReferencedSSAValues(rewriter, referencedValues, callerDevice,
                                       argMap, clonedDefOpsInsertPoint,
                                       runOp))) {
      return failure();
    }

    // Now, also inline symbol definitions referenced in the callee body;
    // this may pull in additional SSA values referenced by the symbol
    // definitions.
    rewriter.setInsertionPoint(runOp);
    mlir::OpBuilder::InsertPoint clonedOpInsertionPoint =
        rewriter.saveInsertionPoint();
    llvm::DenseMap<SymbolRefAttr, SymbolRefAttr> previouslyInlinedSymbolMap;
    for (Operation &op : calleeBody.getOps()) {
      rewriter.restoreInsertionPoint(clonedOpInsertionPoint);
      Operation *clonedOp = rewriter.clone(op, argMap);
      clonedOpInsertionPoint = rewriter.saveInsertionPoint();

      if (failed(inlineReferencedSymbolDefinitions(
              rewriter, clonedOp, calleeRuntimeSequence.getOperation(), argMap,
              previouslyInlinedSymbolMap, callerDevice,
              clonedDefOpsInsertPoint))) {
        return failure();
      }
    }

    // The aiex.run op has been inlined; erase it.
    rewriter.eraseOp(runOp);

    return success();
  }
};

struct AIEMaterializeRuntimeSequencesPass
    : AIEMaterializeRuntimeSequencesBase<AIEMaterializeRuntimeSequencesPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Process each device in the module
    for (AIE::DeviceOp deviceOp : moduleOp.getOps<AIE::DeviceOp>()) {

      // Verify all runtime sequences before materialization
      for (AIE::RuntimeSequenceOp runtimeSequenceOp :
           deviceOp.getOps<AIE::RuntimeSequenceOp>()) {
        if (failed(runtimeSequenceOp.verifyBeforeMaterialization())) {
          return signalPassFailure();
        }
      }

      // Check for cycles in runtime sequence calls
      for (AIE::RuntimeSequenceOp runtimeSequenceOp :
           deviceOp.getOps<AIE::RuntimeSequenceOp>()) {
        AnalysisManager am =
            getAnalysisManager().nest(deviceOp).nest(runtimeSequenceOp);
        RuntimeCallGraphCyclicityAnalysis cyclicity =
            am.getAnalysis<RuntimeCallGraphCyclicityAnalysis>();
        if (!cyclicity.isValid) {
          return signalPassFailure();
        }
        if (cyclicity.isCyclic) {
          runtimeSequenceOp.emitError(
              "Runtime sequence call graph contains a cycle");
          return signalPassFailure();
        }
      }

      // Greedily inline all runtime sequences that can be inlined;
      // this will start with runtime sequences that do not call other runtime
      // sequences (leaves); once their callers inline them, the callers can
      // be inlined as well, and so on
      MLIRContext *ctx = &getContext();
      GreedyRewriteConfig rewriter_config = GreedyRewriteConfig();
      rewriter_config.setRegionSimplificationLevel(
          GreedySimplifyRegionLevel::Disabled);

      RewritePatternSet patterns_0(ctx);
      patterns_0.insert<InlineRuntimeCallsPattern>(ctx);
      if (failed(applyPatternsGreedily(deviceOp, std::move(patterns_0),
                                       rewriter_config))) {
        return signalPassFailure();
      }

      // Insert LoadPDI ops for each aiex.configure op
      for (AIE::RuntimeSequenceOp runtimeSequenceOp :
           deviceOp.getOps<AIE::RuntimeSequenceOp>()) {
        RewritePatternSet patterns_1(ctx);
        patterns_1.insert<InsertLoadPdiForConfigurePattern>(ctx);
        walkAndApplyPatterns(runtimeSequenceOp, std::move(patterns_1));
      }

      // Flatten the IR: hoist all operations inside aiex.configure to be direct
      // children of the runtime sequence, preserving order
      for (AIE::RuntimeSequenceOp runtimeSequenceOp :
           deviceOp.getOps<AIE::RuntimeSequenceOp>()) {
        SmallVector<ConfigureOp> configureOps;

        for (ConfigureOp configureOp :
             runtimeSequenceOp.getOps<ConfigureOp>()) {
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

    } // end for each device
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
AIEX::createAIEMaterializeRuntimeSequencesPass() {
  return std::make_unique<AIEMaterializeRuntimeSequencesPass>();
}
