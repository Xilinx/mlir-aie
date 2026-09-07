//===- AIEObjectFifoUnroll.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOUNROLL
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-objectFifo-unroll"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// Statically known trip count of a loop, or nullopt if it cannot be computed.
static std::optional<int64_t> getStaticTripCount(scf::ForOp forOp) {
  if (forOp.getSingleLowerBound() && forOp.getSingleUpperBound() &&
      forOp.getSingleStep()) {
    if (std::optional<llvm::APInt> tc = forOp.getStaticTripCount()) {
      return tc->getSExtValue();
    }
  }
  return std::nullopt;
}

// Remove redundant AIE1 binary-lock acquires.
//
// The dynamic lowering acquires the whole sliding window ([counter, counter +
// acqNumber)) on every loop iteration; after unrolling and constant folding
// that expands to an explicit `Acquire` of each element's lock.
// This drops the needless re-acquire of locks already held.
//
// The analysis is block-local and conservatively assumes nothing is held on
// block entry. That keeps the very first acquire of each unrolled loop body
// (which legitimately re-establishes the window across the back-edge) while
// dropping the intra-body re-acquires of locks whose element is still held.
static void removeRedundantBinaryAcquires(DeviceOp device) {
  device.walk([&](Block *block) {
    llvm::DenseSet<Value> held;
    SmallVector<UseLockOp> toErase;
    for (Operation &op : *block) {
      auto useLock = dyn_cast<UseLockOp>(&op);
      if (!useLock) {
        continue;
      }
      Value lock = useLock.getLock();
      if (useLock.acquire()) {
        // A binary `Acquire` of a lock already in the held set is redundant.
        if (!held.insert(lock).second) {
          toErase.push_back(useLock);
        }
      } else if (useLock.release()) {
        held.erase(lock);
      }
    }
    for (UseLockOp op : toErase) {
      op.erase();
    }
  });
}

// Marks a loop already offered to the peeling trial below, so that neither the
// peeled loop nor the reverted copy is considered again.
static constexpr llvm::StringLiteral kPeelTriedAttrName = "aie.peel_tried";

// Number of `use_lock` ops in `op` whose lock value is not a compile-time
// constant. UseLockOp::getConstantValue() reports an error when it fails, so
// this probes the value directly.
static int64_t countRuntimeLockValues(Operation *op) {
  int64_t count = 0;
  op->walk([&](UseLockOp useLockOp) {
    if (!getConstantIntValue(useLockOp.getValue())) {
      count++;
    }
  });
  return count;
}

struct AIEObjectFifoUnrollPass
    : xilinx::AIE::impl::AIEObjectFifoUnrollBase<AIEObjectFifoUnrollPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
    registry.insert<scf::SCFDialect>();
  }

  /// Peel the first iteration off each loop whose lock values are still
  /// computed at run time, keeping a peel only where it makes them constant.
  ///
  /// Whether a peel pays off depends on foldings that cannot be predicted from
  /// the objectFifo access pattern alone (a conditional acquire may or may not
  /// collapse, for instance), so each candidate is peeled for real and the
  /// result measured. While the fold pipeline runs, the unpeeled core is held
  /// aside as a detached clone, so exactly one version of the loop is ever
  /// live; that clone is swapped back in when the peel bought nothing, and
  /// destroyed otherwise.
  ///
  /// The measurement is a per-core count, so attributing a change to a peel
  /// needs one peel per core per fold; the loop below drains the candidates one
  /// round at a time, peeling every core in parallel so a round costs a single
  /// run of the fold pipeline. Candidates are marked before they are tried and
  /// never unmarked while draining, so the number of rounds is bounded by the
  /// most objectFifo loops in any one core.
  LogicalResult peelObjectFifoLoops(DeviceOp device) {
    OpPassManager foldPipeline(DeviceOp::getOperationName());
    foldPipeline.addPass(mlir::createCanonicalizerPass());
    foldPipeline.addPass(mlir::createSCCPPass());
    foldPipeline.addPass(mlir::createCanonicalizerPass());

    IRRewriter rewriter(device.getContext());
    struct Trial {
      CoreOp core;
      Operation *unpeeled;
      int64_t runtimeLocksBefore;
    };

    while (true) {
      SmallVector<Trial> trials;
      for (CoreOp core : SmallVector<CoreOp>(device.getOps<CoreOp>())) {
        scf::ForOp target;
        // Post-order, so the innermost untried loop is taken first.
        core.walk([&](scf::ForOp forOp) {
          if (forOp->hasAttr(kPeelTriedAttrName) ||
              countRuntimeLockValues(forOp) == 0) {
            return WalkResult::advance();
          }
          // A single-iteration loop has nothing to peel off.
          std::optional<int64_t> trip = getStaticTripCount(forOp);
          if (trip && *trip <= 1) {
            return WalkResult::advance();
          }
          target = forOp;
          return WalkResult::interrupt();
        });
        if (!target) {
          continue;
        }
        // Marked before the core is cloned, so neither the peeled loop nor the
        // fallback copy is offered as a candidate again.
        target->setAttr(kPeelTriedAttrName, rewriter.getUnitAttr());

        Trial trial{core, core->clone(), countRuntimeLockValues(core)};
        scf::ForOp firstIteration;
        if (failed(scf::peelForLoopFirstIteration(rewriter, target,
                                                  firstIteration))) {
          trial.unpeeled->erase();
          continue;
        }
        trials.push_back(trial);
      }
      if (trials.empty()) {
        break;
      }

      if (failed(runPipeline(foldPipeline, device))) {
        for (Trial &trial : trials) {
          trial.unpeeled->erase();
        }
        return failure();
      }

      for (Trial &trial : trials) {
        if (countRuntimeLockValues(trial.core) < trial.runtimeLocksBefore) {
          trial.unpeeled->erase();
          continue;
        }
        rewriter.setInsertionPoint(trial.core);
        Operation *restored = rewriter.insert(trial.unpeeled);
        rewriter.replaceOp(trial.core, restored->getResults());
      }
    }

    device.walk(
        [&](scf::ForOp forOp) { forOp->removeAttr(kPeelTriedAttrName); });
    return success();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();

    bool peelFirstIteration = false;
    if (clPeelFirstIteration == "auto") {
      peelFirstIteration = !clDefaultDynamic;
    } else if (clPeelFirstIteration == "true") {
      peelFirstIteration = true;
    } else if (clPeelFirstIteration != "false") {
      device.emitOpError("invalid peel-first-iteration value '")
          << clPeelFirstIteration << R"('; expected "true", "false" or "auto")";
      return signalPassFailure();
    }

    for (auto coreOp : device.getOps<CoreOp>()) {
      // `default-dynamic` picks the lowering for cores that do not pin their
      // own choice; a core's `dynamic_objfifo_lowering` attribute overrides it
      // in either direction. Dynamic cores keep their loops rolled (drop the
      // hints, skip unrolling); their runtime bookkeeping is tidied by the
      // shared fold/cleanup pipeline below.
      if (coreOp.getDynamicObjfifoLowering().value_or(clDefaultDynamic)) {
        coreOp.walk([&](scf::ForOp forOp) {
          forOp->removeAttr(kObjectFifoUnrollHintAttrName);
        });
        continue;
      }
      SmallVector<scf::ForOp> loops;
      coreOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

      // Operation::walk uses post-order traversal by default, so a nested loop
      // is visited before its enclosing loop; iterating the list in order thus
      // processes the innermost loops first. Unrolling innermost loops first
      // avoids invalidating references to inner loops when an outer loop (which
      // duplicates its nested loops) is unrolled.
      for (scf::ForOp forOp : loops) {
        // Loops without the objectFifo unroll hint (the LCM of the accessed
        // fifo depths) carry no objectFifo access; factor 1 leaves them rolled.
        auto hint =
            forOp->getAttrOfType<IntegerAttr>(kObjectFifoUnrollHintAttrName);
        int64_t unrollFactor = hint ? hint.getInt() : 1;
        if (unrollFactor <= 1) {
          continue;
        }

        std::optional<int64_t> trip = getStaticTripCount(forOp);
        // When the loop performs fewer iterations than a full rotation of the
        // objectFifos, unroll it completely: every iteration must map to an
        // explicit buffer/lock slot.
        if (trip && *trip <= unrollFactor) {
          if (failed(mlir::loopUnrollFull(forOp))) {
            forOp.emitOpError()
                << "failed to fully unroll objectFifo loop (trip count "
                << *trip << ")";
            return signalPassFailure();
          }
          continue;
        }

        // Otherwise unroll by the rotation period. loopUnrollByFactor peels a
        // cleanup/epilogue loop for the remaining iterations when the trip
        // count is not an exact multiple of the factor.
        FailureOr<mlir::UnrolledLoopInfo> info = mlir::loopUnrollByFactor(
            forOp, static_cast<uint64_t>(unrollFactor));
        if (failed(info)) {
          forOp.emitOpError()
              << "failed to unroll objectFifo loop by factor " << unrollFactor;
          return signalPassFailure();
        }

        // The epilogue runs the remaining (< factor) iterations. Fully unroll
        // it as well so that each of those iterations maps to an explicit
        // buffer/lock rotation slot. This is best-effort: an epilogue with a
        // non-constant trip count cannot be fully unrolled and is left rolled.
        std::optional<scf::ForOp> epilogue =
            info->epilogueLoopOp; // NOLINT(bugprone-unchecked-optional-access)
        if (epilogue) {
          (void)mlir::loopUnrollFull(*epilogue);
        }
      }

      // Drop any lingering unroll hints so they do not leak into the output.
      coreOp.walk([&](scf::ForOp forOp) {
        forOp->removeAttr(kObjectFifoUnrollHintAttrName);
      });
    }

    // --aie-objectfifo-lower-cores promotes the buffer-selection and lock
    // bookkeeping counters to loop-carried SSA values. Once the loops have been
    // unrolled by their rotation period those counters become loop-invariant,
    // so every buffer selection (scf.index_switch) and lock value collapses to
    // a constant. Run that fold here as a scoped sub-pipeline: canonicalize
    // exposes the constants, SCCP propagates them across any remainder loop
    // that survives a partial unroll, and a final canonicalize deletes the
    // now-dead counter arithmetic and iter_args.
    OpPassManager foldPipeline(DeviceOp::getOperationName());
    foldPipeline.addPass(mlir::createCanonicalizerPass());
    foldPipeline.addPass(mlir::createSCCPPass());
    foldPipeline.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(foldPipeline, device))) {
      return signalPassFailure();
    }

    if (peelFirstIteration && failed(peelObjectFifoLoops(device))) {
      return signalPassFailure();
    }

    // With the window acquires now folded to concrete per-lock `Acquire` ops,
    // drop the ones that re-acquire an already-held AIE1 binary lock, then run
    // a final canonicalize to delete the constants they leave dead.
    removeRedundantBinaryAcquires(device);
    OpPassManager cleanupPipeline(DeviceOp::getOperationName());
    cleanupPipeline.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(cleanupPipeline, device))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
xilinx::AIE::createAIEObjectFifoUnrollPass() {
  return std::make_unique<AIEObjectFifoUnrollPass>();
}
