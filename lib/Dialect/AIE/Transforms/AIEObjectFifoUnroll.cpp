//===- AIEObjectFifoUnroll.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
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

struct AIEObjectFifoUnrollPass
    : xilinx::AIE::impl::AIEObjectFifoUnrollBase<AIEObjectFifoUnrollPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();

    // Global dynamic lowering: leave every loop rolled and just drop the hints.
    if (clDynamicObjectFifos) {
      device.walk([&](scf::ForOp forOp) {
        forOp->removeAttr(kObjectFifoUnrollHintAttrName);
      });
      return;
    }

    for (auto coreOp : device.getOps<CoreOp>()) {
      // A core can opt into the dynamic form via its `dynamic_objfifo_lowering`
      // attribute; leave its loops rolled (drop the hints, skip unrolling).
      if (coreOp.getDynamicObjfifoLowering().value_or(false)) {
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
        if (info->epilogueLoopOp) {
          (void)mlir::loopUnrollFull(*info->epilogueLoopOp);
        }
      }

      // Drop any lingering unroll hints so they do not leak into the output.
      coreOp.walk([&](scf::ForOp forOp) {
        forOp->removeAttr(kObjectFifoUnrollHintAttrName);
      });
    }

    // AIEObjectFifoStatefulTransform promotes the buffer-selection and lock
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
