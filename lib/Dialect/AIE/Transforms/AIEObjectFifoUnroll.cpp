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

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOUNROLL
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-objectFifo-unroll"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// The unroll factor for a loop is the value of the `aie.unroll_hint` attribute
// attached by AIEObjectFifoStatefulTransform (the least common multiple of the
// depths of the objectFifos accessed within the loop). Loops without the hint
// carry no objectFifo access and are not unrolled (factor 1).
static int64_t unrollFactorForLoop(scf::ForOp forOp) {
  if (auto hint =
          forOp->getAttrOfType<IntegerAttr>(kObjectFifoUnrollHintAttrName))
    return hint.getInt();
  return 1;
}

// Statically known trip count of a loop, or nullopt if it cannot be computed.
static std::optional<int64_t> staticTripCount(scf::ForOp forOp) {
  if (forOp.getSingleLowerBound() && forOp.getSingleUpperBound() &&
      forOp.getSingleStep())
    if (std::optional<llvm::APInt> tc = forOp.getStaticTripCount())
      return tc->getSExtValue();
  return std::nullopt;
}

// True if the loop carries an objectFifo access, i.e. it was annotated with the
// unroll hint by the objectFifo stateful transform.
static bool loopHasObjectFifoOp(scf::ForOp forOp) {
  return forOp->hasAttr(kObjectFifoUnrollHintAttrName);
}

struct AIEObjectFifoUnrollPass
    : xilinx::AIE::impl::AIEObjectFifoUnrollBase<AIEObjectFifoUnrollPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();

    for (auto coreOp : device.getOps<CoreOp>()) {
      // Collect every scf.for loop that carries an objectFifo access (i.e. was
      // annotated with the unroll hint). Ancestor loops are annotated too, so
      // they are naturally included.
      SmallVector<scf::ForOp> loops;
      coreOp.walk([&](scf::ForOp forOp) {
        if (loopHasObjectFifoOp(forOp))
          loops.push_back(forOp);
      });

      // Operation::walk uses post-order traversal by default, so a nested loop
      // is visited before its enclosing loop; iterating the list in order thus
      // processes the innermost loops first. Unrolling innermost loops first
      // avoids invalidating references to inner loops when an outer loop (which
      // duplicates its nested loops) is unrolled.
      for (scf::ForOp forOp : loops) {
        int64_t unrollFactor = unrollFactorForLoop(forOp);
        if (unrollFactor <= 1)
          continue;

        std::optional<int64_t> trip = staticTripCount(forOp);
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
          scf::ForOp epilogue = *info->epilogueLoopOp;
          if (loopHasObjectFifoOp(epilogue))
            (void)mlir::loopUnrollFull(epilogue);
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
    // a constant. Run that fold here as a scoped sub-pipeline instead of
    // relying on the caller to chain the passes: canonicalize exposes the
    // constants, SCCP propagates them across any remainder loop that survives a
    // partial unroll, and a final canonicalize deletes the now-dead counter
    // arithmetic and iter_args.
    OpPassManager foldPipeline(DeviceOp::getOperationName());
    foldPipeline.addPass(mlir::createCanonicalizerPass());
    foldPipeline.addPass(mlir::createSCCPPass());
    foldPipeline.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(foldPipeline, device)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
xilinx::AIE::createAIEObjectFifoUnrollPass() {
  return std::make_unique<AIEObjectFifoUnrollPass>();
}
