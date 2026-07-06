//===- AIEUnrollRuntimeSequenceLoops.cpp ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Fully unrolls constant-bound scf.for loops inside aie.runtime_sequence
// bodies, then resolves each unrolled aie.dma_bd's bd_id from its pre-
// allocated rotating window.
//
// This pass must run AFTER aie-assign-runtime-sequence-bd-ids (which writes
// `bd_id_window` onto rotating BDs) and BEFORE aie-dma-tasks-to-npu (which
// rejects windowed BDs it cannot lower).
//
// After unrolling a depth-D ping-pong window of width W = D+1, each unrolled
// copy i gets bd_id = window[i % W] and the bd_id_window attribute is removed.
// BDs that are not part of a window (non-rotating, straight-line) are
// unaffected. Runtime-bound loops (non-constant trip count) are left alone;
// they will be handled by the dynamic EmitC path in a later phase.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEUNROLLRUNTIMESEQUENCELOOPS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

// Return the trip count of `forOp` as a compile-time constant, or nullopt if
// the bounds or step are not constants.
static std::optional<uint64_t> getConstantTripCount(scf::ForOp forOp) {
  auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb || !ub || !step)
    return std::nullopt;
  int64_t lbv = lb.value(), ubv = ub.value(), sv = step.value();
  if (sv <= 0 || ubv <= lbv)
    return 0;
  return static_cast<uint64_t>((ubv - lbv + sv - 1) / sv);
}

// After loopUnrollFull, collect every aie.dma_bd inside `seq` that still
// carries a bd_id_window attribute and resolve each one to a concrete bd_id
// based on its copy index within its window.
//
// loopUnrollFull clones the loop body `tripCount` times and inserts the copies
// in program order. BDs in the same window (same ordered set of physical IDs)
// are resolved round-robin, first-come-first-served in program order within
// their enclosing configure-task chain.
static void resolveWindows(RuntimeSequenceOp seq) {
  // Track per-window how many BDs have been assigned so far.
  // Key: the window contents (stable storage in windowStorage).
  llvm::SmallVector<SmallVector<int32_t>> windowStorage;
  llvm::SmallVector<unsigned> windowNextIdx;

  auto getWindowSlot = [&](ArrayRef<int32_t> window) -> unsigned & {
    for (auto [i, w] : llvm::enumerate(windowStorage))
      if (ArrayRef<int32_t>(w) == window)
        return windowNextIdx[i];
    windowStorage.push_back(SmallVector<int32_t>(window));
    windowNextIdx.push_back(0);
    return windowNextIdx.back();
  };

  seq.walk([&](DMABDOp bd) {
    auto windowAttr = bd.getBdIdWindow();
    if (!windowAttr || windowAttr->empty())
      return;
    ArrayRef<int32_t> window = *windowAttr;
    unsigned &idx = getWindowSlot(window);
    bd.setBdId(static_cast<uint32_t>(window[idx % window.size()]));
    ++idx;
    bd.removeBdIdWindowAttr();
  });
}

struct AIEUnrollRuntimeSequenceLoopsPass
    : xilinx::AIEX::impl::AIEUnrollRuntimeSequenceLoopsBase<
          AIEUnrollRuntimeSequenceLoopsPass> {

  void runOnOperation() override {
    DeviceOp device = getOperation();

    device.walk([&](RuntimeSequenceOp seq) {
      // Iterate to fixed point: after outer loops unroll, inner loops become
      // direct children of the sequence body and need another sweep.
      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<scf::ForOp> toUnroll;
        // Only look at immediate children of the sequence body — loopUnrollFull
        // handles inner nesting within each loop.
        for (Operation &op : seq.getBody().front()) {
          if (auto forOp = dyn_cast<scf::ForOp>(&op))
            if (getConstantTripCount(forOp).has_value())
              toUnroll.push_back(forOp);
        }
        for (scf::ForOp forOp : toUnroll) {
          if (succeeded(loopUnrollFull(forOp)))
            changed = true;
        }
      }

      // Resolve any remaining bd_id_window attributes to concrete ids.
      resolveWindows(seq);
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEUnrollRuntimeSequenceLoopsPass() {
  return std::make_unique<AIEUnrollRuntimeSequenceLoopsPass>();
}
