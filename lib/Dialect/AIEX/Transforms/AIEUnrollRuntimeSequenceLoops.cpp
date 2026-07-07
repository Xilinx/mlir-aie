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

// After loopUnrollFull, collect every aie.dma_bd inside `seq` that still
// carries a bd_id_window attribute and resolve each one to a concrete bd_id
// based on its copy index within its window.
//
// loopUnrollFull clones the loop body `tripCount` times and inserts the copies
// in program order. BDs in the same window are resolved round-robin,
// first-come-first-served in program order within their enclosing
// configure-task chain.
//
// The round-robin counter is keyed by (bd_id_window_group, window contents),
// not by contents alone: distinct rotation groups can carry identical windows
// (different tiles allocate ids independently; sequential loops on one tile
// reuse the freed pool), so the contents alone cannot tell them apart. The
// group id (stamped by --aie-assign-runtime-sequence-bd-ids) separates
// colliding groups, while the contents separate the per-descriptor windows
// within one multi-BD chain (which share a group id but rotate through
// disjoint id ranges).
static void resolveWindows(RuntimeSequenceOp seq) {
  struct Slot {
    int32_t group;
    SmallVector<int32_t> window;
    unsigned nextIdx;
  };
  llvm::SmallVector<Slot> slots;

  auto getWindowSlot = [&](int32_t group,
                           ArrayRef<int32_t> window) -> unsigned & {
    for (Slot &s : slots)
      if (s.group == group && ArrayRef<int32_t>(s.window) == window)
        return s.nextIdx;
    slots.push_back({group, SmallVector<int32_t>(window), 0});
    return slots.back().nextIdx;
  };

  seq.walk([&](DMABDOp bd) {
    auto windowAttr = bd.getBdIdWindow();
    if (!windowAttr || windowAttr->empty())
      return;
    ArrayRef<int32_t> window = *windowAttr;
    int32_t group = bd.getBdIdWindowGroup().value_or(0);
    unsigned &idx = getWindowSlot(group, window);
    bd.setBdId(static_cast<uint32_t>(window[idx % window.size()]));
    ++idx;
    bd.removeBdIdWindowAttr();
    bd.removeBdIdWindowGroupAttr();
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
            if (forOp.getStaticTripCount().has_value())
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
