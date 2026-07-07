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
// bodies. Running this before aie-assign-runtime-sequence-bd-ids leaves the
// allocator with straight-line IR (no back-edges), so a task freed in a later
// iteration than it is configured becomes several distinct configures whose BD
// ids ordinary liveness-based allocation recycles -- no rolled-loop rotation
// analysis is needed.
//
// Runtime-bound loops (non-constant trip count) are left in place; they will be
// handled by the dynamic EmitC path in a later phase.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEUNROLLRUNTIMESEQUENCELOOPS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

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
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEUnrollRuntimeSequenceLoopsPass() {
  return std::make_unique<AIEUnrollRuntimeSequenceLoopsPass>();
}
