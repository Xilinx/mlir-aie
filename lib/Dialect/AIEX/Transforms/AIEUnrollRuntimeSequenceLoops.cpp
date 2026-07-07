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
      // Collect constant-trip loops in post-order (innermost first), then
      // unroll in that order. loopUnrollFull unrolls only the given loop,
      // cloning any nested loops as-is -- so nested loops need their own calls.
      // Post-order guarantees an inner loop is unrolled (and gone) before its
      // enclosing loop, so every collected handle is still valid when reached
      // and no re-walk is needed. Reaches loops anywhere in the sequence,
      // including inside scf.if arms.
      SmallVector<scf::ForOp> loops;
      seq.walk([&](scf::ForOp forOp) {
        if (forOp.getStaticTripCount().has_value())
          loops.push_back(forOp);
      });
      for (scf::ForOp forOp : loops)
        (void)loopUnrollFull(forOp);
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEUnrollRuntimeSequenceLoopsPass() {
  return std::make_unique<AIEUnrollRuntimeSequenceLoopsPass>();
}
