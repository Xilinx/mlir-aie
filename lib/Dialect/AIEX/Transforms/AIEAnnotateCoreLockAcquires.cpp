//===- AIEAnnotateCoreLockAcquires.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEANNOTATECORELOCKACQUIRES
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct AIEAnnotateCoreLockAcquiresPass
    : xilinx::AIEX::impl::AIEAnnotateCoreLockAcquiresBase<
          AIEAnnotateCoreLockAcquiresPass> {
  void runOnOperation() override {
    WalkResult walk = getOperation().walk([&](CoreOp core) {
      // A core given as an ELF has no body to read; one annotated before keeps
      // its record.
      if (core.getElfFileAttr())
        return WalkResult::advance();
      std::optional<ArrayAttr> acquires = lockAcquires(core);
      if (!acquires)
        return WalkResult::interrupt();
      core->setAttr("aiex.lock_acquires", *acquires);
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEAnnotateCoreLockAcquiresPass() {
  return std::make_unique<AIEAnnotateCoreLockAcquiresPass>();
}
