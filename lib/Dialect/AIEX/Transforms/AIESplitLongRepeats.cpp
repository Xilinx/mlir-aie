//===- AIESplitLongRepeats.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A queue push carries its repeat count in a narrow field, but a task start may
// ask for more. This pass issues such a start as several starts of the same
// task, so that every start after it is exactly one push.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIESPLITLONGREPEATS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIESplitLongRepeatsPass
    : xilinx::AIEX::impl::AIESplitLongRepeatsBase<AIESplitLongRepeatsPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    uint32_t maxRepeat = device.getTargetModel().getMaxRepeatCount();
    int64_t queueDepth = device.getTargetModel().getDmaTaskQueueDepth();
    // A target without a repeat field has nothing to split into; its push
    // verifier reports the count instead.
    if (maxRepeat == 0)
      return;
    SmallVector<DMAStartTaskOp> starts;
    device.walk([&](DMAStartTaskOp start) { starts.push_back(start); });
    for (DMAStartTaskOp start : starts) {
      // A start's own count needs no configure. Otherwise the count is the
      // task's, read through any control flow that carries the task.
      std::optional<int64_t> rc;
      if (std::optional<uint32_t> own = start.getRepeatCount()) {
        rc = *own;
      } else if (Value ownVal = start.getRepeatCountVal()) {
        rc = getConstantIntValue(ownVal);
      } else {
        DMAConfigureTaskOp cfg = start.getTaskOp();
        if (!cfg)
          cfg = getUniqueReachableConfigure(start.getTask());
        if (!cfg)
          continue;
        rc = getConstantIntValue(start.getPushRepeatCount(cfg));
      }
      if (!rc || *rc <= maxRepeat)
        continue;
      int64_t pushes = (*rc + maxRepeat + 1) / (maxRepeat + 1);
      if (pushes > static_cast<int64_t>(maxPushes)) {
        start.emitOpError("repeat count ")
            << *rc << " needs " << pushes
            << " queue pushes, more than max-pushes (" << maxPushes.getValue()
            << ")";
        return signalPassFailure();
      }
      if (queueDepth > 0 && pushes > queueDepth)
        start.emitWarning("repeat count ")
            << *rc << " needs " << pushes << " queue pushes, more than the "
            << queueDepth
            << " the channel's queue holds, so the sequence waits for this "
               "channel to drain before going on. A transfer this one waits "
               "on must be started before it, or the wait never ends.";
      // Leading starts withhold the token, so an await on the task still
      // returns only after the last pass.
      OpBuilder b(start);
      int64_t runs = *rc + 1;
      for (; runs > maxRepeat + 1; runs -= maxRepeat + 1)
        DMAStartTaskOp::create(b, start.getLoc(), start.getTask(),
                               b.getI32IntegerAttr(maxRepeat),
                               /*repeat_count_val=*/nullptr,
                               /*no_token=*/b.getUnitAttr());
      start.setRepeatCountAttr(b.getI32IntegerAttr(runs - 1));
      start.getRepeatCountValMutable().clear();
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIESplitLongRepeatsPass() {
  return std::make_unique<AIESplitLongRepeatsPass>();
}
