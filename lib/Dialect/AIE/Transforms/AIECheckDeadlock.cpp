//===- AIECheckDeadlock.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEDeadlockModel.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/FormatVariadic.h"

#include <set>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIECHECKDEADLOCK
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

struct AIECheckDeadlockPass
    : public xilinx::AIE::impl::AIECheckDeadlockBase<AIECheckDeadlockPass> {
  using Base::Base;

  std::set<TileDMAChannel> waitedAnywhere;

  void runOnOperation() override {
    DeviceOp device = getOperation();
    SmallVector<RuntimeSequenceOp> sequences(
        device.getOps<RuntimeSequenceOp>());
    if (sequences.empty())
      sequences.push_back(nullptr);
    for (RuntimeSequenceOp sequence : sequences) {
      if (!sequence)
        continue;
      DeadlockModel model(device, sequence);
      waitedAnywhere.insert(model.waitedChannels().begin(),
                            model.waitedChannels().end());
    }
    bool failedAny = false;
    for (RuntimeSequenceOp sequence : sequences)
      failedAny |= failed(check(device, sequence));
    if (failedAny)
      signalPassFailure();
  }

  LogicalResult check(DeviceOp device, RuntimeSequenceOp sequence) {
    Operation *anchor = sequence ? sequence.getOperation() : device;
    std::string what = sequence
                           ? ("dispatching @" + sequence.getSymName()).str()
                           : std::string("running the device");
    auto notes = [](InFlightDiagnostic &diag,
                    ArrayRef<DeadlockModel::Note> list) {
      for (auto &[op, text] : list) {
        if (op)
          diag.attachNote(op->getLoc()) << text;
        else
          diag.attachNote() << text;
      }
    };
    DeadlockModel model(device, sequence);
    if (!model.outside().empty()) {
      auto diag =
          clAllowUndecided ? anchor->emitWarning() : anchor->emitError();
      diag << "cannot decide whether " << what << " can deadlock";
      notes(diag, model.outside());
      return failure(!clAllowUndecided);
    }
    DeadlockModel::Outcome least = model.run(0, 0);
    DeadlockModel::Outcome most = model.run(clBuffering, clShimBuffering);
    if (least.deadlock && most.deadlock) {
      auto diag = anchor->emitError();
      diag << what << " deadlocks";
      notes(diag, most.blocked);
      return failure();
    }
    if (least.deadlock) {
      auto diag =
          clAllowUndecided ? anchor->emitWarning() : anchor->emitError();
      diag << what
           << " deadlocks unless the fabric buffers more than it is "
              "known to; whether it does is undecided";
      notes(diag, least.blocked);
      return failure(!clAllowUndecided);
    }
    // A leftover token matters to the next wait on its channel, which
    // returns early on it; a channel nothing waits on is never asked.
    for (auto &[ch, n] : most.leftoverTokens)
      if (waitedAnywhere.count(ch))
        most.inFlight.push_back(
            {nullptr, llvm::formatv("{0} has {1} completion token(s) nobody "
                                    "waited for, which the next wait on it "
                                    "would take",
                                    model.describe(ch), n)
                          .str()});
    if (!most.inFlight.empty()) {
      auto diag = anchor->emitError();
      diag << what
           << " ends with work still in flight, which the next "
              "dispatch would find";
      notes(diag, most.inFlight);
      return failure();
    }
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIECheckDeadlockPass() {
  return std::make_unique<AIECheckDeadlockPass>();
}
