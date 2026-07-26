//===- AIEVerifyRuntimeRearm.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EXPERIMENTAL, opt-in diagnostic (not in the default pipeline; validate on
// hardware before relying on it). Rejects an aiex.dma_channel_reset whose
// objectFIFO locks are not re-armed anywhere in the module. That is a
// necessary, not sufficient, condition for a safe channel reset.
//
// A channel reset drains the DMA task queue and leaves the objectFIFO lock
// counters frozen; if a bound lock is never re-armed the channel's peer blocks
// on an acquire that never releases (host: `qds_device::wait() unexpected
// command state`). This catches the missing-lock-re-arm mistake. It does NOT
// prove the reset is safe: the complementary start-queue re-push (no
// runtime-sequence op yet for a resident core/mem channel) is not checked, so a
// design that re-arms locks but not the queue still deadlocks and still passes.
//
// A lock counts as re-armed by an aiex.set_lock. The pass runs before set_lock
// is lowered, so it matches the semantic op instead of decoding a lowered
// npu.write32 (which would duplicate the set_lock encoding here). Re-arms are
// collected module-wide, so a re-arm in a later dispatch's sequence is
// honoured. Assumes the objectFIFO stateful transform has run (channels and
// locks materialised).
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <map>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEVERIFYRUNTIMEREARM
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

using ChannelKey = std::array<int, 4>; // {col, row, direction, channel}

struct AIEVerifyRuntimeRearmPass
    : xilinx::AIEX::impl::AIEVerifyRuntimeRearmBase<AIEVerifyRuntimeRearmPass> {

  // channel -> the objectFIFO locks its BD chain uses, built in one device
  // walk.
  std::map<ChannelKey, SmallVector<LockOp>> mapChannelLocks(DeviceOp dev) {
    std::map<ChannelKey, SmallVector<LockOp>> m;
    dev.walk([&](DMAStartOp start) {
      Operation *memOp = start->getParentOp();
      TileOp tile;
      if (auto x = dyn_cast<MemOp>(memOp))
        tile = x.getTileOp();
      else if (auto x = dyn_cast<MemTileDMAOp>(memOp))
        tile = x.getTileOp();
      if (!tile)
        return;
      ChannelKey key{tile.getCol(), tile.getRow(),
                     static_cast<int>(start.getChannelDir()),
                     static_cast<int>(start.getChannelIndex())};
      SmallVector<LockOp> &locks = m[key];
      // Walk the BD chain (dest, then next_bd successors); the last next_bd
      // loops back, so stop on a revisit.
      llvm::SmallPtrSet<Block *, 8> visited;
      for (Block *b = start.getDest(); b && visited.insert(b).second;
           b = (b->getNumSuccessors() == 1 ? b->getSuccessor(0) : nullptr))
        for (auto use : b->getOps<UseLockOp>())
          if (auto l = dyn_cast_or_null<LockOp>(use.getLock().getDefiningOp()))
            if (!llvm::is_contained(locks, l))
              locks.push_back(l);
    });
    return m;
  }

  void runOnOperation() override {
    DeviceOp dev = getOperation();
    std::map<ChannelKey, SmallVector<LockOp>> channelLocks =
        mapChannelLocks(dev);

    // Locks re-armed anywhere in the module by an aiex.set_lock. Module-wide so
    // a re-arm in another dispatch's sequence counts. The pass runs before
    // set_lock is lowered, so matching the op is enough; no need to decode a
    // lowered npu.write32 to the lock's register.
    llvm::SmallPtrSet<Operation *, 16> reArmedBySetLock;
    dev.walk([&](SetLockOp sl) {
      reArmedBySetLock.insert(sl.getLockOp().getOperation());
    });

    auto isReArmed = [&](LockOp l) -> bool {
      return reArmedBySetLock.contains(l.getOperation());
    };

    WalkResult wr = dev.walk([&](DmaChannelResetOp reset) -> WalkResult {
      TileOp tile = reset.getTileOp();
      if (!tile)
        return WalkResult::advance();
      ChannelKey key{tile.getCol(), tile.getRow(),
                     static_cast<int>(reset.getDirection()),
                     static_cast<int>(reset.getChannel())};
      auto it = channelLocks.find(key);
      if (it == channelLocks.end())
        return WalkResult::advance();
      SmallVector<LockOp> frozen;
      for (LockOp l : it->second)
        if (!isReArmed(l))
          frozen.push_back(l);
      if (frozen.empty())
        return WalkResult::advance();
      InFlightDiagnostic err = reset.emitOpError(
          "resets a DMA channel whose objectFIFO lock is never re-armed; the "
          "semaphore counter stays frozen and the channel's peer will block "
          "forever on acquire (a runtime deadlock). Re-arm each bound lock "
          "with an aiex.set_lock");
      for (LockOp l : frozen)
        err.attachNote(l.getLoc())
            << "this lock is bound to the reset channel and is not re-armed";
      return WalkResult::interrupt();
    });
    if (wr.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIEVerifyRuntimeRearmPass() {
  return std::make_unique<AIEVerifyRuntimeRearmPass>();
}
