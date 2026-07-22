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
// A lock counts as re-armed by an aiex.set_lock, or a write to its lock
// register (npu.write32 / npu.maskwrite32 at its local address). Re-arms are
// collected module-wide, so a re-arm in a later dispatch's sequence is
// honoured. Assumes the objectFIFO stateful transform has run (channels and
// locks materialised).
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include <array>
#include <map>
#include <set>
#include <tuple>

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
using AddrKey = std::tuple<uint64_t, int, int>; // {local address, col, row}

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

  // A lock-register write (npu.write32 / npu.maskwrite32) to a constant local
  // address on a named tile, as (address, col, row). set_lock lowers to exactly
  // this, and a hand-written re-arm may use it directly.
  static std::optional<AddrKey> lockWriteKey(Operation *op) {
    Value addr;
    IntegerAttr colA, rowA;
    if (auto w = dyn_cast<NpuWrite32Op>(op)) {
      addr = w.getAddress();
      colA = w.getColumnAttr();
      rowA = w.getRowAttr();
    } else if (auto w = dyn_cast<NpuMaskWrite32Op>(op)) {
      addr = w.getAddress();
      colA = w.getColumnAttr();
      rowA = w.getRowAttr();
    } else {
      return std::nullopt;
    }
    APInt c;
    if (!colA || !rowA || !matchPattern(addr, m_ConstantInt(&c)))
      return std::nullopt;
    return AddrKey{c.getZExtValue(), (int)colA.getInt(), (int)rowA.getInt()};
  }

  void runOnOperation() override {
    DeviceOp dev = getOperation();
    const AIETargetModel &tm = dev.getTargetModel();
    std::map<ChannelKey, SmallVector<LockOp>> channelLocks =
        mapChannelLocks(dev);

    // Locks re-armed anywhere in the module, by set_lock or a lock-register
    // write. Module-wide so a re-arm in another dispatch's sequence counts.
    llvm::SmallPtrSet<Operation *, 16> reArmedBySetLock;
    std::set<AddrKey> reArmedByWrite;
    dev.walk([&](Operation *op) {
      if (auto sl = dyn_cast<SetLockOp>(op))
        reArmedBySetLock.insert(sl.getLockOp().getOperation());
      else if (auto key = lockWriteKey(op))
        reArmedByWrite.insert(*key);
    });

    auto isReArmed = [&](LockOp l) -> bool {
      if (reArmedBySetLock.contains(l.getOperation()))
        return true;
      if (!l.getLockID())
        return true; // unassigned id: cannot prove a violation, defer
      auto addr = tm.getLocalLockAddress(l.getLockIDValue(), l.getTileID());
      if (!addr)
        return true;
      return reArmedByWrite.count(
          AddrKey{*addr, (int)l.colIndex(), (int)l.rowIndex()});
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
          "with "
          "an aiex.set_lock");
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
