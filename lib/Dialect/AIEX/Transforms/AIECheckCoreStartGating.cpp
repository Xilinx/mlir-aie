//===- AIECheckCoreStartGating.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Proves which MM2S channels and cores of a configuration cannot act before the
// configuration is delivered, so control packet reconfiguration may start them
// before the other columns' packets land.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FormatVariadic.h"

#include <map>
#include <optional>
#include <set>
#include <string>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECHECKCORESTARTGATING
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// std::nullopt when proved, otherwise the reason it is not.
using Verdict = std::optional<std::string>;

// Whether a device's agents can act before its configuration is delivered.
// Cores and the host cannot, since cores are enabled after the configuration
// and the host's own writes follow it in the runtime sequence. An MM2S channel
// whose first buffer descriptor does not wait can, and so can whatever the data
// it sends sets off.
struct Checker {
  explicit Checker(DeviceOp device) : device(device) {}

  DeviceOp device;
  // Every release of a lock, keyed on the aie.lock.
  DenseMap<Value, SmallVector<UseLockOp>> releases;
  // (tile, MM2S?, channel) -> the channel's aie.dma_start.
  std::map<std::tuple<Operation *, bool, int>, DMAStartOp> starts;
  // BD block -> (tile, MM2S?, channel) of the chain it is on.
  DenseMap<Block *, std::tuple<Operation *, bool, int>> blockChannel;
  // A source of a stream into an S2MM channel.
  struct Source {
    Operation *tile;
    WireBundle bundle;
    int channel;
  };
  // (tile, S2MM channel) -> the sources streams come from; a null tile is
  // one the device does not declare.
  std::map<std::pair<Operation *, int>, SmallVector<Source>> sources;

  DenseMap<Value, Verdict> lockMemo;
  DenseSet<Value> lockBusy;

  static LockOp lockOf(Value lock) {
    return dyn_cast_or_null<LockOp>(lock.getDefiningOp());
  }

  void collect() {
    device.walk([&](UseLockOp u) {
      if (u.release())
        releases[u.getLock()].push_back(u);
    });
    device.walk([&](DMAStartOp s) {
      auto index = dyn_cast<IntegerAttr>(s.getChannel());
      if (!index)
        return;
      Operation *tile = s->getParentOp()->getOperand(0).getDefiningOp();
      std::tuple<Operation *, bool, int> key{tile, s.isSend(),
                                             static_cast<int>(index.getInt())};
      starts[key] = s;
      SmallVector<Block *> work{s.getDest()};
      DenseSet<Block *> seen;
      while (!work.empty()) {
        Block *b = work.pop_back_val();
        if (!seen.insert(b).second)
          continue;
        blockChannel[b] = key;
        if (auto next = dyn_cast<NextBDOp>(b->getTerminator()))
          work.push_back(next.getDest());
      }
    });
    // The streams into S2MM channels, as the flows ask for them before routing
    // and as the switchboxes route them after.
    DenseMap<TileID, Operation *> tiles;
    device.walk([&](TileOp t) { tiles[{t.colIndex(), t.rowIndex()}] = t; });
    auto addStream = [&](const RoutedStream &s) {
      Operation *dst = tiles.lookup(s.dst.tile);
      if (s.dst.port.bundle != WireBundle::DMA || !dst)
        return;
      sources[{dst, s.dst.port.channel}].push_back(
          {tiles.lookup(s.src.tile), s.src.port.bundle, s.src.port.channel});
    };
    for (const RoutedStream &s : requestedStreams(device))
      addStream(s);
    for (const RoutedStream &s : traceRoutedStreams(device))
      addStream(s);
  }

  // Whether an acquire cannot succeed on the lock's initial value; nullopt if
  // the value or the initial value is unknown. The configuration sets a lock
  // to its initializer when it has one and an id, and zeroes every lock of a
  // tile with a core (AIERTControl::initLocks).
  static std::optional<bool> gates(UseLockOp u) {
    auto c = u.getValue().getDefiningOp<arith::ConstantOp>();
    auto value = c ? dyn_cast<IntegerAttr>(c.getValue()) : nullptr;
    LockOp l = lockOf(u.getLock());
    if (!value || !l)
      return std::nullopt;
    TileOp t = l.getTileOp();
    std::optional<int> init;
    if (l.getInit() && l.getLockID())
      init = *l.getInit();
    else if (!t.isShimTile() && t.getCoreOp())
      init = 0;
    if (!init)
      return std::nullopt;
    return *init < value.getInt();
  }

  // A channel cannot start moving data when its first buffer descriptor waits
  // on a lock that nothing acting during the configuration releases.
  Verdict channelGated(Operation *tile, int channel) {
    auto it = starts.find({tile, true, channel});
    if (it == starts.end())
      return std::string("an MM2S channel with no aie.dma_start");
    Block *first = it->second.getDest();
    for (UseLockOp u : first->getOps<UseLockOp>()) {
      if (!u.acquire() && !u.acquireGE())
        continue;
      if (!u.getAcqEn() ||
          (u.getBlocking() && *u.getBlocking() == LockBlocking::NonBlocking))
        return std::string("its first buffer descriptor's acquire does not "
                           "wait");
      LockOp l = lockOf(u.getLock());
      if (!l)
        return std::string("a lock not resolved to aie.lock");
      if (l.getTileOp().getOperation() != tile)
        return std::string("its first buffer descriptor acquires another "
                           "tile's lock");
      std::optional<bool> g = gates(u);
      if (!g)
        return std::string("a lock value or initial value that is not a "
                           "constant");
      if (!*g)
        return std::string("the lock's initial value satisfies its first "
                           "buffer descriptor's acquire");
      return lockQuiet(u.getLock());
    }
    return std::string("its first buffer descriptor does not acquire a lock");
  }

  Verdict lockQuiet(Value lock) {
    if (auto it = lockMemo.find(lock); it != lockMemo.end())
      return it->second;
    if (!lockBusy.insert(lock).second)
      return std::string("a cycle through a lock");
    Verdict v = lockQuietImpl(lock);
    lockBusy.erase(lock);
    return lockMemo[lock] = v;
  }

  // Every release of the lock waits for something that starts only after the
  // configuration.
  Verdict lockQuietImpl(Value lock) {
    for (UseLockOp r : releases.lookup(lock)) {
      if (r->getParentOfType<CoreOp>())
        continue;
      auto it = blockChannel.find(r->getBlock());
      if (it == blockChannel.end())
        return std::string("a release outside a core or a buffer descriptor "
                           "chain");
      auto [tile, mm2s, channel] = it->second;
      Verdict v = mm2s ? channelGated(tile, channel)
                       : s2mmQuiet(r->getBlock(), tile, channel);
      if (v)
        return v;
    }
    return std::nullopt;
  }

  // An S2MM buffer descriptor completes only when data arrives on its
  // channel. Whether a zero-length one completes with none is not modelled.
  Verdict s2mmQuiet(Block *b, Operation *tile, int channel) {
    auto bds = b->getOps<DMABDOp>();
    if (bds.empty())
      return std::string("an S2MM buffer descriptor block with no aie.dma_bd");
    for (DMABDOp bd : bds) {
      std::optional<int64_t> len = bd.getConstantLen();
      if (!len && !bd.getLen())
        if (auto t = dyn_cast<MemRefType>(bd.getBuffer().getType());
            t && t.hasStaticShape())
          len = t.getNumElements();
      if (!len || *len <= 0)
        return std::string("an S2MM buffer descriptor whose length is not a "
                           "known nonzero constant");
    }
    auto it = sources.find({tile, channel});
    if (it == sources.end() || it->second.empty())
      return std::string("an S2MM channel no flow feeds");
    for (const Source &src : it->second) {
      if (src.bundle == WireBundle::Core)
        continue;
      if (!src.tile)
        return std::string("an S2MM channel fed from a tile the device does "
                           "not declare");
      if (src.bundle != WireBundle::DMA)
        return llvm::formatv("an S2MM channel fed from a {0} port",
                             stringifyWireBundle(src.bundle))
            .str();
      // A shim channel no aie.dma_start of the device starts is started only
      // by the runtime sequence, after the configuration.
      if (cast<TileOp>(src.tile).isShimTile() &&
          !starts.count({src.tile, true, src.channel}))
        continue;
      if (Verdict v = channelGated(src.tile, src.channel))
        return v;
    }
    return std::nullopt;
  }
};

struct AIECheckCoreStartGatingPass
    : xilinx::AIEX::impl::AIECheckCoreStartGatingBase<
          AIECheckCoreStartGatingPass> {
  void runOnOperation() override {
    AIEX::AIEXDialect::CtrlPktGatedAttrHelper gated(&getContext());
    getOperation().walk([&](DeviceOp device) {
      Checker checker(device);
      checker.collect();
      device.walk([&](DMAStartOp start) {
        auto index = dyn_cast<IntegerAttr>(start.getChannel());
        if (!start.isSend() || !index)
          return;
        auto t =
            cast<TileOp>(start->getParentOp()->getOperand(0).getDefiningOp());
        int channel = index.getInt();
        if (Verdict v = checker.channelGated(t, channel)) {
          if (gated.isAttrPresent(start))
            gated.removeAttr(start);
          start.emitRemark()
              << "MM2S channel " << channel << " of tile (" << t.getCol()
              << ", " << t.getRow() << ") not proved gated: " << *v;
        } else {
          gated.setAttr(start, UnitAttr::get(&getContext()));
        }
      });
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIECheckCoreStartGatingPass() {
  return std::make_unique<AIECheckCoreStartGatingPass>();
}
