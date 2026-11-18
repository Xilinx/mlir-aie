//===- AIEReorderCtrlPacketsByTile.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "CtrlPacketClassifier.h"
#include "CtrlPacketDevices.h"
#include "CtrlPacketRuns.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEREORDERCTRLPACKETSBYTILE
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIEReorderCtrlPacketsByTilePass
    : xilinx::AIEX::impl::AIEReorderCtrlPacketsByTileBase<
          AIEReorderCtrlPacketsByTilePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConfiguredDevices devices(module);
    for (auto device : module.getOps<AIE::DeviceOp>()) {
      PacketClassifier classifier(device.getTargetModel());
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>())
        for (GroupRun &run : groupRuns(seq))
          if (failed(reorder(run, classifier, devices)))
            return signalPassFailure();
    }
  }

  // A packet of the run with its sort key and, once ordered, its phase (0 for
  // the configuration) and place.
  struct Placed {
    std::tuple<WriteKind, uint32_t, uint32_t> key;
    GroupedPacket entry;
    Classified write;
    unsigned phase = 0;
    size_t rank = 0;
  };

  // Puts the run's packets in the order the pass description gives.
  LogicalResult reorder(GroupRun &run, PacketClassifier &classifier,
                        ConfiguredDevices &devices) {
    SmallVector<Placed> placed;
    for (GroupedPacket entry : run) {
      FailureOr<Classified> c = classifier.classify(entry.packet);
      if (failed(c))
        return failure();
      AIE::TileID tile = entry.packet.getTileFromAddr();
      WriteKind part = c->kind;
      // With gated-enable, a proved core's enable goes last among its
      // column's other writes.
      uint32_t row = tile.row;
      FlatSymbolRefAttr device = entry.group.getDeviceRefAttr();
      if (part == WriteKind::ChannelStart &&
          devices.startGated(device, {tile, c->channel})) {
        part = WriteKind::Other;
      } else if (part == WriteKind::CoreEnable && gatedEnable &&
                 devices.startGated(device, {tile, -1})) {
        part = WriteKind::Other;
        row = std::numeric_limits<uint32_t>::max();
      }
      std::tuple<WriteKind, uint32_t, uint32_t> key{part, 0, 0};
      if (byTile)
        key = {part, static_cast<uint32_t>(tile.col), row};
      placed.push_back({key, entry, *c});
    }
    std::stable_sort(
        placed.begin(), placed.end(),
        [](const Placed &a, const Placed &b) { return a.key < b.key; });
    if (failed(orderEnablePhase(placed, devices)))
      return failure();
    std::stable_sort(
        placed.begin(), placed.end(), [](const Placed &a, const Placed &b) {
          return std::tie(a.phase, a.rank) < std::tie(b.phase, b.rank);
        });

    // The packets go back after the run, each in a group like its old one; a
    // barrier ends each phase.
    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(run.back().group);
    llvm::SetVector<Operation *> oldGroups;
    for (GroupedPacket entry : run)
      oldGroups.insert(entry.group);
    ControlPacketGroupOp current;
    std::optional<unsigned> phase;
    for (Placed &p : placed) {
      GroupedPacket entry = p.entry;
      if (phase && *phase != p.phase) {
        ControlPacketBarrierOp::create(builder, entry.packet.getLoc());
        current = nullptr;
      }
      phase = p.phase;
      if (!current) {
        current = ControlPacketGroupOp::create(builder, entry.group.getLoc(),
                                               entry.group.getKindAttr(),
                                               entry.group.getDeviceRefAttr());
        current.getBody().emplaceBlock();
      }
      Block &body = current.getBody().front();
      entry.packet->moveBefore(&body, body.end());
    }
    for (Operation *group : oldGroups)
      group->erase();
    return success();
  }

  // Places the enable phase of `placed`, sorted by key, keeping key order
  // except that each packet that starts streams goes after every packet there
  // whose control packets take an arbiter those may hold, stalled, in an
  // earlier phase: one sent before it on the same channel may still wait on
  // its way. Serially another column's packets are awaited first anyway.
  LogicalResult orderEnablePhase(MutableArrayRef<Placed> placed,
                                 ConfiguredDevices &devices) {
    size_t n = placed.size();
    auto enable = llvm::find_if(placed, [](const Placed &p) {
      return std::get<0>(p.key) != WriteKind::Other;
    });
    size_t start = enable - placed.begin();
    for (size_t i = 0; i < start; ++i)
      placed[i].rank = i;
    if (start == n)
      return success();
    FlatSymbolRefAttr device = placed[start].entry.group.getDeviceRefAttr();
    // For each packet, the packets that wait for it and whether in a later
    // phase.
    SmallVector<SmallVector<std::pair<size_t, bool>>> waiters(n);
    SmallVector<SmallVector<size_t>> waitsFor(n);
    for (size_t m = start; m < n; ++m) {
      AIE::TileID tile = placed[m].entry.packet.getTileFromAddr();
      std::set<Arbiter> held =
          devices.startedArbiters(device, tile, placed[m].write);
      if (held.empty())
        continue;
      for (size_t p = start; p < n; ++p) {
        AIE::TileID dst = placed[p].entry.packet.getTileFromAddr();
        if (p == m || !devices.controlArbiterIn(device, dst, held))
          continue;
        waiters[p].push_back({m, parallelShims || dst.col == tile.col});
        waitsFor[m].push_back(p);
      }
    }
    // Each step places the first packet in key order that waits for none.
    SmallVector<size_t> left(n);
    std::set<size_t> ready;
    for (size_t i = start; i < n; ++i) {
      left[i] = waitsFor[i].size();
      if (!left[i])
        ready.insert(i);
    }
    size_t rank = start;
    while (!ready.empty()) {
      size_t p = *ready.begin();
      ready.erase(ready.begin());
      placed[p].rank = rank++;
      for (auto [m, later] : waiters[p]) {
        placed[m].phase = std::max(placed[m].phase, placed[p].phase + later);
        if (!--left[m])
          ready.insert(m);
      }
    }
    if (rank < n) {
      // Stepping back from an unplaced packet n times ends in a cycle.
      size_t m = start;
      while (!left[m])
        ++m;
      for (size_t i = start; i < n; ++i)
        m = *llvm::find_if(waitsFor[m], [&](size_t p) { return left[p]; });
      return placed[m].entry.packet.emitOpError("is ")
             << describeStart(placed[m].entry.packet.getTileFromAddr(),
                              placed[m].write)
             << ", in a cycle of packets that start streams, each of which "
                "may hold, stalled, an arbiter the next one's control packets "
                "take";
    }
    for (size_t i = start; i < n; ++i)
      placed[i].phase += 1;
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEReorderCtrlPacketsByTilePass() {
  return std::make_unique<AIEReorderCtrlPacketsByTilePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEReorderCtrlPacketsByTilePass(
    const AIEReorderCtrlPacketsByTileOptions &options) {
  return std::make_unique<AIEReorderCtrlPacketsByTilePass>(options);
}
