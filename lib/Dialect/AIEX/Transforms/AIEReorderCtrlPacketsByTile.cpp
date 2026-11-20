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
#define GEN_PASS_DEF_AIESCHEDULECTRLPACKETACTIVATION
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIEScheduleCtrlPacketActivationPass
    : xilinx::AIEX::impl::AIEScheduleCtrlPacketActivationBase<
          AIEScheduleCtrlPacketActivationPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConfiguredDevices devices(module);
    for (auto device : module.getOps<AIE::DeviceOp>()) {
      PacketClassifier classifier(device.getTargetModel());
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>())
        for (PacketRun &run : packetRuns(seq))
          if (failed(reorder(run, classifier, devices)))
            return signalPassFailure();
    }
  }

  // A packet of the run with its sort key and, once ordered, its phase (0 for
  // the configuration) and place.
  struct Placed {
    std::tuple<WriteKind, uint32_t, uint32_t> key;
    ContextualPacket entry;
    Classified write;
    unsigned phase = 0;
    size_t rank = 0;
  };

  // Puts the run's packets in the order the pass description gives.
  LogicalResult reorder(PacketRun &run, PacketClassifier &classifier,
                        ConfiguredDevices &devices) {
    SmallVector<Placed> placed;
    for (ContextualPacket entry : run) {
      FailureOr<Classified> c = classifier.classify(entry.packet);
      if (failed(c))
        return failure();
      AIE::TileID tile = entry.packet.getTileFromAddr();
      WriteKind part = c->kind;
      // With gated-enable, a proved core's enable goes last among its
      // column's other writes.
      uint32_t row = tile.row;
      FlatSymbolRefAttr device = entry.device;
      if (useGating && part == WriteKind::ChannelStart &&
          devices.startGated(device, {tile, c->channel})) {
        part = WriteKind::Other;
      } else if (useGating && part == WriteKind::CoreEnable && gatedEnable &&
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

    Block *block = run.back().packet->getBlock();
    auto end = std::next(run.back().packet->getIterator());
    OpBuilder builder(block, end);
    std::optional<unsigned> phase;
    for (Placed &p : placed) {
      ContextualPacket entry = p.entry;
      if (phase && *phase != p.phase)
        ControlPacketBarrierOp::create(builder, entry.packet.getLoc());
      phase = p.phase;
      entry.packet->moveBefore(block, end);
    }
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
    FlatSymbolRefAttr device = placed[start].entry.device;
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

struct AIEReorderCtrlPacketsByTilePass
    : xilinx::AIEX::impl::AIEReorderCtrlPacketsByTileBase<
          AIEReorderCtrlPacketsByTilePass> {
  static void normalize(SmallVectorImpl<NpuControlPacketOp> &packets) {
    if (packets.size() < 2) {
      packets.clear();
      return;
    }
    Block *block = packets.back()->getBlock();
    auto end = std::next(packets.back()->getIterator());
    llvm::DenseSet<uint32_t> written;
    SmallVector<NpuControlPacketOp> surviving;
    for (NpuControlPacketOp packet : llvm::reverse(packets)) {
      auto data = packet.getData().value();
      SmallVector<bool, 4> live;
      for (size_t index = 0; index < data.size(); ++index)
        live.push_back(written.insert(packet.getAddress() + 4 * index).second);
      if (llvm::all_of(live, [](bool value) { return value; })) {
        surviving.push_back(packet);
        continue;
      }
      OpBuilder builder(packet);
      for (size_t begin = 0; begin < data.size();) {
        if (!live[begin]) {
          ++begin;
          continue;
        }
        size_t limit = begin + 1;
        while (limit < data.size() && live[limit])
          ++limit;
        auto fragment = NpuControlPacketOp::create(
            builder, packet.getLoc(),
            builder.getUI32IntegerAttr(packet.getAddress() + 4 * begin),
            IntegerAttr{}, packet.getOpcodeAttr(), packet.getStreamIdAttr(),
            builder.getDenseI32ArrayAttr(data.slice(begin, limit - begin)));
        surviving.push_back(fragment);
        begin = limit;
      }
      packet.erase();
    }
    llvm::sort(surviving,
               [](NpuControlPacketOp left, NpuControlPacketOp right) {
                 return left.getAddress() < right.getAddress();
               });
    for (NpuControlPacketOp packet : surviving)
      packet->moveBefore(block, end);
    packets.clear();
  }

  void runOnOperation() override {
    getOperation().walk([&](AIE::RuntimeSequenceOp sequence) {
      sequence.walk([&](Block *block) {
        SmallVector<NpuControlPacketOp> packets;
        for (Operation &operation : llvm::make_early_inc_range(*block)) {
          auto packet = dyn_cast<NpuControlPacketOp>(operation);
          if (!packet || packet.getOpcode() != 0 || !packet.getData() ||
              packet.getMulticastId()) {
            normalize(packets);
            continue;
          }
          if (!packets.empty() &&
              packets.back().getStreamId() != packet.getStreamId())
            normalize(packets);
          packets.push_back(packet);
        }
        normalize(packets);
      });
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEReorderCtrlPacketsByTilePass() {
  return std::make_unique<AIEReorderCtrlPacketsByTilePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEScheduleCtrlPacketActivationPass() {
  return std::make_unique<AIEScheduleCtrlPacketActivationPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEScheduleCtrlPacketActivationPass(
    const AIEScheduleCtrlPacketActivationOptions &options) {
  return std::make_unique<AIEScheduleCtrlPacketActivationPass>(options);
}
