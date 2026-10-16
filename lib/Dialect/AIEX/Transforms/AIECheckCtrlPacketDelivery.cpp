//===- AIECheckCtrlPacketDelivery.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks that the order aie-ctrl-packet-to-dma delivers control packets in
// keeps every tile's writes in their order, and that no packet stream a started
// channel sends can hold up the control packets that follow it.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "ControlPacketClassifier.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <map>
#include <set>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECHECKCTRLPACKETDELIVERY
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// The tiles each multicast id of a configured device reaches, its MM2S channels
// proved gated, and its routed streams.
struct DeviceFacts {
  std::map<int64_t, SmallVector<AIE::TileID>> multicast;
  std::set<std::tuple<int, int, int>> gated;
  std::vector<AIE::RoutedStream> streams;
};

struct AIECheckCtrlPacketDeliveryPass
    : xilinx::AIEX::impl::AIECheckCtrlPacketDeliveryBase<
          AIECheckCtrlPacketDeliveryPass> {
  using Base::Base;

  llvm::DenseMap<StringAttr, DeviceFacts> facts;

  const DeviceFacts *factsOf(ModuleOp module, FlatSymbolRefAttr device) {
    if (!device)
      return nullptr;
    auto it = facts.find(device.getAttr());
    if (it != facts.end())
      return &it->second;
    DeviceFacts f;
    if (auto d = module.lookupSymbol<AIE::DeviceOp>(device)) {
      AIE::AIEDialect::CtrlPktMulticastIdAttrHelper id(&getContext());
      AIEXDialect::CtrlPktGatedAttrHelper gated(&getContext());
      d.walk([&](AIE::TileOp t) {
        if (auto a = id.getAttr(t))
          f.multicast[a.getInt()].push_back({t.getCol(), t.getRow()});
      });
      d.walk([&](AIE::DMAStartOp start) {
        auto index = dyn_cast<IntegerAttr>(start.getChannel());
        if (!start.isSend() || !index || !gated.isAttrPresent(start))
          return;
        auto t = cast<AIE::TileOp>(
            start->getParentOp()->getOperand(0).getDefiningOp());
        f.gated.insert({t.getCol(), t.getRow(), (int)index.getInt()});
      });
      f.streams = AIE::traceRoutedStreams(d);
    }
    return &facts.try_emplace(device.getAttr(), std::move(f)).first->second;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool failedAny = false;
    for (auto device : module.getOps<AIE::DeviceOp>()) {
      PacketClassifier classifier(device.getTargetModel());
      uint32_t rowShift = device.getTargetModel().getRowShift();
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>())
        seq.walk([&](Block *block) {
          SmallVector<NpuControlPacketOp> run;
          auto check = [&]() {
            if (!run.empty() &&
                failed(checkRun(module, run, classifier, rowShift)))
              failedAny = true;
            run.clear();
          };
          for (Operation &op : *block) {
            if (auto packet = dyn_cast<NpuControlPacketOp>(op))
              run.push_back(packet);
            else
              check();
          }
          check();
        });
    }
    if (failedAny)
      signalPassFailure();
  }

  // The phase a packet is delivered in: one per run serially; under parallel
  // shims the base phase, the rest, then the same two of the enable phase.
  int phaseOf(NpuControlPacketOp packet) {
    if (!parallelShims)
      return 0;
    AIEXDialect::CtrlPktEnablePhaseAttrHelper enablePhase(&getContext());
    AIEXDialect::CtrlPktBasePhaseAttrHelper basePhase(&getContext());
    return (enablePhase.isAttrPresent(packet) ? 2 : 0) +
           (basePhase.isAttrPresent(packet) ? 0 : 1);
  }

  // Position of each packet in its run.
  llvm::DenseMap<Operation *, size_t> position;

  // Whether `a` surely lands before `b`. Serially it does when it comes first;
  // in parallel, when it is of an earlier phase, or of the same phase and first
  // on the same shim channel (one per column).
  bool landsBefore(NpuControlPacketOp a, NpuControlPacketOp b) {
    bool first = position.lookup(a) < position.lookup(b);
    if (!parallelShims)
      return first;
    int pa = phaseOf(a), pb = phaseOf(b);
    return pa < pb || (pa == pb && first &&
                       a.getColumnFromAddr() == b.getColumnFromAddr());
  }

  LogicalResult checkRun(ModuleOp module, ArrayRef<NpuControlPacketOp> run,
                         PacketClassifier &classifier, uint32_t rowShift) {
    AIEXDialect::CtrlPktDeviceAttrHelper deviceStamp(&getContext());
    AIEXDialect::CtrlPktMulticastIdAttrHelper multicastStamp(&getContext());
    position.clear();
    for (auto [i, packet] : llvm::enumerate(run))
      position[packet] = i;

    // Each tile's packets in program order, multicasts at every tile they
    // reach.
    std::map<AIE::TileID, SmallVector<NpuControlPacketOp>> byTile;
    for (NpuControlPacketOp packet : run) {
      AIE::TileID own{(int)packet.getColumnFromAddr(),
                      (int)packet.getRowFromAddr()};
      auto id = multicastStamp.getAttr(packet);
      const DeviceFacts *f = factsOf(module, deviceStamp.getAttr(packet));
      if (id && (!f || !f->multicast.count(id.getInt())))
        return packet.emitOpError("is sent on multicast id ")
               << id.getInt() << ", which no tile of its device carries";
      if (id)
        for (AIE::TileID t : f->multicast.at(id.getInt()))
          byTile[t].push_back(packet);
      else
        byTile[own].push_back(packet);
    }

    // Two writes to a tile that do not commute must land in program order,
    // which serial delivery keeps.
    if (parallelShims) {
      llvm::DenseMap<Operation *, Classified> info;
      for (NpuControlPacketOp packet : run) {
        FailureOr<Classified> c = classifier.classify(packet);
        if (failed(c))
          return failure();
        info[packet] = std::move(*c);
      }
      auto commute = [&](NpuControlPacketOp a, NpuControlPacketOp b) {
        return info[a].commutesWith(info[b]);
      };
      for (auto &[tile, packets] : byTile)
        for (size_t j = 1; j < packets.size(); ++j)
          for (size_t i = 0; i < j; ++i)
            if (!commute(packets[i], packets[j]) &&
                !landsBefore(packets[i], packets[j])) {
              InFlightDiagnostic d =
                  packets[j].emitOpError("may land on tile (")
                  << tile.col << ", " << tile.row
                  << ") before an earlier control packet it does not "
                     "commute with";
              d.attachNote(packets[i].getLoc()) << "the earlier packet";
              return failure();
            }
    }

    // A packet stream a self-starting MM2S channel sends as soon as the run
    // starts it may stall at a receiver nothing has started yet, holding the
    // arbiters on its way until its last word. A control packet that lands
    // after the start must not need one of them.
    for (NpuControlPacketOp packet : run) {
      FailureOr<Classified> c = classifier.classify(packet);
      if (failed(c))
        return failure();
      if (c->part != Part::ChannelStart)
        continue;
      const DeviceFacts *f = factsOf(module, deviceStamp.getAttr(packet));
      if (!f)
        continue;
      AIE::TileID tile{(int)packet.getColumnFromAddr(),
                       (int)packet.getRowFromAddr()};
      if (f->gated.count({tile.col, tile.row, c->channel}))
        continue;
      std::set<std::pair<AIE::TileID, int>> held;
      for (const AIE::RoutedStream &s : f->streams)
        if (s.src.tile == tile && s.src.port.bundle == AIE::WireBundle::DMA &&
            s.src.port.channel == c->channel)
          for (const AIE::StreamHop &hop : s.hops)
            if (hop.arbiter)
              held.insert({hop.tile, *hop.arbiter});
      if (held.empty())
        continue;
      for (NpuControlPacketOp later : run) {
        if (later == packet || landsBefore(later, packet))
          continue;
        AIE::TileID dst{(int)later.getColumnFromAddr(),
                        (int)later.getRowFromAddr()};
        for (const AIE::RoutedStream &s : f->streams) {
          if (s.dst.tile != dst ||
              s.dst.port.bundle != AIE::WireBundle::TileControl)
            continue;
          for (const AIE::StreamHop &hop : s.hops)
            if (hop.arbiter && held.count({hop.tile, *hop.arbiter})) {
              InFlightDiagnostic d =
                  later.emitOpError("may wait forever: its control packets "
                                    "take arbiter ")
                  << *hop.arbiter << " of switchbox (" << hop.tile.col << ", "
                  << hop.tile.row << "), which MM2S channel " << c->channel
                  << " of tile (" << tile.col << ", " << tile.row
                  << ") may hold, stalled, once started";
              d.attachNote(packet.getLoc()) << "the channel is started here";
              return failure();
            }
        }
      }
    }
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIECheckCtrlPacketDeliveryPass() {
  return std::make_unique<AIECheckCtrlPacketDeliveryPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIECheckCtrlPacketDeliveryPass(
    const AIECheckCtrlPacketDeliveryOptions &options) {
  return std::make_unique<AIECheckCtrlPacketDeliveryPass>(options);
}
