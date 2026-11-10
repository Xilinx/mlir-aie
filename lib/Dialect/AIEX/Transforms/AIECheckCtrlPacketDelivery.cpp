//===- AIECheckCtrlPacketDelivery.cpp ---------------------------*- C++ -*-===//
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
#include "CtrlPacketMulticast.h"
#include "CtrlPacketRuns.h"

#include "mlir/Pass/Pass.h"

#include <map>
#include <optional>
#include <set>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECHECKCTRLPACKETDELIVERY
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// Which packets of a run surely land before which, as aie-ctrl-packet-to-dma
// delivers them (see groupRuns), by their index in the run.
struct RunOrder {
  GroupRun &run;
  bool parallelShims;
  OverlayMulticasts &multicasts;
  SmallVector<unsigned> steps;

  RunOrder(GroupRun &run, bool parallelShims, OverlayMulticasts &multicasts)
      : run(run), parallelShims(parallelShims), multicasts(multicasts),
        steps(awaitSteps(run, parallelShims)) {}

  unsigned phaseOf(size_t i) const { return run[i].phase; }

  const OverlayMulticasts::Info *infoOf(size_t i) const {
    return multicasts.of(run[i].group.getDeviceRefAttr());
  }

  // Whether `a` has landed before `b` is sent: an earlier one on the same
  // channel may still wait on its way where their routes part.
  bool awaitedBefore(size_t a, size_t b) const { return steps[a] < steps[b]; }

  // Whether `a` surely lands on `tile` before `b`. Serially each column's
  // transfers are awaited before another column's; a multicast and a tile's
  // own packets keep their order only along one trail.
  bool landsBeforeOn(size_t a, size_t b, AIE::TileID tile) const {
    if (phaseOf(a) != phaseOf(b))
      return phaseOf(a) < phaseOf(b);
    if (a > b)
      return false;
    bool ma = run[a].packet.getMulticastId().has_value(),
         mb = run[b].packet.getMulticastId().has_value();
    if (ma == mb)
      return sameColumn(a, b) || !parallelShims;
    size_t multicast = ma ? a : b;
    const OverlayMulticasts::Info *info = infoOf(multicast);
    const OverlayMulticasts::Multicast *m =
        info ? info->multicastOf(run[multicast].packet) : nullptr;
    return m && info->keepsOrder(*m, tile, parallelShims);
  }

  // The tiles packet `i` reaches.
  SmallVector<AIE::TileID> tilesOf(size_t i) const {
    return tilesReached(run[i].packet, infoOf(i));
  }

  // The first arbiter of `held` packet `i` takes, if any.
  std::optional<Arbiter> arbiterTaken(size_t i, ConfiguredDevices &devices,
                                      const std::set<Arbiter> &held) const {
    return AIEX::arbiterTaken(run[i].packet, infoOf(i), devices, held);
  }

  bool sameColumn(size_t a, size_t b) const {
    return run[a].packet.getColumnFromAddr() ==
           run[b].packet.getColumnFromAddr();
  }
};

struct AIECheckCtrlPacketDeliveryPass
    : xilinx::AIEX::impl::AIECheckCtrlPacketDeliveryBase<
          AIECheckCtrlPacketDeliveryPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConfiguredDevices devices(module);
    OverlayMulticasts multicasts(module);
    bool failedAny = false;
    for (auto device : module.getOps<AIE::DeviceOp>()) {
      PacketClassifier classifier(device.getTargetModel());
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>())
        for (GroupRun &run : groupRuns(seq, {KindOrDeviceChange::NextPhase,
                                             /*atBarriers=*/false}))
          if (failed(checkRun(run, classifier, devices, multicasts)))
            failedAny = true;
    }
    if (failedAny)
      signalPassFailure();
  }

  // Checks one run against the pass description's rules.
  LogicalResult checkRun(GroupRun &run, PacketClassifier &classifier,
                         ConfiguredDevices &devices,
                         OverlayMulticasts &multicasts) {
    SmallVector<Classified> writes;
    for (GroupedPacket &p : run) {
      FailureOr<Classified> c = classifier.classify(p.packet);
      if (failed(c))
        return failure();
      writes.push_back(std::move(*c));
    }
    RunOrder order{run, parallelShims, multicasts};
    if (failed(checkOrder(order, writes)) ||
        failed(checkHolds(order, writes, devices)))
      return failure();
    return parallelShims ? checkMulticastArbiters(order) : success();
  }

  // The first rule: the packets that reach a tile land in program order,
  // except where they commute.
  static LogicalResult checkOrder(const RunOrder &order,
                                  ArrayRef<Classified> writes) {
    GroupRun &run = order.run;
    // Each tile's packets in program order, a multicast's at every tile it
    // reaches, by index in the run.
    std::map<AIE::TileID, SmallVector<size_t>> byTile;
    for (auto [i, p] : llvm::enumerate(run)) {
      AIE::TileID own = p.packet.getTileFromAddr();
      std::optional<uint32_t> id = p.packet.getMulticastId();
      if (!id) {
        byTile[own].push_back(i);
        continue;
      }
      const OverlayMulticasts::Info *info = order.infoOf(i);
      if (!info || !info->multicastOf(p.packet))
        return p.packet.emitOpError("is sent on multicast id ")
               << *id << " from tile " << tileName(own)
               << ", which is not the lowest tile of a multicast on that id "
                  "in the overlay of its device";
      for (AIE::TileID t : order.tilesOf(i))
        byTile[t].push_back(i);
    }
    for (auto &[tile, indices] : byTile)
      for (size_t j = 1; j < indices.size(); ++j)
        for (size_t i = 0; i < j; ++i) {
          size_t a = indices[i], b = indices[j];
          if (order.landsBeforeOn(a, b, tile) ||
              writes[a].commutesWith(writes[b]))
            continue;
          InFlightDiagnostic d =
              run[b].packet.emitOpError("may land on tile ")
              << tileName(tile)
              << " before an earlier control packet it does not commute with";
          d.attachNote(run[a].packet.getLoc()) << "the earlier packet";
          return failure();
        }
    return success();
  }

  // The second rule: no control packet may wait behind a stream that may
  // stall.
  static LogicalResult checkHolds(const RunOrder &order,
                                  ArrayRef<Classified> writes,
                                  ConfiguredDevices &devices) {
    GroupRun &run = order.run;
    for (auto [i, p] : llvm::enumerate(run)) {
      FlatSymbolRefAttr device = p.group.getDeviceRefAttr();
      for (AIE::TileID tile : order.tilesOf(i)) {
        std::set<Arbiter> held =
            devices.startedArbiters(device, tile, writes[i]);
        if (held.empty())
          continue;
        for (size_t j = 0; j < run.size(); ++j) {
          if (j == i || order.awaitedBefore(j, i))
            continue;
          if (std::optional<Arbiter> hold =
                  order.arbiterTaken(j, devices, held)) {
            InFlightDiagnostic d =
                run[j].packet.emitOpError("may wait forever: its control "
                                          "packets take arbiter ")
                << hold->second << " of switchbox " << tileName(hold->first)
                << ", which a stream that " << describeStart(tile, writes[i])
                << " starts may hold, stalled";
            d.attachNote(p.packet.getLoc()) << "the stream is started here";
            return failure();
          }
        }
      }
    }
    return success();
  }

  // The third rule: multicasts from two shim channels sent in one phase take
  // no common arbiter.
  static LogicalResult checkMulticastArbiters(const RunOrder &order) {
    GroupRun &run = order.run;
    // The first packet of each phase on each column's multicast.
    std::map<unsigned, std::map<int, size_t>> byPhase;
    for (auto [i, p] : llvm::enumerate(run)) {
      const OverlayMulticasts::Info *info = order.infoOf(i);
      const OverlayMulticasts::Multicast *b =
          info ? info->multicastOf(p.packet) : nullptr;
      if (!b)
        continue;
      auto tb = info->trees.find(b->column);
      std::map<int, size_t> &sent = byPhase[order.phaseOf(i)];
      for (auto [column, k] : sent) {
        const OverlayMulticasts::Info *otherInfo = order.infoOf(k);
        auto ta = otherInfo->trees.find(column);
        if (column == b->column || ta == otherInfo->trees.end() ||
            tb == info->trees.end())
          continue;
        for (const Arbiter &hold : tb->second)
          if (ta->second.count(hold))
            return p.packet.emitOpError("may deadlock: the multicasts from "
                                        "columns ")
                   << column << " and " << b->column
                   << " are sent together and both take arbiter " << hold.second
                   << " of switchbox " << tileName(hold.first);
      }
      sent.try_emplace(b->column, i);
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
