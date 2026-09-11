//===- AIEFindFlows.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEFINDFLOWS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-find-flows"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

using MaskValue = struct MaskValue {
  int mask;
  int value;
};

using PortConnection = struct PortConnection {
  Operation *op;
  Port port;
};

using PortMaskValue = struct PortMaskValue {
  Port port;
  MaskValue mv;
  // The interconnect ops that implement this hop (a circuit ConnectOp, or a
  // packet PacketRuleOp + MasterSetOp + AMSelOp). Recorded so the flows that
  // consume them can be lifted out of the physical IR.
  llvm::SmallVector<Operation *, 3> ops;
};

// One intermediate stop of a flow: the tile whose switchbox the stream passes
// through, and the ingress/egress ports it uses there.
using Via = struct Via {
  Value tile;
  Port ingress;
  Port egress;
};

using PacketConnection = struct PacketConnection {
  PortConnection portConnection;
  MaskValue mv;
  llvm::SmallVector<Via, 4> vias;
  // Interconnect ops traversed on the way to this endpoint.
  llvm::SmallVector<Operation *, 8> usedOps;
};

class ConnectivityAnalysis {
  DeviceOp &device;
  // Fan-out splitting: when enabled, a linear traversal stops at any switchbox
  // input port that drives more than one output (a fan-out), leaving the
  // fan-out switchbox explicit and recording its output ports as new section
  // sources.
  bool splitFanouts = false;
  mutable llvm::DenseSet<std::pair<Operation *, int>> fanoutIngresses;
  mutable llvm::DenseSet<std::pair<Operation *, int>> fanoutEgressSeeds;

  static int encodePort(Port p) {
    return static_cast<int>(p.bundle) * 64 + p.channel;
  }

public:
  ConnectivityAnalysis(DeviceOp &d) : device(d) {}

  // Enable fan-out splitting and precompute which interconnect input ports fan
  // out (drive more than one output).
  void enableFanoutSplitting() {
    splitFanouts = true;
    auto scan = [&](Operation *sw, Region &connections) {
      llvm::SmallVector<Port, 8> sources;
      Block &b = connections.front();
      for (auto connectOp : b.getOps<ConnectOp>())
        if (!llvm::is_contained(sources, connectOp.sourcePort()))
          sources.push_back(connectOp.sourcePort());
      for (auto rulesOp : b.getOps<PacketRulesOp>())
        if (!llvm::is_contained(sources, rulesOp.sourcePort()))
          sources.push_back(rulesOp.sourcePort());
      // Fan-out: a source that drives more than one output.
      for (Port p : sources)
        if (getConnectionsThroughSwitchbox(connections, p).size() > 1)
          fanoutIngresses.insert({sw, encodePort(p)});
      // Fan-in: an output driven from more than one source (a packet merge).
      // Every source feeding a shared output is a section boundary as well, so
      // the merge node stays materialized and the linear sections stop at it.
      llvm::DenseMap<int, llvm::SmallVector<Port, 4>> outputSources;
      for (Port p : sources)
        for (PortMaskValue &pmv :
             getConnectionsThroughSwitchbox(connections, p)) {
          auto &v = outputSources[encodePort(pmv.port)];
          if (!llvm::is_contained(v, p))
            v.push_back(p);
        }
      for (auto &[egress, srcs] : outputSources)
        if (srcs.size() > 1)
          for (Port p : srcs)
            fanoutIngresses.insert({sw, encodePort(p)});
      // Break at hops whose configuration carries state a lifted flow cannot
      // reconstruct, so only that switchbox stays materialized while the rest
      // of the flow still lifts (split-at-change). keep_pkt_header on a real
      // endpoint is carried on the flow (so it does not break), and control
      // overlays are kept whole elsewhere; every other attribute breaks here.
      auto isFabric = [](WireBundle bnd) {
        return bnd == WireBundle::North || bnd == WireBundle::South ||
               bnd == WireBundle::East || bnd == WireBundle::West;
      };
      auto hasNonCtrlDiscardable = [](Operation *op) {
        return llvm::any_of(op->getDiscardableAttrDictionary(),
                            [](NamedAttribute a) {
                              return a.getName() != "is_ctrl_pkt_overlay";
                            });
      };
      for (Port p : sources)
        for (PortMaskValue &pmv :
             getConnectionsThroughSwitchbox(connections, p))
          for (Operation *op : pmv.ops) {
            if (!op)
              continue;
            bool unliftable = hasNonCtrlDiscardable(op);
            if (auto r = dyn_cast<PacketRuleOp>(op))
              if (Operation *par = r->getParentOp())
                unliftable |= hasNonCtrlDiscardable(par);
            if (auto ms = dyn_cast<MasterSetOp>(op))
              unliftable |=
                  ms.getKeepPktHeaderAttr() && isFabric(ms.getDestBundle());
            if (unliftable)
              fanoutIngresses.insert({sw, encodePort(p)});
          }
    };
    for (auto switchOp : device.getOps<SwitchboxOp>())
      scan(switchOp, switchOp.getConnections());
    for (auto shimMuxOp : device.getOps<ShimMuxOp>())
      scan(shimMuxOp, shimMuxOp.getConnections());
  }

  // Fan-out output ports discovered while traversing, each a source for a new
  // linear section.  Consumed and cleared by the pass.
  llvm::DenseSet<std::pair<Operation *, int>> &getFanoutEgressSeeds() {
    return fanoutEgressSeeds;
  }
  static Port decodePort(int e) {
    return {static_cast<WireBundle>(e / 64), e % 64};
  }

private:
  std::optional<PortConnection>
  getConnectionThroughWire(Operation *op, Port masterPort) const {
    LLVM_DEBUG(llvm::dbgs() << "Wire:" << *op << " "
                            << stringifyWireBundle(masterPort.bundle) << " "
                            << masterPort.channel << "\n");
    for (auto wireOp : device.getOps<WireOp>()) {
      if (wireOp.getSource().getDefiningOp() == op &&
          wireOp.getSourceBundle() == masterPort.bundle) {
        Operation *other = wireOp.getDest().getDefiningOp();
        Port otherPort = {wireOp.getDestBundle(), masterPort.channel};
        LLVM_DEBUG(llvm::dbgs() << "Connects To:" << *other << " "
                                << stringifyWireBundle(otherPort.bundle) << " "
                                << otherPort.channel << "\n");

        return PortConnection{other, otherPort};
      }
      if (wireOp.getDest().getDefiningOp() == op &&
          wireOp.getDestBundle() == masterPort.bundle) {
        Operation *other = wireOp.getSource().getDefiningOp();
        Port otherPort = {wireOp.getSourceBundle(), masterPort.channel};
        LLVM_DEBUG(llvm::dbgs() << "Connects To:" << *other << " "
                                << stringifyWireBundle(otherPort.bundle) << " "
                                << otherPort.channel << "\n");
        return PortConnection{other, otherPort};
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "*** Missing Wire!\n");
    return std::nullopt;
  }

  std::vector<PortMaskValue>
  getConnectionsThroughSwitchbox(Region &r, Port sourcePort) const {
    LLVM_DEBUG(llvm::dbgs() << "Switchbox:\n");
    Block &b = r.front();
    std::vector<PortMaskValue> portSet;
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        MaskValue maskValue = {0, 0};
        portSet.push_back({connectOp.destPort(), maskValue, {connectOp}});
        LLVM_DEBUG(llvm::dbgs()
                   << "To:" << stringifyWireBundle(connectOp.destPort().bundle)
                   << " " << connectOp.destPort().channel << "\n");
      }
    }
    for (auto connectOp : b.getOps<PacketRulesOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Packet From: "
                   << stringifyWireBundle(connectOp.sourcePort().bundle) << " "
                   << sourcePort.channel << "\n");
        for (auto masterSetOp : b.getOps<MasterSetOp>())
          for (Value amsel : masterSetOp.getAmsels())
            for (auto ruleOp :
                 connectOp.getRules().front().getOps<PacketRuleOp>()) {
              if (ruleOp.getAmsel() == amsel) {
                LLVM_DEBUG(llvm::dbgs()
                           << "To:"
                           << stringifyWireBundle(masterSetOp.destPort().bundle)
                           << " " << masterSetOp.destPort().channel << "\n");
                MaskValue maskValue = {ruleOp.maskInt(), ruleOp.valueInt()};
                portSet.push_back(
                    {masterSetOp.destPort(),
                     maskValue,
                     {ruleOp, masterSetOp, amsel.getDefiningOp()}});
              }
            }
      }
    }
    return portSet;
  }

  std::vector<PacketConnection> maskSwitchboxConnections(
      Operation *switchOp, Port ingressPort, ArrayRef<Via> currentVias,
      ArrayRef<Operation *> currentUsedOps,
      const std::vector<PortMaskValue> &nextPortMaskValues, MaskValue maskValue,
      bool keepPartialFlows,
      std::vector<PacketConnection> &partialEndpoints) const {
    std::vector<PacketConnection> worklist;
    // Only switchbox hops become vias; the shim-mux is not stream-switch
    // configuration and is regenerated by routing.
    Value viaTile;
    if (auto sb = dyn_cast<SwitchboxOp>(switchOp))
      viaTile = sb.getTile();
    for (auto &nextPortMaskValue : nextPortMaskValues) {
      Port nextPort = nextPortMaskValue.port;
      MaskValue nextMaskValue = nextPortMaskValue.mv;
      int maskConflicts = nextMaskValue.mask & maskValue.mask;
      LLVM_DEBUG(llvm::dbgs() << "Mask: " << maskValue.mask << " "
                              << maskValue.value << "\n");
      LLVM_DEBUG(llvm::dbgs() << "NextMask: " << nextMaskValue.mask << " "
                              << nextMaskValue.value << "\n");
      LLVM_DEBUG(llvm::dbgs() << maskConflicts << "\n");

      if ((maskConflicts & nextMaskValue.value) !=
          (maskConflicts & maskValue.value)) {
        // Incoming packets cannot match this rule. Skip it.
        continue;
      }
      MaskValue newMaskValue = {maskValue.mask | nextMaskValue.mask,
                                maskValue.value |
                                    (nextMaskValue.mask & nextMaskValue.value)};
      SmallVector<Via, 4> newVias(currentVias.begin(), currentVias.end());
      if (viaTile)
        newVias.push_back({viaTile, ingressPort, nextPort});
      SmallVector<Operation *, 8> newUsedOps(currentUsedOps.begin(),
                                             currentUsedOps.end());
      newUsedOps.append(nextPortMaskValue.ops.begin(),
                        nextPortMaskValue.ops.end());
      auto nextConnection = getConnectionThroughWire(switchOp, nextPort);

      if (!nextConnection) {
        // The switchbox drives nextPort but no wire continues from it (an array
        // edge or an off-fabric consumer). Under partial recovery this output
        // port is itself the flow's endpoint.
        if (keepPartialFlows)
          partialEndpoints.push_back(
              {{switchOp, nextPort}, newMaskValue, newVias, newUsedOps});
        continue;
      }

      worklist.push_back({*nextConnection, newMaskValue, newVias, newUsedOps});
    }
    return worklist;
  }

public:
  // Follow the single upstream wire out of an interconnect input port.
  std::optional<PortConnection> upstreamOf(Operation *op, Port port) const {
    return getConnectionThroughWire(op, port);
  }

  // Whether `port` is driven (is the destination of a connect or masterset)
  // inside the given interconnect op.
  bool drivesPort(Operation *op, Port port) const {
    Region *r = nullptr;
    if (auto sb = dyn_cast<SwitchboxOp>(op))
      r = &sb.getConnections();
    else if (auto sm = dyn_cast<ShimMuxOp>(op))
      r = &sm.getConnections();
    if (!r)
      return false;
    Block &b = r->front();
    for (auto connectOp : b.getOps<ConnectOp>())
      if (connectOp.destPort() == port)
        return true;
    for (auto masterSetOp : b.getOps<MasterSetOp>())
      if (masterSetOp.destPort() == port)
        return true;
    return false;
  }

  // Traverse forward from switchbox/shim-mux input ports, collecting the
  // endpoints each stream reaches: flow-endpoint tiles, or -- under
  // keepPartialFlows -- driven interconnect ports that have no onward wire.
  std::vector<PacketConnection> traverse(std::vector<PacketConnection> worklist,
                                         bool keepPartialFlows) const {
    std::vector<PacketConnection> connectedTiles;
    while (!worklist.empty()) {
      PacketConnection t = worklist.back();
      worklist.pop_back();
      Operation *other = t.portConnection.op;
      Port otherPort = t.portConnection.port;
      MaskValue maskValue = t.mv;
      if (other && other->hasTrait<IsFlowEndPoint>()) {
        // If we got to a tile, then add it to the result.
        connectedTiles.push_back(t);
        continue;
      }
      Region *connections = nullptr;
      if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(other))
        connections = &switchOp.getConnections();
      else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(other))
        connections = &switchOp.getConnections();
      if (!connections) {
        LLVM_DEBUG(llvm::dbgs()
                   << "*** Connection Terminated at unknown operation: ");
        LLVM_DEBUG(other->dump());
        continue;
      }
      // A fan-out port ends the current linear section: the flow terminates at
      // this input, the fan-out switchbox stays explicit (its ops are not
      // lifted), and every output it drives becomes a new section source.
      if (splitFanouts &&
          fanoutIngresses.count({other, encodePort(otherPort)})) {
        connectedTiles.push_back(t);
        for (PortMaskValue &pmv :
             getConnectionsThroughSwitchbox(*connections, otherPort))
          fanoutEgressSeeds.insert({other, encodePort(pmv.port)});
        continue;
      }
      std::vector<PortMaskValue> nextPortMaskValues =
          getConnectionsThroughSwitchbox(*connections, otherPort);
      std::vector<PacketConnection> partialEndpoints;
      std::vector<PacketConnection> newWorkList = maskSwitchboxConnections(
          other, otherPort, t.vias, t.usedOps, nextPortMaskValues, maskValue,
          keepPartialFlows, partialEndpoints);
      worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
      connectedTiles.insert(connectedTiles.end(), partialEndpoints.begin(),
                            partialEndpoints.end());
      if (!nextPortMaskValues.empty() && newWorkList.empty() &&
          partialEndpoints.empty()) {
        // No rule matched some incoming packet.  This is likely a
        // configuration error.
        LLVM_DEBUG(llvm::dbgs() << "No rule matched incoming packet here: ");
        LLVM_DEBUG(other->dump());
      }
    }
    return connectedTiles;
  }

  // Get the tiles connected to the given tile, starting from the given
  // output port of the tile.  This is 1:N relationship because each
  // switchbox can broadcast.
  std::vector<PacketConnection> getConnectedTiles(TileOp tileOp, Port port,
                                                  bool keepPartialFlows) const {
    LLVM_DEBUG(llvm::dbgs()
               << "getConnectedTile(" << stringifyWireBundle(port.bundle) << " "
               << port.channel << ")");
    LLVM_DEBUG(tileOp.dump());
    // Traverse from the tile to its connected switchbox.
    auto t = getConnectionThroughWire(tileOp.getOperation(), port);
    // If there is no wire to traverse, then just return no connection
    if (!t)
      return {};
    return traverse({PacketConnection{*t, {0, 0}, {}, {}}}, keepPartialFlows);
  }

  // Get the endpoints reached by a stream that enters the fabric at the given
  // input port of an interconnect (switchbox / shim-mux).  Used to lift flows
  // whose source is not a core or DMA.
  std::vector<PacketConnection>
  getConnectedTilesFromInput(Operation *switchOp, Port inputPort,
                             bool keepPartialFlows) const {
    return traverse(
        {PacketConnection{PortConnection{switchOp, inputPort}, {0, 0}, {}, {}}},
        keepPartialFlows);
  }

  // Get the endpoints reached by a stream leaving the given output port of a
  // fan-out interconnect: cross the outgoing wire and traverse the next linear
  // section.
  std::vector<PacketConnection>
  getConnectedTilesFromEgress(Operation *switchOp, Port egressPort,
                              bool keepPartialFlows) const {
    auto t = getConnectionThroughWire(switchOp, egressPort);
    if (!t)
      return {};
    return traverse({PacketConnection{*t, {0, 0}, {}, {}}}, keepPartialFlows);
  }
};

// Identifies a flow by its two endpoints -- tile coordinates plus port -- and
// the packet claim it carries, using kCircuitFlow for circuit-switched flows.
// Coordinates rather than SSA values, so flows written against different
// aie.tile ops for the same tile still compare equal.
static constexpr int kCircuitFlow = -1;
using FlowKey = std::tuple<int, int, int, int, int, int, int, int, int, int>;
using FlowKeySet = std::set<FlowKey>;

// Returns nullopt when either endpoint's coordinates are unknown, in which
// case the caller cannot tell the flow apart from any other and must not
// dedupe it away.
static std::optional<FlowKey> tryGetFlowKey(Value srcTileValue, Port srcPort,
                                            Value destTileValue, Port destPort,
                                            int packetID, int packetMask) {
  auto coords = [](Value v) -> std::optional<std::pair<int, int>> {
    auto tile = llvm::dyn_cast_or_null<TileLike>(v.getDefiningOp());
    if (!tile)
      return std::nullopt;
    std::optional<int> col = tile.tryGetCol();
    std::optional<int> row = tile.tryGetRow();
    if (!col || !row)
      return std::nullopt;
    return std::make_pair(*col, *row);
  };
  std::optional<std::pair<int, int>> src = coords(srcTileValue);
  std::optional<std::pair<int, int>> dest = coords(destTileValue);
  if (!src || !dest)
    return std::nullopt;
  return FlowKey{src->first,
                 src->second,
                 static_cast<int>(srcPort.bundle),
                 srcPort.channel,
                 dest->first,
                 dest->second,
                 static_cast<int>(destPort.bundle),
                 destPort.channel,
                 packetID,
                 packetMask};
}

// The tile an endpoint of a lifted flow belongs to.
//
// aie.flow names tiles: --aie-create-pathfinder-flows casts both endpoints of
// every flow to TileOp. A tile element such as an aie.shim_dma therefore has to
// resolve to its owning tile, or the flow this pass emits crashes the router it
// is meant to feed. TileElement covers every element that can end a flow, and
// Interconnect extends it, so a switchbox or shim-mux port resolves the same
// way.
static Value resolveEndpointTile(Operation *op) {
  if (isa<TileOp>(op))
    return op->getResult(0);
  if (auto element = dyn_cast<TileElement>(op))
    return element.getTile();
  return nullptr;
}

static void emitFlows(OpBuilder &rewriter, Location loc, Value srcTile,
                      WireBundle srcBundle, int srcChannel,
                      const std::vector<PacketConnection> &endpoints,
                      bool emitVias, bool dropIntraTile, int idMask,
                      FlowKeySet &seen, llvm::DenseSet<Operation *> &consumed) {
  for (const PacketConnection &c : endpoints) {
    Operation *destOp = c.portConnection.op;
    Port destPort = c.portConnection.port;
    MaskValue maskValue = c.mv;
    Value destTile = resolveEndpointTile(destOp);
    if (!destTile)
      continue;
    // Never silently drop configuration. Control overlays (in any mode) are
    // kept materialized, and when pinning so is any section carrying op state a
    // lifted flow cannot rebuild -- keep_pkt_header on a non-destination hop,
    // or any discardable annotation (present or future) the rebuild does not
    // carry. Leaving it materialized keeps it faithful and visible rather than
    // dropping it and having the round-trip silently diverge.
    bool keepMaterialized = false;
    for (Operation *op : c.usedOps) {
      Operation *rulesParent =
          isa_and_nonnull<PacketRuleOp>(op) ? op->getParentOp() : nullptr;
      if (op->hasAttr("is_ctrl_pkt_overlay") ||
          (rulesParent && rulesParent->hasAttr("is_ctrl_pkt_overlay"))) {
        keepMaterialized = true;
      } else if (emitVias) {
        if (!op->getDiscardableAttrDictionary().empty() ||
            (rulesParent &&
             !rulesParent->getDiscardableAttrDictionary().empty()))
          keepMaterialized = true;
        else if (auto ms = dyn_cast<MasterSetOp>(op))
          keepMaterialized = ms.getKeepPktHeaderAttr() &&
                             (ms.getDestBundle() != destPort.bundle ||
                              ms.getDestChannel() != destPort.channel);
      }
      if (keepMaterialized)
        break;
    }
    if (keepMaterialized)
      continue;
    // In pinning mode a path realized only by shim-mux connections has no
    // switchbox transit and therefore no via to pin it; split-flow-vias cannot
    // materialize it and re-routing cannot honor it. Leave that shim-mux
    // configuration in place rather than lifting an unroutable flow.
    if (emitVias && c.vias.empty())
      continue;
    // A packet endpoint becomes a logical packet flow, and the pass reclaims
    // its physical configuration.
    //
    // The rules decide this, not the mask: a rule may carry mask 0, which
    // accepts every id, and a circuit path accumulates the same mask 0.
    bool isPacket = llvm::any_of(c.usedOps, [](Operation *op) {
      return isa_and_nonnull<PacketRuleOp>(op);
    });
    // The traversal reaches one endpoint more than once when a broadcast
    // re-converges on it, and the seeds can arrive at one route from different
    // directions. Those repeats describe one flow, and DeviceOp::verify rejects
    // a flow declared twice.
    std::optional<FlowKey> key =
        tryGetFlowKey(srcTile, {srcBundle, srcChannel}, destTile, destPort,
                      isPacket ? maskValue.value : kCircuitFlow,
                      isPacket ? maskValue.mask : 0);
    if (key && !seen.insert(*key).second)
      continue;
    if (isPacket && !emitVias) {
      for (Operation *op : c.usedOps)
        if (op)
          consumed.insert(op);
      // The lowering stores keep_pkt_header on the master set that drives the
      // destination. Carry it onto the recovered flow, or re-lowering strips
      // the header and misroutes the packet downstream.
      BoolAttr keepPktHeader;
      for (Operation *op : c.usedOps)
        if (auto ms = dyn_cast_or_null<MasterSetOp>(op))
          if (ms.getDestBundle() == destPort.bundle &&
              ms.getDestChannel() == destPort.channel)
            keepPktHeader = ms.getKeepPktHeaderAttr();
      // The rules along the path accept every id that agrees with value on the
      // bits mask selects. Which of those a running design sends is not
      // decidable here, so state the pair and let routing rebuild the same
      // rules.
      //
      // A full-width mask selects one id, which the id states on its own.
      // Leaving the attribute off keeps such a flow mergeable with the others
      // that share its route, the way routing found it.
      IntegerAttr mask;
      if (maskValue.mask != idMask)
        mask = rewriter.getI8IntegerAttr(maskValue.mask);
      auto flowOp = PacketFlowOp::create(
          rewriter, loc, rewriter.getI8IntegerAttr(maskValue.value), mask,
          keepPktHeader, BoolAttr(), mlir::ValueRange{}, nullptr, nullptr,
          nullptr, nullptr);
      PacketFlowOp::ensureTerminator(flowOp.getPorts(), rewriter, loc);
      OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(flowOp.getPorts().front().getTerminator());
      PacketSourceOp::create(rewriter, loc, srcTile, srcBundle, srcChannel);
      PacketDestOp::create(rewriter, loc, destTile, destPort.bundle,
                           destPort.channel);
      rewriter.restoreInsertionPoint(ip);
      continue;
    }
    // A section that lifts no interconnect configuration carries no routing:
    // it is either a fan-out output feeding its own tile with no hop, or the
    // physical wire between two retained fan nodes. The wire is implicit and
    // the switchboxes at both ends stay explicit, so emitting a flow for it
    // would only fight their fixed connections on re-routing.
    if (c.usedOps.empty())
      continue;
    // A flow seeded from a fabric-entry / fan-out output that terminates on a
    // real endpoint of the same tile never leaves that tile -- it is a dead or
    // orphan interconnect connection (e.g. an undriven shim-mux write path),
    // not a routable flow, so do not lift it.
    if (dropIntraTile && destTile == srcTile && isa<TileOp>(destOp))
      continue;
    // Every interconnect op on this endpoint's path is lifted into the flow.
    for (Operation *op : c.usedOps)
      if (op)
        consumed.insert(op);
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<Value> viaTiles;
    SmallVector<int32_t> ingressBundles, ingressChannels, egressBundles,
        egressChannels;
    if (emitVias)
      for (const Via &via : c.vias) {
        viaTiles.push_back(via.tile);
        ingressBundles.push_back(static_cast<int32_t>(via.ingress.bundle));
        ingressChannels.push_back(via.ingress.channel);
        egressBundles.push_back(static_cast<int32_t>(via.egress.bundle));
        egressChannels.push_back(via.egress.channel);
      }
    auto ib = viaTiles.empty() ? nullptr
                               : DenseI32ArrayAttr::get(ctx, ingressBundles);
    auto ic = viaTiles.empty() ? nullptr
                               : DenseI32ArrayAttr::get(ctx, ingressChannels);
    auto eb =
        viaTiles.empty() ? nullptr : DenseI32ArrayAttr::get(ctx, egressBundles);
    auto ec = viaTiles.empty() ? nullptr
                               : DenseI32ArrayAttr::get(ctx, egressChannels);
    if (maskValue.mask == 0) {
      FlowOp::create(rewriter, loc, srcTile, srcBundle, srcChannel, destTile,
                     destPort.bundle, destPort.channel, viaTiles, ib, ic, eb,
                     ec);
    } else {
      // A straight-line packet section: pin its route with vias and record the
      // ID mask so --aie-split-flow-vias can rebuild the switchbox rules. The
      // fan-out/fan-in nodes at either end stay materialized in place.
      // keep_pkt_header lives on the master set that drives the section's
      // destination; carry it on the lifted flow so re-lowering restores it
      // (dropping it would strip the header and misroute the packet
      // downstream).
      BoolAttr keepPktHeader;
      for (Operation *op : c.usedOps)
        if (auto ms = dyn_cast_or_null<MasterSetOp>(op))
          if (ms.getDestBundle() == destPort.bundle &&
              ms.getDestChannel() == destPort.channel)
            keepPktHeader = ms.getKeepPktHeaderAttr();
      auto flowOp = PacketFlowOp::create(
          rewriter, loc, rewriter.getI8IntegerAttr(maskValue.value),
          rewriter.getI8IntegerAttr(maskValue.mask), keepPktHeader, BoolAttr(),
          viaTiles, ib, ic, eb, ec);
      PacketFlowOp::ensureTerminator(flowOp.getPorts(), rewriter, loc);
      OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(flowOp.getPorts().front().getTerminator());
      PacketSourceOp::create(rewriter, loc, srcTile, srcBundle, srcChannel);
      PacketDestOp::create(rewriter, loc, destTile, destPort.bundle,
                           destPort.channel);
      rewriter.restoreInsertionPoint(ip);
    }
  }
}

static void findFlowsFrom(TileOp op, ConnectivityAnalysis &analysis,
                          OpBuilder &rewriter, bool keepPartialFlows,
                          bool emitVias, int idMask, FlowKeySet &seen,
                          llvm::DenseSet<Operation *> &consumed) {
  Operation *Op = op.getOperation();
  rewriter.setInsertionPoint(Op->getBlock()->getTerminator());

  std::vector bundles = {WireBundle::Core, WireBundle::DMA};
  for (WireBundle bundle : bundles) {
    LLVM_DEBUG(llvm::dbgs()
               << op << stringifyWireBundle(bundle) << " has "
               << op.getNumSourceConnections(bundle) << " Connections\n");
    for (size_t i = 0; i < op.getNumSourceConnections(bundle); i++) {
      std::vector<PacketConnection> tiles =
          analysis.getConnectedTiles(op, {bundle, (int)i}, keepPartialFlows);
      LLVM_DEBUG(llvm::dbgs() << tiles.size() << " Flows\n");
      emitFlows(rewriter, Op->getLoc(), Op->getResult(0), bundle, (int)i, tiles,
                emitVias, /*dropIntraTile=*/false, idMask, seen, consumed);
    }
  }
}

// Lift flows whose source is not a core/DMA.  Seed from every switchbox /
// shim-mux input port that drives a connection but is not itself driven by an
// upstream interconnect connection.  Ports driven by a core/DMA tile are
// already covered by findFlowsFrom; ports driven by an upstream interconnect
// are covered by that interconnect's own source, so both are skipped here to
// avoid emitting a flow twice.
static void findFlowsFromInterconnect(Operation *switchOp,
                                      ConnectivityAnalysis &analysis,
                                      OpBuilder &rewriter,
                                      bool keepPartialFlows, bool emitVias,
                                      int idMask, FlowKeySet &seen,
                                      llvm::DenseSet<Operation *> &consumed) {
  Region *connections = nullptr;
  if (auto sb = dyn_cast<SwitchboxOp>(switchOp))
    connections = &sb.getConnections();
  else if (auto sm = dyn_cast<ShimMuxOp>(switchOp))
    connections = &sm.getConnections();
  if (!connections)
    return;
  rewriter.setInsertionPoint(switchOp->getBlock()->getTerminator());

  // Distinct source ports of this interconnect's connections and packet rules.
  llvm::SmallVector<Port, 8> sourcePorts;
  auto addPort = [&](Port p) {
    if (!llvm::is_contained(sourcePorts, p))
      sourcePorts.push_back(p);
  };
  Block &b = connections->front();
  for (auto connectOp : b.getOps<ConnectOp>())
    addPort(connectOp.sourcePort());
  for (auto rulesOp : b.getOps<PacketRulesOp>())
    addPort(rulesOp.sourcePort());

  for (Port p : sourcePorts) {
    Value srcTile;
    WireBundle srcBundle = p.bundle;
    int srcChannel = p.channel;
    if (auto up = analysis.upstreamOf(switchOp, p)) {
      Operation *upOp = up->op;
      Port upPort = up->port;
      if (upOp && upOp->hasTrait<IsFlowEndPoint>()) {
        // Driven by a tile.  Core/DMA sources are handled by findFlowsFrom;
        // recover the remaining tile-source bundles (e.g. PLIO) from here.
        if (upPort.bundle == WireBundle::Core ||
            upPort.bundle == WireBundle::DMA)
          continue;
        srcTile = upOp->getResult(0);
        srcBundle = upPort.bundle;
        srcChannel = upPort.channel;
      } else if (analysis.drivesPort(upOp, upPort)) {
        // Mid-chain: an upstream interconnect drives this port.
        continue;
      } else {
        // Wire exists but nothing drives it: this input is a fabric entry.
        srcTile = resolveEndpointTile(switchOp);
      }
    } else {
      // No upstream wire: this input is a fabric entry (array edge).
      srcTile = resolveEndpointTile(switchOp);
    }
    if (!srcTile)
      continue;
    std::vector<PacketConnection> tiles =
        analysis.getConnectedTilesFromInput(switchOp, p, keepPartialFlows);
    emitFlows(rewriter, switchOp->getLoc(), srcTile, srcBundle, srcChannel,
              tiles, emitVias, /*dropIntraTile=*/true, idMask, seen, consumed);
  }
}

struct AIEFindFlowsPass
    : public xilinx::AIE::impl::AIEFindFlowsBase<AIEFindFlowsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  // An interconnect whose region holds nothing but its terminator carries no
  // configuration and can be dropped.
  static bool isEmptyInterconnect(Region &connections) {
    Block &b = connections.front();
    return b.getOps<ConnectOp>().empty() && b.getOps<PacketRulesOp>().empty() &&
           b.getOps<MasterSetOp>().empty() && b.getOps<AMSelOp>().empty();
  }

  void runOnOperation() override {

    DeviceOp d = getOperation();
    ConnectivityAnalysis analysis(d);
    d.getTargetModel().validate();
    if (clEmitVias)
      analysis.enableFanoutSplitting();

    // In pinning mode the packet routes are re-emitted as straight-line section
    // flows carrying vias; the original packet_flow ops they replace are
    // recorded now so they can be dropped once the sections are in place.
    SmallVector<PacketFlowOp> originalPacketFlows;
    if (clEmitVias)
      for (auto pf : d.getOps<PacketFlowOp>())
        originalPacketFlows.push_back(pf);

    // Widest mask the target's packet ids can carry.
    const int idMask =
        (1 << llvm::Log2_32_Ceil(d.getTargetModel().getMaxPacketId() + 1)) - 1;

    llvm::DenseSet<Operation *> consumed;
    FlowKeySet seen;
    OpBuilder builder = OpBuilder::atBlockTerminator(d.getBody());
    for (auto tile : d.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder, clKeepPartialFlows, clEmitVias,
                    idMask, seen, consumed);
    }
    // Lift flows whose source is not a core/DMA (transit fills, packet routing
    // steered at runtime, PLIO/edge entries) directly from the interconnect.
    if (clKeepPartialFlows) {
      for (auto switchOp : d.getOps<SwitchboxOp>())
        findFlowsFromInterconnect(switchOp, analysis, builder,
                                  clKeepPartialFlows, clEmitVias, idMask, seen,
                                  consumed);
      for (auto shimMuxOp : d.getOps<ShimMuxOp>())
        findFlowsFromInterconnect(shimMuxOp, analysis, builder,
                                  clKeepPartialFlows, clEmitVias, idMask, seen,
                                  consumed);
    }

    // Each output of a fan-out node starts a new linear section; drain those
    // seeds to a fixpoint (a branch may reach further fan-outs).  The fan-out
    // nodes themselves stay explicit.
    if (clEmitVias) {
      builder.setInsertionPoint(d.getBody()->getTerminator());
      llvm::DenseSet<std::pair<Operation *, int>> seeded;
      bool progress = true;
      while (progress) {
        progress = false;
        llvm::SmallVector<std::pair<Operation *, int>> snapshot(
            analysis.getFanoutEgressSeeds().begin(),
            analysis.getFanoutEgressSeeds().end());
        for (auto &seed : snapshot) {
          if (!seeded.insert(seed).second)
            continue;
          progress = true;
          Operation *switchOp = seed.first;
          Port egress = ConnectivityAnalysis::decodePort(seed.second);
          Value srcTile = resolveEndpointTile(switchOp);
          if (!srcTile)
            continue;
          std::vector<PacketConnection> tiles =
              analysis.getConnectedTilesFromEgress(switchOp, egress,
                                                   clKeepPartialFlows);
          emitFlows(builder, switchOp->getLoc(), srcTile, egress.bundle,
                    egress.channel, tiles, clEmitVias, /*dropIntraTile=*/true,
                    idMask, seen, consumed);
        }
      }
    }

    // The lifted section flows now describe the packet routes; drop the
    // originals so they are not routed a second time.
    for (PacketFlowOp pf : originalPacketFlows)
      pf.erase();

    if (!clRemoveLifted)
      return;

    // Every recovered flow makes the interconnect ops it traversed redundant;
    // drop exactly those, leaving any configuration that could not be lifted
    // (e.g. an unreachable connect) in place.  Leaf routing ops go first;
    // amsels that lose all users and now-empty rule containers follow.
    for (Operation *op : consumed)
      if (isa<ConnectOp, PacketRuleOp, MasterSetOp>(op))
        op->erase();
    auto cleanupInterconnect = [](Region &connections) {
      for (auto amselOp :
           llvm::make_early_inc_range(connections.getOps<AMSelOp>()))
        if (amselOp.use_empty())
          amselOp.erase();
      for (auto rulesOp :
           llvm::make_early_inc_range(connections.getOps<PacketRulesOp>()))
        if (rulesOp.getRules().front().getOps<PacketRuleOp>().empty())
          rulesOp.erase();
    };
    for (auto switchOp : d.getOps<SwitchboxOp>())
      cleanupInterconnect(switchOp.getConnections());
    for (auto shimMuxOp : d.getOps<ShimMuxOp>())
      cleanupInterconnect(shimMuxOp.getConnections());

    // An interconnect still holding configuration was not lifted: a control
    // overlay, a connect this pass could not reach, or -- under
    // keep-partial-flows=false -- a partial route left in place. Routing
    // regenerates a wire only for the flows it routes, so a wire reaching such
    // an interconnect has to stay, or that configuration ends up unreachable.
    llvm::DenseSet<Operation *> retained;
    auto retainNonEmpty = [&](Operation *op, Region &connections) {
      if (!isEmptyInterconnect(connections))
        retained.insert(op);
    };
    for (auto switchOp : d.getOps<SwitchboxOp>())
      retainNonEmpty(switchOp, switchOp.getConnections());
    for (auto shimMuxOp : d.getOps<ShimMuxOp>())
      retainNonEmpty(shimMuxOp, shimMuxOp.getConnections());

    llvm::DenseSet<Operation *> wiredToRetained;
    for (auto wireOp : llvm::make_early_inc_range(d.getOps<WireOp>())) {
      Operation *src = wireOp.getSource().getDefiningOp();
      Operation *dst = wireOp.getDest().getDefiningOp();
      if (!retained.contains(src) && !retained.contains(dst)) {
        wireOp.erase();
        continue;
      }
      wiredToRetained.insert(src);
      wiredToRetained.insert(dst);
    }

    // A surviving wire keeps its operands alive, so an empty interconnect one
    // of them names outlives its configuration.
    auto eraseIfUnused = [&](Operation *op, Region &connections) {
      return isEmptyInterconnect(connections) && !wiredToRetained.contains(op);
    };
    for (auto switchOp : llvm::make_early_inc_range(d.getOps<SwitchboxOp>()))
      if (eraseIfUnused(switchOp, switchOp.getConnections()))
        switchOp.erase();
    for (auto shimMuxOp : llvm::make_early_inc_range(d.getOps<ShimMuxOp>()))
      if (eraseIfUnused(shimMuxOp, shimMuxOp.getConnections()))
        shimMuxOp.erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEFindFlowsPass() {
  return std::make_unique<AIEFindFlowsPass>();
}
