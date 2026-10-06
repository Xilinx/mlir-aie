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
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"

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
  // packet PacketRuleOp + MasterSetOp + AMSelOp). Erased after lifting.
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

using CircuitFanoutSeed = struct CircuitFanoutSeed {
  Operation *op;
  Port ingress;
  Port egress;
  llvm::SmallVector<Operation *, 3> ops;
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
  mutable llvm::SmallVector<CircuitFanoutSeed, 8> circuitFanoutSeeds;

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

  // Consumed and cleared by the pass.
  llvm::DenseSet<std::pair<Operation *, int>> &getFanoutEgressSeeds() {
    return fanoutEgressSeeds;
  }
  llvm::SmallVector<CircuitFanoutSeed, 8> &getCircuitFanoutSeeds() {
    return circuitFanoutSeeds;
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
        // In rule order, which is slot order.
        for (auto ruleOp : connectOp.getRules().front().getOps<PacketRuleOp>())
          for (auto masterSetOp : b.getOps<MasterSetOp>())
            for (Value amsel : masterSetOp.getAmsels()) {
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
    // A slave port hands a packet to the first rule that matches its id, so
    // once a rule matches every id that can arrive, later rules get none.
    Operation *takenBy = nullptr;
    // Only switchbox hops become vias; the shim-mux is not stream-switch
    // configuration and is regenerated by routing.
    Value viaTile;
    if (auto sb = dyn_cast<SwitchboxOp>(switchOp))
      viaTile = sb.getTile();
    for (auto &nextPortMaskValue : nextPortMaskValues) {
      Port nextPort = nextPortMaskValue.port;
      MaskValue nextMaskValue = nextPortMaskValue.mv;
      Operation *rule = nextPortMaskValue.ops.front();
      if (!isa<PacketRuleOp>(rule)) {
        rule = nullptr;
      }
      if (takenBy && rule != takenBy) {
        continue;
      }
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
      if (rule && (nextMaskValue.mask & ~maskValue.mask) == 0) {
        takenBy = rule;
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
        if (keepPartialFlows) {
          partialEndpoints.push_back(
              {{switchOp, nextPort}, newMaskValue, newVias, newUsedOps});
        }
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
    if (auto sb = dyn_cast<SwitchboxOp>(op)) {
      r = &sb.getConnections();
    } else if (auto sm = dyn_cast<ShimMuxOp>(op)) {
      r = &sm.getConnections();
    }
    if (!r) {
      return false;
    }
    Block &b = r->front();
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.destPort() == port) {
        return true;
      }
    }
    for (auto masterSetOp : b.getOps<MasterSetOp>()) {
      if (masterSetOp.destPort() == port) {
        return true;
      }
    }
    return false;
  }

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
      if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(other)) {
        connections = &switchOp.getConnections();
      } else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(other)) {
        connections = &switchOp.getConnections();
      }
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
             getConnectionsThroughSwitchbox(*connections, otherPort)) {
          if (pmv.mv.mask == 0) {
            auto sameSeed = [&](const CircuitFanoutSeed &seed) {
              return seed.op == other && seed.ingress == otherPort &&
                     seed.egress == pmv.port;
            };
            if (!llvm::any_of(circuitFanoutSeeds, sameSeed))
              circuitFanoutSeeds.push_back(
                  {other, otherPort, pmv.port, pmv.ops});
          } else {
            fanoutEgressSeeds.insert({other, encodePort(pmv.port)});
          }
        }
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
  // `claim` selects the packet ids followed; the default follows them all.
  std::vector<PacketConnection>
  getConnectedTiles(TileOp tileOp, Port port, bool keepPartialFlows,
                    MaskValue claim = {0, 0}) const {
    LLVM_DEBUG(llvm::dbgs()
               << "getConnectedTile(" << stringifyWireBundle(port.bundle) << " "
               << port.channel << ")");
    LLVM_DEBUG(tileOp.dump());
    // Traverse from the tile to its connected switchbox.
    auto t = getConnectionThroughWire(tileOp.getOperation(), port);
    // If there is no wire to traverse, then just return no connection
    if (!t) {
      return {};
    }
  return traverse({PacketConnection{*t, claim, {}, {}}}, keepPartialFlows);
  }

  std::vector<PacketConnection>
  getConnectedTilesFromInput(Operation *switchOp, Port inputPort,
                             bool keepPartialFlows,
                             MaskValue claim = {0, 0}) const {
    return traverse(
                              {PacketConnection{PortConnection{switchOp, inputPort}, claim, {}, {}}},
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

  std::vector<PacketConnection>
  getConnectedTilesFromCircuitFanout(const CircuitFanoutSeed &seed,
                                     bool keepPartialFlows) const {
    auto connection = getConnectionThroughWire(seed.op, seed.egress);
    if (!connection)
      return {};
    Value tile = cast<SwitchboxOp>(seed.op).getTile();
    SmallVector<Operation *, 8> usedOps(seed.ops.begin(), seed.ops.end());
    PacketConnection branch{*connection,
                            {0, 0},
                            {{tile, seed.ingress, seed.egress}},
                            std::move(usedOps)};
    return traverse({std::move(branch)}, keepPartialFlows);
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

// aie.flow names tiles, so an endpoint that is an element of a tile, such as an
// aie.shim_dma, has to resolve to the tile that owns it.
static Value resolveEndpointTile(Operation *op) {
  if (isa<TileOp>(op)) {
    return op->getResult(0);
  }
  if (auto element = dyn_cast<TileElement>(op)) {
    return element.getTile();
  }
  return nullptr;
}

// The interconnect ops the lifted flows make redundant, and those a route left
// materialized still needs. An op can be on both, such as a shim-mux connect
// that user flows and the control overlay share, and then it stays. Each lifted
// packet flow is listed with the ops it traversed.
struct LiftedOps {
  llvm::DenseSet<Operation *> consumed;
  llvm::DenseSet<Operation *> kept;
  std::vector<std::pair<PacketFlowOp, SmallVector<Operation *, 8>>> packetFlows;
};

static void emitFlows(OpBuilder &rewriter, Location loc, Value srcTile,
                      WireBundle srcBundle, int srcChannel,
                      const std::vector<PacketConnection> &endpoints,
                      bool emitVias, bool dropIntraTile, int idMask,
                      FlowKeySet &seen,
                      LiftedOps &lifted) {
  AIEDialect::IsCtrlPktOverlayAttrHelper overlay(rewriter.getContext());
  AIEDialect::PriorityRouteAttrHelper prioritizedRule(rewriter.getContext());
  for (const PacketConnection &c : endpoints) {
    Operation *destOp = c.portConnection.op;
    Port destPort = c.portConnection.port;
    MaskValue maskValue = c.mv;
    Value destTile = resolveEndpointTile(destOp);
    if (!destTile) {
      continue;
    }
    // A control overlay stays materialized: its configuration carries the
    // is_ctrl_pkt_overlay marker, and a lifted flow cannot rebuild it. When
    // pinning, so does a hop carrying state the rebuild drops: keep_pkt_header
    // away from the destination, or a discardable attribute.
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
    if (keepMaterialized) {
      for (Operation *op : c.usedOps)
        if (op)
          lifted.kept.insert(op);
      continue;
    }
    // A path realized only by shim-mux connections crosses no switchbox, so it
    // has no via to pin it and re-routing cannot reproduce it.
    if (emitVias && c.vias.empty()) {
      continue;
    }
    // A packet endpoint becomes a logical packet flow, and the pass erases
    // the lowered physical configuration.
    bool isPacket = llvm::any_of(c.usedOps, [](Operation *op) {
      return isa_and_nonnull<PacketRuleOp>(op);
    });
    // The same endpoint is reached twice when a broadcast splits and rejoins,
    // and the tile, switchbox and shim-mux seeds can each reach one route.
    // DeviceOp::verify rejects a flow declared twice.
    std::optional<FlowKey> key =
        tryGetFlowKey(srcTile, {srcBundle, srcChannel}, destTile, destPort,
                      isPacket ? maskValue.value : kCircuitFlow,
                      isPacket ? maskValue.mask : 0);
    if (key && !seen.insert(*key).second) {
      continue;
    }
    if (isPacket && !emitVias) {
      for (Operation *op : c.usedOps) {
        if (op) {
          lifted.consumed.insert(op);
        }
      }
      // The lowering stores keep_pkt_header on the master set that drives the
      // destination. A priority_route flow marks that master set too, and the
      // rule it starts by at its source. Either can be shared, the master set
      // with other sources and the rule with the source's other ids, but a
      // flow that has both is a priority_route one or comes from the same
      // source as one, which routes the same.
      BoolAttr keepPktHeader, priorityRoute;
      bool markedDest = false, markedSource = false;
      for (Operation *op : c.usedOps) {
        if (auto ms = dyn_cast_or_null<MasterSetOp>(op)) {
          if (ms.getDestBundle() == destPort.bundle &&
              ms.getDestChannel() == destPort.channel) {
            keepPktHeader = ms.getKeepPktHeaderAttr();
            markedDest = overlay.isAttrPresent(ms);
          }
        }
        if (isa_and_nonnull<PacketRuleOp>(op) &&
            prioritizedRule.isAttrPresent(op)) {
          markedSource = true;
        }
      }
      if (markedDest && markedSource) {
        priorityRoute = rewriter.getBoolAttr(true);
      }
      // The rules along the path accept every id that agrees with value on the
      // bits mask selects. Which of those a running design sends is not
      // decidable here, so state the pair and let routing rebuild the same
      // rules.
      //
      // A full-width mask selects exactly one id; in flow IR we express this
      // by leaving off the mask. The router may merge such flows.
      IntegerAttr mask;
      if (maskValue.mask != idMask) {
        mask = rewriter.getI8IntegerAttr(maskValue.mask);
      }
      auto flowOp = PacketFlowOp::create(
          rewriter, loc, rewriter.getI8IntegerAttr(maskValue.value), mask,
          keepPktHeader, priorityRoute, mlir::ValueRange{}, nullptr, nullptr,
          nullptr, nullptr);
      PacketFlowOp::ensureTerminator(flowOp.getPorts(), rewriter, loc);
      OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(flowOp.getPorts().front().getTerminator());
      PacketSourceOp::create(rewriter, loc, srcTile, srcBundle, srcChannel);
      PacketDestOp::create(rewriter, loc, destTile, destPort.bundle,
                           destPort.channel);
      rewriter.restoreInsertionPoint(ip);
      lifted.packetFlows.emplace_back(flowOp, c.usedOps);
      continue;
    }
    // No switchbox configuration stands between the endpoints, so there is no
    // flow to lift.
    if (c.usedOps.empty()) {
      continue;
    }
    // A fabric entry or fan-out output that ends on its own tile is an orphan
    // connection, such as an undriven shim-mux write path.
    if (dropIntraTile && destTile == srcTile && isa<TileOp>(destOp)) {
      continue;
    }
    for (Operation *op : c.usedOps) {
      if (op) {
        lifted.consumed.insert(op);
      }
    }
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
      // The mask lets --aie-split-flow-vias rebuild the switchbox rules.
      // keep_pkt_header lives on the master set that drives the section's
      // destination. Carry it onto the recovered flow.
      BoolAttr keepPktHeader;
      for (Operation *op : c.usedOps) {
        if (auto ms = dyn_cast_or_null<MasterSetOp>(op)) {
          if (ms.getDestBundle() == destPort.bundle &&
              ms.getDestChannel() == destPort.channel) {
            keepPktHeader = ms.getKeepPktHeaderAttr();
          }
        }
      }
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

// Where rules on a port overlap, which ids each one takes depends on the rules
// before it, and a traversal of cubes cannot follow that. So each packet id is
// traced on its own, and the ids that reach the same destinations are
// described by disjoint cubes. Two flows from one source then claim either the
// same ids or none in common.
static std::vector<PacketConnection> exactPacketEndpoints(
    std::vector<PacketConnection> endpoints, int idMask,
    llvm::function_ref<std::vector<PacketConnection>(MaskValue)> trace) {
  auto isPacket = [](const PacketConnection &c) {
    return llvm::any_of(c.usedOps, [](Operation *op) {
      return isa_and_nonnull<PacketRuleOp>(op);
    });
  };
  if (llvm::none_of(endpoints, isPacket)) {
    return endpoints;
  }
  std::vector<PacketConnection> result;
  llvm::copy_if(endpoints, std::back_inserter(result),
                [&](const PacketConnection &c) { return !isPacket(c); });

  auto sameDest = [](const PortConnection &a, const PortConnection &b) {
    return a.op == b.op && a.port == b.port;
  };
  SmallVector<PortConnection> dests;
  auto destIndex = [&](const PortConnection &d) {
    auto *it = llvm::find_if(
        dests, [&](const PortConnection &e) { return sameDest(d, e); });
    if (it != dests.end()) {
      return static_cast<unsigned>(it - dests.begin());
    }
    dests.push_back(d);
    return static_cast<unsigned>(dests.size() - 1);
  };
  for (const PacketConnection &c : endpoints) {
    if (isPacket(c)) {
      destIndex(c.portConnection);
    }
  }

  // The destinations each id reaches, and the ops it passes to reach each.
  std::map<SmallVector<unsigned>, SmallVector<int>> idsByDests;
  std::map<std::pair<int, unsigned>, SmallVector<Operation *, 8>> opsTo;
  for (int id = 0; id <= idMask; ++id) {
    SmallVector<unsigned> reached;
    for (const PacketConnection &c : trace({idMask, id})) {
      if (!isPacket(c)) {
        continue;
      }
      unsigned d = destIndex(c.portConnection);
      if (!llvm::is_contained(reached, d)) {
        reached.push_back(d);
      }
      SmallVector<Operation *, 8> &ops = opsTo[{id, d}];
      for (Operation *op : c.usedOps) {
        if (!llvm::is_contained(ops, op)) {
          ops.push_back(op);
        }
      }
    }
    if (!reached.empty()) {
      llvm::sort(reached);
      idsByDests[reached].push_back(id);
    }
  }

  // Largest cube first, each inside the ids not yet described.
  auto disjointCubes = [&](ArrayRef<int> ids) {
    llvm::SmallBitVector left(idMask + 1);
    for (int id : ids) {
      left.set(id);
    }
    SmallVector<MaskValue> cubes;
    while (left.any()) {
      MaskValue best = {idMask, left.find_first()};
      int bestSize = 1;
      for (int mask = 0; mask <= idMask; ++mask) {
        int size = 1 << llvm::popcount(static_cast<unsigned>(idMask & ~mask));
        if (size <= bestSize) {
          continue;
        }
        for (int value = 0; value <= idMask && size > bestSize; ++value) {
          if ((value & mask) != value) {
            continue;
          }
          bool inside = true;
          for (int id = value; id <= idMask && inside; ++id) {
            inside = (id & mask) != value || left.test(id);
          }
          if (inside) {
            best = {mask, value};
            bestSize = size;
          }
        }
      }
      for (int id = 0; id <= idMask; ++id) {
        if ((id & best.mask) == best.value) {
          left.reset(id);
        }
      }
      cubes.push_back(best);
    }
    return cubes;
  };

  // In id order, which is the order of each group's lowest id.
  SmallVector<std::pair<const SmallVector<unsigned> *, ArrayRef<int>>> groups;
  for (const auto &[reached, ids] : idsByDests) {
    groups.push_back({&reached, ids});
  }
  llvm::sort(groups, [](const auto &a, const auto &b) {
    return a.second.front() < b.second.front();
  });
  for (const auto &[reached, ids] : groups) {
    SmallVector<MaskValue> cubes = disjointCubes(ids);
    for (unsigned d : *reached) {
      for (MaskValue cube : cubes) {
        SmallVector<Operation *, 8> ops;
        for (int id : ids) {
          if ((id & cube.mask) != cube.value) {
            continue;
          }
          for (Operation *op : opsTo[{id, d}]) {
            if (!llvm::is_contained(ops, op)) {
              ops.push_back(op);
            }
          }
        }
        result.push_back({dests[d], cube, ops});
      }
    }
  }
  return result;
}

static void findFlowsFrom(TileOp op, ConnectivityAnalysis &analysis,
                          OpBuilder &rewriter, bool keepPartialFlows,
                          bool emitVias, int idMask, FlowKeySet &seen,
                          LiftedOps &lifted) {
  Operation *Op = op.getOperation();
  rewriter.setInsertionPoint(Op->getBlock()->getTerminator());

  std::vector bundles = {WireBundle::Core, WireBundle::DMA};
  for (WireBundle bundle : bundles) {
    LLVM_DEBUG(llvm::dbgs()
               << op << stringifyWireBundle(bundle) << " has "
               << op.getNumSourceConnections(bundle) << " Connections\n");
    for (size_t i = 0; i < op.getNumSourceConnections(bundle); i++) {
      Port port = {bundle, (int)i};
      std::vector<PacketConnection> tiles = exactPacketEndpoints(
          analysis.getConnectedTiles(op, port, keepPartialFlows), idMask,
          [&](MaskValue claim) {
            return analysis.getConnectedTiles(op, port, keepPartialFlows,
                                              claim);
          });
      LLVM_DEBUG(llvm::dbgs() << tiles.size() << " Flows\n");
      emitFlows(rewriter, Op->getLoc(), Op->getResult(0), bundle, (int)i, tiles,
            emitVias, /*dropIntraTile=*/false, idMask, seen, lifted);
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
                                      LiftedOps &lifted) {
  Region *connections = nullptr;
  if (auto sb = dyn_cast<SwitchboxOp>(switchOp)) {
    connections = &sb.getConnections();
  } else if (auto sm = dyn_cast<ShimMuxOp>(switchOp)) {
    connections = &sm.getConnections();
  }
  if (!connections) {
    return;
  }
  rewriter.setInsertionPoint(switchOp->getBlock()->getTerminator());

  // Distinct source ports of this interconnect's connections and packet rules.
  llvm::SmallVector<Port, 8> sourcePorts;
  auto addPort = [&](Port p) {
    if (!llvm::is_contained(sourcePorts, p)) {
      sourcePorts.push_back(p);
    }
  };
  Block &b = connections->front();
  for (auto connectOp : b.getOps<ConnectOp>()) {
    addPort(connectOp.sourcePort());
  }
  for (auto rulesOp : b.getOps<PacketRulesOp>()) {
    addPort(rulesOp.sourcePort());
  }

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
            upPort.bundle == WireBundle::DMA) {
          continue;
        }
        srcTile = resolveEndpointTile(upOp);
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
    if (!srcTile) {
      continue;
    }
    std::vector<PacketConnection> tiles = exactPacketEndpoints(
        analysis.getConnectedTilesFromInput(switchOp, p, keepPartialFlows),
        idMask, [&](MaskValue claim) {
          return analysis.getConnectedTilesFromInput(switchOp, p,
                                                     keepPartialFlows, claim);
        });
    emitFlows(rewriter, switchOp->getLoc(), srcTile, srcBundle, srcChannel,
          tiles, emitVias, /*dropIntraTile=*/true, idMask, seen, lifted);
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

    LiftedOps lifted;
    FlowKeySet seen;
    OpBuilder builder = OpBuilder::atBlockTerminator(d.getBody());
    for (auto tile : d.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder, clKeepPartialFlows, clEmitVias,
            idMask, seen, lifted);
    }
    // Lift flows whose source is not a core/DMA (transit fills, packet routing
    // steered at runtime, PLIO/edge entries) directly from the interconnect.
    if (clKeepPartialFlows) {
      for (auto switchOp : d.getOps<SwitchboxOp>()) {
        findFlowsFromInterconnect(switchOp, analysis, builder,
                                  clKeepPartialFlows, clEmitVias, idMask, seen,
                                  lifted);
      }
      for (auto shimMuxOp : d.getOps<ShimMuxOp>()) {
        findFlowsFromInterconnect(shimMuxOp, analysis, builder,
                                  clKeepPartialFlows, clEmitVias, idMask, seen,
                                  lifted);
      }
    }

    // Each output of a fan-out node starts a new linear section; drain those
    // seeds to a fixpoint (a branch may reach further fan-outs).  The fan-out
    // nodes themselves stay explicit.
    if (clEmitVias) {
      builder.setInsertionPoint(d.getBody()->getTerminator());
      for (size_t i = 0; i < analysis.getCircuitFanoutSeeds().size(); ++i) {
        CircuitFanoutSeed seed = analysis.getCircuitFanoutSeeds()[i];
        Value srcTile = resolveEndpointTile(seed.op);
        if (!srcTile)
          continue;
        std::vector<PacketConnection> tiles =
            analysis.getConnectedTilesFromCircuitFanout(seed,
                                                        clKeepPartialFlows);
        emitFlows(builder, seed.op->getLoc(), srcTile, seed.ingress.bundle,
                  seed.ingress.channel, tiles, clEmitVias,
                  /*dropIntraTile=*/false, idMask, seen, consumed);
      }
      llvm::DenseSet<std::pair<Operation *, int>> seeded;
      bool progress = true;
      while (progress) {
        progress = false;
        llvm::SmallVector<std::pair<Operation *, int>> snapshot(
            analysis.getFanoutEgressSeeds().begin(),
            analysis.getFanoutEgressSeeds().end());
        for (auto &seed : snapshot) {
          if (!seeded.insert(seed).second) {
            continue;
          }
          progress = true;
          Operation *switchOp = seed.first;
          Port egress = ConnectivityAnalysis::decodePort(seed.second);
          Value srcTile = resolveEndpointTile(switchOp);
          if (!srcTile) {
            continue;
          }
          std::vector<PacketConnection> tiles =
              analysis.getConnectedTilesFromEgress(switchOp, egress,
                                                   clKeepPartialFlows);
          emitFlows(builder, switchOp->getLoc(), srcTile, egress.bundle,
                    egress.channel, tiles, clEmitVias, /*dropIntraTile=*/true,
                    idMask, seen, lifted);
        }
      }
    }

    // The lifted section flows now describe the packet routes; drop the
    // originals so they are not routed a second time.
    for (PacketFlowOp pf : originalPacketFlows) {
      pf.erase();
    }

    if (!clRemoveLifted) {
      return;
    }

    // Every recovered flow makes the interconnect ops it traversed redundant;
    // drop exactly those, leaving any configuration that could not be lifted
    // (e.g. an unreachable connect) in place.
    // A packet flow that shares a rule with a route left materialized stays
    // materialized too: the rule claims its ids, so a rerouted copy of the flow
    // could not claim them again.
    for (bool changed = true; changed;) {
      changed = false;
      for (auto &[flow, usedOps] : lifted.packetFlows) {
        if (!flow || llvm::none_of(usedOps, [&](Operation *op) {
              return isa_and_nonnull<PacketRuleOp>(op) &&
                     lifted.kept.contains(op);
            })) {
          continue;
        }
        for (Operation *op : usedOps) {
          if (op) {
            lifted.kept.insert(op);
          }
        }
        flow.erase();
        flow = nullptr;
        changed = true;
      }
    }
    for (Operation *op : lifted.consumed) {
      if (isa<ConnectOp, PacketRuleOp, MasterSetOp>(op) &&
          !lifted.kept.contains(op)) {
        op->erase();
      }
    }
    auto cleanupInterconnect = [](Region &connections) {
      for (auto amselOp :
           llvm::make_early_inc_range(connections.getOps<AMSelOp>())) {
        if (amselOp.use_empty()) {
          amselOp.erase();
        }
      }
      for (auto rulesOp :
           llvm::make_early_inc_range(connections.getOps<PacketRulesOp>())) {
        if (rulesOp.getRules().front().getOps<PacketRuleOp>().empty()) {
          rulesOp.erase();
        }
      }
    };
    for (auto switchOp : d.getOps<SwitchboxOp>()) {
      cleanupInterconnect(switchOp.getConnections());
    }
    for (auto shimMuxOp : d.getOps<ShimMuxOp>()) {
      cleanupInterconnect(shimMuxOp.getConnections());
    }

    // Routing regenerates a wire only for the flows it routes, so a wire that
    // reaches an interconnect still holding configuration has to stay, or that
    // configuration ends up unreachable.
    llvm::DenseSet<Operation *> retained;
    auto retainNonEmpty = [&](Operation *op, Region &connections) {
      if (!isEmptyInterconnect(connections)) {
        retained.insert(op);
      }
    };
    for (auto switchOp : d.getOps<SwitchboxOp>()) {
      retainNonEmpty(switchOp, switchOp.getConnections());
    }
    for (auto shimMuxOp : d.getOps<ShimMuxOp>()) {
      retainNonEmpty(shimMuxOp, shimMuxOp.getConnections());
    }

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
    for (auto switchOp : llvm::make_early_inc_range(d.getOps<SwitchboxOp>())) {
      if (eraseIfUnused(switchOp, switchOp.getConnections())) {
        switchOp.erase();
      }
    }
    for (auto shimMuxOp : llvm::make_early_inc_range(d.getOps<ShimMuxOp>())) {
      if (eraseIfUnused(shimMuxOp, shimMuxOp.getConnections())) {
        shimMuxOp.erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEFindFlowsPass() {
  return std::make_unique<AIEFindFlowsPass>();
}
