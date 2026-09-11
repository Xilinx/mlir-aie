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

using PacketConnection = struct PacketConnection {
  PortConnection portConnection;
  MaskValue mv;
  // Interconnect ops traversed on the way to this endpoint.
  llvm::SmallVector<Operation *, 8> usedOps;
};

class ConnectivityAnalysis {
  DeviceOp &device;

public:
  ConnectivityAnalysis(DeviceOp &d) : device(d) {}

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
      Operation *switchOp, ArrayRef<Operation *> currentUsedOps,
      const std::vector<PortMaskValue> &nextPortMaskValues, MaskValue maskValue,
      bool keepPartialFlows,
      std::vector<PacketConnection> &partialEndpoints) const {
    std::vector<PacketConnection> worklist;
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
              {{switchOp, nextPort}, newMaskValue, newUsedOps});
        continue;
      }

      worklist.push_back({*nextConnection, newMaskValue, newUsedOps});
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
      std::vector<PortMaskValue> nextPortMaskValues =
          getConnectionsThroughSwitchbox(*connections, otherPort);
      std::vector<PacketConnection> partialEndpoints;
      std::vector<PacketConnection> newWorkList = maskSwitchboxConnections(
          other, t.usedOps, nextPortMaskValues, maskValue, keepPartialFlows,
          partialEndpoints);
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
    return traverse({PacketConnection{*t, {0, 0}, {}}}, keepPartialFlows);
  }

  // Get the endpoints reached by a stream that enters the fabric at the given
  // input port of an interconnect (switchbox / shim-mux).  Used to lift flows
  // whose source is not a core or DMA.
  std::vector<PacketConnection>
  getConnectedTilesFromInput(Operation *switchOp, Port inputPort,
                             bool keepPartialFlows) const {
    return traverse(
        {PacketConnection{PortConnection{switchOp, inputPort}, {0, 0}, {}}},
        keepPartialFlows);
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
                      bool dropIntraTile, int idMask, FlowKeySet &seen,
                      llvm::DenseSet<Operation *> &consumed) {
  for (const PacketConnection &c : endpoints) {
    Operation *destOp = c.portConnection.op;
    Port destPort = c.portConnection.port;
    MaskValue maskValue = c.mv;
    Value destTile = resolveEndpointTile(destOp);
    if (!destTile)
      continue;
    // A control overlay stays materialized. Its switchbox configuration carries
    // the is_ctrl_pkt_overlay marker, and a lifted flow cannot rebuild it.
    bool keepMaterialized = false;
    for (Operation *op : c.usedOps) {
      Operation *rulesParent =
          isa_and_nonnull<PacketRuleOp>(op) ? op->getParentOp() : nullptr;
      if (op->hasAttr("is_ctrl_pkt_overlay") ||
          (rulesParent && rulesParent->hasAttr("is_ctrl_pkt_overlay"))) {
        keepMaterialized = true;
        break;
      }
    }
    if (keepMaterialized)
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
    // re-converges on it, and the three seeds can arrive at one route from
    // different directions. Those repeats describe one flow, and
    // DeviceOp::verify rejects a flow declared twice.
    std::optional<FlowKey> key =
        tryGetFlowKey(srcTile, {srcBundle, srcChannel}, destTile, destPort,
                      isPacket ? maskValue.value : kCircuitFlow,
                      isPacket ? maskValue.mask : 0);
    if (key && !seen.insert(*key).second)
      continue;
    if (isPacket) {
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
          keepPktHeader, BoolAttr());
      PacketFlowOp::ensureTerminator(flowOp.getPorts(), rewriter, loc);
      OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(flowOp.getPorts().front().getTerminator());
      PacketSourceOp::create(rewriter, loc, srcTile, srcBundle, srcChannel);
      PacketDestOp::create(rewriter, loc, destTile, destPort.bundle,
                           destPort.channel);
      rewriter.restoreInsertionPoint(ip);
      continue;
    }
    // A section that lifts no interconnect configuration carries no routing: it
    // is the physical wire between two retained nodes. The wire is implicit and
    // the switchboxes at both ends stay explicit, so a flow emitted for it
    // would only fight their fixed connections on re-routing.
    if (c.usedOps.empty())
      continue;
    // A flow seeded from a fabric entry that terminates on a real endpoint of
    // the same tile never leaves that tile. It is a dead or orphan interconnect
    // connection, such as an undriven shim-mux write path, so the pass leaves
    // it alone.
    if (dropIntraTile && destTile == srcTile && isa<TileOp>(destOp))
      continue;
    // Every interconnect op on this endpoint's path is lifted into the flow.
    for (Operation *op : c.usedOps)
      if (op)
        consumed.insert(op);
    FlowOp::create(rewriter, loc, srcTile, srcBundle, srcChannel, destTile,
                   destPort.bundle, destPort.channel);
  }
}

static void findFlowsFrom(TileOp op, ConnectivityAnalysis &analysis,
                          OpBuilder &rewriter, bool keepPartialFlows,
                          int idMask, FlowKeySet &seen,
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
                /*dropIntraTile=*/false, idMask, seen, consumed);
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
                                      bool keepPartialFlows, int idMask,
                                      FlowKeySet &seen,
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
              tiles, /*dropIntraTile=*/true, idMask, seen, consumed);
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

    // Widest mask the target's packet ids can carry.
    const int idMask =
        (1 << llvm::Log2_32_Ceil(d.getTargetModel().getMaxPacketId() + 1)) - 1;

    llvm::DenseSet<Operation *> consumed;
    FlowKeySet seen;
    OpBuilder builder = OpBuilder::atBlockTerminator(d.getBody());
    for (auto tile : d.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder, clKeepPartialFlows, idMask, seen,
                    consumed);
    }
    // Lift flows whose source is not a core/DMA (transit fills, packet routing
    // steered at runtime, PLIO/edge entries) directly from the interconnect.
    if (clKeepPartialFlows) {
      for (auto switchOp : d.getOps<SwitchboxOp>())
        findFlowsFromInterconnect(switchOp, analysis, builder,
                                  clKeepPartialFlows, idMask, seen, consumed);
      for (auto shimMuxOp : d.getOps<ShimMuxOp>())
        findFlowsFromInterconnect(shimMuxOp, analysis, builder,
                                  clKeepPartialFlows, idMask, seen, consumed);
    }

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
