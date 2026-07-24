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

#include "llvm/ADT/STLExtras.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEFINDFLOWS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-find-flows"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

typedef struct MaskValue {
  int mask;
  int value;
} MaskValue;

typedef struct PortConnection {
  Operation *op;
  Port port;
} PortConnection;

typedef struct PortMaskValue {
  Port port;
  MaskValue mv;
} PortMaskValue;

// One intermediate stop of a flow: the tile whose switchbox the stream passes
// through, and the ingress/egress ports it uses there.
typedef struct Via {
  Value tile;
  Port ingress;
  Port egress;
} Via;

typedef struct PacketConnection {
  PortConnection portConnection;
  MaskValue mv;
  llvm::SmallVector<Via, 4> vias;
} PacketConnection;

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
        portSet.push_back({connectOp.destPort(), maskValue});
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
                portSet.push_back({masterSetOp.destPort(), maskValue});
              }
            }
      }
    }
    return portSet;
  }

  std::vector<PacketConnection>
  maskSwitchboxConnections(Operation *switchOp, Port ingressPort,
                           ArrayRef<Via> currentVias,
                           std::vector<PortMaskValue> nextPortMaskValues,
                           MaskValue maskValue, bool keepPartialFlows,
                           std::vector<PacketConnection> &partialEndpoints)
      const {
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
      auto nextConnection = getConnectionThroughWire(switchOp, nextPort);

      if (!nextConnection) {
        // The switchbox drives nextPort but no wire continues from it (an array
        // edge or an off-fabric consumer). Under partial recovery this output
        // port is itself the flow's endpoint.
        if (keepPartialFlows)
          partialEndpoints.push_back({{switchOp, nextPort}, newMaskValue,
                                      newVias});
        continue;
      }

      worklist.push_back({*nextConnection, newMaskValue, newVias});
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
          other, otherPort, t.vias, nextPortMaskValues, maskValue,
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
    return traverse({{*t, {0, 0}}}, keepPartialFlows);
  }

  // Get the endpoints reached by a stream that enters the fabric at the given
  // input port of an interconnect (switchbox / shim-mux).  Used to lift flows
  // whose source is not a core or DMA.
  std::vector<PacketConnection>
  getConnectedTilesFromInput(Operation *switchOp, Port inputPort,
                             bool keepPartialFlows) const {
    return traverse({{PortConnection{switchOp, inputPort}, {0, 0}}},
                    keepPartialFlows);
  }
};

// The endpoint of a lifted flow is a tile: a flow-endpoint tile directly, or
// the tile owning a directional switchbox / shim-mux port (partial flow).
static Value resolveEndpointTile(Operation *op) {
  if (isa<TileOp>(op))
    return op->getResult(0);
  if (auto sb = dyn_cast<SwitchboxOp>(op))
    return sb.getTile();
  if (auto sm = dyn_cast<ShimMuxOp>(op))
    return sm.getTile();
  return nullptr;
}

static void emitFlows(OpBuilder &rewriter, Location loc, Value srcTile,
                      WireBundle srcBundle, int srcChannel,
                      const std::vector<PacketConnection> &endpoints,
                      bool emitVias) {
  for (const PacketConnection &c : endpoints) {
    Operation *destOp = c.portConnection.op;
    Port destPort = c.portConnection.port;
    MaskValue maskValue = c.mv;
    Value destTile = resolveEndpointTile(destOp);
    if (!destTile)
      continue;
    if (maskValue.mask == 0) {
      if (emitVias && !c.vias.empty()) {
        SmallVector<Value> viaTiles;
        SmallVector<int32_t> ingressBundles, ingressChannels, egressBundles,
            egressChannels;
        for (const Via &via : c.vias) {
          viaTiles.push_back(via.tile);
          ingressBundles.push_back(static_cast<int32_t>(via.ingress.bundle));
          ingressChannels.push_back(via.ingress.channel);
          egressBundles.push_back(static_cast<int32_t>(via.egress.bundle));
          egressChannels.push_back(via.egress.channel);
        }
        MLIRContext *ctx = rewriter.getContext();
        FlowOp::create(rewriter, loc, srcTile, srcBundle, srcChannel, destTile,
                       destPort.bundle, destPort.channel, viaTiles,
                       DenseI32ArrayAttr::get(ctx, ingressBundles),
                       DenseI32ArrayAttr::get(ctx, ingressChannels),
                       DenseI32ArrayAttr::get(ctx, egressBundles),
                       DenseI32ArrayAttr::get(ctx, egressChannels));
      } else {
        FlowOp::create(rewriter, loc, srcTile, srcBundle, srcChannel, destTile,
                       destPort.bundle, destPort.channel);
      }
    } else {
      // Packet flows have no via representation; emit the plain packet flow.
      auto flowOp =
          PacketFlowOp::create(rewriter, loc, maskValue.value, nullptr, nullptr);
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
                          bool emitVias) {
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
                emitVias);
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
                                      OpBuilder &rewriter, bool keepPartialFlows,
                                      bool emitVias) {
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
              tiles, emitVias);
  }
}

struct AIEFindFlowsPass
    : public xilinx::AIE::impl::AIEFindFlowsBase<AIEFindFlowsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }
  void runOnOperation() override {

    DeviceOp d = getOperation();
    ConnectivityAnalysis analysis(d);
    d.getTargetModel().validate();

    OpBuilder builder = OpBuilder::atBlockTerminator(d.getBody());
    for (auto tile : d.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder, clKeepPartialFlows, clEmitVias);
    }
    // Lift flows whose source is not a core/DMA (transit fills, packet routing
    // steered at runtime, PLIO/edge entries) directly from the interconnect.
    if (clKeepPartialFlows) {
      for (auto switchOp : d.getOps<SwitchboxOp>())
        findFlowsFromInterconnect(switchOp, analysis, builder,
                                  clKeepPartialFlows, clEmitVias);
      for (auto shimMuxOp : d.getOps<ShimMuxOp>())
        findFlowsFromInterconnect(shimMuxOp, analysis, builder,
                                  clKeepPartialFlows, clEmitVias);
    }

    // With vias, each recovered flow fully encodes the switchbox configuration
    // it traversed, so the physical interconnect is redundant: drop it, leaving
    // a purely logical, re-routable representation.  Wires reference the
    // interconnect results, so erase them first.
    if (clEmitVias) {
      for (auto wireOp : llvm::make_early_inc_range(d.getOps<WireOp>()))
        wireOp.erase();
      for (auto switchOp : llvm::make_early_inc_range(d.getOps<SwitchboxOp>()))
        switchOp.erase();
      for (auto shimMuxOp : llvm::make_early_inc_range(d.getOps<ShimMuxOp>()))
        shimMuxOp.erase();
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEFindFlowsPass() {
  return std::make_unique<AIEFindFlowsPass>();
}
