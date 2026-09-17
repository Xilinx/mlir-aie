//===- AIEPathfinder.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"
#include "d_ary_heap.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"

#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-pathfinder"

// Strong-attract heuristic: after a packet flow's destination routes, its edges
// are discounted by this factor so the flow's next destination reuses the same
// trunk when equal-cost. Not a tuned value.
static constexpr double kTrunkReuseDiscount = 0.001;

// Design-aware control freeze keeps control column-local. Control is packet-
// switched and shares a channel with data 32-way, so it must never leave its
// source column just to dodge data -- a cross-column (East/West) detour splits
// the control multicast's coherent spine into extra output ports, fragmenting
// its packet-rule cover. This penalty on every cross-column hop dominates any
// in-column data-avoid penalty (DESIGN_AVOID_PENALTY x column height), so
// control prefers sharing an in-column channel with data over an East/West
// escape; it is finite, so a column with no in-column path (e.g. no shim) can
// still fall back to a cross-column ingress. Seeded only in the design-aware
// capture (alongside the design field), so blind/off routing is untouched.
static constexpr double kControlCrossColumnPenalty = 1.0e6;

std::array<std::pair<TileID, Port>, 4> AIE::getCardinalNeighbors(TileID coords,
                                                                 int channel) {
  return {{{{coords.col, coords.row - 1}, {WireBundle::North, channel}},
           {{coords.col - 1, coords.row}, {WireBundle::East, channel}},
           {{coords.col, coords.row + 1}, {WireBundle::South, channel}},
           {{coords.col + 1, coords.row}, {WireBundle::West, channel}}}};
}

// Index of port `p` in `ports`, or -1 if absent.
static int portIndex(const std::vector<Port> &ports, Port p) {
  auto it = llvm::find(ports, p);
  return it == ports.end() ? -1
                           : static_cast<int>(std::distance(ports.begin(), it));
}

LogicalResult DynamicTileAnalysis::runAnalysis(DeviceOp &device,
                                               bool skipControlFlows,
                                               const DesignField *baseline) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin DynamicTileAnalysis Constructor---\n");
  // find the maxCol and maxRow
  maxCol = device.getTargetModel().columns();
  maxRow = device.getTargetModel().rows();

  pathfinder->initialize(maxCol, maxRow, device.getTargetModel());

  // Design-aware freeze: seed the persistent per-cell demand field before any
  // flow is routed. initialize() has built the full graph and buildRoutingGraph
  // (inside findPaths) only reads it, so the seed survives to updateDemand.
  if (baseline) {
    pathfinder->seedDesignDemand(*baseline);
    // Design-aware capture routes control only; keep it column-local so a
    // data-avoid detour never crosses columns and fragments control's cover.
    pathfinder->seedColumnLocalControl(kControlCrossColumnPenalty);
    // Route each control multicast farthest-first so it commits one coherent
    // trunk per column instead of weaving onto free channels cell-by-cell.
    pathfinder->setCoherentControlCapture(true);
  }

  // Consolidate a control multicast onto a shared trunk for any control-overlay
  // (reconfiguration) compile: the design-aware capture (baseline) or a device
  // carrying the generated control overlay (has_ctrl_pkt_overlay, set for
  // freeze on OR off). A plain, non-reconfiguration design has neither, so its
  // packet routing stays byte-identical to upstream.
  bool devHasCtrlPktOverlay = false;
  if (auto a = device->getAttrOfType<mlir::BoolAttr>("has_ctrl_pkt_overlay"))
    devHasCtrlPktOverlay = a.getValue();
  pathfinder->setControlOverlayRouting(baseline != nullptr ||
                                       devHasCtrlPktOverlay);

  // For each flow (circuit + packet) in the device, add it to pathfinder. Each
  // source can map to multiple different destinations (fanout). Control packet
  // flows to be routed (as prioritized routings). Then followed by normal
  // packet flows.
  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    bool priorityFlow = pktFlowOp.getPriorityRoute().value_or(false);
    // Design-aware freeze demand capture routes config DATA only: skip the
    // control (priority_route) packet flows entirely.
    if (skipControlFlows && priorityFlow)
      continue;
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    SmallVector<std::pair<TileID, Port>, 4> sources;
    // Pass 1: collect all sources (order-independent; supports fan-in).
    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        auto srcTile = cast<TileOp>(pktSource.getTile().getDefiningOp());
        sources.push_back(
            {{srcTile.colIndex(), srcTile.rowIndex()}, pktSource.port()});
      }
    }
    if (sources.empty())
      return pktFlowOp.emitOpError("packet_flow has no packet_source");

    // Pass 2: add a flow from every source to every destination so
    // fan-in topologies are routed (not just the last source).
    for (Operation &Op : b.getOperations()) {
      if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
        auto dstTile = cast<TileOp>(pktDest.getTile().getDefiningOp());
        Port dstPort = pktDest.port();
        TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
        for (auto &[srcCoords, srcPort] : sources) {
          LLVM_DEBUG(llvm::dbgs()
                     << "\tAdding Packet Flow: (" << srcCoords.col << ", "
                     << srcCoords.row << ")"
                     << stringifyWireBundle(srcPort.bundle) << srcPort.channel
                     << " -> (" << dstCoords.col << ", " << dstCoords.row << ")"
                     << stringifyWireBundle(dstPort.bundle) << dstPort.channel
                     << "\n");
          pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort,
                              pktFlowOp.IDInt(), priorityFlow);
        }
      }
    }

    // Under freeze, AIEFreezeControlFabric annotated this control flow with its
    // captured canonical route. Decode it from THIS device's own IR (never
    // shared state, so parallel per-device findPaths stays correct) and pin the
    // flow so findPaths replays it instead of re-routing, freezing control
    // against data-driven drift. Absent the annotation (non-freeze) this is a
    // no-op and the routing is byte-identical.
    if (auto attr = pktFlowOp->getAttr(kPinnedRouteAttr))
      for (auto &[pinSrc, pinnedRoute] : decodePinnedRoutes(attr))
        pathfinder->pinRoute(pinSrc, pinnedRoute);
  }

  // Add circuit flows.
  for (FlowOp flowOp : device.getOps<FlowOp>()) {
    TileOp srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileOp dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    Port srcPort = {flowOp.getSourceBundle(), flowOp.getSourceChannel()};
    Port dstPort = {flowOp.getDestBundle(), flowOp.getDestChannel()};
    LLVM_DEBUG(llvm::dbgs()
               << "\tAdding Flow: (" << srcCoords.col << ", " << srcCoords.row
               << ")" << stringifyWireBundle(srcPort.bundle) << srcPort.channel
               << " -> (" << dstCoords.col << ", " << dstCoords.row << ")"
               << stringifyWireBundle(dstPort.bundle) << dstPort.channel
               << "\n");
    pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort,
                        /*packetId=*/std::nullopt, /*isPriorityFlow=*/false);
  }

  // Canonicalize all flows after both packet and circuit flows are collected.
  pathfinder->sortFlows();

  // add existing connections so Pathfinder knows which resources are
  // available search all existing SwitchBoxOps for exising connections
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    if (!pathfinder->addFixedConnection(switchboxOp))
      return switchboxOp.emitOpError() << "Unable to add fixed connections";
  }

  // all flows are now populated, call the congestion-aware pathfinder
  // algorithm
  // check whether the pathfinder algorithm creates a legal routing
  if (auto maybeFlowSolutions = pathfinder->findPaths(maxIterations))
    flowSolutions = maybeFlowSolutions.value();
  else
    return device.emitError("Unable to find a legal routing");

  // initialize all flows as unprocessed to prep for rewrite
  for (const auto &[PathEndPoint, switchSetting] : flowSolutions) {
    processedFlows[PathEndPoint] = false;
  }

  // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
  for (auto tileOp : device.getOps<TileOp>()) {
    int col, row;
    col = tileOp.colIndex();
    row = tileOp.rowIndex();
    assert(coordToTile.count({col, row}) == 0);
    coordToTile[{col, row}] = tileOp;
  }
  for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
    int col = switchboxOp.colIndex();
    int row = switchboxOp.rowIndex();
    assert(coordToSwitchbox.count({col, row}) == 0);
    coordToSwitchbox[{col, row}] = switchboxOp;
  }
  for (auto shimmuxOp : device.getOps<ShimMuxOp>()) {
    int col = shimmuxOp.colIndex();
    int row = shimmuxOp.rowIndex();
    assert(coordToShimMux.count({col, row}) == 0);
    coordToShimMux[{col, row}] = shimmuxOp;
  }

  LLVM_DEBUG(llvm::dbgs() << "\t---End DynamicTileAnalysis Constructor---\n");
  return success();
}

TileOp DynamicTileAnalysis::getTile(OpBuilder &builder, int col, int row) {
  if (coordToTile.count({col, row})) {
    return coordToTile[{col, row}];
  }
  auto tileOp = TileOp::create(builder, builder.getUnknownLoc(), col, row);
  coordToTile[{col, row}] = tileOp;
  return tileOp;
}

TileOp DynamicTileAnalysis::getTile(OpBuilder &builder, const TileID &tileId) {
  return getTile(builder, tileId.col, tileId.row);
}

SwitchboxOp DynamicTileAnalysis::getSwitchbox(OpBuilder &builder, int col,
                                              int row) {
  assert(col >= 0);
  assert(row >= 0);
  if (coordToSwitchbox.count({col, row})) {
    return coordToSwitchbox[{col, row}];
  }
  auto switchboxOp = SwitchboxOp::create(builder, builder.getUnknownLoc(),
                                         getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToSwitchbox[{col, row}] = switchboxOp;
  return switchboxOp;
}

ShimMuxOp DynamicTileAnalysis::getShimMux(OpBuilder &builder, int col) {
  assert(col >= 0);
  int row = 0;
  if (coordToShimMux.count({col, row})) {
    return coordToShimMux[{col, row}];
  }
  assert(getTile(builder, col, row).isShimNOCorPLTile());
  auto switchboxOp = ShimMuxOp::create(builder, builder.getUnknownLoc(),
                                       getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToShimMux[{col, row}] = switchboxOp;
  return switchboxOp;
}

void Pathfinder::initialize(int maxCol, int maxRow,
                            const AIETargetModel &targetModel) {
  // Reset all state so a Pathfinder instance can be safely reused across
  // analyses/devices. In particular the dense-graph cache below must be
  // rebuilt for the new topology; leaving graphBuilt set would reuse stale
  // node IDs and adjacency.
  graph.clear();
  flows.clear();
  graphBuilt = false;
  nodeIds.clear();
  nodes.clear();
  adjacency.clear();
  distance.clear();
  indexInHeap.clear();
  colors.clear();
  preds.clear();
  predEdge.clear();
  pinnedRoutes.clear();

  std::map<WireBundle, int> maxChannels;
  auto intraconnect = [&](int col, int row) {
    TileID coords = {col, row};
    SwitchboxConnect sb = {coords};

    for (int i = 0, e = getMaxEnumValForWireBundle() + 1; i < e; ++i) {
      WireBundle bundle = symbolizeWireBundle(i).value();
      // get all ports into current switchbox
      int channels =
          targetModel.getNumSourceSwitchboxConnections(col, row, bundle);
      if (channels == 0 && targetModel.isShimNOCorPLTile(col, row)) {
        // wordaround for shimMux
        channels = targetModel.getNumSourceShimMuxConnections(col, row, bundle);
      }
      for (int channel = 0; channel < channels; channel++) {
        sb.srcPorts.push_back(Port{bundle, channel});
      }
      // get all ports out of current switchbox
      channels = targetModel.getNumDestSwitchboxConnections(col, row, bundle);
      if (channels == 0 && targetModel.isShimNOCorPLTile(col, row)) {
        // wordaround for shimMux
        channels = targetModel.getNumDestShimMuxConnections(col, row, bundle);
      }
      for (int channel = 0; channel < channels; channel++) {
        sb.dstPorts.push_back(Port{bundle, channel});
      }
      maxChannels[bundle] = channels;
    }
    // initialize matrices
    sb.resize();
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      for (size_t j = 0; j < sb.dstPorts.size(); j++) {
        auto &pIn = sb.srcPorts[i];
        auto &pOut = sb.dstPorts[j];
        if (targetModel.isLegalTileConnection(col, row, pIn.bundle, pIn.channel,
                                              pOut.bundle, pOut.channel))
          sb.connectivity[i][j] = Connectivity::AVAILABLE;
        else {
          sb.connectivity[i][j] = Connectivity::INVALID;
          if (targetModel.isShimNOCorPLTile(col, row)) {
            // wordaround for shimMux
            auto isBundleInList = [](WireBundle bundle,
                                     std::vector<WireBundle> bundles) {
              return llvm::find(bundles, bundle) != bundles.end();
            };
            const std::vector<WireBundle> bundles = {
                WireBundle::DMA, WireBundle::NOC, WireBundle::PLIO};
            if (isBundleInList(pIn.bundle, bundles) ||
                isBundleInList(pOut.bundle, bundles))
              sb.connectivity[i][j] = Connectivity::AVAILABLE;
          }
        }
      }
    }
    graph[std::make_pair(coords, coords)] = sb;
  };

  auto interconnect = [&](int col, int row, int targetCol, int targetRow,
                          WireBundle srcBundle, WireBundle dstBundle) {
    SwitchboxConnect sb = {{col, row}, {targetCol, targetRow}};
    for (int channel = 0; channel < maxChannels[srcBundle]; channel++) {
      sb.srcPorts.push_back(Port{srcBundle, channel});
      sb.dstPorts.push_back(Port{dstBundle, channel});
    }
    sb.resize();
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      sb.connectivity[i][i] = Connectivity::AVAILABLE;
    }
    graph[std::make_pair(TileID{col, row}, TileID{targetCol, targetRow})] = sb;
  };

  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      maxChannels.clear();
      // connections within the same switchbox
      intraconnect(col, row);

      // connections between switchboxes
      if (row > 0) {
        // from south to north
        interconnect(col, row, col, row - 1, WireBundle::South,
                     WireBundle::North);
      }
      if (row < maxRow) {
        // from north to south
        interconnect(col, row, col, row + 1, WireBundle::North,
                     WireBundle::South);
      }
      if (col > 0) {
        // from east to west
        interconnect(col, row, col - 1, row, WireBundle::West,
                     WireBundle::East);
      }
      if (col < maxCol) {
        // from west to east
        interconnect(col, row, col + 1, row, WireBundle::East,
                     WireBundle::West);
      }
    }
  }
}

// Add a flow from src to dst can have an arbitrary number of dst locations
// due to fanout.
void Pathfinder::addFlow(TileID srcCoords, Port srcPort, TileID dstCoords,
                         Port dstPort, std::optional<int> packetId,
                         bool isPriorityFlow) {
  // check if a flow with this source already exists
  for (auto &[_, prioritized, src, dsts, pid, dstPids] : flows) {
    if (src.coords == srcCoords && src.port == srcPort) {
      if (isPriorityFlow) {
        prioritized = true;
        dsts.emplace(dsts.begin(), dstCoords, dstPort);
        dstPids.emplace(dstPids.begin(), packetId);
      } else {
        dsts.emplace_back(dstCoords, dstPort);
        dstPids.emplace_back(packetId);
      }
      return;
    }
  }

  // Assign a group ID for packet flows
  // any overlapping in source/destination will lead to the same group ID
  // channel sharing will happen within the same group ID
  // for circuit flows, group ID is always -1, and no channel sharing
  int packetGroupId = -1;
  if (packetId.has_value()) {
    bool found = false;
    for (auto &[existingId, _prio, src, dsts, pid, dstPids] : flows) {
      if (src.coords == srcCoords && src.port == srcPort) {
        packetGroupId = existingId;
        found = true;
        break;
      }
      for (auto &dst : dsts) {
        if (dst.coords == dstCoords && dst.port == dstPort) {
          packetGroupId = existingId;
          found = true;
          break;
        }
      }
      packetGroupId = std::max(packetGroupId, existingId);
    }
    if (!found) {
      packetGroupId++;
    }
  }
  // If no existing flow was found with this source, create a new flow.
  flows.push_back(
      Flow{packetGroupId, isPriorityFlow, PathEndPoint{srcCoords, srcPort},
           std::vector<PathEndPoint>{PathEndPoint{dstCoords, dstPort}},
           packetId, std::vector<std::optional<int>>{packetId}});
}

// Sort flows to (1) get deterministic routing, and (2) perform routings on
// prioritized flows before others, for routing consistency on those flows.
void Pathfinder::sortFlows() {
  auto endpointLess = [](const PathEndPoint &lhs, const PathEndPoint &rhs) {
    return std::make_tuple(lhs.coords.col, lhs.coords.row,
                           getWireBundleAsInt(lhs.port.bundle),
                           lhs.port.channel) <
           std::make_tuple(rhs.coords.col, rhs.coords.row,
                           getWireBundleAsInt(rhs.port.bundle),
                           rhs.port.channel);
  };

  for (auto &flow : flows) {
    // dstPacketIds is index-parallel to dsts; sort them together (zipped) so
    // reordering dsts cannot desync which id belongs to which destination.
    std::vector<std::pair<PathEndPoint, std::optional<int>>> zipped;
    zipped.reserve(flow.dsts.size());
    for (size_t i = 0; i < flow.dsts.size(); i++)
      zipped.emplace_back(flow.dsts[i], flow.dstPacketIds[i]);
    std::sort(zipped.begin(), zipped.end(),
              [&](const auto &lhs, const auto &rhs) {
                return endpointLess(lhs.first, rhs.first);
              });
    for (size_t i = 0; i < zipped.size(); i++) {
      flow.dsts[i] = zipped[i].first;
      flow.dstPacketIds[i] = zipped[i].second;
    }
  }

  auto flowRank = [](const Flow &flow) {
    if (flow.isPriorityFlow)
      return 0;
    if (flow.packetGroupId >= 0)
      return 1;
    return 2;
  };
  std::sort(flows.begin(), flows.end(), [&](const Flow &lhs, const Flow &rhs) {
    if (flowRank(lhs) != flowRank(rhs))
      return flowRank(lhs) < flowRank(rhs);
    return endpointLess(lhs.src, rhs.src);
  });
}

// Keep track of connections already used in the AIE; Pathfinder algorithm
// will avoid using these.
bool Pathfinder::addFixedConnection(SwitchboxOp switchboxOp) {
  int col = switchboxOp.colIndex();
  int row = switchboxOp.rowIndex();
  TileID coords = {col, row};
  auto &sb = graph[std::make_pair(coords, coords)];
  for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
    bool found = false;
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      if (sb.srcPorts[i] != connectOp.sourcePort())
        continue;
      // A circuit-switched ConnectOp monopolizes its entire source port;
      // mark all connectivity[i][*] INVALID so the pathfinder cannot route
      // any packet flow through this source port.
      for (size_t j = 0; j < sb.dstPorts.size(); j++) {
        if (sb.dstPorts[j] == connectOp.destPort() &&
            sb.connectivity[i][j] == Connectivity::AVAILABLE)
          found = true;
        sb.connectivity[i][j] = Connectivity::INVALID;
      }
    }
    if (!found) {
      // ConnectOp references a (srcPort, dstPort) pair absent from or already
      // fully invalidated in the switchbox model; IR/graph mismatch.
      return false;
    }
  }
  return true;
}

// Register a pinned route for `src`; findPaths replays it verbatim.
void Pathfinder::pinRoute(const PathEndPoint &src,
                          const SwitchSettings &route) {
  pinnedRoutes[src] = route;
}

// Seed the per-cell demand field from a design-demand map (design-aware
// freeze). Iterate the (small) field, resolve each (srcCoords, dstCoords,
// srcPort, dstPort) key to its switchbox-connect cell, and assign the weight
// (assignment, not accumulation: a cell any config's data uses gets one fixed
// penalty). initialize() has already built `graph`, so the cells exist.
void Pathfinder::seedDesignDemand(const DesignField &field) {
  for (const auto &[key, weight] : field) {
    const auto &[srcCoords, dstCoords, srcPort, dstPort] = key;
    auto it = graph.find(std::make_pair(srcCoords, dstCoords));
    if (it == graph.end())
      continue;
    SwitchboxConnect &sb = it->second;
    int i = portIndex(sb.srcPorts, srcPort);
    int j = portIndex(sb.dstPorts, dstPort);
    if (i < 0 || j < 0)
      continue;
    sb.designDemand[i][j] = weight;
  }
}

// Add `penalty` to every cross-column (East/West) output cell so a design-aware
// control capture keeps control column-local (see kControlCrossColumnPenalty).
// Additive so it composes with the per-cell design field; finite so a column
// with no in-column path can still fall back to a cross-column hop.
void Pathfinder::seedColumnLocalControl(double penalty) {
  for (auto &[key, sb] : graph)
    for (size_t j = 0; j < sb.dstPorts.size(); j++)
      if (sb.dstPorts[j].bundle == WireBundle::East ||
          sb.dstPorts[j].bundle == WireBundle::West)
        for (size_t i = 0; i < sb.srcPorts.size(); i++)
          sb.designDemand[i][j] += penalty;
}

// See the header for the invariant. Two passes over `pinnedRoutes`: collect
// control's own source rows per master port, then INVALIDATE every OTHER source
// row on each control master column so no foreign (data) flow can egress a
// control master. Runs once, before buildRoutingGraph bakes connectivity into
// the dense graph. Empty pinnedRoutes -> no-op -> OFF byte-identical.
bool Pathfinder::reservePinnedControlMasters() {
  // PASS 1: allowedRows[{tile, masterPort}] = control's own source rows there.
  // Accumulate across ALL pinned routes so multi-source control fan-in onto one
  // master is preserved, not clobbered.
  std::map<std::pair<TileID, Port>, std::set<Port>> allowedRows;
  for (const auto &[src, route] : pinnedRoutes) {
    for (const auto &[coords, setting] : route)
      for (size_t k = 0; k < setting.srcs.size(); k++)
        allowedRows[{coords, setting.dsts[k]}].insert(setting.srcs[k]);
  }

  // PASS 2: on each control master column, reserve against non-control rows.
  for (const auto &[key, ctrlRows] : allowedRows) {
    const TileID &coords = key.first;
    const Port &master = key.second;
    auto it = graph.find({coords, coords});
    if (it == graph.end())
      continue;
    SwitchboxConnect &sb = it->second;
    int col = -1;
    for (size_t j = 0; j < sb.dstPorts.size(); j++)
      if (sb.dstPorts[j] == master) {
        col = static_cast<int>(j);
        break;
      }
    // A control master that is not a routable intra-switchbox dst (e.g. a
    // shim-mux-rewritten port) has no crossbar column for data to grab, so
    // there is nothing to reserve.
    if (col < 0)
      continue;
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      if (ctrlRows.count(sb.srcPorts[i])) {
        // Control's own row -- preserved so a co-sourced data leg can ride
        // control's shared slave. It must not already be monopolized by a pre-
        // placed circuit ConnectOp (addFixedConnection sets connectivity but
        // never usedCapacity), or the replay's accountEdge would double-use it.
        if (sb.connectivity[i][col] == Connectivity::INVALID)
          return false;
      } else {
        // Foreign (data) row -- may not egress this control master.
        sb.connectivity[i][col] = Connectivity::INVALID;
      }
    }
  }
  return true;
}

// Encode a captured route as a flat i32 array attribute. Layout:
//   [ srcCol, srcRow, srcBundle, srcChannel,
//     numTiles,
//     { tileCol, tileRow, numConns,
//       { srcBundle, srcChannel, dstBundle, dstChannel } * numConns } *
//       numTiles
//   ]
// WireBundles travel as their enum int (getWireBundleAsInt /
// symbolizeWireBundle).
mlir::Attribute AIE::encodePinnedRoute(mlir::MLIRContext *ctx, TileID srcCoords,
                                       Port srcPort,
                                       const SwitchSettings &settings) {
  SmallVector<int32_t> v;
  v.push_back(srcCoords.col);
  v.push_back(srcCoords.row);
  v.push_back(getWireBundleAsInt(srcPort.bundle));
  v.push_back(srcPort.channel);
  v.push_back(static_cast<int32_t>(settings.size()));
  for (const auto &[coords, setting] : settings) {
    assert(setting.srcs.size() == setting.dsts.size());
    v.push_back(coords.col);
    v.push_back(coords.row);
    v.push_back(static_cast<int32_t>(setting.srcs.size()));
    for (size_t k = 0; k < setting.srcs.size(); k++) {
      v.push_back(getWireBundleAsInt(setting.srcs[k].bundle));
      v.push_back(setting.srcs[k].channel);
      v.push_back(getWireBundleAsInt(setting.dsts[k].bundle));
      v.push_back(setting.dsts[k].channel);
    }
  }
  return mlir::DenseI32ArrayAttr::get(ctx, v);
}

std::vector<std::pair<PathEndPoint, SwitchSettings>>
AIE::decodePinnedRoutes(mlir::Attribute attr) {
  std::vector<std::pair<PathEndPoint, SwitchSettings>> out;
  auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr);
  if (!arr)
    return out;
  for (mlir::Attribute e : arr) {
    auto da = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(e);
    if (!da)
      continue;
    ArrayRef<int32_t> v = da.asArrayRef();
    size_t p = 0;
    bool bad = false;
    // Bounds-checked read. encodePinnedRoute always emits a well-formed array,
    // so a truncated/garbage attr is only reachable via hand-authored IR;
    // degrade by dropping the entry rather than indexing past the array.
    auto rd = [&]() -> int32_t {
      if (p >= v.size()) {
        bad = true;
        return 0;
      }
      return v[p++];
    };
    auto rdBundle = [&]() -> WireBundle {
      auto b = symbolizeWireBundle(rd());
      if (!b) {
        bad = true;
        return WireBundle::Core;
      }
      return *b;
    };
    TileID srcCoords = {rd(), rd()};
    Port srcPort = {rdBundle(), rd()};
    int numTiles = rd();
    SwitchSettings settings;
    for (int t = 0; t < numTiles && !bad; t++) {
      TileID coords = {rd(), rd()};
      int numConns = rd();
      SwitchSetting setting;
      for (int c = 0; c < numConns && !bad; c++) {
        Port s = {rdBundle(), rd()};
        Port d = {rdBundle(), rd()};
        setting.srcs.push_back(s);
        setting.dsts.push_back(d);
      }
      settings[coords] = setting;
    }
    if (bad)
      continue;
    out.emplace_back(PathEndPoint{srcCoords, srcPort}, settings);
  }
  return out;
}

static constexpr double INF = std::numeric_limits<double>::max();

namespace {
enum Color : int8_t { WHITE = 0, GRAY = 1, BLACK = 2 };
} // namespace

int Pathfinder::getOrAddNodeId(const PathEndPoint &pep) {
  auto it = nodeIds.find(pep);
  if (it != nodeIds.end())
    return it->second;
  int id = static_cast<int>(nodes.size());
  nodeIds[pep] = id;
  nodes.push_back(pep);
  return id;
}

// Build the dense integer node numbering and per-node adjacency once. The graph
// topology is fixed across congestion iterations (only the demand weights
// change), so this is computed a single time and the edges carry live pointers
// into `graph` for demand lookups. Edge order per node matches the legacy
// PathEndPoint-sorted channel order to preserve identical routing output.
void Pathfinder::buildRoutingGraph() {
  // Seed the dense node set with all flow endpoints (the only nodes Dijkstra is
  // ever started from or traced back to). Remaining nodes are discovered lazily
  // as edge destinations below, exactly mirroring the legacy on-demand channel
  // expansion in dijkstraShortestPaths.
  for (auto &f : flows) {
    getOrAddNodeId(f.src);
    for (auto &d : f.dsts)
      getOrAddNodeId(d);
  }

  // Process nodes by growing index; getOrAddNodeId() may append new nodes as
  // edge destinations are discovered, so re-read nodes.size() each iteration.
  for (size_t id = 0; id < nodes.size(); id++) {
    PathEndPoint src = nodes[id];
    // Collect destination PathEndPoints exactly as the legacy lazy channel
    // discovery did, then sort by PathEndPoint for deterministic edge order.
    std::vector<PathEndPoint> dests;
    auto intraIt = graph.find(std::make_pair(src.coords, src.coords));
    if (intraIt != graph.end()) {
      auto &sb = intraIt->second;
      for (size_t i = 0; i < sb.srcPorts.size(); i++)
        for (size_t j = 0; j < sb.dstPorts.size(); j++)
          if (sb.srcPorts[i] == src.port &&
              sb.connectivity[i][j] == Connectivity::AVAILABLE)
            dests.emplace_back(src.coords, sb.dstPorts[j]);
    }
    for (const auto &[neighborCoords, neighborPort] :
         getCardinalNeighbors(src.coords, src.port.channel)) {
      auto nIt = graph.find(std::make_pair(src.coords, neighborCoords));
      if (nIt != graph.end() &&
          src.port.bundle == getConnectingBundle(neighborPort.bundle)) {
        auto &sb = nIt->second;
        if (llvm::find(sb.dstPorts, neighborPort) != sb.dstPorts.end())
          dests.emplace_back(neighborCoords, neighborPort);
      }
    }
    std::sort(dests.begin(), dests.end());

    std::vector<Edge> edges;
    edges.reserve(dests.size());
    for (auto &dest : dests) {
      auto &sb = graph[std::make_pair(src.coords, dest.coords)];
      int i = portIndex(sb.srcPorts, src.port);
      int j = portIndex(sb.dstPorts, dest.port);
      assert(i >= 0 && j >= 0);
      int destId = getOrAddNodeId(dest);
      edges.push_back(Edge{destId, &sb, i, j});
    }
    // getOrAddNodeId above may have reallocated `adjacency` via index growth in
    // later iterations, but we only assign this node's edges now.
    if (adjacency.size() < nodes.size())
      adjacency.resize(nodes.size());
    adjacency[id] = std::move(edges);
  }
  size_t n = nodes.size();
  if (adjacency.size() < n)
    adjacency.resize(n);

  // Size the reusable Dijkstra scratch buffers. These are indexed by state id,
  // i.e. two entries per node -- one per side of the port.
  distance.assign(2 * n, INF);
  indexInHeap.assign(2 * n, 0);
  colors.assign(2 * n, WHITE);
  preds.assign(2 * n, -1);
  predEdge.assign(2 * n, Edge{-1, nullptr, 0, 0});
  graphBuilt = true;
}

// Dijkstra over the dense graph from dense node `srcId`, searching states
// (node, PortSide) rather than bare nodes. Fills the `preds` and `predEdge`
// scratch buffers, both indexed by state id. The push/relax control flow
// (including the WHITE-node always-push behavior and the absence of a heap
// decrease-key) is inherited from the legacy PathEndPoint-keyed version.
void Pathfinder::dijkstraShortestPaths(int srcId) {
  llvm::fill(distance, INF);
  llvm::fill(colors, static_cast<int8_t>(WHITE));
  llvm::fill(preds, -1);
  llvm::fill(indexInHeap, uint64_t{0});

  using MutableQueue = d_ary_heap_indirect<
      /*Value=*/int, /*Arity=*/4,
      /*IndexInHeapPropertyMap=*/std::vector<uint64_t> &,
      /*DistanceMap=*/std::vector<double> &,
      /*Compare=*/std::less<>>;
  MutableQueue Q(distance, indexInHeap);

  // The flow source port feeds into its switchbox, so the search starts on the
  // In side and the first edge taken is necessarily a crossbar hop.
  int srcState = stateId(srcId, In);
  distance[srcState] = 0.0;
  Q.push(srcState);
  while (!Q.empty()) {
    int s = Q.top();
    Q.pop();
    // In takes crossbar edges and lands on the Out side of the port it picks;
    // Out takes the wire to the neighbour and lands on that tile's In side. Any
    // other pairing would either turn the stream around inside a switchbox or
    // ride a wire the crossbar was never set to drive.
    const bool sIsOut = (s & 1) == Out;
    for (Edge &e : adjacency[stateNode(s)]) {
      const bool isIntra = e.sb->srcCoords == e.sb->dstCoords;
      if (sIsOut == isIntra)
        continue;
      int dst = stateId(e.dst, isIntra ? Out : In);
      double w = e.sb->demand[e.i][e.j];
      bool relax = distance[s] + w < distance[dst];
      if (colors[dst] == WHITE) {
        if (relax) {
          distance[dst] = distance[s] + w;
          preds[dst] = s;
          predEdge[dst] = e;
          colors[dst] = GRAY;
        }
        Q.push(dst);
      } else if (colors[dst] == GRAY && relax) {
        distance[dst] = distance[s] + w;
        preds[dst] = s;
        predEdge[dst] = e;
      }
    }
    colors[s] = BLACK;
  }
}

// Apply one routed edge's per-iteration accounting to switchbox-connect `sb` at
// cell (i, j): mark the priority flag, run the packet-id sharing bookkeeping,
// bump usedCapacity, and re-weight demand. Factored out of the Dijkstra trace
// so the pinned-route replay consumes capacity identically (order matters,
// hence a single shared implementation).
void Pathfinder::accountEdge(SwitchboxConnect &sb, int i, int j,
                             bool isPriority, int packetGroupId,
                             std::optional<int> packetId) {
  sb.isPriority[i][j] = isPriority;
  // Packet flows in the same group may share a channel, but only if their ids
  // differ, so two same-id flows never merge onto a channel and then fan back
  // out to separate destinations.
  if (packetGroupId >= 0 && packetId.has_value() &&
      (sb.packetGroupId[i][j] == -1 ||
       sb.packetGroupId[i][j] == packetGroupId) &&
      sb.packetIds[i][j].count(*packetId) == 0) {
    for (size_t k = 0; k < sb.srcPorts.size(); k++) {
      for (size_t l = 0; l < sb.dstPorts.size(); l++) {
        if (k == static_cast<size_t>(i) || l == static_cast<size_t>(j)) {
          sb.packetGroupId[k][l] = packetGroupId;
          sb.packetIds[k][l].insert(*packetId);
        }
      }
    }
    sb.packetFlowCount[i][j]++;
    // maximum packet stream sharing per channel
    if (sb.packetFlowCount[i][j] >= MAX_PACKET_STREAM_CAPACITY) {
      sb.packetFlowCount[i][j] = 0;
      sb.usedCapacity[i][j]++;
    }
  } else {
    sb.usedCapacity[i][j]++;
  }
  // if at capacity, bump demand to discourage using this Channel
  // this means the order matters!
  sb.bumpDemand(i, j);
}

// Replay a pinned flow's captured route. The captured SwitchSettings stores
// only intra-tile srcs[k]->dsts[k] connections; reconstruct each implicit
// inter-tile hop with the same neighbor logic buildRoutingGraph uses, running
// accountEdge on every edge so data flows negotiate around control's TRUE
// cross-tile footprint. Every port the route touches is stamped processed: a
// data leg that shares this flow's source (addFlow merges co-sourced flows into
// one) then back-traces only to where it rejoins control, so the two share that
// slave. Control itself does not move -- findPaths installs the captured route
// as its solution directly.
void Pathfinder::replayPinnedRoute(bool isPriority, int packetGroupId,
                                   std::optional<int> packetId,
                                   const SwitchSettings &route,
                                   std::vector<uint32_t> &processedStamp,
                                   uint32_t curStamp) {
  auto markNode = [&](TileID coords, Port port) {
    auto it = nodeIds.find(PathEndPoint{coords, port});
    if (it != nodeIds.end()) {
      // The pathfinder graph splits each port node into In/Out states
      // (stateId). Control fully owns every port its captured route touches, so
      // stamp both sides processed: a pure-control flow then traces nothing
      // (its dst is already covered on the Out side) and a co-sourced data
      // leg's back-trace stops wherever it rejoins control.
      processedStamp[stateId(it->second, In)] = curStamp;
      processedStamp[stateId(it->second, Out)] = curStamp;
    }
  };
  for (const auto &[coords, setting] : route) {
    auto intraIt = graph.find(std::make_pair(coords, coords));
    if (intraIt == graph.end())
      continue;
    SwitchboxConnect &intra = intraIt->second;
    for (size_t k = 0; k < setting.srcs.size(); k++) {
      Port sp = setting.srcs[k];
      Port dp = setting.dsts[k];
      markNode(coords, sp);
      markNode(coords, dp);
      // intra-tile crossbar edge (source port -> output port)
      int i = portIndex(intra.srcPorts, sp);
      int j = portIndex(intra.dstPorts, dp);
      if (i >= 0 && j >= 0)
        accountEdge(intra, i, j, isPriority, packetGroupId, packetId);
      // inter-tile hop leaving this tile's output port `dp`. Only a directional
      // (North/South/East/West) output has a neighbor; terminal ports (DMA,
      // TileControl, ...) match none and are a natural no-op.
      for (const auto &[neighborCoords, neighborPort] :
           getCardinalNeighbors(coords, dp.channel)) {
        if (dp.bundle != getConnectingBundle(neighborPort.bundle))
          continue;
        auto nIt = graph.find(std::make_pair(coords, neighborCoords));
        if (nIt == graph.end())
          continue;
        SwitchboxConnect &sb = nIt->second;
        markNode(neighborCoords, neighborPort);
        int ii = portIndex(sb.srcPorts, dp);
        int jj = portIndex(sb.dstPorts, neighborPort);
        if (ii >= 0 && jj >= 0)
          accountEdge(sb, ii, jj, isPriority, packetGroupId, packetId);
        break;
      }
    }
  }
}

// Perform congestion-aware routing for all flows which have been added.
// Use Dijkstra's shortest path to find routes, and use "demand" as the
// weights. If the routing finds too much congestion, update the demand
// weights and repeat the process until a valid solution is found. Returns a
// map specifying switchbox settings for all flows. If no legal routing can be
// found after maxIterations, returns empty vector.
std::optional<std::map<PathEndPoint, SwitchSettings>>
Pathfinder::findPaths(const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin Pathfinder::findPaths---\n");
  std::map<PathEndPoint, SwitchSettings> routingSolution;
  // Build the dense routing graph once; topology is invariant across
  // iterations. Under freeze, reserve control's master ports FIRST so the
  // reserved connectivity is baked into the dense graph (a foreign data flow
  // then has no edge into a control master); a collision with a pre-placed
  // circuit connection fails closed.
  if (!graphBuilt) {
    if (!reservePinnedControlMasters())
      return std::nullopt;
    buildRoutingGraph();
  }
  // Stamp-based "processed" set (avoids O(n) clears per flow).
  std::vector<uint32_t> processedStamp(2 * nodes.size(), 0);
  uint32_t curStamp = 0;
  // initialize all Channel histories to 0
  for (auto &[_, sb] : graph) {
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      for (size_t j = 0; j < sb.dstPorts.size(); j++) {
        sb.usedCapacity[i][j] = 0;
        sb.overCapacity[i][j] = 0;
        sb.isPriority[i][j] = false;
      }
    }
  }

  // group flows based on packetGroupId
  llvm::MapVector<int, std::vector<Flow>> groupedFlows;
  for (auto &f : flows) {
    if (groupedFlows.count(f.packetGroupId) == 0) {
      groupedFlows[f.packetGroupId] = std::vector<Flow>();
    }
    groupedFlows[f.packetGroupId].push_back(f);
  }

  int iterationCount = -1;
  int illegalEdges = 0;
#ifndef NDEBUG
  int totalPathLength = 0;
#endif
  do {
    // if reach maxIterations, throw an error since no routing can be found
    if (++iterationCount >= maxIterations) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\t\tPathfinder: maxIterations has been exceeded ("
                 << maxIterations
                 << " iterations)...unable to find routing for flows.\n");
      return std::nullopt;
    }

    LLVM_DEBUG(llvm::dbgs() << "\t\t---Begin findPaths iteration #"
                            << iterationCount << "---\n");
    // update demand at the beginning of each iteration
    for (auto &[_, sb] : graph) {
      sb.updateDemand();
    }

    // "rip up" all routes
    illegalEdges = 0;
#ifndef NDEBUG
    totalPathLength = 0;
#endif
    routingSolution.clear();
    for (auto &[_, sb] : graph) {
      for (size_t i = 0; i < sb.srcPorts.size(); i++) {
        for (size_t j = 0; j < sb.dstPorts.size(); j++) {
          sb.usedCapacity[i][j] = 0;
          sb.packetFlowCount[i][j] = 0;
          sb.packetGroupId[i][j] = -1;
          sb.packetIds[i][j].clear();
        }
      }
    }

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them

    for (const auto &[_, flows] : groupedFlows) {
      for (const auto &[packetGroupId, isPriority, src, dsts, packetId,
                        dstPacketIds] : flows) {
        int srcId = nodeIds.at(src);
        // A pinned flow (control under freeze) does not route its OWN
        // destinations: seed the solution with the captured route, account its
        // edges (INF demand on control's channels so data steers clear), and
        // stamp its ports processed. addFlow merges a co-sourced data leg into
        // this same flow; its destination is NOT in the captured route, so the
        // trace below still Dijkstra-routes it around control, rejoining at the
        // shared source slave. A pure control flow has all destinations covered
        // and traces nothing -> its solution stays byte-identical to captured.
        SwitchSettings switchSettings;
        ++curStamp;
        if (auto pinIt = pinnedRoutes.find(src); pinIt != pinnedRoutes.end()) {
          switchSettings = pinIt->second;
          replayPinnedRoute(isPriority, packetGroupId, packetId, pinIt->second,
                            processedStamp, curStamp);
        }
        // Consolidation-aware routing for a merged multi-destination PACKET
        // flow: a packet channel carries many distinct ids, so a co-sourced
        // flow's destinations share one trunk (peeling one id per tile). Route
        // each destination on its own Dijkstra pass and, after tracing a
        // destination, discount the edges it used so the next destination of
        // this flow reuses that trunk. Consolidation is best-effort: a
        // destination that cannot ride the discounted trunk routes
        // independently and is still correct -- the discount only steers among
        // equal-viable channels, never changing correctness. The discount is
        // intra-flow, restored after the flow so it never leaks into another
        // flow's demand.
        //
        // Scoped to control-overlay (reconfiguration) routing: a merged control
        // multicast needs a coherent trunk only when the compile carries the
        // control overlay (controlOverlayRouting, set for freeze on OR off) or
        // replays a pinned control route. A plain, non-reconfiguration compile
        // keeps the upstream single-Dijkstra, all-dsts-against-one-tree path,
        // so generic packet routing stays byte-identical to upstream. Circuit
        // flows (no packet id) never consolidate.
        const bool isPacketFlow = packetId.has_value();
        const bool consolidate =
            isPacketFlow && (controlOverlayRouting || !pinnedRoutes.empty());
        // (SwitchboxConnect*, i, j, pre-discount demand) to restore at flow
        // end.
        std::vector<std::tuple<SwitchboxConnect *, int, int, double>>
            trunkDiscounts;
        // Non-consolidated flows (circuit flows, and packet flows outside the
        // design-aware path) route all destinations against one tree. For a
        // consolidated pinned flow the per-destination passes route only the
        // uncovered co-sourced legs around control's just-accounted footprint.
        if (!consolidate)
          dijkstraShortestPaths(srcId);

        // trace the path of the flow backwards via predecessors
        // increment used_capacity for the associated channels
        processedStamp[stateId(srcId, In)] = curStamp;
        // Order in which this flow's destinations are traced. Default = IR
        // order (byte-identical). Under the design-aware capture, trace a
        // priority control multicast farthest-first: the longest path leaves
        // the source on one channel and every nearer destination (a prefix of
        // it) reuses that trunk via the discount, so the source emits a single
        // coherent output channel instead of one fresh channel per destination.
        std::vector<size_t> dstOrder(dsts.size());
        for (size_t di = 0; di < dsts.size(); di++)
          dstOrder[di] = di;
        if (coherentControlCapture && isPacketFlow && isPriority) {
          // Copy out of the structured bindings first (capturing them in a
          // lambda is a C++20 extension). Sort by Manhattan distance from the
          // source, farthest first.
          const int srcCol = src.coords.col, srcRow = src.coords.row;
          const std::vector<PathEndPoint> &dstList = dsts;
          auto srcDist = [&dstList, srcCol, srcRow](size_t k) {
            int dc = dstList[k].coords.col - srcCol;
            int dr = dstList[k].coords.row - srcRow;
            return (dc < 0 ? -dc : dc) + (dr < 0 ? -dr : dr);
          };
          llvm::stable_sort(dstOrder, [&](size_t a, size_t b) {
            return srcDist(a) > srcDist(b);
          });
        }
        for (size_t di : dstOrder) {
          const PathEndPoint &endPoint = dsts[di];
          // Id used for this destination's per-edge accounting. The
          // consolidated (reconfiguration) path routes each co-sourced
          // destination on its own Dijkstra pass, so it accounts under that
          // destination's DISTINCT id. The non-consolidated path routes all
          // destinations against one tree exactly like upstream, so it accounts
          // under the flow's representative packetId -- keeping generic
          // (non-reconfiguration) packet routing byte-identical to upstream
          // even when one source drives distinct ids.
          std::optional<int> accountId =
              consolidate ? dstPacketIds[di] : packetId;
          if (endPoint == src) {
            // Route to self: the port is both ends, so there is no path to
            // trace. The source was stamped on the In side, so falling through
            // would trace back from an Out state Dijkstra never reached.
            switchSettings[src.coords].srcs.push_back(src.port);
            switchSettings[src.coords].dsts.push_back(src.port);
            continue;
          }
          // Recompute the tree per destination so the trunk discount from
          // earlier destinations of this flow is visible.
          if (consolidate)
            dijkstraShortestPaths(srcId);
          // A destination port is driven by its switchbox, so it is reached on
          // the Out side.
          int currId = stateId(nodeIds.at(endPoint), Out);
          // trace backwards until a vertex already processed is reached
          while (processedStamp[currId] != curStamp) {
            // If Dijkstra never reached this node it has no predecessor; the
            // destination is unroutable under the current demand. Bail out of
            // this iteration rather than indexing with a -1 predecessor.
            if (preds[currId] < 0)
              return std::nullopt;
            const PathEndPoint &curr = nodes[stateNode(currId)];
            const Edge &e = predEdge[currId];
            int predId = preds[currId];
            const PathEndPoint &pred = nodes[stateNode(predId)];
            accountEdge(*e.sb, e.i, e.j, isPriority, packetGroupId, accountId);
            if (consolidate) {
              // Discount this edge so the next same-flow destination reuses it.
              // Only this flow's unique tails are touched (shared-trunk edges
              // are stamped and skipped), so each edge is discounted at most
              // once per flow.
              double cur = e.sb->demand[e.i][e.j];
              trunkDiscounts.emplace_back(e.sb, e.i, e.j, cur);
              e.sb->demand[e.i][e.j] = cur * kTrunkReuseDiscount;
            }
            if (pred.coords == curr.coords) {
              switchSettings[pred.coords].srcs.push_back(pred.port);
              switchSettings[curr.coords].dsts.push_back(curr.port);
            }
            processedStamp[currId] = curStamp;
            currId = predId;
          }
        }
        // Undo the intra-flow trunk discount so it does not bias any other
        // flow's routing. Restore in reverse to recover the original demand.
        for (auto it = trunkDiscounts.rbegin(); it != trunkDiscounts.rend();
             ++it) {
          auto &[sb, i, j, orig] = *it;
          sb->demand[i][j] = orig;
        }
        // add this flow to the proposed solution
        routingSolution[src] = switchSettings;
      }
      for (auto &[_, sb] : graph) {
        for (size_t i = 0; i < sb.srcPorts.size(); i++) {
          for (size_t j = 0; j < sb.dstPorts.size(); j++) {
            // fix used capacity for packet flows
            if (sb.packetFlowCount[i][j] > 0) {
              sb.packetFlowCount[i][j] = 0;
              sb.usedCapacity[i][j]++;
            }
            sb.bumpDemand(i, j);
          }
        }
      }
    }

    for (auto &[_, sb] : graph) {
      for (size_t i = 0; i < sb.srcPorts.size(); i++) {
        for (size_t j = 0; j < sb.dstPorts.size(); j++) {
          // check that every channel does not exceed max capacity
          if (sb.usedCapacity[i][j] > MAX_CIRCUIT_STREAM_CAPACITY) {
            sb.overCapacity[i][j]++;
            illegalEdges++;
            LLVM_DEBUG(
                llvm::dbgs()
                << "\t\t\tToo much capacity on (" << sb.srcCoords.col << ","
                << sb.srcCoords.row << ") " << sb.srcPorts[i].bundle
                << sb.srcPorts[i].channel << " -> (" << sb.dstCoords.col << ","
                << sb.dstCoords.row << ") " << sb.dstPorts[j].bundle
                << sb.dstPorts[j].channel << ", used_capacity = "
                << sb.usedCapacity[i][j] << ", demand = " << sb.demand[i][j]
                << ", over_capacity_count = " << sb.overCapacity[i][j] << "\n");
          }
#ifndef NDEBUG
          // calculate total path length (across switchboxes)
          if (sb.srcCoords != sb.dstCoords) {
            totalPathLength += sb.usedCapacity[i][j];
          }
#endif
        }
      }
    }

#ifndef NDEBUG
    for (const auto &[PathEndPoint, switchSetting] : routingSolution) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\t\t\tFlow starting at (" << PathEndPoint.coords.col << ","
                 << PathEndPoint.coords.row << "):\t");
      LLVM_DEBUG(llvm::dbgs() << switchSetting);
    }
    LLVM_DEBUG(llvm::dbgs()
               << "\t\t---End findPaths iteration #" << iterationCount
               << " , illegal edges count = " << illegalEdges
               << ", total path length = " << totalPathLength << "---\n");
#endif
  } while (illegalEdges >
           0); // continue iterations until a legal routing is found

  LLVM_DEBUG(llvm::dbgs() << "\t---End Pathfinder::findPaths---\n");
  return routingSolution;
}

// Get enum int value from WireBundle.
int AIE::getWireBundleAsInt(WireBundle bundle) {
  return static_cast<typename std::underlying_type<WireBundle>::type>(bundle);
}
