//===- AIEPathfinder.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"
#include "d_ary_heap.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

#include <numeric>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-pathfinder"

LogicalResult DynamicTileAnalysis::runAnalysis(DeviceOp &device) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin DynamicTileAnalysis Constructor---\n");
  // find the maxCol and maxRow
  maxCol = device.getTargetModel().columns();
  maxRow = device.getTargetModel().rows();

  pathfinder->initialize(maxCol, maxRow, device.getTargetModel());

  // For each flow (circuit + packet) in the device, add it to pathfinder. Each
  // source can map to multiple different destinations (fanout). Control packet
  // flows to be routed (as prioritized routings). Then followed by normal
  // packet flows.
  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
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

    bool priorityFlow = pktFlowOp.getPriorityRoute().value_or(false);
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
  if (auto maybeFlowSolutions = pathfinder->findPaths(maxIterations)) {
    flowSolutions = maybeFlowSolutions.value();
  } else {
    std::string reason = pathfinder->getFailureReason();
    if (reason.empty())
      reason = routingFailureReason;
    if (reason.empty())
      reason = pathfinder->getOveruseReason();
    if (reason.empty())
      return device.emitError("Unable to find a legal routing");
    return device.emitError("Unable to find a legal routing: ") << reason;
  }

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
  packetIdsTo.clear();
  graphBuilt = false;
  nodeIds.clear();
  nodes.clear();
  adjacency.clear();
  distance.clear();
  indexInHeap.clear();
  colors.clear();
  preds.clear();
  predEdge.clear();

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
  if (packetId) {
    auto &ids = packetIdsTo[{{srcCoords, srcPort}, {dstCoords, dstPort}}];
    if (!llvm::is_contained(ids, *packetId))
      ids.push_back(*packetId);
  }
  // check if a flow with this source already exists
  for (auto &[_, prioritized, src, dsts, pid] : flows) {
    if (src.coords == srcCoords && src.port == srcPort) {
      if (isPriorityFlow) {
        prioritized = true;
        dsts.emplace(dsts.begin(), dstCoords, dstPort);
      } else
        dsts.emplace_back(dstCoords, dstPort);
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
    for (auto &[existingId, _, src, dsts, pid] : flows) {
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
      if (found)
        break;
      packetGroupId = std::max(packetGroupId, existingId);
    }
    if (!found) {
      packetGroupId++;
    }
  }
  // If no existing flow was found with this source, create a new flow.
  flows.push_back(Flow{
      packetGroupId, isPriorityFlow, PathEndPoint{srcCoords, srcPort},
      std::vector<PathEndPoint>{PathEndPoint{dstCoords, dstPort}}, packetId});
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

  for (auto &flow : flows)
    std::sort(flow.dsts.begin(), flow.dsts.end(), endpointLess);

  // Packet flows that share a destination, directly or through others, form
  // one group, whatever order they were added in. A source has one flow.
  std::vector<size_t> parent(flows.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](size_t a) {
    while (parent[a] != a)
      a = parent[a] = parent[parent[a]];
    return a;
  };
  std::map<PathEndPoint, size_t> firstTo;
  for (auto [k, flow] : llvm::enumerate(flows)) {
    if (flow.packetGroupId < 0)
      continue;
    for (const PathEndPoint &dst : flow.dsts) {
      auto [it, fresh] = firstTo.try_emplace(dst, k);
      if (!fresh)
        parent[root(k)] = root(it->second);
    }
  }
  std::map<size_t, int> groupOf;
  for (auto [k, flow] : llvm::enumerate(flows))
    if (flow.packetGroupId >= 0)
      flow.packetGroupId =
          groupOf.try_emplace(root(k), groupOf.size()).first->second;

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
  // Validate every connect against the original connectivity before reserving
  // any resource, so a broadcast (one source fanning out to several dests) does
  // not invalidate its own sibling connects mid-scan. A destination port may be
  // driven only once; a source port may legally recur across a broadcast.
  llvm::SmallVector<std::pair<int, int>, 8> reserved;
  llvm::SmallDenseSet<int, 8> claimedDsts;
  for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
    int srcIdx = -1, dstIdx = -1;
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      if (sb.srcPorts[i] == connectOp.sourcePort()) {
        srcIdx = static_cast<int>(i);
        break;
      }
    }
    for (size_t j = 0; j < sb.dstPorts.size(); j++) {
      if (sb.dstPorts[j] == connectOp.destPort()) {
        dstIdx = static_cast<int>(j);
        break;
      }
    }
    // Reject an illegal pair (absent from the switchbox model) or a second
    // driver on the same output port; a repeated source port is a broadcast.
    if (srcIdx < 0 || dstIdx < 0 ||
        sb.connectivity[srcIdx][dstIdx] != Connectivity::AVAILABLE ||
        !claimedDsts.insert(dstIdx).second) {
      return false;
    }
    reserved.emplace_back(srcIdx, dstIdx);
  }
  // A pre-placed packet-switched output (an aie.masterset) also monopolizes its
  // destination port. That stream-switch output is already configured for
  // packet switching, so no circuit stream may drive it, and the router cannot
  // emit a second masterset on the same port for another packet flow. The
  // circuit pathfinder does not read packet ops, so without this reservation it
  // may route a flow onto an output the masterset owns and produce two ops
  // driving one destination.
  llvm::SmallVector<int, 8> reservedMasterDsts;
  for (MasterSetOp masterSetOp : switchboxOp.getOps<MasterSetOp>()) {
    int dstIdx = -1;
    for (size_t j = 0; j < sb.dstPorts.size(); j++) {
      if (sb.dstPorts[j] == masterSetOp.destPort()) {
        dstIdx = static_cast<int>(j);
        break;
      }
    }
    // Reject an output port absent from the switchbox model or already driven
    // by a circuit connect or another masterset.
    if (dstIdx < 0 || !claimedDsts.insert(dstIdx).second) {
      return false;
    }
    reservedMasterDsts.push_back(dstIdx);
  }
  // A circuit-switched ConnectOp monopolizes both its source port (the stream
  // switch input) and its destination port (the output): no other stream may
  // inject on that input, and the output can carry only this one stream.
  // Reserving the whole column also reserves the outgoing wire, since that wire
  // is reachable only by driving this output port.
  for (auto [srcIdx, dstIdx] : reserved) {
    for (size_t j = 0; j < sb.dstPorts.size(); j++) {
      sb.connectivity[srcIdx][j] = Connectivity::INVALID;
    }
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      sb.connectivity[i][dstIdx] = Connectivity::INVALID;
    }
  }
  // A masterset fixes only its output port. Its inputs arrive through arbiters
  // that packet flows share, so the source rows stay free.
  for (int dstIdx : reservedMasterDsts) {
    for (size_t i = 0; i < sb.srcPorts.size(); i++) {
      sb.connectivity[i][dstIdx] = Connectivity::INVALID;
    }
  }
  // At a shim, the DMA, NOC and PLIO ports reach the switchbox through the shim
  // mux on South channels, so a fixed op on such a channel claims them too.
  if (switchboxOp.getTileOp().isShimNOCorPLTile()) {
    auto southDst = [](Port p) {
      if (p.bundle == WireBundle::DMA)
        return p.channel == 0 ? 2 : 3;
      if (p.bundle == WireBundle::NOC)
        return p.channel + 2;
      return p.channel;
    };
    auto southSrc = [](Port p) {
      if (p.bundle == WireBundle::DMA)
        return p.channel == 0 ? 3 : 7;
      if (p.bundle == WireBundle::NOC)
        return p.channel >= 2 ? p.channel + 4 : p.channel + 2;
      return p.channel;
    };
    auto isMuxed = [](Port p) {
      return p.bundle == WireBundle::DMA || p.bundle == WireBundle::NOC ||
             p.bundle == WireBundle::PLIO;
    };
    llvm::SmallDenseSet<int, 8> southDsts, southSrcs;
    for (int dstIdx : claimedDsts)
      if (sb.dstPorts[dstIdx].bundle == WireBundle::South)
        southDsts.insert(sb.dstPorts[dstIdx].channel);
    for (auto [srcIdx, dstIdx] : reserved)
      if (sb.srcPorts[srcIdx].bundle == WireBundle::South)
        southSrcs.insert(sb.srcPorts[srcIdx].channel);
    for (size_t j = 0; j < sb.dstPorts.size(); j++)
      if (isMuxed(sb.dstPorts[j]) && southDsts.count(southDst(sb.dstPorts[j])))
        for (size_t i = 0; i < sb.srcPorts.size(); i++)
          sb.connectivity[i][j] = Connectivity::INVALID;
    for (size_t i = 0; i < sb.srcPorts.size(); i++)
      if (isMuxed(sb.srcPorts[i]) && southSrcs.count(southSrc(sb.srcPorts[i])))
        for (size_t j = 0; j < sb.dstPorts.size(); j++)
          sb.connectivity[i][j] = Connectivity::INVALID;
  }
  for (PacketRulesOp rulesOp : switchboxOp.getOps<PacketRulesOp>()) {
    auto it = llvm::find(sb.srcPorts, rulesOp.sourcePort());
    if (it == sb.srcPorts.end())
      return false;
    sb.packetOnlySrc[it - sb.srcPorts.begin()] = true;
  }
  return true;
}

static constexpr double INF = std::numeric_limits<double>::max();

static std::string endpointString(const PathEndPoint &p) {
  return "(" + std::to_string(p.coords.col) + ", " +
         std::to_string(p.coords.row) + ") " +
         stringifyWireBundle(p.port.bundle).str() + ":" +
         std::to_string(p.port.channel);
}

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
    std::vector<std::pair<TileID, Port>> neighbors = {
        {{src.coords.col, src.coords.row - 1},
         {WireBundle::North, src.port.channel}},
        {{src.coords.col - 1, src.coords.row},
         {WireBundle::East, src.port.channel}},
        {{src.coords.col, src.coords.row + 1},
         {WireBundle::South, src.port.channel}},
        {{src.coords.col + 1, src.coords.row},
         {WireBundle::West, src.port.channel}}};
    for (const auto &[neighborCoords, neighborPort] : neighbors) {
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
      int i = static_cast<int>(std::distance(
          sb.srcPorts.begin(), llvm::find(sb.srcPorts, src.port)));
      int j = static_cast<int>(std::distance(
          sb.dstPorts.begin(), llvm::find(sb.dstPorts, dest.port)));
      assert(i < static_cast<int>(sb.srcPorts.size()));
      assert(j < static_cast<int>(sb.dstPorts.size()));
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

double Pathfinder::edgeWeight(const Edge &e, std::optional<int> packetId,
                              const llvm::BitVector *avoid,
                              const llvm::BitVector *avoidBranch) {
  double w = e.sb->demand[e.i][e.j];
  // Sharing it would take a second stream's capacity, which the demand
  // only shows once the group is routed.
  if (packetId && e.sb->packetFlowCount[e.i][e.j] > 0 &&
      e.sb->packetIds[e.i][e.j].count(*packetId))
    w *= DEMAND_COEFF;
  if (e.sb->srcCoords == e.sb->dstCoords)
    for (const llvm::BitVector *flows : {avoid, avoidBranch})
      if (flows && llvm::any_of(e.sb->unitPacketFlows[e.sb->unitOf(e.j)],
                                [&](int f) { return flows->test(f); }))
        w += CONFLICT_SHARE_PENALTY;
  return w;
}

// Dijkstra over the dense graph from the states in `seeds`, searching states
// (node, PortSide) rather than bare nodes. Fills the `preds` and `predEdge`
// scratch buffers, both indexed by state id. The push/relax control flow
// (including the WHITE-node always-push behavior and the absence of a heap
// decrease-key) is inherited from the legacy PathEndPoint-keyed version.
void Pathfinder::dijkstraShortestPaths(
    ArrayRef<int> seeds, ArrayRef<double> seedCosts,
    std::optional<int> packetId, const llvm::BitVector *avoid,
    const llvm::DenseMap<int, llvm::BitVector> *branchAvoid,
    const llvm::DenseSet<int> *stops) {
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

  for (auto [seed, cost] : llvm::zip_equal(seeds, seedCosts)) {
    distance[seed] = cost;
    colors[seed] = GRAY;
    Q.push(seed);
  }
  while (!Q.empty()) {
    int s = Q.top();
    Q.pop();
    if (stops && stops->count(s)) {
      colors[s] = BLACK;
      continue;
    }
    // In takes crossbar edges and lands on the Out side of the port it picks;
    // Out takes the wire to the neighbour and lands on that tile's In side. Any
    // other pairing would either turn the stream around inside a switchbox or
    // ride a wire the crossbar was never set to drive.
    const bool sIsOut = (s & 1) == Out;
    const llvm::BitVector *avoidBranch = nullptr;
    if (branchAvoid)
      if (auto it = branchAvoid->find(s); it != branchAvoid->end())
        avoidBranch = &it->second;
    for (Edge &e : adjacency[stateNode(s)]) {
      const bool isIntra = e.sb->srcCoords == e.sb->dstCoords;
      if (sIsOut == isIntra ||
          (isIntra && !packetId && e.sb->packetOnlySrc[e.i]))
        continue;
      int dst = stateId(e.dst, isIntra ? Out : In);
      double w = edgeWeight(e, packetId, avoid, avoidBranch);
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

// Perform congestion-aware routing for all flows which have been added.
// Use Dijkstra's shortest path to find routes, and use "demand" as the
// weights. If the routing finds too much congestion, update the demand
// weights and repeat the process until a valid solution is found. Returns a
// map specifying switchbox settings for all flows. If no legal routing can be
// found after maxIterations, returns empty vector.
std::optional<std::map<PathEndPoint, SwitchSettings>>
Pathfinder::findPaths(const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin Pathfinder::findPaths---\n");
  failureReason.clear();
  overuseReason.clear();
  std::map<PathEndPoint, SwitchSettings> routingSolution;
  // Build the dense routing graph once; topology is invariant across
  // iterations.
  if (!graphBuilt)
    buildRoutingGraph();
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

  // group flows based on packetGroupId; pinned trees go in first, so the
  // flows routed around them see them.
  llvm::MapVector<int, std::vector<Flow>> groupedFlows;
  for (auto &f : flows) {
    int group = pinnedTrees.count(f.src) ? std::numeric_limits<int>::min()
                                         : f.packetGroupId;
    if (groupedFlows.count(group) == 0) {
      groupedFlows[group] = std::vector<Flow>();
    }
    groupedFlows[group].push_back(f);
  }

  // Packet flows that conflict, by source; a source has one flow.
  std::map<PathEndPoint, int> flowIndex;
  for (auto [k, f] : llvm::enumerate(flows))
    flowIndex[f.src] = k;
  std::vector<llvm::BitVector> conflicting(flows.size(),
                                           llvm::BitVector(flows.size()));
  if (packetConflict)
    for (size_t a = 0; a < flows.size(); a++)
      for (size_t b = a + 1; b < flows.size(); b++)
        if (flows[a].packetId && flows[b].packetId &&
            packetConflict(flows[a].src, flows[b].src)) {
          conflicting[a].set(b);
          conflicting[b].set(a);
        }
  // The packet ids each flow carries, to each destination and in all.
  auto idsTo = [&](int flow, int dstState) -> ArrayRef<int> {
    auto it = packetIdsTo.find({flows[flow].src, nodes[stateNode(dstState)]});
    return it == packetIdsTo.end() ? ArrayRef<int>()
                                   : ArrayRef<int>(it->second);
  };
  std::vector<std::set<int>> flowIds(flows.size());
  for (const auto &[ends, ids] : packetIdsTo)
    flowIds[flowIndex.at(ends.first)].insert(ids.begin(), ids.end());
  std::vector<SmallVector<int, 4>> sameId(flows.size());
  for (size_t a = 0; a < flows.size(); a++)
    for (size_t b = 0; b < flows.size(); b++)
      if (a != b && llvm::any_of(flowIds[a],
                                 [&](int id) { return flowIds[b].count(id); }))
        sameId[a].push_back(b);
  // Each packet flow's tree as routed so far this iteration: the state each
  // hop reaches, from which state and by which edge; and its destinations.
  std::vector<llvm::DenseMap<int, std::pair<int, Edge>>> treeOf(flows.size());
  std::vector<SmallVector<int, 4>> treeDsts(flows.size());
  // The hops each flow takes only by joining another flow's tree, by the flow
  // it joined. A join the routing check faults is not made again.
  std::vector<SmallVector<std::pair<int, Edge>, 8>> joinedHops(flows.size());
  std::set<std::pair<int, int>> noJoin;
  // Where the routing check split each flow's tree: the state of the slave
  // port it branches at, and the ids to branch apart there.
  std::vector<std::set<std::tuple<int, int, int>>> splitsOf(flows.size());

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
      // A prioritized flow keeps the route it takes alone, so the others
      // may have had to fit around it.
      auto overused = [](const SwitchboxConnect &sb) {
        return sb.srcCoords != sb.dstCoords &&
               llvm::any_of(sb.usedCapacity, [](const auto &row) {
                 return llvm::any_of(row, [](int used) {
                   return used > MAX_CIRCUIT_STREAM_CAPACITY;
                 });
               });
      };
      auto link = [](const SwitchboxConnect &sb) {
        return "from tile (" + std::to_string(sb.srcCoords.col) + ", " +
               std::to_string(sb.srcCoords.row) + ") to (" +
               std::to_string(sb.dstCoords.col) + ", " +
               std::to_string(sb.dstCoords.row) + ")";
      };
      const Flow *prioritized = nullptr;
      llvm::SetVector<const SwitchboxConnect *> held;
      for (auto [k, f] : llvm::enumerate(flows)) {
        if (!f.isPriorityFlow)
          continue;
        prioritized = prioritized ? prioritized : &f;
        for (const auto &[_, hop] : treeOf[k]) {
          if (overused(*hop.second.sb)) {
            failureReason = "packet flows from " + endpointString(f.src) +
                            " are prioritized (priority_route), so they keep "
                            "the route they take alone, and it holds a "
                            "channel " +
                            link(*hop.second.sb) + " the other flows need.";
            return std::nullopt;
          }
          if (hop.second.sb->srcCoords != hop.second.sb->dstCoords)
            held.insert(hop.second.sb);
        }
      }
      if (prioritized && llvm::any_of(graph, [&](const auto &entry) {
            return overused(entry.second);
          })) {
        failureReason = "packet flows from " +
                        endpointString(prioritized->src) +
                        " are prioritized (priority_route), so they keep the "
                        "route they take alone, and the router found no "
                        "routing for the other flows around the channels it "
                        "holds";
        for (auto [i, sb] : llvm::enumerate(held))
          failureReason += (i ? ", " : " ") + link(*sb);
        failureReason += ".";
      }
      if (!failureReason.empty())
        return std::nullopt;
      // Name the channel the last iteration overused that was overused in the
      // most iterations, and the flows the last one routed through it.
      const SwitchboxConnect *worst = nullptr;
      int worstI = 0, worstJ = 0, worstCount = 0;
      for (const auto &[_, sb] : graph)
        for (size_t i = 0; i < sb.srcPorts.size(); i++)
          for (size_t j = 0; j < sb.dstPorts.size(); j++)
            if (sb.usedCapacity[i][j] > MAX_CIRCUIT_STREAM_CAPACITY &&
                sb.overCapacity[i][j] > worstCount) {
              worst = &sb;
              worstI = i;
              worstJ = j;
              worstCount = sb.overCapacity[i][j];
            }
      if (!worst)
        return std::nullopt;
      bool crossbar = worst->srcCoords == worst->dstCoords;
      std::vector<std::string> users;
      for (const auto &[src, settings] : routingSolution) {
        auto it = settings.find(worst->srcCoords);
        if (it == settings.end())
          continue;
        const SwitchSetting &s = it->second;
        bool uses = false;
        for (size_t k = 0; k < s.dsts.size() && !uses; k++)
          uses = crossbar ? k < s.srcs.size() &&
                                s.srcs[k] == worst->srcPorts[worstI] &&
                                s.dsts[k] == worst->dstPorts[worstJ]
                          : llvm::is_contained(worst->srcPorts, s.dsts[k]);
        if (uses)
          users.push_back(endpointString(src));
      }
      std::string where =
          crossbar
              ? "the connection from " +
                    stringifyWireBundle(worst->srcPorts[worstI].bundle).str() +
                    ":" + std::to_string(worst->srcPorts[worstI].channel) +
                    " to " +
                    stringifyWireBundle(worst->dstPorts[worstJ].bundle).str() +
                    ":" + std::to_string(worst->dstPorts[worstJ].channel) +
                    " at tile (" + std::to_string(worst->srcCoords.col) + ", " +
                    std::to_string(worst->srcCoords.row) + ")"
              : "the links " + link(*worst);
      if (users.empty()) {
        overuseReason = "the router found no routing that fits " + where + ".";
        return std::nullopt;
      }
      constexpr size_t shown = 4;
      if (users.size() > shown) {
        size_t more = users.size() - shown;
        users.resize(shown);
        users.push_back(std::to_string(more) + " more");
      }
      overuseReason = "the flows from ";
      for (auto [k, user] : llvm::enumerate(users))
        overuseReason += (k == 0                  ? ""
                          : k + 1 == users.size() ? " and "
                                                  : ", ") +
                         user;
      overuseReason += " need " + where +
                       ", and the router found no routing that fits them.";
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
      sb.resetUnits();
    }
    for (auto &tree : treeOf)
      tree.clear();
    for (auto &dsts : treeDsts)
      dsts.clear();
    for (auto &hops : joinedHops)
      hops.clear();
    packetTrees.clear();

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them

    for (const auto &[_, flows] : groupedFlows) {
      for (const auto &[packetGroupId, isPriority, src, dsts, packetId] :
           flows) {
        // Grow the flow's tree one destination at a time: Dijkstra, given the
        // current demand, from everything the tree reaches so far to the next
        // destination, whose path is then traced back to the tree. Growing
        // from the tree rather than the source lets destinations share hops.
        // A branch off a hop of the tree starts from TREE_SEED_FACTOR per hop
        // back to the source; the demand of those hops is already paid.
        int srcId = nodeIds.at(src);
        int flow = flowIndex.at(src);
        SwitchSettings switchSettings;
        ++curStamp;
        const llvm::BitVector *avoid = packetId ? &conflicting[flow] : nullptr;
        auto pin = pinnedTrees.find(src);
        // The flow source port feeds into its switchbox, so the tree starts
        // on its In side and the first edge taken is necessarily a crossbar
        // hop.
        SmallVector<int, 16> tree{stateId(srcId, In)};
        SmallVector<int, 16> treeHops{0};
        SmallVector<int, 16> seeds;
        SmallVector<double, 16> seedCosts;
        processedStamp[tree.front()] = curStamp;
        // The tree's hops, by the state each reaches, from which state and by
        // which edge, in the order they were traced. They take effect once the
        // tree is final.
        llvm::MapVector<int, std::pair<int, Edge>> planned;
        llvm::DenseMap<int, int> children;
        // A branch off a port the tree already crosses puts its master port on
        // the arbiter of the ports the tree leaves by there, so it avoids the
        // flows conflicting with any flow on those arbiters too.
        llvm::DenseMap<int, llvm::BitVector> branchAvoid;
        SmallVector<PathEndPoint, 4> pending;
        for (auto endPoint : dsts) {
          // Route to self: the port is both ends. Where its switchbox cannot
          // connect it to itself (Core to Core), the stream has to leave and
          // come back, which Dijkstra finds from the In side to the Out side.
          if (endPoint == src &&
              llvm::any_of(adjacency[srcId],
                           [&](const Edge &e) { return e.dst == srcId; })) {
            switchSettings[src.coords].srcs.push_back(src.port);
            switchSettings[src.coords].dsts.push_back(src.port);
            continue;
          }
          pending.push_back(endPoint);
        }
        // A destination port is driven by its switchbox, so it is reached on
        // the Out side.
        auto dstState = [&](const PathEndPoint &p) {
          return stateId(nodeIds.at(p), Out);
        };
        // A switchbox routes on the id alone, so packets that reach a master
        // port another source's tree takes an id they share by go wherever
        // that id goes from there. The flow may join the tree there only if
        // it still has to reach each of those destinations, with just the ids
        // they share.
        llvm::DenseMap<int, SmallVector<int, 4>> joins;
        llvm::DenseMap<int, SmallVector<int, 2>> joinOwners;
        llvm::DenseSet<int> stops, unjoinable;
        auto shared = [&](ArrayRef<int> ids, int other) {
          std::set<int> common;
          for (int id : ids)
            if (flowIds[other].count(id))
              common.insert(id);
          return common;
        };
        for (int other : sameId[flow])
          for (int dst : treeDsts[other]) {
            std::set<int> common = shared(idsTo(other, dst), flow);
            if (common.empty())
              continue;
            // Joining shares the other flow's arbiters below the join. A
            // prioritized flow's tree is pinned, so it joins only trees
            // pinned with it.
            bool agree = common == shared(idsTo(flow, dst), other) &&
                         !conflicting[flow].test(other) &&
                         !noJoin.count({flow, other}) &&
                         (!isPriority || this->flows[other].isPriorityFlow);
            for (auto it = treeOf[other].find(dst); it != treeOf[other].end();
                 it = treeOf[other].find(it->second.first)) {
              int up = it->second.first;
              if ((up & 1) != Out)
                continue;
              stops.insert(up);
              if (!agree)
                unjoinable.insert(up);
              if (!llvm::is_contained(joins[up], dst))
                joins[up].push_back(dst);
              if (!llvm::is_contained(joinOwners[up], other))
                joinOwners[up].push_back(other);
            }
          }
        auto isPending = [&](int state) {
          return llvm::any_of(pending, [&](const PathEndPoint &p) {
            return dstState(p) == state;
          });
        };
        // Dijkstra from the tree, less the states in `drop`, and not through
        // those in `off`.
        auto search = [&](const llvm::DenseSet<int> &drop,
                          const llvm::DenseSet<int> &off = {}) {
          branchAvoid.clear();
          if (avoid)
            for (const auto &[state, hop] : planned) {
              const auto &[from, e] = hop;
              if (e.sb->srcCoords != e.sb->dstCoords || drop.count(state))
                continue;
              llvm::BitVector &avoid =
                  branchAvoid.try_emplace(from, conflicting.size())
                      .first->second;
              for (int other : e.sb->unitPacketFlows[e.sb->unitOf(e.j)])
                if (other != flow)
                  avoid |= conflicting[other];
              // A flow sharing an arbiter with itself is no conflict.
              avoid.reset(flow);
            }
          seeds.clear();
          seedCosts.clear();
          for (auto [state, hops] : llvm::zip_equal(tree, treeHops))
            if (!drop.count(state) && !off.count(state)) {
              seeds.push_back(state);
              seedCosts.push_back(TREE_SEED_FACTOR * hops);
            }
          llvm::DenseSet<int> blocked;
          if (!off.empty()) {
            blocked = stops;
            blocked.insert(off.begin(), off.end());
          }
          dijkstraShortestPaths(seeds, seedCosts, packetId, avoid, &branchAvoid,
                                off.empty() ? &stops : &blocked);
        };
        // Trace the path Dijkstra found to `currId` back to the tree.
        auto trace = [&](int currId) {
          size_t grown = tree.size();
          while (processedStamp[currId] != curStamp) {
            // If Dijkstra never reached this node it has no predecessor; the
            // destination is unroutable under the current demand.
            if (preds[currId] < 0)
              return false;
            int predId = preds[currId];
            planned.insert({currId, {predId, predEdge[currId]}});
            ++children[predId];
            processedStamp[currId] = curStamp;
            tree.push_back(currId);
            currId = predId;
          }
          // The new hops were traced from the destination back to the tree.
          int hops = treeHops[llvm::find(tree, currId) - tree.begin()] +
                     static_cast<int>(tree.size() - grown);
          while (treeHops.size() < tree.size())
            treeHops.push_back(hops--);
          return true;
        };
        SmallVector<int, 4> reached;
        // The path to `dst` stays off the hops another destination's path
        // takes below where the check split the two apart.
        auto splitOff = [&](int dst) {
          llvm::DenseSet<int> off;
          auto carries = [&](int d, int id) {
            return llvm::is_contained(idsTo(flow, d), id);
          };
          for (auto [at, a, b] : splitsOf[flow])
            for (int other : reached) {
              if (!(carries(dst, a) && !carries(dst, b) && carries(other, b) &&
                    !carries(other, a)) &&
                  !(carries(dst, b) && !carries(dst, a) && carries(other, a) &&
                    !carries(other, b)))
                continue;
              SmallVector<int, 8> below;
              int s = other;
              for (auto hop = planned.find(s); s != at && hop != planned.end();
                   hop = planned.find(s)) {
                below.push_back(s);
                s = hop->second.first;
              }
              if (s == at)
                off.insert(below.begin(), below.end());
            }
          return off;
        };
        SmallVector<std::pair<int, const SmallVector<int, 4> *>, 2> joinedAt;
        SmallVector<std::pair<int, std::pair<int, Edge>>, 8> pinnedJoins;
        SmallVector<int, 4> pinnedJoinDsts;
        if (pin != pinnedTrees.end()) {
          for (const auto &[from, to, joined] : pin->second) {
            bool intra = from.coords == to.coords;
            auto fromId = nodeIds.find(from), toId = nodeIds.find(to);
            const Edge *e = nullptr;
            if (fromId != nodeIds.end() && toId != nodeIds.end())
              for (const Edge &out : adjacency[fromId->second])
                if (out.dst == toId->second &&
                    (out.sb->srcCoords == out.sb->dstCoords) == intra)
                  e = &out;
            if (!e) {
              failureReason = "the route packet flows from " +
                              endpointString(src) +
                              " take alone does not fit this design.";
              return std::nullopt;
            }
            std::pair<int, std::pair<int, Edge>> hop{
                stateId(toId->second, intra ? Out : In),
                {stateId(fromId->second, intra ? In : Out), *e}};
            if (joined)
              pinnedJoins.push_back(hop);
            else
              planned.insert(hop);
          }
          for (const PathEndPoint &p : pending)
            (planned.count(dstState(p)) ? reached : pinnedJoinDsts)
                .push_back(dstState(p));
          pending.clear();
        }
        while (!pending.empty()) {
          search({});
          // The nearest destination joins the tree next, or the nearest join
          // brings every destination below it.
          auto nearest = llvm::min_element(
              pending, [&](const PathEndPoint &a, const PathEndPoint &b) {
                return distance[dstState(a)] < distance[dstState(b)];
              });
          PathEndPoint endPoint = *nearest;
          int currId = dstState(endPoint);
          if (llvm::DenseSet<int> off = splitOff(currId); !off.empty()) {
            search({}, off);
            if (distance[currId] == INF)
              search({});
          }
          const SmallVector<int, 4> *joined = nullptr;
          for (const auto &[at, dsts] : joins)
            if (distance[at] < distance[currId] && !unjoinable.count(at) &&
                llvm::all_of(dsts, isPending)) {
              currId = at;
              joined = &dsts;
            }
          if (joined) {
            llvm::erase_if(pending, [&](const PathEndPoint &p) {
              return llvm::is_contained(*joined, dstState(p));
            });
            joinedAt.push_back({currId, joined});
            // The tree it joins goes on below.
            ++children[currId];
          } else {
            pending.erase(nearest);
            reached.push_back(currId);
          }
          if (!trace(currId)) {
            failureReason = "no path leads from " + endpointString(src) +
                            " to " + endpointString(endPoint) +
                            " through the connections the switchboxes "
                            "allow and existing routing leaves free.";
            return std::nullopt;
          }
        }
        // A destination's path was chosen before the tree reached the later
        // ones, so reroute each destination's own branch from the rest of the
        // tree where that is cheaper.
        if (pin == pinnedTrees.end() && reached.size() + joinedAt.size() > 1)
          for (int dst : reached) {
            llvm::DenseSet<int> branch;
            int top = dst;
            for (int below = 0;
                 top != tree.front() && children.lookup(top) == below;
                 below = 1) {
              branch.insert(top);
              top = planned.find(top)->second.first;
            }
            if (branch.empty())
              continue;
            search(branch, splitOff(dst));
            double cost = TREE_SEED_FACTOR *
                          treeHops[llvm::find(tree, top) - tree.begin()];
            for (int s = dst; s != top;) {
              const auto &[from, e] = planned.find(s)->second;
              auto avoidBranch = branchAvoid.find(from);
              cost += edgeWeight(e, packetId, avoid,
                                 avoidBranch == branchAvoid.end()
                                     ? nullptr
                                     : &avoidBranch->second);
              s = from;
            }
            if (distance[dst] + REROUTE_MIN_SAVING >= cost)
              continue;
            for (int s : branch) {
              processedStamp[s] = 0;
              planned.erase(s);
              children.erase(s);
            }
            --children[top];
            for (size_t k = tree.size(); k-- > 0;)
              if (branch.count(tree[k])) {
                tree.erase(tree.begin() + k);
                treeHops.erase(treeHops.begin() + k);
              }
            (void)trace(dst);
          }
        llvm::DenseMap<int, std::pair<SwitchboxConnect *, int>> branchPort;
        for (const auto &[currId, hop] : planned) {
          const auto &[predId, e] = hop;
          const PathEndPoint &curr = nodes[stateNode(currId)];
          const PathEndPoint &pred = nodes[stateNode(predId)];
          SwitchboxConnect &sb = *e.sb;
          int i = e.i;
          int j = e.j;
          if (packetId)
            treeOf[flow].try_emplace(currId, predId, e);
          sb.isPriority[i][j] = isPriority;
          // Packet flows in the same group may share a channel, but only if
          // their ids differ, so two same-id flows never merge onto a channel
          // and then fan back out to separate destinations.
          // packetGroupId only becomes >= 0 when packetId has a value (see
          // Pathfinder::addFlow), so the dereferences below are safe; the
          // checker just can't correlate the two across this loop's back edge.
          // NOLINTBEGIN(bugprone-unchecked-optional-access)
          bool sameGroupUnseen = packetGroupId >= 0 && packetId.has_value() &&
                                 (sb.packetGroupId[i][j] == -1 ||
                                  sb.packetGroupId[i][j] == packetGroupId) &&
                                 sb.packetIds[i][j].count(*packetId) == 0;
          if (sameGroupUnseen) {
            int packetIdValue = *packetId;
            // NOLINTEND(bugprone-unchecked-optional-access)
            for (size_t k = 0; k < sb.srcPorts.size(); k++) {
              for (size_t l = 0; l < sb.dstPorts.size(); l++) {
                if (k == static_cast<size_t>(i) ||
                    l == static_cast<size_t>(j)) {
                  sb.packetGroupId[k][l] = packetGroupId;
                  sb.packetIds[k][l].insert(packetIdValue);
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
          if (pred.coords == curr.coords) {
            switchSettings[pred.coords].srcs.push_back(pred.port);
            switchSettings[curr.coords].dsts.push_back(curr.port);
            if (packetId) {
              sb.addToUnit(j, flow);
              auto [it, first] = branchPort.try_emplace(predId, &sb, j);
              if (!first)
                sb.joinUnits(it->second.second, j);
            }
          }
        }
        if (packetId) {
          treeDsts[flow].append(reached.begin(), reached.end());
          for (const auto &[currId, hop] : planned)
            packetTrees[src].push_back(
                {nodes[stateNode(hop.first)], nodes[stateNode(currId)], false});
        }
        // Below a join the flow's packets follow the tree they joined.
        auto join = [&](int state, int predId, const Edge &e) {
          if (!treeOf[flow].try_emplace(state, predId, e).second)
            return false;
          const PathEndPoint &pred = nodes[stateNode(predId)];
          const PathEndPoint &curr = nodes[stateNode(state)];
          packetTrees[src].push_back({pred, curr, true});
          if (pred.coords == curr.coords) {
            switchSettings[pred.coords].srcs.push_back(pred.port);
            switchSettings[curr.coords].dsts.push_back(curr.port);
            e.sb->addToUnit(e.j, flow);
          }
          return true;
        };
        for (const auto &[joinAt, joined] : joinedAt) {
          for (int other : joinOwners[joinAt]) {
            joinedHops[flow].push_back(
                {other, planned.find(joinAt)->second.second});
            for (int dst : *joined) {
              const auto &t = treeOf[other];
              SmallVector<std::pair<int, std::pair<int, Edge>>, 8> path;
              auto it = t.find(dst);
              for (; it != t.end() && it->first != joinAt;
                   it = t.find(it->second.first))
                path.push_back(*it);
              if (it == t.end())
                continue;
              for (const auto &[state, hop] : path)
                if (join(state, hop.first, hop.second))
                  joinedHops[flow].push_back({other, hop.second});
            }
          }
          treeDsts[flow].append(joined->begin(), joined->end());
        }
        for (const auto &[state, hop] : pinnedJoins)
          join(state, hop.first, hop.second);
        treeDsts[flow].append(pinnedJoinDsts.begin(), pinnedJoinDsts.end());
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

    // A routing that fits the fabric can still be one the caller cannot use;
    // steer away from the connections it names as if they were overused.
    if (illegalEdges == 0 && routingCheck) {
      RoutingFaults faults = routingCheck(routingSolution);
      for (const TreeSplit &split : faults.splits) {
        auto flow = flowIndex.find(split.src);
        auto at = nodeIds.find(split.at);
        if (flow != flowIndex.end() && at != nodeIds.end() &&
            splitsOf[flow->second]
                .insert({stateId(at->second, In), split.a, split.b})
                .second)
          illegalEdges++;
      }
      for (const auto &[tile, conn] : faults.connections) {
        illegalEdges++;
        auto it = graph.find({tile, tile});
        if (it == graph.end())
          continue;
        SwitchboxConnect &sb = it->second;
        auto i = llvm::find(sb.srcPorts, conn.src);
        auto j = llvm::find(sb.dstPorts, conn.dst);
        if (i == sb.srcPorts.end() || j == sb.dstPorts.end())
          continue;
        sb.overCapacity[i - sb.srcPorts.begin()][j - sb.dstPorts.begin()] +=
            ROUTING_CHECK_PENALTY;
        for (auto [flow, hops] : llvm::enumerate(joinedHops))
          for (const auto &[other, e] : hops)
            if (e.sb == &sb && sb.srcPorts[e.i] == conn.src &&
                sb.dstPorts[e.j] == conn.dst)
              noJoin.insert({static_cast<int>(flow), other});
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
