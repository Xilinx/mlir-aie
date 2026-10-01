//===- AIEPathFinder.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"
#include "aie/Dialect/AIE/Transforms/AIERoutingDiagnostics.h"
#include "d_ary_heap.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include <limits>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-pathfinder"

namespace {
// A connection's demand, the cost Dijkstra weighs it by, is
// (demandBase + overCapacityCoeff * iterations it was over capacity) *
// (demandBase + usedCapacityCoeff * streams using it).
constexpr double overCapacityCoeff = 0.1;
constexpr double usedCapacityCoeff = 0.02;
constexpr double demandBase = 1.0;
// A full connection's demand grows by this factor, or a prioritized flow's
// by priorityDemandCoeff, each time another stream takes it.
constexpr double demandCoeff = 1.1;
constexpr double priorityDemandCoeff =
    static_cast<double>(std::numeric_limits<int>::max());
constexpr int maxCircuitStreamCapacity = 1;
constexpr int maxPacketStreamCapacity = 32;
// History added to a connection each time the routing check rejects it.
constexpr int routingCheckPenalty = 5;
// Cost added to a hop that shares an arbiter unit with a flow to avoid.
constexpr double conflictSharePenalty = 4;
// Channels per direction packet streams leave a capped tile by (see
// Pathfinder::relax). A heuristic: fewer channels mean fewer master
// sets per tile, but more flows on each.
constexpr int packetFanoutCap = 2;
// A multicast's next destination may branch off any hop its tree already
// takes, starting at this cost per hop back to the source: enough of a
// discount to share hops, while still preferring the shortest path to each
// destination.
constexpr double treeSeedFactor = 0.9;
// A destination's branch is rerouted only when that saves more than this.
constexpr double rerouteMinSaving = 1e-6;
} // namespace

void SwitchboxConnect::updateDemand() {
  for (Cell &c : cells)
    c.demand = (demandBase + overCapacityCoeff * c.overCapacity) *
               (demandBase + usedCapacityCoeff * c.usedCapacity);
}

void SwitchboxConnect::bumpDemand(Cell &c) {
  if (c.usedCapacity >= maxCircuitStreamCapacity)
    c.demand *= c.isPriority ? priorityDemandCoeff : demandCoeff;
}

char RoutingFailure::ID = 0;

void RoutingFailure::log(llvm::raw_ostream &os) const {
  os << "Unable to find a legal routing";
  if (!reason.empty())
    os << ": " << reason;
}

llvm::Error DynamicTileAnalysis::runAnalysis(DeviceOp &device) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin DynamicTileAnalysis Constructor---\n");
  // find the maxCol and maxRow
  maxCol = device.getTargetModel().columns();
  maxRow = device.getTargetModel().rows();

  pathfinder.initialize(maxCol, maxRow, device.getTargetModel());

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
          pathfinder.addFlow(srcCoords, srcPort, dstCoords, dstPort,
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
    pathfinder.addFlow(srcCoords, srcPort, dstCoords, dstPort,
                       /*packetId=*/std::nullopt, /*isPriorityFlow=*/false);
  }

  // Canonicalize all flows after both packet and circuit flows are collected.
  pathfinder.sortFlows();

  // add existing connections so Pathfinder knows which resources are
  // available search all existing SwitchBoxOps for exising connections
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    if (failed(pathfinder.addFixedConnection(switchboxOp)))
      return llvm::make_error<RoutingFailure>(
          llvm::formatv("cannot add the fixed connections of the switchbox "
                        "at tile ({0}, {1})",
                        switchboxOp.colIndex(), switchboxOp.rowIndex()),
          RoutingFaults{}, switchboxOp.getLoc());
  }

  // all flows are now populated, call the congestion-aware pathfinder
  // algorithm
  // check whether the pathfinder algorithm creates a legal routing
  llvm::Expected<Routing> found = pathfinder.findPaths(maxIterations);
  if (!found)
    return found.takeError();
  routing = std::move(*found);

  // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
  for (auto tileOp : device.getOps<TileOp>()) {
    [[maybe_unused]] bool fresh =
        coordToTile.try_emplace(tileOp.getTileID(), tileOp).second;
    assert(fresh);
  }
  for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
    [[maybe_unused]] bool fresh =
        coordToSwitchbox
            .try_emplace({switchboxOp.colIndex(), switchboxOp.rowIndex()},
                         switchboxOp)
            .second;
    assert(fresh);
  }
  for (auto shimmuxOp : device.getOps<ShimMuxOp>()) {
    [[maybe_unused]] bool fresh =
        coordToShimMux
            .try_emplace({shimmuxOp.colIndex(), shimmuxOp.rowIndex()},
                         shimmuxOp)
            .second;
    assert(fresh);
  }

  LLVM_DEBUG(llvm::dbgs() << "\t---End DynamicTileAnalysis Constructor---\n");
  return llvm::Error::success();
}

TileOp DynamicTileAnalysis::getTile(OpBuilder &builder, int col, int row) {
  TileOp &tileOp = coordToTile[{col, row}];
  if (!tileOp)
    tileOp = TileOp::create(builder, builder.getUnknownLoc(), col, row);
  return tileOp;
}

TileOp DynamicTileAnalysis::getTile(OpBuilder &builder, const TileID &tileId) {
  return getTile(builder, tileId.col, tileId.row);
}

SwitchboxOp DynamicTileAnalysis::getSwitchbox(OpBuilder &builder, int col,
                                              int row) {
  assert(col >= 0);
  assert(row >= 0);
  if (SwitchboxOp switchboxOp = lookupSwitchbox({col, row}))
    return switchboxOp;
  auto switchboxOp = SwitchboxOp::create(builder, builder.getUnknownLoc(),
                                         getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToSwitchbox[{col, row}] = switchboxOp;
  return switchboxOp;
}

ShimMuxOp DynamicTileAnalysis::getShimMux(OpBuilder &builder, int col) {
  assert(col >= 0);
  if (ShimMuxOp shimMuxOp = lookupShimMux(col))
    return shimMuxOp;
  assert(getTile(builder, col, 0).isShimNOCorPLTile());
  auto shimMuxOp = ShimMuxOp::create(builder, builder.getUnknownLoc(),
                                     getTile(builder, col, 0));
  ShimMuxOp::ensureTerminator(shimMuxOp.getConnections(), builder,
                              builder.getUnknownLoc());
  coordToShimMux[{col, 0}] = shimMuxOp;
  return shimMuxOp;
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
    SwitchboxConnect &sb =
        graph.try_emplace({coords, coords}, coords).first->second;

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
    sb.resize();
    // A shim's mux reaches its DMA, NOC and PLIO ports from any port.
    auto isMuxed = [&](Port p) {
      return targetModel.isShimNOCorPLTile(col, row) &&
             llvm::is_contained(
                 {WireBundle::DMA, WireBundle::NOC, WireBundle::PLIO},
                 p.bundle);
    };
    for (auto [i, pIn] : llvm::enumerate(sb.srcPorts))
      for (auto [j, pOut] : llvm::enumerate(sb.dstPorts))
        sb.at(i, j).available =
            targetModel.isLegalTileConnection(col, row, pIn.bundle, pIn.channel,
                                              pOut.bundle, pOut.channel) ||
            isMuxed(pIn) || isMuxed(pOut);
  };

  auto interconnect = [&](int col, int row, int targetCol, int targetRow,
                          WireBundle srcBundle, WireBundle dstBundle) {
    TileID src = {col, row}, dst = {targetCol, targetRow};
    SwitchboxConnect &sb =
        graph.try_emplace({src, dst}, src, dst).first->second;
    for (int channel = 0; channel < maxChannels[srcBundle]; channel++) {
      sb.srcPorts.push_back(Port{srcBundle, channel});
      sb.dstPorts.push_back(Port{dstBundle, channel});
    }
    sb.resize();
    for (size_t i = 0; i < sb.srcPorts.size(); i++)
      sb.at(i, i).available = true;
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
  // A source has one flow, with all its destinations.
  PathEndPoint src{srcCoords, srcPort}, dst{dstCoords, dstPort};
  auto flow = llvm::find_if(flows, [&](const Flow &f) { return f.src == src; });
  if (flow == flows.end()) {
    // sortFlows assigns the packet groups.
    flows.push_back(
        Flow{packetId ? 0 : -1, isPriorityFlow, src, {dst}, packetId});
  } else if (isPriorityFlow) {
    flow->isPriorityFlow = true;
    flow->dsts.insert(flow->dsts.begin(), dst);
  } else {
    flow->dsts.push_back(dst);
  }
}

bool Pathfinder::setShareChannels(bool share) {
  bool apart = false;
  for (const Flow &f : flows)
    apart |= f.packetGroupId > 0;
  shareChannels = share;
  return share && apart;
}

bool Pathfinder::capCrowdedFanOut() {
  size_t capped = cappedTiles.size();
  cappedTiles.insert(crowdedTiles.begin(), crowdedTiles.end());
  return cappedTiles.size() > capped;
}

bool Pathfinder::routeIdsApart() {
  if (idsApart)
    return false;
  idsApart = true;
  shareChannels = false;
  cappedTiles.clear();
  std::map<PathEndPoint, std::set<int>> ids;
  for (const auto &[ends, sent] : packetIdsTo)
    ids[ends.first].insert(sent.begin(), sent.end());
  return llvm::any_of(ids, [](const auto &s) { return s.second.size() > 1; });
}

bool Pathfinder::relax() {
  if (!packetsFailed) {
    LLVM_DEBUG(llvm::dbgs() << "No packet stream crosses an overused link\n");
    return false;
  }
  for (;;) {
    switch (relaxStep++) {
    case 0:
    case 3:
      if (setShareChannels(true)) {
        LLVM_DEBUG(llvm::dbgs() << "Relax: share channels\n");
        return true;
      }
      break;
    case 1:
    case 4:
      if (capCrowdedFanOut()) {
        LLVM_DEBUG(llvm::dbgs() << "Relax: cap crowded tiles\n");
        return true;
      }
      break;
    case 2:
      if (routeIdsApart()) {
        LLVM_DEBUG(llvm::dbgs() << "Relax: route ids apart\n");
        return true;
      }
      relaxStep = 5;
      return false;
    default:
      return false;
    }
  }
}

// Sort flows to (1) get deterministic routing, and (2) perform routings on
// prioritized flows before others, for routing consistency on those flows.
void Pathfinder::sortFlows() {
  for (auto &flow : flows)
    llvm::sort(flow.dsts);

  // The groups setShareChannels describes, whatever order the flows were
  // added in: packet flows to a common destination are in one.
  llvm::IntEqClasses groups(flows.size());
  std::map<PathEndPoint, unsigned> firstTo;
  for (auto [k, flow] : llvm::enumerate(flows)) {
    if (flow.packetGroupId < 0)
      continue;
    for (const PathEndPoint &dst : flow.dsts) {
      auto [it, fresh] = firstTo.try_emplace(dst, k);
      if (!fresh)
        groups.join(k, it->second);
    }
  }
  llvm::DenseMap<unsigned, int> groupOf;
  for (auto [k, flow] : llvm::enumerate(flows))
    if (flow.packetGroupId >= 0)
      flow.packetGroupId =
          shareChannels
              ? 0
              : groupOf.try_emplace(groups.findLeader(k), groupOf.size())
                    .first->second;

  auto flowRank = [](const Flow &flow) {
    if (flow.isPriorityFlow)
      return 0;
    if (flow.packetGroupId >= 0)
      return 1;
    return 2;
  };
  llvm::sort(flows, [&](const Flow &lhs, const Flow &rhs) {
    return std::make_pair(flowRank(lhs), lhs.src) <
           std::make_pair(flowRank(rhs), rhs.src);
  });
}

int xilinx::AIE::shimMuxChannelFrom(Port src) {
  // DMA0 -> N3, DMA1 -> N7; NOC0/1 -> N2/3, NOC2/3 -> N6/7.
  if (src.bundle == WireBundle::DMA)
    return src.channel == 0 ? 3 : 7;
  if (src.bundle == WireBundle::NOC)
    return src.channel >= 2 ? src.channel + 4 : src.channel + 2;
  return src.channel;
}

int xilinx::AIE::shimMuxChannelTo(Port dst) {
  // N2 -> DMA0, N3 -> DMA1; N2-5 -> NOC0-3.
  if (dst.bundle == WireBundle::DMA)
    return dst.channel == 0 ? 2 : 3;
  if (dst.bundle == WireBundle::NOC)
    return dst.channel + 2;
  return dst.channel;
}

LogicalResult Pathfinder::addFixedConnection(SwitchboxOp switchboxOp) {
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
    int srcIdx = sb.srcIndex(connectOp.sourcePort());
    int dstIdx = sb.dstIndex(connectOp.destPort());
    // Reject an illegal pair (absent from the switchbox model) or a second
    // driver on the same output port; a repeated source port is a broadcast.
    if (srcIdx < 0 || dstIdx < 0 || !sb.at(srcIdx, dstIdx).available ||
        !claimedDsts.insert(dstIdx).second) {
      return failure();
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
    int dstIdx = sb.dstIndex(masterSetOp.destPort());
    // Reject an output port absent from the switchbox model or already driven
    // by a circuit connect or another masterset.
    if (dstIdx < 0 || !claimedDsts.insert(dstIdx).second) {
      return failure();
    }
    reservedMasterDsts.push_back(dstIdx);
  }
  // A circuit-switched ConnectOp monopolizes both its source port (the stream
  // switch input) and its destination port (the output): no other stream may
  // inject on that input, and the output can carry only this one stream.
  // Reserving the whole column also reserves the outgoing wire, since that wire
  // is reachable only by driving this output port.
  for (auto [srcIdx, dstIdx] : reserved) {
    for (size_t j = 0; j < sb.dstPorts.size(); j++)
      sb.at(srcIdx, j).available = false;
    for (size_t i = 0; i < sb.srcPorts.size(); i++)
      sb.at(i, dstIdx).available = false;
  }
  // A masterset fixes only its output port. Its inputs arrive through arbiters
  // that packet flows share, so the source rows stay free.
  for (int dstIdx : reservedMasterDsts)
    for (size_t i = 0; i < sb.srcPorts.size(); i++)
      sb.at(i, dstIdx).available = false;
  // A fixed op on a shim's South channel claims the port the shim mux carries
  // on it (see shimMuxChannelFrom).
  if (switchboxOp.getTileOp().isShimNOCorPLTile()) {
    auto isMuxed = [](Port p) {
      return llvm::is_contained(
          {WireBundle::DMA, WireBundle::NOC, WireBundle::PLIO}, p.bundle);
    };
    llvm::SmallDenseSet<int, 8> southDsts, southSrcs;
    for (int dstIdx : claimedDsts)
      if (sb.dstPorts[dstIdx].bundle == WireBundle::South)
        southDsts.insert(sb.dstPorts[dstIdx].channel);
    for (auto [srcIdx, dstIdx] : reserved)
      if (sb.srcPorts[srcIdx].bundle == WireBundle::South)
        southSrcs.insert(sb.srcPorts[srcIdx].channel);
    for (size_t j = 0; j < sb.dstPorts.size(); j++)
      if (isMuxed(sb.dstPorts[j]) &&
          southDsts.count(shimMuxChannelTo(sb.dstPorts[j])))
        for (size_t i = 0; i < sb.srcPorts.size(); i++)
          sb.at(i, j).available = false;
    for (size_t i = 0; i < sb.srcPorts.size(); i++)
      if (isMuxed(sb.srcPorts[i]) &&
          southSrcs.count(shimMuxChannelFrom(sb.srcPorts[i])))
        for (size_t j = 0; j < sb.dstPorts.size(); j++)
          sb.at(i, j).available = false;
  }
  for (PacketRulesOp rulesOp : switchboxOp.getOps<PacketRulesOp>()) {
    int srcIdx = sb.srcIndex(rulesOp.sourcePort());
    if (srcIdx < 0)
      return failure();
    sb.packetOnlySrc.set(srcIdx);
  }
  return success();
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
    auto intraIt = graph.find({src.coords, src.coords});
    if (intraIt != graph.end()) {
      auto &sb = intraIt->second;
      for (auto [i, pIn] : llvm::enumerate(sb.srcPorts))
        for (auto [j, pOut] : llvm::enumerate(sb.dstPorts))
          if (pIn == src.port && sb.at(i, j).available)
            dests.emplace_back(src.coords, pOut);
    }
    std::pair<TileID, Port> neighbors[] = {
        {{src.coords.col, src.coords.row - 1},
         {WireBundle::North, src.port.channel}},
        {{src.coords.col - 1, src.coords.row},
         {WireBundle::East, src.port.channel}},
        {{src.coords.col, src.coords.row + 1},
         {WireBundle::South, src.port.channel}},
        {{src.coords.col + 1, src.coords.row},
         {WireBundle::West, src.port.channel}}};
    for (const auto &[neighborCoords, neighborPort] : neighbors) {
      auto nIt = graph.find({src.coords, neighborCoords});
      if (nIt != graph.end() &&
          src.port.bundle == getConnectingBundle(neighborPort.bundle) &&
          llvm::is_contained(nIt->second.dstPorts, neighborPort))
        dests.emplace_back(neighborCoords, neighborPort);
    }
    llvm::sort(dests);

    std::vector<Edge> edges;
    edges.reserve(dests.size());
    for (auto &dest : dests) {
      auto &sb = graph.at({src.coords, dest.coords});
      int i = sb.srcIndex(src.port);
      int j = sb.dstIndex(dest.port);
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

double Pathfinder::edgeWeight(const Edge &e, std::optional<int> packetId,
                              const llvm::BitVector *avoid,
                              const llvm::BitVector *avoidBranch) const {
  const SwitchboxConnect::Cell &cell = e.sb->at(e.i, e.j);
  double w = cell.demand;
  // Sharing it would take a second stream's capacity, which the demand
  // only shows once the group is routed.
  if (packetId && cell.packetFlowCount > 0 && cell.packetIds.count(*packetId))
    w *= demandCoeff;
  if (e.sb->srcCoords == e.sb->dstCoords)
    for (const llvm::BitVector *flows : {avoid, avoidBranch})
      if (flows && llvm::any_of(e.sb->unitFlows(e.j),
                                [&](int f) { return flows->test(f); }))
        w += conflictSharePenalty;
  return w;
}

// Dijkstra over the dense graph from the states in `seeds`, searching states
// (node, PortSide) rather than bare nodes. Fills the `preds` and `predEdge`
// scratch buffers, both indexed by state id.
void Pathfinder::dijkstraShortestPaths(
    ArrayRef<int> seeds, ArrayRef<double> seedCosts,
    std::optional<int> packetId, const llvm::BitVector *avoid,
    const llvm::DenseMap<int, llvm::BitVector> *branchAvoid,
    const llvm::DenseSet<int> *stops, ArrayRef<int> targets) {
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
  // A settled state's distance and predecessor never change again.
  SmallVector<int, 8> unsettled(targets);
  while (!Q.empty()) {
    int s = Q.top();
    Q.pop();
    if (llvm::is_contained(unsettled, s)) {
      llvm::erase(unsettled, s);
      if (unsettled.empty())
        break;
    }
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
          (isIntra && !packetId && e.sb->packetOnlySrc[e.i]) ||
          (isIntra && packetId && e.sb->circuitOnlyDst[e.j]))
        continue;
      int dst = stateId(e.dst, isIntra ? Out : In);
      double w = edgeWeight(e, packetId, avoid, avoidBranch);
      if (colors[dst] == BLACK || distance[s] + w >= distance[dst])
        continue;
      distance[dst] = distance[s] + w;
      preds[dst] = s;
      predEdge[dst] = e;
      if (colors[dst] == WHITE) {
        colors[dst] = GRAY;
        Q.push(dst);
      } else {
        Q.update(dst);
      }
    }
    colors[s] = BLACK;
  }
}

// What findPaths keeps from one flow and iteration to the next.
struct Pathfinder::RouteState {
  explicit RouteState(const Pathfinder &pf);

  int groupOf(const Flow &f) const;
  SmallVector<int, 4> idsTo(int flow, int dstState) const;
  void findSameIds();
  int splitPart(int flow, int id);

  const Pathfinder &pf;
  Routing routing;
  // Stamp-based "processed" set (avoids O(n) clears per flow).
  std::vector<uint32_t> processedStamp;
  uint32_t curStamp = 0;
  // The flows to route: one per source, less the packets split off into parts
  // of their own (see below), which route as flows from the same source.
  std::vector<Flow> parts;
  std::map<PathEndPoint, SmallVector<int, 2>> partsOf;
  // group flows based on packetGroupId; pinned trees go in first, so the
  // flows routed around them see them.
  llvm::MapVector<int, SmallVector<int, 8>> groupedFlows;
  // Packet flows that conflict.
  std::vector<llvm::BitVector> conflicting;
  // The packet ids each flow carries, in all and to each destination.
  std::vector<std::set<int>> flowIds;
  std::vector<SmallVector<int, 4>> sameId;
  // Each packet flow's tree as routed so far this iteration: the state each
  // hop reaches, from which state and by which edge; and its destinations.
  std::vector<llvm::DenseMap<int, std::pair<int, Edge>>> treeOf;
  std::vector<SmallVector<int, 4>> treeDsts;
  // The hops each flow takes only by joining another flow's tree, by the flow
  // it joined. A join the routing check faults is not made again.
  std::vector<SmallVector<std::pair<int, Edge>, 8>> joinedHops;
  std::set<std::pair<int, int>> noJoin;
  // Where the routing check split each flow's tree: the tile it branches at,
  // the ids to branch apart there, and the state of the destination that
  // reaches the tile apart, or -1.
  std::vector<std::set<std::tuple<TileID, int, int, int>>> splitsOf;
  // A split one tree cannot make, since a destination takes both ids, moves
  // the second id to a part of its own. Parts of a source split apart at a
  // tile leave it by different master ports: the part routed later keeps off
  // those the other's tree takes there.
  std::vector<std::set<std::pair<TileID, int>>> partSplits;
};

Pathfinder::RouteState::RouteState(const Pathfinder &pf)
    : pf(pf), processedStamp(2 * pf.nodes.size(), 0), parts(pf.flows),
      conflicting(parts.size(), llvm::BitVector(parts.size())),
      flowIds(parts.size()), treeOf(parts.size()), treeDsts(parts.size()),
      joinedHops(parts.size()), splitsOf(parts.size()),
      partSplits(parts.size()) {
  for (auto [k, f] : llvm::enumerate(parts))
    partsOf[f.src].push_back(k);
  for (auto [k, f] : llvm::enumerate(parts))
    groupedFlows[groupOf(f)].push_back(k);
  if (pf.constraints.conflict)
    for (size_t a = 0; a < parts.size(); a++)
      for (size_t b = a + 1; b < parts.size(); b++)
        if (parts[a].packetId && parts[b].packetId &&
            pf.constraints.conflict(parts[a].src, parts[b].src)) {
          conflicting[a].set(b);
          conflicting[b].set(a);
        }
  for (const auto &[ends, ids] : pf.packetIdsTo)
    flowIds[partsOf.at(ends.first).front()].insert(ids.begin(), ids.end());
  findSameIds();
}

int Pathfinder::RouteState::groupOf(const Flow &f) const {
  return pf.constraints.pinned.count(f.src) ? std::numeric_limits<int>::min()
                                            : f.packetGroupId;
}

SmallVector<int, 4> Pathfinder::RouteState::idsTo(int flow,
                                                  int dstState) const {
  SmallVector<int, 4> ids;
  auto it =
      pf.packetIdsTo.find({parts[flow].src, pf.nodes[stateNode(dstState)]});
  if (it != pf.packetIdsTo.end())
    for (int id : it->second)
      if (flowIds[flow].count(id))
        ids.push_back(id);
  return ids;
}

void Pathfinder::RouteState::findSameIds() {
  sameId.assign(parts.size(), {});
  for (size_t a = 0; a < parts.size(); a++)
    for (size_t b = 0; b < parts.size(); b++)
      if (a != b && llvm::any_of(flowIds[a],
                                 [&](int id) { return flowIds[b].count(id); }))
        sameId[a].push_back(b);
}

int Pathfinder::RouteState::splitPart(int flow, int id) {
  int part = parts.size();
  Flow f = parts[flow];
  LLVM_DEBUG(llvm::dbgs() << "\t\tRouting id " << id << " from "
                          << describeTilePort(f.src.coords, f.src.port)
                          << " apart\n");
  flowIds[flow].erase(id);
  flowIds.push_back({id});
  auto carries = [&](int k, const PathEndPoint &p) {
    auto it = pf.packetIdsTo.find({f.src, p});
    return it != pf.packetIdsTo.end() &&
           llvm::any_of(it->second, [&](int i) { return flowIds[k].count(i); });
  };
  llvm::erase_if(f.dsts,
                 [&](const PathEndPoint &p) { return !carries(part, p); });
  llvm::erase_if(parts[flow].dsts,
                 [&](const PathEndPoint &p) { return !carries(flow, p); });
  f.packetId = id;
  if (parts[flow].packetId == id)
    parts[flow].packetId = *flowIds[flow].begin();
  parts.push_back(f);
  partsOf[f.src].push_back(part);
  groupedFlows[groupOf(f)].push_back(part);
  for (llvm::BitVector &row : conflicting)
    row.resize(parts.size());
  conflicting.push_back(conflicting[flow]);
  for (int k : conflicting[part].set_bits())
    conflicting[k].set(part);
  findSameIds();
  treeOf.emplace_back();
  treeDsts.emplace_back();
  joinedHops.emplace_back();
  for (auto [a, b] : std::set<std::pair<int, int>>(noJoin))
    if (a == flow || b == flow)
      noJoin.insert({a == flow ? part : a, b == flow ? part : b});
  splitsOf.push_back(splitsOf[flow]);
  partSplits.push_back(partSplits[flow]);
  for (auto [at, other] : partSplits[flow])
    partSplits[other].insert({at, part});
  for (auto [at, a, b, apart] : splitsOf[flow])
    if ((a == id || b == id) && flowIds[flow].count(a == id ? b : a)) {
      partSplits[flow].insert({at, part});
      partSplits[part].insert({at, flow});
    }
  return part;
}

// A link with a channel no flow takes and a free master port driving it
// was overused only by the way the flows went; its flows could spread.
bool Pathfinder::hasRoom(const SwitchboxConnect &sb) const {
  if (sb.srcCoords == sb.dstCoords)
    return true;
  auto xbar = graph.find({sb.srcCoords, sb.srcCoords});
  for (size_t i = 0; i < sb.srcPorts.size(); i++)
    for (size_t j = 0; j < sb.dstPorts.size(); j++) {
      const SwitchboxConnect::Cell &c = sb.at(i, j);
      if (!c.available || c.usedCapacity > 0)
        continue;
      if (xbar == graph.end())
        return true;
      const SwitchboxConnect &x = xbar->second;
      int k = x.dstIndex(sb.srcPorts[i]);
      if (k < 0)
        return true;
      for (size_t row = 0; row < x.srcPorts.size(); row++)
        if (x.at(row, k).available)
          return true;
    }
  return false;
}

std::string Pathfinder::explainNoRouting(const RouteState &st) const {
  // A prioritized flow keeps the route it takes alone, so the others
  // may have had to fit around it.
  auto overused = [](const SwitchboxConnect &sb) {
    return sb.srcCoords != sb.dstCoords &&
           llvm::any_of(sb.cells, [](const SwitchboxConnect::Cell &c) {
             return c.usedCapacity > maxCircuitStreamCapacity;
           });
  };
  auto link = [](const SwitchboxConnect &sb) -> std::string {
    return llvm::formatv("from tile ({0}, {1}) to ({2}, {3})", sb.srcCoords.col,
                         sb.srcCoords.row, sb.dstCoords.col, sb.dstCoords.row);
  };
  const Flow *prioritized = nullptr;
  llvm::SetVector<const SwitchboxConnect *> held;
  for (auto [k, f] : llvm::enumerate(st.parts)) {
    if (!f.isPriorityFlow)
      continue;
    prioritized = prioritized ? prioritized : &f;
    for (const auto &[_, hop] : st.treeOf[k]) {
      if (overused(*hop.second.sb)) {
        return describePrioritized(describeTilePort(f.src.coords, f.src.port)) +
               ", and it holds a channel " + link(*hop.second.sb) +
               " the other flows need.";
      }
      if (hop.second.sb->srcCoords != hop.second.sb->dstCoords)
        held.insert(hop.second.sb);
    }
  }
  if (prioritized && llvm::any_of(graph, [&](const auto &entry) {
        return overused(entry.second);
      })) {
    return describePrioritized(describeTilePort(prioritized->src.coords,
                                                prioritized->src.port)) +
           ", and the router found no routing for the other flows around the "
           "channels it holds " +
           joinNames(llvm::to_vector(
               llvm::map_range(llvm::make_pointee_range(held), link))) +
           ".";
  }
  // The routing check says more than the overuse it led to.
  if (!checkReason.empty())
    return checkReason;
  // Name the channel the last iteration overused that was overused in the
  // most iterations, on a link with no room if there is one, and the flows
  // the last one routed through it.
  const SwitchboxConnect *worst = nullptr;
  int worstI = 0, worstJ = 0;
  std::pair<bool, int> worstRank{false, 0};
  for (const auto &[_, sb] : graph) {
    std::optional<bool> roomless;
    for (size_t i = 0; i < sb.srcPorts.size(); i++)
      for (size_t j = 0; j < sb.dstPorts.size(); j++) {
        if (sb.at(i, j).usedCapacity <= maxCircuitStreamCapacity)
          continue;
        if (!roomless)
          roomless = !hasRoom(sb);
        std::pair<bool, int> rank{*roomless, sb.at(i, j).overCapacity};
        if (rank > worstRank) {
          worst = &sb;
          worstI = i;
          worstJ = j;
          worstRank = rank;
        }
      }
  }
  if (!worst)
    return {};
  bool crossbar = worst->srcCoords == worst->dstCoords;
  std::vector<std::string> users;
  for (const auto &[src, settings] : st.routing.settings) {
    auto it = settings.find(worst->srcCoords);
    if (it == settings.end())
      continue;
    const SwitchSetting &s = it->second;
    bool uses = false;
    for (size_t k = 0; k < s.dsts.size() && !uses; k++)
      uses = crossbar
                 ? k < s.srcs.size() && s.srcs[k] == worst->srcPorts[worstI] &&
                       s.dsts[k] == worst->dstPorts[worstJ]
                 : llvm::is_contained(worst->srcPorts, s.dsts[k]);
    if (uses)
      users.push_back(describeTilePort(src.coords, src.port));
  }
  std::string where =
      crossbar ? llvm::formatv("the connection from {0} to {1} at tile ({2}, "
                               "{3})",
                               describePort(worst->srcPorts[worstI]),
                               describePort(worst->dstPorts[worstJ]),
                               worst->srcCoords.col, worst->srcCoords.row)
                     .str()
               : "the links " + link(*worst);
  if (users.empty()) {
    return "the router found no routing that fits " + where + ".";
  }
  return "the flows from " + joinNames(users) + " need " + where +
         ", and the router found no routing that fits them.";
}

// One flow's tree, grown by routePart.
struct Pathfinder::TreeBuilder {
  TreeBuilder(Pathfinder &pf, RouteState &st, int flow);

  // A destination port is driven by its switchbox, so it is reached on
  // the Out side.
  int dstState(const PathEndPoint &p) const {
    return stateId(pf.nodeIds.at(p), Out);
  }
  bool isPending(int state) const;
  bool joinable(int at, ArrayRef<int> dsts) const;
  void findJoins();
  void search(const llvm::DenseSet<int> &drop, ArrayRef<int> targets,
              const llvm::DenseSet<int> &off = {});
  bool trace(int state);
  llvm::DenseSet<int> splitOff(int dst) const;
  llvm::Error placePinned();
  llvm::Error grow();
  void reroute();
  void claim();
  bool join(int state, int predId, const Edge &e);
  void joinTrees();
  void record();

  Pathfinder &pf;
  RouteState &st;
  int flow;
  const Flow &part;
  const llvm::BitVector *avoid;
  // The route the flow takes alone, if it is pinned to it.
  const std::vector<TreeHop> *pinned = nullptr;
  SwitchSettings switchSettings;
  // The states the tree reaches, and how many hops from the source each is.
  SmallVector<int, 16> tree, treeHops;
  SmallVector<int, 16> seeds;
  SmallVector<double, 16> seedCosts;
  // The tree's hops, by the state each reaches, from which state and by
  // which edge, in the order they were traced. They take effect once the
  // tree is final.
  llvm::MapVector<int, std::pair<int, Edge>> planned;
  llvm::DenseMap<int, int> children;
  // A branch off a port the tree already crosses joins that port's unit
  // (see planArbiters), so it avoids the flows conflicting with any flow
  // on those arbiters too.
  llvm::DenseMap<int, llvm::BitVector> branchAvoid;
  SmallVector<PathEndPoint, 4> pending;
  // A switchbox routes on the id alone, so packets that reach a master
  // port another source's tree takes an id they share by go wherever
  // that id goes from there. The flow may join the tree there only if
  // it still has to reach each of those destinations, with just the ids
  // they share.
  llvm::DenseMap<int, SmallVector<int, 4>> joins;
  llvm::DenseMap<int, SmallVector<int, 2>> joinOwners;
  llvm::DenseSet<int> stops, unjoinable;
  // The master ports the source's other parts take at the tiles the flow
  // was split apart from them at.
  llvm::DenseSet<int> partOff;
  SmallVector<int, 4> reached;
  SmallVector<std::pair<int, const SmallVector<int, 4> *>, 2> joinedAt;
  SmallVector<std::pair<int, std::pair<int, Edge>>, 8> pinnedJoins;
  SmallVector<int, 4> pinnedJoinDsts;
};

Pathfinder::TreeBuilder::TreeBuilder(Pathfinder &pf, RouteState &st, int flow)
    : pf(pf), st(st), flow(flow), part(st.parts[flow]),
      avoid(part.packetId ? &st.conflicting[flow] : nullptr) {
  if (auto pin = pf.constraints.pinned.find(part.src);
      pin != pf.constraints.pinned.end())
    pinned = &pin->second;
  // The flow source port feeds into its switchbox, so the tree starts
  // on its In side and the first edge taken is necessarily a crossbar
  // hop.
  int srcId = pf.nodeIds.at(part.src);
  tree.push_back(stateId(srcId, In));
  treeHops.push_back(0);
  st.processedStamp[tree.front()] = ++st.curStamp;
  for (const PathEndPoint &endPoint : part.dsts) {
    // Route to self: the port is both ends. Where its switchbox cannot
    // connect it to itself (Core to Core), the stream has to leave and
    // come back, which Dijkstra finds from the In side to the Out side.
    if (endPoint == part.src &&
        llvm::any_of(pf.adjacency[srcId],
                     [&](const Edge &e) { return e.dst == srcId; })) {
      switchSettings[part.src.coords].srcs.push_back(part.src.port);
      switchSettings[part.src.coords].dsts.push_back(part.src.port);
      continue;
    }
    pending.push_back(endPoint);
  }
  findJoins();
  for (auto [at, other] : st.partSplits[flow])
    for (const auto &[state, _] : st.treeOf[other])
      if ((state & 1) == Out && pf.nodes[stateNode(state)].coords == at)
        partOff.insert(state);
}

bool Pathfinder::TreeBuilder::isPending(int state) const {
  return llvm::any_of(
      pending, [&](const PathEndPoint &p) { return dstState(p) == state; });
}

bool Pathfinder::TreeBuilder::joinable(int at, ArrayRef<int> dsts) const {
  return !unjoinable.count(at) &&
         llvm::all_of(dsts, [&](int dst) { return isPending(dst); });
}

void Pathfinder::TreeBuilder::findJoins() {
  auto shared = [&](ArrayRef<int> ids, int other) {
    std::set<int> common;
    for (int id : ids)
      if (st.flowIds[other].count(id))
        common.insert(id);
    return common;
  };
  for (int other : st.sameId[flow])
    for (int dst : st.treeDsts[other]) {
      std::set<int> common = shared(st.idsTo(other, dst), flow);
      if (common.empty())
        continue;
      // Joining shares the other flow's arbiters below the join. A
      // prioritized flow's tree is pinned, so it joins only trees
      // pinned with it.
      bool agree = common == shared(st.idsTo(flow, dst), other) &&
                   !st.conflicting[flow].test(other) &&
                   !st.noJoin.count({flow, other}) &&
                   (!part.isPriorityFlow || st.parts[other].isPriorityFlow);
      for (auto it = st.treeOf[other].find(dst); it != st.treeOf[other].end();
           it = st.treeOf[other].find(it->second.first)) {
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
}

// Dijkstra from the tree, less the states in `drop`, and not through
// those in `off`, until the states in `targets` are settled.
void Pathfinder::TreeBuilder::search(const llvm::DenseSet<int> &drop,
                                     ArrayRef<int> targets,
                                     const llvm::DenseSet<int> &off) {
  branchAvoid.clear();
  llvm::DenseMap<int, llvm::BitVector> onArbiter;
  if (avoid)
    for (const auto &[state, hop] : planned) {
      const auto &[from, e] = hop;
      if (e.sb->srcCoords != e.sb->dstCoords || drop.count(state))
        continue;
      llvm::BitVector &away =
          branchAvoid.try_emplace(from, st.conflicting.size()).first->second;
      llvm::BitVector &on =
          onArbiter.try_emplace(from, st.conflicting.size()).first->second;
      for (int other : e.sb->unitFlows(e.j)) {
        on.set(other);
        if (other != flow)
          away |= st.conflicting[other];
      }
    }
  // A flow sharing an arbiter with itself is no conflict, and flows
  // already on those arbiters share them whatever the branch does.
  for (auto &[from, away] : branchAvoid) {
    away.reset(onArbiter.find(from)->second);
    away.reset(flow);
  }
  seeds.clear();
  seedCosts.clear();
  for (auto [state, hops] : llvm::zip_equal(tree, treeHops))
    if (!drop.count(state) && !off.count(state)) {
      seeds.push_back(state);
      seedCosts.push_back(treeSeedFactor * hops);
    }
  llvm::DenseSet<int> blocked;
  if (!off.empty()) {
    blocked = stops;
    blocked.insert(off.begin(), off.end());
  }
  pf.dijkstraShortestPaths(seeds, seedCosts, part.packetId, avoid, &branchAvoid,
                           off.empty() ? &stops : &blocked, targets);
}

// Trace the path Dijkstra found to `state` back to the tree.
bool Pathfinder::TreeBuilder::trace(int state) {
  size_t grown = tree.size();
  while (st.processedStamp[state] != st.curStamp) {
    // If Dijkstra never reached this node it has no predecessor; the
    // destination is unroutable under the current demand.
    int pred = pf.preds[state];
    if (pred < 0)
      return false;
    planned.insert({state, {pred, pf.predEdge[state]}});
    ++children[pred];
    st.processedStamp[state] = st.curStamp;
    tree.push_back(state);
    state = pred;
  }
  // The new hops were traced from the destination back to the tree.
  int hops = treeHops[llvm::find(tree, state) - tree.begin()] +
             static_cast<int>(tree.size() - grown);
  while (treeHops.size() < tree.size())
    treeHops.push_back(hops--);
  return true;
}

// The path to `dst` stays off the hops another destination's path
// takes below where the check split the two apart, and off the
// slave port it takes there too if one of them has to reach the tile
// on its own.
llvm::DenseSet<int> Pathfinder::TreeBuilder::splitOff(int dst) const {
  llvm::DenseSet<int> off = partOff;
  auto carries = [&](int d, int id) {
    return llvm::is_contained(st.idsTo(flow, d), id);
  };
  auto only = [&](int d, int id, int other) {
    return carries(d, id) && !carries(d, other);
  };
  for (auto [at, a, b, apart] : st.splitsOf[flow])
    for (int other : reached) {
      bool splits = apart < 0 ? (only(dst, a, b) && only(other, b, a)) ||
                                    (only(dst, b, a) && only(other, a, b))
                              : (only(dst, a, b) && other == apart) ||
                                    (dst == apart && only(other, a, b));
      if (!splits)
        continue;
      SmallVector<int, 8> below;
      int s = other;
      auto entersAt = [&, tile = at](int state) {
        return (state & 1) == In && pf.nodes[stateNode(state)].coords == tile;
      };
      for (auto *hop = planned.find(s); !entersAt(s) && hop != planned.end();
           hop = planned.find(s)) {
        below.push_back(s);
        s = hop->second.first;
      }
      if (!entersAt(s))
        continue;
      off.insert(below.begin(), below.end());
      if (apart >= 0)
        off.insert(s);
    }
  return off;
}

// Lay the tree along the route the flow takes alone.
llvm::Error Pathfinder::TreeBuilder::placePinned() {
  for (const auto &[from, to, joined] : *pinned) {
    bool intra = from.coords == to.coords;
    auto fromId = pf.nodeIds.find(from), toId = pf.nodeIds.find(to);
    const Edge *e = nullptr;
    if (fromId != pf.nodeIds.end() && toId != pf.nodeIds.end())
      for (const Edge &out : pf.adjacency[fromId->second])
        if (out.dst == toId->second &&
            (out.sb->srcCoords == out.sb->dstCoords) == intra)
          e = &out;
    if (!e)
      return llvm::make_error<RoutingFailure>(
          "the route packet flows from " +
          describeTilePort(part.src.coords, part.src.port) +
          " take alone does not fit this design.");
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
  return llvm::Error::success();
}

// Grow the tree one destination at a time: Dijkstra, given the current
// demand, from everything the tree reaches so far to the next destination,
// whose path is then traced back to the tree. Growing from the tree rather
// than the source lets destinations share hops; see treeSeedFactor for what
// a branch off the tree costs.
llvm::Error Pathfinder::TreeBuilder::grow() {
  while (!pending.empty()) {
    SmallVector<int, 8> targets;
    for (const PathEndPoint &p : pending)
      targets.push_back(dstState(p));
    for (const auto &[at, dsts] : joins)
      if (joinable(at, dsts))
        targets.push_back(at);
    search({}, targets);
    // The nearest destination joins the tree next, or the nearest join
    // brings every destination below it.
    auto *nearest = llvm::min_element(
        pending, [&](const PathEndPoint &a, const PathEndPoint &b) {
          return pf.distance[dstState(a)] < pf.distance[dstState(b)];
        });
    PathEndPoint endPoint = *nearest;
    int currId = dstState(endPoint);
    if (llvm::DenseSet<int> off = splitOff(currId); !off.empty()) {
      search({}, targets, off);
      if (pf.distance[currId] == INF)
        search({}, targets);
    }
    const SmallVector<int, 4> *joined = nullptr;
    for (const auto &[at, dsts] : joins)
      if (pf.distance[at] < pf.distance[currId] && joinable(at, dsts)) {
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
    if (!trace(currId))
      return llvm::make_error<RoutingFailure>(
          "no path leads from " +
          describeTilePort(part.src.coords, part.src.port) + " to " +
          describeTilePort(endPoint.coords, endPoint.port) +
          " through the connections the switchboxes allow and existing "
          "routing leaves free.");
  }
  return llvm::Error::success();
}

// A destination's path was chosen before the tree reached the later
// ones, so reroute each destination's own branch from the rest of the
// tree where that is cheaper.
void Pathfinder::TreeBuilder::reroute() {
  for (int dst : reached) {
    llvm::DenseSet<int> branch;
    int top = dst;
    for (int below = 0; top != tree.front() && children.lookup(top) == below;
         below = 1) {
      branch.insert(top);
      top = planned.find(top)->second.first;
    }
    if (branch.empty())
      continue;
    search(branch, dst, splitOff(dst));
    double cost =
        treeSeedFactor * treeHops[llvm::find(tree, top) - tree.begin()];
    for (int s = dst; s != top;) {
      const auto &[from, e] = planned.find(s)->second;
      auto away = branchAvoid.find(from);
      cost +=
          pf.edgeWeight(e, part.packetId, avoid,
                        away == branchAvoid.end() ? nullptr : &away->second);
      s = from;
    }
    if (pf.distance[dst] + rerouteMinSaving >= cost)
      continue;
    for (int s : branch) {
      st.processedStamp[s] = 0;
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
}

// Take the channels the tree's hops cross.
void Pathfinder::TreeBuilder::claim() {
  const std::optional<int> &packetId = part.packetId;
  int packetGroupId = part.packetGroupId;
  llvm::DenseMap<int, std::pair<SwitchboxConnect *, int>> branchPort;
  for (const auto &[currId, hop] : planned) {
    const auto &[predId, e] = hop;
    const PathEndPoint &curr = pf.nodes[stateNode(currId)];
    const PathEndPoint &pred = pf.nodes[stateNode(predId)];
    SwitchboxConnect &sb = *e.sb;
    int i = e.i;
    int j = e.j;
    SwitchboxConnect::Cell &cell = sb.at(i, j);
    if (packetId)
      st.treeOf[flow].try_emplace(currId, predId, e);
    cell.isPriority = part.isPriorityFlow;
    // Packet flows in the same group may share a channel, but only if
    // their ids differ, so two same-id flows never merge onto a channel
    // and then fan back out to separate destinations. The flow's own
    // tree branching at a port never merges back.
    // packetGroupId only becomes >= 0 when packetId has a value (see
    // Pathfinder::addFlow), so the dereferences below are safe; the
    // checker just can't correlate the two across this loop's back edge.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    auto seen =
        packetId ? cell.packetIds.find(*packetId) : cell.packetIds.end();
    bool sameGroupUnseen =
        packetGroupId >= 0 && packetId.has_value() &&
        (cell.packetGroupId == -1 || cell.packetGroupId == packetGroupId) &&
        (seen == cell.packetIds.end() || seen->second == flow);
    if (sameGroupUnseen) {
      int packetIdValue = *packetId;
      // NOLINTEND(bugprone-unchecked-optional-access)
      auto claim = [&](SwitchboxConnect::Cell &c) {
        c.packetGroupId = packetGroupId;
        c.packetIds.try_emplace(packetIdValue, flow);
      };
      for (size_t k = 0; k < sb.srcPorts.size(); k++)
        claim(sb.at(k, j));
      for (size_t l = 0; l < sb.dstPorts.size(); l++)
        if (l != static_cast<size_t>(j))
          claim(sb.at(i, l));
      // maximum packet stream sharing per channel
      if (++cell.packetFlowCount >= maxPacketStreamCapacity) {
        cell.packetFlowCount = 0;
        cell.usedCapacity++;
      }
    } else {
      cell.usedCapacity++;
    }
    // if at capacity, bump demand to discourage using this Channel
    // this means the order matters!
    SwitchboxConnect::bumpDemand(cell);
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
    st.treeDsts[flow].append(reached.begin(), reached.end());
    for (const auto &[currId, hop] : planned)
      st.routing.packetTrees[part.src].push_back(
          {pf.nodes[stateNode(hop.first)], pf.nodes[stateNode(currId)], false});
  }
}

// Below a join the flow's packets follow the tree they joined.
bool Pathfinder::TreeBuilder::join(int state, int predId, const Edge &e) {
  if (!st.treeOf[flow].try_emplace(state, predId, e).second)
    return false;
  const PathEndPoint &pred = pf.nodes[stateNode(predId)];
  const PathEndPoint &curr = pf.nodes[stateNode(state)];
  st.routing.packetTrees[part.src].push_back({pred, curr, true});
  if (pred.coords == curr.coords) {
    switchSettings[pred.coords].srcs.push_back(pred.port);
    switchSettings[curr.coords].dsts.push_back(curr.port);
    e.sb->addToUnit(e.j, flow);
  }
  return true;
}

void Pathfinder::TreeBuilder::joinTrees() {
  for (const auto &[joinAt, joined] : joinedAt) {
    for (int other : joinOwners[joinAt]) {
      st.joinedHops[flow].push_back(
          {other, planned.find(joinAt)->second.second});
      for (int dst : *joined) {
        const auto &t = st.treeOf[other];
        SmallVector<std::pair<int, std::pair<int, Edge>>, 8> path;
        auto it = t.find(dst);
        for (; it != t.end() && it->first != joinAt;
             it = t.find(it->second.first))
          path.push_back(*it);
        if (it == t.end())
          continue;
        for (const auto &[state, hop] : path)
          if (join(state, hop.first, hop.second))
            st.joinedHops[flow].push_back({other, hop.second});
      }
    }
    st.treeDsts[flow].append(joined->begin(), joined->end());
  }
  for (const auto &[state, hop] : pinnedJoins)
    join(state, hop.first, hop.second);
  st.treeDsts[flow].append(pinnedJoinDsts.begin(), pinnedJoinDsts.end());
}

// Add the tree to the routing.
void Pathfinder::TreeBuilder::record() {
  const PathEndPoint &src = part.src;
  if (st.partsOf.at(src).size() == 1) {
    st.routing.settings[src] = switchSettings;
    return;
  }
  for (int id : st.flowIds[flow])
    st.routing.idSettings[{src, id}] = switchSettings;
  for (const auto &[tile, setting] : switchSettings) {
    SwitchSetting &all = st.routing.settings[src][tile];
    for (auto [in, out] : llvm::zip(setting.srcs, setting.dsts))
      if (!llvm::is_contained(llvm::zip(all.srcs, all.dsts),
                              std::make_tuple(in, out))) {
        all.srcs.push_back(in);
        all.dsts.push_back(out);
      }
  }
}

llvm::Error Pathfinder::routePart(RouteState &st, int flow) {
  TreeBuilder tree(*this, st, flow);
  if (llvm::Error err = tree.pinned ? tree.placePinned() : tree.grow())
    return err;
  if (!tree.pinned && tree.reached.size() + tree.joinedAt.size() > 1)
    tree.reroute();
  tree.claim();
  tree.joinTrees();
  tree.record();
  return llvm::Error::success();
}

int Pathfinder::applyRoutingFaults(RouteState &st,
                                   const RoutingFaults &faults) {
  int illegalEdges = 0;
  for (const TreeSplit &split : faults.splits) {
    auto src = st.partsOf.find(split.src);
    if (src == st.partsOf.end())
      continue;
    auto partWith = [&](int id) {
      for (int k : src->second)
        if (st.flowIds[k].count(id))
          return k;
      return -1;
    };
    int flow = partWith(split.a), other = partWith(split.b);
    if (flow < 0 || other < 0)
      continue;
    int apart = -1;
    if (split.apart) {
      auto it = nodeIds.find(*split.apart);
      if (it == nodeIds.end())
        continue;
      apart = stateId(it->second, Out);
    }
    // Whether a destination taking both ids is reached through the tile.
    auto inseparable = [&] {
      const Flow &f = st.parts[flow];
      for (const PathEndPoint &dst : f.dsts) {
        auto ids = packetIdsTo.find({f.src, dst});
        if (ids == packetIdsTo.end() ||
            !llvm::is_contained(ids->second, split.a) ||
            !llvm::is_contained(ids->second, split.b))
          continue;
        const auto &tree = st.treeOf[flow];
        auto it = tree.find(stateId(nodeIds.at(dst), Out));
        if (it == tree.end() && dst == f.src && dst.coords == split.at)
          return true;
        for (; it != tree.end(); it = tree.find(it->second.first)) {
          int up = it->second.first;
          if ((up & 1) == In && nodes[stateNode(up)].coords == split.at)
            return true;
        }
      }
      return false;
    };
    if (idsApart && flow == other && !split.apart &&
        !st.parts[flow].isPriorityFlow && inseparable())
      other = st.splitPart(flow, split.b);
    if (flow != other) {
      if (st.partSplits[flow].insert({split.at, other}).second)
        illegalEdges++;
      st.partSplits[other].insert({split.at, flow});
    } else if (st.splitsOf[flow]
                   .insert({split.at, split.a, split.b, apart})
                   .second) {
      illegalEdges++;
    }
  }
  crowdedTiles.insert(faults.crowded.begin(), faults.crowded.end());
  for (const auto &[tile, conn] : faults.connections) {
    illegalEdges++;
    auto it = graph.find({tile, tile});
    if (it == graph.end())
      continue;
    SwitchboxConnect &sb = it->second;
    int i = sb.srcIndex(conn.src);
    int j = sb.dstIndex(conn.dst);
    if (i < 0 || j < 0)
      continue;
    sb.at(i, j).overCapacity += routingCheckPenalty;
    for (auto [flow, hops] : llvm::enumerate(st.joinedHops))
      for (const auto &[other, e] : hops)
        if (e.sb == &sb && sb.srcPorts[e.i] == conn.src &&
            sb.dstPorts[e.j] == conn.dst)
          st.noJoin.insert({static_cast<int>(flow), other});
  }
  return illegalEdges;
}

// Perform congestion-aware routing for all flows which have been added.
// Use Dijkstra's shortest path to find routes, and use "demand" as the
// weights. If the routing finds too much congestion, update the demand
// weights and repeat the process until a valid solution is found. Returns
// the switchbox settings for all flows, or a RoutingFailure if no legal
// routing is found after maxIterations.
llvm::Expected<Routing> Pathfinder::findPaths(const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin Pathfinder::findPaths---\n");
  checkReason.clear();
  packetsFailed = true;
  // Build the dense routing graph once; topology is invariant across
  // iterations.
  if (!graphBuilt)
    buildRoutingGraph();
  crowdedTiles.clear();
  // initialize all Channel histories to 0
  for (auto &[_, sb] : graph) {
    if (sb.srcCoords == sb.dstCoords)
      for (auto [j, port] : llvm::enumerate(sb.dstPorts))
        sb.circuitOnlyDst[j] =
            cappedTiles.contains(sb.srcCoords) &&
            port.channel >= packetFanoutCap &&
            llvm::is_contained({WireBundle::North, WireBundle::South,
                                WireBundle::East, WireBundle::West},
                               port.bundle);
    for (SwitchboxConnect::Cell &c : sb.cells) {
      c.usedCapacity = 0;
      c.overCapacity = 0;
      c.isPriority = false;
    }
  }

  RouteState st(*this);

  int iterationCount = -1;
  int illegalEdges = 0;
  do {
    // if reach maxIterations, throw an error since no routing can be found
    if (++iterationCount >= maxIterations) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\t\tPathfinder: maxIterations has been exceeded ("
                 << maxIterations
                 << " iterations)...unable to find routing for flows.\n");
      packetsFailed =
          !checkReason.empty() || llvm::any_of(graph, [](const auto &entry) {
            ArrayRef<SwitchboxConnect::Cell> cells = entry.second.cells;
            return llvm::any_of(cells,
                                [](const SwitchboxConnect::Cell &c) {
                                  return c.packetGroupId >= 0;
                                }) &&
                   llvm::any_of(cells, [](const SwitchboxConnect::Cell &c) {
                     return c.usedCapacity > maxCircuitStreamCapacity;
                   });
          });
      return llvm::make_error<RoutingFailure>(explainNoRouting(st));
    }

    LLVM_DEBUG(llvm::dbgs() << "\t\t---Begin findPaths iteration #"
                            << iterationCount << "---\n");
    // update demand at the beginning of each iteration
    for (auto &[_, sb] : graph) {
      sb.updateDemand();
    }

    // "rip up" all routes
    illegalEdges = 0;
    st.routing = {};
    for (auto &[_, sb] : graph) {
      for (SwitchboxConnect::Cell &c : sb.cells) {
        c.usedCapacity = 0;
        c.packetFlowCount = 0;
        c.packetGroupId = -1;
        c.packetIds.clear();
      }
      sb.resetUnits();
    }
    for (auto &tree : st.treeOf)
      tree.clear();
    for (auto &dsts : st.treeDsts)
      dsts.clear();
    for (auto &hops : st.joinedHops)
      hops.clear();

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them

    for (const auto &[_, group] : st.groupedFlows) {
      for (int flow : group)
        if (llvm::Error err = routePart(st, flow))
          return std::move(err);
      for (auto &[_, sb] : graph) {
        for (SwitchboxConnect::Cell &c : sb.cells) {
          // fix used capacity for packet flows
          if (c.packetFlowCount > 0) {
            c.packetFlowCount = 0;
            c.usedCapacity++;
          }
          SwitchboxConnect::bumpDemand(c);
        }
      }
    }

    for (auto &[_, sb] : graph) {
      for (size_t i = 0; i < sb.srcPorts.size(); i++) {
        for (size_t j = 0; j < sb.dstPorts.size(); j++) {
          SwitchboxConnect::Cell &c = sb.at(i, j);
          // check that every channel does not exceed max capacity
          if (c.usedCapacity > maxCircuitStreamCapacity) {
            c.overCapacity++;
            illegalEdges++;
            LLVM_DEBUG(llvm::dbgs()
                       << "\t\t\tToo much capacity on (" << sb.srcCoords.col
                       << "," << sb.srcCoords.row << ") "
                       << sb.srcPorts[i].bundle << sb.srcPorts[i].channel
                       << " -> (" << sb.dstCoords.col << "," << sb.dstCoords.row
                       << ") " << sb.dstPorts[j].bundle
                       << sb.dstPorts[j].channel << ", used_capacity = "
                       << c.usedCapacity << ", demand = " << c.demand
                       << ", over_capacity_count = " << c.overCapacity << "\n");
          }
        }
      }
    }

    // A routing that fits the fabric can still be one the caller cannot use;
    // steer away from the connections it names as if they were overused.
    if (illegalEdges == 0 && constraints.check)
      llvm::handleAllErrors(
          constraints.check(st.routing), [&](RoutingFailure &rejected) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Routing rejected: " << rejected.reason << '\n');
            illegalEdges += applyRoutingFaults(st, rejected.faults);
            checkReason = std::move(rejected.reason);
          });

    LLVM_DEBUG({
      for (const auto &[src, settings] : st.routing.settings)
        llvm::dbgs() << "\t\t\tFlow starting at (" << src.coords.col << ","
                     << src.coords.row << "):\t" << settings;
      // total path length, across switchboxes
      int totalPathLength = 0;
      for (const auto &[_, sb] : graph)
        if (sb.srcCoords != sb.dstCoords)
          for (const SwitchboxConnect::Cell &c : sb.cells)
            totalPathLength += c.usedCapacity;
      llvm::dbgs() << "\t\t---End findPaths iteration #" << iterationCount
                   << " , illegal edges count = " << illegalEdges
                   << ", total path length = " << totalPathLength << "---\n";
    });
    // continue iterations until a legal routing is found
  } while (illegalEdges > 0);

  LLVM_DEBUG(llvm::dbgs() << "\t---End Pathfinder::findPaths---\n");
  return std::move(st.routing);
}
