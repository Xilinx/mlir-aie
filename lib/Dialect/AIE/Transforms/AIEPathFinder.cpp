//===- AIEPathfinder.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"
#include "d_ary_heap.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"

#include "llvm/ADT/MapVector.h"

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
    Port srcPort, dstPort;
    TileOp srcTile, dstTile;
    TileID srcCoords, dstCoords;
    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        srcTile = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
        srcPort = pktSource.port();
        srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
        dstTile = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
        dstPort = pktDest.port();
        dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
        LLVM_DEBUG(llvm::dbgs()
                   << "\tAdding Packet Flow: (" << srcCoords.col << ", "
                   << srcCoords.row << ")"
                   << stringifyWireBundle(srcPort.bundle) << srcPort.channel
                   << " -> (" << dstCoords.col << ", " << dstCoords.row << ")"
                   << stringifyWireBundle(dstPort.bundle) << dstPort.channel
                   << "\n");
        // todo: support many-to-one & many-to-many?
        bool priorityFlow =
            pktFlowOp.getPriorityRoute()
                ? *pktFlowOp.getPriorityRoute()
                : false; // Flows such as control packet flows are routed in
                         // priority, to ensure routing consistency.
        pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort,
                            /*isPktFlow*/ true, priorityFlow);
      }
    }
  }

  // Sort ctrlPktFlows into a deterministic order; concat ctrlPktFlows to flows
  pathfinder->sortFlows(device.getTargetModel().columns(),
                        device.getTargetModel().rows());

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
                        /*isPktFlow*/ false, /*isPriorityFlow*/ false);
  }

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
  auto tileOp = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
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
  auto switchboxOp = builder.create<SwitchboxOp>(builder.getUnknownLoc(),
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
  auto switchboxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
                                               getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToShimMux[{col, row}] = switchboxOp;
  return switchboxOp;
}

void Pathfinder::initialize(int maxCol, int maxRow,
                            const AIETargetModel &targetModel) {

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
              return std::find(bundles.begin(), bundles.end(), bundle) !=
                     bundles.end();
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
                         Port dstPort, bool isPacketFlow, bool isPriorityFlow) {
  // check if a flow with this source already exists
  for (auto &[_, prioritized, src, dsts] : flows) {
    if (src.coords == srcCoords && src.port == srcPort) {
      if (isPriorityFlow) {
        prioritized = true;
        dsts.emplace(dsts.begin(), PathEndPoint{dstCoords, dstPort});
      } else
        dsts.emplace_back(PathEndPoint{dstCoords, dstPort});
      return;
    }
  }

  // Assign a group ID for packet flows
  // any overlapping in source/destination will lead to the same group ID
  // channel sharing will happen within the same group ID
  // for circuit flows, group ID is always -1, and no channel sharing
  int packetGroupId = -1;
  if (isPacketFlow) {
    bool found = false;
    for (auto &[existingId, _, src, dsts] : flows) {
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
           std::vector<PathEndPoint>{PathEndPoint{dstCoords, dstPort}}});
}

// Sort flows to (1) get deterministic routing, and (2) perform routings on
// prioritized flows before others, for routing consistency on those flows.
void Pathfinder::sortFlows(const int maxCol, const int maxRow) {
  std::vector<Flow> priorityFlows;
  std::vector<Flow> normalFlows;
  for (auto f : flows) {
    if (f.isPriorityFlow)
      priorityFlows.push_back(f);
    else
      normalFlows.push_back(f);
  }
  std::sort(priorityFlows.begin(), priorityFlows.end(),
            [](const auto &lhs, const auto &rhs) {
              // Compare tuple of properties in priority order:
              // (col, row, bundle, channel)
              auto lhsKey = std::make_tuple(lhs.src.coords.col, lhs.src.coords.row,
                                            getWireBundleAsInt(lhs.src.port.bundle),
                                            lhs.src.port.channel);
              auto rhsKey = std::make_tuple(rhs.src.coords.col, rhs.src.coords.row,
                                            getWireBundleAsInt(rhs.src.port.bundle),
                                            rhs.src.port.channel);
              return lhsKey < rhsKey;
            });
  flows = priorityFlows;
  flows.insert(flows.end(), normalFlows.begin(), normalFlows.end());
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
      for (size_t j = 0; j < sb.dstPorts.size(); j++) {
        if (sb.srcPorts[i] == connectOp.sourcePort() &&
            sb.dstPorts[j] == connectOp.destPort() &&
            sb.connectivity[i][j] == Connectivity::AVAILABLE) {
          sb.connectivity[i][j] = Connectivity::INVALID;
          found = true;
        }
      }
    }
    if (!found) {
      // could not add such a fixed connection
      return false;
    }
  }
  return true;
}

static constexpr double INF = std::numeric_limits<double>::max();

std::map<PathEndPoint, PathEndPoint>
Pathfinder::dijkstraShortestPaths(PathEndPoint src) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you
  // overwrite tombstones.
  std::map<PathEndPoint, double> distance;
  std::map<PathEndPoint, PathEndPoint> preds;
  std::map<PathEndPoint, uint64_t> indexInHeap;
  enum Color { WHITE, GRAY, BLACK };
  std::map<PathEndPoint, Color> colors;
  typedef d_ary_heap_indirect<
      /*Value=*/PathEndPoint, /*Arity=*/4,
      /*IndexInHeapPropertyMap=*/std::map<PathEndPoint, uint64_t>,
      /*DistanceMap=*/std::map<PathEndPoint, double> &,
      /*Compare=*/std::less<>>
      MutableQueue;
  MutableQueue Q(distance, indexInHeap);

  distance[src] = 0.0;
  Q.push(src);
  while (!Q.empty()) {
    src = Q.top();
    Q.pop();

    // get all channels src connects to
    if (channels.count(src) == 0) {
      auto &sb = graph[std::make_pair(src.coords, src.coords)];
      for (size_t i = 0; i < sb.srcPorts.size(); i++) {
        for (size_t j = 0; j < sb.dstPorts.size(); j++) {
          if (sb.srcPorts[i] == src.port &&
              sb.connectivity[i][j] == Connectivity::AVAILABLE) {
            // connections within the same switchbox
            channels[src].push_back(PathEndPoint{src.coords, sb.dstPorts[j]});
          }
        }
      }
      // connections to neighboring switchboxes
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
        if (graph.count(std::make_pair(src.coords, neighborCoords)) > 0 &&
            src.port.bundle == getConnectingBundle(neighborPort.bundle)) {
          auto &sb = graph[std::make_pair(src.coords, neighborCoords)];
          if (std::find(sb.dstPorts.begin(), sb.dstPorts.end(), neighborPort) !=
              sb.dstPorts.end())
            channels[src].push_back({neighborCoords, neighborPort});
        }
      }
      std::sort(channels[src].begin(), channels[src].end());
    }

    for (auto &dest : channels[src]) {
      if (distance.count(dest) == 0)
        distance[dest] = INF;
      auto &sb = graph[std::make_pair(src.coords, dest.coords)];
      size_t i = std::distance(
          sb.srcPorts.begin(),
          std::find(sb.srcPorts.begin(), sb.srcPorts.end(), src.port));
      size_t j = std::distance(
          sb.dstPorts.begin(),
          std::find(sb.dstPorts.begin(), sb.dstPorts.end(), dest.port));
      assert(i < sb.srcPorts.size());
      assert(j < sb.dstPorts.size());
      bool relax = distance[src] + sb.demand[i][j] < distance[dest];
      if (colors.count(dest) == 0) {
        // was WHITE
        if (relax) {
          distance[dest] = distance[src] + sb.demand[i][j];
          preds[dest] = src;
          colors[dest] = GRAY;
        }
        Q.push(dest);
      } else if (colors[dest] == GRAY && relax) {
        distance[dest] = distance[src] + sb.demand[i][j];
        preds[dest] = src;
      }
    }
    colors[src] = BLACK;
  }

  return preds;
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
        }
      }
    }

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them

    for (const auto &[_, flows] : groupedFlows) {
      for (const auto &[packetGroupId, isPriority, src, dsts] : flows) {
        // Use dijkstra to find path given current demand from the start
        // switchbox; find the shortest paths to each other switchbox. Output is
        // in the predecessor map, which must then be processed to get
        // individual switchbox settings
        std::set<PathEndPoint> processed;
        std::map<PathEndPoint, PathEndPoint> preds = dijkstraShortestPaths(src);

        // trace the path of the flow backwards via predecessors
        // increment used_capacity for the associated channels
        SwitchSettings switchSettings;
        processed.insert(src);
        for (auto endPoint : dsts) {
          if (endPoint == src) {
            // route to self
            switchSettings[src.coords].srcs.push_back(src.port);
            switchSettings[src.coords].dsts.push_back(src.port);
          }
          auto curr = endPoint;
          // trace backwards until a vertex already processed is reached
          while (!processed.count(curr)) {
            auto &sb = graph[std::make_pair(preds[curr].coords, curr.coords)];
            size_t i =
                std::distance(sb.srcPorts.begin(),
                              std::find(sb.srcPorts.begin(), sb.srcPorts.end(),
                                        preds[curr].port));
            size_t j = std::distance(
                sb.dstPorts.begin(),
                std::find(sb.dstPorts.begin(), sb.dstPorts.end(), curr.port));
            assert(i < sb.srcPorts.size());
            assert(j < sb.dstPorts.size());
            sb.isPriority[i][j] = isPriority;
            if (packetGroupId >= 0 &&
                (sb.packetGroupId[i][j] == -1 ||
                 sb.packetGroupId[i][j] == packetGroupId)) {
              for (size_t k = 0; k < sb.srcPorts.size(); k++) {
                for (size_t l = 0; l < sb.dstPorts.size(); l++) {
                  if (k == i || l == j) {
                    sb.packetGroupId[k][l] = packetGroupId;
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
            if (preds[curr].coords == curr.coords) {
              switchSettings[preds[curr].coords].srcs.push_back(
                  preds[curr].port);
              switchSettings[curr.coords].dsts.push_back(curr.port);
            }
            processed.insert(curr);
            curr = preds[curr];
          }
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