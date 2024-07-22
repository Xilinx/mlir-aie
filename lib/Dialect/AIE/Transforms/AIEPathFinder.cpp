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

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-pathfinder"
#define OVER_CAPACITY_COEFF 0.1
#define USED_CAPACITY_COEFF 0.1
#define DEMAND_COEFF 1.1
#define MAX_CIRCUIT_STREAM_CAPACITY 1
#define MAX_PACKET_STREAM_CAPACITY 32

LogicalResult DynamicTileAnalysis::runAnalysis(DeviceOp &device) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin DynamicTileAnalysis Constructor---\n");
  // find the maxCol and maxRow
  maxCol = 0;
  maxRow = 0;
  for (TileOp tileOp : device.getOps<TileOp>()) {
    maxCol = std::max(maxCol, tileOp.colIndex());
    maxRow = std::max(maxRow, tileOp.rowIndex());
  }

  pathfinder->initialize(maxCol, maxRow, device.getTargetModel());

  // for each flow in the device, add it to pathfinder
  // each source can map to multiple different destinations (fanout)
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
    pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort, false);
  }

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
        pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort, true);
      }
    }
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
  for (const auto &[pathNode, switchSetting] : flowSolutions) {
    processedFlows[pathNode] = false;
  }

  // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
  for (auto tileOp : device.getOps<TileOp>()) {
    int col, row;
    col = tileOp.colIndex();
    row = tileOp.rowIndex();
    maxCol = std::max(maxCol, col);
    maxRow = std::max(maxRow, row);
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
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return tileOp;
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
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}

ShimMuxOp DynamicTileAnalysis::getShimMux(OpBuilder &builder, int col) {
  assert(col >= 0);
  int row = 0;
  if (coordToShimMux.count({col, row})) {
    return coordToShimMux[{col, row}];
  }
  assert(getTile(builder, col, row).isShimNOCTile());
  auto switchboxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
                                               getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToShimMux[{col, row}] = switchboxOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}

void Pathfinder::initialize(int maxCol, int maxRow,
                            const AIETargetModel &targetModel) {

  auto insertAndConnect = [&](int colSrc, int rowSrc, WireBundle bundleSrc,
                              int channelSrc, int colDest, int rowDest,
                              WireBundle bundleDest, int channelDest) {
    auto sourcePair =
        pathNodes.insert({TileID{colSrc, rowSrc}, Port{bundleSrc, channelSrc}});
    auto targetPair = pathNodes.insert(
        {TileID{colDest, rowDest}, Port{bundleDest, channelDest}});

    auto &sourceIt = std::get<0>(sourcePair);
    auto &targetIt = std::get<0>(targetPair);

    pathEdges.push_back(PathEdge{const_cast<PathNode *>(&(*sourceIt)),
                                 const_cast<PathNode *>(&(*targetIt))});
  };

  const std::vector<WireBundle> bundles = {
      WireBundle::Core, WireBundle::DMA,   WireBundle::FIFO, WireBundle::South,
      WireBundle::West, WireBundle::North, WireBundle::East, WireBundle::PLIO,
      WireBundle::NOC,  WireBundle::Trace, WireBundle::Ctrl};

  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      std::vector<Port> inPorts, outPorts;
      for (WireBundle bundle : bundles) {
        // get all ports into current switchbox
        int maxCapacity =
            targetModel.getNumSourceSwitchboxConnections(col, row, bundle);
        if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
          // wordaround for shimMux, todo: integrate shimMux into routable grid
          maxCapacity =
              targetModel.getNumSourceShimMuxConnections(col, row, bundle);
        }
        for (int channel = 0; channel < maxCapacity; channel++) {
          inPorts.push_back({bundle, channel});
        }

        // get all ports out of current switchbox
        maxCapacity =
            targetModel.getNumDestSwitchboxConnections(col, row, bundle);
        if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
          // wordaround for shimMux, todo: integrate shimMux into routable grid
          maxCapacity =
              targetModel.getNumDestShimMuxConnections(col, row, bundle);
        }
        for (int channel = 0; channel < maxCapacity; channel++) {
          outPorts.push_back({bundle, channel});
        }
      }

      for (auto &inPort : inPorts) {
        for (auto &outPort : outPorts) {
          Connectivity status = Connectivity::AVAILABLE;

          if (!targetModel.isLegalTileConnection(col, row, inPort.bundle,
                                                 inPort.channel, outPort.bundle,
                                                 outPort.channel))
            status = Connectivity::INVALID;

          if (targetModel.isShimNOCorPLTile(col, row)) {
            // wordaround for shimMux, todo: integrate shimMux into routable
            // grid
            auto isBundleInList = [](WireBundle bundle,
                                     std::vector<WireBundle> bundles) {
              return std::find(bundles.begin(), bundles.end(), bundle) !=
                     bundles.end();
            };
            std::vector<WireBundle> bundles = {WireBundle::DMA, WireBundle::NOC,
                                               WireBundle::PLIO};
            if (isBundleInList(inPort.bundle, bundles) ||
                isBundleInList(outPort.bundle, bundles))
              status = Connectivity::AVAILABLE;
          }

          if (status == Connectivity::AVAILABLE) {
            // connection inside switchboxes
            insertAndConnect(col, row, inPort.bundle, inPort.channel, col, row,
                             outPort.bundle, outPort.channel);
          }
        }

        if (inPort.bundle == WireBundle::South && row > 0) {
          // connection between switchboxes, from north to south
          insertAndConnect(col, row - 1, WireBundle::North, inPort.channel, col,
                           row, inPort.bundle, inPort.channel);
        }
        if (inPort.bundle == WireBundle::North && row < maxRow) {
          // connection between switchboxes, from south to north
          insertAndConnect(col, row + 1, WireBundle::South, inPort.channel, col,
                           row, inPort.bundle, inPort.channel);
        }
        if (inPort.bundle == WireBundle::West && col > 0) {
          // connection between switchboxes, from east to west
          insertAndConnect(col - 1, row, WireBundle::East, inPort.channel, col,
                           row, inPort.bundle, inPort.channel);
        }
        if (inPort.bundle == WireBundle::East && col < maxCol) {
          // connection between switchboxes, from west to east
          insertAndConnect(col + 1, row, WireBundle::West, inPort.channel, col,
                           row, inPort.bundle, inPort.channel);
        }
      }
    }
  }
}

// Add a flow from src to dst can have an arbitrary number of dst locations due
// to fanout.
void Pathfinder::addFlow(TileID srcCoords, Port srcPort, TileID dstCoords,
                         Port dstPort, bool isPacketFlow) {
  // check if a flow with this source already exists
  for (auto &[isPkt, src, dsts] : flows) {
    if (src->sb == srcCoords && src->port == srcPort) {
      // find the vertex corresponding to the destination
      auto dstIt = pathNodes.find(PathNode{dstCoords, dstPort});
      assert(dstIt != pathNodes.end());
      dsts.emplace_back(const_cast<PathNode *>(&(*dstIt)));
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  auto srcIt = pathNodes.find(PathNode{srcCoords, srcPort});
  auto dstIt = pathNodes.find(PathNode{dstCoords, dstPort});
  assert(srcIt != pathNodes.end() && dstIt != pathNodes.end());
  flows.push_back(
      FlowNode{isPacketFlow, const_cast<PathNode *>(&(*srcIt)),
               std::vector<PathNode *>{const_cast<PathNode *>(&(*dstIt))}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Pathfinder::addFixedConnection(SwitchboxOp switchboxOp) {
  int col = switchboxOp.colIndex();
  int row = switchboxOp.rowIndex();
  TileID coords = {col, row};

  for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
    Port srcPort = connectOp.sourcePort();
    Port destPort = connectOp.destPort();

    auto srcIt = pathNodes.find(PathNode{coords, srcPort});
    auto destIt = pathNodes.find(PathNode{coords, destPort});

    if (srcIt == pathNodes.end() || destIt == pathNodes.end()) {
      return false;
    }

    PathNode *pathNodeSrcPtr = const_cast<PathNode *>(&(*srcIt));
    PathNode *pathNodeDstPtr = const_cast<PathNode *>(&(*destIt));
    // remove the edge from the list of edges
    bool found = false;
    for (auto it = pathEdges.begin(); it != pathEdges.end(); ++it) {
      if ((*it).source == pathNodeSrcPtr && (*it).target == pathNodeDstPtr) {
        pathEdges.erase(it);
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }

  return true;
}

static constexpr double INF = std::numeric_limits<double>::max();

std::map<PathNode *, PathNode *>
Pathfinder::dijkstraShortestPaths(PathNode *src) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you overwrite
  // tombstones.
  auto distance = std::map<PathNode *, double>();
  auto preds = std::map<PathNode *, PathNode *>();
  std::map<PathNode *, uint64_t> indexInHeap;
  typedef d_ary_heap_indirect<
      /*Value=*/PathNode *, /*Arity=*/4,
      /*IndexInHeapPropertyMap=*/std::map<PathNode *, uint64_t>,
      /*DistanceMap=*/std::map<PathNode *, double> &,
      /*Compare=*/std::less<>>
      MutableQueue;
  MutableQueue Q(distance, indexInHeap);

  for (auto &pathNode : pathNodes) {
    PathNode *pathNodePtr = const_cast<PathNode *>(&pathNode);
    distance.emplace(pathNodePtr, INF);
  }
  distance[src] = 0.0;

  std::map<PathNode *, std::vector<PathEdge *>> channels;

  enum Color { WHITE, GRAY, BLACK };
  std::map<PathNode *, Color> colors;
  for (auto &pathNode : pathNodes) {
    PathNode *pathNodePtr = const_cast<PathNode *>(&pathNode);
    colors[pathNodePtr] = WHITE;
    for (auto &e : pathEdges) {
      if (e.source == pathNodePtr) {
        channels[pathNodePtr].push_back(&e);
      }
    }
    // sort the channels in descending order
    std::sort(channels[pathNodePtr].begin(), channels[pathNodePtr].end(),
              [](const PathEdge *c1, PathEdge *c2) {
                return *(c2->target) < *(c1->target);
              });
  }

  Q.push(src);
  while (!Q.empty()) {
    src = Q.top();
    Q.pop();
    for (PathEdge *e : channels[src]) {
      PathNode *dest = e->target;
      bool relax = distance[src] + demand[e] < distance[dest];
      if (colors[dest] == WHITE) {
        if (relax) {
          distance[dest] = distance[src] + demand[e];
          preds[dest] = src;
          colors[dest] = GRAY;
        }
        Q.push(dest);
      } else if (colors[dest] == GRAY && relax) {
        distance[dest] = distance[src] + demand[e];
        preds[dest] = src;
      }
    }
    colors[src] = BLACK;
  }

  return preds;
}

// Perform congestion-aware routing for all flows which have been added.
// Use Dijkstra's shortest path to find routes, and use "demand" as the weights.
// If the routing finds too much congestion, update the demand weights
// and repeat the process until a valid solution is found.
// Returns a map specifying switchbox settings for all flows.
// If no legal routing can be found after maxIterations, returns empty vector.
std::optional<std::map<PathNode, SwitchSettings>>
Pathfinder::findPaths(const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin Pathfinder::findPaths---\n");
  std::map<PathNode, SwitchSettings> routingSolution;

  // initialize all Channel histories to 0
  for (auto &ch : pathEdges) {
    overCapacity[&ch] = 0;
    usedCapacity[&ch] = 0;
  }

  int iterationCount = -1;
  int illegalEdges = 0;
  int totalPathLength = 0;
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
    // update demand on all channels
    for (auto &ch : pathEdges) {
      double history = 1.0 + OVER_CAPACITY_COEFF * overCapacity[&ch];
      double congestion = 1.0 + USED_CAPACITY_COEFF * usedCapacity[&ch];
      demand[&ch] = history * congestion;
    }

    // "rip up" all routes
    routingSolution.clear();
    for (auto &ch : pathEdges) {
      usedCapacity[&ch] = 0;
      packetFlowCount[&ch] = 0;
    }

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for (const auto &[isPkt, src, dsts] : flows) {
      // Use dijkstra to find path given current demand from the start
      // switchbox; find the shortest paths to each other switchbox. Output is
      // in the predecessor map, which must then be processed to get individual
      // switchbox settings
      std::set<PathNode *> processed;
      std::map<PathNode *, PathNode *> preds = dijkstraShortestPaths(src);

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings;
      // set the input bundle for the source endpoint
      switchSettings[src->sb].src = src->port;
      processed.insert(src);
      for (auto endPoint : dsts) {
        PathNode *curr = endPoint;
        // set the output bundle for this destination endpoint
        switchSettings[endPoint->sb].dsts.insert(endPoint->port);
        // trace backwards until a vertex already processed is reached
        while (!processed.count(curr)) {
          // find the incoming edge from the pred to curr
          PathEdge *ch = nullptr;
          for (auto &e : pathEdges) {
            if (e.source == preds[curr] && e.target == curr) {
              ch = &e;
              break;
            }
          }
          assert(ch != nullptr);
          if (preds[curr]->sb == curr->sb) {
            switchSettings[preds[curr]->sb].src = preds[curr]->port;
            switchSettings[curr->sb].dsts.insert(curr->port);
          }

          if (isPkt) {
            packetFlowCount[ch]++;
            // maximum packet stream per channel
            if (packetFlowCount[ch] >= MAX_PACKET_STREAM_CAPACITY) {
              packetFlowCount[ch] = 0;
              usedCapacity[ch]++;
            }
          } else {
            packetFlowCount[ch] = 0;
            usedCapacity[ch]++;
          }

          // if at capacity, bump demand to discourage using this Channel
          if (usedCapacity[ch] >= MAX_CIRCUIT_STREAM_CAPACITY) {
            // this means the order matters!
            demand[ch] *= DEMAND_COEFF;
          }

          processed.insert(curr);
          curr = preds[curr];
        }
      }
      // add this flow to the proposed solution
      routingSolution[*src] = switchSettings;
    }

    // fix used capacity for packet flows
    for (auto &ch : pathEdges) {
      if (packetFlowCount[&ch] > 0) {
        packetFlowCount[&ch] = 0;
        usedCapacity[&ch]++;
      }
    }

    illegalEdges = 0;
    totalPathLength = 0;
    for (auto &e : pathEdges) {
      // Calculate total path length across switchboxes
      if (e.source->sb != e.target->sb) {
        totalPathLength += usedCapacity[&e];
      }
      // Check that every channel does not exceed max capacity.
      if (usedCapacity[&e] > MAX_CIRCUIT_STREAM_CAPACITY) {
        overCapacity[&e]++;
        illegalEdges++;
        LLVM_DEBUG(llvm::dbgs()
                   << "\t\t\tToo much capacity on " << e << ", used_capacity = "
                   << usedCapacity[&e] << ", demand = " << demand[&e]
                   << ", over_capacity_count = " << overCapacity[&e] << "\n");
      }
    }

#ifndef NDEBUG
    for (const auto &[pathNode, switchSetting] : routingSolution) {
      LLVM_DEBUG(llvm::dbgs() << "\t\t\tFlow starting at (" << pathNode.sb.col
                              << "," << pathNode.sb.row << "):\t");
      LLVM_DEBUG(llvm::dbgs() << switchSetting);
    }
#endif
    LLVM_DEBUG(llvm::dbgs()
               << "\t\t---End findPaths iteration #" << iterationCount
               << " , illegal edges count = " << illegalEdges
               << ", total path length = " << totalPathLength << "---\n");

  } while (illegalEdges >
           0); // continue iterations until a legal routing is found

  LLVM_DEBUG(llvm::dbgs() << "\t---End Pathfinder::findPaths---\n");
  return routingSolution;
}
