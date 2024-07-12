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
#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1

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
  for (const auto &[pathEndPoint, switchSetting] : flowSolutions) {
    processedFlows[pathEndPoint] = false;
    LLVM_DEBUG(llvm::dbgs() << "Flow starting at (" << pathEndPoint.sb.col
                            << "," << pathEndPoint.sb.row << "):\t");
    LLVM_DEBUG(llvm::dbgs() << switchSetting);
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
  // make grid of switchboxes
  int id = 0;
  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      grid.insert({{col, row},
                   SwitchboxNode{col, row, id++, maxCol, maxRow, targetModel}});
      SwitchboxNode &thisNode = grid.at({col, row});
      if (row > 0) { // if not in row 0 add channel to North/South
        SwitchboxNode &southernNeighbor = grid.at({col, row - 1});
        // get the number of outgoing connections on the south side - outgoing
        // because these correspond to rhs of a connect op
        if (targetModel.getNumDestSwitchboxConnections(col, row,
                                                       WireBundle::South)) {
          edges.emplace_back(&thisNode, &southernNeighbor);
        }
        // get the number of incoming connections on the south side - incoming
        // because they correspond to connections on the southside that are then
        // routed using internal connect ops through the switchbox (i.e., lhs of
        // connect ops)
        if (targetModel.getNumSourceSwitchboxConnections(col, row,
                                                         WireBundle::South)) {
          edges.emplace_back(&southernNeighbor, &thisNode);
        }
      }

      if (col > 0) { // if not in col 0 add channel to East/West
        SwitchboxNode &westernNeighbor = grid.at({col - 1, row});
        if (targetModel.getNumDestSwitchboxConnections(col, row,
                                                       WireBundle::West)) {
          edges.emplace_back(&thisNode, &westernNeighbor);
        }
        if (targetModel.getNumSourceSwitchboxConnections(col, row,
                                                         WireBundle::West)) {
          edges.emplace_back(&westernNeighbor, &thisNode);
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
    SwitchboxNode *existingSrcPtr = src.sb;
    assert(existingSrcPtr && "nullptr flow source");
    if (Port existingPort = src.port; existingSrcPtr->col == srcCoords.col &&
                                      existingSrcPtr->row == srcCoords.row &&
                                      existingPort == srcPort) {
      // find the vertex corresponding to the destination
      SwitchboxNode *matchingDstSbPtr = &grid.at(dstCoords);
      dsts.emplace_back(matchingDstSbPtr, dstPort);
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  SwitchboxNode *matchingSrcSbPtr = &grid.at(srcCoords);
  SwitchboxNode *matchingDstSbPtr = &grid.at(dstCoords);
  flows.push_back({isPacketFlow, PathEndPointNode{matchingSrcSbPtr, srcPort},
                   std::vector<PathEndPointNode>{{matchingDstSbPtr, dstPort}}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Pathfinder::addFixedConnection(SwitchboxOp switchboxOp) {
  int col = switchboxOp.colIndex();
  int row = switchboxOp.rowIndex();
  SwitchboxNode &sb = grid.at({col, row});
  std::set<int> invalidInId, invalidOutId;

  for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
    Port srcPort = connectOp.sourcePort();
    Port destPort = connectOp.destPort();
    if (sb.inPortToId.count(srcPort) == 0 ||
        sb.outPortToId.count(destPort) == 0)
      return false;
    int inId = sb.inPortToId.at(srcPort);
    int outId = sb.outPortToId.at(destPort);
    if (sb.connectionMatrix[inId][outId] != 0)
      return false;
    invalidInId.insert(inId);
    invalidOutId.insert(outId);
  }

  for (const auto &[inPort, inId] : sb.inPortToId) {
    for (const auto &[outPort, outId] : sb.outPortToId) {
      if (invalidInId.find(inId) != invalidInId.end() ||
          invalidOutId.find(outId) != invalidOutId.end()) {
        // mark as invalid
        sb.connectionMatrix[inId][outId] = -1;
      }
    }
  }
  return true;
}

static constexpr double INF = std::numeric_limits<double>::max();

std::map<SwitchboxNode *, SwitchboxNode *>
Pathfinder::dijkstraShortestPaths(SwitchboxNode *src) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you overwrite
  // tombstones.
  auto distance = std::map<SwitchboxNode *, double>();
  auto preds = std::map<SwitchboxNode *, SwitchboxNode *>();
  std::map<SwitchboxNode *, uint64_t> indexInHeap;
  typedef d_ary_heap_indirect<
      /*Value=*/SwitchboxNode *, /*Arity=*/4,
      /*IndexInHeapPropertyMap=*/std::map<SwitchboxNode *, uint64_t>,
      /*DistanceMap=*/std::map<SwitchboxNode *, double> &,
      /*Compare=*/std::less<>>
      MutableQueue;
  MutableQueue Q(distance, indexInHeap);

  for (auto &[_, sb] : grid)
    distance.emplace(&sb, INF);
  distance[src] = 0.0;

  std::map<SwitchboxNode *, std::vector<ChannelEdge *>> channels;

  enum Color { WHITE, GRAY, BLACK };
  std::map<SwitchboxNode *, Color> colors;
  for (auto &[_, sb] : grid) {
    SwitchboxNode *sbPtr = &sb;
    colors[sbPtr] = WHITE;
    for (auto &e : edges) {
      if (e.src == sbPtr) {
        channels[sbPtr].push_back(&e);
      }
    }
    std::sort(channels[sbPtr].begin(), channels[sbPtr].end(),
              [](const ChannelEdge *c1, ChannelEdge *c2) {
                return c1->target->id < c2->target->id;
              });
  }

  Q.push(src);
  while (!Q.empty()) {
    src = Q.top();
    Q.pop();
    for (ChannelEdge *e : channels[src]) {
      SwitchboxNode *dest = e->target;
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
std::optional<std::map<PathEndPoint, SwitchSettings>>
Pathfinder::findPaths(const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "Begin Pathfinder::findPaths\n");
  int iterationCount = 0;
  std::map<PathEndPoint, SwitchSettings> routingSolution;

  // initialize all Channel histories to 0
  for (auto &ch : edges) {
    overCapacity[&ch] = 0;
    usedCapacity[&ch] = 0;
  }
  // assume legal until found otherwise
  bool isLegal = true;

  do {
    LLVM_DEBUG(llvm::dbgs()
               << "Begin findPaths iteration #" << iterationCount << "\n");
    // update demand on all channels
    for (auto &ch : edges) {
      double history = 1.0 + OVER_CAPACITY_COEFF * overCapacity[&ch];
      double congestion = 1.0 + USED_CAPACITY_COEFF * usedCapacity[&ch];
      demand[&ch] = history * congestion;
    }
    // if reach maxIterations, throw an error since no routing can be found
    if (++iterationCount > maxIterations) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pathfinder: maxIterations has been exceeded ("
                 << maxIterations
                 << " iterations)...unable to find routing for flows.\n");
      return std::nullopt;
    }

    // "rip up" all routes
    routingSolution.clear();
    for (auto &[tileID, node] : grid) {
      node.clearAllocation();
    }
    isLegal = true;

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for (const auto &[isPkt, src, dsts] : flows) {
      // Use dijkstra to find path given current demand from the start
      // switchbox; find the shortest paths to each other switchbox. Output is
      // in the predecessor map, which must then be processed to get individual
      // switchbox settings
      assert(src.sb && "nonexistent flow source");
      std::set<SwitchboxNode *> processed;
      std::map<SwitchboxNode *, SwitchboxNode *> preds =
          dijkstraShortestPaths(src.sb);

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings;
      // set the input bundle for the source endpoint
      switchSettings[*src.sb].src = src.port;
      processed.insert(src.sb);
      // destination ports used by src.sb
      std::vector<Port> srcDestPorts;
      for (const PathEndPointNode &endPoint : dsts) {
        SwitchboxNode *curr = endPoint.sb;
        assert(curr && "endpoint has no source switchbox");
        // set the output bundle for this destination endpoint
        switchSettings[*curr].dsts.insert(endPoint.port);
        Port lastDestPort = endPoint.port;
        // trace backwards until a vertex already processed is reached
        while (!processed.count(curr)) {
          // find the incoming edge from the pred to curr
          ChannelEdge *ch = nullptr;
          for (auto &e : edges) {
            if (e.src == preds[curr] && e.target == curr) {
              ch = &e;
              break;
            }
          }
          assert(ch != nullptr && "couldn't find ch");
          int channel =
              curr->outPortToId.count(lastDestPort) > 0
                  ? curr->findAvailableChannelIn(
                        getConnectingBundle(ch->bundle), lastDestPort, isPkt)
                  : -1;
          if (channel >= 0) {
            bool succeed =
                curr->allocate({getConnectingBundle(ch->bundle), channel},
                               lastDestPort, isPkt);
            if (!succeed)
              assert(false && "invalid allocation");
          } else {
            // if no channel available, use a virtual channel id and mark
            // routing as being invalid
            channel = usedCapacity[ch];
            LLVM_DEBUG(llvm::dbgs()
                       << "Too much capacity on Edge (" << ch->target->col
                       << ", " << ch->target->row << ") . "
                       << stringifyWireBundle(ch->bundle)
                       << "\t: used_capacity = " << usedCapacity[ch]
                       << "\t: Demand = " << demand[ch] << "\n");
            if (isLegal) {
              overCapacity[ch]++;
            }
            LLVM_DEBUG(llvm::dbgs()
                       << "over_capacity_count = " << overCapacity[ch] << "\n");
            isLegal = false;
          }
          usedCapacity[ch]++;

          // add the entrance port for this Switchbox
          Port currSourcePort = {getConnectingBundle(ch->bundle), channel};
          switchSettings[*curr].src = {currSourcePort};

          // add the current Switchbox to the map of the predecessor
          Port PredDestPort = {ch->bundle, channel};
          switchSettings[*preds[curr]].dsts.insert(PredDestPort);
          lastDestPort = PredDestPort;

          // if at capacity, bump demand to discourage using this Channel
          if (usedCapacity[ch] >= ch->maxCapacity) {
            LLVM_DEBUG(llvm::dbgs() << "ch over capacity: " << ch << "\n");
            // this means the order matters!
            demand[ch] *= DEMAND_COEFF;
          }

          processed.insert(curr);
          curr = preds[curr];
        }
        if (src.sb->outPortToId.count(lastDestPort) &&
            std::find(srcDestPorts.begin(), srcDestPorts.end(), lastDestPort) ==
                srcDestPorts.end()) {
          bool succeed = src.sb->allocate(src.port, lastDestPort, isPkt);
          if (!succeed)
            assert(false && "invalid allocation");
          srcDestPorts.push_back(lastDestPort);
        }
      }
      // add this flow to the proposed solution
      routingSolution[src] = switchSettings;
    }

  } while (!isLegal); // continue iterations until a legal routing is found

  return routingSolution;
}
