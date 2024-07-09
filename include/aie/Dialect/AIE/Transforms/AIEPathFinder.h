//===- AIEPathfinder.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PATHFINDER_H
#define AIE_PATHFINDER_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include <algorithm>
#include <iostream>
#include <list>
#include <set>

namespace xilinx::AIE {

using SwitchboxNode = struct SwitchboxNode {

  SwitchboxNode(int col, int row, int id, int maxCol, int maxRow,
                const AIETargetModel &targetModel)
      : col{col}, row{row}, id{id} {

    std::vector<WireBundle> bundles = {WireBundle::Core, WireBundle::DMA,
                                       WireBundle::FIFO, WireBundle::Trace,
                                       WireBundle::Ctrl};

    if (col > 0)
      bundles.push_back(WireBundle::West);
    if (col < maxCol)
      bundles.push_back(WireBundle::East);
    if (row > 0)
      bundles.push_back(WireBundle::South);
    if (row < maxRow)
      bundles.push_back(WireBundle::North);

    for (WireBundle bundle : bundles) {
      int maxCapacity =
          targetModel.getNumSourceSwitchboxConnections(col, row, bundle);
      if (row == 0 && bundle == WireBundle::DMA) // ZY: hack for shimmux
        maxCapacity = 2;
      for (int channel = 0; channel < maxCapacity; channel++) {
        Port inPort = {bundle, channel};
        inPortToId[inPort] = inPortId;
        inPortId++;
      }

      maxCapacity =
          targetModel.getNumDestSwitchboxConnections(col, row, bundle);
      if (row == 0 && bundle == WireBundle::DMA) // ZY: hack for shimmux
        maxCapacity = 2;
      for (int channel = 0; channel < maxCapacity; channel++) {
        Port outPort = {bundle, channel};
        outPortToId[outPort] = outPortId;
        outPortId++;
      }
    }

    // -1:illegal, 0:available, 1:used
    connectionMatrix.resize(inPortId, std::vector<int>(outPortId, 0));

    // illegal connection
    for (const auto &[inPort, inId] : inPortToId) {
      for (const auto &[outPort, outId] : outPortToId) {
        if (!targetModel.isLegalTileConnection(col, row, inPort.bundle,
                                               inPort.channel, outPort.bundle,
                                               outPort.channel))
          connectionMatrix[inId][outId] = -1;

        // ZY: hack for shimmux
        if (row == 0 && (inPort.bundle == WireBundle::DMA ||
                         outPort.bundle == WireBundle::DMA))
          connectionMatrix[inId][outId] = 0;
      }
    }
  }

  // given a outPort, find availble input channel
  std::vector<int> findAvailableChannelIn(WireBundle inBundle, Port outPort,
                                          bool isPkt) {
    if (outPortToId.count(outPort) == 0) {
      printf("ERR%6s:%d ", stringifyWireBundle(outPort.bundle).str().c_str(),
             outPort.channel);
      return {};
    }

    int outId = outPortToId[outPort];
    std::vector<int> availableChannels;

    for (const auto &[inPort, inId] : inPortToId) {
      if (inPort.bundle == inBundle && connectionMatrix[inId][outId] == 0) {
        bool available = true;
        int streamCount = 0;
        int maxStreams = 1;
        if (isPkt &&
            std::find(pktInId.begin(), pktInId.end(), inId) != pktInId.end())
          maxStreams = 32;
        for (const auto &[outPort, outId] : outPortToId) {
          streamCount += connectionMatrix[inId][outId];
          if (streamCount >= maxStreams) {
            available = false;
            break;
          }
        }
        if (available)
          availableChannels.push_back(inPort.channel);
      }
    }
    return availableChannels;
  }

  void allocate(Port inPort, Port outPort, bool isPkt) {
    int inId = inPortToId.at(inPort);
    int outId = outPortToId.at(outPort);
    assert(connectionMatrix[inId][outId] >= 0 && "invalid allocation");
    connectionMatrix[inId][outId]++;
    if (isPkt &&
        std::find(pktInId.begin(), pktInId.end(), inId) == pktInId.end())
      pktInId.push_back(inId);
  }

  void clearAllocation() {
    for (int inId = 0; inId < inPortId; inId++) {
      for (int outId = 0; outId < outPortId; outId++) {
        if (connectionMatrix[inId][outId] == 1) {
          connectionMatrix[inId][outId] = 0;
        }
      }
    }
  }

  void visualize() {
    printf("Switchbox@ col:%d, row:%d\n", col, row);

    // Print header for columns (outPorts)
    printf("         ");
    for (const auto &[outPort, outId] : outPortToId) {
      printf("%6s:%d ", stringifyWireBundle(outPort.bundle).str().c_str(),
             outPort.channel);
    }
    printf("\n");

    for (const auto &[inPort, inId] : inPortToId) {
      // Print header for rows (inPorts)
      printf("%6s:%d ", stringifyWireBundle(inPort.bundle).str().c_str(),
             inPort.channel);
      for (const auto &[outPort, outId] : outPortToId) {
        printf("%8d ", connectionMatrix[inId][outId]);
      }
      printf("\n");
    }
    printf("\n");
  }

  friend std::ostream &operator<<(std::ostream &os, const SwitchboxNode &s) {
    os << "Switchbox(" << s.col << ", " << s.row << ")";
    return os;
  }

  GENERATE_TO_STRING(SwitchboxNode);

  bool operator<(const SwitchboxNode &rhs) const { return this->id < rhs.id; }

  bool operator==(const SwitchboxNode &rhs) const { return this->id == rhs.id; }

  int col, row, id;
  int inPortId = 0, outPortId = 0;
  std::map<Port, int> inPortToId, outPortToId;
  std::vector<std::vector<int>> connectionMatrix;
  std::vector<int> pktInId;
};

using ChannelEdge = struct ChannelEdge {
  ChannelEdge(SwitchboxNode *src, SwitchboxNode *target)
      : src(src), target(target) {}

  friend std::ostream &operator<<(std::ostream &os, const ChannelEdge &c) {
    os << "Channel(src=" << c.src << ", dst=" << c.target << ")";
    return os;
  }

  GENERATE_TO_STRING(ChannelEdge)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const ChannelEdge &c) {
    os << to_string(c);
    return os;
  }

  WireBundle getBundle() {
    if (src->col == target->col)
      if (src->row > target->row)
        return WireBundle::South;
      else
        return WireBundle::North;
    else if (src->col > target->col)
      return WireBundle::West;
    else
      return WireBundle::East;
  }

  int getMaxCapacity() {
    auto bundle = getBundle();
    int maxCapacity = 0; // maximum number of routing resources
    for (auto &[outPort, _] : src->outPortToId) {
      if (outPort.bundle == bundle) {
        maxCapacity++;
      }
    }
    return maxCapacity;
  }

  int getUsedCapacity() {
    auto bundle = getBundle();
    int usedCapacity = 0; // how many flows are actually using this Channel
    for (auto &[outPort, outPortId] : src->outPortToId) {
      if (outPort.bundle == bundle) {
        for (auto &[inPort, inPortId] : src->inPortToId) {
          if (src->connectionMatrix[inPortId][outPortId] > 0) {
            usedCapacity++;
            break;
          }
        }
      }
    }
    return usedCapacity;
  }

  SwitchboxNode *src;
  SwitchboxNode *target;

  std::set<int> fixedCapacity; // channels not available to the algorithm
};

// A SwitchSetting defines the required settings for a SwitchboxNode for a flow
// SwitchSetting.src is the incoming signal
// SwitchSetting.dsts is the fanout
using SwitchSetting = struct SwitchSetting {
  SwitchSetting() = default;
  SwitchSetting(Port src) : src(src) {}
  SwitchSetting(Port src, std::set<Port> dsts)
      : src(src), dsts(std::move(dsts)) {}
  Port src;
  std::set<Port> dsts;

  // friend definition (will define the function as a non-member function of the
  // namespace surrounding the class).
  friend std::ostream &operator<<(std::ostream &os,
                                  const SwitchSetting &setting) {
    os << setting.src << " -> "
       << "{"
       << join(llvm::map_range(setting.dsts,
                               [](const Port &port) {
                                 std::ostringstream ss;
                                 ss << port;
                                 return ss.str();
                               }),
               ", ")
       << "}";
    return os;
  }

  GENERATE_TO_STRING(SwitchSetting)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SwitchSetting &s) {
    os << to_string(s);
    return os;
  }

  bool operator<(const SwitchSetting &rhs) const { return src < rhs.src; }
};

using SwitchSettings = std::map<SwitchboxNode, SwitchSetting>;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
using PathEndPoint = struct PathEndPoint {
  SwitchboxNode sb;
  Port port;

  friend std::ostream &operator<<(std::ostream &os, const PathEndPoint &s) {
    os << "PathEndPoint(" << s.sb << ": " << s.port << ")";
    return os;
  }

  GENERATE_TO_STRING(PathEndPoint)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const PathEndPoint &s) {
    os << to_string(s);
    return os;
  }

  // Needed for the std::maps that store PathEndPoint.
  bool operator<(const PathEndPoint &rhs) const {
    return std::tie(sb, port) < std::tie(rhs.sb, rhs.port);
  }

  bool operator==(const PathEndPoint &rhs) const {
    return std::tie(sb, port) == std::tie(rhs.sb, rhs.port);
  }
};

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
using PathEndPointNode = struct PathEndPointNode : PathEndPoint {
  PathEndPointNode(SwitchboxNode *sb, Port port)
      : PathEndPoint{*sb, port}, sb(sb) {}
  SwitchboxNode *sb;
};

using FlowNode = struct FlowNode {
  bool isPacketFlow;
  PathEndPointNode src;
  std::vector<PathEndPointNode> dsts;
};

class Router {
public:
  Router() = default;
  // This has to go first so it can serve as a key function.
  // https://lld.llvm.org/missingkeyfunction
  virtual ~Router() = default;
  virtual void initialize(int maxCol, int maxRow,
                          const AIETargetModel &targetModel) = 0;
  virtual void addFlow(TileID srcCoords, Port srcPort, TileID dstCoords,
                       Port dstPort, bool isPacketFlow) = 0;
  // virtual bool addFixedConnection(ConnectOp connectOp) = 0; // ZY, todo:
  // FIXME
  virtual std::optional<std::map<PathEndPoint, SwitchSettings>>
  findPaths(int maxIterations) = 0;
  virtual SwitchboxNode getSwitchboxNode(TileID coords) = 0;
};

class Pathfinder : public Router {
public:
  Pathfinder() = default;
  void initialize(int maxCol, int maxRow,
                  const AIETargetModel &targetModel) override;
  void addFlow(TileID srcCoords, Port srcPort, TileID dstCoords, Port dstPort,
               bool isPacketFlow) override;
  // bool addFixedConnection(ConnectOp connectOp) override; // ZY, todo: FIXME
  std::optional<std::map<PathEndPoint, SwitchSettings>>
  findPaths(int maxIterations) override;

  std::map<SwitchboxNode *, SwitchboxNode *>
  dijkstraShortestPaths(SwitchboxNode *src);

  SwitchboxNode getSwitchboxNode(TileID coords) override {
    return grid.at(coords);
  }

private:
  std::vector<FlowNode> flows;
  std::map<TileID, SwitchboxNode> grid;
  // Use a list instead of a vector because nodes have an edge list of raw
  // pointers to edges (so growing a vector would invalidate the pointers).
  std::list<ChannelEdge> edges;

  // Use Dijkstra's shortest path to find routes, and use "demand" as the
  // weights.
  std::map<std::pair<SwitchboxNode *, SwitchboxNode *>, double> demandMatrix;
  // History of Channel being over capacity
  std::map<std::pair<SwitchboxNode *, SwitchboxNode *>, int> overCapacityMatrix;
};

// DynamicTileAnalysis integrates the Pathfinder class into the MLIR
// environment. It passes flows to the Pathfinder as ordered pairs of ints.
// Detailed routing is received as SwitchboxSettings
// It then converts these settings to MLIR operations
class DynamicTileAnalysis {
public:
  int maxCol, maxRow;
  std::shared_ptr<Router> pathfinder;
  std::map<PathEndPoint, SwitchSettings> flowSolutions;
  std::map<PathEndPoint, bool> processedFlows;

  llvm::DenseMap<TileID, TileOp> coordToTile;
  llvm::DenseMap<TileID, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileID, ShimMuxOp> coordToShimMux;
  llvm::DenseMap<int, PLIOOp> coordToPLIO;

  const int maxIterations = 1000; // how long until declared unroutable

  DynamicTileAnalysis() : pathfinder(std::make_shared<Pathfinder>()) {}
  DynamicTileAnalysis(std::shared_ptr<Router> p) : pathfinder(std::move(p)) {}

  mlir::LogicalResult runAnalysis(DeviceOp &device);

  int getMaxCol() const { return maxCol; }
  int getMaxRow() const { return maxRow; }

  TileOp getTile(mlir::OpBuilder &builder, int col, int row);

  SwitchboxOp getSwitchbox(mlir::OpBuilder &builder, int col, int row);

  ShimMuxOp getShimMux(mlir::OpBuilder &builder, int col);
};

} // namespace xilinx::AIE

namespace llvm {

inline raw_ostream &operator<<(raw_ostream &os,
                               const xilinx::AIE::SwitchSettings &ss) {
  std::stringstream s;
  s << "\tSwitchSettings: ";
  for (const auto &[sb, setting] : ss) {
    s << sb << ": " << setting << " | ";
  }
  s << "\n";
  os << s.str();
  return os;
}

} // namespace llvm

template <>
struct std::hash<xilinx::AIE::SwitchboxNode> {
  std::size_t operator()(const xilinx::AIE::SwitchboxNode &s) const noexcept {
    return std::hash<xilinx::AIE::TileID>{}({s.col, s.row});
  }
};

template <>
struct std::hash<xilinx::AIE::PathEndPoint> {
  std::size_t operator()(const xilinx::AIE::PathEndPoint &pe) const noexcept {
    std::size_t h1 = std::hash<xilinx::AIE::Port>{}(pe.port);
    std::size_t h2 = std::hash<xilinx::AIE::SwitchboxNode>{}(pe.sb);
    return h1 ^ (h2 << 1);
  }
};

#endif
