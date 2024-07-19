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

enum class Connectivity { INVALID = -1, AVAILABLE = 0, OCCUPIED = 1 };

using PathNode = struct PathNode {
  PathNode(TileID sb, Port port) : sb(sb), port(port) {}

  TileID sb;
  Port port;

  friend std::ostream &operator<<(std::ostream &os, const PathNode &s) {
    os << "PathNode(" << s.sb << ": " << s.port << ")";
    return os;
  }

  GENERATE_TO_STRING(PathNode)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const PathNode &s) {
    os << to_string(s);
    return os;
  }

  // Needed for the std::maps that store PathNode.
  bool operator<(const PathNode &rhs) const {
    return std::tie(sb, port) < std::tie(rhs.sb, rhs.port);
  }

  bool operator==(const PathNode &rhs) const {
    return std::tie(sb, port) == std::tie(rhs.sb, rhs.port);
  }
};

using PathEdge = struct PathEdge {
  PathEdge(PathNode *source, PathNode *target)
      : source(source), target(target) {}
  PathNode *source;
  PathNode *target;

  friend std::ostream &operator<<(std::ostream &os, const PathEdge &s) {
    os << "PathEdge(" << *s.source << " -> " << *s.target << ")";
    return os;
  }

  GENERATE_TO_STRING(PathEdge)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const PathEdge &s) {
    os << to_string(s);
    return os;
  }
};

using FlowNode = struct FlowNode {
  bool isPacketFlow;
  PathNode *src;
  std::vector<PathNode *> dsts;
};

// A SwitchSetting defines the required settings for a Switchbox for a flow
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

using SwitchSettings = std::map<TileID, SwitchSetting>;

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
  // virtual bool addFixedConnection(SwitchboxOp switchboxOp) = 0;
  virtual std::optional<std::map<PathNode, SwitchSettings>>
  findPaths(int maxIterations) = 0;
};

class Pathfinder : public Router {
public:
  Pathfinder() = default;
  void initialize(int maxCol, int maxRow,
                  const AIETargetModel &targetModel) override;
  void addFlow(TileID srcCoords, Port srcPort, TileID dstCoords, Port dstPort,
               bool isPacketFlow) override;
  // bool addFixedConnection(SwitchboxOp switchboxOp) override;
  std::optional<std::map<PathNode, SwitchSettings>>
  findPaths(int maxIterations) override;

  std::map<PathNode *, PathNode *> dijkstraShortestPaths(PathNode *src);

private:
  // Flows to be routed
  std::vector<FlowNode> flows;

  std::set<PathNode> pathNodes;
  std::vector<PathEdge> pathEdges;

  // Use Dijkstra's shortest path to find routes, and use "demand" as the
  // weights.
  std::map<PathEdge *, double> demand;

  // History of Channel being over capacity
  std::map<PathEdge *, int> overCapacity;

  // how many flows are actually using this Channel
  std::map<PathEdge *, int> usedCapacity;
};

// DynamicTileAnalysis integrates the Pathfinder class into the MLIR
// environment. It passes flows to the Pathfinder as ordered pairs of ints.
// Detailed routing is received as SwitchboxSettings
// It then converts these settings to MLIR operations
class DynamicTileAnalysis {
public:
  int maxCol, maxRow;
  std::shared_ptr<Router> pathfinder;
  std::map<PathNode, SwitchSettings> flowSolutions;
  std::map<PathNode, bool> processedFlows;

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

#endif
