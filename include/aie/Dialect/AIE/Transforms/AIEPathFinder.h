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

#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/GraphTraits.h"

#include <algorithm>
#include <list>
#include <set>

namespace xilinx::AIE {

class Switchbox;
class Channel;
using SwitchboxBase = llvm::DGNode<Switchbox, Channel>;
using ChannelBase = llvm::DGEdge<Switchbox, Channel>;
using SwitchboxGraphBase = llvm::DirectedGraph<Switchbox, Channel>;

class Switchbox : public SwitchboxBase {
public:
  Switchbox() = delete;
  Switchbox(const int col, const int row) : col(col), row(row) {}

  // friend definition (will define the function as a non-member function in the
  // namespace surrounding the class).
  friend std::ostream &operator<<(std::ostream &os, const Switchbox &s) {
    os << "Switchbox(" << s.col << ", " << s.row << ")";
    return os;
  }

  GENERATE_TO_STRING(Switchbox)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Switchbox &s) {
    os << to_string(s);
    return os;
  }

  int col, row;
};

class Channel : public ChannelBase {
public:
  explicit Channel(Switchbox &target) = delete;
  Channel(Switchbox &src, Switchbox &target, WireBundle bundle, int maxCapacity)
      : ChannelBase(target), src(src), bundle(bundle),
        maxCapacity(maxCapacity) {}

  // Default deleted because of &src and &ChannelBase::TargetNode.
  Channel(const Channel &E)
      : ChannelBase(E), src(E.src), bundle(E.bundle),
        maxCapacity(E.maxCapacity), demand(E.demand),
        usedCapacity(E.usedCapacity), fixedCapacity(E.fixedCapacity),
        overCapacityCount(E.overCapacityCount) {}

  // Default deleted because of &src and &ChannelBase::TargetNode.
  Channel &operator=(Channel &&E) {
    ChannelBase::operator=(std::move(E));
    src = std::move(E.src);
    bundle = E.bundle;
    maxCapacity = E.maxCapacity;
    demand = E.demand;
    usedCapacity = E.usedCapacity;
    fixedCapacity = E.fixedCapacity;
    overCapacityCount = E.overCapacityCount;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const Channel &c) {
    os << "Channel(src=" << c.src << ", dst=" << c.getTargetNode() << ")";
    return os;
  }

  GENERATE_TO_STRING(Channel)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Channel &c) {
    os << to_string(c);
    return os;
  }

  Switchbox &src;
  WireBundle bundle;
  int maxCapacity = 0;  // maximum number of routing resources
  double demand = 0.0;  // indicates how many flows want to use this Channel
  int usedCapacity = 0; // how many flows are actually using this Channel
  std::set<int> fixedCapacity; // channels not available to the algorithm
  int overCapacityCount = 0;   // history of Channel being over capacity
};

class SwitchboxGraph : public SwitchboxGraphBase {
public:
  SwitchboxGraph() = default;
  ~SwitchboxGraph() = default;
};

// A SwitchSetting defines the required settings for a Switchbox for a flow
// SwitchSetting.first is the incoming signal
// SwitchSetting.second is the fanout
typedef struct SwitchSetting {
  Port src;
  std::set<Port> dsts;

  // friend definition (will define the function as a non-member function of the
  // namespace surrounding the class).
  friend std::ostream &operator<<(std::ostream &os,
                                  const SwitchSetting &setting) {
    os << setting.src << " -> "
       << "{"
       << llvm::join(llvm::map_range(setting.dsts,
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
} SwitchSetting;

typedef std::map<Switchbox *, SwitchSetting> SwitchSettings;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
typedef struct PathEndPoint {
  Switchbox *sb;
  Port port;

  bool operator<(const PathEndPoint &rhs) const {
    return sb->col == rhs.sb->col
               ? (sb->row == rhs.sb->row
                      ? (port.bundle == rhs.port.bundle
                             ? port.channel < rhs.port.channel
                             : port.bundle < rhs.port.bundle)
                      : sb->row < rhs.sb->row)
               : sb->col < rhs.sb->col;
  }

} PathEndPoint;

typedef struct Flow {
  PathEndPoint src;
  std::vector<PathEndPoint> dsts;
} Flow;

class Pathfinder {
  SwitchboxGraph graph;
  std::vector<Flow> flows;
  bool maxIterReached{};
  std::map<TileID, Switchbox> grid;
  // Use a list instead of a vector because nodes have an edge list of raw
  // pointers to edges (so growing a vector would invalidate the pointers).
  std::list<Channel> edges;

public:
  Pathfinder() = default;
  virtual void initialize(int maxCol, int maxRow,
                          const AIETargetModel &targetModel);
  void addFlow(TileID srcCoords, Port srcPort, TileID dstCoords, Port dstPort);
  bool addFixedConnection(TileID coords, Port port);
  bool isLegal();
  std::map<PathEndPoint, SwitchSettings> findPaths(int maxIterations = 1000);

  Switchbox *getSwitchbox(TileID coords) {
    auto sb = std::find_if(graph.begin(), graph.end(), [&](Switchbox *sb) {
      return sb->col == coords.col && sb->row == coords.row;
    });
    assert(sb != graph.end() && "couldn't find sb");
    return *sb;
  }
};

// DynamicTileAnalysis integrates the Pathfinder class into the MLIR
// environment. It passes flows to the Pathfinder as ordered pairs of ints.
// Detailed routing is received as SwitchboxSettings
// It then converts these settings to MLIR operations
class DynamicTileAnalysis {
public:
  int maxCol, maxRow;
  std::shared_ptr<Pathfinder> pathfinder;
  std::map<PathEndPoint, SwitchSettings> flowSolutions;
  std::map<PathEndPoint, bool> processedFlows;

  llvm::DenseMap<TileID, TileOp> coordToTile;
  llvm::DenseMap<TileID, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileID, ShimMuxOp> coordToShimMux;
  llvm::DenseMap<int, PLIOOp> coordToPLIO;

  const int maxIterations = 1000; // how long until declared unroutable

  DynamicTileAnalysis() : pathfinder(std::make_shared<Pathfinder>()) {}
  DynamicTileAnalysis(std::shared_ptr<Pathfinder> p) : pathfinder(p) {}

  void runAnalysis(DeviceOp &device);

  int getMaxCol() const { return maxCol; }
  int getMaxRow() const { return maxRow; }

  TileOp getTile(mlir::OpBuilder &builder, int col, int row);

  SwitchboxOp getSwitchbox(mlir::OpBuilder &builder, int col, int row);

  ShimMuxOp getShimMux(mlir::OpBuilder &builder, int col);
};

} // namespace xilinx::AIE

namespace std {
template <> struct less<xilinx::AIE::Switchbox *> {
  bool operator()(const xilinx::AIE::Switchbox *a,
                  const xilinx::AIE::Switchbox *b) const {
    return a->col == b->col ? a->row < b->row : a->col < b->col;
  }
};
} // namespace std

namespace llvm {

template <> struct GraphTraits<xilinx::AIE::Switchbox *> {
  using NodeRef = xilinx::AIE::Switchbox *;

  static xilinx::AIE::Switchbox *SwitchboxGraphGetSwitchbox(
      DGEdge<xilinx::AIE::Switchbox, xilinx::AIE::Channel> *P) {
    return &P->getTargetNode();
  }

  // Provide a mapped iterator so that the GraphTrait-based implementations can
  // find the target nodes without having to explicitly go through the edges.
  using ChildIteratorType =
      mapped_iterator<xilinx::AIE::Switchbox::iterator,
                      decltype(&SwitchboxGraphGetSwitchbox)>;
  using ChildEdgeIteratorType = xilinx::AIE::Switchbox::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) {
    return {N->begin(), &SwitchboxGraphGetSwitchbox};
  }
  static ChildIteratorType child_end(NodeRef N) {
    return {N->end(), &SwitchboxGraphGetSwitchbox};
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<xilinx::AIE::SwitchboxGraph *>
    : GraphTraits<xilinx::AIE::Switchbox *> {
  using nodes_iterator = xilinx::AIE::SwitchboxGraph::iterator;
  static NodeRef getEntryNode(xilinx::AIE::SwitchboxGraph *DG) {
    return *DG->begin();
  }
  static nodes_iterator nodes_begin(xilinx::AIE::SwitchboxGraph *DG) {
    return DG->begin();
  }
  static nodes_iterator nodes_end(xilinx::AIE::SwitchboxGraph *DG) {
    return DG->end();
  }
};

inline raw_ostream &operator<<(llvm::raw_ostream &os,
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
