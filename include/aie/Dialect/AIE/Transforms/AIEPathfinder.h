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

#include "aie/Dialect/AIE/IR/AIEDialect.h" // for WireBundle and Port

#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/GraphTraits.h"

#include <list>

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

  int col, row;
};

class Channel : public ChannelBase {
public:
  explicit Channel(Switchbox &target) = delete;
  Channel(Switchbox &src, Switchbox &target, WireBundle bundle,
          uint32_t maxCapacity)
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

  Switchbox &src;
  WireBundle bundle;
  uint32_t maxCapacity = 0; // maximum number of routing resources
  float demand = 0.0;       // indicates how many flows want to use this Channel
  uint32_t usedCapacity = 0; // how many flows are actually using this Channel
  std::set<uint32_t> fixedCapacity; // channels not available to the algorithm
  uint32_t overCapacityCount = 0;   // history of Channel being over capacity
};

class SwitchboxGraph : public SwitchboxGraphBase {
public:
  SwitchboxGraph() = default;
  ~SwitchboxGraph() = default;
};

// A SwitchSetting defines the required settings for a Switchbox for a flow
// SwitchSetting.first is the incoming signal
// SwitchSetting.second is the fanout
typedef std::pair<Port, std::set<Port>> SwitchSetting;
typedef std::map<Switchbox *, SwitchSetting> SwitchSettings;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
typedef std::pair<Switchbox *, Port> PathEndPoint;
typedef std::pair<PathEndPoint, std::vector<PathEndPoint>> Flow;

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
  Pathfinder(int maxCol, int maxRow, DeviceOp &d);
  void addFlow(TileID srcCoords, Port srcPort, TileID dstCoords, Port dstPort);
  void addFixedConnection(TileID coord, Port port);
  bool isLegal();
  std::map<PathEndPoint, SwitchSettings> findPaths(int maxIterations = 1000);

  Switchbox *getSwitchbox(TileID coords) {
    auto sb = std::find_if(graph.begin(), graph.end(), [&](Switchbox *sb) {
      return sb->col == coords.first && sb->row == coords.second;
    });
    assert(sb != graph.end() && "couldn't find sb");
    return *sb;
  }
};

} // namespace xilinx::AIE

namespace llvm {
using namespace xilinx::AIE;

template <> struct GraphTraits<Switchbox *> {
  using NodeRef = Switchbox *;

  static Switchbox *SwitchboxGraphGetSwitchbox(DGEdge<Switchbox, Channel> *P) {
    return &P->getTargetNode();
  }

  // Provide a mapped iterator so that the GraphTrait-based implementations can
  // find the target nodes without having to explicitly go through the edges.
  using ChildIteratorType =
      mapped_iterator<Switchbox::iterator,
                      decltype(&SwitchboxGraphGetSwitchbox)>;
  using ChildEdgeIteratorType = Switchbox::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &SwitchboxGraphGetSwitchbox);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &SwitchboxGraphGetSwitchbox);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<SwitchboxGraph *> : public GraphTraits<Switchbox *> {
  using nodes_iterator = SwitchboxGraph::iterator;
  static NodeRef getEntryNode(SwitchboxGraph *DG) { return *DG->begin(); }
  static nodes_iterator nodes_begin(SwitchboxGraph *DG) { return DG->begin(); }
  static nodes_iterator nodes_end(SwitchboxGraph *DG) { return DG->end(); }
};

inline raw_ostream &operator<<(raw_ostream &OS, const Switchbox &S) {
  OS << "Switchbox(" << S.col << ", " << S.row << ")";
  return OS;
}

inline raw_ostream &operator<<(raw_ostream &OS, const Channel &C) {
  OS << "Channel(src=" << C.src << ", dst=" << C.getTargetNode() << ")";
  return OS;
}

} // namespace llvm

#endif
