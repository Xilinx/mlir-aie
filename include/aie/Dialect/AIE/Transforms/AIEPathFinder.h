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

    std::vector<WireBundle> bundles = {
        WireBundle::Core,  WireBundle::DMA,  WireBundle::FIFO,
        WireBundle::South, WireBundle::West, WireBundle::North,
        WireBundle::East,  WireBundle::PLIO, WireBundle::NOC,
        WireBundle::Trace, WireBundle::Ctrl};

    for (WireBundle bundle : bundles) {
      int maxCapacity =
          targetModel.getNumSourceSwitchboxConnections(col, row, bundle);
      if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
        // wordaround for shimMux, todo: integrate shimMux into routable grid
        maxCapacity =
            (bundle == WireBundle::PLIO)
                ? 8
                : targetModel.getNumSourceShimMuxConnections(col, row, bundle);
      }

      for (int channel = 0; channel < maxCapacity; channel++) {
        Port inPort = {bundle, channel};
        inPortToId[inPort] = inPortId;
        inPortId++;
      }

      maxCapacity =
          targetModel.getNumDestSwitchboxConnections(col, row, bundle);
      if (targetModel.isShimNOCorPLTile(col, row) && maxCapacity == 0) {
        // wordaround for shimMux, todo: integrate shimMux into routable grid
        maxCapacity =
            (bundle == WireBundle::PLIO)
                ? 8
                : targetModel.getNumDestShimMuxConnections(col, row, bundle);
      }
      for (int channel = 0; channel < maxCapacity; channel++) {
        Port outPort = {bundle, channel};
        outPortToId[outPort] = outPortId;
        outPortId++;
      }
    }

    connectionMatrix.resize(inPortId, std::vector<int>(outPortId, 0));

    // illegal connection
    for (const auto &[inPort, inId] : inPortToId) {
      for (const auto &[outPort, outId] : outPortToId) {
        if (!targetModel.isLegalTileConnection(col, row, inPort.bundle,
                                               inPort.channel, outPort.bundle,
                                               outPort.channel))
          connectionMatrix[inId][outId] = -1;

        if (targetModel.isShimNOCorPLTile(col, row)) {
          // wordaround for shimMux, todo: integrate shimMux into routable grid
          auto isBundleInList = [](WireBundle bundle,
                                   std::vector<WireBundle> bundles) {
            return std::find(bundles.begin(), bundles.end(), bundle) !=
                   bundles.end();
          };
          std::vector<WireBundle> bundles = {WireBundle::DMA, WireBundle::NOC,
                                             WireBundle::PLIO};
          if (isBundleInList(inPort.bundle, bundles) ||
              isBundleInList(outPort.bundle, bundles))
            connectionMatrix[inId][outId] = 0;
        }
      }
    }
  }

  // given a outPort, find availble input channel
  int findAvailableChannelIn(WireBundle inBundle, Port outPort, bool isPkt) {
    int outId = outPortToId.at(outPort);

    if (isPkt) {
      for (const auto &[inPort, inId] : inPortToId) {
        if (inPort.bundle == inBundle && connectionMatrix[inId][outId] >= 0) {
          bool available = true;
          if (inPortPktCount.count(inPort) == 0) {
            for (const auto &[outPort, outId] : outPortToId) {
              if (connectionMatrix[inId][outId] == 1) {
                // occupied by others as circuit-switched
                available = false;
                break;
              }
            }
          } else {
            if (inPortPktCount[inPort] >= maxPktStream) {
              // occupied by others as packet-switched but exceed max packet
              // stream capacity
              available = false;
            }
          }
          if (available)
            return inPort.channel;
        }
      }
    } else {
      for (const auto &[inPort, inId] : inPortToId) {
        if (inPort.bundle == inBundle && connectionMatrix[inId][outId] == 0) {
          bool available = true;
          for (const auto &[outPort, outId] : outPortToId) {
            if (connectionMatrix[inId][outId] == 1) {
              available = false;
              break;
            }
          }
          if (available)
            return inPort.channel;
        }
      }
    }

    // couldn't find any availale channel
    return -1;
  }

  bool allocate(Port inPort, Port outPort, bool isPkt) {
    int inId = inPortToId.at(inPort);
    int outId = outPortToId.at(outPort);

    // invalid connection
    if (connectionMatrix[inId][outId] == -1)
      return false;

    if (isPkt) {
      // a packet-switched stream to be allocated
      if (inPortPktCount.count(inPort) == 0) {
        for (const auto &[outPort, outId] : outPortToId) {
          if (connectionMatrix[inId][outId] == 1)
            // occupied by others as circuit-switched, allocation fail!
            return false;
        }
        // empty channel, allocation succeed!
        inPortPktCount[inPort] = 1;
        connectionMatrix[inId][outId] = 1;
        return true;
      } else {
        if (inPortPktCount[inPort] >= maxPktStream) {
          // occupied by others as packet-switched but exceed max packet stream
          // capacity, allocation fail!
          return false;
        } else {
          // valid packet-switched, allocation succeed!
          inPortPktCount[inPort]++;
          return true;
        }
      }
    } else {
      // a circuit-switched stream to be allocated
      if (connectionMatrix[inId][outId] == 0) {
        // empty channel, allocation succeed!
        connectionMatrix[inId][outId] = 1;
        return true;
      } else {
        // occupied by others, allocation fail!
        return false;
      }
    }
  }

  void clearAllocation() {
    for (int inId = 0; inId < inPortId; inId++) {
      for (int outId = 0; outId < outPortId; outId++) {
        if (connectionMatrix[inId][outId] != -1) {
          connectionMatrix[inId][outId] = 0;
        }
      }
    }
    inPortPktCount.clear();
  }

  friend std::ostream &operator<<(std::ostream &os, const SwitchboxNode &s) {
    os << "Switchbox(" << s.col << ", " << s.row << ")";
    return os;
  }

  GENERATE_TO_STRING(SwitchboxNode);

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SwitchboxNode &s) {
    os << to_string(s);
    return os;
  }

  bool operator<(const SwitchboxNode &rhs) const { return this->id < rhs.id; }

  bool operator==(const SwitchboxNode &rhs) const { return this->id == rhs.id; }

  int col, row, id;
  int inPortId = 0, outPortId = 0;
  std::map<Port, int> inPortToId, outPortToId;

  // tenary representation of switchbox connectivity
  // -1: invalid in arch, 0: empty and available, 1: occupued
  std::vector<std::vector<int>> connectionMatrix;

  // input ports with incoming packet-switched streams
  std::map<Port, int> inPortPktCount;

  // up to 32 packet-switched stram through a port
  const int maxPktStream = 32;
};

using ChannelEdge = struct ChannelEdge {
  ChannelEdge(SwitchboxNode *src, SwitchboxNode *target)
      : src(src), target(target) {

    // get bundle from src to target
    if (src->col == target->col) {
      if (src->row > target->row)
        bundle = WireBundle::South;
      else
        bundle = WireBundle::North;
    } else {
      if (src->col > target->col)
        bundle = WireBundle::West;
      else
        bundle = WireBundle::East;
    }

    // maximum number of routing resources
    maxCapacity = 0;
    for (auto &[outPort, _] : src->outPortToId) {
      if (outPort.bundle == bundle) {
        maxCapacity++;
      }
    }
  }

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

  SwitchboxNode *src;
  SwitchboxNode *target;

  int maxCapacity;
  WireBundle bundle;
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
  virtual bool addFixedConnection(SwitchboxOp switchboxOp) = 0;
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
  bool addFixedConnection(SwitchboxOp switchboxOp) override;
  std::optional<std::map<PathEndPoint, SwitchSettings>>
  findPaths(int maxIterations) override;

  std::map<SwitchboxNode *, SwitchboxNode *>
  dijkstraShortestPaths(SwitchboxNode *src);

  SwitchboxNode getSwitchboxNode(TileID coords) override {
    return grid.at(coords);
  }

private:
  // Flows to be routed
  std::vector<FlowNode> flows;

  // Grid of switchboxes available
  std::map<TileID, SwitchboxNode> grid;

  // Use a list instead of a vector because nodes have an edge list of raw
  // pointers to edges (so growing a vector would invalidate the pointers).
  std::list<ChannelEdge> edges;

  // Use Dijkstra's shortest path to find routes, and use "demand" as the
  // weights.
  std::map<ChannelEdge *, double> demand;

  // History of Channel being over capacity
  std::map<ChannelEdge *, int> overCapacity;

  // how many flows are actually using this Channel
  std::map<ChannelEdge *, int> usedCapacity;
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
