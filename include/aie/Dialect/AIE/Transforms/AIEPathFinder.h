//===- AIEPathfinder.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PATHFINDER_H
#define AIE_PATHFINDER_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include <algorithm>
#include <iostream>
#include <list>
#include <optional>
#include <set>

namespace xilinx::AIE {

#define OVER_CAPACITY_COEFF 0.1
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1
#define DEMAND_BASE 1.0
#define MAX_CIRCUIT_STREAM_CAPACITY 1
#define MAX_PACKET_STREAM_CAPACITY 32

enum class Connectivity { INVALID = 0, AVAILABLE = 1 };

using SwitchboxConnect = struct SwitchboxConnect {
  SwitchboxConnect() = default;
  SwitchboxConnect(TileID coords) : srcCoords(coords), dstCoords(coords) {}
  SwitchboxConnect(TileID srcCoords, TileID dstCoords)
      : srcCoords(srcCoords), dstCoords(dstCoords) {}

  TileID srcCoords, dstCoords;
  std::vector<Port> srcPorts;
  std::vector<Port> dstPorts;
  // connectivity between ports
  std::vector<std::vector<Connectivity>> connectivity;
  // weights of Dijkstra's shortest path
  std::vector<std::vector<double>> demand;
  // history of Channel being over capacity
  std::vector<std::vector<int>> overCapacity;
  // how many circuit streams are actually using this Channel
  std::vector<std::vector<int>> usedCapacity;
  // how many packet streams are actually using this Channel
  std::vector<std::vector<int>> packetFlowCount;
  // only sharing the channel with the same packet group id
  std::vector<std::vector<int>> packetGroupId;
  // packet ids currently routed through each channel (and its crossbar
  // row/column); a channel may be shared only among distinct ids.
  std::vector<std::vector<std::set<int>>> packetIds;
  // flags indicating priority routings
  std::vector<std::vector<bool>> isPriority;

  // resize the matrices to the size of srcPorts and dstPorts
  void resize() {
    connectivity.resize(
        srcPorts.size(),
        std::vector<Connectivity>(dstPorts.size(), Connectivity::INVALID));
    demand.resize(srcPorts.size(), std::vector<double>(dstPorts.size(), 0.0));
    overCapacity.resize(srcPorts.size(), std::vector<int>(dstPorts.size(), 0));
    usedCapacity.resize(srcPorts.size(), std::vector<int>(dstPorts.size(), 0));
    packetFlowCount.resize(srcPorts.size(),
                           std::vector<int>(dstPorts.size(), 0));
    packetGroupId.resize(srcPorts.size(), std::vector<int>(dstPorts.size(), 0));
    packetIds.resize(srcPorts.size(),
                     std::vector<std::set<int>>(dstPorts.size()));
    isPriority.resize(srcPorts.size(),
                      std::vector<bool>(dstPorts.size(), false));
  }

  // update demand at the beginning of each dijkstraShortestPaths iteration
  void updateDemand() {
    for (size_t i = 0; i < srcPorts.size(); i++) {
      for (size_t j = 0; j < dstPorts.size(); j++) {
        double history = DEMAND_BASE + OVER_CAPACITY_COEFF * overCapacity[i][j];
        double congestion =
            DEMAND_BASE + USED_CAPACITY_COEFF * usedCapacity[i][j];
        demand[i][j] = history * congestion;
      }
    }
  }

  // Inside each dijkstraShortestPaths interation, bump demand when exceeds
  // capacity. If isPriority is true, then set demand to INF to ensure routing
  // consistency for prioritized flows
  void bumpDemand(size_t i, size_t j) {
    if (usedCapacity[i][j] >= MAX_CIRCUIT_STREAM_CAPACITY) {
      demand[i][j] *=
          isPriority[i][j] ? std::numeric_limits<int>::max() : DEMAND_COEFF;
    }
  }
};

using PathEndPoint = struct PathEndPoint {
  PathEndPoint() = default;
  PathEndPoint(TileID coords, Port port) : coords(coords), port(port) {}

  TileID coords;
  Port port;

  friend std::ostream &operator<<(std::ostream &os, const PathEndPoint &s) {
    os << "PathEndPoint(" << s.coords << ": " << s.port << ")";
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
    return std::tie(coords, port) < std::tie(rhs.coords, rhs.port);
  }

  bool operator==(const PathEndPoint &rhs) const {
    return std::tie(coords, port) == std::tie(rhs.coords, rhs.port);
  }
};

using Flow = struct Flow {
  int packetGroupId;
  bool isPriorityFlow;
  PathEndPoint src;
  std::vector<PathEndPoint> dsts;
  // packet id carried by this flow (nullopt for circuit flows); a channel may
  // be shared only among distinct ids so same-id flows never merge then fan
  // out.
  std::optional<int> packetId;
};

// A SwitchSetting defines the required settings for a Switchbox for a flow
// SwitchSetting.srcs is the fanin
// SwitchSetting.dsts is the fanout
using SwitchSetting = struct SwitchSetting {
  SwitchSetting() = default;
  SwitchSetting(std::vector<Port> srcs) : srcs(std::move(srcs)) {}
  SwitchSetting(std::vector<Port> srcs, std::vector<Port> dsts)
      : srcs(std::move(srcs)), dsts(std::move(dsts)) {}

  std::vector<Port> srcs;
  std::vector<Port> dsts;

  // friend definition (will define the function as a non-member function of
  // the namespace surrounding the class).
  friend std::ostream &operator<<(std::ostream &os,
                                  const SwitchSetting &setting) {
    os << "{"
       << join(llvm::map_range(setting.srcs,
                               [](const Port &port) {
                                 std::ostringstream ss;
                                 ss << port;
                                 return ss.str();
                               }),
               ", ")
       << " -> "
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

  bool operator<(const SwitchSetting &rhs) const { return srcs < rhs.srcs; }
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
                       Port dstPort, std::optional<int> packetId,
                       bool isPriorityFlow) = 0;
  virtual void sortFlows() = 0;
  virtual bool addFixedConnection(SwitchboxOp switchboxOp) = 0;
  virtual std::optional<std::map<PathEndPoint, SwitchSettings>>
  findPaths(int maxIterations) = 0;
};

class Pathfinder : public Router {
public:
  Pathfinder() = default;
  void initialize(int maxCol, int maxRow,
                  const AIETargetModel &targetModel) override;
  void addFlow(TileID srcCoords, Port srcPort, TileID dstCoords, Port dstPort,
               std::optional<int> packetId, bool isPriorityFlow) override;
  void sortFlows() override;
  bool addFixedConnection(SwitchboxOp switchboxOp) override;
  std::optional<std::map<PathEndPoint, SwitchSettings>>
  findPaths(int maxIterations) override;

private:
  // A directed edge in the dense routing graph: from some node to node `dst`,
  // realized by switchbox-connect `sb` at matrix position (i, j). `sb`, `i` and
  // `j` index live into `graph` so demand reads always see the current
  // iteration's weights.
  struct Edge {
    int dst;
    SwitchboxConnect *sb;
    int i;
    int j;
  };

  // A `Port` is (bundle, channel) with no direction, so one dense node stands
  // for both a switchbox port's input side and its output side. Dijkstra
  // therefore searches over states, not nodes: a state is a node paired with
  // the side of that port the stream is currently on. A stream enters a
  // switchbox on an input port, crosses the crossbar once to an output port,
  // and then rides the wire to the neighbour's input port -- so In only ever
  // takes intra-switchbox edges and Out only ever takes inter-switchbox ones.
  // Without the split, Dijkstra can chain two crossbar hops through one port
  // and turn the stream around inside a switchbox; the settings it emits then
  // dead-end and AIECreatePathFindFlows reports the flow as unroutable.
  enum PortSide : int { In = 0, Out = 1 };
  static int stateId(int nodeId, PortSide side) { return 2 * nodeId + side; }
  static int stateNode(int stateId) { return stateId >> 1; }

  // Build the dense integer node numbering and per-node adjacency from `graph`
  // and `flows`. Topology is fixed across congestion iterations, so this runs
  // once. Edge order per node matches the legacy PathEndPoint-sorted order to
  // preserve identical routing output.
  void buildRoutingGraph();

  // Dijkstra over the dense graph from dense node `srcId`, whose port is the
  // stream's entry into its switchbox and so starts on the In side. Fills
  // `preds` (predecessor state id, or -1) and `predEdge` (the edge taken to
  // reach each state). Reuses the scratch buffers below.
  void dijkstraShortestPaths(int srcId);

  // Flows to be routed
  std::vector<Flow> flows;
  // Represent all routable paths as a graph
  // The key is a pair of TileIDs representing the connectivity from srcTile to
  // dstTile If srcTile == dstTile, it represents connections inside the same
  // switchbox otherwise, it represents connections (South, North, West, East)
  // accross two switchboxes
  std::map<std::pair<TileID, TileID>, SwitchboxConnect> graph;

  // Dense routing graph (built once by buildRoutingGraph()).
  bool graphBuilt = false;
  std::map<PathEndPoint, int> nodeIds;      // PathEndPoint -> dense id
  std::vector<PathEndPoint> nodes;          // dense id -> PathEndPoint
  std::vector<std::vector<Edge>> adjacency; // dense id -> out-edges

  // Dijkstra scratch, indexed by state id (2 * nodes.size()) and reused across
  // calls.
  std::vector<double> distance;
  std::vector<uint64_t> indexInHeap;
  std::vector<int8_t> colors;
  std::vector<int> preds;
  std::vector<Edge> predEdge;

  int getOrAddNodeId(const PathEndPoint &pep);
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
  DynamicTileAnalysis(mlir::Operation *op)
      : pathfinder(std::make_shared<Pathfinder>()) {}

  mlir::LogicalResult runAnalysis(DeviceOp &device);

  int getMaxCol() const { return maxCol; }
  int getMaxRow() const { return maxRow; }

  TileOp getTile(mlir::OpBuilder &builder, int col, int row);
  TileOp getTile(mlir::OpBuilder &builder, const TileID &tileId);

  SwitchboxOp getSwitchbox(mlir::OpBuilder &builder, int col, int row);

  ShimMuxOp getShimMux(mlir::OpBuilder &builder, int col);
};

// Get enum int value from WireBundle.
int getWireBundleAsInt(WireBundle bundle);

} // namespace xilinx::AIE

namespace llvm {

inline raw_ostream &operator<<(raw_ostream &os,
                               const xilinx::AIE::SwitchSettings &ss) {
  std::stringstream s;
  s << "\tSwitchSettings: ";
  for (const auto &[coords, setting] : ss) {
    s << coords << ": " << setting << " | ";
  }
  s << "\n";
  os << s.str();
  return os;
}

} // namespace llvm

#endif
