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

#include "llvm/ADT/BitVector.h"

#include <algorithm>
#include <functional>
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
#define ROUTING_CHECK_PENALTY 5
#define CONFLICT_SHARE_PENALTY 4
// See Router::capCrowdedFanOut.
#define PACKET_FANOUT_CAP 2
// A multicast's next destination may branch off any hop its tree already
// takes, starting at this cost per hop back to the source: enough of a
// discount to share hops, while still preferring the shortest path to each
// destination.
#define TREE_SEED_FACTOR 0.9

// A destination's branch is rerouted only when that saves more than this.
#define REROUTE_MIN_SAVING 1e-6

enum class Connectivity { INVALID = 0, AVAILABLE = 1 };

// A shim's DMA, NOC and PLIO ports reach its switchbox through the shim mux,
// on the South channel these return for a port that sends or receives.
int shimMuxChannelFrom(Port src);
int shimMuxChannelTo(Port dst);

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
  // row/column), by the flow routing each; a channel may be shared only among
  // distinct ids.
  std::vector<std::vector<std::map<int, int>>> packetIds;
  // Units of dst ports tied to one arbiter (see planArbiters in
  // AIECreatePathFindFlows.cpp); each dst port links to its unit, and a unit's
  // root lists the packet flows (indices into the router's flows) leaving by
  // any of its ports
  std::vector<int> dstUnit;
  std::vector<llvm::SmallVector<int, 2>> unitPacketFlows;
  // flags indicating priority routings
  std::vector<std::vector<bool>> isPriority;
  // source ports the design already gives packet rules, which circuit streams
  // cannot enter
  std::vector<bool> packetOnlySrc;
  // dst ports packet streams may no longer take
  std::vector<bool> circuitOnlyDst;

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
                     std::vector<std::map<int, int>>(dstPorts.size()));
    isPriority.resize(srcPorts.size(),
                      std::vector<bool>(dstPorts.size(), false));
    packetOnlySrc.resize(srcPorts.size(), false);
    circuitOnlyDst.resize(dstPorts.size(), false);
    unitPacketFlows.resize(dstPorts.size());
    resetUnits();
  }

  void resetUnits() {
    dstUnit.resize(dstPorts.size());
    for (size_t j = 0; j < dstPorts.size(); j++)
      dstUnit[j] = j;
    for (auto &flows : unitPacketFlows)
      flows.clear();
  }

  int unitOf(int j) {
    while (dstUnit[j] != j)
      j = dstUnit[j] = dstUnit[dstUnit[j]];
    return j;
  }

  void addToUnit(int j, int flow) {
    auto &flows = unitPacketFlows[unitOf(j)];
    if (!llvm::is_contained(flows, flow))
      flows.push_back(flow);
  }

  void joinUnits(int j, int k) {
    j = unitOf(j);
    k = unitOf(k);
    if (j == k)
      return;
    dstUnit[k] = j;
    for (int flow : unitPacketFlows[k])
      addToUnit(j, flow);
    unitPacketFlows[k].clear();
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

/// A hop of a packet flow's tree, from one port to the next; `joined` when the
/// flow takes it by joining another flow's tree.
struct TreeHop {
  PathEndPoint from, to;
  bool joined;
};
using PacketTrees = std::map<PathEndPoint, std::vector<TreeHop>>;

/// Whether packet flows from the two sources can deadlock if they share an
/// arbiter; see StreamConflicts::conflict.
using PacketConflict =
    std::function<bool(const PathEndPoint &, const PathEndPoint &)>;

/// Packets `src` sends with ids `a` and `b` share a master port they leave
/// tile `at` by, which puts them on one arbiter. If `apart` is set, the
/// packets with id `b` for that destination on the tile reach it by a slave
/// port of their own instead.
struct TreeSplit {
  PathEndPoint src;
  TileID at;
  int a, b;
  std::optional<PathEndPoint> apart;
};

/// What makes a routing unusable: switchbox connections to move, and where a
/// source's tree has to branch, and crowded tiles (see capCrowdedFanOut).
struct RoutingFaults {
  std::vector<std::pair<TileID, Connect>> connections;
  std::vector<TreeSplit> splits;
  std::vector<TileID> crowded;
};

/// Checks a routing that fits the fabric. Returns what makes it unusable,
/// nothing when it is accepted.
using RoutingCheck = std::function<RoutingFaults(
    const std::map<PathEndPoint, SwitchSettings> &)>;

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
  /// Routings the check rejects count as illegal; the connections it names
  /// are penalized like overused channels so later iterations avoid them, and
  /// the trees it splits branch where it says from then on.
  virtual void setRoutingCheck(RoutingCheck check) {}
  /// Packet flows that conflict are steered off each other's master ports; see
  /// edgeWeight.
  virtual void setPacketConflict(PacketConflict conflict) {}
  /// Why the last findPaths found no routing, empty if it cannot say.
  virtual std::string getFailureReason() const { return {}; }
  /// The channel the last findPaths left overused, empty if none; the routing
  /// check's reason, when it has one, says more.
  virtual std::string getOveruseReason() const { return {}; }
  /// The packet flows' trees in the routing the last findPaths found, by
  /// source.
  virtual PacketTrees getPacketTrees() const { return {}; }
  /// Packet flows from these sources take these trees instead of being
  /// routed.
  virtual void pinPacketTrees(PacketTrees trees) {}
  /// Packet flows share channels only with flows they share a destination
  /// with, directly or through others, unless `share`; then with any other
  /// packet flow. Returns whether that lets flows the last routing kept apart
  /// share.
  virtual bool setShareChannels(bool share) { return false; }
  /// Tiles the routing check found out of packet rules or arbiter msels, with
  /// no split to free any, during the last findPaths: packet streams leave
  /// them by PACKET_FANOUT_CAP channels per direction from now on, so by fewer
  /// sets of master ports. Returns whether that caps any tile not capped
  /// before.
  virtual bool capCrowdedFanOut() { return false; }
  /// Routes the second id of each TreeSplit the routing check asked for at a
  /// tile the tree cannot branch at apart from the rest from now on, and
  /// resets channel sharing and tile caps as at the start (see
  /// AIEPathfinderPass::route). Returns whether a source sends more than one
  /// id.
  virtual bool routeIdsApart() { return false; }
  /// The switch settings of the packets `src` sends with id `id`, if the last
  /// findPaths routed them apart from others `src` sends; else null.
  virtual const SwitchSettings *getIdSettings(const PathEndPoint &src,
                                              int id) const {
    return nullptr;
  }
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
  void setRoutingCheck(RoutingCheck check) override {
    routingCheck = std::move(check);
  }
  void setPacketConflict(PacketConflict conflict) override {
    packetConflict = std::move(conflict);
  }
  std::string getFailureReason() const override { return failureReason; }
  std::string getOveruseReason() const override { return overuseReason; }
  PacketTrees getPacketTrees() const override { return packetTrees; }
  void pinPacketTrees(PacketTrees trees) override {
    pinnedTrees = std::move(trees);
  }
  bool setShareChannels(bool share) override;
  bool capCrowdedFanOut() override;
  bool routeIdsApart() override;
  const SwitchSettings *getIdSettings(const PathEndPoint &src,
                                      int id) const override {
    auto it = idSettings.find({src, id});
    return it == idSettings.end() ? nullptr : &it->second;
  }

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
  static int stateNode(int state) { return state >> 1; }

  // Build the dense integer node numbering and per-node adjacency from `graph`
  // and `flows`. Topology is fixed across congestion iterations, so this runs
  // once. Edge order per node matches the legacy PathEndPoint-sorted order to
  // preserve identical routing output.
  void buildRoutingGraph();

  // The cost of taking `e`, as dijkstraShortestPaths weighs it.
  double edgeWeight(const Edge &e, std::optional<int> packetId,
                    const llvm::BitVector *avoid,
                    const llvm::BitVector *avoidBranch);

  // Dijkstra over the dense graph from the states in `seeds`, each starting at
  // its cost in `seedCosts`. Fills
  // `preds` (predecessor state id, or -1) and `predEdge` (the edge taken to
  // reach each state). Reuses the scratch buffers below. Master ports on an
  // arbiter with flows in `avoid` cost CONFLICT_SHARE_PENALTY more, and from a
  // state in `branchAvoid`, as much again for the flows it maps to. A channel
  // a flow with the same `packetId` already shares costs as a full one. States
  // in `stops` are reached but not left. The search ends once every state in
  // `targets` is settled; only their `distance` and paths are final then.
  void dijkstraShortestPaths(
      llvm::ArrayRef<int> seeds, llvm::ArrayRef<double> seedCosts,
      std::optional<int> packetId, const llvm::BitVector *avoid = nullptr,
      const llvm::DenseMap<int, llvm::BitVector> *branchAvoid = nullptr,
      const llvm::DenseSet<int> *stops = nullptr,
      llvm::ArrayRef<int> targets = {});

  struct RouteState;
  // Route `flow`'s tree around those routed before it this iteration. False,
  // with failureReason set, if it reaches no path to a destination.
  bool routePart(RouteState &st, int flow);
  // Steer the next iteration away from what the routing check faulted.
  // Returns the illegal edges the faults count as.
  int applyRoutingFaults(RouteState &st, const RoutingFaults &faults);
  // Set failureReason or overuseReason to why the last iteration's routing
  // does not fit.
  void explainNoRouting(const RouteState &st);
  bool hasRoom(const SwitchboxConnect &sb) const;

  // Flows to be routed
  std::vector<Flow> flows;
  // The packet ids each source sends each destination.
  std::map<std::pair<PathEndPoint, PathEndPoint>, llvm::SmallVector<int, 2>>
      packetIdsTo;
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

  RoutingCheck routingCheck;
  PacketConflict packetConflict;
  std::string failureReason, overuseReason;
  PacketTrees packetTrees, pinnedTrees;
  bool shareChannels = false, idsApart = false;
  std::set<TileID> crowdedTiles, cappedTiles;
  std::map<std::pair<PathEndPoint, int>, SwitchSettings> idSettings;
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
  /// Why the last routing the routing check rejected was unusable, reported
  /// if no usable routing is found.
  std::string routingFailureReason;

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
