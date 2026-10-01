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
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/Support/Error.h"

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

// History added to a connection each time the routing check rejects it.
constexpr int routingCheckPenalty = 5;
// Cost added to a hop that shares an arbiter unit with a flow to avoid.
constexpr double conflictSharePenalty = 4;
// Channels per direction packet streams leave a capped tile by (see
// Pathfinder::relax). A heuristic: fewer channels mean fewer master
// sets per tile, but more flows on each.
constexpr int packetFanoutCap = 2;
// A multicast's next destination may branch off any hop its tree already
// takes, starting at this cost per hop back to the source: enough of a
// discount to share hops, while still preferring the shortest path to each
// destination.
constexpr double treeSeedFactor = 0.9;
// A destination's branch is rerouted only when that saves more than this.
constexpr double rerouteMinSaving = 1e-6;

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
  // AIECreatePathFindFlows.cpp); a unit's leader lists the packet flows
  // (indices into the router's flows) leaving by any of its ports
  llvm::IntEqClasses dstUnits;
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
    dstUnits.clear();
    dstUnits.grow(dstPorts.size());
    for (auto &flows : unitPacketFlows)
      flows.clear();
  }

  llvm::ArrayRef<int> unitFlows(int j) const {
    return unitPacketFlows[dstUnits.findLeader(j)];
  }

  void addToUnit(int j, int flow) {
    auto &flows = unitPacketFlows[dstUnits.findLeader(j)];
    if (!llvm::is_contained(flows, flow))
      flows.push_back(flow);
  }

  void joinUnits(int j, int k) {
    j = dstUnits.findLeader(j);
    k = dstUnits.findLeader(k);
    if (j == k)
      return;
    int leader = dstUnits.join(j, k);
    int other = leader == j ? k : j;
    for (int flow : unitPacketFlows[other])
      addToUnit(leader, flow);
    unitPacketFlows[other].clear();
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
/// source's tree has to branch, and crowded tiles (see Pathfinder::relax).
struct RoutingFaults {
  std::vector<std::pair<TileID, Connect>> connections;
  std::vector<TreeSplit> splits;
  std::vector<TileID> crowded;
};

/// A routing findPaths found.
struct Routing {
  /// The switch settings of each source's flows.
  std::map<PathEndPoint, SwitchSettings> settings;
  /// The packet flows' trees, by source.
  PacketTrees packetTrees;
  /// The switch settings of the packets a source sends with an id, where they
  /// are routed apart from the others the source sends.
  std::map<std::pair<PathEndPoint, int>, SwitchSettings> idSettings;
};

/// Why no legal routing was found, reported at `loc`, or the device if unset,
/// with `reason` if the router can say. A routing check's failure also names
/// the faults the router has to move.
class RoutingFailure : public llvm::ErrorInfo<RoutingFailure> {
public:
  static char ID;

  explicit RoutingFailure(std::string reason, RoutingFaults faults = {},
                          std::optional<mlir::Location> loc = std::nullopt)
      : reason(std::move(reason)), faults(std::move(faults)), loc(loc) {}

  void log(llvm::raw_ostream &os) const override;
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  std::string reason;
  RoutingFaults faults;
  std::optional<mlir::Location> loc;
};

/// Checks a routing that fits the fabric: a RoutingFailure if it is unusable,
/// success if it is accepted.
using RoutingCheck = std::function<llvm::Error(const Routing &)>;

/// What packet flows are routed under.
struct PacketConstraints {
  /// Routings the check rejects count as illegal; the connections it names
  /// are penalized like overused channels so later iterations avoid them, and
  /// the trees it splits branch where it says from then on.
  RoutingCheck check;
  /// Packet flows that conflict are steered off each other's master ports; see
  /// edgeWeight.
  PacketConflict conflict;
  /// Packet flows from these sources take these trees instead of being
  /// routed.
  PacketTrees pinned;
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
                       Port dstPort, std::optional<int> packetId,
                       bool isPriorityFlow) = 0;
  virtual void sortFlows() = 0;
  virtual bool addFixedConnection(SwitchboxOp switchboxOp) = 0;
  /// A RoutingFailure if no legal routing is found in `maxIterations`.
  virtual llvm::Expected<Routing> findPaths(int maxIterations) = 0;
  virtual void setPacketConstraints(PacketConstraints constraints) {}
  /// Loosens the packet constraints by a step after findPaths found no
  /// routing. Returns false once no step is left.
  virtual bool relax() { return false; }
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
  llvm::Expected<Routing> findPaths(int maxIterations) override;
  void setPacketConstraints(PacketConstraints c) override {
    constraints = std::move(c);
  }
  /// Packet flows first share channels only with flows they share a
  /// destination with, directly or through others. The steps, each skipped
  /// if it changes nothing: they share channels with any packet flow; tiles
  /// the routing check found out of packet rules or arbiter msels, with no
  /// split to free any, are capped, so packet streams leave them by
  /// packetFanoutCap channels per direction, so by fewer sets of master
  /// ports. Then, if a source sends more than one id, the second id of each
  /// TreeSplit at a tile the tree cannot branch at is routed apart from the
  /// rest, sharing and caps start over, and the two steps follow again.
  bool relax() override;

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
  // arbiter with flows in `avoid` cost conflictSharePenalty more, and from a
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
  // Route `flow`'s tree around those routed before it this iteration. A
  // RoutingFailure if it reaches no path to a destination.
  llvm::Error routePart(RouteState &st, int flow);
  // Steer the next iteration away from what the routing check faulted.
  // Returns the illegal edges the faults count as.
  int applyRoutingFaults(RouteState &st, const RoutingFaults &faults);
  // Why the last iteration's routing does not fit.
  std::string explainNoRouting(const RouteState &st) const;
  bool hasRoom(const SwitchboxConnect &sb) const;
  // The steps of relax.
  bool setShareChannels(bool share);
  bool capCrowdedFanOut();
  bool routeIdsApart();

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

  PacketConstraints constraints;
  // Why the routing check rejected the last routing it rejected.
  std::string checkReason;
  bool shareChannels = false, idsApart = false;
  int relaxStep = 0;
  std::set<TileID> crowdedTiles, cappedTiles;
};

// DynamicTileAnalysis integrates the Pathfinder class into the MLIR
// environment. It passes flows to the Pathfinder as ordered pairs of ints.
// Detailed routing is received as SwitchboxSettings
// It then converts these settings to MLIR operations
class DynamicTileAnalysis {
public:
  int maxCol, maxRow;
  std::shared_ptr<Router> pathfinder;
  Routing routing;
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

  /// A RoutingFailure if the flows in `device` have no legal routing.
  llvm::Error runAnalysis(DeviceOp &device);

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
