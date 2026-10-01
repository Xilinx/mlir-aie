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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace xilinx::AIE {

// A connection's demand, the cost Dijkstra weighs it by, is
// (demandBase + overCapacityCoeff * iterations it was over capacity) *
// (demandBase + usedCapacityCoeff * streams using it).
constexpr double overCapacityCoeff = 0.1;
constexpr double usedCapacityCoeff = 0.02;
constexpr double demandBase = 1.0;
// A full connection's demand grows by this factor, or a prioritized flow's
// by priorityDemandCoeff, each time another stream takes it.
constexpr double demandCoeff = 1.1;
constexpr double priorityDemandCoeff = std::numeric_limits<int>::max();
constexpr int maxCircuitStreamCapacity = 1;
constexpr int maxPacketStreamCapacity = 32;
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

// A shim's DMA, NOC and PLIO ports reach its switchbox through the shim mux,
// on the South channel these return for a port that sends or receives.
int shimMuxChannelFrom(Port src);
int shimMuxChannelTo(Port dst);

/// The connections from the source ports to the destination ports of a
/// switchbox, if `srcCoords == dstCoords`, or of the wires from one switchbox
/// to its neighbour at `dstCoords`.
struct SwitchboxConnect {
  /// A connection from a source port to a destination port.
  struct Cell {
    bool available = false;
    // weight of Dijkstra's shortest path
    double demand = 0;
    // iterations the connection was over capacity
    int overCapacity = 0;
    // circuit streams using the connection
    int usedCapacity = 0;
    // packet streams using the connection
    int packetFlowCount = 0;
    // only packet streams of this group may share the connection
    int packetGroupId = -1;
    // packet ids routed through the connection (and its crossbar row and
    // column), by the flow routing each; it may be shared only among distinct
    // ids.
    llvm::SmallDenseMap<int, int, 2> packetIds;
    // whether a prioritized flow uses the connection
    bool isPriority = false;
  };

  SwitchboxConnect() = default;
  SwitchboxConnect(TileID coords) : srcCoords(coords), dstCoords(coords) {}
  SwitchboxConnect(TileID srcCoords, TileID dstCoords)
      : srcCoords(srcCoords), dstCoords(dstCoords) {}

  TileID srcCoords, dstCoords;
  std::vector<Port> srcPorts;
  std::vector<Port> dstPorts;
  // The connection from srcPorts[i] to dstPorts[j] is cells[i * dstPorts.size()
  // + j].
  std::vector<Cell> cells;
  // Units of dst ports tied to one arbiter (see planArbiters in
  // AIECreatePathFindFlows.cpp); a unit's leader lists the packet flows
  // (indices into the router's flows) leaving by any of its ports
  llvm::IntEqClasses dstUnits;
  std::vector<llvm::SmallVector<int, 2>> unitPacketFlows;
  // source ports the design already gives packet rules, which circuit streams
  // cannot enter
  std::vector<bool> packetOnlySrc;
  // dst ports packet streams may no longer take
  std::vector<bool> circuitOnlyDst;

  Cell &at(size_t i, size_t j) { return cells[i * dstPorts.size() + j]; }
  const Cell &at(size_t i, size_t j) const {
    return cells[i * dstPorts.size() + j];
  }

  // The index of `p` in srcPorts or dstPorts, or -1.
  int srcIndex(Port p) const { return indexIn(srcPorts, p); }
  int dstIndex(Port p) const { return indexIn(dstPorts, p); }

  // Size the cells and the per-port state to srcPorts and dstPorts.
  void resize() {
    cells.assign(srcPorts.size() * dstPorts.size(), Cell());
    packetOnlySrc.assign(srcPorts.size(), false);
    circuitOnlyDst.assign(dstPorts.size(), false);
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
    for (Cell &c : cells)
      c.demand = (demandBase + overCapacityCoeff * c.overCapacity) *
                 (demandBase + usedCapacityCoeff * c.usedCapacity);
  }

  // Inside each dijkstraShortestPaths iteration, bump demand when it exceeds
  // capacity, all but ruling the connection out if a prioritized flow uses it,
  // to keep prioritized flows' routes.
  static void bumpDemand(Cell &c) {
    if (c.usedCapacity >= maxCircuitStreamCapacity)
      c.demand *= c.isPriority ? priorityDemandCoeff : demandCoeff;
  }

private:
  static int indexIn(llvm::ArrayRef<Port> ports, Port p) {
    const Port *it = llvm::find(ports, p);
    return it == ports.end() ? -1 : it - ports.begin();
  }
};

struct PathEndPoint {
  PathEndPoint() = default;
  PathEndPoint(TileID coords, Port port) : coords(coords), port(port) {}

  TileID coords;
  Port port;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const PathEndPoint &s) {
    return os << "PathEndPoint(" << s.coords << ": " << s.port << ")";
  }

  bool operator<(const PathEndPoint &rhs) const {
    return std::tie(coords, port) < std::tie(rhs.coords, rhs.port);
  }

  bool operator==(const PathEndPoint &rhs) const {
    return std::tie(coords, port) == std::tie(rhs.coords, rhs.port);
  }
};

struct Flow {
  // Packet flows of one group may share channels; -1 for a circuit flow.
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
struct SwitchSetting {
  SwitchSetting() = default;
  SwitchSetting(std::vector<Port> srcs) : srcs(std::move(srcs)) {}
  SwitchSetting(std::vector<Port> srcs, std::vector<Port> dsts)
      : srcs(std::move(srcs)), dsts(std::move(dsts)) {}

  std::vector<Port> srcs;
  std::vector<Port> dsts;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SwitchSetting &s) {
    os << "{";
    llvm::interleaveComma(s.srcs, os);
    os << " -> {";
    llvm::interleaveComma(s.dsts, os);
    return os << "}";
  }

  bool operator<(const SwitchSetting &rhs) const { return srcs < rhs.srcs; }
};

using SwitchSettings = std::map<TileID, SwitchSetting>;

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SwitchSettings &ss) {
  os << "\tSwitchSettings: ";
  for (const auto &[coords, setting] : ss)
    os << coords << ": " << setting << " | ";
  return os << "\n";
}

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

/// Congestion-negotiated routing: each iteration routes every flow by
/// Dijkstra, given each connection's demand, and raises the demand of the
/// connections used over capacity, until no connection is.
class Pathfinder {
public:
  void initialize(int maxCol, int maxRow, const AIETargetModel &targetModel);
  void addFlow(TileID srcCoords, Port srcPort, TileID dstCoords, Port dstPort,
               std::optional<int> packetId, bool isPriorityFlow);
  void sortFlows();
  bool addFixedConnection(SwitchboxOp switchboxOp);
  /// A RoutingFailure if no legal routing is found in `maxIterations`.
  llvm::Expected<Routing> findPaths(int maxIterations);
  void setPacketConstraints(PacketConstraints c) { constraints = std::move(c); }
  /// Loosens the packet constraints by a step after findPaths found no
  /// routing. Returns false once no step is left.
  ///
  /// Packet flows first share channels only with flows they share a
  /// destination with, directly or through others. The steps, each skipped
  /// if it changes nothing: they share channels with any packet flow; tiles
  /// the routing check found out of packet rules or arbiter msels, with no
  /// split to free any, are capped, so packet streams leave them by
  /// packetFanoutCap channels per direction, so by fewer sets of master
  /// ports. Then, if a source sends more than one id, the second id of each
  /// TreeSplit at a tile the tree cannot branch at is routed apart from the
  /// rest, sharing and caps start over, and the two steps follow again.
  bool relax();

private:
  // A directed edge in the dense routing graph: from some node to node `dst`,
  // realized by cell (i, j) of `sb`. `sb` points into `graph`, so demand reads
  // always see the current iteration's weights.
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
  // once. A node's edges are in the order of the PathEndPoints they reach, so
  // Dijkstra breaks ties the same way every run.
  void buildRoutingGraph();

  // The cost of taking `e`, as dijkstraShortestPaths weighs it.
  double edgeWeight(const Edge &e, std::optional<int> packetId,
                    const llvm::BitVector *avoid,
                    const llvm::BitVector *avoidBranch);

  // Dijkstra over the dense graph from the states in `seeds`, each starting at
  // its cost in `seedCosts`. Fills `preds` (predecessor state id, or -1) and
  // `predEdge` (the edge taken to reach each state). Reuses the scratch buffers
  // below. Master ports on an arbiter with flows in `avoid` cost
  // conflictSharePenalty more, and from a state in `branchAvoid`, as much again
  // for the flows it maps to. A channel a flow with the same `packetId` already
  // shares costs as a full one. States in `stops` are reached but not left. The
  // search ends once every state in `targets` is settled; only their `distance`
  // and paths are final then.
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
  // The routing graph, by the tiles a SwitchboxConnect connects: a tile to
  // itself for its switchbox, or to a neighbour for the wires between them.
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
  llvm::DenseSet<TileID> crowdedTiles, cappedTiles;
};

/// Routes the flows of a device with a Pathfinder, and finds or creates the
/// tiles, switchboxes and shim muxes the routing is lowered onto.
class DynamicTileAnalysis {
public:
  /// A RoutingFailure if the flows in `device` have no legal routing.
  llvm::Error runAnalysis(DeviceOp &device);

  int getMaxCol() const { return maxCol; }
  int getMaxRow() const { return maxRow; }

  TileOp getTile(mlir::OpBuilder &builder, int col, int row);
  TileOp getTile(mlir::OpBuilder &builder, const TileID &tileId);
  SwitchboxOp getSwitchbox(mlir::OpBuilder &builder, int col, int row);
  ShimMuxOp getShimMux(mlir::OpBuilder &builder, int col);

  /// The ops at `tile` the analysis has found or created, or null.
  TileOp lookupTile(TileID tile) const { return coordToTile.lookup(tile); }
  SwitchboxOp lookupSwitchbox(TileID tile) const {
    return coordToSwitchbox.lookup(tile);
  }
  ShimMuxOp lookupShimMux(int col) const {
    return coordToShimMux.lookup({col, 0});
  }

  Pathfinder pathfinder;
  /// The routing runAnalysis found.
  Routing routing;
  /// The sources of the circuit flows lowered so far.
  std::set<PathEndPoint> processedFlows;

private:
  // how long until declared unroutable
  static constexpr int maxIterations = 1000;

  int maxCol = 0, maxRow = 0;
  llvm::DenseMap<TileID, TileOp> coordToTile;
  llvm::DenseMap<TileID, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileID, ShimMuxOp> coordToShimMux;
};

} // namespace xilinx::AIE

#endif
