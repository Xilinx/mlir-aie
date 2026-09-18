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
#include <array>
#include <iostream>
#include <list>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

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
  // persistent per-cell demand added on top of the congestion weight each
  // updateDemand iteration. Seeded once before findPaths for an adaptive
  // control pinning (steers control off cells config data uses); 0.0 for every
  // other analysis, so demand stays byte-identical when it is not seeded.
  std::vector<std::vector<double>> designDemand;
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
    designDemand.resize(srcPorts.size(),
                        std::vector<double>(dstPorts.size(), 0.0));
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
        demand[i][j] = history * congestion + designDemand[i][j];
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
  // packet id carried by this flow (nullopt for circuit flows); representative
  // (first) id, kept for group-id assignment and pinned-route replay.
  std::optional<int> packetId;
  // per-destination packet id, index-parallel to dsts. A co-sourced multi-dest
  // packet flow carries a DISTINCT id per destination; routing each dst under
  // its own id lets accountEdge share one trunk across distinct ids and fan
  // out.
  std::vector<std::optional<int>> dstPacketIds;
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

// A design-demand field: per switchbox-connect cell
// (srcCoords, dstCoords, srcPort, dstPort) -> extra demand. Seeded into the
// pathfinder before an adaptive control pinning so control routes around the
// ports config data uses. Keyed by ports (not matrix [i][j]) so it stays valid
// regardless of per-instance port ordering.
using DesignField = std::map<std::tuple<TileID, TileID, Port, Port>, double>;

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
  // Pin a flow (keyed by its source) to a captured route: findPaths replays it
  // verbatim instead of running Dijkstra, so the flow cannot drift across
  // congestion iterations or across sibling devices.
  virtual void pinRoute(const PathEndPoint &src,
                        const SwitchSettings &route) = 0;
  // Seed a persistent per-cell demand field (adaptive pinning). Added on
  // top of the congestion demand each iteration; an empty field leaves routing
  // byte-identical. Default no-op so routers that never seed are unaffected.
  virtual void seedDesignDemand(const DesignField &field) {}
  // Penalize cross-column (East/West) hops so an adaptive control capture
  // stays column-local. Default no-op so unaffected routers stay
  // byte-identical.
  virtual void seedColumnLocalControl(double penalty) {}
  // Route a priority control multicast's destinations farthest-first so the
  // longest path establishes the column trunk and nearer destinations reuse it
  // (one coherent output channel per source), instead of each destination
  // opening a fresh channel and fragmenting the shim packet-rule cover. Enabled
  // only for the adaptive capture; default off so other routing (blind
  // capture, per-device replay) stays byte-identical.
  virtual void setCoherentControlCapture(bool on) {}
  // Mark this a control-overlay (reconfiguration) routing run. Scopes the
  // multi-destination packet trunk consolidation to control-overlay compiles
  // (pinning on OR off) so a plain, non-reconfiguration design routes
  // byte-identically to upstream. Default off.
  virtual void setControlOverlayRouting(bool on) {}
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
  void pinRoute(const PathEndPoint &src, const SwitchSettings &route) override;
  void seedDesignDemand(const DesignField &field) override;
  void seedColumnLocalControl(double penalty) override;
  void setCoherentControlCapture(bool on) override {
    coherentControlCapture = on;
  }
  void setControlOverlayRouting(bool on) override {
    controlOverlayRouting = on;
  }
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
  static int stateNode(int state) { return state >> 1; }

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

  // Apply one routed edge's per-iteration accounting (priority flag, packet-id
  // sharing bookkeeping, usedCapacity, bumpDemand) to switchbox-connect `sb` at
  // matrix cell (i, j). Shared by the Dijkstra trace and the pinned-route
  // replay so both consume a channel's capacity identically.
  void accountEdge(SwitchboxConnect &sb, int i, int j, bool isPriority,
                   int packetGroupId, std::optional<int> packetId);

  // Replay a pinned flow's captured route: walk its intra-tile connections and
  // the implicit inter-tile hops (reconstructed with buildRoutingGraph's
  // neighbor logic), running accountEdge on each so data negotiates around
  // control's true footprint, and stamp every port the route touches as
  // processed so a co-sourced data leg's back-trace stops where it rejoins
  // control (a shared slave). Control itself is fixed to the captured route.
  void replayPinnedRoute(bool isPriority, int packetGroupId,
                         std::optional<int> packetId,
                         const SwitchSettings &route,
                         std::vector<uint32_t> &processedStamp,
                         uint32_t curStamp);

  // Structurally reserve every control master port a pinned route drives so a
  // non-co-sourced data flow cannot be routed onto it: on each control master
  // column, mark connectivity INVALID for every source row that is NOT one of
  // control's own. buildRoutingGraph then omits those edges, so a forced data-
  // onto-control-master share becomes a clean "Unable to find a legal routing"
  // instead of a silent in-band repoint wedge. A no-op when pinnedRoutes is
  // empty (non-pinning), so OFF routing is unchanged. Returns false if
  // control's pinned route collides with a pre-placed circuit connection (a
  // control cell already INVALID), which must fail closed rather than
  // double-use the cell.
  bool reservePinnedControlMasters();

  // Flows to be routed
  std::vector<Flow> flows;

  // Flows pinned to a captured route (keyed by source): replayed, not routed.
  std::map<PathEndPoint, SwitchSettings> pinnedRoutes;
  // Represent all routable paths as a graph
  // The key is a pair of TileIDs representing the connectivity from srcTile to
  // dstTile If srcTile == dstTile, it represents connections inside the same
  // switchbox otherwise, it represents connections (South, North, West, East)
  // accross two switchboxes
  std::map<std::pair<TileID, TileID>, SwitchboxConnect> graph;

  // Design-aware capture only: route each priority control multicast's
  // destinations farthest-first so the trunk is established once and reused.
  bool coherentControlCapture = false;

  // Control-overlay (reconfiguration) routing run: gates the multi-destination
  // packet trunk consolidation so a plain, non-reconfiguration design routes
  // byte-identically to upstream. Set for control-overlay compiles (pinning on
  // or off) and the adaptive capture.
  bool controlOverlayRouting = false;

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

  // skipControlFlows drops priority_route control packet flows (route config
  // DATA only, for adaptive pinning demand capture). baseline, when set,
  // seeds the pathfinder's per-cell demand field before routing.
  mlir::LogicalResult runAnalysis(DeviceOp &device,
                                  bool skipControlFlows = false,
                                  const DesignField *baseline = nullptr);

  int getMaxCol() const { return maxCol; }
  int getMaxRow() const { return maxRow; }

  TileOp getTile(mlir::OpBuilder &builder, int col, int row);
  TileOp getTile(mlir::OpBuilder &builder, const TileID &tileId);

  SwitchboxOp getSwitchbox(mlir::OpBuilder &builder, int col, int row);

  ShimMuxOp getShimMux(mlir::OpBuilder &builder, int col);
};

// Get enum int value from WireBundle.
int getWireBundleAsInt(WireBundle bundle);

// The four cardinal-neighbor (tile, input-port) pairs a stream on `channel` can
// hop to when leaving a switchbox output -- North/East/South/West, in that
// order, matching the inter-tile adjacency the pathfinder graph is built on.
// The caller applies its own connectivity / getConnectingBundle check; this
// only builds the list, so the demand-capture, replay, and graph-build sites
// share one definition of the neighbor convention.
std::array<std::pair<TileID, Port>, 4> getCardinalNeighbors(TileID coords,
                                                            int channel);

// Attribute key under which AIEPinControlOverlay stashes a control flow's
// captured canonical route (one entry per source), decoded in the per-device
// pathfinder to pin the flow. The route rides the config's own IR, so parallel
// per-device passes never share mutable state.
constexpr llvm::StringLiteral kPinnedRouteAttr = "ctrl_pkt_pinned_route";

// Encode one source's captured route (its routed SwitchSettings) as a flat
// DenseI32ArrayAttr for annotation onto a control packet_flow op.
mlir::Attribute encodePinnedRoute(mlir::MLIRContext *ctx, TileID srcCoords,
                                  Port srcPort, const SwitchSettings &settings);

// Decode a kPinnedRouteAttr ArrayAttr back into (source, captured route) pairs.
std::vector<std::pair<PathEndPoint, SwitchSettings>>
decodePinnedRoutes(mlir::Attribute attr);

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
