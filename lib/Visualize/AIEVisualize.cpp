//===- AIEVisualize.cpp ---------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Visualize/AIEVisualize.h"

#include "ArrowRenderer.h"
#include "LayoutGeometry.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace xilinx;

namespace {

using xilinx::AIE::ArrowPath;
using xilinx::AIE::ArrowRenderer;
using xilinx::AIE::ArrowRequest;
using xilinx::AIE::createAvoidanceArrowRenderer;
using xilinx::AIE::createChannelArrowRenderer;
using xilinx::AIE::createStraightArrowRenderer;
using xilinx::AIE::LayoutGeometry;
using xilinx::AIE::Point;

struct PortNode {
  int col;
  int row;
  AIE::WireBundle bundle;
  int channel;
  std::optional<AIE::DMAChannelDir> dmaDirection;
  /// Which packet crosses a switchbox port. Two packet flows sharing one
  /// port are not the same place to a reader following either of them, and
  /// keeping them apart lets each contract on its own.
  std::optional<unsigned> packet;

  auto asTuple() const {
    return std::make_tuple(col, row, static_cast<int>(bundle), channel,
                           dmaDirection, packet);
  }
  bool operator<(const PortNode &other) const {
    return asTuple() < other.asTuple();
  }
  bool operator==(const PortNode &other) const {
    return asTuple() == other.asTuple();
  }
};

struct Segment {
  PortNode source;
  PortNode dest;

  bool operator<(const Segment &other) const {
    return std::tie(source, dest) < std::tie(other.source, other.dest);
  }
};

struct FlowRoute {
  unsigned id;
  std::optional<unsigned> packetID;
  std::optional<unsigned> packetMask;
  /// Every packet this flow carries. A mask lets one flow claim several.
  std::set<unsigned> carried;
  std::optional<bool> keepsPacketHeader;
  bool priorityRoute;
  std::vector<PortNode> points;
  /// Where the flow this came from is written, so a complaint about it can
  /// point at the flow rather than at the device.
  mlir::Location loc = mlir::UnknownLoc::get(nullptr);
};

struct ArbitrationRule {
  Segment segment;
  unsigned priority;
  unsigned mask;
  unsigned value;
  unsigned arbiter;
  unsigned msel;
  bool keepsPacketHeader;
  bool priorityRoute;
  bool controlOverlay;
  mlir::Location loc;
};

struct BufferInfo {
  AIE::BufferOp op;
  unsigned id;
  int col;
  int row;
  unsigned tileIndex;
};

using DMAChannelKey = std::tuple<int, int, AIE::DMAChannelDir, int>;
using DMAChannelBuffers = std::map<DMAChannelKey, std::vector<AIE::BufferOp>>;

struct RuntimeBufferUse {
  DMAChannelKey channel;
  unsigned argumentIndex;
  std::string argument;
};

struct RuntimeArgumentInfo {
  unsigned id;
  std::string name;
  std::vector<DMAChannelKey> channels;
  mlir::Location loc;
};

struct CallInfo {
  unsigned id;
  int col;
  int row;
  unsigned tileIndex;
  std::string callee;
  std::vector<AIE::BufferOp> buffers;
  mlir::Location loc;
};

using SourceLocations = std::map<std::string, std::set<unsigned>>;

static SourceLocations sourceLocations(mlir::Location location) {
  SourceLocations result;
  location->walk([&](mlir::Location nested) {
    if (auto file = llvm::dyn_cast<mlir::FileLineColLoc>(nested))
      result[file.getFilename().str()].insert(file.getLine());
    return mlir::WalkResult::advance();
  });
  return result;
}

static void absorbSourceLocations(SourceLocations &dest,
                                  const SourceLocations &source) {
  for (const auto &[file, lines] : source)
    dest[file].insert(lines.begin(), lines.end());
}

enum class SemanticNodeKind {
  Tile,
  /// A switchbox port: somewhere a flow passes through.
  Port,
  /// A DMA channel: where a flow enters or leaves a tile.
  DMA,
  Buffer,
  RuntimeArgument,
  Call
};

struct SemanticNode {
  SemanticNodeKind kind;
  std::optional<std::pair<int, int>> tile;
  std::optional<PortNode> port;
  SourceLocations source;
  std::string label;
};

/// Which two things an edge joins. That is its whole identity: what it
/// carries is an attribute, so that contraction can merge two edges by
/// merging what they carry rather than by deciding they are the same.
struct EdgeKey {
  std::string source;
  std::string dest;

  bool operator<(const EdgeKey &other) const {
    return std::tie(source, dest) < std::tie(other.source, other.dest);
  }
};

/// What an edge carries. Contraction unions these, so an edge that stands
/// for a chain says everything the chain said.
struct EdgeAttributes {
  /// The routes that took this edge.
  std::set<unsigned> routes;
  /// How many equivalent paths this edge stands for.
  unsigned count = 1;
  SourceLocations source;

  void absorb(const EdgeAttributes &other) {
    routes.insert(other.routes.begin(), other.routes.end());
    absorbSourceLocations(source, other.source);
  }
};

using SemanticEdges = std::map<EdgeKey, EdgeAttributes>;

struct SemanticGraph {
  std::map<std::string, SemanticNode> nodes;
  SemanticEdges edges;

  void addNode(StringRef id, SemanticNodeKind kind,
               std::optional<std::pair<int, int>> tile = std::nullopt,
               std::optional<PortNode> port = std::nullopt,
               SourceLocations source = {}, StringRef label = {}) {
    auto [entry, inserted] = nodes.try_emplace(
        id.str(), SemanticNode{kind, tile, port, source, label.str()});
    if (!inserted) {
      absorbSourceLocations(entry->second.source, source);
      if (entry->second.label.empty())
        entry->second.label = label.str();
    }
  }

  void addEdge(StringRef source, StringRef dest,
               std::optional<unsigned> route = std::nullopt,
               SourceLocations locations = {}) {
    EdgeAttributes &attributes = edges[{source.str(), dest.str()}];
    if (route)
      attributes.routes.insert(*route);
    absorbSourceLocations(attributes.source, locations);
  }

  bool hasIncoming(StringRef node) const {
    return llvm::any_of(
        edges, [&](const auto &entry) { return entry.first.dest == node; });
  }

  bool hasOutgoing(StringRef node) const {
    return llvm::any_of(
        edges, [&](const auto &entry) { return entry.first.source == node; });
  }

  bool hasEdge(StringRef source, StringRef dest) const {
    return edges.count({source.str(), dest.str()}) != 0;
  }

  /// Splices out a hidden node that only passes one edge through. Fanout and
  /// fan-in remain visible unless the balanced-region rewrite below matches.
  SemanticEdges contract(const std::set<SemanticNodeKind> &hiddenKinds,
                         bool contractBalancedFanout) const {
    SemanticEdges result = edges;
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto &[id, node] : nodes) {
        if (!hiddenKinds.count(node.kind))
          continue;
        SmallVector<EdgeKey> incoming;
        SmallVector<EdgeKey> outgoing;
        for (const auto &[key, attributes] : result) {
          if (key.dest == id)
            incoming.push_back(key);
          if (key.source == id)
            outgoing.push_back(key);
        }
        if (incoming.size() != 1 || outgoing.size() != 1)
          continue;

        SmallVector<std::pair<EdgeKey, EdgeAttributes>> spliced;
        for (const EdgeKey &before : incoming) {
          for (const EdgeKey &after : outgoing) {
            if (before.source == after.dest)
              continue;
            EdgeAttributes merged = result.at(before);
            merged.absorb(result.at(after));
            absorbSourceLocations(merged.source, node.source);
            merged.count *= result.at(after).count;
            spliced.push_back({{before.source, after.dest}, std::move(merged)});
          }
        }
        for (const EdgeKey &key : incoming)
          result.erase(key);
        for (const EdgeKey &key : outgoing)
          result.erase(key);
        for (auto &[key, attributes] : spliced) {
          auto [entry, inserted] = result.try_emplace(key, attributes);
          if (!inserted) {
            entry->second.absorb(attributes);
            entry->second.count += attributes.count;
          }
        }
        changed = true;
        break;
      }
      if (changed || !contractBalancedFanout)
        continue;

      for (const auto &[source, sourceNode] : nodes) {
        SmallVector<EdgeKey> outgoing;
        for (const auto &[key, attributes] : result)
          if (key.source == source)
            outgoing.push_back(key);
        if (outgoing.size() < 2)
          continue;

        std::optional<std::string> sink;
        SmallVector<std::string> sources;
        SmallVector<EdgeKey> secondLegs;
        bool balanced = true;
        for (const EdgeKey &first : outgoing) {
          if (first.dest == source) {
            balanced = false;
            break;
          }
          SmallVector<EdgeKey> incoming, leaving;
          for (const auto &[key, attributes] : result) {
            if (key.dest == first.dest)
              incoming.push_back(key);
            if (key.source == first.dest)
              leaving.push_back(key);
          }
          if (incoming.empty() || leaving.size() != 1) {
            balanced = false;
            break;
          }
          SmallVector<std::string> branchSources;
          llvm::transform(incoming, std::back_inserter(branchSources),
                          [](const EdgeKey &key) { return key.source; });
          if (sources.empty())
            sources = branchSources;
          if (sources != branchSources ||
              !llvm::is_contained(sources, source)) {
            balanced = false;
            break;
          }
          if (!sink)
            sink = leaving.front().dest;
          if (*sink != leaving.front().dest || *sink == source) {
            balanced = false;
            break;
          }
          secondLegs.push_back(leaving.front());
        }
        if (!balanced || !sink)
          continue;

        const std::string sinkID = *sink;
        unsigned sinkDegree = llvm::count_if(result, [&](const auto &entry) {
          return entry.first.dest == sinkID;
        });
        if (sinkDegree != outgoing.size())
          continue;

        for (StringRef branchSource : sources) {
          SmallVector<std::string> destinations;
          for (const auto &[key, attributes] : result)
            if (key.source == branchSource)
              destinations.push_back(key.dest);
          SmallVector<std::string> branches;
          llvm::transform(outgoing, std::back_inserter(branches),
                          [](const EdgeKey &key) { return key.dest; });
          llvm::sort(destinations);
          llvm::sort(branches);
          if (destinations != branches) {
            balanced = false;
            break;
          }
        }
        if (!balanced)
          continue;

        std::map<std::string, EdgeAttributes> merged;
        for (StringRef branchSource : sources) {
          EdgeAttributes &attributes = merged[branchSource.str()];
          attributes.count = 0;
          for (const EdgeKey &second : secondLegs) {
            EdgeKey first{branchSource.str(), second.source};
            EdgeAttributes path = result.at(first);
            path.absorb(result.at(second));
            const SourceLocations &branchSourceLocations =
                nodes.at(second.source).source;
            absorbSourceLocations(path.source, branchSourceLocations);
            path.count *= result.at(second).count;
            attributes.absorb(path);
            attributes.count += path.count;
          }
        }
        for (StringRef branchSource : sources)
          for (const EdgeKey &second : secondLegs)
            result.erase({branchSource.str(), second.source});
        for (const EdgeKey &key : secondLegs)
          result.erase(key);
        for (auto &[branchSource, attributes] : merged)
          result[{branchSource, sinkID}] = std::move(attributes);
        changed = true;
        break;
      }
    }
    return result;
  }
};

/// Where what crosses an edge came from, and where it can end up. An origin
/// is a node nothing feeds; a fate is a node that feeds nothing. Both are
/// read off the edges alone, so it makes no difference whether an edge came
/// from a switchbox hop, a buffer access or a kernel call.
struct Provenance {
  /// Node to the interned set of origins reaching it, and of fates it reaches.
  std::map<std::string, unsigned> sourceGroup;
  std::map<std::string, unsigned> destGroup;
  /// Members of each interned set, in node-ID order.
  std::vector<std::vector<std::string>> sourceSets;
  std::vector<std::vector<std::string>> destSets;
};

struct TransitiveMembership {
  std::map<std::string, std::set<std::string>> origins;
  std::map<std::string, std::set<std::string>> fates;
};

static TransitiveMembership
computeTransitiveMembership(const SemanticGraph &graph,
                            const SemanticEdges &edges) {
  TransitiveMembership membership;
  for (const auto &[key, attributes] : edges) {
    membership.origins[key.source].insert(key.source);
    membership.origins[key.dest].insert(key.dest);
    membership.fates[key.source].insert(key.source);
    membership.fates[key.dest].insert(key.dest);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &[key, attributes] : edges) {
      auto isAnchor = [&](const std::string &id) {
        SemanticNodeKind kind = graph.nodes.at(id).kind;
        return kind == SemanticNodeKind::Tile || kind == SemanticNodeKind::Call;
      };
      if (isAnchor(key.source)) {
        changed |= membership.origins[key.dest].insert(key.source).second;
      } else {
        for (const std::string &origin : membership.origins[key.source])
          changed |= membership.origins[key.dest].insert(origin).second;
      }
      if (isAnchor(key.dest)) {
        changed |= membership.fates[key.source].insert(key.dest).second;
      } else {
        for (const std::string &fate : membership.fates[key.dest])
          changed |= membership.fates[key.source].insert(fate).second;
      }
    }
  }
  return membership;
}

/// Accumulates boundary nodes along the edges until nothing changes. Splicing
/// a node that only passes things through cannot change what reaches an edge,
/// so this answers the same before and after contraction.
static std::map<std::string, std::set<std::string>>
accumulateBoundary(const SemanticGraph &graph, const SemanticEdges &edges,
                   bool forward) {
  std::map<std::string, std::vector<std::string>> feeders;
  std::set<std::string> fed, feeding, nodes;
  for (const auto &[key, attributes] : edges) {
    const std::string &from = forward ? key.source : key.dest;
    const std::string &to = forward ? key.dest : key.source;
    feeders[to].push_back(from);
    fed.insert(to);
    feeding.insert(from);
    nodes.insert(from);
    nodes.insert(to);
  }

  // Tiles and calls join unrelated inputs and outputs. Propagating through
  // either one would assign every input origin to every output.
  auto anchors = [&](const std::string &node) {
    auto found = graph.nodes.find(node);
    return found != graph.nodes.end() &&
           (found->second.kind == SemanticNodeKind::Tile ||
            found->second.kind == SemanticNodeKind::Call);
  };

  std::map<std::string, std::set<std::string>> reach;
  for (const std::string &node : nodes)
    if (feeding.count(node) && (anchors(node) || !fed.count(node)))
      reach[node] = {node};

  bool changed = true;
  while (changed) {
    changed = false;
    for (const std::string &node : nodes) {
      if (!fed.count(node) || anchors(node))
        continue;
      std::set<std::string> merged;
      for (const std::string &from : feeders[node]) {
        auto found = reach.find(from);
        if (found != reach.end())
          merged.insert(found->second.begin(), found->second.end());
      }
      std::set<std::string> &current = reach[node];
      if (merged != current) {
        current = std::move(merged);
        changed = true;
      }
    }
  }
  return reach;
}

static Provenance computeProvenance(const SemanticGraph &graph,
                                    const SemanticEdges &edges) {
  std::map<std::string, std::set<std::string>> origins =
      accumulateBoundary(graph, edges, /*forward=*/true);
  std::map<std::string, std::set<std::string>> fates =
      accumulateBoundary(graph, edges, /*forward=*/false);

  Provenance provenance;
  // Interning on the sorted members keeps IDs stable across runs.
  auto intern = [](const std::set<std::string> &members,
                   std::vector<std::vector<std::string>> &sets,
                   std::map<std::vector<std::string>, unsigned> &ids) {
    std::vector<std::string> key(members.begin(), members.end());
    auto [entry, inserted] = ids.try_emplace(key, sets.size());
    if (inserted)
      sets.push_back(std::move(key));
    return entry->second;
  };
  std::map<std::vector<std::string>, unsigned> sourceIDs, destIDs;
  for (const auto &[id, node] : graph.nodes) {
    auto origin = origins.find(id);
    auto fate = fates.find(id);
    provenance.sourceGroup[id] = intern(
        origin == origins.end() ? std::set<std::string>{} : origin->second,
        provenance.sourceSets, sourceIDs);
    provenance.destGroup[id] =
        intern(fate == fates.end() ? std::set<std::string>{} : fate->second,
               provenance.destSets, destIDs);
  }
  return provenance;
}

static FailureOr<PortNode>
getPortNode(mlir::Value tile, AIE::WireBundle bundle, int channel,
            Operation *owner,
            std::optional<AIE::DMAChannelDir> dmaDirection = std::nullopt) {
  auto tileOp = dyn_cast_or_null<AIE::TileLike>(tile.getDefiningOp());
  if (!tileOp)
    return owner->emitOpError("route endpoint is not a tile-like operation");
  std::optional<int> col = tileOp.tryGetCol();
  std::optional<int> row = tileOp.tryGetRow();
  if (!col || !row)
    return owner->emitOpError("route endpoint has unresolved coordinates");
  if (bundle != AIE::WireBundle::DMA)
    dmaDirection = std::nullopt;
  return PortNode{*col, *row, bundle, channel, dmaDirection};
}

template <typename FlowTy>
static LogicalResult appendVias(FlowTy flow, std::vector<PortNode> &points) {
  if (flow.getVias().empty())
    return success();
  ArrayRef<int32_t> ingressBundles =
      flow.getViaIngressBundlesAttr().asArrayRef();
  ArrayRef<int32_t> ingressChannels =
      flow.getViaIngressChannelsAttr().asArrayRef();
  ArrayRef<int32_t> egressBundles = flow.getViaEgressBundlesAttr().asArrayRef();
  ArrayRef<int32_t> egressChannels =
      flow.getViaEgressChannelsAttr().asArrayRef();
  for (auto [index, tile] : llvm::enumerate(flow.getVias())) {
    std::optional<PortNode> ingress =
        getPortNode(tile, static_cast<AIE::WireBundle>(ingressBundles[index]),
                    ingressChannels[index], flow, AIE::DMAChannelDir::MM2S);
    std::optional<PortNode> egress =
        getPortNode(tile, static_cast<AIE::WireBundle>(egressBundles[index]),
                    egressChannels[index], flow, AIE::DMAChannelDir::S2MM);
    if (!ingress)
      return failure();
    if (!egress)
      return failure();
    points.push_back(*ingress);
    points.push_back(*egress);
  }
  return success();
}

static FailureOr<std::vector<FlowRoute>> collectRoutes(AIE::DeviceOp device) {
  std::vector<FlowRoute> routes;
  for (Operation &operation : *device.getBody()) {
    if (auto flow = dyn_cast<AIE::FlowOp>(operation)) {
      FlowRoute route{static_cast<unsigned>(routes.size()),
                      std::nullopt,
                      std::nullopt,
                      {},
                      std::nullopt,
                      false,
                      {},
                      flow.getLoc()};
      std::optional<PortNode> source =
          getPortNode(flow.getSource(), flow.getSourceBundle(),
                      flow.getSourceChannel(), flow, AIE::DMAChannelDir::MM2S);
      std::optional<PortNode> dest =
          getPortNode(flow.getDest(), flow.getDestBundle(),
                      flow.getDestChannel(), flow, AIE::DMAChannelDir::S2MM);
      if (!source)
        return failure();
      if (!dest)
        return failure();
      route.points.push_back(*source);
      if (failed(appendVias(flow, route.points)))
        return failure();
      route.points.push_back(*dest);
      routes.push_back(std::move(route));
      continue;
    }
    auto packetFlow = dyn_cast<AIE::PacketFlowOp>(operation);
    if (!packetFlow)
      continue;
    auto sources = packetFlow.getOps<AIE::PacketSourceOp>();
    auto dests = packetFlow.getOps<AIE::PacketDestOp>();
    if (!llvm::hasSingleElement(sources) || !llvm::hasSingleElement(dests))
      return packetFlow.emitOpError(
          "requires exactly one source and destination per routed section");
    AIE::PacketSourceOp sourceOp = *sources.begin();
    AIE::PacketDestOp destOp = *dests.begin();
    std::optional<unsigned> packetMask;
    if (std::optional<int8_t> mask = packetFlow.getMask())
      packetMask = static_cast<uint8_t>(*mask);
    FlowRoute route{static_cast<unsigned>(routes.size()),
                    static_cast<uint8_t>(packetFlow.getID()),
                    packetMask,
                    {},
                    packetFlow.getKeepPktHeader(),
                    packetFlow.getPriorityRoute().value_or(false),
                    {},
                    packetFlow.getLoc()};
    std::optional<PortNode> source = getPortNode(
        sourceOp.getTile(), sourceOp.getBundle(), sourceOp.getChannel(),
        packetFlow, AIE::DMAChannelDir::MM2S);
    std::optional<PortNode> dest =
        getPortNode(destOp.getTile(), destOp.getBundle(), destOp.getChannel(),
                    packetFlow, AIE::DMAChannelDir::S2MM);
    if (!source)
      return failure();
    if (!dest)
      return failure();
    route.points.push_back(*source);
    if (failed(appendVias(packetFlow, route.points)))
      return failure();
    route.points.push_back(*dest);
    routes.push_back(std::move(route));
  }
  return routes;
}

static std::vector<ArbitrationRule>
collectArbitrationRules(AIE::DeviceOp device) {
  AIE::AIEDialect::IsCtrlPktOverlayAttrHelper overlay(device.getContext());
  auto marksOverlay = [&](Operation *op) {
    // Older IR spells the attribute without the dialect prefix.
    return overlay.isAttrPresent(op) || op->hasAttr("is_ctrl_pkt_overlay");
  };
  std::vector<ArbitrationRule> rules;
  for (AIE::SwitchboxOp switchbox : device.getOps<AIE::SwitchboxOp>()) {
    AIE::TileOp tile = switchbox.getTileOp();
    int col = tile.colIndex();
    int row = tile.rowIndex();
    for (AIE::PacketRulesOp packetRules :
         switchbox.getOps<AIE::PacketRulesOp>()) {
      PortNode source{col, row, packetRules.getSourceBundle(),
                      packetRules.getSourceChannel(),
                      packetRules.getSourceBundle() == AIE::WireBundle::DMA
                          ? std::optional(AIE::DMAChannelDir::MM2S)
                          : std::nullopt};
      unsigned priority = 0;
      for (AIE::PacketRuleOp rule : packetRules.getOps<AIE::PacketRuleOp>()) {
        auto amsel = rule.getAmsel().getDefiningOp<AIE::AMSelOp>();
        if (!amsel) {
          ++priority;
          continue;
        }
        for (AIE::MasterSetOp master : switchbox.getOps<AIE::MasterSetOp>()) {
          if (!llvm::is_contained(master.getAmsels(), amsel.getResult()))
            continue;
          PortNode dest{col, row, master.getDestBundle(),
                        master.getDestChannel(),
                        master.getDestBundle() == AIE::WireBundle::DMA
                            ? std::optional(AIE::DMAChannelDir::S2MM)
                            : std::nullopt};
          rules.push_back({{source, dest},
                           priority,
                           static_cast<uint8_t>(rule.getMask()),
                           static_cast<uint8_t>(rule.getValue()),
                           static_cast<uint8_t>(amsel.getArbiterID()),
                           static_cast<uint8_t>(amsel.getMsel()),
                           master.keepsPktHeader(),
                           rule->hasAttr("aie.priority_route"),
                           marksOverlay(rule) || marksOverlay(packetRules) ||
                               marksOverlay(master),
                           rule.getLoc()});
        }
        ++priority;
      }
    }
  }
  return rules;
}

static FailureOr<std::vector<BufferInfo>> collectBuffers(AIE::DeviceOp device) {
  std::vector<BufferInfo> buffers;
  std::map<std::pair<int, int>, unsigned> tileCounts;
  for (AIE::BufferOp buffer : device.getOps<AIE::BufferOp>()) {
    AIE::TileOp tile = buffer.getTileOp();
    if (!tile)
      return buffer.emitOpError("buffer owner is not a resolved tile");
    std::pair<int, int> coordinate{tile.colIndex(), tile.rowIndex()};
    buffers.push_back({buffer, static_cast<unsigned>(buffers.size()),
                       coordinate.first, coordinate.second,
                       tileCounts[coordinate]++});
  }
  return buffers;
}

static DMAChannelBuffers collectDMAChannelBuffers(AIE::DeviceOp device) {
  DMAChannelBuffers channels;
  for (auto program : device.getOps<AIE::DmaBody>()) {
    auto tile =
        dyn_cast_or_null<AIE::TileLike>(program.getTile().getDefiningOp());
    if (!tile)
      continue;
    std::optional<int> col = tile.tryGetCol();
    std::optional<int> row = tile.tryGetRow();
    if (!col || !row)
      continue;
    for (Block &block : program.getDmaBody()) {
      for (AIE::DMAStartOp start : block.getOps<AIE::DMAStartOp>()) {
        if (start.getEndpoint())
          continue;
        DMAChannelKey key{*col, *row, start.getChannelDir(),
                          start.getChannelIndex()};
        std::vector<AIE::BufferOp> &buffers = channels[key];
        SmallVector<Block *> worklist{start.getDest()};
        llvm::SmallPtrSet<Block *, 8> visited;
        while (!worklist.empty()) {
          Block *current = worklist.pop_back_val();
          if (!current || !visited.insert(current).second)
            continue;
          for (AIE::DMABDOp descriptor : current->getOps<AIE::DMABDOp>()) {
            auto buffer = descriptor.getBuffer().getDefiningOp<AIE::BufferOp>();
            if (buffer && !llvm::is_contained(buffers, buffer))
              buffers.push_back(buffer);
          }
          Operation *terminator = current->getTerminator();
          if (isa<AIE::DMAStartOp>(terminator))
            continue;
          llvm::append_range(worklist, terminator->getSuccessors());
        }
      }
    }
  }
  return channels;
}

using DMAChannelBytes = std::map<DMAChannelKey, uint64_t>;

/// Bytes one pass over each channel's buffer-descriptor chain moves. A chain
/// repeats for the life of the design, so this is a per-iteration rate, not a
/// total; it is the only volume the IR states without a trip count.
static DMAChannelBytes collectDMAChannelBytes(AIE::DeviceOp device) {
  DMAChannelBytes channels;
  for (auto program : device.getOps<AIE::DmaBody>()) {
    auto tile =
        dyn_cast_or_null<AIE::TileLike>(program.getTile().getDefiningOp());
    if (!tile)
      continue;
    std::optional<int> col = tile.tryGetCol();
    std::optional<int> row = tile.tryGetRow();
    if (!col || !row)
      continue;
    for (Block &block : program.getDmaBody()) {
      for (AIE::DMAStartOp start : block.getOps<AIE::DMAStartOp>()) {
        if (start.getEndpoint())
          continue;
        DMAChannelKey key{*col, *row, start.getChannelDir(),
                          start.getChannelIndex()};
        uint64_t &bytes = channels[key];
        SmallVector<Block *> worklist{start.getDest()};
        llvm::SmallPtrSet<Block *, 8> visited;
        while (!worklist.empty()) {
          Block *current = worklist.pop_back_val();
          if (!current || !visited.insert(current).second)
            continue;
          for (AIE::DMABDOp descriptor : current->getOps<AIE::DMABDOp>())
            bytes += descriptor.getLenInBytes();
          Operation *terminator = current->getTerminator();
          if (isa<AIE::DMAStartOp>(terminator))
            continue;
          llvm::append_range(worklist, terminator->getSuccessors());
        }
      }
    }
  }
  return channels;
}

/// Maps bytes to stroke width on a log scale spanning the design's own range.
/// Transfer sizes here span several orders of magnitude, so a linear map would
/// collapse everything below the largest flow into one hairline.
struct EdgeWeightScale {
  uint64_t lightest = 0;
  uint64_t heaviest = 0;
  bool enabled = false;

  double narrowest = 0.5;
  double widest = 7.0;

  static EdgeWeightScale build(ArrayRef<uint64_t> groupBytes, double narrowest,
                               double widest) {
    EdgeWeightScale scale;
    scale.narrowest = narrowest;
    scale.widest = widest;
    for (uint64_t bytes : groupBytes) {
      if (bytes == 0)
        continue;
      if (scale.lightest == 0 || bytes < scale.lightest)
        scale.lightest = bytes;
      scale.heaviest = std::max(scale.heaviest, bytes);
    }
    scale.enabled = scale.heaviest > 0;
    return scale;
  }

  double width(uint64_t bytes) const {
    if (!enabled || bytes == 0)
      return narrowest;
    if (heaviest == lightest)
      return widest;
    double position = (std::log2(static_cast<double>(bytes)) -
                       std::log2(static_cast<double>(lightest))) /
                      (std::log2(static_cast<double>(heaviest)) -
                       std::log2(static_cast<double>(lightest)));
    return narrowest + position * (widest - narrowest);
  }
};

static std::string formatBytes(uint64_t bytes) {
  static constexpr const char *units[] = {"B", "KiB", "MiB", "GiB"};
  double value = static_cast<double>(bytes);
  unsigned unit = 0;
  while (value >= 1024.0 && unit + 1 < std::size(units)) {
    value /= 1024.0;
    ++unit;
  }
  SmallString<16> text;
  llvm::raw_svector_ostream stream(text);
  stream << llvm::format(value < 10.0 && unit > 0 ? "%.1f" : "%.0f", value)
         << ' ' << units[unit];
  return std::string(text);
}

static std::vector<DMAChannelKey> getAllocationChannel(AIE::DeviceOp device,
                                                       SymbolRefAttr symbol) {
  Operation *definition =
      SymbolTable::lookupSymbolIn(device, symbol.getRootReference());
  if (auto allocation =
          dyn_cast_or_null<AIE::ShimDMAAllocationOp>(definition)) {
    AIE::TileOp tile = allocation.getTileOp();
    if (!tile)
      return {};
    return {{tile.colIndex(), tile.rowIndex(), allocation.getChannelDir(),
             allocation.getChannelIndex()}};
  }
  auto endpoint = dyn_cast_or_null<AIE::RouteEndpointOp>(definition);
  if (!endpoint || endpoint.getBundle() != AIE::WireBundle::DMA)
    return {};
  std::optional<uint8_t> channel = endpoint.getChannelIndex();
  if (!channel)
    return {};
  auto tile =
      dyn_cast_or_null<AIE::TileLike>(endpoint.getTile().getDefiningOp());
  if (!tile)
    return {};
  std::optional<int> col = tile.tryGetCol();
  std::optional<int> row = tile.tryGetRow();
  if (!col || !row)
    return {};
  return {{*col, *row, AIE::DMAChannelDir::MM2S, *channel},
          {*col, *row, AIE::DMAChannelDir::S2MM, *channel}};
}

static std::optional<std::pair<unsigned, std::string>>
getRuntimeArgument(AIE::RuntimeSequenceOp sequence, mlir::Value value) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner()->getParentOp() != sequence)
    return std::nullopt;
  unsigned index = argument.getArgNumber();
  if (auto names = sequence->getAttrOfType<ArrayAttr>("runtime_arg_names")) {
    if (index < names.size())
      return std::make_pair(index,
                            cast<StringAttr>(names[index]).getValue().str());
  }
  return std::make_pair(index, "arg" + std::to_string(index));
}

static std::vector<RuntimeBufferUse>
collectRuntimeBufferUses(AIE::DeviceOp device,
                         AIE::RuntimeSequenceOp sequence) {
  std::vector<RuntimeBufferUse> uses;
  if (!sequence)
    return uses;
  auto addUses = [&](ArrayRef<DMAChannelKey> channels, mlir::Value value) {
    std::optional<std::pair<unsigned, std::string>> argument =
        getRuntimeArgument(sequence, value);
    if (!argument)
      return;
    for (const DMAChannelKey &channel : channels) {
      RuntimeBufferUse use{channel, argument->first, argument->second};
      if (!llvm::any_of(uses, [&](const RuntimeBufferUse &existing) {
            return existing.channel == use.channel &&
                   existing.argumentIndex == use.argumentIndex;
          }))
        uses.push_back(std::move(use));
    }
  };
  sequence.walk([&](Operation *operation) {
    if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(operation)) {
      AIE::TileOp tile = task.getTileOp();
      if (!tile)
        return;
      DMAChannelKey channel{tile.colIndex(), tile.rowIndex(),
                            task.getDirection(),
                            static_cast<int>(task.getChannel())};
      task.getBody().walk([&](AIE::DMABDOp descriptor) {
        addUses(ArrayRef(channel), descriptor.getBuffer());
      });
      return;
    }
    if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(operation)) {
      std::vector<DMAChannelKey> channels =
          getAllocationChannel(device, task.getAlloc());
      task.getBody().walk([&](AIE::DMABDOp descriptor) {
        addUses(channels, descriptor.getBuffer());
      });
      return;
    }
    if (auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(operation)) {
      addUses(getAllocationChannel(device, memcpy.getMetadata()),
              memcpy.getMemref());
    }
  });
  return uses;
}

static std::vector<CallInfo> collectCalls(AIE::DeviceOp device) {
  std::vector<CallInfo> calls;
  std::map<std::pair<int, int>, unsigned> tileCounts;
  std::map<Operation *, unsigned> bufferOrder;
  for (auto [index, buffer] : llvm::enumerate(device.getOps<AIE::BufferOp>()))
    bufferOrder[buffer] = index;

  auto traceBuffers = [&](mlir::Value root) {
    std::vector<AIE::BufferOp> buffers;
    SmallVector<mlir::Value> worklist{root};
    llvm::SmallDenseSet<mlir::Value, 8> visited;
    while (!worklist.empty()) {
      mlir::Value value = worklist.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      if (auto buffer = value.getDefiningOp<AIE::BufferOp>()) {
        buffers.push_back(buffer);
        continue;
      }
      if (Operation *definition = value.getDefiningOp()) {
        if (auto view = dyn_cast<ViewLikeOpInterface>(definition))
          worklist.push_back(view.getViewSource());
        continue;
      }
      auto argument = dyn_cast<BlockArgument>(value);
      if (!argument)
        continue;
      Block *block = argument.getOwner();
      for (Block *predecessor : block->getPredecessors()) {
        Operation *terminator = predecessor->getTerminator();
        auto branch = dyn_cast<BranchOpInterface>(terminator);
        if (!branch)
          continue;
        for (auto [index, successor] :
             llvm::enumerate(terminator->getSuccessors())) {
          if (successor != block)
            continue;
          SuccessorOperands operands = branch.getSuccessorOperands(index);
          if (argument.getArgNumber() >= operands.size())
            continue;
          if (mlir::Value incoming = operands[argument.getArgNumber()])
            worklist.push_back(incoming);
        }
      }
    }
    llvm::sort(buffers, [&](AIE::BufferOp first, AIE::BufferOp second) {
      return bufferOrder[first] < bufferOrder[second];
    });
    buffers.erase(llvm::unique(buffers), buffers.end());
    return buffers;
  };

  for (AIE::CoreOp core : device.getOps<AIE::CoreOp>()) {
    AIE::TileOp tile = core.getTileOp();
    if (!tile)
      continue;
    core.walk([&](func::CallOp call) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || !callee.isExternal())
        return;
      std::vector<AIE::BufferOp> buffers;
      for (mlir::Value operand : call.getOperands()) {
        for (AIE::BufferOp buffer : traceBuffers(operand))
          if (!llvm::is_contained(buffers, buffer))
            buffers.push_back(buffer);
      }
      std::pair<int, int> coordinate{tile.colIndex(), tile.rowIndex()};
      calls.push_back({static_cast<unsigned>(calls.size()), coordinate.first,
                       coordinate.second, tileCounts[coordinate]++,
                       call.getCallee().str(), std::move(buffers),
                       call.getLoc()});
    });
  }
  return calls;
}

/// Drops the content-hash prefix kernel symbols carry. The hash identifies a
/// build variant, not a function, so two tiles running the same kernel read as
/// unrelated while it is still attached. Requires a digit so that an
/// all-letters word that happens to be hex ("decade") survives.
static StringRef stripKernelHash(StringRef callee) {
  size_t separator = callee.find('_');
  if (separator == StringRef::npos || separator < 8 || separator > 16)
    return callee;
  StringRef prefix = callee.take_front(separator);
  if (!llvm::all_of(prefix, isHexDigit) || !llvm::any_of(prefix, isDigit))
    return callee;
  return callee.drop_front(separator + 1);
}

/// What a tile does: each kernel it runs named once, with how many times it
/// runs. Tiles that share a role are doing the same job on different data.
struct TileRole {
  std::vector<std::pair<std::string, unsigned>> kernels;

  /// At most `limit` kernels, then a count of the rest. One tile with a long
  /// kernel list would otherwise set the height of every tile in its row.
  std::string text(unsigned limit) const {
    std::string result;
    for (const auto &[kernel, count] : ArrayRef(kernels).take_front(limit)) {
      if (!result.empty())
        result += "\\n";
      result += kernel;
      if (count > 1)
        result += " x" + std::to_string(count);
    }
    if (kernels.size() > limit)
      result += "\\n+" + std::to_string(kernels.size() - limit) + " more";
    return result;
  }

  unsigned lines(unsigned limit) const {
    return std::min<unsigned>(kernels.size(), limit) +
           (kernels.size() > limit ? 1 : 0);
  }

  bool operator<(const TileRole &other) const {
    return kernels < other.kernels;
  }
};

static std::map<std::pair<int, int>, TileRole>
collectTileRoles(ArrayRef<CallInfo> calls) {
  std::map<std::pair<int, int>, TileRole> roles;
  for (const CallInfo &call : calls) {
    TileRole &role = roles[{call.col, call.row}];
    std::string kernel = stripKernelHash(call.callee).str();
    auto existing = llvm::find_if(
        role.kernels, [&](const auto &entry) { return entry.first == kernel; });
    if (existing == role.kernels.end())
      role.kernels.emplace_back(kernel, 1);
    else
      ++existing->second;
  }
  return roles;
}

/// A tile that shares its role with others, and which of those groups it is in.
struct RoleClass {
  std::string label;
  unsigned id;
};

/// Groups tiles that share a role, so a reader can see that four tiles are the
/// same worker without comparing four lists of kernel names. Roles held by a
/// single tile get no class: there is no repetition to point out.
static std::map<std::pair<int, int>, RoleClass>
collectRoleClasses(const std::map<std::pair<int, int>, TileRole> &roles) {
  std::map<TileRole, std::vector<std::pair<int, int>>> classes;
  for (const auto &[coordinate, role] : roles)
    classes[role].push_back(coordinate);
  std::map<std::pair<int, int>, RoleClass> assigned;
  unsigned next = 0;
  for (const auto &[role, members] : classes) {
    if (members.size() < 2)
      continue;
    std::string name(1, static_cast<char>('A' + next % 26));
    if (next >= 26)
      name += std::to_string(next / 26);
    for (const std::pair<int, int> &coordinate : members)
      assigned[coordinate] = {name + " x" + std::to_string(members.size()),
                              next};
    ++next;
  }
  return assigned;
}

/// Fill for a repeated role. Tiles doing the same job read as a group before
/// any label is read. The tints stay pale enough to sit under black text and
/// under the flow edges drawn across them, and far enough apart in hue to
/// survive being seen one tile at a time rather than side by side.
/// Every color the view draws in, so that a scheme is one table rather than
/// a hunt through the emitter.
struct Palette {
  StringRef background;
  StringRef tileEdge;
  StringRef tileText;
  StringRef coreTile, memTile, shimTile, otherTile;
  StringRef boxFill, boxEdge;
  StringRef callFill, callEdge;
  StringRef argumentFill, argumentEdge;
  StringRef ruleFill, ruleEdge, ruleText;
  StringRef runtimeLink, callLink, quietEdge;
  StringRef legendText, legendRule;
  /// Fills for tiles that share a role, and colors for the flows.
  ArrayRef<const char *> roles;
  ArrayRef<const char *> flows;
};

/// Seaborn's mako, sampled away from both ends: the darkest steps vanish into
/// a black page and the lightest wash out against it.
static constexpr const char *kMakoFlows[] = {
    "#3e4c8a", "#38699e", "#3487a6", "#35a1ab", "#43bbad",
    "#6cd3ad", "#9fe0b0", "#4a3f7a", "#2e7ba3", "#57c7ae"};
/// The same ramp held back to tints a label can be read on.
static constexpr const char *kMakoRoles[] = {
    "#1d2b45", "#163a4a", "#17484c", "#1c5553", "#24604f",
    "#38393f", "#2a2f55", "#113f3f", "#214a63", "#2f2f33"};

static constexpr const char *kLightFlows[] = {
    "#d73027", "#4575b4", "#1a9850", "#984ea3", "#ff7f00",
    "#00a6a6", "#e7298a", "#6a3d9a", "#a6761d", "#1f78b4"};
static constexpr const char *kLightRoles[] = {
    "#cfe3f7", "#ffdfd3", "#d6efd6", "#ebdcf5", "#ffeec2",
    "#ccebe8", "#f6d9e6", "#e3e6c9", "#dcdcf0", "#e6e6e6"};

static const Palette &paletteFor(AIE::ColorScheme scheme) {
  static const Palette light{
      "white",   "#b8b8b8", "#555555", "#eef7ee",   "#fff2df",  "#e8f1fb",
      "#f3edf8", "#ffffff", "#666666", "#f7f7f7",   "#555555",  "#fff7d6",
      "#8f7c32", "#fffdf4", "#9a8b52", "#554b2c",   "#39789b",  "#666666",
      "#c2c2c2", "#333333", "#555555", kLightRoles, kLightFlows};
  // On black the tiles step just off the background rather than drawing a
  // line of their own, and every label is light.
  static const Palette mako{
      "#06080c", "#1b2430", "#dfe8ee", "#101722",  "#141b16", "#0d1620",
      "#141320", "#161d28", "#46586b", "#1b2430",  "#5a6e82", "#1a1f2e",
      "#6f7fa6", "#141b26", "#55617a", "#aab6c6",  "#6cd3ad", "#46586b",
      "#39424f", "#dfe8ee", "#8a97a8", kMakoRoles, kMakoFlows};
  return scheme == AIE::ColorScheme::Mako ? mako : light;
}

static StringRef roleFill(const Palette &palette, unsigned id) {
  return palette.roles[id % palette.roles.size()];
}

/// A packet ID is five bits wide, so a rule whose mask covers all of them
/// accepts exactly the one packet it names.
static constexpr unsigned kPacketIDMask = 0x1f;

/// The port a wire out of this one reaches.
static std::optional<PortNode> wireNeighbor(const AIE::AIETargetModel &model,
                                            const PortNode &port) {
  std::optional<std::pair<AIE::TileID, AIE::Port>> endpoint =
      AIE::getConnectingWireEndpoint(model, AIE::TileID{port.col, port.row},
                                     AIE::Port{port.bundle, port.channel});
  if (!endpoint)
    return std::nullopt;
  PortNode other = port;
  other.col = endpoint->first.col;
  other.row = endpoint->first.row;
  other.bundle = endpoint->second.bundle;
  other.channel = endpoint->second.channel;
  return other;
}

/// The wires between neighboring tiles that the design uses, each listed
/// from the output port to the input port it feeds. No op states a wire, so
/// a chain of switchbox rules stops at the edge of a tile without one.
///
/// A PortNode does not record whether it is an output or an input, so both
/// ends of a channel have the same name and either could be the output. A
/// route lists its hops in order, and a rule names its source and its
/// destination, so both say which end of a channel the design writes.
static std::vector<std::pair<PortNode, PortNode>>
collectWires(const AIE::AIETargetModel &model, ArrayRef<FlowRoute> routes,
             ArrayRef<ArbitrationRule> rules) {
  auto base = [](PortNode port) {
    port.packet = std::nullopt;
    return port;
  };
  std::set<PortNode> outputs;
  std::set<PortNode> inputs;
  std::set<std::pair<PortNode, PortNode>> routed;
  for (const FlowRoute &route : routes) {
    for (auto [index, point] : llvm::enumerate(route.points))
      (index % 2 == 0 ? outputs : inputs).insert(base(point));
    for (auto pair : llvm::zip_equal(ArrayRef(route.points).drop_back(),
                                     ArrayRef(route.points).drop_front())) {
      const auto &[from, to] = pair;
      if (std::optional<PortNode> step = wireNeighbor(model, base(from)))
        if (*step == base(to))
          routed.insert({base(from), base(to)});
    }
  }
  for (const ArbitrationRule &rule : rules) {
    inputs.insert(base(rule.segment.source));
    outputs.insert(base(rule.segment.dest));
  }

  std::vector<std::pair<PortNode, PortNode>> wires;
  for (const PortNode &port : outputs) {
    std::optional<PortNode> neighbor = wireNeighbor(model, port);
    if (!neighbor || !inputs.count(*neighbor))
      continue;
    // Skip this wire when a route already crosses the channel the other way.
    if (routed.count({*neighbor, port}))
      continue;
    wires.emplace_back(port, *neighbor);
  }
  return wires;
}

/// Which packets each flow carries. A flow claims every ID that agrees with
/// its own on the bits its mask selects, so one flow can carry several that
/// a switchbox further on routes apart. Only an ID something in the design
/// names is worth telling apart.
static void assignCarriedPackets(std::vector<FlowRoute> &routes,
                                 ArrayRef<ArbitrationRule> rules) {
  std::set<unsigned> named;
  for (const FlowRoute &route : routes)
    if (route.packetID)
      named.insert(*route.packetID);
  for (const ArbitrationRule &rule : rules)
    if (rule.mask == kPacketIDMask)
      named.insert(rule.value);

  for (FlowRoute &route : routes) {
    if (!route.packetID)
      continue;
    unsigned id = *route.packetID;
    route.carried.insert(id);
    if (!route.packetMask)
      continue;
    unsigned mask = *route.packetMask;
    for (unsigned packet : named)
      if ((packet & mask) == id)
        route.carried.insert(packet);
  }
}

/// Which packets cross each switchbox port, keyed by the port without a
/// packet ID in its name.
///
/// A rule whose mask covers the whole ID accepts one packet, and a flow
/// states the ones it carries. Rules and wires then carry each packet
/// onwards until no port gains one.
static std::map<PortNode, std::set<unsigned>>
collectPortPackets(ArrayRef<FlowRoute> routes, ArrayRef<ArbitrationRule> rules,
                   ArrayRef<std::pair<PortNode, PortNode>> wires) {
  std::map<PortNode, std::set<unsigned>> packets;
  for (const FlowRoute &route : routes)
    for (const PortNode &point : route.points) {
      PortNode base = point;
      base.packet = std::nullopt;
      packets[base].insert(route.carried.begin(), route.carried.end());
    }
  for (const ArbitrationRule &rule : rules)
    if (rule.mask == kPacketIDMask)
      packets[rule.segment.source].insert(rule.value);

  bool changed = true;
  while (changed) {
    changed = false;
    for (const ArbitrationRule &rule : rules) {
      std::set<unsigned> arriving = packets[rule.segment.source];
      for (unsigned packet : arriving)
        if ((packet & rule.mask) == rule.value)
          changed |= packets[rule.segment.dest].insert(packet).second;
    }
    for (const auto &[from, to] : wires) {
      std::set<unsigned> arriving = packets[from];
      for (unsigned packet : arriving)
        changed |= packets[to].insert(packet).second;
    }
  }
  return packets;
}

/// The ends of one hop of a route, one pair per packet the flow carries.
static std::vector<std::pair<PortNode, PortNode>>
hopSegments(const FlowRoute &route, const PortNode &from, const PortNode &to,
            bool splitPacketPorts) {
  if (!splitPacketPorts || route.carried.empty())
    return {{from, to}};
  std::vector<std::pair<PortNode, PortNode>> segments;
  for (unsigned packet : route.carried) {
    PortNode source = from;
    PortNode dest = to;
    if (source.bundle != AIE::WireBundle::DMA)
      source.packet = packet;
    if (dest.bundle != AIE::WireBundle::DMA)
      dest.packet = packet;
    segments.emplace_back(source, dest);
  }
  return segments;
}

/// The ends a wire joins, one pair per packet crossing it.
static std::vector<std::pair<PortNode, PortNode>>
wireSegments(const std::map<PortNode, std::set<unsigned>> &portPackets,
             const PortNode &from, const PortNode &to, bool splitPacketPorts) {
  auto found = portPackets.find(from);
  if (!splitPacketPorts || found == portPackets.end() || found->second.empty())
    return {{from, to}};
  std::vector<std::pair<PortNode, PortNode>> segments;
  for (unsigned packet : found->second) {
    PortNode driven = from;
    PortNode read = to;
    driven.packet = packet;
    read.packet = packet;
    segments.emplace_back(driven, read);
  }
  return segments;
}

/// The two ports a rule connects, once per packet that both arrives at its
/// source and matches its mask and value. A DMA channel is one node whatever
/// crosses it, so only a switchbox port takes the packet ID in its name.
static std::vector<std::pair<PortNode, PortNode>>
claimedSegments(const std::map<PortNode, std::set<unsigned>> &portPackets,
                const Segment &segment, unsigned mask, unsigned value,
                bool splitPacketPorts) {
  auto found = portPackets.find(segment.source);
  if (found == portPackets.end() || found->second.empty())
    return {{segment.source, segment.dest}};

  std::vector<std::pair<PortNode, PortNode>> claimed;
  for (unsigned packet : found->second) {
    if ((packet & mask) != value)
      continue;
    PortNode source = segment.source;
    PortNode dest = segment.dest;
    if (splitPacketPorts && source.bundle != AIE::WireBundle::DMA)
      source.packet = packet;
    if (splitPacketPorts && dest.bundle != AIE::WireBundle::DMA)
      dest.packet = packet;
    claimed.emplace_back(source, dest);
  }
  return claimed;
}

static std::string portNodeID(const PortNode &port) {
  std::string id = "p_" + std::to_string(port.col) + "_" +
                   std::to_string(port.row) + "_" +
                   std::to_string(static_cast<int>(port.bundle)) + "_" +
                   std::to_string(port.channel);
  if (port.dmaDirection)
    id += *port.dmaDirection == AIE::DMAChannelDir::MM2S ? "_m" : "_s";
  if (port.packet)
    id += "_k" + std::to_string(*port.packet);
  return id;
}

enum class PortRole { Ingress, Egress };

static PortRole circuitPointRole(const FlowRoute &route, unsigned index) {
  if (index == 0 && route.points.size() > 1 &&
      route.points[0] == route.points[1])
    return PortRole::Ingress;
  if (index + 1 == route.points.size() && index > 0 &&
      route.points[index - 1] == route.points[index])
    return PortRole::Egress;
  return index % 2 == 0 ? PortRole::Egress : PortRole::Ingress;
}

static std::string circuitPortNodeID(const PortNode &port, PortRole role,
                                     unsigned routeGroup) {
  if (port.bundle == AIE::WireBundle::DMA)
    return portNodeID(port);
  return portNodeID(port) + (role == PortRole::Ingress ? "_in_r" : "_out_r") +
         std::to_string(routeGroup);
}

static std::vector<unsigned> circuitRouteGroups(ArrayRef<FlowRoute> routes) {
  std::vector<unsigned> parent(routes.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](unsigned id) {
    while (parent[id] != id) {
      parent[id] = parent[parent[id]];
      id = parent[id];
    }
    return id;
  };
  auto unite = [&](unsigned first, unsigned second) {
    first = root(first);
    second = root(second);
    if (first != second)
      parent[std::max(first, second)] = std::min(first, second);
  };
  for (const FlowRoute &first : routes) {
    if (first.packetID)
      continue;
    for (const FlowRoute &second : routes.drop_front(first.id + 1)) {
      if (second.packetID)
        continue;
      bool sameSource =
          first.points.front() == second.points.front() &&
          circuitPointRole(first, 0) == circuitPointRole(second, 0);
      bool continues = (first.points.back() == second.points.front() &&
                        circuitPointRole(first, first.points.size() - 1) ==
                            circuitPointRole(second, 0)) ||
                       (second.points.back() == first.points.front() &&
                        circuitPointRole(second, second.points.size() - 1) ==
                            circuitPointRole(first, 0));
      if (sameSource || continues)
        unite(first.id, second.id);
    }
  }
  for (unsigned &id : parent)
    id = root(id);
  return parent;
}

static std::string bufferNodeID(unsigned id) {
  return "buffer_" + std::to_string(id);
}

static std::string callNodeID(unsigned id) {
  return "call_" + std::to_string(id);
}

static std::string runtimeArgumentNodeID(unsigned id) {
  return "runtime_arg_" + std::to_string(id);
}

/// How far back from the tip an arrowhead's base sits, in inches: ten
/// points of head at the 0.65 arrowsize the edges are drawn with.
static constexpr double kArrowLength = 10.0 * 0.65 / 72.0;

static std::string tileNodeID(int col, int row) {
  return "tile_" + std::to_string(col) + "_" + std::to_string(row);
}

/// Bytes each interned boundary set moves per iteration, measured at the DMA
/// channels carrying it. An origin is rarely a channel itself -- it is the
/// runtime argument or kernel upstream of one -- so the volume is read off
/// the channels that share its provenance.
static void boundarySetBytes(const Provenance &provenance,
                             const DMAChannelBytes &channelBytes,
                             std::vector<uint64_t> &sourceBytes,
                             std::vector<uint64_t> &destBytes) {
  sourceBytes.assign(provenance.sourceSets.size(), 0);
  destBytes.assign(provenance.destSets.size(), 0);
  for (const auto &[key, bytes] : channelBytes) {
    auto [col, row, direction, channel] = key;
    PortNode port{col, row, AIE::WireBundle::DMA, channel, direction};
    std::string id = portNodeID(port);
    if (direction == AIE::DMAChannelDir::MM2S) {
      auto group = provenance.sourceGroup.find(id);
      if (group != provenance.sourceGroup.end() &&
          !provenance.sourceSets[group->second].empty())
        sourceBytes[group->second] += bytes;
      continue;
    }
    auto group = provenance.destGroup.find(id);
    if (group != provenance.destGroup.end() &&
        !provenance.destSets[group->second].empty())
      destBytes[group->second] += bytes;
  }
}

static std::string bufferLabel(AIE::BufferOp buffer, unsigned id);

static SemanticGraph buildSemanticGraph(
    AIE::DeviceOp device, ArrayRef<FlowRoute> routes,
    ArrayRef<ArbitrationRule> arbitrationRules,
    const std::map<PortNode, std::set<unsigned>> &portPackets,
    ArrayRef<std::pair<PortNode, PortNode>> wires, bool splitPacketPorts,
    const DMAChannelBuffers &channelBuffers,
    const std::set<DMAChannelKey> &activeChannels, ArrayRef<BufferInfo> buffers,
    ArrayRef<RuntimeArgumentInfo> runtimeArguments, ArrayRef<CallInfo> calls,
    const std::set<Operation *> &buffersFeedingDMA,
    const std::set<Operation *> &buffersFedByDMA) {
  SemanticGraph graph;
  std::map<std::pair<int, int>, SourceLocations> tileSources;
  for (AIE::TileOp tile : device.getOps<AIE::TileOp>())
    tileSources[{tile.colIndex(), tile.rowIndex()}] =
        sourceLocations(tile.getLoc());
  auto addPort = [&](const PortNode &port, SourceLocations source = {}) {
    std::string id = portNodeID(port);
    graph.addNode(id,
                  port.bundle == AIE::WireBundle::DMA ? SemanticNodeKind::DMA
                                                      : SemanticNodeKind::Port,
                  std::make_pair(port.col, port.row), port, std::move(source));
    return id;
  };
  auto addCircuitPort = [&](const PortNode &port, PortRole role,
                            unsigned routeGroup, SourceLocations source = {}) {
    std::string id = circuitPortNodeID(port, role, routeGroup);
    graph.addNode(id,
                  port.bundle == AIE::WireBundle::DMA ? SemanticNodeKind::DMA
                                                      : SemanticNodeKind::Port,
                  std::make_pair(port.col, port.row), port, std::move(source));
    return id;
  };
  auto addTile = [&](int col, int row) {
    std::string id = tileNodeID(col, row);
    graph.addNode(id, SemanticNodeKind::Tile, std::make_pair(col, row),
                  std::nullopt, tileSources[{col, row}],
                  "(" + std::to_string(col) + "," + std::to_string(row) + ")");
    return id;
  };

  // Every hop a route takes, not only where it starts and ends. A view that
  // keeps ports then has them to draw, and one that contracts them removes
  // them here rather than never collecting them.
  std::vector<unsigned> routeGroups = circuitRouteGroups(routes);
  for (const FlowRoute &route : routes) {
    SourceLocations routeSource = sourceLocations(route.loc);
    for (auto [index, pair] : llvm::enumerate(
             llvm::zip_equal(ArrayRef(route.points).drop_back(),
                             ArrayRef(route.points).drop_front()))) {
      const auto &[from, to] = pair;
      for (const auto &[source, dest] :
           hopSegments(route, from, to, splitPacketPorts)) {
        if (route.packetID) {
          if (source == dest)
            continue;
          graph.addEdge(addPort(source, routeSource),
                        addPort(dest, routeSource), route.id, routeSource);
          continue;
        }
        std::string sourceID =
            addCircuitPort(source, circuitPointRole(route, index),
                           routeGroups[route.id], routeSource);
        std::string destID =
            addCircuitPort(dest, circuitPointRole(route, index + 1),
                           routeGroups[route.id], routeSource);
        if (sourceID != destID)
          graph.addEdge(sourceID, destID, route.id, routeSource);
      }
    }
    addTile(route.points.front().col, route.points.front().row);
    addTile(route.points.back().col, route.points.back().row);
  }

  // Neither a packet merge -- two ingress ports sharing an arbiter slot --
  // nor the wire from one switchbox to the next can be written as a flow, so
  // aie-find-flows leaves both in the configuration. Reading them here is
  // what keeps a flow arriving at such a port from appearing to stop.
  for (const ArbitrationRule &rule : arbitrationRules)
    for (const auto &[ingress, egress] :
         claimedSegments(portPackets, rule.segment, rule.mask, rule.value,
                         splitPacketPorts)) {
      if (portNodeID(ingress) == portNodeID(egress))
        continue;
      SourceLocations ruleSource = sourceLocations(rule.loc);
      graph.addEdge(addPort(ingress, ruleSource), addPort(egress, ruleSource),
                    std::nullopt, ruleSource);
    }
  for (const auto &[from, to] : wires)
    for (const auto &[driven, read] :
         wireSegments(portPackets, from, to, splitPacketPorts)) {
      auto packets = portPackets.find(from);
      if (packets == portPackets.end() || packets->second.empty())
        continue;
      std::string drivenID = portNodeID(driven);
      std::string readID = portNodeID(read);
      // Both ends of a channel have the same port name, so two flows using it
      // in opposite directions would join the same two nodes twice.
      if (graph.hasEdge(readID, drivenID))
        continue;
      graph.addEdge(addPort(driven), addPort(read));
    }

  std::map<Operation *, unsigned> bufferIDs;
  for (const BufferInfo &buffer : buffers) {
    bufferIDs[buffer.op] = buffer.id;
    graph.addNode(bufferNodeID(buffer.id), SemanticNodeKind::Buffer,
                  std::make_pair(buffer.col, buffer.row), std::nullopt,
                  sourceLocations(buffer.op->getLoc()),
                  bufferLabel(buffer.op, buffer.id));
    addTile(buffer.col, buffer.row);
  }
  for (const auto &[key, channelBufferList] : channelBuffers) {
    if (!activeChannels.count(key))
      continue;
    auto [col, row, direction, channel] = key;
    PortNode port{col, row, AIE::WireBundle::DMA, channel, direction};
    std::string portID = addPort(port);
    for (AIE::BufferOp buffer : channelBufferList) {
      std::string bufferID = bufferNodeID(bufferIDs.at(buffer));
      bool draining = direction == AIE::DMAChannelDir::MM2S;
      graph.addEdge(draining ? bufferID : portID, draining ? portID : bufferID);
    }
  }

  for (const RuntimeArgumentInfo &argument : runtimeArguments) {
    std::string argumentID = runtimeArgumentNodeID(argument.id);
    SourceLocations argumentSource = sourceLocations(argument.loc);
    graph.addNode(argumentID, SemanticNodeKind::RuntimeArgument, std::nullopt,
                  std::nullopt, argumentSource, argument.name);
    for (const DMAChannelKey &key : argument.channels) {
      auto [col, row, direction, channel] = key;
      PortNode port{col, row, AIE::WireBundle::DMA, channel, direction};
      std::string portID = addPort(port);
      std::string source =
          direction == AIE::DMAChannelDir::MM2S ? argumentID : portID;
      std::string dest =
          direction == AIE::DMAChannelDir::MM2S ? portID : argumentID;
      graph.addEdge(source, dest, std::nullopt, argumentSource);
    }
  }

  for (const CallInfo &call : calls) {
    std::string callID = callNodeID(call.id);
    graph.addNode(callID, SemanticNodeKind::Call,
                  std::make_pair(call.col, call.row), std::nullopt,
                  sourceLocations(call.loc), call.callee);
    addTile(call.col, call.row);
    for (AIE::BufferOp buffer : call.buffers) {
      std::string bufferID = bufferNodeID(bufferIDs.at(buffer));
      if (buffersFeedingDMA.count(buffer))
        graph.addEdge(callID, bufferID);
      if (buffersFedByDMA.count(buffer))
        graph.addEdge(bufferID, callID);
    }
  }

  // Join every port with nothing on one side to its tile, so that each chain
  // starts and ends somewhere. The tile is all the design says about where
  // such a stream comes from or goes to.
  std::vector<std::pair<std::string, std::pair<int, int>>> toTile;
  std::vector<std::pair<std::string, std::pair<int, int>>> fromTile;
  for (const auto &[id, node] : graph.nodes) {
    if (node.kind != SemanticNodeKind::Port &&
        node.kind != SemanticNodeKind::DMA)
      continue;
    if (!node.tile)
      continue;
    if (!graph.hasIncoming(id))
      fromTile.emplace_back(id, *node.tile);
    if (!graph.hasOutgoing(id))
      toTile.emplace_back(id, *node.tile);
  }
  for (const auto &[id, tile] : fromTile)
    graph.addEdge(addTile(tile.first, tile.second), id);
  for (const auto &[id, tile] : toTile)
    graph.addEdge(id, addTile(tile.first, tile.second));
  return graph;
}

static std::string escapeDotLabel(StringRef value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char character : value) {
    if (character == '\\' || character == '\"')
      escaped.push_back('\\');
    escaped.push_back(character);
  }
  return escaped;
}

static std::string bufferLabel(AIE::BufferOp buffer, unsigned id) {
  if (auto name = buffer->getAttrOfType<StringAttr>("sym_name"))
    return escapeDotLabel(name.getValue());
  return "buffer " + std::to_string(id);
}

static std::pair<double, double> portPosition(const PortNode &port,
                                              const LayoutGeometry &layout) {
  double x = layout.tileCenterX(port.col);
  double y = layout.tileCenterY(port.row);
  double halfWidth = layout.tileWidth / 2;
  double halfHeight = layout.tileHeight(port.col, port.row) / 2;
  double channelOffset = (port.channel - 1.5) * 0.18;
  switch (port.bundle) {
  case AIE::WireBundle::North:
    return {x + channelOffset, y + halfHeight};
  case AIE::WireBundle::South:
    return {x + channelOffset, y - halfHeight};
  case AIE::WireBundle::East:
    return {x + halfWidth, y + channelOffset};
  case AIE::WireBundle::West:
    return {x - halfWidth, y + channelOffset};
  case AIE::WireBundle::DMA:
    return {x + (port.dmaDirection == AIE::DMAChannelDir::S2MM
                     ? halfWidth - 0.2
                     : 0.2 - halfWidth),
            y + (port.channel - 2.5) * 0.26};
  case AIE::WireBundle::Core:
    return {x + halfWidth - 0.2, y + channelOffset};
  default:
    return {x, y + channelOffset};
  }
}

/// Which origins reach a node or edge and which fates it reaches, as the
/// interned IDs the viewer matches on. An edge takes its origins from the end
/// it leaves and its fates from the end it arrives at, so a fan-out shares an
/// origin while its branches differ in where they end up.
struct Boundary {
  unsigned source = 0;
  unsigned dest = 0;
  bool hasSource = false;
  bool hasDest = false;
};

struct FlowMembership {
  std::set<std::string> origins;
  std::set<std::string> fates;
  std::optional<std::string> boundary;
};

static std::string flowClasses(StringRef kind, const Boundary &boundary,
                               const FlowMembership &membership) {
  std::string classes = kind.str();
  if (boundary.hasSource || boundary.hasDest)
    classes += " flow";
  if (boundary.hasSource)
    classes += " src-" + std::to_string(boundary.source);
  if (boundary.hasDest)
    classes += " dst-" + std::to_string(boundary.dest);
  for (const std::string &origin : membership.origins)
    classes += " origin-" + origin;
  for (const std::string &fate : membership.fates)
    classes += " fate-" + fate;
  if (membership.boundary)
    classes += " boundary-" + *membership.boundary;
  return classes;
}

static std::string flowColor(const Palette &palette, unsigned id) {
  return palette.flows[id % palette.flows.size()];
}

static std::string shortPortName(const PortNode &port) {
  switch (port.bundle) {
  case AIE::WireBundle::DMA:
    return (port.dmaDirection == AIE::DMAChannelDir::S2MM ? "S2MM" : "MM2S") +
           std::to_string(port.channel);
  case AIE::WireBundle::Core:
    return "C" + std::to_string(port.channel);
  case AIE::WireBundle::North:
    return "N" + std::to_string(port.channel);
  case AIE::WireBundle::South:
    return "S" + std::to_string(port.channel);
  case AIE::WireBundle::East:
    return "E" + std::to_string(port.channel);
  case AIE::WireBundle::West:
    return "W" + std::to_string(port.channel);
  default:
    return stringifyWireBundle(port.bundle).str() +
           std::to_string(port.channel);
  }
}

static StringRef semanticNodeKindName(SemanticNodeKind kind) {
  switch (kind) {
  case SemanticNodeKind::Tile:
    return "tile";
  case SemanticNodeKind::Port:
    return "port";
  case SemanticNodeKind::DMA:
    return "dma";
  case SemanticNodeKind::Buffer:
    return "buffer";
  case SemanticNodeKind::RuntimeArgument:
    return "runtime-argument";
  case SemanticNodeKind::Call:
    return "call";
  }
  llvm_unreachable("unknown semantic node kind");
}

static std::optional<SemanticNodeKind> parseSemanticNodeKind(StringRef kind) {
  for (SemanticNodeKind candidate :
       {SemanticNodeKind::Tile, SemanticNodeKind::Port, SemanticNodeKind::DMA,
        SemanticNodeKind::Buffer, SemanticNodeKind::RuntimeArgument,
        SemanticNodeKind::Call})
    if (semanticNodeKindName(candidate) == kind)
      return candidate;
  return std::nullopt;
}

struct SelectedTopology {
  SemanticEdges edges;
  Provenance provenance;
  TransitiveMembership membership;
};

static FailureOr<SelectedTopology>
selectTopology(AIE::DeviceOp device, const SemanticGraph &graph,
               const AIE::VisualizeOptions &options) {
  std::set<SemanticNodeKind> contractKinds;
  for (StringRef kind : options.contractedNodeKinds) {
    std::optional<SemanticNodeKind> parsed = parseSemanticNodeKind(kind);
    if (!parsed) {
      device.emitOpError() << "--contract references unknown node kind '"
                           << kind << "'";
      return failure();
    }
    contractKinds.insert(*parsed);
  }

  SemanticEdges edges =
      graph.contract(contractKinds, options.contractBalancedFanout);
  Provenance provenance = computeProvenance(graph, edges);
  TransitiveMembership membership = computeTransitiveMembership(graph, edges);
  std::set<unsigned> only(options.onlyFlows.begin(), options.onlyFlows.end());
  for (unsigned id : only) {
    if (id < provenance.sourceSets.size())
      continue;
    device.emitOpError() << "--only-flow references unknown flow " << id
                         << "; valid IDs are 0 through "
                         << (provenance.sourceSets.empty()
                                 ? 0
                                 : provenance.sourceSets.size() - 1);
    return failure();
  }
  if (!only.empty())
    for (auto entry = edges.begin(); entry != edges.end();)
      entry = only.count(provenance.sourceGroup.at(entry->first.source))
                  ? std::next(entry)
                  : edges.erase(entry);
  return SelectedTopology{std::move(edges), std::move(provenance),
                          std::move(membership)};
}

static json::Array sourceLocationsJson(const SourceLocations &locations) {
  json::Array result;
  for (const auto &[file, sourceLines] : locations) {
    json::Array lines;
    for (unsigned line : sourceLines)
      lines.push_back(line);
    result.push_back(
        json::Object{{"file", file}, {"lines", std::move(lines)}});
  }
  return result;
}

static std::string edgeID(const EdgeKey &key) {
  return "edge__" + key.source + "__" + key.dest;
}

static void writeSemanticGraphJson(AIE::DeviceOp device,
                                   const SemanticGraph &graph,
                                   const SelectedTopology &topology,
                                   raw_ostream &output) {
  std::set<std::string> selectedNodes;
  for (const auto &[key, attributes] : topology.edges) {
    selectedNodes.insert(key.source);
    selectedNodes.insert(key.dest);
  }

  json::Array nodes;
  for (const auto &[id, node] : graph.nodes) {
    if (!selectedNodes.count(id))
      continue;
    json::Object object{{"id", id},
                        {"kind", semanticNodeKindName(node.kind)},
                        {"label", node.label.empty() ? id : node.label},
                        {"sources", sourceLocationsJson(node.source)}};
    if (node.tile)
      object["tile"] = json::Array{node.tile->first, node.tile->second};
    if (node.port) {
      json::Object port{{"bundle", stringifyWireBundle(node.port->bundle)},
                        {"channel", node.port->channel}};
      if (node.port->dmaDirection)
        port["direction"] = *node.port->dmaDirection == AIE::DMAChannelDir::MM2S
                                ? "MM2S"
                                : "S2MM";
      if (node.port->packet)
        port["packet"] = *node.port->packet;
      object["port"] = std::move(port);
    }
    nodes.push_back(std::move(object));
  }

  json::Array edges;
  for (const auto &[key, attributes] : topology.edges) {
    json::Array routes;
    for (unsigned route : attributes.routes)
      routes.push_back(route);
    edges.push_back(json::Object{
        {"id", edgeID(key)},
        {"source", key.source},
        {"target", key.dest},
        {"routes", std::move(routes)},
        {"pathCount", attributes.count},
        {"sources", sourceLocationsJson(attributes.source)}});
  }

  json::Object root{{"format", "aie-semantic-graph"},
                    {"version", 2},
                    {"device", device.getSymName().str()},
                    {"nodes", std::move(nodes)},
                    {"edges", std::move(edges)}};
  output << formatv("{0:2}\n", json::Value(std::move(root)));
}

struct SemanticGraphAnalysis {
  std::vector<FlowRoute> routes;
  std::vector<ArbitrationRule> arbitrationRules;
  std::vector<BufferInfo> buffers;
  DMAChannelBuffers channelBuffers;
  DMAChannelBytes channelBytes;
  std::vector<CallInfo> calls;
  std::vector<std::pair<PortNode, PortNode>> wires;
  std::map<PortNode, std::set<unsigned>> portPackets;
  std::vector<RuntimeArgumentInfo> runtimeArguments;
  SemanticGraph graph;
};

static FailureOr<SemanticGraphAnalysis>
analyzeSemanticGraph(AIE::DeviceOp device,
                     AIE::RuntimeSequenceOp runtimeSequence,
                     const AIE::VisualizeOptions &options) {
  FailureOr<std::vector<FlowRoute>> routes = collectRoutes(device);
  if (failed(routes))
    return failure();
  FailureOr<std::vector<BufferInfo>> buffers = collectBuffers(device);
  if (failed(buffers))
    return failure();

  SemanticGraphAnalysis analysis;
  analysis.routes = std::move(*routes);
  analysis.buffers = std::move(*buffers);
  analysis.arbitrationRules = collectArbitrationRules(device);
  if (!options.controlOverlay)
    llvm::erase_if(analysis.arbitrationRules, [](const ArbitrationRule &rule) {
      return rule.controlOverlay;
    });
  analysis.channelBuffers = collectDMAChannelBuffers(device);
  analysis.channelBytes = collectDMAChannelBytes(device);
  analysis.calls = collectCalls(device);

  assignCarriedPackets(analysis.routes, analysis.arbitrationRules);
  const AIE::AIETargetModel &model = device.getTargetModel();
  analysis.wires =
      collectWires(model, analysis.routes, analysis.arbitrationRules);
  analysis.portPackets = collectPortPackets(
      analysis.routes, analysis.arbitrationRules, analysis.wires);

  std::set<DMAChannelKey> activeChannels;
  std::set<Operation *> buffersFeedingDMA;
  std::set<Operation *> buffersFedByDMA;
  auto noteChannel = [&](const PortNode &port) {
    if (port.bundle != AIE::WireBundle::DMA || !port.dmaDirection)
      return;
    DMAChannelKey key{port.col, port.row, *port.dmaDirection, port.channel};
    activeChannels.insert(key);
    bool draining = *port.dmaDirection == AIE::DMAChannelDir::MM2S;
    for (AIE::BufferOp buffer : analysis.channelBuffers[key])
      (draining ? buffersFeedingDMA : buffersFedByDMA).insert(buffer);
  };
  auto noteSegment = [&](const PortNode &from, const PortNode &to) {
    noteChannel(from);
    noteChannel(to);
  };
  for (const FlowRoute &route : analysis.routes)
    for (auto pair : llvm::zip_equal(ArrayRef(route.points).drop_back(),
                                     ArrayRef(route.points).drop_front())) {
      const auto &[from, to] = pair;
      for (const auto &[source, dest] :
           hopSegments(route, from, to, options.splitPacketPorts))
        noteSegment(source, dest);
    }
  for (const ArbitrationRule &rule : analysis.arbitrationRules)
    for (const auto &[ingress, egress] :
         claimedSegments(analysis.portPackets, rule.segment, rule.mask,
                         rule.value, options.splitPacketPorts))
      noteSegment(ingress, egress);
  for (const auto &[from, to] : analysis.wires)
    for (const auto &[driven, read] :
         wireSegments(analysis.portPackets, from, to, options.splitPacketPorts))
      noteSegment(driven, read);

  std::vector<RuntimeBufferUse> runtimeBufferUses =
      collectRuntimeBufferUses(device, runtimeSequence);
  std::map<unsigned, unsigned> argumentIDs;
  for (const RuntimeBufferUse &use : runtimeBufferUses) {
    auto [entry, inserted] = argumentIDs.try_emplace(
        use.argumentIndex, analysis.runtimeArguments.size());
    if (inserted)
      analysis.runtimeArguments.push_back(
          {use.argumentIndex, use.argument, {}, runtimeSequence.getLoc()});
    RuntimeArgumentInfo &argument = analysis.runtimeArguments[entry->second];
    if (!llvm::is_contained(argument.channels, use.channel))
      argument.channels.push_back(use.channel);
  }
  llvm::sort(analysis.runtimeArguments, [](const RuntimeArgumentInfo &left,
                                           const RuntimeArgumentInfo &right) {
    return left.id < right.id;
  });

  analysis.graph = buildSemanticGraph(
      device, analysis.routes, analysis.arbitrationRules, analysis.portPackets,
      analysis.wires, options.splitPacketPorts, analysis.channelBuffers,
      activeChannels, analysis.buffers, analysis.runtimeArguments,
      analysis.calls, buffersFeedingDMA, buffersFedByDMA);
  return analysis;
}

static LogicalResult emitDeviceRouteDot(AIE::DeviceOp device,
                                        raw_ostream &output,
                                        AIE::RuntimeSequenceOp runtimeSequence,
                                        const AIE::VisualizeOptions &options) {
  FailureOr<SemanticGraphAnalysis> analysis =
      analyzeSemanticGraph(device, runtimeSequence, options);
  if (failed(analysis))
    return failure();
  auto *routes = &analysis->routes;
  auto *buffers = &analysis->buffers;
  auto &arbitrationRules = analysis->arbitrationRules;
  auto &channelBytes = analysis->channelBytes;
  auto &calls = analysis->calls;
  auto &portPackets = analysis->portPackets;
  auto &runtimeArguments = analysis->runtimeArguments;
  auto &semanticGraph = analysis->graph;
  bool drawsPorts = !llvm::is_contained(options.contractedNodeKinds, "port");
  if (drawsPorts)
    for (const FlowRoute &route : *routes)
      if (route.points.size() < 3)
        return mlir::emitError(route.loc)
               << "flow carries no vias, so its route cannot be drawn; run "
                  "aie-find-flows with emit-vias=true, or contract ports";
  const Palette &palette = paletteFor(options.colors);
  std::unique_ptr<ArrowRenderer> arrows;
  switch (options.arrows) {
  case AIE::ArrowStyle::Straight:
    arrows = createStraightArrowRenderer();
    break;
  case AIE::ArrowStyle::Channel:
    arrows = createChannelArrowRenderer();
    break;
  case AIE::ArrowStyle::Avoidance:
    arrows = createAvoidanceArrowRenderer();
    break;
  }

  double widest = options.maxEdgeWidth;
  if (std::optional<double> room = arrows->widestLine())
    widest = std::min(widest, *room);
  widest *= options.scale;

  const AIE::AIETargetModel &model = device.getTargetModel();

  std::set<std::pair<int, int>> designTiles;
  for (AIE::TileOp tile : device.getOps<AIE::TileOp>())
    designTiles.insert({tile.colIndex(), tile.rowIndex()});

  std::map<std::pair<int, int>, unsigned> localCounts;
  std::map<Operation *, unsigned> bufferLocalIndices;
  std::map<unsigned, unsigned> callLocalIndices;

  FailureOr<SelectedTopology> selected =
      selectTopology(device, semanticGraph, options);
  if (failed(selected))
    return failure();
  SemanticEdges &topologyEdges = selected->edges;
  Provenance &provenance = selected->provenance;
  TransitiveMembership &membership = selected->membership;

  auto boundaryOfNode = [&](const std::string &id) {
    Boundary boundary;
    auto source = provenance.sourceGroup.find(id);
    if (source != provenance.sourceGroup.end()) {
      boundary.source = source->second;
      boundary.hasSource = !provenance.sourceSets[boundary.source].empty();
    }
    auto dest = provenance.destGroup.find(id);
    if (dest != provenance.destGroup.end()) {
      boundary.dest = dest->second;
      boundary.hasDest = !provenance.destSets[boundary.dest].empty();
    }
    return boundary;
  };
  auto boundaryOfEdge = [&](const EdgeKey &key) {
    Boundary leaving = boundaryOfNode(key.source);
    Boundary arriving = boundaryOfNode(key.dest);
    return Boundary{leaving.source, arriving.dest, leaving.hasSource,
                    arriving.hasDest};
  };
  auto membershipOfNode = [&](const std::string &id) {
    return FlowMembership{membership.origins.at(id), membership.fates.at(id),
                          id};
  };
  auto membershipOfEdge = [&](const EdgeKey &key) {
    auto anchor = [&](const std::string &id) {
      SemanticNodeKind kind = semanticGraph.nodes.at(id).kind;
      return kind == SemanticNodeKind::Tile || kind == SemanticNodeKind::Call;
    };
    const std::set<std::string> &origins = membership.origins.at(key.source);
    const std::set<std::string> &fates = membership.fates.at(key.dest);
    if (anchor(key.source))
      return FlowMembership{{key.source}, fates, std::nullopt};
    if (anchor(key.dest))
      return FlowMembership{origins, {key.dest}, std::nullopt};
    return FlowMembership{origins, fates, std::nullopt};
  };

  std::vector<uint64_t> sourceBytes, destBytes;
  boundarySetBytes(provenance, channelBytes, sourceBytes, destBytes);
  // An origin that states no volume -- a kernel rather than a buffer
  // descriptor -- leaves the question to where the data ends up.
  auto edgeBytes = [&](const Boundary &boundary) {
    uint64_t bytes = boundary.hasSource ? sourceBytes[boundary.source] : 0;
    if (bytes == 0 && boundary.hasDest)
      bytes = destBytes[boundary.dest];
    return bytes;
  };
  EdgeWeightScale weightScale =
      options.weighEdges
          ? EdgeWeightScale::build(sourceBytes,
                                   options.minEdgeWidth * options.scale, widest)
          : EdgeWeightScale();

  // Contraction removes a node by splicing it out of the edges it joined, so
  // what is left to draw is what still has an edge. A buffer no flow reaches
  // joins nothing and says nothing; the tile headline still names the kernels
  // it holds.
  std::set<std::string> topologyNodes;
  for (const auto &[key, attributes] : topologyEdges) {
    topologyNodes.insert(key.source);
    topologyNodes.insert(key.dest);
  }
  for (const BufferInfo &buffer : *buffers) {
    if (!topologyNodes.count(bufferNodeID(buffer.id)))
      continue;
    std::pair<int, int> coordinate{buffer.col, buffer.row};
    bufferLocalIndices[buffer.op] = localCounts[coordinate]++;
  }
  for (const CallInfo &call : calls) {
    if (!topologyNodes.count(callNodeID(call.id)))
      continue;
    callLocalIndices[call.id] = localCounts[{call.col, call.row}]++;
  }

  std::map<std::string, unsigned> indegree, outdegree;
  for (const auto &[key, attributes] : topologyEdges) {
    ++outdegree[key.source];
    ++indegree[key.dest];
  }
  auto nodeKind = [&](const std::string &id) {
    return semanticGraph.nodes.at(id).kind;
  };
  // A port that only passes things on is a via: detail a reader asks for,
  // not a place where anything is decided. One where flows branch or join
  // is, which is the same test contraction applies.
  auto isVia = [&](const std::string &id) {
    return nodeKind(id) == SemanticNodeKind::Port && indegree[id] == 1 &&
           outdegree[id] == 1;
  };

  if (::getenv("AIE_VISUALIZE_GRAPH_STATS")) {
    llvm::errs() << "graph nodes=" << semanticGraph.nodes.size()
                 << " edges=" << semanticGraph.edges.size()
                 << " contracted=" << topologyEdges.size() << '\n';
  }

  std::map<std::pair<int, int>, TileRole> roles;
  std::map<std::pair<int, int>, RoleClass> roleClasses;
  if (options.tileRoles) {
    roles = collectTileRoles(calls);
    roleClasses = collectRoleClasses(roles);
  }
  LayoutGeometry layout;
  // Only nop2 leaves supplied edge geometry alone; every laying-out engine
  // recomputes it.
  layout.preLaidOut = arrows->suppliesLayout();
  layout.scale = options.scale;

  // The headline sits inside the tile box, so the box reserves room for it
  // before the buffers and calls start.
  auto headroom = [&](std::pair<int, int> coordinate) {
    if (!options.tileRoles)
      return layout.text(0.45);
    auto role = roles.find(coordinate);
    unsigned lines = role == roles.end()
                         ? 1
                         : 1 + role->second.lines(options.maxHeadlineKernels);
    // One line of padding above the headline and one below it.
    return layout.text(0.34 + 0.17 * static_cast<double>(lines + 1));
  };

  // A renderer that needs room between tiles to put arrows in says so here,
  // rather than the caller having to know which style wants what spacing.
  double gapX = options.tileGapX;
  double gapY = options.tileGapY;
  if (std::optional<double> wanted =
          arrows->preferredTileGap(layout.text(1.0))) {
    gapX = std::max(gapX, *wanted);
    gapY = std::max(gapY, *wanted);
  }
  // A tile is sized by the arrows drawn inside it and by the widest label it
  // holds: a box narrower than its own text spills over its neighbors.
  double tileWidth = layout.text(options.tileWidth);
  size_t longestLabel = 0;
  for (const BufferInfo &buffer : *buffers)
    longestLabel =
        std::max(longestLabel, bufferLabel(buffer.op, buffer.id).size());
  for (const CallInfo &call : calls)
    longestLabel = std::max(longestLabel, call.callee.size());
  for (const RuntimeArgumentInfo &argument : runtimeArguments)
    longestLabel = std::max(longestLabel, argument.name.size() + 1);
  // Helvetica at the label's point size comes to about this much of an inch
  // per character.
  double contentWidth =
      std::clamp(layout.text(0.065 * static_cast<double>(longestLabel)),
                 layout.text(1.0), layout.text(3.0));
  std::map<std::pair<int, int>, unsigned> insideTile;
  auto contentTile =
      [&](const std::string &id) -> std::optional<std::pair<int, int>> {
    const SemanticNode &node = semanticGraph.nodes.at(id);
    if (node.kind == SemanticNodeKind::Buffer ||
        node.kind == SemanticNodeKind::Call)
      return node.tile;
    return std::nullopt;
  };
  for (const auto &[key, attributes] : topologyEdges) {
    if (auto tile = contentTile(key.source))
      ++insideTile[*tile];
    else if (auto tile = contentTile(key.dest))
      ++insideTile[*tile];
  }
  unsigned busiest = 0;
  for (const auto &[coordinate, count] : localCounts)
    busiest = std::max(busiest, count);
  for (const auto &[coordinate, count] : insideTile)
    busiest = std::max(busiest, count);
  if (std::optional<double> wanted =
          arrows->preferredTileWidth(busiest, contentWidth, layout.text(1.0)))
    tileWidth = std::max(tileWidth, *wanted);

  layout.tileWidth = tileWidth;
  layout.tileGapX = gapX;
  layout.tileGapY = gapY;
  bool compactTiles = !options.contractedNodeKinds.empty();
  double minimumTileHeight =
      compactTiles ? 0.0 : layout.text(options.minimumTileHeight);
  layout.rowHeights.assign(model.rows(), minimumTileHeight);
  double localSpacing = layout.text(options.localSpacing);
  for (const std::pair<int, int> &coordinate : designTiles) {
    auto count = localCounts.find(coordinate);
    double height =
        std::max(minimumTileHeight,
                 headroom(coordinate) + layout.text(0.45) +
                     (count == localCounts.end()
                          ? 0.0
                          : static_cast<double>(count->second) * localSpacing));
    layout.ownHeights[coordinate] = height;
    layout.rowHeights[coordinate.second] =
        std::max(layout.rowHeights[coordinate.second], height);
  }
  layout.rowCenters.assign(model.rows(), 0.0);
  for (int row = 1; row < model.rows(); ++row)
    layout.rowCenters[row] = layout.rowCenters[row - 1] +
                             layout.rowHeights[row - 1] / 2 + gapY +
                             layout.rowHeights[row] / 2;
  auto localNodeY = [&](int col, int row, unsigned index) {
    return layout.tileTop(col, row) - headroom({col, row}) -
           index * localSpacing;
  };
  // Every drawn edge becomes an arrow, and every node it can end on says
  // where it may be met. Arrows are laid out together: separating the ones
  // that share a channel, and ordering them to cross less, are properties of
  // the whole set.
  //
  // A tile's ports are numbered here rather than placed, so that a renderer
  // free to move them can stack them in a fixed order instead of scattering
  // them around the perimeter by whichever bundle each faces.
  std::map<std::pair<std::pair<int, int>, AIE::Facing>, int> switchSlots;
  std::map<std::pair<int, int>, int> dmaSlots;
  std::map<std::string, AIE::ArrowEnd> ends;
  for (const auto &[id, node] : semanticGraph.nodes) {
    if (!node.port || !topologyNodes.count(id))
      continue;
    const PortNode &port = *node.port;
    std::pair<int, int> tile{port.col, port.row};
    bool dma = port.bundle == AIE::WireBundle::DMA;
    AIE::Facing facing = AIE::Facing::Local;
    switch (port.bundle) {
    case AIE::WireBundle::North:
      facing = AIE::Facing::North;
      break;
    case AIE::WireBundle::South:
      facing = AIE::Facing::South;
      break;
    case AIE::WireBundle::East:
      facing = AIE::Facing::East;
      break;
    case AIE::WireBundle::West:
      facing = AIE::Facing::West;
      break;
    default:
      break;
    }
    ends[id] = {id,
                tile,
                dma ? AIE::EndKind::DMA : AIE::EndKind::Switch,
                dma ? dmaSlots[tile]++ : switchSlots[{tile, facing}]++,
                portPosition(port, layout),
                dma ? 0.41 : 0.045,
                dma ? 0.09 : 0.045,
                facing};
  }

  // The boxes fill the tile, leaving the arrows inside it a lane either side.
  double localNodeWidth = std::max(
      layout.text(0.8), std::min(contentWidth, tileWidth - layout.text(0.4)));

  auto argumentPoint = [&, localNodeWidth](size_t index) -> Point {
    std::set<int> designColumns;
    for (auto [col, row] : designTiles)
      designColumns.insert(col);
    double designLeft = layout.tileCenterX(*designColumns.begin());
    double designRight = layout.tileCenterX(*designColumns.rbegin());
    double center = (designLeft + designRight) / 2;
    double spacing = localNodeWidth + layout.text(0.3);
    if (runtimeArguments.size() > 1)
      spacing = std::max(spacing, (designRight - designLeft) /
                                      (runtimeArguments.size() - 1));
    double left = center - spacing * (runtimeArguments.size() - 1) / 2;
    // Far enough below the array to leave a band the runs reaching an
    // argument can travel along without touching the arguments themselves.
    return {left + index * spacing, layout.channelY(0) - layout.tileGapY * 1.3};
  };

  std::map<Operation *, unsigned> bufferIDs;
  for (const BufferInfo &buffer : *buffers)
    if (bufferLocalIndices.count(buffer.op))
      bufferIDs[buffer.op] = buffer.id;

  auto localNodePoint = [&](int col, int row, unsigned index) -> Point {
    return {layout.tileCenterX(col), localNodeY(col, row, index)};
  };

  for (const BufferInfo &buffer : *buffers) {
    auto localIndex = bufferLocalIndices.find(buffer.op);
    if (localIndex == bufferLocalIndices.end())
      continue;
    std::string id = bufferNodeID(buffer.id);
    ends[id] = {id,
                {buffer.col, buffer.row},
                AIE::EndKind::Content,
                0,
                localNodePoint(buffer.col, buffer.row, localIndex->second),
                localNodeWidth / 2,
                0.17};
  }
  for (const CallInfo &call : calls) {
    auto localIndex = callLocalIndices.find(call.id);
    if (localIndex == callLocalIndices.end())
      continue;
    std::string id = callNodeID(call.id);
    ends[id] = {id,
                {call.col, call.row},
                AIE::EndKind::Content,
                1,
                localNodePoint(call.col, call.row, localIndex->second),
                localNodeWidth / 2,
                0.15};
  }
  for (auto [index, argument] : llvm::enumerate(runtimeArguments)) {
    std::string id = runtimeArgumentNodeID(argument.id);
    ends[id] = {id,
                {0, 0},
                AIE::EndKind::External,
                0,
                argumentPoint(index),
                localNodeWidth / 2,
                0.15};
  }
  // A flow with no stated source or destination is anchored to the tile it
  // leaves from or arrives at, so it has somewhere to come from rather than
  // appearing out of the air.
  for (const auto &[id, node] : semanticGraph.nodes) {
    if (node.kind != SemanticNodeKind::Tile || !node.tile)
      continue;
    auto [col, row] = *node.tile;
    ends[id] = {id,
                *node.tile,
                AIE::EndKind::Content,
                2,
                {layout.tileCenterX(col), layout.tileCenterY(col, row)},
                localNodeWidth / 2,
                0.17};
  }

  SmallVector<ArrowRequest> arrowRequests;
  std::map<EdgeKey, unsigned> edgeArrows;
  for (const auto &[key, attributes] : topologyEdges) {
    auto from = ends.find(key.source);
    auto to = ends.find(key.dest);
    if (from == ends.end() || to == ends.end())
      continue;
    unsigned id = arrowRequests.size();
    Boundary boundary = boundaryOfEdge(key);
    arrowRequests.push_back({id, from->second, to->second, boundary.source,
                             weightScale.width(edgeBytes(boundary))});
    edgeArrows[key] = id;
  }

  std::map<std::string, Point> placedNodes =
      arrows->nodePositions(arrowRequests, layout);
  auto portPoint = [&](const PortNode &port) {
    auto found = placedNodes.find(portNodeID(port));
    return found == placedNodes.end() ? portPosition(port, layout)
                                      : found->second;
  };
  auto semanticPortPoint = [&](StringRef id, const PortNode &port) {
    auto found = placedNodes.find(id.str());
    return found == placedNodes.end() ? portPosition(port, layout)
                                      : found->second;
  };

  double legendX = layout.tileCenterX(0) - options.tileWidth / 2;
  double legendY = (runtimeArguments.empty() ? layout.channelY(0)
                                             : argumentPoint(0).second) -
                   1.0;
  std::vector<uint64_t> legendSamples;
  if (weightScale.enabled) {
    legendSamples = {weightScale.lightest, weightScale.heaviest};
    if (weightScale.heaviest > weightScale.lightest)
      legendSamples.insert(legendSamples.begin() + 1,
                           static_cast<uint64_t>(std::sqrt(
                               static_cast<double>(weightScale.lightest) *
                               static_cast<double>(weightScale.heaviest))));
    legendSamples.erase(llvm::unique(legendSamples), legendSamples.end());
  }

  struct ArbitrationAnnotation {
    const ArbitrationRule *rule;
    EdgeKey edge;
    Point center;
    std::string label;
  };
  SmallVector<ArbitrationAnnotation> arbitrationAnnotations;
  if (drawsPorts) {
    std::map<Segment, unsigned> segmentAnnotationCounts;
    for (const ArbitrationRule &rule : arbitrationRules) {
      auto edge = topologyEdges.end();
      for (const auto &[ingress, egress] :
           claimedSegments(portPackets, rule.segment, rule.mask, rule.value,
                           options.splitPacketPorts)) {
        edge = topologyEdges.find({portNodeID(ingress), portNodeID(egress)});
        if (edge != topologyEdges.end())
          break;
      }
      if (edge == topologyEdges.end())
        continue;
      bool usedByAnyRoute = false;
      for (unsigned routeID : edge->second.routes) {
        const FlowRoute &route = (*routes)[routeID];
        if (!route.packetID)
          continue;
        bool usesRule = false;
        for (unsigned packetID = 0; packetID <= 255; ++packetID) {
          bool claimedByFlow =
              route.packetMask
                  ? (packetID & *route.packetMask) == *route.packetID
                  : packetID == *route.packetID;
          if (!claimedByFlow)
            continue;
          for (const ArbitrationRule &candidate : arbitrationRules) {
            if (!(candidate.segment.source == rule.segment.source) ||
                candidate.priority > rule.priority)
              continue;
            if ((packetID & candidate.mask) != candidate.value)
              continue;
            usesRule = candidate.priority == rule.priority;
            break;
          }
          if (usesRule)
            break;
        }
        if (usesRule)
          usedByAnyRoute = true;
      }
      if (!usedByAnyRoute)
        continue;

      auto [sourceX, sourceY] = portPoint(rule.segment.source);
      auto [destX, destY] = portPoint(rule.segment.dest);
      unsigned offsetIndex = segmentAnnotationCounts[rule.segment]++;
      Point center{(sourceX + destX) / 2, (sourceY + destY) / 2};
      double deltaX = destX - sourceX;
      double deltaY = destY - sourceY;
      double length = std::hypot(deltaX, deltaY);
      if (length > 0) {
        double side = offsetIndex % 2 == 0 ? 1.0 : -1.0;
        double offset = side * (0.8 + 0.28 * (offsetIndex / 2));
        center.first -= deltaY / length * offset;
        center.second += deltaX / length * offset;
      }
      std::string label = "r" + std::to_string(rule.priority) +
                          " pkt=" + std::to_string(rule.value) + "/" +
                          std::to_string(rule.mask) + " -> A" +
                          std::to_string(rule.arbiter) + ".M" +
                          std::to_string(rule.msel);
      if (rule.keepsPacketHeader)
        label += " header";
      if (rule.priorityRoute)
        label += " priority";
      if (rule.controlOverlay)
        label += " ctrl-overlay";
      arbitrationAnnotations.push_back({&rule, edge->first, center, label});
    }
  }

  SmallVector<AIE::RoutingObstacle> obstacles;
  for (auto [col, row] : designTiles) {
    double centerX = layout.tileCenterX(col);
    double centerY = layout.tileCenterY(col, row);
    double halfWidth = layout.tileWidth / 2;
    double halfHeight = layout.tileHeight(col, row) / 2;
    obstacles.push_back({tileNodeID(col, row),
                         std::pair{col, row},
                         {centerX - halfWidth, centerX + halfWidth,
                          centerY - halfHeight, centerY + halfHeight},
                         AIE::ObstacleKind::Tile});
  }
  for (const auto &[id, end] : ends) {
    if (!topologyNodes.count(id) ||
        semanticGraph.nodes.at(id).kind == SemanticNodeKind::Tile)
      continue;
    auto placed = placedNodes.find(id);
    Point center = placed == placedNodes.end() ? end.pinned : placed->second;
    double halfWidth = end.halfWidth;
    double halfHeight = end.halfHeight;
    SemanticNodeKind kind = semanticGraph.nodes.at(id).kind;
    if (kind == SemanticNodeKind::DMA) {
      halfWidth = layout.text(0.41);
      halfHeight = layout.text(0.09);
    } else if (kind == SemanticNodeKind::Buffer) {
      halfHeight = layout.text(0.17);
    } else if (kind == SemanticNodeKind::Call) {
      halfHeight = layout.text(0.15);
    } else if (kind == SemanticNodeKind::RuntimeArgument) {
      halfHeight = layout.text(0.26);
    }
    obstacles.push_back(
        {id,
         end.kind == AIE::EndKind::External
             ? std::nullopt
             : std::optional<std::pair<int, int>>(end.tile),
         {center.first - halfWidth, center.first + halfWidth,
          center.second - halfHeight, center.second + halfHeight},
         AIE::ObstacleKind::Node});
  }
  for (auto [index, annotation] : llvm::enumerate(arbitrationAnnotations)) {
    double halfWidth = layout.text(
        0.04 + 0.025 * static_cast<double>(annotation.label.size()));
    double halfHeight = layout.text(0.08);
    obstacles.push_back({"arbitration_" + std::to_string(index),
                         std::pair{annotation.rule->segment.source.col,
                                   annotation.rule->segment.source.row},
                         {annotation.center.first - halfWidth,
                          annotation.center.first + halfWidth,
                          annotation.center.second - halfHeight,
                          annotation.center.second + halfHeight},
                         AIE::ObstacleKind::Node});
  }
  if (!legendSamples.empty()) {
    double legendBottom = legendY -
                          0.34 * static_cast<double>(legendSamples.size() - 1) -
                          layout.text(0.12);
    obstacles.push_back(
        {"legend",
         std::nullopt,
         {legendX - layout.text(0.1), legendX + 1.55 + layout.text(0.45),
          legendBottom, legendY + 0.45 + layout.text(0.12)},
         AIE::ObstacleKind::Node});
  }
  std::map<unsigned, ArrowPath> arrowPaths;
  for (ArrowPath &path : arrows->route(arrowRequests, obstacles, layout))
    arrowPaths[path.id] = std::move(path);
  // Graphviz draws no arrowhead on a supplied spline unless it is told where
  // the tip goes. Only a hop that arrives somewhere is told: a head on every
  // switchbox a flow passes through suggests it ends at each of them.
  auto writeArrowGeometry = [&](unsigned id, bool tipped) {
    auto found = arrowPaths.find(id);
    if (found == arrowPaths.end())
      return;
    SmallVector<Point> points(found->second.points);
    output << ", pos=\"";
    if (tipped && points.size() >= 4) {
      // The tip is where the head's point goes; the curve has to stop at its
      // base, or graphviz has no direction to turn the head to and draws it
      // back along the line.
      Point tip = points.back();
      Point before = points[points.size() - 4];
      double dx = tip.first - before.first, dy = tip.second - before.second;
      double length = std::hypot(dx, dy);
      if (length > 1e-6) {
        double back = std::min(kArrowLength, length * 0.9);
        Point base{tip.first - dx / length * back,
                   tip.second - dy / length * back};
        points[points.size() - 3] = {
            before.first + (base.first - before.first) / 3,
            before.second + (base.second - before.second) / 3};
        points[points.size() - 2] = {
            before.first + 2 * (base.first - before.first) / 3,
            before.second + 2 * (base.second - before.second) / 3};
        points.back() = base;
      }
      output << "e," << layout.pos(tip.first, tip.second) << ' ';
    }
    for (auto [index, point] : llvm::enumerate(points)) {
      if (index)
        output << ' ';
      output << layout.pos(point.first, point.second);
    }
    output << '"';
  };
  output << "digraph aie_routes {\n"
         << (layout.preLaidOut ? "  graph [layout=nop2, overlap=true"
                               : "  graph [layout=neato, overlap=true")
         << (layout.preLaidOut ? "" : ", splines=curved")
         << ", outputorder=nodesfirst, bgcolor=\""
         << palette.background
         // The widest a line may be drawn before it reaches the one beside
         // it, for a viewer offering to draw them all alike.
         << "\", class=\"uniform-width-" << llvm::format("%.2f", widest)
         << "\", pad=\"0.45\"];\n"
         << "  node [fontname=\"Helvetica\"];\n"
         << "  edge [fontname=\"Helvetica\", fontsize=9, arrowsize=0.65];\n";
  if (weightScale.enabled) {
    output << "  legend_title [shape=plaintext, pos=\""
           << layout.pos(legendX + 0.75, legendY + 0.45)
           << "\", label=\"bytes moved per iteration\", fontsize=10, "
              "fontcolor=\""
           << palette.legendText << "\", class=\"legend\"];\n";
    for (auto [index, bytes] : llvm::enumerate(legendSamples)) {
      double y = legendY - static_cast<double>(index) * 0.34;
      output << "  legend_" << index << "_from [shape=point, width=0.01, "
             << R"(color="none", pos=")" << layout.pos(legendX, y)
             << "\", class=\"legend\"];\n"
             << "  legend_" << index << "_to [shape=point, width=0.01, "
             << R"(color="none", pos=")" << layout.pos(legendX + 1.1, y)
             << "\", class=\"legend\"];\n"
             << "  legend_" << index << "_text [shape=plaintext, pos=\""
             << layout.pos(legendX + 1.55, y) << "\", label=\""
             << formatBytes(bytes) << "\", fontsize=9, fontcolor=\""
             << palette.legendText << "\", class=\"legend\"];\n"
             << "  legend_" << index << "_from -> legend_" << index
             << "_to [penwidth=\""
             << llvm::format("%.2f", weightScale.width(bytes)) << "\", color=\""
             << palette.legendRule;
      if (layout.preLaidOut) {
        Point from{legendX, y};
        Point to{legendX + 1.1, y};
        output << "\", pos=\"" << layout.pos(from.first, from.second) << ' '
               << layout.pos(from.first + 1.1 / 3.0, from.second) << ' '
               << layout.pos(from.first + 2.2 / 3.0, from.second) << ' '
               << layout.pos(to.first, to.second);
      }
      output << "\", arrowhead=none, class=\"legend\"];\n";
    }
  }
  for (int col = 0; col < model.columns(); ++col) {
    for (int row = 0; row < model.rows(); ++row) {
      if (!designTiles.count({col, row}))
        continue;
      StringRef fill = model.isCoreTile(col, row)      ? palette.coreTile
                       : model.isMemTile(col, row)     ? palette.memTile
                       : model.isShimNOCTile(col, row) ? palette.shimTile
                                                       : palette.otherTile;
      // A role fill replaces the tile-kind fill, so that color carries one
      // meaning at a time. Only repeated roles take one; a tile left in its
      // kind color is one of a kind.
      auto roleClass = roleClasses.find({col, row});
      if (options.tileRoles && roleClass != roleClasses.end())
        fill = roleFill(palette, roleClass->second.id);
      std::string id = tileNodeID(col, row);
      output << "  " << id << " [shape=box, id=\"" << id << "\"";
      output << ", fixedsize=true, width=" << layout.size(tileWidth)
             << ", height=" << layout.size(layout.tileHeight(col, row))
             << ", pos=\""
             << layout.pos(layout.tileCenterX(col),
                           layout.tileCenterY(col, row))
             << '"';
      output << ", label=\"" << (options.tileRoles ? "\\n" : "") << '(' << col
             << ',' << row << ')';
      if (options.tileRoles) {
        if (roleClass != roleClasses.end())
          output << "  " << roleClass->second.label;
        auto role = roles.find({col, row});
        if (role != roles.end())
          output << "\\n" << role->second.text(options.maxHeadlineKernels);
        output << R"(", labelloc="t", fontsize=11)";
      } else {
        output << '"';
      }
            output << R"(, class="semantic-node", style="rounded,filled", fillcolor=")"
              << fill
             << "\", color=\"" << palette.tileEdge << "\", fontcolor=\""
             << palette.tileText << "\", penwidth=\"2.0\"];\n";
    }
  }
  for (const auto &[id, node] : semanticGraph.nodes) {
    if (!node.port || !topologyNodes.count(id))
      continue;
    const PortNode &port = *node.port;
    auto [x, y] = semanticPortPoint(id, port);
    output << "  " << id;
    if (port.bundle == AIE::WireBundle::DMA) {
      output << " [shape=box, fixedsize=true, width="
             << layout.size(layout.text(0.82))
             << ", height=" << layout.size(layout.text(0.18)) << ", pos=\""
             << layout.pos(x, y) << "\", label=\"" << shortPortName(port)
             << "\", id=\"" << id << "\", class=\""
             << flowClasses("port", boundaryOfNode(id), membershipOfNode(id))
             << "\", fontsize=7, style=filled, fillcolor=\"" << palette.boxFill
             << "\", fontcolor=\"" << palette.tileText << "\", " << "color=\""
             << palette.boxEdge << "\"];\n";
    } else {
      std::string kind = isVia(id) ? "port detail-vias" : "port";
      output << " [shape=point, width=" << layout.size(0.09) << ", pos=\""
             << layout.pos(x, y) << "\", color=\"" << palette.boxEdge
             << "\", xlabel=\"" << shortPortName(port)
             << "\", fontsize=7, fontcolor=\"" << palette.tileText
             << "\", id=\"" << id << "\", class=\""
             << flowClasses(kind, boundaryOfNode(id), membershipOfNode(id))
             << "\"];\n";
    }
  }
  // What a switchbox does with a packet can only be anchored where the
  // switchbox hops are still drawn.
  if (drawsPorts) {
    unsigned annotationID = 0;
    std::map<Segment, unsigned> segmentAnnotationCounts;
    for (const ArbitrationRule &rule : arbitrationRules) {
      auto edge = topologyEdges.end();
      for (const auto &[ingress, egress] :
           claimedSegments(portPackets, rule.segment, rule.mask, rule.value,
                           options.splitPacketPorts)) {
        edge = topologyEdges.find({portNodeID(ingress), portNodeID(egress)});
        if (edge != topologyEdges.end())
          break;
      }
      if (edge == topologyEdges.end())
        continue;
      bool usedByAnyRoute = false;
      for (unsigned routeID : edge->second.routes) {
        const FlowRoute &route = (*routes)[routeID];
        if (!route.packetID)
          continue;
        bool usesRule = false;
        for (unsigned packetID = 0; packetID <= 255; ++packetID) {
          bool claimedByFlow =
              route.packetMask
                  ? (packetID & *route.packetMask) == *route.packetID
                  : packetID == *route.packetID;
          if (!claimedByFlow)
            continue;
          for (const ArbitrationRule &candidate : arbitrationRules) {
            if (!(candidate.segment.source == rule.segment.source) ||
                candidate.priority > rule.priority)
              continue;
            if ((packetID & candidate.mask) != candidate.value)
              continue;
            usesRule = candidate.priority == rule.priority;
            break;
          }
          if (usesRule)
            break;
        }
        if (usesRule)
          usedByAnyRoute = true;
      }
      if (!usedByAnyRoute)
        continue;
      auto [sourceX, sourceY] = portPoint(rule.segment.source);
      auto [destX, destY] = portPoint(rule.segment.dest);
      unsigned offsetIndex = segmentAnnotationCounts[rule.segment]++;
      double x = (sourceX + destX) / 2;
      double y = (sourceY + destY) / 2;
      double deltaX = destX - sourceX;
      double deltaY = destY - sourceY;
      double length = std::hypot(deltaX, deltaY);
      if (length > 0) {
        double side = offsetIndex % 2 == 0 ? 1.0 : -1.0;
        double offset = side * (0.8 + 0.28 * (offsetIndex / 2));
        x -= deltaY / length * offset;
        y += deltaX / length * offset;
      }
      output << "  arbitration_" << annotationID++
             << R"( [shape=box, fixedsize=false, margin="0.04,0.02", pos=")"
             << layout.pos(x, y) << "\", label=\"r" << rule.priority
             << " pkt=" << rule.value << '/' << rule.mask << " -> A"
             << rule.arbiter << ".M" << rule.msel;
      if (rule.keepsPacketHeader)
        output << " header";
      if (rule.priorityRoute)
        output << " priority";
      if (rule.controlOverlay)
        output << " ctrl-overlay";
      output << "\", tooltip=\"rule " << rule.priority << ": packet ID & "
             << rule.mask << " = " << rule.value << ", arbiter " << rule.arbiter
             << ", msel " << rule.msel << ", keep packet header: "
             << (rule.keepsPacketHeader ? "yes" : "no")
             << ", priority route: " << (rule.priorityRoute ? "yes" : "no")
             << ", control overlay: " << (rule.controlOverlay ? "yes" : "no")
             << "\", class=\""
             << flowClasses("arbitration detail-arbitration detail-vias",
                            boundaryOfEdge(edge->first),
                            membershipOfEdge(edge->first))
             << "\", fontsize=6, style=filled, fillcolor=\"" << palette.ruleFill
             << "\", color=\"" << palette.ruleEdge << "\", fontcolor=\""
             << palette.ruleText << "\"];\n";
    }
  }
  for (const BufferInfo &buffer : *buffers) {
    std::string id = bufferNodeID(buffer.id);
    auto localIndex = bufferLocalIndices.find(buffer.op);
    if (localIndex == bufferLocalIndices.end() || !topologyNodes.count(id))
      continue;
    auto [x, y] = localNodePoint(buffer.col, buffer.row, localIndex->second);
    output << "  " << id << " [shape=box, fixedsize=true, width="
           << layout.size(localNodeWidth)
           << ", height=" << layout.size(layout.text(0.34)) << ", pos=\""
           << layout.pos(x, y) << "\", label=\""
           << bufferLabel(buffer.op, buffer.id) << "\", id=\"" << id
           << "\", class=\""
           << flowClasses("buffer detail-buffers", boundaryOfNode(id),
                          membershipOfNode(id))
           << "\", fontsize=8, style=filled, fillcolor=\"" << palette.boxFill
           << "\", fontcolor=\"" << palette.tileText << "\", " << "color=\""
           << palette.boxEdge << "\"];\n";
  }
  for (auto [index, argument] : llvm::enumerate(runtimeArguments)) {
    std::string id = runtimeArgumentNodeID(argument.id);
    if (!topologyNodes.count(id))
      continue;
    auto [x, y] = argumentPoint(index);
    output << "  " << id << " [shape=ellipse, fixedsize=true, width="
           << layout.size(localNodeWidth)
           << ", height=" << layout.size(layout.text(0.52)) << ", pos=\""
           << layout.pos(x, y) << "\", label=\"%"
           << escapeDotLabel(argument.name) << "\", id=\"" << id
           << "\", class=\""
           << flowClasses("runtime-argument", boundaryOfNode(id),
                          membershipOfNode(id))
           << R"(", fontsize=8, style="rounded,filled", fillcolor=")"
           << palette.argumentFill << "\", color=\"" << palette.argumentEdge
           << "\", fontcolor=\"" << palette.tileText << "\"];\n";
  }
  for (const CallInfo &call : calls) {
    std::string id = callNodeID(call.id);
    auto localIndex = callLocalIndices.find(call.id);
    if (localIndex == callLocalIndices.end() || !topologyNodes.count(id))
      continue;
    auto [x, y] = localNodePoint(call.col, call.row, localIndex->second);
    output << "  " << id << " [shape=box, fixedsize=true, width="
           << layout.size(localNodeWidth)
           << ", height=" << layout.size(layout.text(0.3)) << ", pos=\""
           << layout.pos(x, y) << "\", label=\"" << escapeDotLabel(call.callee)
           << "\", id=\"" << id << "\", class=\""
           << flowClasses("call detail-calls", boundaryOfNode(id),
                          membershipOfNode(id))
           << R"(", fontsize=8, style="rounded,filled", fillcolor=")"
           << palette.callFill << "\", color=\"" << palette.callEdge
           << "\", fontcolor=\"" << palette.tileText << "\"];\n";
  }

  auto writeWeight = [&](const Boundary &boundary) {
    if (!weightScale.enabled) {
      output << ", penwidth=\"" << llvm::format("%.2f", 2.4 * options.scale)
             << '"';
      return;
    }
    uint64_t bytes = edgeBytes(boundary);
    output << ", penwidth=\"" << llvm::format("%.2f", weightScale.width(bytes))
           << '"';
    // A flow with no buffer descriptor at either end states no volume. Saying
    // "0 B" would claim it carries nothing, which is a different fact.
    if (bytes == 0)
      output << ", tooltip=\"volume not stated in the IR\"";
    else
      output << ", tooltip=\"" << formatBytes(bytes) << " per iteration\"";
  };
  auto writeColors = [&](const Boundary &boundary) {
    if (!boundary.hasSource) {
      output << palette.quietEdge;
      return;
    }
    output << flowColor(palette, boundary.source);
  };
  auto writeLabel = [&](const FlowRoute &route, const Boundary &boundary) {
    output << "<FONT COLOR=\""
           << (boundary.hasSource ? flowColor(palette, boundary.source)
                                  : palette.quietEdge)
           << "\">F" << boundary.source;
    if (route.packetID) {
      output << " pkt=" << *route.packetID;
      if (route.packetMask)
        output << '/' << *route.packetMask;
      if (route.keepsPacketHeader)
        output << (*route.keepsPacketHeader ? " header" : " strip-header");
      if (route.priorityRoute)
        output << " priority";
    }
    output << "</FONT>";
  };

  // A route is named once, on the run between tiles in the middle of its
  // path, or where it branches or joins if it does either -- that is the edge
  // a reader is asking about. Repeating the name down the whole chain says
  // nothing more.
  std::map<EdgeKey, std::set<unsigned>> labeledEdges;
  for (const FlowRoute &route : *routes) {
    std::vector<const EdgeKey *> carrying;
    std::vector<const EdgeKey *> crossing;
    for (const auto &[key, attributes] : topologyEdges) {
      if (!attributes.routes.count(route.id))
        continue;
      carrying.push_back(&key);
      const SemanticNode &source = semanticGraph.nodes.at(key.source);
      const SemanticNode &dest = semanticGraph.nodes.at(key.dest);
      if (source.tile && dest.tile && *source.tile != *dest.tile)
        crossing.push_back(&key);
    }
    if (carrying.empty())
      continue;
    ArrayRef<const EdgeKey *> candidates =
        crossing.empty() ? ArrayRef(carrying) : ArrayRef(crossing);
    const EdgeKey *chosen = candidates[candidates.size() / 2];
    for (const EdgeKey *key : candidates) {
      unsigned leaving = llvm::count_if(topologyEdges, [&](const auto &entry) {
        return entry.second.routes.count(route.id) &&
               entry.first.source == key->dest;
      });
      unsigned arriving = llvm::count_if(topologyEdges, [&](const auto &entry) {
        return entry.second.routes.count(route.id) &&
               entry.first.dest == key->source;
      });
      if (leaving > 1 || arriving > 1) {
        chosen = key;
        break;
      }
    }
    labeledEdges[*chosen].insert(route.id);
  }

  // Heaviest last, so a dominant flow is never hidden under a hairline.
  std::vector<const EdgeKey *> emissionOrder;
  emissionOrder.reserve(topologyEdges.size());
  for (const auto &[key, attributes] : topologyEdges)
    emissionOrder.push_back(&key);
  if (weightScale.enabled)
    llvm::stable_sort(emissionOrder, [&](const EdgeKey *a, const EdgeKey *b) {
      return edgeBytes(boundaryOfEdge(*a)) < edgeBytes(boundaryOfEdge(*b));
    });

  for (const EdgeKey *key : emissionOrder) {
    const EdgeAttributes &attributes = topologyEdges.at(*key);
    SemanticNodeKind sourceKind = nodeKind(key->source);
    SemanticNodeKind destKind = nodeKind(key->dest);
    // What an edge is, is what it joins.
    std::string kind;
    if (sourceKind == SemanticNodeKind::RuntimeArgument ||
        destKind == SemanticNodeKind::RuntimeArgument)
      kind = "runtime-link";
    else if (sourceKind == SemanticNodeKind::Call ||
             destKind == SemanticNodeKind::Call)
      kind = "call-link detail-calls";
    else if (sourceKind == SemanticNodeKind::Buffer ||
             destKind == SemanticNodeKind::Buffer)
      kind = "buffer-link";
    else
      kind = "route";
    // Hiding the buffers has to take the edges that end on one with them.
    if (sourceKind == SemanticNodeKind::Buffer ||
        destKind == SemanticNodeKind::Buffer)
      kind += " detail-buffers";
    auto labeled = labeledEdges.find(*key);
    if (labeled != labeledEdges.end())
      kind += " detail-packet-ids";
    if (isVia(key->dest))
      kind += " detail-vias";

    Boundary boundary = boundaryOfEdge(*key);
    output << "  " << key->source << " -> " << key->dest << " [color=\"";
    writeColors(boundary);
    output << "\", id=\"" << edgeID(*key) << "\", class=\""
          << flowClasses(kind, boundary, membershipOfEdge(*key)) << '"';
    writeWeight(boundary);
    auto arrow = edgeArrows.find(*key);
    if (arrow != edgeArrows.end())
      writeArrowGeometry(arrow->second,
                         /*tipped=*/destKind != SemanticNodeKind::Port);

    // One name per distinct packet treatment: routes that say the same thing
    // say it once, however many of them a contracted edge stands for.
    std::vector<const FlowRoute *> labelRoutes;
    std::set<std::tuple<std::optional<unsigned>, std::optional<unsigned>,
                        std::optional<bool>, bool>>
        seen;
    if (labeled != labeledEdges.end())
      for (unsigned routeID : labeled->second) {
        const FlowRoute &route = (*routes)[routeID];
        if (seen.insert({route.packetID, route.packetMask,
                         route.keepsPacketHeader, route.priorityRoute})
                .second)
          labelRoutes.push_back(&route);
      }
    bool writeCount = attributes.count > 1;
    if (!labelRoutes.empty() || writeCount) {
      output << ", label=<";
      for (auto [index, route] : llvm::enumerate(labelRoutes)) {
        if (index)
          output << "<BR/>";
        writeLabel(*route, boundary);
      }
      if (!labelRoutes.empty() && writeCount)
        output << "<BR/>";
      if (writeCount)
        output << "<FONT COLOR=\"" << palette.tileText << "\">&#215;"
               << attributes.count << "</FONT>";
      output << '>';
    }
    output << "];\n";
  }
  output << "}\n";
  return success();
}

enum class PhysicalPortRole { Source, Dest };
enum class PhysicalPortDomain { Switchbox, ShimMux };

struct PhysicalPort {
  int col;
  int row;
  AIE::WireBundle bundle;
  unsigned channel;
  PhysicalPortRole role;
  PhysicalPortDomain domain = PhysicalPortDomain::Switchbox;

  auto asTuple() const {
    return std::make_tuple(col, row, static_cast<int>(bundle), channel, role,
                           domain);
  }
  bool operator<(const PhysicalPort &other) const {
    return asTuple() < other.asTuple();
  }
};

static std::string physicalPortID(const PhysicalPort &port) {
  StringRef prefix = port.domain == PhysicalPortDomain::ShimMux
                         ? "shim_mux_port_"
                         : "physical_port_";
  return prefix.str() + std::to_string(port.col) + "_" +
         std::to_string(port.row) + "_" +
         std::to_string(static_cast<int>(port.bundle)) + "_" +
         std::to_string(port.channel) +
         (port.role == PhysicalPortRole::Source ? "_source" : "_dest");
}

static std::pair<double, double>
physicalPortPosition(const PhysicalPort &port) {
  double tileX = static_cast<double>(port.col) * 8.0;
  double tileY = static_cast<double>(port.row) * 7.0;
  double offset = (static_cast<double>(port.channel) - 2.5) * 0.34;
  double lane = port.role == PhysicalPortRole::Source ? 0.18 : -0.18;
  if (port.domain == PhysicalPortDomain::ShimMux) {
    if (port.bundle == AIE::WireBundle::South)
      return {tileX + offset,
              tileY - 2.58 +
                  (port.role == PhysicalPortRole::Source ? 0.08 : -0.08)};
    double bundleOffset = 0.0;
    if (port.bundle == AIE::WireBundle::DMA)
      bundleOffset = -0.72;
    else if (port.bundle == AIE::WireBundle::PLIO)
      bundleOffset = -0.35;
    else if (port.bundle == AIE::WireBundle::NOC)
      bundleOffset = 0.35;
    double externalOffset =
        bundleOffset + (static_cast<double>(port.channel) - 2.5) * 0.09;
    return {tileX + externalOffset,
            tileY - 3.42 +
                (port.role == PhysicalPortRole::Source ? 0.08 : -0.08)};
  }
  switch (port.bundle) {
  case AIE::WireBundle::North:
    return {tileX + offset, tileY + 2.0 + lane};
  case AIE::WireBundle::South:
    return {tileX + offset, tileY - 2.0 - lane};
  case AIE::WireBundle::East:
    return {tileX + 2.0 + lane, tileY + offset};
  case AIE::WireBundle::West:
    return {tileX - 2.0 - lane, tileY + offset};
  default:
    return {tileX + (port.role == PhysicalPortRole::Source ? -1.45 : 1.45),
            tileY + 1.45 -
                static_cast<double>(static_cast<int>(port.bundle)) * 0.20 -
                static_cast<double>(port.channel) * 0.14};
  }
}

static std::string physicalCoordinate(double value) {
  constexpr double pointsPerInch = 72.0;
  return formatv("{0:F6}", value * pointsPerInch).str();
}

static ArrayRef<AIE::WireBundle> physicalBundles() {
  static const AIE::WireBundle bundles[] = {
      AIE::WireBundle::Core,        AIE::WireBundle::DMA,
      AIE::WireBundle::FIFO,        AIE::WireBundle::South,
      AIE::WireBundle::West,        AIE::WireBundle::North,
      AIE::WireBundle::East,        AIE::WireBundle::PLIO,
      AIE::WireBundle::NOC,         AIE::WireBundle::Trace,
      AIE::WireBundle::TileControl, AIE::WireBundle::Controller32};
  return bundles;
}

static std::optional<PhysicalPortRole>
semanticPortRole(StringRef id, const SemanticNode &node,
                 const SemanticGraph &graph) {
  if (!node.port)
    return std::nullopt;
  if (node.port->bundle == AIE::WireBundle::DMA)
    return node.port->dmaDirection == AIE::DMAChannelDir::MM2S
               ? PhysicalPortRole::Source
               : PhysicalPortRole::Dest;
  for (const auto &[key, attributes] : graph.edges) {
    if (key.source != id && key.dest != id)
      continue;
    const SemanticNode &other =
        graph.nodes.at(key.source == id ? key.dest : key.source);
    if (!other.port || other.port->col != node.port->col ||
        other.port->row != node.port->row)
      continue;
    return key.source == id ? PhysicalPortRole::Source : PhysicalPortRole::Dest;
  }
  return std::nullopt;
}

static LogicalResult
emitDevicePhysicalRouteDot(AIE::DeviceOp device, raw_ostream &output,
                           AIE::RuntimeSequenceOp runtimeSequence,
                           const AIE::VisualizeOptions &options) {
  FailureOr<SemanticGraphAnalysis> analysis =
      analyzeSemanticGraph(device, runtimeSequence, options);
  if (failed(analysis))
    return failure();

  const AIE::AIETargetModel &model = device.getTargetModel();
  std::set<PhysicalPort> usedPorts;
  std::set<std::pair<PhysicalPort, PhysicalPort>> usedTracks;
  std::map<std::string, PhysicalPort> semanticPorts;
  std::set<std::tuple<int, int, unsigned>> allocatedBDs;
  for (AIE::DmaBody program : device.getOps<AIE::DmaBody>()) {
    auto tile =
        dyn_cast_or_null<AIE::TileLike>(program.getTile().getDefiningOp());
    if (!tile || !tile.tryGetCol() || !tile.tryGetRow())
      continue;
    program.getDmaBody().walk([&](AIE::DMABDOp descriptor) {
      if (auto id = descriptor.getBdId())
        allocatedBDs.insert(
            {*tile.tryGetCol(), *tile.tryGetRow(), static_cast<unsigned>(*id)});
    });
  }
  for (const auto &[id, node] : analysis->graph.nodes) {
    std::optional<PhysicalPortRole> role =
        semanticPortRole(id, node, analysis->graph);
    if (!node.port || !role)
      continue;
    AIE::WireBundle bundle = node.port->bundle;
    bool shimMuxStreamSwitchPort = false;
    if (bundle == AIE::WireBundle::North &&
        model.isShimNOCorPLTile(node.port->col, node.port->row)) {
      for (const auto &[key, attributes] : analysis->graph.edges) {
        if (key.source != id && key.dest != id)
          continue;
        const SemanticNode &other =
            analysis->graph.nodes.at(key.source == id ? key.dest : key.source);
        if (other.port && other.port->col == node.port->col &&
            other.port->row == node.port->row &&
            llvm::is_contained({AIE::WireBundle::DMA, AIE::WireBundle::NOC,
                                AIE::WireBundle::PLIO},
                               other.port->bundle)) {
          shimMuxStreamSwitchPort = true;
          break;
        }
      }
    }
    if (shimMuxStreamSwitchPort) {
      bundle = AIE::WireBundle::South;
      *role = *role == PhysicalPortRole::Source ? PhysicalPortRole::Dest
                                                : PhysicalPortRole::Source;
    }
    PhysicalPort port{node.port->col, node.port->row, bundle,
                      static_cast<unsigned>(node.port->channel), *role};
    if (model.isShimNOCorPLTile(port.col, port.row) &&
        llvm::is_contained(
            {AIE::WireBundle::DMA, AIE::WireBundle::NOC, AIE::WireBundle::PLIO},
            port.bundle))
      port.domain = PhysicalPortDomain::ShimMux;
    semanticPorts[id] = port;
    usedPorts.insert(port);
  }
  for (const auto &[key, attributes] : analysis->graph.edges) {
    auto source = semanticPorts.find(key.source);
    auto dest = semanticPorts.find(key.dest);
    if (source == semanticPorts.end() || dest == semanticPorts.end())
      continue;
    const PhysicalPort &sourcePort = source->second;
    const PhysicalPort &destPort = dest->second;
    bool shimVia = sourcePort.col == destPort.col &&
                   sourcePort.row == destPort.row &&
                   model.isShimNOCorPLTile(sourcePort.col, sourcePort.row);
    if (shimVia && sourcePort.domain == PhysicalPortDomain::ShimMux &&
        destPort.bundle == AIE::WireBundle::South) {
      PhysicalPort muxSouth{
          sourcePort.col,         sourcePort.row,
          AIE::WireBundle::South, destPort.channel,
          PhysicalPortRole::Dest, PhysicalPortDomain::ShimMux};
      PhysicalPort switchboxSouth{sourcePort.col, sourcePort.row,
                                  AIE::WireBundle::South, destPort.channel,
                                  PhysicalPortRole::Source};
      usedPorts.insert(muxSouth);
      usedPorts.insert(switchboxSouth);
      usedTracks.insert({sourcePort, muxSouth});
      usedTracks.insert({muxSouth, switchboxSouth});
    } else if (shimVia && sourcePort.bundle == AIE::WireBundle::South &&
               destPort.domain == PhysicalPortDomain::ShimMux) {
      PhysicalPort switchboxSouth{sourcePort.col, sourcePort.row,
                                  AIE::WireBundle::South, sourcePort.channel,
                                  PhysicalPortRole::Dest};
      PhysicalPort muxSouth{
          sourcePort.col,           sourcePort.row,
          AIE::WireBundle::South,   sourcePort.channel,
          PhysicalPortRole::Source, PhysicalPortDomain::ShimMux};
      usedPorts.insert(switchboxSouth);
      usedPorts.insert(muxSouth);
      usedTracks.insert({switchboxSouth, muxSouth});
      usedTracks.insert({muxSouth, destPort});
    } else {
      usedTracks.insert({sourcePort, destPort});
    }
  }

  output << "digraph aie_physical {\n"
            "  graph [layout=neato, overlap=true, splines=line, inputscale=72, "
            "outputorder=edgesfirst, bgcolor=white];\n"
            "  node [fontname=Helvetica, fontsize=7, pin=true];\n"
            "  edge [arrowsize=0.35];\n";
  for (int row = 0; row < model.rows(); ++row) {
    for (int col = 0; col < model.columns(); ++col) {
      double tileX = static_cast<double>(col) * 8.0;
      double tileY = static_cast<double>(row) * 7.0;
      std::string coordinate = std::to_string(col) + "_" + std::to_string(row);
      output << "  switchbox_" << coordinate
             << " [shape=box, width=4.0, height=4.0, fixedsize=true, "
                "label=\"switchbox ("
             << col << ',' << row << ")\", color=\"#b8bec4\", "
             << "fontcolor=\"#687078\", pos=\"" << physicalCoordinate(tileX)
             << ',' << physicalCoordinate(tileY)
             << "!\", class=\"physical-switchbox\"];\n";

      std::vector<PhysicalPort> ports;
      for (AIE::WireBundle bundle : physicalBundles()) {
        for (unsigned channel = 0;
             channel < model.getNumSourceSwitchboxConnections(col, row, bundle);
             ++channel)
          ports.push_back(
              {col, row, bundle, channel, PhysicalPortRole::Source});
        for (unsigned channel = 0;
             channel < model.getNumDestSwitchboxConnections(col, row, bundle);
             ++channel)
          ports.push_back({col, row, bundle, channel, PhysicalPortRole::Dest});
      }
      for (const PhysicalPort &port : ports) {
        auto [x, y] = physicalPortPosition(port);
        bool used = usedPorts.count(port);
        output << "  " << physicalPortID(port)
               << " [shape=point, width=" << (used ? "0.09" : "0.045")
               << ", color=\"" << (used ? "#c51b32" : "#d9dde1") << "\", pos=\""
               << physicalCoordinate(x) << ',' << physicalCoordinate(y)
               << "!\", xlabel=\""
               << shortPortName({col, row, port.bundle,
                                 static_cast<int>(port.channel),
                                 port.bundle == AIE::WireBundle::DMA
                                     ? std::optional<AIE::DMAChannelDir>(
                                           port.role == PhysicalPortRole::Source
                                               ? AIE::DMAChannelDir::MM2S
                                               : AIE::DMAChannelDir::S2MM)
                                     : std::nullopt})
               << (port.role == PhysicalPortRole::Source ? " in" : " out")
               << "\", class=\"physical-port " << (used ? "used" : "unused")
               << "\"];\n";
      }

      for (const PhysicalPort &source : ports) {
        if (source.role != PhysicalPortRole::Source)
          continue;
        for (const PhysicalPort &dest : ports) {
          if (dest.role != PhysicalPortRole::Dest ||
              !model.isLegalTileConnection(col, row, source.bundle,
                                           source.channel, dest.bundle,
                                           dest.channel))
            continue;
          bool used = usedTracks.count({source, dest});
          output << "  " << physicalPortID(source) << " -> "
                 << physicalPortID(dest) << " [color=\""
                 << (used ? "#c51b32" : "#e7eaed")
                 << "\", penwidth=" << (used ? "1.6" : "0.18")
                 << ", class=\"physical-track " << (used ? "used" : "unused")
                 << "\"];\n";
        }
      }

      std::vector<PhysicalPort> shimMuxPorts;
      for (AIE::WireBundle bundle : physicalBundles()) {
        for (unsigned channel = 0;
             channel < model.getNumSourceShimMuxConnections(col, row, bundle);
             ++channel)
          shimMuxPorts.push_back({col, row, bundle, channel,
                                  PhysicalPortRole::Source,
                                  PhysicalPortDomain::ShimMux});
        for (unsigned channel = 0;
             channel < model.getNumDestShimMuxConnections(col, row, bundle);
             ++channel)
          shimMuxPorts.push_back({col, row, bundle, channel,
                                  PhysicalPortRole::Dest,
                                  PhysicalPortDomain::ShimMux});
      }
      for (const PhysicalPort &port : shimMuxPorts) {
        auto [x, y] = physicalPortPosition(port);
        bool used = usedPorts.count(port);
        output << "  " << physicalPortID(port)
               << " [shape=point, width=" << (used ? "0.09" : "0.045")
               << ", color=\"" << (used ? "#c51b32" : "#d9dde1") << "\", pos=\""
               << physicalCoordinate(x) << ',' << physicalCoordinate(y)
               << "!\", xlabel=\""
               << shortPortName({col, row, port.bundle,
                                 static_cast<int>(port.channel),
                                 port.bundle == AIE::WireBundle::DMA
                                     ? std::optional<AIE::DMAChannelDir>(
                                           port.role == PhysicalPortRole::Source
                                               ? AIE::DMAChannelDir::MM2S
                                               : AIE::DMAChannelDir::S2MM)
                                     : std::nullopt})
               << (port.role == PhysicalPortRole::Source ? " in" : " out")
               << "\", class=\"physical-shim-mux-port "
               << (used ? "used" : "unused") << "\"];\n";
      }
      for (const PhysicalPort &source : shimMuxPorts) {
        if (source.role != PhysicalPortRole::Source)
          continue;
        for (const PhysicalPort &dest : shimMuxPorts) {
          if (dest.role != PhysicalPortRole::Dest ||
              !model.isLegalShimMuxConnection(col, row, source.bundle,
                                              source.channel, dest.bundle,
                                              dest.channel))
            continue;
          bool used = usedTracks.count({source, dest});
          output << "  " << physicalPortID(source) << " -> "
                 << physicalPortID(dest) << " [color=\""
                 << (used ? "#c51b32" : "#e7eaed")
                 << "\", penwidth=" << (used ? "1.6" : "0.18")
                 << ", class=\"physical-shim-mux-track "
                 << (used ? "used" : "unused") << "\"];\n";
        }
      }
      for (const PhysicalPort &muxPort : shimMuxPorts) {
        if (muxPort.bundle != AIE::WireBundle::South)
          continue;
        PhysicalPort switchboxPort{col, row, AIE::WireBundle::South,
                                   muxPort.channel,
                                   muxPort.role == PhysicalPortRole::Source
                                       ? PhysicalPortRole::Dest
                                       : PhysicalPortRole::Source};
        bool used = muxPort.role == PhysicalPortRole::Source
                        ? usedTracks.count({switchboxPort, muxPort})
                        : usedTracks.count({muxPort, switchboxPort});
        output << "  "
               << physicalPortID(muxPort.role == PhysicalPortRole::Source
                                     ? switchboxPort
                                     : muxPort)
               << " -> "
               << physicalPortID(muxPort.role == PhysicalPortRole::Source
                                     ? muxPort
                                     : switchboxPort)
               << " [color=\"" << (used ? "#c51b32" : "#e1e5e8")
               << "\", penwidth=" << (used ? "1.6" : "0.22")
               << ", class=\"physical-shim-mux-link "
               << (used ? "used" : "unused") << "\"];\n";
      }

      unsigned numBDs = model.getNumBDs(col, row);
      unsigned dmaSources = model.getNumSourceSwitchboxConnections(
          col, row, AIE::WireBundle::DMA);
      unsigned dmaDests =
          model.getNumDestSwitchboxConnections(col, row, AIE::WireBundle::DMA);
      unsigned shimDmaSources =
          model.getNumSourceShimMuxConnections(col, row, AIE::WireBundle::DMA);
      unsigned shimDmaDests =
          model.getNumDestShimMuxConnections(col, row, AIE::WireBundle::DMA);
      bool hasDMA = dmaSources || dmaDests || shimDmaSources || shimDmaDests;
      if (hasDMA) {
        output << "  dma_box_" << coordinate
               << " [shape=box, width=2.7, height=4.8, fixedsize=true, "
                  "label=\"\", color=\"#cdd2d7\", "
                  "fontcolor=\"#7b838a\", pos=\""
               << physicalCoordinate(tileX + 3.8) << ','
               << physicalCoordinate(tileY) << "!\", "
               << "class=\"physical-dma-box\"];\n";
        output
            << "  dma_title_" << coordinate
            << " [shape=plain, label=\"DMA resources\", fontcolor=\"#7b838a\", "
               "pos=\""
            << physicalCoordinate(tileX + 3.8) << ','
            << physicalCoordinate(tileY + 2.28)
            << "!\", class=\"physical-dma-title\"];\n";
        unsigned mm2sChannels = std::max(dmaSources, shimDmaSources);
        unsigned s2mmChannels = std::max(dmaDests, shimDmaDests);
        for (unsigned channel = 0; channel < mm2sChannels; ++channel) {
          PhysicalPort port{col, row, AIE::WireBundle::DMA, channel,
                            PhysicalPortRole::Source};
          if (shimDmaSources)
            port.domain = PhysicalPortDomain::ShimMux;
          bool used = usedPorts.count(port);
          output << "  dma_mm2s_" << coordinate << '_' << channel
                 << " [shape=box, width=0.55, height=0.2, fixedsize=true, "
                    "label=\"M"
                 << channel << "\", color=\"" << (used ? "#c51b32" : "#e1e4e7")
                 << "\", pos=\"" << physicalCoordinate(tileX + 3.15) << ','
                 << physicalCoordinate(tileY + 0.9 - channel * 0.3)
                 << "!\", class=\"physical-dma-channel "
                 << (used ? "used" : "unused") << "\"];\n";
        }
        for (unsigned channel = 0; channel < s2mmChannels; ++channel) {
          PhysicalPort port{col, row, AIE::WireBundle::DMA, channel,
                            PhysicalPortRole::Dest};
          if (shimDmaDests)
            port.domain = PhysicalPortDomain::ShimMux;
          bool used = usedPorts.count(port);
          output << "  dma_s2mm_" << coordinate << '_' << channel
                 << " [shape=box, width=0.55, height=0.2, fixedsize=true, "
                    "label=\"S"
                 << channel << "\", color=\"" << (used ? "#c51b32" : "#e1e4e7")
                 << "\", pos=\"" << physicalCoordinate(tileX + 4.45) << ','
                 << physicalCoordinate(tileY + 0.9 - channel * 0.3)
                 << "!\", class=\"physical-dma-channel "
                 << (used ? "used" : "unused") << "\"];\n";
        }
        std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>> bdGroups;
        for (unsigned channel = 0;
             channel < std::max(mm2sChannels, s2mmChannels); ++channel)
          bdGroups[model.getBdIdRangeForChannel(col, row, channel)].push_back(
              channel);
        unsigned bdGroupIndex = 0;
        for (const auto &[range, channels] : bdGroups) {
          if (range.first == range.second)
            continue;
          output << "  bd_group_" << coordinate << '_' << range.first << '_'
                 << range.second << " [shape=plain, label=\"channels ";
          llvm::interleaveComma(channels, output);
          output << ": BD" << range.first << "-" << range.second - 1
                 << "\", fontcolor=\"#7b838a\", pos=\""
                 << physicalCoordinate(tileX + 3.8) << ','
                 << physicalCoordinate(tileY + 2.05 - bdGroupIndex++ * 0.22)
                 << "!\", class=\"physical-bd-group\"];\n";
        }
        for (unsigned bd = 0; bd < numBDs; ++bd) {
          double x = tileX + 2.75 + static_cast<double>(bd % 4) * 0.68;
          double y = tileY + 1.65 - static_cast<double>(bd / 4) * 0.30;
          bool allocated = allocatedBDs.count({col, row, bd});
          output << "  bd_" << coordinate << '_' << bd
                 << " [shape=box, width=0.45, height=0.18, fixedsize=true, "
                    "label=\"BD"
                 << bd << "\", color=\"" << (allocated ? "#c51b32" : "#e1e4e7")
                 << "\", fontcolor=\"" << (allocated ? "#7e1422" : "#b0b5ba")
                 << "\", " << "pos=\"" << physicalCoordinate(x) << ','
                 << physicalCoordinate(y) << "!\", class=\"physical-bd "
                 << (allocated ? "used" : "unused") << "\"];\n";
        }
      }
      if (shimDmaSources || shimDmaDests) {
        output << "  shim_mux_" << coordinate
               << " [shape=box, width=2.2, height=1.0, fixedsize=true, "
                  "label=\"shim mux\", color=\"#cdd2d7\", pos=\""
               << physicalCoordinate(tileX) << ','
               << physicalCoordinate(tileY - 3.0)
               << "!\", class=\"physical-shim-mux\"];\n";
      }
    }
  }

  for (int row = 0; row < model.rows(); ++row) {
    for (int col = 0; col < model.columns(); ++col) {
      for (AIE::WireBundle outputBundle :
           {AIE::WireBundle::North, AIE::WireBundle::East}) {
        unsigned outputs =
            model.getNumDestSwitchboxConnections(col, row, outputBundle);
        for (unsigned channel = 0; channel < outputs; ++channel) {
          std::optional<std::pair<AIE::TileID, AIE::Port>> endpoint =
              AIE::getConnectingWireEndpoint(
                  model, AIE::TileID{col, row},
                  AIE::Port{outputBundle, static_cast<int>(channel)});
          if (!endpoint)
            continue;
          if (endpoint->second.channel >=
              static_cast<int>(model.getNumSourceSwitchboxConnections(
                  endpoint->first.col, endpoint->first.row,
                  endpoint->second.bundle)))
            continue;
          PhysicalPort source{col, row, outputBundle, channel,
                              PhysicalPortRole::Dest};
          PhysicalPort dest{endpoint->first.col, endpoint->first.row,
                            endpoint->second.bundle,
                            static_cast<unsigned>(endpoint->second.channel),
                            PhysicalPortRole::Source};
          bool used = usedTracks.count({source, dest});
          output << "  " << physicalPortID(source) << " -> "
                 << physicalPortID(dest) << " [color=\""
                 << (used ? "#c51b32" : "#e1e5e8")
                 << "\", penwidth=" << (used ? "1.6" : "0.22")
                 << ", class=\"physical-intertile-track "
                 << (used ? "used" : "unused") << "\"];\n";
        }
      }
    }
  }

  for (const auto &[id, node] : analysis->graph.nodes) {
    std::optional<std::pair<double, double>> position;
    auto port = semanticPorts.find(id);
    if (port != semanticPorts.end()) {
      position = physicalPortPosition(port->second);
    } else if (node.tile) {
      double tileX = static_cast<double>(node.tile->first) * 8.0;
      double tileY = static_cast<double>(node.tile->second) * 7.0;
      position = node.kind == SemanticNodeKind::DMA
                     ? std::make_pair(tileX + 3.8, tileY)
                     : std::make_pair(tileX, tileY);
    }
    if (!position)
      continue;
    output << "  physical_alias_" << id
           << " [shape=circle, width=0.13, height=0.13, fixedsize=true, "
              "label=\"\", color=\"#c51b32\", fillcolor=\"#c51b32\", "
              "style=filled, pos=\""
           << physicalCoordinate(position->first) << ','
           << physicalCoordinate(position->second)
           << "!\", id=\"physical_alias_" << id
           << "\", class=\"semantic-node physical-alias semantic-id-" << id
           << "\", tooltip=\""
           << escapeDotLabel(id) << "\"];\n";
  }
  output << "}\n";
  return success();
}

} // namespace

static FailureOr<std::pair<AIE::DeviceOp, AIE::RuntimeSequenceOp>>
selectDeviceAndRuntime(ModuleOp module, const AIE::VisualizeOptions &options) {
  SmallVector<AIE::DeviceOp> devices(module.getOps<AIE::DeviceOp>());
  AIE::DeviceOp device;
  if (!options.deviceName.empty()) {
    for (AIE::DeviceOp candidate : devices) {
      if (candidate.getSymName() == options.deviceName) {
        device = candidate;
        break;
      }
    }
    if (!device)
      return module.emitError()
             << "no aie.device named '" << options.deviceName << "'";
  } else if (devices.size() == 1) {
    device = devices.front();
  } else if (devices.empty()) {
    return module.emitError() << "input contains no aie.device operations";
  } else {
    return module.emitError()
           << "input contains multiple aie.device operations; select one with "
              "a device name";
  }

  if (options.skipRuntime && !options.runtimeSequenceName.empty())
    return device.emitError()
           << "a runtime sequence cannot be both skipped and named";

  AIE::RuntimeSequenceOp runtimeSequence;
  if (!options.skipRuntime) {
    SmallVector<AIE::RuntimeSequenceOp> sequences(
        device.getOps<AIE::RuntimeSequenceOp>());
    if (!options.runtimeSequenceName.empty()) {
      for (AIE::RuntimeSequenceOp candidate : sequences) {
        if (candidate.getSymName() == options.runtimeSequenceName) {
          runtimeSequence = candidate;
          break;
        }
      }
      if (!runtimeSequence)
        return device.emitError()
               << "no aie.runtime_sequence named '"
               << options.runtimeSequenceName << "' in selected device";
    } else if (sequences.size() == 1) {
      runtimeSequence = sequences.front();
    }
  }

  return std::make_pair(device, runtimeSequence);
}

LogicalResult AIE::emitRouteDot(ModuleOp module, raw_ostream &output,
                                const AIE::VisualizeOptions &options) {
  FailureOr<std::pair<AIE::DeviceOp, AIE::RuntimeSequenceOp>> selected =
      selectDeviceAndRuntime(module, options);
  if (failed(selected))
    return failure();
  return emitDeviceRouteDot(selected->first, output, selected->second, options);
}

LogicalResult AIE::emitPhysicalRouteDot(ModuleOp module, raw_ostream &output,
                                        const AIE::VisualizeOptions &options) {
  FailureOr<std::pair<AIE::DeviceOp, AIE::RuntimeSequenceOp>> selected =
      selectDeviceAndRuntime(module, options);
  if (failed(selected))
    return failure();
  return emitDevicePhysicalRouteDot(selected->first, output, selected->second,
                                    options);
}

LogicalResult AIE::emitSemanticGraphJson(ModuleOp module, raw_ostream &output,
                                         const AIE::VisualizeOptions &options) {
  FailureOr<std::pair<AIE::DeviceOp, AIE::RuntimeSequenceOp>> selected =
      selectDeviceAndRuntime(module, options);
  if (failed(selected))
    return failure();
  FailureOr<SemanticGraphAnalysis> analysis =
      analyzeSemanticGraph(selected->first, selected->second, options);
  if (failed(analysis))
    return failure();
  FailureOr<SelectedTopology> topology =
      selectTopology(selected->first, analysis->graph, options);
  if (failed(topology))
    return failure();
  writeSemanticGraphJson(selected->first, analysis->graph, *topology, output);
  return success();
}
