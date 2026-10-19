//===- aie-visualize.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace xilinx;

static cl::opt<std::string> fileName(cl::Positional, cl::desc("<input mlir>"),
                                     cl::Required);
static cl::opt<bool> emitDot("emit-dot",
                             cl::desc("Emit a DOT route visualization"));
static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));
static cl::list<unsigned>
    onlyFlows("only-flow", cl::desc("Emit only a flow with this numeric ID"),
              cl::CommaSeparated, cl::ZeroOrMore);
static cl::list<std::string> contractedNodeKinds(
    "contract",
    cl::desc("Contract semantic node kinds: tile, port, buffer, "
             "runtime-argument, call"),
    cl::value_desc("kind"), cl::CommaSeparated, cl::ZeroOrMore);
static cl::opt<bool> noFollowBuffers(
    "no-follow-buffers",
    cl::desc("Do not group flows through buffers or shared endpoints"));
static cl::opt<std::string>
    deviceName("device", cl::desc("Select an aie.device by symbol name"),
               cl::value_desc("symbol"));
static cl::opt<std::string>
    runtimeSequenceName("runtime-sequence",
                        cl::desc("Select an aie.runtime_sequence by symbol"),
                        cl::value_desc("symbol"));
static cl::opt<bool>
    skipRuntime("skip-runtime",
                cl::desc("Do not annotate DMA endpoints from runtime setup"));
static cl::opt<bool>
    skipVerify("skip-verify",
               cl::desc("Do not verify analysis IR after parsing"));
static cl::opt<bool>
    weighEdges("weigh-edges",
               cl::desc("Scale edge width by the bytes each flow group moves"));
static cl::opt<double>
    minEdgeWidth("min-edge-width",
                 cl::desc("Edge width in points for the lightest flow group"),
                 cl::init(0.5));
static cl::opt<double>
    maxEdgeWidth("max-edge-width",
                 cl::desc("Edge width in points for the heaviest flow group"),
                 cl::init(7.0));
static cl::opt<bool>
    tileRoles("tile-roles",
              cl::desc("Headline each tile with the kernels it runs and mark "
                       "tiles that run the same kernels"));
static cl::opt<unsigned>
    maxHeadlineKernels("max-headline-kernels",
                       cl::desc("Name at most this many kernels in a tile "
                                "headline, counting the rest"),
                       cl::init(4));
static cl::opt<double> tileWidth("tile-width", cl::desc("Tile width in inches"),
                                 cl::init(2.15));
static cl::opt<double>
    minimumTileHeight("tile-height", cl::desc("Minimum tile height in inches"),
                      cl::init(2.15));
static cl::opt<double>
    tileGapX("tile-gap-x", cl::desc("Horizontal gap between tiles in inches"),
             cl::init(0.85));
static cl::opt<double>
    tileGapY("tile-gap-y", cl::desc("Vertical gap between tiles in inches"),
             cl::init(0.85));
static cl::opt<double>
    localSpacing("local-spacing",
                 cl::desc("Vertical spacing between tile-local nodes"),
                 cl::init(0.4));

const std::string bold("\033[0;1m");
const std::string dim("\033[0;2m");
const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string blue("\033[1;34m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string bwhite("\033[0;47m");
const std::string reset("\033[0m");
const std::string bgray("\033[48;5;239m");

namespace {

struct PortNode {
  int col;
  int row;
  AIE::WireBundle bundle;
  int channel;
  std::optional<AIE::DMAChannelDir> dmaDirection;

  auto asTuple() const {
    return std::make_tuple(col, row, static_cast<int>(bundle), channel,
                           dmaDirection);
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
  std::optional<bool> keepsPacketHeader;
  bool priorityRoute;
  std::vector<PortNode> points;
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
  std::set<unsigned> flowGroups;
};

struct LayoutGeometry {
  std::vector<double> rowHeights;
  std::vector<double> rowCenters;
  std::map<std::pair<int, int>, double> ownHeights;

  double tileCenterX(int col) const { return col * (tileWidth + tileGapX); }
  double tileCenterY(int row) const { return rowCenters.at(row); }
  double tileHeight(int row) const { return rowHeights.at(row); }

  /// A tile is only as tall as its own contents. Sizing every tile to the
  /// tallest one in its row shrinks the text in all the others to fit a
  /// neighbour's buffer list. Rows still reserve the tallest tile's height so
  /// that boxes never overlap.
  double tileHeight(int col, int row) const {
    auto found = ownHeights.find({col, row});
    return found == ownHeights.end() ? rowHeights.at(row) : found->second;
  }
};

struct CallInfo {
  unsigned id;
  int col;
  int row;
  unsigned tileIndex;
  std::string callee;
  std::vector<AIE::BufferOp> buffers;
};

struct FlowGroups {
  std::vector<unsigned> routeGroups;
  unsigned count;
};

struct BufferSegment {
  PortNode port;
  Operation *buffer;
  bool intoBuffer;

  bool operator<(const BufferSegment &other) const {
    return std::tie(port, buffer, intoBuffer) <
           std::tie(other.port, other.buffer, other.intoBuffer);
  }
};

enum class SemanticNodeKind { Tile, Port, Buffer, RuntimeArgument, Call };

struct SemanticNode {
  SemanticNodeKind kind;
  std::optional<std::pair<int, int>> tile;
};

struct SemanticEdge {
  std::string source;
  std::string dest;
  unsigned flowGroup;
  unsigned count = 1;

  bool operator<(const SemanticEdge &other) const {
    return std::tie(source, dest, flowGroup) <
           std::tie(other.source, other.dest, other.flowGroup);
  }
};

struct SemanticGraph {
  std::map<std::string, SemanticNode> nodes;
  std::set<SemanticEdge> edges;

  void addNode(StringRef id, SemanticNodeKind kind,
               std::optional<std::pair<int, int>> tile = std::nullopt) {
    nodes.try_emplace(id.str(), SemanticNode{kind, tile});
  }

  void addEdge(StringRef source, StringRef dest, unsigned flowGroup) {
    edges.insert({source.str(), dest.str(), flowGroup});
  }

  bool hasIncoming(StringRef node, unsigned flowGroup) const {
    return llvm::any_of(edges, [&](const SemanticEdge &edge) {
      return edge.dest == node && edge.flowGroup == flowGroup;
    });
  }

  bool hasOutgoing(StringRef node, unsigned flowGroup) const {
    return llvm::any_of(edges, [&](const SemanticEdge &edge) {
      return edge.source == node && edge.flowGroup == flowGroup;
    });
  }

  std::set<SemanticEdge>
  contract(const std::set<SemanticNodeKind> &hiddenKinds) const {
    std::set<SemanticEdge> result = edges;
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto &[id, node] : nodes) {
        if (!hiddenKinds.count(node.kind))
          continue;
        SmallVector<SemanticEdge> incoming;
        SmallVector<SemanticEdge> outgoing;
        for (const SemanticEdge &edge : result) {
          if (edge.dest == id)
            incoming.push_back(edge);
          if (edge.source == id)
            outgoing.push_back(edge);
        }
        if (incoming.size() != 1 || outgoing.size() != 1 ||
            incoming.front().flowGroup != outgoing.front().flowGroup ||
            incoming.front().source == outgoing.front().dest)
          continue;
        result.erase(incoming.front());
        result.erase(outgoing.front());
        SemanticEdge contracted{incoming.front().source, outgoing.front().dest,
                                incoming.front().flowGroup,
                                incoming.front().count *
                                    outgoing.front().count};
        auto [edge, inserted] = result.insert(contracted);
        if (!inserted) {
          contracted.count += edge->count;
          result.erase(edge);
          result.insert(contracted);
        }
        changed = true;
        break;
      }
    }
    return result;
  }
};

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
    return flow.emitOpError(
        "requires vias; run aie-find-flows with emit-vias=true");
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
                      std::nullopt,
                      false,
                      {}};
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
      if (contractedNodeKinds.empty() && failed(appendVias(flow, route.points)))
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
                    packetFlow.getKeepPktHeader(),
                    packetFlow.getPriorityRoute().value_or(false),
                    {}};
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
    if (contractedNodeKinds.empty() &&
        failed(appendVias(packetFlow, route.points)))
      return failure();
    route.points.push_back(*dest);
    routes.push_back(std::move(route));
  }
  return routes;
}

static std::vector<ArbitrationRule>
collectArbitrationRules(AIE::DeviceOp device) {
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
          rules.push_back(
              {{source, dest},
               priority,
               static_cast<uint8_t>(rule.getMask()),
               static_cast<uint8_t>(rule.getValue()),
               static_cast<uint8_t>(amsel.getArbiterID()),
               static_cast<uint8_t>(amsel.getMsel()),
               master.keepsPktHeader(),
               rule->hasAttr("aie.priority_route"),
               rule->hasAttr("aie.is_ctrl_pkt_overlay") ||
                   packetRules->hasAttr("aie.is_ctrl_pkt_overlay") ||
                   master->hasAttr("aie.is_ctrl_pkt_overlay")});
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

/// Bytes each flow group moves per iteration, measured at the group's sources.
/// Counting the sources alone avoids double counting a broadcast, whose one
/// source feeds several destinations. Groups whose sources are core ports
/// rather than DMA channels fall back to their destinations, and a group with
/// no DMA endpoint at all stays at zero.
static std::vector<uint64_t>
collectFlowGroupBytes(ArrayRef<FlowRoute> routes, const FlowGroups &groups,
                      const DMAChannelBytes &channelBytes) {
  std::vector<std::set<DMAChannelKey>> sources(groups.count);
  std::vector<std::set<DMAChannelKey>> destinations(groups.count);
  for (const FlowRoute &route : routes) {
    unsigned group = groups.routeGroups[route.id];
    const PortNode &source = route.points.front();
    if (source.bundle == AIE::WireBundle::DMA)
      sources[group].insert(
          {source.col, source.row, AIE::DMAChannelDir::MM2S, source.channel});
    const PortNode &dest = route.points.back();
    if (dest.bundle == AIE::WireBundle::DMA)
      destinations[group].insert(
          {dest.col, dest.row, AIE::DMAChannelDir::S2MM, dest.channel});
  }
  std::vector<uint64_t> bytes(groups.count, 0);
  for (unsigned group = 0; group < groups.count; ++group) {
    const std::set<DMAChannelKey> &endpoints =
        sources[group].empty() ? destinations[group] : sources[group];
    for (const DMAChannelKey &channel : endpoints) {
      auto found = channelBytes.find(channel);
      if (found != channelBytes.end())
        bytes[group] += found->second;
    }
  }
  return bytes;
}

/// Maps bytes to stroke width on a log scale spanning the design's own range.
/// Transfer sizes here span several orders of magnitude, so a linear map would
/// collapse everything below the largest flow into one hairline.
struct EdgeWeightScale {
  uint64_t lightest = 0;
  uint64_t heaviest = 0;
  bool enabled = false;

  static EdgeWeightScale build(ArrayRef<uint64_t> groupBytes) {
    EdgeWeightScale scale;
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
      return minEdgeWidth;
    if (heaviest == lightest)
      return maxEdgeWidth;
    double position = (std::log2(static_cast<double>(bytes)) -
                       std::log2(static_cast<double>(lightest))) /
                      (std::log2(static_cast<double>(heaviest)) -
                       std::log2(static_cast<double>(lightest)));
    return minEdgeWidth + position * (maxEdgeWidth - minEdgeWidth);
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
                       call.getCallee().str(), std::move(buffers)});
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

/// Labels for tiles that share a role, so a reader can see that four tiles are
/// the same worker without comparing four lists of kernel names. Roles held by
/// a single tile get no label: there is no repetition to point out.
static std::map<std::pair<int, int>, std::string>
collectRoleLabels(const std::map<std::pair<int, int>, TileRole> &roles) {
  std::map<TileRole, std::vector<std::pair<int, int>>> classes;
  for (const auto &[coordinate, role] : roles)
    classes[role].push_back(coordinate);
  std::map<std::pair<int, int>, std::string> labels;
  unsigned next = 0;
  for (const auto &[role, members] : classes) {
    if (members.size() < 2)
      continue;
    std::string name(1, static_cast<char>('A' + next % 26));
    if (next >= 26)
      name += std::to_string(next / 26);
    ++next;
    for (const std::pair<int, int> &coordinate : members)
      labels[coordinate] = name + " x" + std::to_string(members.size());
  }
  return labels;
}

static FlowGroups collectFlowGroups(ArrayRef<FlowRoute> routes,
                                    const DMAChannelBuffers &channelBuffers) {
  FlowGroups groups;
  groups.routeGroups.resize(routes.size());
  if (noFollowBuffers) {
    std::iota(groups.routeGroups.begin(), groups.routeGroups.end(), 0);
    groups.count = routes.size();
    return groups;
  }

  std::vector<unsigned> parents(routes.size());
  std::iota(parents.begin(), parents.end(), 0);
  auto findRoot = [&](unsigned route) {
    while (parents[route] != route) {
      parents[route] = parents[parents[route]];
      route = parents[route];
    }
    return route;
  };
  auto merge = [&](unsigned first, unsigned second) {
    unsigned firstRoot = findRoot(first);
    unsigned secondRoot = findRoot(second);
    if (firstRoot != secondRoot)
      parents[secondRoot] = firstRoot;
  };

  using EndpointKey =
      std::tuple<PortNode, std::optional<int>, std::optional<int>>;
  std::map<Operation *, std::vector<unsigned>> bufferRoutes;
  std::map<EndpointKey, std::vector<unsigned>> endpointRoutes;
  for (const FlowRoute &route : routes) {
    const PortNode &source = route.points.front();
    if (source.bundle != AIE::WireBundle::DMA &&
        source.bundle != AIE::WireBundle::Core)
      endpointRoutes[{source, route.packetID, route.packetMask}].push_back(
          route.id);
    if (source.bundle == AIE::WireBundle::DMA) {
      DMAChannelKey key{source.col, source.row, AIE::DMAChannelDir::MM2S,
                        source.channel};
      auto channels = channelBuffers.find(key);
      if (channels != channelBuffers.end())
        for (AIE::BufferOp buffer : channels->second)
          bufferRoutes[buffer].push_back(route.id);
    }
    const PortNode &dest = route.points.back();
    if (dest.bundle != AIE::WireBundle::DMA &&
        dest.bundle != AIE::WireBundle::Core)
      endpointRoutes[{dest, route.packetID, route.packetMask}].push_back(
          route.id);
    if (dest.bundle == AIE::WireBundle::DMA) {
      DMAChannelKey key{dest.col, dest.row, AIE::DMAChannelDir::S2MM,
                        dest.channel};
      auto channels = channelBuffers.find(key);
      if (channels != channelBuffers.end())
        for (AIE::BufferOp buffer : channels->second)
          bufferRoutes[buffer].push_back(route.id);
    }
  }
  for (const auto &[buffer, routes] : bufferRoutes) {
    for (unsigned route : ArrayRef(routes).drop_front())
      merge(routes.front(), route);
  }
  for (const auto &[endpoint, routes] : endpointRoutes) {
    for (unsigned route : ArrayRef(routes).drop_front())
      merge(routes.front(), route);
  }

  std::map<unsigned, unsigned> groupIDs;
  for (const FlowRoute &route : routes) {
    unsigned root = findRoot(route.id);
    auto [group, inserted] = groupIDs.try_emplace(root, groupIDs.size());
    groups.routeGroups[route.id] = group->second;
  }
  groups.count = groupIDs.size();
  return groups;
}

static std::string portNodeID(const PortNode &port) {
  std::string id = "p_" + std::to_string(port.col) + "_" +
                   std::to_string(port.row) + "_" +
                   std::to_string(static_cast<int>(port.bundle)) + "_" +
                   std::to_string(port.channel);
  if (port.dmaDirection)
    id += *port.dmaDirection == AIE::DMAChannelDir::MM2S ? "_m" : "_s";
  return id;
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

static std::string dotCoordinate(double value) { return std::to_string(value); }

static std::string tileNodeID(int col, int row) {
  return "tile_" + std::to_string(col) + "_" + std::to_string(row);
}

static SemanticGraph buildSemanticGraph(
    ArrayRef<FlowRoute> routes, const FlowGroups &groups,
    const DMAChannelBuffers &channelBuffers,
    const std::map<DMAChannelKey, std::set<unsigned>> &channelFlowGroups,
    ArrayRef<BufferInfo> buffers,
    ArrayRef<RuntimeArgumentInfo> runtimeArguments, ArrayRef<CallInfo> calls,
    const std::map<Operation *, std::set<unsigned>> &bufferSourceFlowGroups,
    const std::map<Operation *, std::set<unsigned>> &bufferDestFlowGroups,
    const std::set<unsigned> &visibleGroups) {
  SemanticGraph graph;
  auto isVisible = [&](unsigned group) {
    return visibleGroups.empty() || visibleGroups.count(group);
  };
  auto addPort = [&](const PortNode &port) {
    std::string id = portNodeID(port);
    graph.addNode(id, SemanticNodeKind::Port,
                  std::make_pair(port.col, port.row));
    return id;
  };
  auto addTile = [&](int col, int row) {
    std::string id = tileNodeID(col, row);
    graph.addNode(id, SemanticNodeKind::Tile, std::make_pair(col, row));
    return id;
  };

  for (const FlowRoute &route : routes) {
    unsigned group = groups.routeGroups[route.id];
    if (!isVisible(group))
      continue;
    const PortNode &source = route.points.front();
    const PortNode &dest = route.points.back();
    graph.addEdge(addPort(source), addPort(dest), group);
    addTile(source.col, source.row);
    addTile(dest.col, dest.row);
  }

  std::map<Operation *, unsigned> bufferIDs;
  for (const BufferInfo &buffer : buffers) {
    bufferIDs[buffer.op] = buffer.id;
    graph.addNode(bufferNodeID(buffer.id), SemanticNodeKind::Buffer,
                  std::make_pair(buffer.col, buffer.row));
    addTile(buffer.col, buffer.row);
  }
  for (const auto &[key, channelBufferList] : channelBuffers) {
    auto [col, row, direction, channel] = key;
    PortNode port{col, row, AIE::WireBundle::DMA, channel, direction};
    std::string portID = addPort(port);
    auto flowGroups = channelFlowGroups.find(key);
    if (flowGroups == channelFlowGroups.end())
      continue;
    for (AIE::BufferOp buffer : channelBufferList) {
      std::string bufferID = bufferNodeID(bufferIDs.at(buffer));
      for (unsigned group : flowGroups->second) {
        if (!isVisible(group))
          continue;
        if (direction == AIE::DMAChannelDir::MM2S)
          graph.addEdge(bufferID, portID, group);
        else
          graph.addEdge(portID, bufferID, group);
      }
    }
  }

  for (const RuntimeArgumentInfo &argument : runtimeArguments) {
    std::string argumentID = runtimeArgumentNodeID(argument.id);
    graph.addNode(argumentID, SemanticNodeKind::RuntimeArgument);
    for (const DMAChannelKey &key : argument.channels) {
      auto [col, row, direction, channel] = key;
      PortNode port{col, row, AIE::WireBundle::DMA, channel, direction};
      std::string portID = addPort(port);
      auto flowGroups = channelFlowGroups.find(key);
      if (flowGroups == channelFlowGroups.end())
        continue;
      for (unsigned group : flowGroups->second) {
        if (!isVisible(group))
          continue;
        if (direction == AIE::DMAChannelDir::MM2S)
          graph.addEdge(argumentID, portID, group);
        else
          graph.addEdge(portID, argumentID, group);
      }
    }
  }

  for (const CallInfo &call : calls) {
    std::string callID = callNodeID(call.id);
    graph.addNode(callID, SemanticNodeKind::Call,
                  std::make_pair(call.col, call.row));
    addTile(call.col, call.row);
    for (AIE::BufferOp buffer : call.buffers) {
      std::string bufferID = bufferNodeID(bufferIDs.at(buffer));
      auto sourceGroups = bufferSourceFlowGroups.find(buffer);
      if (sourceGroups != bufferSourceFlowGroups.end()) {
        for (unsigned group : sourceGroups->second) {
          if (!isVisible(group))
            continue;
          graph.addEdge(callID, bufferID, group);
        }
      }
      auto destGroups = bufferDestFlowGroups.find(buffer);
      if (destGroups != bufferDestFlowGroups.end()) {
        for (unsigned group : destGroups->second) {
          if (!isVisible(group))
            continue;
          graph.addEdge(bufferID, callID, group);
        }
      }
    }
  }

  for (const FlowRoute &route : routes) {
    unsigned group = groups.routeGroups[route.id];
    if (!isVisible(group))
      continue;
    const PortNode &source = route.points.front();
    const PortNode &dest = route.points.back();
    std::string sourceID = portNodeID(source);
    std::string destID = portNodeID(dest);
    if (!graph.hasIncoming(sourceID, group))
      graph.addEdge(addTile(source.col, source.row), sourceID, group);
    if (!graph.hasOutgoing(destID, group))
      graph.addEdge(destID, addTile(dest.col, dest.row), group);
  }
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
  double halfWidth = tileWidth / 2;
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

static std::string flowClasses(const std::set<unsigned> &groups,
                               StringRef kind) {
  std::string classes = kind.str();
  if (!groups.empty())
    classes += " flow";
  for (unsigned group : groups)
    classes += " flow-" + std::to_string(group);
  return classes;
}

static std::string flowClasses(ArrayRef<const FlowRoute *> routes,
                               const FlowGroups &groups, StringRef kind) {
  std::set<unsigned> groupIDs;
  for (const FlowRoute *route : routes)
    groupIDs.insert(groups.routeGroups[route->id]);
  return flowClasses(groupIDs, kind);
}

static std::string flowColor(unsigned id) {
  static constexpr const char *colors[] = {
      "#d73027", "#4575b4", "#1a9850", "#984ea3", "#ff7f00",
      "#00a6a6", "#e7298a", "#6a3d9a", "#a6761d", "#1f78b4"};
  return colors[id % std::size(colors)];
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

static LogicalResult emitRouteDot(AIE::DeviceOp device, raw_ostream &output,
                                  AIE::RuntimeSequenceOp runtimeSequence) {
  FailureOr<std::vector<FlowRoute>> routes = collectRoutes(device);
  if (failed(routes))
    return failure();
  std::vector<ArbitrationRule> arbitrationRules =
      collectArbitrationRules(device);
  FailureOr<std::vector<BufferInfo>> buffers = collectBuffers(device);
  if (failed(buffers))
    return failure();
  DMAChannelBuffers channelBuffers = collectDMAChannelBuffers(device);
  std::vector<RuntimeBufferUse> runtimeBufferUses =
      collectRuntimeBufferUses(device, runtimeSequence);
  std::vector<CallInfo> calls = collectCalls(device);
  FlowGroups groups = collectFlowGroups(*routes, channelBuffers);
  std::vector<uint64_t> groupBytes =
      collectFlowGroupBytes(*routes, groups, collectDMAChannelBytes(device));
  EdgeWeightScale weightScale =
      weighEdges ? EdgeWeightScale::build(groupBytes) : EdgeWeightScale();

  std::set<unsigned> only(onlyFlows.begin(), onlyFlows.end());
  auto validateIDs = [&](const std::set<unsigned> &ids,
                         StringRef option) -> LogicalResult {
    for (unsigned id : ids) {
      if (id >= groups.count) {
        device.emitOpError() << option << " references unknown flow " << id
                             << "; valid IDs are 0 through "
                             << (groups.count == 0 ? 0 : groups.count - 1);
        return failure();
      }
    }
    return success();
  };
  if (failed(validateIDs(only, "--only-flow")))
    return failure();
  auto isVisible = [&](unsigned id) { return only.empty() || only.count(id); };

  std::set<SemanticNodeKind> contractKinds;
  for (StringRef kind : contractedNodeKinds) {
    if (kind == "tile")
      contractKinds.insert(SemanticNodeKind::Tile);
    else if (kind == "port")
      contractKinds.insert(SemanticNodeKind::Port);
    else if (kind == "buffer")
      contractKinds.insert(SemanticNodeKind::Buffer);
    else if (kind == "runtime-argument")
      contractKinds.insert(SemanticNodeKind::RuntimeArgument);
    else if (kind == "call")
      contractKinds.insert(SemanticNodeKind::Call);
    else {
      device.emitOpError() << "--contract references unknown node kind '"
                           << kind << "'";
      return failure();
    }
  }
  bool semanticView = !contractKinds.empty();

  std::set<PortNode> ports;
  std::set<std::pair<int, int>> topologyTiles;
  std::map<Segment, std::vector<const FlowRoute *>> segments;
  std::map<BufferSegment, std::vector<const FlowRoute *>> bufferSegments;
  std::map<unsigned, Segment> labeledSegments;
  std::set<Segment> arrowSegments;
  std::map<PortNode, std::set<unsigned>> portFlowGroups;
  std::map<DMAChannelKey, std::set<unsigned>> channelFlowGroups;
  std::map<Operation *, std::set<unsigned>> bufferFlowGroups;
  std::map<Operation *, std::set<unsigned>> bufferSourceFlowGroups;
  std::map<Operation *, std::set<unsigned>> bufferDestFlowGroups;
  for (const FlowRoute &route : *routes) {
    unsigned groupID = groups.routeGroups[route.id];
    if (!isVisible(groupID))
      continue;
    if (semanticView) {
      ports.insert(route.points.front());
      ports.insert(route.points.back());
      topologyTiles.insert(
          {route.points.front().col, route.points.front().row});
      topologyTiles.insert({route.points.back().col, route.points.back().row});
    } else {
      ports.insert(route.points.begin(), route.points.end());
    }
    std::vector<Segment> routeSegments;
    std::vector<Segment> physicalLinks;
    for (auto pair : llvm::zip_equal(ArrayRef(route.points).drop_back(),
                                     ArrayRef(route.points).drop_front())) {
      const auto &[source, dest] = pair;
      if (source == dest)
        continue;
      Segment segment{source, dest};
      segments[segment].push_back(&route);
      routeSegments.push_back(segment);
      if (source.col != dest.col || source.row != dest.row)
        physicalLinks.push_back(segment);
    }
    ArrayRef<Segment> labelCandidates = physicalLinks.empty()
                                            ? ArrayRef(routeSegments)
                                            : ArrayRef(physicalLinks);
    if (!labelCandidates.empty())
      labeledSegments[route.id] = labelCandidates[labelCandidates.size() / 2];
    if (!routeSegments.empty() &&
        (route.points.back().bundle == AIE::WireBundle::Core ||
         route.points.back().bundle == AIE::WireBundle::DMA))
      arrowSegments.insert(routeSegments.back());
    const PortNode &source = route.points.front();
    if (source.bundle == AIE::WireBundle::DMA) {
      DMAChannelKey key{source.col, source.row, AIE::DMAChannelDir::MM2S,
                        source.channel};
      channelFlowGroups[key].insert(groupID);
      for (AIE::BufferOp buffer : channelBuffers[key]) {
        bufferSegments[{source, buffer, false}].push_back(&route);
        bufferFlowGroups[buffer].insert(groupID);
        bufferSourceFlowGroups[buffer].insert(groupID);
      }
    }
    const PortNode &dest = route.points.back();
    if (dest.bundle == AIE::WireBundle::DMA) {
      DMAChannelKey key{dest.col, dest.row, AIE::DMAChannelDir::S2MM,
                        dest.channel};
      channelFlowGroups[key].insert(groupID);
      for (AIE::BufferOp buffer : channelBuffers[key]) {
        bufferSegments[{dest, buffer, true}].push_back(&route);
        bufferFlowGroups[buffer].insert(groupID);
        bufferDestFlowGroups[buffer].insert(groupID);
      }
    }
  }

  const AIE::AIETargetModel &model = device.getTargetModel();
  std::set<std::pair<int, int>> designTiles;
  for (AIE::TileOp tile : device.getOps<AIE::TileOp>())
    designTiles.insert({tile.colIndex(), tile.rowIndex()});
  std::set<Operation *> visibleBuffers;
  for (const auto &[segment, segmentRoutes] : bufferSegments)
    visibleBuffers.insert(segment.buffer);
  for (const CallInfo &call : calls)
    for (AIE::BufferOp buffer : call.buffers)
      visibleBuffers.insert(buffer);

  std::map<std::pair<int, int>, unsigned> localCounts;
  std::map<Operation *, unsigned> bufferLocalIndices;
  for (const BufferInfo &buffer : *buffers) {
    if (semanticView)
      continue;
    if (!only.empty() && !visibleBuffers.count(buffer.op))
      continue;
    std::pair<int, int> coordinate{buffer.col, buffer.row};
    bufferLocalIndices[buffer.op] = localCounts[coordinate]++;
  }
  std::map<unsigned, unsigned> callLocalIndices;
  std::map<unsigned, std::set<unsigned>> callFlowGroups;
  for (const CallInfo &call : calls) {
    for (AIE::BufferOp buffer : call.buffers)
      callFlowGroups[call.id].insert(bufferFlowGroups[buffer].begin(),
                                     bufferFlowGroups[buffer].end());
    if (!only.empty() && !callFlowGroups[call.id].count(*only.begin()))
      continue;
    callLocalIndices[call.id] = localCounts[{call.col, call.row}]++;
  }

  std::vector<RuntimeArgumentInfo> runtimeArguments;
  std::map<unsigned, unsigned> argumentIDs;
  for (const RuntimeBufferUse &use : runtimeBufferUses) {
    const std::set<unsigned> &flowGroups = channelFlowGroups[use.channel];
    if (!only.empty() && !flowGroups.count(*only.begin()))
      continue;
    auto [entry, inserted] =
        argumentIDs.try_emplace(use.argumentIndex, runtimeArguments.size());
    if (inserted)
      runtimeArguments.push_back(
          {use.argumentIndex, use.argument, {}, flowGroups});
    RuntimeArgumentInfo &argument = runtimeArguments[entry->second];
    if (!llvm::is_contained(argument.channels, use.channel))
      argument.channels.push_back(use.channel);
    argument.flowGroups.insert(flowGroups.begin(), flowGroups.end());
  }
  llvm::sort(runtimeArguments, [](const RuntimeArgumentInfo &left,
                                  const RuntimeArgumentInfo &right) {
    return left.id < right.id;
  });
  SemanticGraph semanticGraph;
  std::set<SemanticEdge> topologyEdges;
  std::set<std::string> topologyNodes;
  if (semanticView) {
    semanticGraph =
        buildSemanticGraph(*routes, groups, channelBuffers, channelFlowGroups,
                           *buffers, runtimeArguments, calls,
                           bufferSourceFlowGroups, bufferDestFlowGroups, only);
    topologyEdges = semanticGraph.contract(contractKinds);
    topologyTiles.clear();
    for (const SemanticEdge &edge : topologyEdges) {
      for (StringRef nodeID : {StringRef(edge.source), StringRef(edge.dest)}) {
        topologyNodes.insert(nodeID.str());
        const SemanticNode &node = semanticGraph.nodes.at(nodeID.str());
        if (node.tile)
          topologyTiles.insert(*node.tile);
      }
    }
    for (const BufferInfo &buffer : *buffers) {
      if (!topologyNodes.count(bufferNodeID(buffer.id)))
        continue;
      std::pair<int, int> coordinate{buffer.col, buffer.row};
      bufferLocalIndices[buffer.op] = localCounts[coordinate]++;
    }
  }

  std::map<std::pair<int, int>, TileRole> roles;
  std::map<std::pair<int, int>, std::string> roleLabels;
  if (tileRoles) {
    roles = collectTileRoles(calls);
    roleLabels = collectRoleLabels(roles);
  }
  // The headline sits inside the tile box, so the box reserves room for it
  // before the buffers and calls start.
  auto headroom = [&](std::pair<int, int> coordinate) {
    if (!tileRoles)
      return 0.45;
    auto role = roles.find(coordinate);
    unsigned lines =
        role == roles.end() ? 1 : 1 + role->second.lines(maxHeadlineKernels);
    return 0.30 + 0.17 * static_cast<double>(lines);
  };

  LayoutGeometry layout;
  layout.rowHeights.assign(model.rows(), minimumTileHeight);
  for (const std::pair<int, int> &coordinate : designTiles) {
    auto count = localCounts.find(coordinate);
    double height =
        std::max(static_cast<double>(minimumTileHeight),
                 headroom(coordinate) + 0.45 +
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
                             layout.rowHeights[row - 1] / 2 + tileGapY +
                             layout.rowHeights[row] / 2;
  auto localNodeY = [&](int col, int row, unsigned index) {
    return layout.tileCenterY(row) + layout.tileHeight(col, row) / 2 -
           headroom({col, row}) - index * localSpacing;
  };
  double localNodeWidth =
      std::clamp(static_cast<double>(tileWidth) - 0.4, 1.0, 1.75);

  output << "digraph aie_routes {\n"
         << "  graph [layout=neato, overlap=true"
         << (semanticView ? ", splines=curved" : "")
         << ", outputorder=nodesfirst, bgcolor=\"white\", pad=\"0.45\"];\n"
         << "  node [fontname=\"Helvetica\"];\n"
         << "  edge [fontname=\"Helvetica\", fontsize=9, arrowsize=0.65];\n";
  if (weightScale.enabled) {
    // Anchored below the shim row, where no tile or runtime argument sits.
    double legendX = layout.tileCenterX(0) - tileWidth / 2;
    double legendY = layout.tileCenterY(0) - layout.tileHeight(0) / 2 - 1.9;
    std::vector<uint64_t> samples{weightScale.lightest, weightScale.heaviest};
    if (weightScale.heaviest > weightScale.lightest)
      samples.insert(samples.begin() + 1,
                     static_cast<uint64_t>(
                         std::sqrt(static_cast<double>(weightScale.lightest) *
                                   static_cast<double>(weightScale.heaviest))));
    samples.erase(llvm::unique(samples), samples.end());
    output << "  legend_title [shape=plaintext, pos=\""
           << dotCoordinate(legendX + 0.75) << ','
           << dotCoordinate(legendY + 0.45)
           << "!\", label=\"bytes moved per iteration\", fontsize=10, "
              "fontcolor=\"#333333\", class=\"legend\"];\n";
    for (auto [index, bytes] : llvm::enumerate(samples)) {
      double y = legendY - static_cast<double>(index) * 0.34;
      output << "  legend_" << index << "_from [shape=point, width=0.01, "
             << R"(color="#ffffff", pos=")" << dotCoordinate(legendX) << ','
             << dotCoordinate(y) << "!\", class=\"legend\"];\n"
             << "  legend_" << index << "_to [shape=point, width=0.01, "
             << R"(color="#ffffff", pos=")" << dotCoordinate(legendX + 1.1)
             << ',' << dotCoordinate(y) << "!\", class=\"legend\"];\n"
             << "  legend_" << index << "_text [shape=plaintext, pos=\""
             << dotCoordinate(legendX + 1.55) << ',' << dotCoordinate(y)
             << "!\", label=\"" << formatBytes(bytes)
             << "\", fontsize=9, fontcolor=\"#333333\", class=\"legend\"];\n"
             << "  legend_" << index << "_from -> legend_" << index
             << "_to [penwidth=\""
             << llvm::format("%.2f", weightScale.width(bytes))
             << "\", color=\"#555555\", arrowhead=none, class=\"legend\"];\n";
    }
  }
  for (int col = 0; col < model.columns(); ++col) {
    for (int row = 0; row < model.rows(); ++row) {
      if (!designTiles.count({col, row}) ||
          (semanticView && !topologyTiles.count({col, row})))
        continue;
      StringRef fill = model.isCoreTile(col, row)      ? "#eef7ee"
                       : model.isMemTile(col, row)     ? "#fff2df"
                       : model.isShimNOCTile(col, row) ? "#e8f1fb"
                                                       : "#f3edf8";
      output << "  " << tileNodeID(col, row) << " [shape=box";
      output << ", fixedsize=true, width=" << dotCoordinate(tileWidth)
             << ", height=" << dotCoordinate(layout.tileHeight(col, row))
             << ", pos=\"" << dotCoordinate(layout.tileCenterX(col)) << ','
             << dotCoordinate(layout.tileCenterY(row)) << "!\"";
      output << ", label=\"(" << col << ',' << row << ')';
      if (tileRoles) {
        auto label = roleLabels.find({col, row});
        if (label != roleLabels.end())
          output << "  " << label->second;
        auto role = roles.find({col, row});
        if (role != roles.end())
          output << "\\n" << role->second.text(maxHeadlineKernels);
        output << R"(", labelloc="t", fontsize=11)";
      } else {
        output << '"';
      }
      output << ", style=filled, fillcolor=\"" << fill
             << "\", color=\"#b8b8b8\", fontcolor=\"#555555\"];\n";
    }
  }
  if (!semanticView) {
    for (const PortNode &port : ports) {
      auto [x, y] = portPosition(port, layout);
      output << "  " << portNodeID(port);
      if (port.bundle == AIE::WireBundle::DMA) {
        AIE::DMAChannelDir direction =
            port.dmaDirection.value_or(AIE::DMAChannelDir::MM2S);
        DMAChannelKey key{port.col, port.row, direction, port.channel};
        output << " [shape=box, fixedsize=true, width=0.82, height=0.18, pos=\""
               << dotCoordinate(x) << ',' << dotCoordinate(y) << "!\", label=\""
               << shortPortName(port) << "\", class=\""
               << flowClasses(channelFlowGroups[key], "port")
               << "\", fontsize=7, style=filled, fillcolor=\"#ffffff\", "
                  "color=\"#777777\"];\n";
      } else {
        StringRef kind =
            port.bundle == AIE::WireBundle::Core ? "port" : "port detail-vias";
        output << " [shape=point, width=0.09, pos=\"" << dotCoordinate(x) << ','
               << dotCoordinate(y) << "!\", xlabel=\"" << shortPortName(port)
               << "\", class=\"" << flowClasses(portFlowGroups[port], kind)
               << "\"];\n";
      }
    }
    unsigned annotationID = 0;
    std::map<Segment, unsigned> segmentAnnotationCounts;
    for (const ArbitrationRule &rule : arbitrationRules) {
      auto segmentRoutes = segments.find(rule.segment);
      if (segmentRoutes == segments.end())
        continue;
      std::set<unsigned> matchingGroups;
      for (const FlowRoute *route : segmentRoutes->second) {
        if (!route->packetID)
          continue;
        bool usesRule = false;
        for (unsigned packetID = 0; packetID <= 255; ++packetID) {
          bool claimedByFlow =
              route->packetMask
                  ? (packetID & *route->packetMask) == *route->packetID
                  : packetID == *route->packetID;
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
          matchingGroups.insert(groups.routeGroups[route->id]);
      }
      if (matchingGroups.empty())
        continue;
      auto [sourceX, sourceY] = portPosition(rule.segment.source, layout);
      auto [destX, destY] = portPosition(rule.segment.dest, layout);
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
             << dotCoordinate(x) << ',' << dotCoordinate(y) << "!\", label=\"r"
             << rule.priority << " pkt=" << rule.value << '/' << rule.mask
             << " -> A" << rule.arbiter << ".M" << rule.msel;
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
             << flowClasses(matchingGroups,
                            "arbitration detail-arbitration detail-vias")
             << "\", fontsize=6, style=filled, fillcolor=\"#fffdf4\", "
                "color=\"#9a8b52\", fontcolor=\"#554b2c\"];\n";
    }
  } else {
    for (const PortNode &port : ports) {
      std::string id = portNodeID(port);
      if (!topologyNodes.count(id))
        continue;
      auto [x, y] = portPosition(port, layout);
      std::set<unsigned> nodeGroups;
      for (const SemanticEdge &edge : topologyEdges)
        if (edge.source == id || edge.dest == id)
          nodeGroups.insert(edge.flowGroup);
      output << "  " << id
             << " [shape=box, fixedsize=true, width=0.82, height=0.18, pos=\""
             << dotCoordinate(x) << ',' << dotCoordinate(y) << "!\", label=\""
             << shortPortName(port) << "\", class=\""
             << flowClasses(nodeGroups, "port")
             << "\", fontsize=7, style=filled, fillcolor=\"#ffffff\", "
                "color=\"#777777\"];\n";
    }
  }
  std::map<Operation *, unsigned> bufferIDs;
  for (const BufferInfo &buffer : *buffers) {
    auto localIndex = bufferLocalIndices.find(buffer.op);
    if (localIndex == bufferLocalIndices.end())
      continue;
    bufferIDs[buffer.op] = buffer.id;
    double x = layout.tileCenterX(buffer.col);
    double y = localNodeY(buffer.col, buffer.row, localIndex->second);
    output << "  " << bufferNodeID(buffer.id)
           << " [shape=box, fixedsize=true, width="
           << dotCoordinate(localNodeWidth) << ", height=0.34, pos=\""
           << dotCoordinate(x) << ',' << dotCoordinate(y) << "!\"";
    output << ", label=\"" << bufferLabel(buffer.op, buffer.id)
           << "\", class=\""
           << flowClasses(bufferFlowGroups[buffer.op], "buffer detail-buffers")
           << "\", fontsize=8, style=filled, fillcolor=\"#ffffff\", "
              "color=\"#666666\"];\n";
  }
  if (!runtimeArguments.empty()) {
    std::set<int> designColumns;
    for (auto [col, row] : designTiles)
      designColumns.insert(col);
    double designLeft = layout.tileCenterX(*designColumns.begin());
    double designRight = layout.tileCenterX(*designColumns.rbegin());
    double center = (designLeft + designRight) / 2;
    double argumentSpacing = localNodeWidth + 0.3;
    if (runtimeArguments.size() > 1)
      argumentSpacing =
          std::max(argumentSpacing,
                   (designRight - designLeft) / (runtimeArguments.size() - 1));
    double argumentLeft =
        center - argumentSpacing * (runtimeArguments.size() - 1) / 2;
    double argumentY = layout.tileCenterY(0) - layout.tileHeight(0) / 2 - 0.55;
    for (auto [index, argument] : llvm::enumerate(runtimeArguments)) {
      double x = argumentLeft + index * argumentSpacing;
      output << "  " << runtimeArgumentNodeID(argument.id)
             << " [shape=ellipse, fixedsize=true, width="
             << dotCoordinate(localNodeWidth) << ", height=0.3, pos=\""
             << dotCoordinate(x) << ',' << dotCoordinate(argumentY)
             << "!\", label=\"%" << escapeDotLabel(argument.name)
             << "\", class=\""
             << flowClasses(argument.flowGroups, "runtime-argument")
             << "\", fontsize=8, style=filled, fillcolor=\"#fff7d6\", "
                "color=\"#8f7c32\"];\n";
    }
    if (!semanticView) {
      for (const RuntimeArgumentInfo &argument : runtimeArguments) {
        for (const DMAChannelKey &key : argument.channels) {
          auto [col, row, direction, channel] = key;
          PortNode port{col, row, AIE::WireBundle::DMA, channel, direction};
          std::string source = runtimeArgumentNodeID(argument.id);
          std::string dest = portNodeID(port);
          if (direction == AIE::DMAChannelDir::S2MM)
            std::swap(source, dest);
          output << "  " << source << " -> " << dest << " [class=\""
                 << flowClasses(channelFlowGroups[key], "runtime-link")
                 << "\", color=\"#39789b\", penwidth=\"1.2\"];\n";
        }
      }
    }
  }
  if (!semanticView || !calls.empty()) {
    for (const CallInfo &call : calls) {
      auto localIndex = callLocalIndices.find(call.id);
      if (localIndex == callLocalIndices.end())
        continue;
      double x = layout.tileCenterX(call.col);
      double y = localNodeY(call.col, call.row, localIndex->second);
      output << "  " << callNodeID(call.id)
             << " [shape=box, fixedsize=true, width="
             << dotCoordinate(localNodeWidth) << ", height=0.3, pos=\""
             << dotCoordinate(x) << ',' << dotCoordinate(y) << "!\", label=\""
             << escapeDotLabel(call.callee) << "\", class=\""
             << flowClasses(callFlowGroups[call.id], "call detail-calls")
             << "\", fontsize=8, style=filled, fillcolor=\"#f7f7f7\", "
                "color=\"#555555\"];\n";
      if (semanticView)
        continue;
      for (AIE::BufferOp buffer : call.buffers) {
        auto bufferID = bufferIDs.find(buffer);
        if (bufferID != bufferIDs.end())
          output << "  " << bufferNodeID(bufferID->second) << " -> "
                 << callNodeID(call.id) << " [class=\""
                 << flowClasses(bufferFlowGroups[buffer],
                                "call-link detail-calls detail-buffers")
                 << "\", color=\"#666666\", arrowsize=0.55];\n";
      }
    }
  }
  auto edgeBytes = [&](ArrayRef<const FlowRoute *> edgeRoutes) {
    std::set<unsigned> counted;
    uint64_t bytes = 0;
    for (const FlowRoute *route : edgeRoutes) {
      unsigned groupID = groups.routeGroups[route->id];
      if (counted.insert(groupID).second)
        bytes += groupBytes[groupID];
    }
    return bytes;
  };
  auto writeWeight = [&](ArrayRef<const FlowRoute *> edgeRoutes) {
    if (!weightScale.enabled) {
      output << R"(, penwidth="2.4")";
      return;
    }
    uint64_t bytes = edgeBytes(edgeRoutes);
    output << ", penwidth=\"" << llvm::format("%.2f", weightScale.width(bytes))
           << '"';
    // A flow with no buffer descriptor at either end states no volume. Saying
    // "0 B" would claim it carries nothing, which is a different fact.
    if (bytes == 0)
      output << ", tooltip=\"volume not stated in the IR\"";
    else
      output << ", tooltip=\"" << formatBytes(bytes) << " per iteration\"";
  };
  auto writeColors = [&](ArrayRef<const FlowRoute *> edgeRoutes) {
    bool first = true;
    std::set<unsigned> writtenGroups;
    for (const FlowRoute *route : edgeRoutes) {
      unsigned groupID = groups.routeGroups[route->id];
      if (!writtenGroups.insert(groupID).second)
        continue;
      if (!first)
        output << ':';
      output << flowColor(groupID);
      first = false;
    }
    if (first)
      output << "#c2c2c2";
  };
  auto writeLabel = [&](const FlowRoute &route) {
    unsigned groupID = groups.routeGroups[route.id];
    output << "<FONT COLOR=\"" << flowColor(groupID) << "\">F" << groupID;
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
  if (semanticView) {
    std::vector<std::vector<const FlowRoute *>> groupRoutes(groups.count);
    for (const FlowRoute &route : *routes) {
      unsigned groupID = groups.routeGroups[route.id];
      if (isVisible(groupID))
        groupRoutes[groupID].push_back(&route);
    }
    std::map<std::pair<std::string, std::string>, std::set<unsigned>> edges;
    std::map<std::pair<std::string, std::string>, unsigned> edgeCounts;
    for (const SemanticEdge &edge : topologyEdges) {
      edges[{edge.source, edge.dest}].insert(edge.flowGroup);
      edgeCounts[{edge.source, edge.dest}] += edge.count;
    }
    std::map<std::pair<std::string, std::string>, std::set<unsigned>>
        labeledEdgeGroups;
    for (unsigned group = 0; group < groups.count; ++group) {
      auto selected =
          llvm::find_if(topologyEdges, [&](const SemanticEdge &edge) {
            if (edge.flowGroup != group)
              return false;
            unsigned outgoing = llvm::count_if(
                topologyEdges, [&](const SemanticEdge &candidate) {
                  return candidate.flowGroup == group &&
                         candidate.source == edge.dest;
                });
            unsigned incoming = llvm::count_if(
                topologyEdges, [&](const SemanticEdge &candidate) {
                  return candidate.flowGroup == group &&
                         candidate.dest == edge.source;
                });
            return outgoing > 1 || incoming > 1;
          });
      if (selected == topologyEdges.end())
        selected = llvm::find_if(topologyEdges, [&](const SemanticEdge &edge) {
          return edge.flowGroup == group;
        });
      if (selected != topologyEdges.end())
        labeledEdgeGroups[{selected->source, selected->dest}].insert(group);
    }
    std::set<unsigned> callDependentGroups;
    for (const auto &[call, callGroups] : callFlowGroups)
      callDependentGroups.insert(callGroups.begin(), callGroups.end());
    // Heaviest last, so a dominant flow is never hidden under a hairline.
    std::vector<std::pair<std::string, std::string>> emissionOrder;
    emissionOrder.reserve(edges.size());
    for (const auto &[endpoints, edgeGroups] : edges)
      emissionOrder.push_back(endpoints);
    if (weightScale.enabled)
      llvm::stable_sort(emissionOrder,
                        [&](const auto &first, const auto &second) {
                          auto weight = [&](const auto &endpoints) {
                            uint64_t bytes = 0;
                            for (unsigned group : edges.at(endpoints))
                              bytes += groupBytes[group];
                            return bytes;
                          };
                          return weight(first) < weight(second);
                        });
    for (const auto &endpoints : emissionOrder) {
      const std::set<unsigned> &edgeGroups = edges.at(endpoints);
      std::vector<const FlowRoute *> edgeRoutes;
      std::vector<const FlowRoute *> labelRoutes;
      for (unsigned group : edgeGroups) {
        std::set<std::tuple<std::optional<unsigned>, std::optional<unsigned>,
                            std::optional<bool>, bool>>
            labels;
        for (const FlowRoute *route : groupRoutes[group]) {
          if (labels
                  .insert({route->packetID, route->packetMask,
                           route->keepsPacketHeader, route->priorityRoute})
                  .second) {
            edgeRoutes.push_back(route);
            if (labeledEdgeGroups[endpoints].count(group))
              labelRoutes.push_back(route);
          }
        }
      }
      std::string kind = "route detail-packet-ids";
      if (llvm::any_of(edgeGroups, [&](unsigned group) {
            return callDependentGroups.count(group);
          }))
        kind += " detail-calls";
      output << "  " << endpoints.first << " -> " << endpoints.second
             << " [color=\"";
      writeColors(edgeRoutes);
      output << "\", class=\"" << flowClasses(edgeRoutes, groups, kind) << '"';
      writeWeight(edgeRoutes);
      bool writeRouteLabels = !labelRoutes.empty();
      bool writeCount = edgeCounts[endpoints] > 1;
      if (writeRouteLabels || writeCount) {
        output << ", label=<";
        if (writeRouteLabels) {
          for (auto [index, route] : llvm::enumerate(labelRoutes)) {
            if (index)
              output << "<BR/>";
            writeLabel(*route);
          }
        }
        if (writeRouteLabels && writeCount)
          output << "<BR/>";
        if (writeCount)
          output << "<FONT COLOR=\"#555555\">&#215;" << edgeCounts[endpoints]
                 << "</FONT>";
        output << '>';
      }
      output << "];\n";
    }
    output << "}\n";
    return success();
  }
  for (const auto &[segment, segmentRoutes] : segments) {
    std::vector<const FlowRoute *> labels;
    for (const FlowRoute *route : segmentRoutes) {
      if (labeledSegments.at(route->id) < segment ||
          segment < labeledSegments.at(route->id))
        continue;
      labels.push_back(route);
    }
    output << "  " << portNodeID(segment.source) << " -> "
           << portNodeID(segment.dest) << " [color=\"";
    writeColors(segmentRoutes);
    std::string kind = labels.empty() ? "route" : "route detail-packet-ids";
    if (!arrowSegments.count(segment))
      kind += " detail-vias";
    output << "\", class=\"" << flowClasses(segmentRoutes, groups, kind) << '"';
    writeWeight(segmentRoutes);
    if (!labels.empty()) {
      output << ", label=<";
      for (auto [index, route] : llvm::enumerate(labels)) {
        if (index)
          output << "<BR/>";
        writeLabel(*route);
      }
      output << '>';
    }
    output << "];\n";
  }
  for (const auto &[segment, segmentRoutes] : bufferSegments) {
    auto bufferID = bufferIDs.find(segment.buffer);
    if (bufferID == bufferIDs.end())
      continue;
    std::string portID = portNodeID(segment.port);
    std::string source =
        segment.intoBuffer ? portID : bufferNodeID(bufferID->second);
    std::string dest =
        segment.intoBuffer ? bufferNodeID(bufferID->second) : portID;
    output << "  " << source << " -> " << dest << " [color=\"";
    writeColors(segmentRoutes);
    output << "\", class=\""
           << flowClasses(segmentRoutes, groups, "buffer-link detail-buffers")
           << '"';
    writeWeight(segmentRoutes);
    output << ", style=dashed];\n";
  }
  output << "}\n";
  return success();
}

} // namespace

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv);

  MLIRContext ctx;
  ParserConfig pcfg(&ctx, /*verifyAfterParse=*/!skipVerify);
  SourceMgr srcMgr;

  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<vector::VectorDialect>();
  xilinx::registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  xilinx::xllvm::registerXLLVMDialectTranslation(registry);
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> owning =
      parseSourceFile<ModuleOp>(fileName, srcMgr, pcfg);

  if (!owning)
    return 1;

  SmallVector<AIE::DeviceOp> deviceOps(owning->getOps<AIE::DeviceOp>());
  AIE::DeviceOp deviceOp;
  if (!deviceName.empty()) {
    for (AIE::DeviceOp candidate : deviceOps) {
      if (candidate.getSymName() == deviceName) {
        deviceOp = candidate;
        break;
      }
    }
    if (!deviceOp) {
      errs() << "no aie.device named '" << deviceName << "'\n";
      return 2;
    }
  } else if (deviceOps.size() == 1) {
    deviceOp = deviceOps.front();
  } else if (deviceOps.empty()) {
    errs() << "input contains no aie.device operations\n";
    return 2;
  } else {
    errs() << "input contains multiple aie.device operations; select one with "
              "--device=<symbol>\n";
    return 2;
  }

  const xilinx::AIE::AIETargetModel &model = deviceOp.getTargetModel();

  model.validate();

  if (skipRuntime && !runtimeSequenceName.empty()) {
    errs() << "--skip-runtime and --runtime-sequence are mutually exclusive\n";
    return 3;
  }
  AIE::RuntimeSequenceOp runtimeSequence;
  if (!skipRuntime) {
    SmallVector<AIE::RuntimeSequenceOp> sequences(
        deviceOp.getOps<AIE::RuntimeSequenceOp>());
    if (!runtimeSequenceName.empty()) {
      for (AIE::RuntimeSequenceOp candidate : sequences) {
        if (candidate.getSymName() == runtimeSequenceName) {
          runtimeSequence = candidate;
          break;
        }
      }
      if (!runtimeSequence) {
        errs() << "no aie.runtime_sequence named '" << runtimeSequenceName
               << "' in selected device\n";
        return 3;
      }
    } else if (sequences.size() == 1) {
      runtimeSequence = sequences.front();
    }
  }

  if (emitDot) {
    std::error_code error;
    ToolOutputFile output(outputFilename, error, sys::fs::OF_Text);
    if (error) {
      errs() << error.message() << '\n';
      return 3;
    }
    if (failed(emitRouteDot(deviceOp, output.os(), runtimeSequence)))
      return 4;
    output.keep();
    return 0;
  }

  std::vector<bool> used(model.columns() * model.rows());
  for (int col = 0; col < model.columns(); col++) {
    for (int row = 0; row < model.rows(); row++) {
      used[col + model.columns() * row] = false;
    }
  }
  for (auto tile : deviceOp.getOps<AIE::TileOp>()) {
    used[tile.getCol() + model.columns() * tile.getRow()] = true;
  }

  std::cout << model.columns() << " Columns and " << model.rows() << " Rows\n";
  for (int row = model.rows() - 1; row >= 0; row--) {
    std::cout << reset << row % 10 << " ";
    for (int col = 0; col < model.columns(); col++) {
      if (used[col + model.columns() * row])
        std::cout << bgray;
      else
        std::cout << dim;
      std::string v = reset + ".";
      if (model.isCoreTile(col, row))
        v = green + 'C';
      else if (model.isMemTile(col, row))
        v = red + 'M';
      else if (model.isShimNOCTile(col, row))
        v = blue + 'D';
      else if (model.isShimPLTile(col, row))
        v = magenta + 'P';
      std::cout << v << reset;
    }
    std::cout << "\n";
  }

  std::cout << reset << "  ";
  for (int col = 0; col < model.columns(); col++)
    std::cout << col % 10;
  std::cout << "\n";

  std::cout << "  ";
  for (int col = 0; col < model.columns(); col++) {
    int coltens = col / 10;
    if (coltens > 0)
      std::cout << coltens;
    else
      std::cout << " ";
  }
  std::cout << "\n";

  return 0;
}
