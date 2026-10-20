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

static FailureOr<std::vector<FlowRoute>> collectRoutes(AIE::DeviceOp device,
                                                       bool requireVias) {
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
      if (requireVias && failed(appendVias(flow, route.points)))
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
    if (requireVias && failed(appendVias(packetFlow, route.points)))
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
/// Every colour the view draws in, so that a scheme is one table rather than
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
  /// Fills for tiles that share a role, and colours for the flows.
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

static FlowGroups collectFlowGroups(ArrayRef<FlowRoute> routes,
                                    const DMAChannelBuffers &channelBuffers,
                                    bool noFollowBuffers) {
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

static LogicalResult emitDeviceRouteDot(AIE::DeviceOp device,
                                        raw_ostream &output,
                                        AIE::RuntimeSequenceOp runtimeSequence,
                                        const AIE::VisualizeOptions &options) {
  bool semanticView = !options.contractedNodeKinds.empty();
  FailureOr<std::vector<FlowRoute>> routes =
      collectRoutes(device, /*requireVias=*/!semanticView);
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
  FlowGroups groups =
      collectFlowGroups(*routes, channelBuffers, options.noFollowBuffers);
  std::vector<uint64_t> groupBytes =
      collectFlowGroupBytes(*routes, groups, collectDMAChannelBytes(device));
  EdgeWeightScale weightScale =
      options.weighEdges
          ? EdgeWeightScale::build(groupBytes, options.minEdgeWidth,
                                   options.maxEdgeWidth)
          : EdgeWeightScale();

  std::set<unsigned> only(options.onlyFlows.begin(), options.onlyFlows.end());
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
  for (StringRef kind : options.contractedNodeKinds) {
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
  std::map<std::pair<int, int>, RoleClass> roleClasses;
  if (options.tileRoles) {
    roles = collectTileRoles(calls);
    roleClasses = collectRoleClasses(roles);
  }
  // The headline sits inside the tile box, so the box reserves room for it
  // before the buffers and calls start.
  auto headroom = [&](std::pair<int, int> coordinate) {
    if (!options.tileRoles)
      return 0.45;
    auto role = roles.find(coordinate);
    unsigned lines = role == roles.end()
                         ? 1
                         : 1 + role->second.lines(options.maxHeadlineKernels);
    // One line of padding above the headline and one below it.
    return 0.34 + 0.17 * static_cast<double>(lines + 1);
  };

  const Palette &palette = paletteFor(options.colors);
  std::unique_ptr<ArrowRenderer> arrows =
      options.arrows == AIE::ArrowStyle::Channel
          ? createChannelArrowRenderer()
          : createStraightArrowRenderer();
  // A renderer that needs room between tiles to put arrows in says so here,
  // rather than the caller having to know which style wants what spacing.
  double gapX = options.tileGapX;
  double gapY = options.tileGapY;
  if (std::optional<double> wanted = arrows->preferredTileGap()) {
    gapX = std::max(gapX, *wanted);
    gapY = std::max(gapY, *wanted);
  }
  double tileWidth = options.tileWidth;
  unsigned busiest = 0;
  for (const auto &[coordinate, count] : localCounts)
    busiest = std::max(busiest, count);
  if (std::optional<double> wanted = arrows->preferredTileWidth(busiest))
    tileWidth = std::max(tileWidth, *wanted);

  LayoutGeometry layout;
  // Only nop2 leaves supplied edge geometry alone; every laying-out engine
  // recomputes it.
  layout.preLaidOut = options.arrows == AIE::ArrowStyle::Channel;
  layout.tileWidth = tileWidth;
  layout.tileGapX = gapX;
  layout.tileGapY = gapY;
  layout.rowHeights.assign(model.rows(), options.minimumTileHeight);
  for (const std::pair<int, int> &coordinate : designTiles) {
    auto count = localCounts.find(coordinate);
    double height = std::max(
        static_cast<double>(options.minimumTileHeight),
        headroom(coordinate) + 0.45 +
            (count == localCounts.end()
                 ? 0.0
                 : static_cast<double>(count->second) * options.localSpacing));
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
           index * options.localSpacing;
  };
  // Arrows are laid out together: separating the ones that share a channel,
  // and ordering them to cross less, are properties of the whole set.
  //
  // A tile's ports are numbered here rather than placed, so that a renderer
  // free to move them can stack them in a fixed order instead of scattering
  // them around the perimeter by whichever bundle each faces.
  std::map<std::pair<std::pair<int, int>, AIE::Facing>, int> switchSlots;
  std::map<std::pair<int, int>, int> dmaSlots;
  std::map<PortNode, AIE::ArrowEnd> portEnds;
  for (const PortNode &port : ports) {
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
    portEnds[port] = {portNodeID(port),
                      tile,
                      dma ? AIE::EndKind::DMA : AIE::EndKind::Switch,
                      dma ? dmaSlots[tile]++ : switchSlots[{tile, facing}]++,
                      portPosition(port, layout),
                      dma ? 0.41 : 0.045,
                      dma ? 0.09 : 0.045,
                      facing};
  }

  SmallVector<ArrowRequest> arrowRequests;
  std::map<Segment, unsigned> arrowIDs;
  std::map<BufferSegment, unsigned> bufferArrowIDs;
  auto addArrow = [&](AIE::ArrowEnd from, AIE::ArrowEnd to, unsigned group) {
    unsigned id = arrowRequests.size();
    arrowRequests.push_back(
        {id, std::move(from), std::move(to), group,
         weightScale.width(group < groupBytes.size() ? groupBytes[group] : 0)});
    return id;
  };
  for (const auto &[segment, segmentRoutes] : segments)
    arrowIDs[segment] =
        addArrow(portEnds[segment.source], portEnds[segment.dest],
                 groups.routeGroups[segmentRoutes.front()->id]);

  double localNodeWidth = std::clamp(tileWidth - 0.4, 1.0, 1.75);

  auto argumentPoint = [&, localNodeWidth](size_t index) -> Point {
    std::set<int> designColumns;
    for (auto [col, row] : designTiles)
      designColumns.insert(col);
    double designLeft = layout.tileCenterX(*designColumns.begin());
    double designRight = layout.tileCenterX(*designColumns.rbegin());
    double center = (designLeft + designRight) / 2;
    double spacing = localNodeWidth + 0.3;
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

  // The stub from a DMA channel to the buffer it serves is the one arrow that
  // enters a tile, so it is routed with the rest rather than left diagonal.
  auto bufferPoint = [&](Operation *buffer, int col, int row) -> Point {
    auto index = bufferLocalIndices.find(buffer);
    return {layout.tileCenterX(col), index == bufferLocalIndices.end()
                                         ? layout.tileCenterY(col, row)
                                         : localNodeY(col, row, index->second)};
  };
  for (const auto &[segment, segmentRoutes] : bufferSegments) {
    AIE::ArrowEnd port = portEnds[segment.port];
    AIE::ArrowEnd content{
        bufferNodeID(bufferIDs.count(segment.buffer) ? bufferIDs[segment.buffer]
                                                     : 0),
        {segment.port.col, segment.port.row},
        AIE::EndKind::Content,
        0,
        bufferPoint(segment.buffer, segment.port.col, segment.port.row),
        localNodeWidth / 2,
        0.17};
    unsigned group = segmentRoutes.empty()
                         ? 0
                         : groups.routeGroups[segmentRoutes.front()->id];
    bufferArrowIDs[segment] = segment.intoBuffer
                                  ? addArrow(port, content, group)
                                  : addArrow(content, port, group);
  }

  std::map<std::pair<unsigned, Operation *>, unsigned> callArrowIDs;
  for (const CallInfo &call : calls) {
    auto localIndex = callLocalIndices.find(call.id);
    if (localIndex == callLocalIndices.end())
      continue;
    Point callAt{layout.tileCenterX(call.col),
                 localNodeY(call.col, call.row, localIndex->second)};
    for (AIE::BufferOp buffer : call.buffers) {
      if (!bufferIDs.count(buffer))
        continue;
      AIE::ArrowEnd from{bufferNodeID(bufferIDs[buffer]),
                         {call.col, call.row},
                         AIE::EndKind::Content,
                         0,
                         bufferPoint(buffer, call.col, call.row),
                         localNodeWidth / 2,
                         0.17};
      AIE::ArrowEnd to{callNodeID(call.id),
                       {call.col, call.row},
                       AIE::EndKind::Content,
                       1,
                       callAt,
                       localNodeWidth / 2,
                       0.15};
      callArrowIDs[{call.id, buffer}] = addArrow(from, to, 0);
    }
  }

  // Runtime arguments sit below the array and reach a shim DMA, so they get
  // routed like anything else rather than cutting a diagonal across it.
  std::map<std::pair<unsigned, DMAChannelKey>, unsigned> runtimeArrowIDs;
  for (auto [index, argument] : llvm::enumerate(runtimeArguments)) {
    for (const DMAChannelKey &key : argument.channels) {
      auto [col, row, direction, channel] = key;
      PortNode port{col, row, AIE::WireBundle::DMA, channel, direction};
      AIE::ArrowEnd outside{runtimeArgumentNodeID(argument.id),
                            {col, row},
                            AIE::EndKind::External,
                            0,
                            argumentPoint(index),
                            localNodeWidth / 2,
                            0.15};
      AIE::ArrowEnd at =
          portEnds.count(port) ? portEnds[port] : AIE::ArrowEnd{};
      unsigned group =
          argument.flowGroups.empty() ? 0 : *argument.flowGroups.begin();
      runtimeArrowIDs[{argument.id, key}] =
          direction == AIE::DMAChannelDir::S2MM ? addArrow(at, outside, group)
                                                : addArrow(outside, at, group);
    }
  }

  std::map<std::string, Point> placedNodes =
      arrows->nodePositions(arrowRequests, layout);
  std::map<unsigned, ArrowPath> arrowPaths;
  for (ArrowPath &path : arrows->route(arrowRequests, layout))
    arrowPaths[path.id] = std::move(path);
  // Graphviz draws no arrowhead on a supplied spline unless it is told where
  // the tip goes. Only the hop that reaches a core or a DMA is told: a flow
  // has one destination, and a head on every switchbox it passes through
  // suggests it has many.
  auto writeArrowGeometry = [&](unsigned id, bool tipped) {
    auto found = arrowPaths.find(id);
    if (found == arrowPaths.end())
      return;
    ArrayRef<Point> points = found->second.points;
    output << ", pos=\"";
    if (tipped)
      output << "e," << layout.pos(points.back().first, points.back().second)
             << ' ';
    for (auto [index, point] : llvm::enumerate(points)) {
      if (index)
        output << ' ';
      output << layout.pos(point.first, point.second);
    }
    output << '"';
  };
  auto portPoint = [&](const PortNode &port) {
    auto found = placedNodes.find(portNodeID(port));
    return found == placedNodes.end() ? portPosition(port, layout)
                                      : found->second;
  };

  output << "digraph aie_routes {\n"
         << (layout.preLaidOut ? "  graph [layout=nop2, overlap=true"
                               : "  graph [layout=neato, overlap=true")
         << (semanticView ? ", splines=curved" : "")
         << ", outputorder=nodesfirst, bgcolor=\"" << palette.background
         << "\", pad=\"0.45\"];\n"
         << "  node [fontname=\"Helvetica\"];\n"
         << "  edge [fontname=\"Helvetica\", fontsize=9, arrowsize=0.65];\n";
  if (weightScale.enabled) {
    // Anchored below the shim row, where no tile or runtime argument sits.
    double legendX = layout.tileCenterX(0) - options.tileWidth / 2;
    // Below everything else, including the runtime arguments.
    double legendY = (runtimeArguments.empty() ? layout.channelY(0)
                                               : argumentPoint(0).second) -
                     1.0;
    std::vector<uint64_t> samples{weightScale.lightest, weightScale.heaviest};
    if (weightScale.heaviest > weightScale.lightest)
      samples.insert(samples.begin() + 1,
                     static_cast<uint64_t>(
                         std::sqrt(static_cast<double>(weightScale.lightest) *
                                   static_cast<double>(weightScale.heaviest))));
    samples.erase(llvm::unique(samples), samples.end());
    output << "  legend_title [shape=plaintext, pos=\""
           << layout.pos(legendX + 0.75, legendY + 0.45)
           << "\", label=\"bytes moved per iteration\", fontsize=10, "
              "fontcolor=\""
           << palette.legendText << "\", class=\"legend\"];\n";
    for (auto [index, bytes] : llvm::enumerate(samples)) {
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
             << palette.legendRule
             << "\", arrowhead=none, class=\"legend\"];\n";
    }
  }
  for (int col = 0; col < model.columns(); ++col) {
    for (int row = 0; row < model.rows(); ++row) {
      if (!designTiles.count({col, row}) ||
          (semanticView && !topologyTiles.count({col, row})))
        continue;
      StringRef fill = model.isCoreTile(col, row)      ? palette.coreTile
                       : model.isMemTile(col, row)     ? palette.memTile
                       : model.isShimNOCTile(col, row) ? palette.shimTile
                                                       : palette.otherTile;
      // A role fill replaces the tile-kind fill, so that colour carries one
      // meaning at a time. Only repeated roles take one; a tile left in its
      // kind colour is one of a kind.
      auto roleClass = roleClasses.find({col, row});
      if (options.tileRoles && roleClass != roleClasses.end())
        fill = roleFill(palette, roleClass->second.id);
      output << "  " << tileNodeID(col, row) << " [shape=box";
      output << ", fixedsize=true, width=" << dotCoordinate(tileWidth)
             << ", height=" << dotCoordinate(layout.tileHeight(col, row))
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
      output << ", style=\"rounded,filled\", fillcolor=\"" << fill
             << "\", color=\"" << palette.tileEdge << "\", fontcolor=\""
             << palette.tileText << "\", penwidth=\"2.0\"];\n";
    }
  }
  if (!semanticView) {
    for (const PortNode &port : ports) {
      auto [x, y] = portPoint(port);
      output << "  " << portNodeID(port);
      if (port.bundle == AIE::WireBundle::DMA) {
        AIE::DMAChannelDir direction =
            port.dmaDirection.value_or(AIE::DMAChannelDir::MM2S);
        DMAChannelKey key{port.col, port.row, direction, port.channel};
        output << " [shape=box, fixedsize=true, width=0.82, height=0.18, pos=\""
               << layout.pos(x, y) << "\", label=\"" << shortPortName(port)
               << "\", class=\"" << flowClasses(channelFlowGroups[key], "port")
               << "\", fontsize=7, style=filled, fillcolor=\""
               << palette.boxFill << "\", fontcolor=\"" << palette.tileText
               << "\", "
               << "color=\"" << palette.boxEdge << "\"];\n";
      } else {
        StringRef kind =
            port.bundle == AIE::WireBundle::Core ? "port" : "port detail-vias";
        output << " [shape=point, width=0.09, pos=\"" << layout.pos(x, y)
               << "\", xlabel=\"" << shortPortName(port) << "\", class=\""
               << flowClasses(portFlowGroups[port], kind) << "\"];\n";
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
             << flowClasses(matchingGroups,
                            "arbitration detail-arbitration detail-vias")
             << "\", fontsize=6, style=filled, fillcolor=\"" << palette.ruleFill
             << "\", color=\"" << palette.ruleEdge << "\", fontcolor=\""
             << palette.ruleText << "\"];\n";
    }
  } else {
    for (const PortNode &port : ports) {
      std::string id = portNodeID(port);
      if (!topologyNodes.count(id))
        continue;
      auto [x, y] = portPoint(port);
      std::set<unsigned> nodeGroups;
      for (const SemanticEdge &edge : topologyEdges)
        if (edge.source == id || edge.dest == id)
          nodeGroups.insert(edge.flowGroup);
      output << "  " << id
             << " [shape=box, fixedsize=true, width=0.82, height=0.18, pos=\""
             << layout.pos(x, y) << "\", label=\"" << shortPortName(port)
             << "\", class=\"" << flowClasses(nodeGroups, "port")
             << "\", fontsize=7, style=filled, fillcolor=\"" << palette.boxFill
             << "\", fontcolor=\"" << palette.tileText << "\", "
             << "color=\"" << palette.boxEdge << "\"];\n";
    }
  }
  for (const BufferInfo &buffer : *buffers) {
    auto localIndex = bufferLocalIndices.find(buffer.op);
    if (localIndex == bufferLocalIndices.end())
      continue;
    double x = layout.tileCenterX(buffer.col);
    double y = localNodeY(buffer.col, buffer.row, localIndex->second);
    output << "  " << bufferNodeID(buffer.id)
           << " [shape=box, fixedsize=true, width="
           << dotCoordinate(localNodeWidth) << ", height=0.34, pos=\""
           << layout.pos(x, y) << '"';
    output << ", label=\"" << bufferLabel(buffer.op, buffer.id)
           << "\", class=\""
           << flowClasses(bufferFlowGroups[buffer.op], "buffer detail-buffers")
           << "\", fontsize=8, style=filled, fillcolor=\"" << palette.boxFill
           << "\", fontcolor=\"" << palette.tileText << "\", "
           << "color=\"" << palette.boxEdge << "\"];\n";
  }
  if (!runtimeArguments.empty()) {
    for (auto [index, argument] : llvm::enumerate(runtimeArguments)) {
      double x = argumentPoint(index).first;
      double argumentY = argumentPoint(index).second;
      output << "  " << runtimeArgumentNodeID(argument.id)
             << " [shape=ellipse, fixedsize=true, width="
             << dotCoordinate(localNodeWidth) << ", height=0.3, pos=\""
             << layout.pos(x, argumentY) << "\", label=\"%"
             << escapeDotLabel(argument.name) << "\", class=\""
             << flowClasses(argument.flowGroups, "runtime-argument")
             << "\", fontsize=8, style=\"rounded,filled\", fillcolor=\""
             << palette.argumentFill << "\", color=\"" << palette.argumentEdge
             << "\", fontcolor=\"" << palette.tileText << "\"];\n";
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
                 << "\", color=\"" << palette.runtimeLink
                 << "\", penwidth=\"1.2\"";
          auto arrow = runtimeArrowIDs.find({argument.id, key});
          if (arrow != runtimeArrowIDs.end())
            writeArrowGeometry(arrow->second, /*tipped=*/true);
          output << "];\n";
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
             << layout.pos(x, y) << "\", label=\""
             << escapeDotLabel(call.callee) << "\", class=\""
             << flowClasses(callFlowGroups[call.id], "call detail-calls")
             << "\", fontsize=8, style=\"rounded,filled\", "
                "fillcolor=\""
             << palette.callFill << "\", color=\"" << palette.callEdge
             << "\", fontcolor=\"" << palette.tileText << "\"];\n";
      if (semanticView)
        continue;
      for (AIE::BufferOp buffer : call.buffers) {
        auto bufferID = bufferIDs.find(buffer);
        if (bufferID != bufferIDs.end()) {
          output << "  " << bufferNodeID(bufferID->second) << " -> "
                 << callNodeID(call.id) << " [class=\""
                 << flowClasses(bufferFlowGroups[buffer],
                                "call-link detail-calls detail-buffers")
                 << "\", color=\"" << palette.callLink << "\", arrowsize=0.55";
          auto arrow = callArrowIDs.find({call.id, buffer});
          if (arrow != callArrowIDs.end())
            writeArrowGeometry(arrow->second, /*tipped=*/true);
          output << "];\n";
        }
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
      output << flowColor(palette, groupID);
      first = false;
    }
    if (first)
      output << palette.quietEdge;
  };
  auto writeLabel = [&](const FlowRoute &route) {
    unsigned groupID = groups.routeGroups[route.id];
    output << "<FONT COLOR=\"" << flowColor(palette, groupID) << "\">F"
           << groupID;
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
          output << "<FONT COLOR=\"" << palette.tileText << "\">&#215;"
                 << edgeCounts[endpoints] << "</FONT>";
        output << '>';
      }
      output << "];\n";
    }
    output << "}\n";
    return success();
  }
  auto writeGeometry = [&](const Segment &segment) {
    writeArrowGeometry(arrowIDs.at(segment), arrowSegments.count(segment) > 0);
  };
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
    writeGeometry(segment);
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
    auto arrow = bufferArrowIDs.find(segment);
    if (arrow != bufferArrowIDs.end())
      writeArrowGeometry(arrow->second, /*tipped=*/true);
    output << ", style=dashed];\n";
  }
  output << "}\n";
  return success();
}

} // namespace

LogicalResult AIE::emitRouteDot(ModuleOp module, raw_ostream &output,
                                const AIE::VisualizeOptions &options) {
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

  return emitDeviceRouteDot(device, output, runtimeSequence, options);
}
