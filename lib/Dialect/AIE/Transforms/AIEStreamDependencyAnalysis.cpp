//===- AIEStreamDependencyAnalysis.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <deque>
#include <set>
#include <variant>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

using ChannelKey = std::tuple<int, int, DMAChannelDir, int>;

ChannelKey channelKey(TileID tile, DMAChannelDir dir, int channel) {
  return {tile.col, tile.row, dir, channel};
}

std::optional<TileID> tileOf(Value tile) {
  if (auto tileOp = dyn_cast_or_null<TileOp>(tile.getDefiningOp()))
    return tileOp.getTileID();
  return std::nullopt;
}

struct DmaChannelProgram {
  Operation *op;
  TileID tile;
  DMAChannelDir dir;
  int channel;
  SmallVector<Block *> bds;
  SmallVector<Region *> regions;
  // Whether the BD chain runs forever.
  bool loops = false;
};

std::optional<DmaChannelProgram> makeProgram(Operation *op, DeviceOp device) {
  auto parentTile = [](Operation *op) -> std::optional<TileID> {
    if (auto element = op->getParentOfType<TileElement>())
      return tileOf(element.getTile());
    return std::nullopt;
  };
  if (auto start = dyn_cast<DMAStartOp>(op)) {
    std::optional<TileID> tile = parentTile(op);
    std::optional<int32_t> channel = start.getChannelIndex();
    if (!tile || !channel)
      return std::nullopt;
    DmaChannelProgram p{op, *tile, start.getChannelDir(), *channel};
    llvm::SmallPtrSet<Block *, 8> seen;
    for (Block *b = start.getDest(); b;) {
      if (!seen.insert(b).second) {
        p.loops = true;
        break;
      }
      p.bds.push_back(b);
      auto next = dyn_cast<NextBDOp>(b->getTerminator());
      b = next ? next.getDest() : nullptr;
    }
    return p;
  }
  if (auto dma = dyn_cast<DMAOp>(op)) {
    std::optional<TileID> tile = parentTile(op);
    if (!tile)
      return std::nullopt;
    DmaChannelProgram p{op, *tile, dma.getChannelDir(),
                        static_cast<int>(dma.getChannelIndex())};
    for (Region &r : dma.getBds())
      p.regions.push_back(&r);
    p.loops = dma.getLoop();
    return p;
  }
  if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op)) {
    std::optional<TileID> tile = tileOf(task.getTile());
    if (!tile)
      return std::nullopt;
    DmaChannelProgram p{op, *tile, task.getDirection(),
                        static_cast<int>(task.getChannel())};
    p.regions.push_back(&task.getBody());
    return p;
  }
  if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
    auto alloc = ShimDMAAllocationOp::getForSymbol(
        device, task.getAlloc().getRootReference());
    if (!alloc)
      return std::nullopt;
    std::optional<TileID> tile = tileOf(alloc.getTile());
    if (!tile)
      return std::nullopt;
    DmaChannelProgram p{op, *tile, alloc.getChannelDir(),
                        static_cast<int>(alloc.getChannelIndex())};
    p.regions.push_back(&task.getBody());
    return p;
  }
  return std::nullopt;
}

// Every DMA channel program: aie.dma_start and aie.dma inside the tile DMA
// ops, and runtime-configured tasks.
SmallVector<DmaChannelProgram> collectDmaPrograms(DeviceOp device) {
  SmallVector<DmaChannelProgram> programs;
  device.walk([&](Operation *op) {
    if (isa<DMAStartOp, DMAOp, AIEX::DMAConfigureTaskOp,
            AIEX::DMAConfigureTaskForOp>(op))
      if (std::optional<DmaChannelProgram> p = makeProgram(op, device))
        programs.push_back(std::move(*p));
  });
  return programs;
}

template <typename OpT>
void forEachInProgram(const DmaChannelProgram &p,
                      llvm::function_ref<void(OpT)> fn) {
  for (Block *b : p.bds)
    for (auto op : b->getOps<OpT>())
      fn(op);
  for (Region *r : p.regions)
    r->walk([&](OpT op) { fn(op); });
}

std::optional<ChannelKey> channelOfSymbol(DeviceOp device, StringRef symbol) {
  auto alloc = ShimDMAAllocationOp::getForSymbol(device, symbol);
  if (!alloc)
    return std::nullopt;
  std::optional<TileID> tile = tileOf(alloc.getTile());
  if (!tile)
    return std::nullopt;
  return channelKey(*tile, alloc.getChannelDir(),
                    static_cast<int>(alloc.getChannelIndex()));
}

std::optional<ChannelKey> channelOfTask(DeviceOp device, Value task) {
  Operation *op = task.getDefiningOp();
  auto onTile = [](Value tileValue, DMAChannelDir dir,
                   int channel) -> std::optional<ChannelKey> {
    std::optional<TileID> tile = tileOf(tileValue);
    if (!tile)
      return std::nullopt;
    return channelKey(*tile, dir, channel);
  };
  if (auto configure = dyn_cast_or_null<AIEX::DMAConfigureTaskOp>(op))
    return onTile(configure.getTile(), configure.getDirection(),
                  configure.getChannel());
  if (auto chain = dyn_cast_or_null<AIEX::DMAStartBdChainOp>(op))
    return onTile(chain.getTile(), chain.getDirection(), chain.getChannel());
  if (auto configure = dyn_cast_or_null<AIEX::DMAConfigureTaskForOp>(op))
    return channelOfSymbol(device, configure.getAlloc().getRootReference());
  if (auto chain = dyn_cast_or_null<AIEX::DMAStartBdChainForOp>(op))
    return channelOfSymbol(device, chain.getAlloc());
  return std::nullopt;
}

// Packet ids each sending channel is programmed with.
std::map<ChannelKey, std::set<int>> collectSentPacketIDs(DeviceOp device) {
  std::map<ChannelKey, std::set<int>> ids;
  for (const DmaChannelProgram &p : collectDmaPrograms(device)) {
    if (p.dir != DMAChannelDir::MM2S)
      continue;
    forEachInProgram<DMABDOp>(p, [&](DMABDOp bd) {
      if (std::optional<PacketInfoAttr> packet = bd.getPacket())
        ids[channelKey(p.tile, p.dir, p.channel)].insert(packet->getPktId());
    });
  }
  for (auto alloc : device.getOps<ShimDMAAllocationOp>()) {
    std::optional<PacketInfoAttr> packet = alloc.getPacket();
    std::optional<TileID> tile = tileOf(alloc.getTile());
    if (packet && tile)
      ids[channelKey(*tile, alloc.getChannelDir(),
                     static_cast<int>(alloc.getChannelIndex()))]
          .insert(packet->getPktId());
  }
  device.walk([&](AIEX::NpuDmaMemcpyNdOp memcpy) {
    std::optional<PacketInfoAttr> packet = memcpy.getPacket();
    if (!packet)
      return;
    if (std::optional<ChannelKey> key =
            channelOfSymbol(device, memcpy.getMetadata().getRootReference()))
      ids[*key].insert(packet->getPktId());
  });
  return ids;
}

bool isDirectional(WireBundle bundle) {
  return bundle == WireBundle::North || bundle == WireBundle::South ||
         bundle == WireBundle::East || bundle == WireBundle::West;
}

class StreamTracer {
public:
  explicit StreamTracer(DeviceOp device) {
    for (auto sb : device.getOps<SwitchboxOp>())
      switchboxes[sb.getTileOp().getTileID()] = sb;
    for (auto mux : device.getOps<ShimMuxOp>())
      shimMuxes[mux.getTileOp().getTileID()] = mux;
    sentIDs = collectSentPacketIDs(device);
  }

  std::vector<RoutedStream> trace() {
    for (auto &[tile, mux] : shimMuxes)
      for (Port p : inputPorts(mux.getConnections()))
        if (p.bundle != WireBundle::North)
          traceFrom({tile, p}, mux, p);
    for (auto &[tile, sb] : switchboxes)
      for (Port p : inputPorts(sb.getConnections()))
        if (!isDirectional(p.bundle))
          traceFrom({tile, p}, sb, p);
    return std::move(streams);
  }

private:
  struct Hop {
    Operation *interconnect;
    Port input;
  };

  static SmallVector<Port> inputPorts(Region &connections) {
    SmallVector<Port> ports;
    for (Operation &op : connections.front()) {
      std::optional<Port> p;
      if (auto connect = dyn_cast<ConnectOp>(op))
        p = connect.sourcePort();
      else if (auto rules = dyn_cast<PacketRulesOp>(op))
        p = rules.sourcePort();
      if (p && !llvm::is_contained(ports, *p))
        ports.push_back(*p);
    }
    return ports;
  }

  void traceFrom(StreamEndpoint src, Operation *interconnect, Port input) {
    std::optional<std::set<int>> ids;
    // A tile sends from its MM2S channel on the port of the same number.
    if (src.port.bundle == WireBundle::DMA) {
      auto it = sentIDs.find(
          channelKey(src.tile, DMAChannelDir::MM2S, src.port.channel));
      if (it != sentIDs.end())
        ids = it->second;
    }
    if (!ids) {
      step(src, {interconnect, input}, std::nullopt, {}, {});
      return;
    }
    for (int id : *ids)
      step(src, {interconnect, input}, id, {}, {});
  }

  // The switchbox input a master port of `from` drives, or the tile port it
  // leaves the fabric at.
  std::variant<Hop, StreamEndpoint> follow(Operation *from, Port out) {
    TileID here = isa<SwitchboxOp>(from)
                      ? cast<SwitchboxOp>(from).getTileOp().getTileID()
                      : cast<ShimMuxOp>(from).getTileOp().getTileID();
    if (isa<ShimMuxOp>(from)) {
      if (out.bundle == WireBundle::North)
        if (auto sb = switchboxes.find(here); sb != switchboxes.end())
          return Hop{sb->second, {WireBundle::South, out.channel}};
      return StreamEndpoint{here, out};
    }
    if (!isDirectional(out.bundle))
      return StreamEndpoint{here, out};
    if (out.bundle == WireBundle::South)
      if (auto mux = shimMuxes.find(here); mux != shimMuxes.end())
        return Hop{mux->second, {WireBundle::North, out.channel}};
    TileID next = here;
    WireBundle in = WireBundle::South;
    switch (out.bundle) {
    case WireBundle::North:
      next.row++;
      in = WireBundle::South;
      break;
    case WireBundle::South:
      next.row--;
      in = WireBundle::North;
      break;
    case WireBundle::East:
      next.col++;
      in = WireBundle::West;
      break;
    default:
      next.col--;
      in = WireBundle::East;
      break;
    }
    if (auto sb = switchboxes.find(next); sb != switchboxes.end())
      return Hop{sb->second, {in, out.channel}};
    return StreamEndpoint{here, out};
  }

  void step(StreamEndpoint src, Hop hop, std::optional<int> id,
            llvm::DenseSet<std::pair<Operation *, int>> visited,
            SmallVector<StreamHop, 8> path) {
    int inputKey =
        static_cast<int>(hop.input.bundle) * 1024 + hop.input.channel;
    if (!visited.insert({hop.interconnect, inputKey}).second)
      return;
    auto sb = dyn_cast<SwitchboxOp>(hop.interconnect);
    Region &connections =
        sb ? sb.getConnections()
           : cast<ShimMuxOp>(hop.interconnect).getConnections();
    Block &b = connections.front();

    auto next = [&](Port out, std::optional<int> nextID,
                    std::optional<int> arbiter, bool keepsPktHeader = false) {
      SmallVector<StreamHop, 8> nextPath = path;
      if (sb)
        nextPath.push_back({sb.getTileOp().getTileID(), hop.input, arbiter});
      std::variant<Hop, StreamEndpoint> to = follow(hop.interconnect, out);
      if (auto *endpoint = std::get_if<StreamEndpoint>(&to)) {
        streams.push_back(
            {src, *endpoint, nextID, keepsPktHeader, std::move(nextPath)});
        return;
      }
      step(src, std::get<Hop>(to), nextID, visited, std::move(nextPath));
    };

    for (auto connect : b.getOps<ConnectOp>())
      if (connect.sourcePort() == hop.input)
        next(connect.destPort(), id, std::nullopt);

    for (auto rules : b.getOps<PacketRulesOp>()) {
      if (rules.sourcePort() != hop.input)
        continue;
      auto route = [&](PacketRuleOp rule, int ruleID) {
        std::optional<int> arbiter;
        if (auto amsel = rule.getAmsel().getDefiningOp<AMSelOp>())
          arbiter = amsel.arbiterIndex();
        for (auto masterSet : b.getOps<MasterSetOp>())
          if (llvm::is_contained(masterSet.getAmsels(), rule.getAmsel()))
            next(masterSet.destPort(), ruleID, arbiter,
                 masterSet.getKeepPktHeader().value_or(false));
      };
      for (auto rule : rules.getRules().front().getOps<PacketRuleOp>()) {
        if (!id) {
          route(rule, rule.valueInt());
          continue;
        }
        if ((*id & rule.maskInt()) == (rule.valueInt() & rule.maskInt())) {
          route(rule, *id);
          break;
        }
      }
    }
  }

  std::map<TileID, SwitchboxOp> switchboxes;
  std::map<TileID, ShimMuxOp> shimMuxes;
  std::map<ChannelKey, std::set<int>> sentIDs;
  std::vector<RoutedStream> streams;
};

} // namespace

std::vector<RoutedStream> AIE::traceRoutedStreams(DeviceOp device) {
  return StreamTracer(device).trace();
}

std::vector<RoutedStream> AIE::requestedStreams(DeviceOp device) {
  std::vector<RoutedStream> streams;
  for (auto flow : device.getOps<FlowOp>()) {
    std::optional<TileID> srcTile = tileOf(flow.getSource());
    std::optional<TileID> dstTile = tileOf(flow.getDest());
    if (srcTile && dstTile)
      streams.push_back(
          {{*srcTile, {flow.getSourceBundle(), flow.sourceIndex()}},
           {*dstTile, {flow.getDestBundle(), flow.destIndex()}},
           std::nullopt});
  }
  for (auto flow : device.getOps<PacketFlowOp>()) {
    Block &b = flow.getPorts().front();
    for (auto src : b.getOps<PacketSourceOp>()) {
      std::optional<TileID> srcTile = tileOf(src.getTile());
      if (!srcTile)
        continue;
      for (auto dst : b.getOps<PacketDestOp>())
        if (std::optional<TileID> dstTile = tileOf(dst.getTile()))
          streams.push_back({{*srcTile, src.port()},
                             {*dstTile, dst.port()},
                             static_cast<int>(flow.IDInt()),
                             flow.getKeepPktHeader().value_or(false)});
    }
  }
  return streams;
}

StreamVolumeAnalysis::StreamVolumeAnalysis(DeviceOp device) : device(device) {
  for (const DmaChannelProgram &p : collectDmaPrograms(device))
    programs[channelKey(p.tile, p.dir, p.channel)].push_back(p.op);
}

static bool inLoop(Operation *op) {
  return op->getParentOfType<LoopLikeOpInterface>() != nullptr;
}

std::optional<uint64_t>
StreamVolumeAnalysis::sendVolume(const RoutedStream &stream) const {
  if (stream.src.port.bundle != WireBundle::DMA)
    return std::nullopt;
  ChannelKey key =
      channelKey(stream.src.tile, DMAChannelDir::MM2S, stream.src.port.channel);
  auto carries = [&](std::optional<PacketInfoAttr> packet) {
    return !packet || !stream.packetID ||
           static_cast<int>(packet->getPktId()) == *stream.packetID;
  };
  constexpr uint64_t pktHeaderBytes = 4;
  auto headerBytes = [&](std::optional<PacketInfoAttr> packet) -> uint64_t {
    return packet && stream.keepsPktHeader ? pktHeaderBytes : 0;
  };
  bool known = false;
  uint64_t total = 0;
  if (auto it = programs.find(key); it != programs.end()) {
    for (Operation *op : it->second) {
      std::optional<DmaChannelProgram> p = makeProgram(op, device);
      if (!p)
        return std::nullopt;
      bool carried = false;
      uint64_t bytes = 0;
      forEachInProgram<DMABDOp>(*p, [&](DMABDOp bd) {
        if (!carries(bd.getPacket()))
          return;
        carried = true;
        bytes += bd.getLenInBytes() + headerBytes(bd.getPacket());
      });
      known = true;
      if (!carried)
        continue;
      if (p->loops)
        return std::nullopt;
      uint64_t runs = 1;
      if (auto start = dyn_cast<DMAStartOp>(op)) {
        runs = start.getRepeatCount() + 1;
      } else if (auto dma = dyn_cast<DMAOp>(op)) {
        runs = dma.getRepeatCount() + 1;
      } else {
        auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op);
        auto taskFor = dyn_cast<AIEX::DMAConfigureTaskForOp>(op);
        if ((task && task.getRepeatCountVal()) ||
            (taskFor && taskFor.getRepeatCountVal()))
          return std::nullopt;
        uint64_t repeat =
            (task ? task.getRepeatCount() : taskFor.getRepeatCount()) + 1;
        runs = 0;
        for (Operation *user : op->getUsers()) {
          if (!isa<AIEX::DMAStartTaskOp>(user))
            continue;
          if (inLoop(user))
            return std::nullopt;
          runs += repeat;
        }
      }
      total += bytes * runs;
    }
  }
  bool unbounded = false;
  device.walk([&](AIEX::NpuDmaMemcpyNdOp memcpy) {
    StringRef symbol = memcpy.getMetadata().getRootReference();
    if (channelOfSymbol(device, symbol) != std::optional(key))
      return;
    auto alloc = ShimDMAAllocationOp::getForSymbol(device, symbol);
    std::optional<PacketInfoAttr> packet = memcpy.getPacket();
    if (!packet)
      packet = alloc.getPacket();
    if (!carries(packet))
      return;
    auto type = dyn_cast<BaseMemRefType>(memcpy.getMemref().getType());
    if (inLoop(memcpy) || !type || !type.getElementType().isIntOrFloat()) {
      unbounded = true;
      return;
    }
    uint64_t elements = 1;
    for (OpFoldResult size : memcpy.getMixedSizes()) {
      std::optional<int64_t> n = getConstantIntValue(size);
      if (!n) {
        unbounded = true;
        return;
      }
      elements *= *n;
    }
    known = true;
    total += elements * type.getElementTypeBitWidth() / 8 + headerBytes(packet);
  });
  if (unbounded || !known)
    return std::nullopt;
  return total;
}

std::optional<uint64_t>
StreamVolumeAnalysis::receiveCapacity(const StreamEndpoint &endpoint) const {
  if (endpoint.port.bundle != WireBundle::DMA)
    return 0;
  auto it = programs.find(
      channelKey(endpoint.tile, DMAChannelDir::S2MM, endpoint.port.channel));
  if (it == programs.end())
    return 0;
  Operation *op = it->second.front();
  std::optional<DmaChannelProgram> p = makeProgram(op, device);
  if (!p || !isa<DMAStartOp, DMAOp>(op))
    return 0;
  uint64_t passes = 1;
  if (auto start = dyn_cast<DMAStartOp>(op))
    passes += start.getRepeatCount();
  else
    passes += cast<DMAOp>(op).getRepeatCount();

  // Run the BD chain from its initial lock values until an acquire blocks.
  SmallVector<Block *> sequence(p->bds);
  for (Region *r : p->regions)
    sequence.push_back(&r->front());
  if (sequence.empty())
    return 0;
  std::map<Operation *, int64_t> lockValues;
  uint64_t bytes = 0;
  constexpr int maxBDs = 1024;
  for (int step = 0; step < maxBDs; step++) {
    size_t i = step % sequence.size();
    if (!p->loops && static_cast<uint64_t>(step) >= passes * sequence.size())
      return bytes;
    for (Operation &bdOp : *sequence[i]) {
      if (auto use = dyn_cast<UseLockOp>(bdOp)) {
        auto lock = use.getLock().getDefiningOp<LockOp>();
        APInt amount;
        if (!lock || !matchPattern(use.getValue(), m_ConstantInt(&amount)))
          return 0;
        auto [value, inserted] =
            lockValues.try_emplace(lock, lock.getInit().value_or(0));
        int64_t n = amount.getSExtValue();
        if (use.release()) {
          value->second += n;
        } else if (value->second < n) {
          return bytes;
        } else {
          value->second -= n;
        }
      } else if (auto bd = dyn_cast<DMABDOp>(bdOp)) {
        bytes += bd.getLenInBytes();
      }
    }
  }
  return std::nullopt;
}

StreamWaitGraph::StreamWaitGraph(DeviceOp device,
                                 ArrayRef<RoutedStream> streams) {
  // Who acquires and who releases each lock.
  std::map<Operation *, llvm::SetVector<unsigned>> acquirers, releasers;
  auto noteLock = [&](UseLockOp use, unsigned agent) {
    Operation *lock = use.getLock().getDefiningOp();
    if (!lock)
      return;
    if (use.getAction() == LockAction::Release)
      releasers[lock].insert(agent);
    else
      acquirers[lock].insert(agent);
  };

  // Agents whose waits the design spells out; see the end of this constructor
  // for the rest.
  for (auto core : device.getOps<CoreOp>()) {
    unsigned agent =
        getOrCreate(core.getTileOp().getTileID(), true, DMAChannelDir::MM2S, 0);
    modeled.insert(agent);
    core.walk([&](UseLockOp use) { noteLock(use, agent); });
  }
  for (const DmaChannelProgram &p : collectDmaPrograms(device)) {
    unsigned agent = getOrCreate(p.tile, false, p.dir, p.channel);
    modeled.insert(agent);
    forEachInProgram<UseLockOp>(p,
                                [&](UseLockOp use) { noteLock(use, agent); });
  }
  // Document order, not pointer order: edge order picks the chain a
  // diagnostic reports.
  for (auto lock : device.getOps<LockOp>()) {
    auto waiting = acquirers.find(lock), it = releasers.find(lock);
    if (waiting == acquirers.end() || it == releasers.end())
      continue;
    for (unsigned p : waiting->second)
      for (unsigned q : it->second)
        if (p != q)
          addEdge(p, q, EdgeKind::Lock);
  }

  auto endpointAgent = [&](const StreamEndpoint &endpoint,
                           bool sending) -> std::optional<unsigned> {
    if (endpoint.port.bundle == WireBundle::Core)
      return getOrCreate(endpoint.tile, true, DMAChannelDir::MM2S, 0);
    if (endpoint.port.bundle == WireBundle::DMA)
      return getOrCreate(endpoint.tile, false,
                         sending ? DMAChannelDir::MM2S : DMAChannelDir::S2MM,
                         endpoint.port.channel);
    return std::nullopt;
  };
  for (const RoutedStream &s : streams) {
    std::optional<unsigned> from = endpointAgent(s.src, true);
    std::optional<unsigned> to = endpointAgent(s.dst, false);
    if (!from || !to || *from == *to)
      continue;
    addEdge(*from, *to, EdgeKind::Stream);
    addEdge(*to, *from, EdgeKind::Stream);
  }

  // The host issues a channel's first transfer only after every wait before
  // it in the runtime sequence completes. Later issues repeat the pattern of
  // earlier ones, so only the first is modeled.
  for (auto sequence : device.getOps<RuntimeSequenceOp>()) {
    llvm::SetVector<unsigned> waited;
    std::set<unsigned> issued;
    auto channelAgent =
        [&](std::optional<ChannelKey> key) -> std::optional<unsigned> {
      if (!key)
        return std::nullopt;
      auto [col, row, dir, channel] = *key;
      return getOrCreate({col, row}, false, dir, channel);
    };
    auto issue = [&](std::optional<ChannelKey> key) {
      std::optional<unsigned> agent = channelAgent(key);
      if (!agent || !issued.insert(*agent).second)
        return;
      modeled.insert(*agent);
      for (unsigned w : waited)
        if (w != *agent)
          addEdge(*agent, w, EdgeKind::Host);
    };
    auto wait = [&](std::optional<ChannelKey> key) {
      if (std::optional<unsigned> agent = channelAgent(key))
        waited.insert(*agent);
    };
    auto bySymbol = [&](SymbolRefAttr symbol) {
      return channelOfSymbol(device, symbol.getRootReference());
    };
    sequence.walk([&](Operation *op) {
      if (auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op))
        issue(bySymbol(memcpy.getMetadata()));
      else if (auto start = dyn_cast<AIEX::DMAStartTaskOp>(op))
        issue(channelOfTask(device, start.getTask()));
      else if (isa<AIEX::DMAStartBdChainOp, AIEX::DMAStartBdChainForOp>(op))
        issue(channelOfTask(device, op->getResult(0)));
      else if (auto dmaWait = dyn_cast<AIEX::NpuDmaWaitOp>(op))
        wait(bySymbol(dmaWait.getSymbolAttr()));
      else if (auto await = dyn_cast<AIEX::DMAAwaitTaskOp>(op))
        wait(channelOfTask(device, await.getTask()));
    });
  }

  // A channel nothing programs here is programmed elsewhere, in ways this
  // cannot see, so it may wait on anything else on its tile.
  for (unsigned a = 0; a < agents.size(); a++) {
    if (modeled.contains(a))
      continue;
    for (unsigned b = 0; b < agents.size(); b++)
      if (b != a && agents[b].tile == agents[a].tile)
        addEdge(a, b, EdgeKind::Lock);
  }
}

unsigned StreamWaitGraph::getOrCreate(TileID tile, bool isCore,
                                      DMAChannelDir dir, int channel) {
  if (std::optional<unsigned> id = lookup(tile, isCore, dir, channel))
    return *id;
  unsigned id = agents.size();
  agents.push_back({tile, isCore, dir, channel});
  edges.emplace_back();
  agentIDs[{tile.col, tile.row, isCore, isCore ? 0 : static_cast<int>(dir),
            isCore ? 0 : channel}] = id;
  return id;
}

std::optional<unsigned> StreamWaitGraph::lookup(TileID tile, bool isCore,
                                                DMAChannelDir dir,
                                                int channel) const {
  auto it =
      agentIDs.find({tile.col, tile.row, isCore,
                     isCore ? 0 : static_cast<int>(dir), isCore ? 0 : channel});
  if (it == agentIDs.end())
    return std::nullopt;
  return it->second;
}

void StreamWaitGraph::addEdge(unsigned from, unsigned to, EdgeKind kind) {
  if (llvm::none_of(edges[from], [&](const Edge &e) {
        return e.to == to && e.kind == kind;
      }))
    edges[from].push_back({to, kind});
}

std::optional<unsigned> StreamWaitGraph::agentAt(const StreamEndpoint &endpoint,
                                                 bool sending) const {
  if (endpoint.port.bundle == WireBundle::Core)
    return lookup(endpoint.tile, true, DMAChannelDir::MM2S, 0);
  if (endpoint.port.bundle == WireBundle::DMA)
    return lookup(endpoint.tile, false,
                  sending ? DMAChannelDir::MM2S : DMAChannelDir::S2MM,
                  endpoint.port.channel);
  return std::nullopt;
}

SmallVector<unsigned> StreamWaitGraph::drainersOf(unsigned agent) const {
  if (agents[agent].isCore)
    return {agent};
  SmallVector<unsigned> drainers;
  for (const Edge &e : edges[agent])
    if (e.kind != EdgeKind::Stream)
      drainers.push_back(e.to);
  return drainers;
}

bool StreamWaitGraph::reaches(ArrayRef<unsigned> from,
                              ArrayRef<unsigned> targets,
                              ArrayRef<unsigned> avoid) const {
  return !waitChain(from, targets, avoid).empty();
}

SmallVector<unsigned>
StreamWaitGraph::waitChain(ArrayRef<unsigned> from, ArrayRef<unsigned> targets,
                           ArrayRef<unsigned> avoid) const {
  llvm::DenseMap<unsigned, std::optional<unsigned>> parent;
  for (unsigned a : avoid)
    parent.try_emplace(a, std::nullopt);
  std::deque<unsigned> worklist;
  for (unsigned a : from)
    if (parent.try_emplace(a, std::nullopt).second)
      worklist.push_back(a);
  while (!worklist.empty()) {
    unsigned a = worklist.front();
    worklist.pop_front();
    if (llvm::is_contained(targets, a)) {
      SmallVector<unsigned> chain;
      for (std::optional<unsigned> at = a; at; at = parent.lookup(*at))
        chain.push_back(*at);
      std::reverse(chain.begin(), chain.end());
      return chain;
    }
    for (const Edge &e : edges[a])
      if (parent.try_emplace(e.to, a).second)
        worklist.push_back(e.to);
  }
  return {};
}

std::string StreamWaitGraph::describe(unsigned id) const {
  const Agent &a = agents[id];
  std::string s = "(" + std::to_string(a.tile.col) + ", " +
                  std::to_string(a.tile.row) + ") ";
  if (a.isCore)
    return s + "core";
  return s + stringifyDMAChannelDir(a.dir).str() + " " +
         std::to_string(a.channel);
}

StreamDeadlockAnalysis::StreamDeadlockAnalysis(
    DeviceOp device, std::vector<RoutedStream> streams)
    : streams(std::move(streams)), volumes(device),
      graph(device, this->streams) {}

bool StreamDeadlockAnalysis::canStall(size_t f) const {
  auto [it, inserted] = stalls.try_emplace(f, false);
  if (!inserted)
    return it->second;
  const StreamEndpoint &dst = streams[f].dst;
  std::optional<uint64_t> capacity = volumes.receiveCapacity(dst);
  if (!capacity)
    return false;
  uint64_t sent = 0;
  for (const RoutedStream &s : streams) {
    if (s.dst.tile != dst.tile || s.dst.port != dst.port)
      continue;
    std::optional<uint64_t> bytes = volumes.sendVolume(s);
    if (!bytes)
      return stalls[f] = true;
    sent += *bytes;
  }
  return stalls[f] = sent > *capacity;
}

SmallVector<unsigned> StreamDeadlockAnalysis::blockingChain(size_t f,
                                                            size_t g) const {
  const RoutedStream &fs = streams[f], &gs = streams[g];
  std::optional<unsigned> fDst = graph.agentAt(fs.dst, false);
  if (!fDst)
    return {};
  SmallVector<unsigned> own;
  if (std::optional<unsigned> fSrc = graph.agentAt(fs.src, true))
    own.push_back(*fSrc);
  own.push_back(*fDst);
  SmallVector<unsigned> drainers = graph.drainersOf(*fDst);
  SmallVector<unsigned> avoid, targets;
  for (unsigned a : own)
    if (!llvm::is_contained(drainers, a))
      avoid.push_back(a);
  for (std::optional<unsigned> a :
       {graph.agentAt(gs.src, true), graph.agentAt(gs.dst, false)})
    if (a && !llvm::is_contained(own, *a))
      targets.push_back(*a);
  if (targets.empty())
    return {};
  return graph.waitChain(drainers, targets, avoid);
}

bool StreamDeadlockAnalysis::canBlock(size_t f, size_t g) const {
  auto [it, inserted] = blocks.try_emplace({f, g}, false);
  if (!inserted)
    return it->second;
  return blocks[{f, g}] = canStall(f) && !silent(f) && !silent(g) &&
                          !blockingChain(f, g).empty();
}

bool StreamDeadlockAnalysis::silent(size_t f) const {
  auto [it, inserted] = silence.try_emplace(f, false);
  if (inserted)
    it->second = volumes.sendVolume(streams[f]) == std::optional<uint64_t>(0);
  return it->second;
}

std::string StreamDeadlockAnalysis::explainBlock(size_t f, size_t g) const {
  const RoutedStream &fs = streams[f], &gs = streams[g];
  SmallVector<unsigned> chain = blockingChain(f, g);
  std::string s = describeStream(fs) + " can fill its receiver, and draining " +
                  "that waits on ";
  for (auto [i, a] : llvm::enumerate(chain))
    s += (i ? ", then " : "") + graph.describe(a);
  s += graph.agentAt(gs.dst, false) == chain.back() ? ", which receives "
                                                    : ", which sends ";
  s += describeStream(gs) + '.';
  for (const std::string &a : assumptions(f, g))
    s += ' ' + a;
  return s;
}

SmallVector<std::string> StreamDeadlockAnalysis::assumptions(size_t f,
                                                             size_t g) const {
  const RoutedStream &fs = streams[f];
  SmallVector<std::string> assumed;
  for (const RoutedStream &other : streams)
    if (other.dst.tile == fs.dst.tile && other.dst.port == fs.dst.port &&
        !volumes.sendVolume(other)) {
      assumed.push_back("The volume " + describeStream(other) +
                        " carries is unknown, so it is assumed to overrun its "
                        "receiver.");
      break;
    }
  SmallVector<unsigned> chain = blockingChain(f, g);
  SmallVector<unsigned> waiters{*graph.agentAt(fs.dst, false)};
  waiters.append(chain.begin(), std::prev(chain.end()));
  for (auto [i, a] : llvm::enumerate(waiters))
    if (!graph.isModeled(a) &&
        !llvm::is_contained(ArrayRef(waiters).take_front(i), a))
      assumed.push_back("Nothing in the design programs " + graph.describe(a) +
                        ", so it is assumed to wait on anything on its tile.");
  return assumed;
}

std::string AIE::describeStream(const RoutedStream &stream) {
  auto endpoint = [](const StreamEndpoint &e) {
    return "(" + std::to_string(e.tile.col) + ", " +
           std::to_string(e.tile.row) + ") " +
           stringifyWireBundle(e.port.bundle).str() + ":" +
           std::to_string(e.port.channel);
  };
  std::string s = (stream.packetID ? "packet flow " : "flow ") +
                  endpoint(stream.src) + " -> " + endpoint(stream.dst);
  if (stream.packetID)
    s += " (id " + std::to_string(*stream.packetID) + ")";
  return s;
}

StreamConflicts::StreamConflicts(DeviceOp device)
    : device(device), streams(requestedStreams(device)),
      numRequested(streams.size()) {
  for (RoutedStream &s : traceRoutedStreams(device))
    streams.push_back(std::move(s));
  std::map<std::tuple<TileID, Port, int, bool>, size_t> treeIDs;
  for (size_t i = 0; i < streams.size(); i++) {
    const RoutedStream &s = streams[i];
    size_t tree = treeMembers.size();
    if (s.packetID)
      tree =
          treeIDs
              .try_emplace(
                  {s.src.tile, s.src.port, *s.packetID, i < numRequested}, tree)
              .first->second;
    if (tree == treeMembers.size())
      treeMembers.emplace_back();
    treeMembers[tree].push_back(i);
    treeOf.push_back(tree);
  }
}

static bool sameEndpoint(const StreamEndpoint &x, const StreamEndpoint &y) {
  return x.tile == y.tile && x.port == y.port;
}

StreamDeadlockAnalysis &StreamConflicts::getAnalysis() {
  if (!analysis)
    analysis.emplace(device, streams);
  return *analysis;
}

bool StreamConflicts::blocks(size_t s, size_t t) {
  const RoutedStream &a = streams[s], &b = streams[t];
  if (sameEndpoint(a.src, b.src) || sameEndpoint(a.dst, b.dst))
    return false;
  return getAnalysis().canBlock(s, t);
}

// Trees from one source, or into one receiver, already wait on each other
// there whatever the routing.
bool StreamConflicts::related(size_t s, size_t t) const {
  if (sameEndpoint(streams[s].src, streams[t].src))
    return true;
  for (size_t m : treeMembers[treeOf[s]])
    for (size_t n : treeMembers[treeOf[t]])
      if (sameEndpoint(streams[m].dst, streams[n].dst))
        return true;
  return false;
}

bool StreamConflicts::conflict(size_t s, size_t t) {
  return !related(s, t) && (blocks(s, t) || blocks(t, s));
}

std::string StreamConflicts::explain(size_t s, size_t t) {
  StreamDeadlockAnalysis &a = getAnalysis();
  return a.canBlock(s, t) ? a.explainBlock(s, t) : a.explainBlock(t, s);
}

SmallVector<std::pair<size_t, size_t>> StreamConflicts::unavoidable() {
  SmallVector<std::pair<size_t, size_t>> pairs;
  for (size_t s = 0; s < numRequested; s++)
    for (size_t t = 0; t < numRequested; t++)
      if (s != t && related(s, t) && getAnalysis().canBlock(s, t))
        pairs.push_back({s, t});
  return pairs;
}

std::optional<HoldCycle>
StreamConflicts::holdCycle(ArrayRef<SmallVector<StreamHop, 8>> routes) {
  // The packets one source sends with one id move down every branch as one.
  struct Tree {
    SmallVector<size_t, 2> members;
    SmallVector<std::pair<TileID, Port>, 8> hops;
    SmallVector<int, 8> parent;
    SmallVector<std::optional<int>, 8> arbiter;
    size_t base = 0;
  };
  std::vector<Tree> trees;
  std::map<std::tuple<TileID, Port, int, bool>, size_t> treeIDs;
  for (size_t i = 0; i < streams.size(); i++) {
    const RoutedStream &s = streams[i];
    if (!s.packetID || getAnalysis().silent(i))
      continue;
    auto [it, inserted] = treeIDs.try_emplace(
        {s.src.tile, s.src.port, *s.packetID, i < numRequested}, trees.size());
    if (inserted)
      trees.emplace_back();
    Tree &tree = trees[it->second];
    tree.members.push_back(i);
    int prev = -1;
    for (const StreamHop &hop : routes[i]) {
      std::pair<TileID, Port> key{hop.tile, hop.input};
      int h = llvm::find(tree.hops, key) - tree.hops.begin();
      if (h == static_cast<int>(tree.hops.size())) {
        tree.hops.push_back(key);
        tree.parent.push_back(prev);
        tree.arbiter.push_back(hop.arbiter);
      }
      prev = h;
    }
  }

  auto related = [&](size_t a, size_t b) {
    return this->related(trees[a].members.front(), trees[b].members.front());
  };

  // Per tree, a node for a head of it stuck entering each hop or at each
  // receiver, one for it stuck anywhere, and one per hop for it stuck
  // anywhere past that hop, so still holding what it took there.
  enum class Kind { Hop, Receiver, Anywhere, Past };
  struct Node {
    size_t tree;
    Kind kind;
    size_t index;
  };
  std::vector<Node> nodes;
  for (size_t t = 0; t < trees.size(); t++) {
    Tree &tree = trees[t];
    tree.base = nodes.size();
    for (size_t h = 0; h < tree.hops.size(); h++)
      nodes.push_back({t, Kind::Hop, h});
    for (size_t m = 0; m < tree.members.size(); m++)
      nodes.push_back({t, Kind::Receiver, m});
    nodes.push_back({t, Kind::Anywhere, 0});
    for (size_t h = 0; h < tree.hops.size(); h++)
      nodes.push_back({t, Kind::Past, h});
  }
  auto receiverNode = [&](size_t t, size_t m) {
    return trees[t].base + trees[t].hops.size() + m;
  };
  auto anywhereNode = [&](size_t t) {
    return receiverNode(t, trees[t].members.size());
  };
  auto pastNode = [&](size_t t, size_t h) { return anywhereNode(t) + 1 + h; };

  std::map<std::pair<TileID, Port>, SmallVector<std::pair<size_t, size_t>, 4>>
      entering;
  std::map<TileID, SmallVector<std::pair<size_t, size_t>, 8>> passing;
  for (size_t t = 0; t < trees.size(); t++)
    for (size_t h = 0; h < trees[t].hops.size(); h++) {
      entering[trees[t].hops[h]].push_back({t, h});
      passing[trees[t].hops[h].first].push_back({t, h});
    }

  struct Edge {
    size_t to;
    std::optional<HoldCycle::Step> step;
  };
  std::vector<std::optional<SmallVector<Edge, 4>>> edges(nodes.size());
  auto successors = [&](size_t n) -> ArrayRef<Edge> {
    if (edges[n])
      return *edges[n];
    SmallVector<Edge, 4> out;
    const Node &node = nodes[n];
    const Tree &u = trees[node.tree];
    switch (node.kind) {
    case Kind::Hop: {
      auto [tile, input] = u.hops[node.index];
      size_t waiting = u.members.front();
      for (auto [t, ht] : entering.at({tile, input})) {
        if (t != node.tree && related(node.tree, t))
          continue;
        size_t sharer = trees[t].members.front();
        if (t != node.tree)
          out.push_back({pastNode(t, ht),
                         HoldCycle::Step{HoldCycle::Wait::Link, waiting, sharer,
                                         sharer, tile, input, input, -1}});
        std::optional<int> arbiter = trees[t].arbiter[ht];
        if (!arbiter)
          continue;
        for (auto [v, hv] : passing.at(tile)) {
          Port holderInput = trees[v].hops[hv].second;
          if (v == node.tree || v == t || holderInput == input ||
              trees[v].arbiter[hv] != arbiter || related(t, v))
            continue;
          out.push_back({pastNode(v, hv),
                         HoldCycle::Step{HoldCycle::Wait::Arbiter, waiting,
                                         sharer, trees[v].members.front(), tile,
                                         input, holderInput, *arbiter}});
        }
      }
      break;
    }
    case Kind::Receiver: {
      size_t s = u.members[node.index];
      for (size_t g = 0; g < trees.size(); g++) {
        if (g == node.tree)
          continue;
        for (size_t m : trees[g].members)
          if (blocks(s, m)) {
            out.push_back({anywhereNode(g),
                           HoldCycle::Step{HoldCycle::Wait::Drain, s, s, m,
                                           TileID{}, Port{}, Port{}, -1}});
            break;
          }
      }
      break;
    }
    case Kind::Anywhere:
    case Kind::Past: {
      llvm::SmallDenseSet<size_t, 8> behind;
      if (node.kind == Kind::Past)
        for (int h = node.index; h >= 0; h = u.parent[h])
          behind.insert(h);
      for (size_t h = 0; h < u.hops.size(); h++)
        if (!behind.contains(h))
          out.push_back({u.base + h, std::nullopt});
      for (size_t m = 0; m < u.members.size(); m++)
        out.push_back({receiverNode(node.tree, m), std::nullopt});
      break;
    }
    }
    return *(edges[n] = std::move(out));
  };

  // A cycle of receivers alone waiting on each other is the design's own, so
  // only cycles through a wait some requested routing adds count.
  auto counts = [&](const Edge &e) {
    if (!e.step || e.step->wait == HoldCycle::Wait::Drain)
      return false;
    return e.step->waiting < numRequested || e.step->sharer < numRequested ||
           e.step->holding < numRequested;
  };
  SmallVector<size_t> roots;
  for (size_t n = 0; n < nodes.size(); n++)
    if (nodes[n].kind == Kind::Hop && llvm::any_of(successors(n), counts))
      roots.push_back(n);

  // Tarjan's strongly connected components, iteratively.
  std::vector<int> index(nodes.size(), -1), low(nodes.size(), 0),
      component(nodes.size(), -1);
  std::vector<bool> onStack(nodes.size(), false);
  std::vector<size_t> stack;
  std::vector<std::pair<size_t, size_t>> frames;
  int counter = 0, components = 0;
  auto visit = [&](size_t n) {
    index[n] = low[n] = counter++;
    stack.push_back(n);
    onStack[n] = true;
    frames.push_back({n, 0});
  };
  for (size_t root : roots) {
    if (index[root] >= 0)
      continue;
    visit(root);
    while (!frames.empty()) {
      auto [n, next] = frames.back();
      ArrayRef<Edge> out = successors(n);
      if (next < out.size()) {
        frames.back().second++;
        size_t w = out[next].to;
        if (index[w] < 0)
          visit(w);
        else if (onStack[w])
          low[n] = std::min(low[n], index[w]);
        continue;
      }
      frames.pop_back();
      if (!frames.empty())
        low[frames.back().first] = std::min(low[frames.back().first], low[n]);
      if (low[n] != index[n])
        continue;
      size_t w;
      do {
        w = stack.back();
        stack.pop_back();
        onStack[w] = false;
        component[w] = components;
      } while (w != n);
      components++;
    }
  }

  // A counted wait within one component lies on a cycle; close it the
  // shortest way back.
  for (size_t x : roots)
    for (const Edge &e : successors(x)) {
      if (!counts(e) || component[e.to] != component[x])
        continue;
      std::map<size_t, std::pair<size_t, const Edge *>> via;
      via[e.to] = {e.to, nullptr};
      std::deque<size_t> worklist{e.to};
      while (!via.count(x)) {
        size_t n = worklist.front();
        worklist.pop_front();
        for (const Edge &next : successors(n))
          if (component[next.to] == component[x] &&
              via.try_emplace(next.to, n, &next).second)
            worklist.push_back(next.to);
      }
      SmallVector<const Edge *> path;
      for (size_t n = x; n != e.to; n = via[n].first)
        path.push_back(via[n].second);
      path.push_back(&e);
      HoldCycle cycle;
      for (const Edge *edge : llvm::reverse(path))
        if (edge->step)
          cycle.steps.push_back(*edge->step);
      return cycle;
    }
  return std::nullopt;
}

std::string StreamConflicts::explain(const HoldCycle &cycle) {
  auto tile = [](TileID t) {
    return "tile (" + std::to_string(t.col) + ", " + std::to_string(t.row) +
           ")";
  };
  std::string s;
  for (const HoldCycle::Step &step : cycle.steps) {
    if (!s.empty())
      s += ' ';
    switch (step.wait) {
    case HoldCycle::Wait::Link:
      s += describeStream(streams[step.waiting]) + " can queue behind " +
           describeStream(streams[step.holding]) + " on " +
           stringifyWireBundle(step.sharerInput.bundle).str() + ":" +
           std::to_string(step.sharerInput.channel) + " into " +
           tile(step.tile) + ".";
      break;
    case HoldCycle::Wait::Arbiter:
      s += describeStream(streams[step.holding]) + " can hold arbiter " +
           std::to_string(step.arbiter) + " at " + tile(step.tile) + " that " +
           describeStream(streams[step.sharer]) + " needs";
      if (step.waiting != step.sharer)
        s += ", and " + describeStream(streams[step.waiting]) +
             " can queue behind it";
      s += ".";
      break;
    case HoldCycle::Wait::Drain:
      s += getAnalysis().explainBlock(step.waiting, step.holding);
      break;
    }
  }
  return s;
}
