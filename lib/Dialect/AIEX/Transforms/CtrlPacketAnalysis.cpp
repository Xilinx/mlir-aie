//===- CtrlPacketAnalysis.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The analyses the control packet delivery passes share.
//
//===----------------------------------------------------------------------===//

#include "CtrlPacketClassifier.h"
#include "CtrlPacketDevices.h"
#include "CtrlPacketMulticast.h"
#include "CtrlPacketRuns.h"

#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"

#include <string>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

FailureOr<Classified> PacketClassifier::classify(NpuControlPacketOp op) {
  auto data = op.getData();
  uint32_t base = op.getLocalAddress();
  if (op.getOpcode() != 0 || !data) {
    Classified c{WriteKind::Other};
    uint32_t words = data ? data->size() : op.getLength().value_or(1);
    for (uint32_t i = 0; i < std::max(words, 1u); ++i)
      c.words.push_back(base + 4 * i);
    return c;
  }
  const TileRegisters *regs = getRegisters(op);
  if (!regs)
    return failure();
  Classified c{WriteKind::Other};
  for (auto [i, word] : llvm::enumerate(*data)) {
    uint32_t offset = base + 4 * i;
    c.words.push_back(offset);
    bool enables = regs->coreControl && offset == *regs->coreControl &&
                   (word & regs->coreEnableMask);
    if (regs->queues.contains(offset) || enables)
      c.startsSomething = true;
    if (auto it = regs->mm2sQueues.find(offset);
        it != regs->mm2sQueues.end() && c.kind != WriteKind::ChannelStart) {
      c.kind = WriteKind::ChannelStart;
      c.channel = it->second;
    }
    if (enables && c.kind == WriteKind::Other)
      c.kind = WriteKind::CoreEnable;
  }
  return c;
}

const PacketClassifier::TileRegisters *
PacketClassifier::getRegisters(NpuControlPacketOp op) {
  AIE::TileID tile = op.getTileFromAddr();
  auto it = registers.find(tile);
  if (it != registers.end())
    return &it->second;
  TileRegisters regs;
  if (tm.isCoreTile(tile.col, tile.row) || tm.isMemTile(tile.col, tile.row) ||
      tm.isShimNOCTile(tile.col, tile.row)) {
    // A shim tile's channels are started through a task queue, the others'
    // through a start queue.
    const char *queue =
        tm.isShimNOCTile(tile.col, tile.row) ? "_Task_Queue" : "_Start_Queue";
    for (const char *dir : {"MM2S", "S2MM"}) {
      for (int channel = 0;; ++channel) {
        std::string name =
            std::string("DMA_") + dir + "_" + std::to_string(channel) + queue;
        const AIE::RegisterInfo *reg =
            tm.lookupRegister(name, tile, /*isMem=*/true);
        if (!reg) {
          if (channel > 0)
            break;
          op.emitOpError("cannot classify control packets: the register "
                         "database has no ")
              << name << " register";
          return nullptr;
        }
        regs.queues.insert(reg->offset);
        if (dir[0] == 'M')
          regs.mm2sQueues[reg->offset] = channel;
      }
    }
  }
  if (tm.isCoreTile(tile.col, tile.row)) {
    const AIE::RegisterInfo *reg = tm.lookupRegister("Core_Control", tile);
    const AIE::BitFieldInfo *enable = reg ? reg->getField("Enable") : nullptr;
    std::optional<uint32_t> mask =
        enable ? tm.getFieldMask(*enable) : std::nullopt;
    if (!mask) {
      op.emitOpError("cannot classify control packets: the register "
                     "database has no Core_Control Enable field");
      return nullptr;
    }
    regs.coreControl = reg->offset;
    regs.coreEnableMask = *mask;
  }
  return &registers.try_emplace(tile, std::move(regs)).first->second;
}

SmallVector<unsigned> AIEX::awaitSteps(const GroupRun &run,
                                       bool parallelShims) {
  SmallVector<unsigned> steps;
  for (size_t i = 0; i < run.size(); ++i) {
    NpuControlPacketOp packet = run[i].packet;
    bool next =
        i && (run[i].phase != run[i - 1].phase ||
              (!parallelShims &&
               packet.getColumnFromAddr() !=
                   NpuControlPacketOp(run[i - 1].packet).getColumnFromAddr()));
    steps.push_back(i ? steps.back() + next : 0);
  }
  return steps;
}

bool AIEX::sameKindAndDevice(ControlPacketGroupOp a, ControlPacketGroupOp b) {
  return a.getKind() == b.getKind() &&
         a.getDeviceRefAttr() == b.getDeviceRefAttr();
}

SmallVector<GroupRun> AIEX::groupRuns(Block &block, RunSplit split) {
  SmallVector<GroupRun> runs;
  GroupRun run;
  ControlPacketGroupOp last;
  unsigned phase = 0;
  auto end = [&]() {
    if (!run.empty())
      runs.push_back(std::move(run));
    run.clear();
    last = nullptr;
    phase = 0;
  };
  for (Operation &op : block) {
    if (isa<ControlPacketBarrierOp>(op) && !split.atBarriers) {
      if (last)
        ++phase;
      continue;
    }
    auto group = dyn_cast<ControlPacketGroupOp>(op);
    bool changes = group && last && !sameKindAndDevice(last, group);
    if (!group ||
        (changes && split.kindOrDevice == KindOrDeviceChange::EndsRun))
      end();
    else if (changes && split.kindOrDevice == KindOrDeviceChange::NextPhase)
      ++phase;
    if (!group)
      continue;
    last = group;
    for (auto packet : group.getBody().front().getOps<NpuControlPacketOp>())
      run.push_back({packet, group, phase});
  }
  end();
  return runs;
}

SmallVector<GroupRun> AIEX::groupRuns(AIE::RuntimeSequenceOp seq,
                                      RunSplit split) {
  SmallVector<GroupRun> runs;
  seq.walk([&](Block *block) {
    if (!isa<ControlPacketGroupOp>(block->getParentOp()))
      llvm::append_range(runs, groupRuns(*block, split));
  });
  return runs;
}

std::optional<Arbiter>
ConfiguredDevices::controlArbiterIn(FlatSymbolRefAttr device, AIE::TileID tile,
                                    const std::set<Arbiter> &held) {
  for (const Arbiter &a : infoOf(device).control.lookup(tile))
    if (held.count(a))
      return a;
  return std::nullopt;
}

ConfiguredDevices::Info &ConfiguredDevices::infoOf(FlatSymbolRefAttr device) {
  auto [it, fresh] = infos.try_emplace(device.getAttr());
  if (fresh)
    it->second = std::make_unique<Info>();
  Info &info = *it->second;
  auto d = module.lookupSymbol<AIE::DeviceOp>(device);
  if (!fresh || !d)
    return info;
  AIEXDialect::StartedByAttrHelper startedBy(module.getContext());
  d.walk([&](Operation *op) {
    ArrayAttr cores = startedBy.getAttr(op);
    if (!cores)
      return;
    StreamSource source;
    if (auto core = dyn_cast<AIE::CoreOp>(op))
      source = {core.getTileOp().getTileID(), -1};
    else if (auto start = dyn_cast<AIE::DMAStartOp>(op))
      source = {cast<AIE::TileElement>(start->getParentOp())
                    .getTile()
                    .getDefiningOp<AIE::TileOp>()
                    .getTileID(),
                start.getChannelIndex()};
    else
      return;
    info.gated.insert(source);
    for (auto core : cores.getAsRange<AIE::TileAttr>())
      info.startedBy[core.getTileID()].push_back(source);
  });
  llvm::DenseMap<AIE::TileID, AIE::PacketInfoAttr> controllerIds =
      controllerIdsOf(d);
  for (const AIE::RoutedStream &s : AIE::traceRoutedStreams(d)) {
    SmallVector<Arbiter> hops;
    for (const AIE::StreamHop &hop : s.hops)
      if (hop.arbiter)
        hops.push_back({hop.tile, *hop.arbiter});
    if (s.src.port.bundle == AIE::WireBundle::DMA)
      info.sent[{s.src.tile, s.src.port.channel}].insert(hops.begin(),
                                                         hops.end());
    else if (s.src.port.bundle == AIE::WireBundle::Core)
      info.sent[{s.src.tile, -1}].insert(hops.begin(), hops.end());
    // A multicast's tree is no part of a tile's own route.
    if (AIE::PacketInfoAttr id = controllerIds.lookup(s.dst.tile);
        s.dst.port.bundle == AIE::WireBundle::TileControl &&
        s.carries(id ? std::optional(id) : std::nullopt))
      llvm::append_range(info.control[s.dst.tile], hops);
  }
  return info;
}

bool OverlayMulticasts::Info::sharesTrail(const Multicast &m,
                                          AIE::TileID tile) const {
  if (tile.col != m.column)
    return false;
  auto own = controllerIds.find(tile);
  if (own == controllerIds.end())
    return false;
  auto a = trails.find({tile, m.id});
  auto b = trails.find({tile, own->second});
  return a != trails.end() && b != trails.end() && a->second == b->second;
}

bool OverlayMulticasts::Info::keepsOrder(const Multicast &m, AIE::TileID tile,
                                         bool parallelShims) const {
  return tile.col != m.column ? !parallelShims : sharesTrail(m, tile);
}

const OverlayMulticasts::Multicast *
OverlayMulticasts::Info::multicastOf(NpuControlPacketOp packet) const {
  std::optional<uint32_t> id = packet.getMulticastId();
  AIE::TileID own = packet.getTileFromAddr();
  auto column = byTile.find(own);
  if (!id || column == byTile.end())
    return nullptr;
  const Multicast &m = multicasts.at(column->second);
  return m.id == static_cast<int32_t>(*id) && m.tiles.front() == own ? &m
                                                                     : nullptr;
}

SmallVector<AIE::TileID>
AIEX::tilesReached(NpuControlPacketOp packet,
                   const OverlayMulticasts::Info *info) {
  if (const OverlayMulticasts::Multicast *m =
          info ? info->multicastOf(packet) : nullptr)
    return m->tiles;
  return {packet.getTileFromAddr()};
}

std::optional<Arbiter> AIEX::arbiterTaken(NpuControlPacketOp packet,
                                          const OverlayMulticasts::Info *info,
                                          ConfiguredDevices &devices,
                                          const std::set<Arbiter> &held) {
  const OverlayMulticasts::Multicast *m =
      info ? info->multicastOf(packet) : nullptr;
  if (!m)
    return devices.controlArbiterIn(
        cast<ControlPacketGroupOp>(packet->getParentOp()).getDeviceRefAttr(),
        packet.getTileFromAddr(), held);
  auto tree = info->trees.find(m->column);
  if (tree != info->trees.end())
    for (const Arbiter &a : tree->second)
      if (held.count(a))
        return a;
  return std::nullopt;
}

// Reads `overlay`'s aie.ctrl_pkt_multicast and the routes of its control
// channels; null when it records no multicast.
static std::unique_ptr<OverlayMulticasts::Info>
readMulticasts(AIE::DeviceOp overlay) {
  MLIRContext *ctx = overlay.getContext();
  ArrayAttr listed =
      AIE::AIEDialect::CtrlPktMulticastAttrHelper(ctx).getAttr(overlay);
  if (!listed)
    return nullptr;
  auto info = std::make_unique<OverlayMulticasts::Info>();
  for (Attribute entry : listed) {
    auto dict = cast<DictionaryAttr>(entry);
    OverlayMulticasts::Multicast m;
    m.id = dict.getAs<IntegerAttr>("id").getInt();
    m.column = dict.getAs<IntegerAttr>("column").getInt();
    for (auto tile :
         dict.getAs<ArrayAttr>("tiles").getAsRange<AIE::TileAttr>()) {
      m.tiles.push_back(tile.getTileID());
      info->byTile[tile.getTileID()] = m.column;
    }
    info->multicasts[m.column] = std::move(m);
  }
  for (auto [tile, id] : controllerIdsOf(overlay))
    info->controllerIds[tile] = id.assignedId();
  std::map<int, int> channels;
  AIE::AIEDialect::IsCtrlPktOverlayAttrHelper marked(ctx);
  for (auto alloc : overlay.getOps<AIE::ShimDMAAllocationOp>())
    if (marked.isAttrPresent(alloc) &&
        alloc.getChannelDir() == AIE::DMAChannelDir::MM2S)
      channels[alloc.getTileOp().colIndex()] =
          static_cast<int>(alloc.getChannelIndex());
  for (const AIE::RoutedStream &s : AIE::traceRoutedStreams(overlay)) {
    auto channel = channels.find(s.src.tile.col);
    if (s.src.tile.row != 0 || channel == channels.end() ||
        s.src.port.bundle != AIE::WireBundle::DMA ||
        s.src.port.channel != channel->second ||
        s.dst.port.bundle != AIE::WireBundle::TileControl)
      continue;
    SmallVector<std::pair<AIE::TileID, AIE::Port>> trail;
    for (const AIE::StreamHop &hop : s.hops)
      trail.push_back({hop.tile, hop.input});
    auto addTrail = [&](int32_t id) {
      info->trails.try_emplace({s.dst.tile, id}, trail);
    };
    if (auto m = info->multicasts.find(s.src.tile.col);
        m != info->multicasts.end() &&
        s.carries(AIE::PacketInfoAttr::get(ctx, 0, m->second.id))) {
      addTrail(m->second.id);
      for (const AIE::StreamHop &hop : s.hops)
        if (hop.arbiter)
          info->trees[m->first].insert({hop.tile, *hop.arbiter});
    }
    auto own = info->controllerIds.find(s.dst.tile);
    if (own != info->controllerIds.end() &&
        s.carries(AIE::PacketInfoAttr::get(ctx, 0, own->second)))
      addTrail(own->second);
  }
  return info;
}

const OverlayMulticasts::Info *
OverlayMulticasts::of(FlatSymbolRefAttr configured) {
  auto device = configured ? module.lookupSymbol<AIE::DeviceOp>(configured)
                           : AIE::DeviceOp();
  if (!device || !device.getOverlayAttr())
    return nullptr;
  auto [it, fresh] = infos.try_emplace(device.getOverlayAttr().getAttr());
  if (fresh)
    if (auto overlay =
            module.lookupSymbol<AIE::DeviceOp>(device.getOverlayAttr()))
      it->second = readMulticasts(overlay);
  return it->second.get();
}

std::set<Arbiter> ConfiguredDevices::startedArbiters(FlatSymbolRefAttr device,
                                                     AIE::TileID tile,
                                                     const Classified &write) {
  std::set<Arbiter> held;
  int channel = write.kind == WriteKind::ChannelStart ? write.channel : -1;
  if (write.kind == WriteKind::Other || startGated(device, {tile, channel}))
    return held;
  Info &info = infoOf(device);
  SmallVector<StreamSource> started{{tile, channel}};
  for (size_t i = 0; i < started.size(); ++i) {
    auto [t, c] = started[i];
    auto it = c == -1 ? info.startedBy.find(t) : info.startedBy.end();
    if (it != info.startedBy.end())
      for (const StreamSource &s : it->second)
        if (!llvm::is_contained(started, s))
          started.push_back(s);
  }
  for (const StreamSource &s : started) {
    const std::set<Arbiter> &sent = info.sent[s];
    held.insert(sent.begin(), sent.end());
  }
  return held;
}

std::string AIEX::describeStart(AIE::TileID tile, const Classified &write) {
  if (write.kind == WriteKind::ChannelStart)
    return "the push to MM2S channel " + std::to_string(write.channel) +
           " of tile " + tileName(tile);
  return "the enable of core " + tileName(tile);
}
