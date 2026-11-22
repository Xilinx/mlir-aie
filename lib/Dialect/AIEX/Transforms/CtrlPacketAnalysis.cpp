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

#include "mlir/Interfaces/SideEffectInterfaces.h"

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

PacketEffects
AIEX::packetEffects(NpuControlPacketOp packet,
                    const llvm::DenseSet<AIE::TileID> *activeCores,
                    std::optional<AIE::TileID> destination) {
  auto data = packet.getData();
  if (packet.getOpcode() != 0 || !data)
    return {};
  const auto &target =
      packet->getParentOfType<AIE::DeviceOp>().getTargetModel();
  AIE::TileID tile = destination.value_or(packet.getTileFromAddr());
  auto active = [&](AIE::TileID core) {
    return !activeCores || activeCores->contains(core);
  };
  std::map<StringRef, SmallVector<Attribute>> modules;
  NamedAttrList direct;
  auto add = [&](StringRef module, AIE::TileID owner) {
    Attribute attribute =
        AIE::TileAttr::get(packet.getContext(), owner.col, owner.row);
    auto &tiles = modules[module];
    if (!llvm::is_contained(tiles, attribute))
      tiles.push_back(attribute);
  };
  bool reorderable = true;
  for (size_t index = 0; index < data->size(); ++index) {
    auto info =
        target.getRegisterWriteInfo(tile, packet.getLocalAddress() + 4 * index);
    if (!info)
      return {};
    reorderable &= info->reorderable;
    add(info->module, tile);
    direct.set(info->module,
               ArrayAttr::get(packet.getContext(),
                              {AIE::TileAttr::get(packet.getContext(), tile.col,
                                                  tile.row)}));
    if (info->reorderable)
      continue;
    for (int column = 0; column < target.columns(); ++column)
      for (int row = 0; row < target.rows(); ++row) {
        AIE::TileID other{column, row};
        if (info->module == "core" && active(tile) &&
            target.isCoreTile(column, row) &&
            target.isLegalMemAffinity(tile.col, tile.row, column, row))
          add("memory", other);
        if (info->module == "memory" && active(other) &&
            target.isCoreTile(column, row) &&
            target.isLegalMemAffinity(column, row, tile.col, tile.row)) {
          add("core", other);
          for (int ownerColumn = 0; ownerColumn < target.columns();
               ++ownerColumn)
            for (int ownerRow = 0; ownerRow < target.rows(); ++ownerRow)
              if (target.isCoreTile(ownerColumn, ownerRow) &&
                  target.isLegalMemAffinity(column, row, ownerColumn, ownerRow))
                add("memory", {ownerColumn, ownerRow});
        }
        if (info->module == "memory" && target.isMemTile(tile.col, tile.row) &&
            target.isMemTile(column, row) &&
            target.isLegalMemAffinity(tile.col, tile.row, column, row))
          add("memory", other);
      }
  }
  NamedAttrList scope;
  for (auto &[module, tiles] : modules)
    scope.set(module, ArrayAttr::get(packet.getContext(), tiles));
  if (packet.getMulticastId() && !destination)
    return {reorderable, {}, {}};
  return {reorderable, scope.getDictionary(packet.getContext()),
          direct.getDictionary(packet.getContext())};
}

FailureOr<llvm::DenseMap<Operation *, PacketEffects>>
AIEX::packetEffects(Block &block) {
  llvm::DenseMap<Operation *, PacketEffects> result;
  const auto &target = AIE::getTargetModel(block.getParentOp());
  PacketClassifier classifier(target);
  SmallVector<Attribute> switches;
  for (int column = 0; column < target.columns(); ++column)
    for (int row = 0; row < target.rows(); ++row)
      switches.push_back(
          AIE::TileAttr::get(block.getParentOp()->getContext(), column, row));
  NamedAttrList switchScope;
  switchScope.set("switch",
                  ArrayAttr::get(block.getParentOp()->getContext(), switches));
  DictionaryAttr switchEffects =
      switchScope.getDictionary(block.getParentOp()->getContext());
  OverlayMulticasts multicasts(block.getParentOp()->getParentOfType<ModuleOp>(),
                               false);
  FlatSymbolRefAttr device;
  bool quiescent = false;
  llvm::DenseSet<AIE::TileID> activeCores;
  DictionaryAttr activeModules =
      DictionaryAttr::get(block.getParentOp()->getContext());
  for (Operation &operation : block) {
    if (auto context = dyn_cast<SetDeviceOp>(operation)) {
      device = context.getDeviceRefAttr();
      quiescent = context.getQuiescent();
      activeCores.clear();
      activeModules = DictionaryAttr::get(operation.getContext());
      continue;
    }
    if (isa<ControlPacketBarrierOp>(operation))
      continue;
    auto packet = dyn_cast<NpuControlPacketOp>(operation);
    if (!packet) {
      quiescent = false;
      continue;
    }
    auto write = classifier.classify(packet);
    if (failed(write))
      return failure();
    const auto *info = multicasts.of(device);
    auto destinations = tilesReached(packet, info);
    if (write->kind == WriteKind::CoreEnable)
      activeCores.insert(destinations.begin(), destinations.end());
    SmallVector<DictionaryAttr> scopes, targets;
    bool storage = true;
    for (AIE::TileID tile : destinations) {
      PacketEffects effects =
          packetEffects(packet, quiescent ? &activeCores : nullptr, tile);
      storage &= effects.reorderable;
      scopes.push_back(effects.scope);
      targets.push_back(effects.target);
    }
    PacketEffects effects{storage, unionScopes(packet.getContext(), scopes),
                          unionScopes(packet.getContext(), targets)};
    if (packet.getMulticastId() && (!info || !info->multicastOf(packet)))
      effects = {};
    if (!effects.scope)
      quiescent = false;
    if (write->startsSomething) {
      effects.scope =
          unionScopes(packet.getContext(), {effects.scope, switchEffects});
      activeModules =
          unionScopes(packet.getContext(), {activeModules, effects.scope});
    }
    if (storage && !quiescent)
      effects = {};
    else if (storage && scopesOverlap(activeModules, effects.scope)) {
      effects.reorderable = false;
      effects.scope =
          unionScopes(packet.getContext(), {effects.scope, activeModules});
    }
    result[packet] = effects;
  }
  return result;
}

bool AIEX::scopesOverlap(DictionaryAttr left, DictionaryAttr right) {
  if (!left || !right)
    return true;
  for (NamedAttribute entry : left)
    if (auto others = right.getAs<ArrayAttr>(entry.getName()))
      for (Attribute tile : cast<ArrayAttr>(entry.getValue()))
        if (llvm::is_contained(others, tile))
          return true;
  return false;
}

DictionaryAttr AIEX::unionScopes(MLIRContext *context,
                                 ArrayRef<DictionaryAttr> scopes) {
  std::map<StringRef, SmallVector<Attribute>> modules;
  for (DictionaryAttr scope : scopes) {
    if (!scope)
      return {};
    for (NamedAttribute entry : scope) {
      auto &tiles = modules[entry.getName().getValue()];
      for (Attribute tile : cast<ArrayAttr>(entry.getValue()))
        if (!llvm::is_contained(tiles, tile))
          tiles.push_back(tile);
    }
  }
  NamedAttrList result;
  for (auto &[name, tiles] : modules)
    result.set(name, ArrayAttr::get(context, tiles));
  return result.getDictionary(context);
}

SmallVector<unsigned> AIEX::awaitSteps(const PacketRun &run,
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

static FlatSymbolRefAttr configuredDeviceAt(Operation *op) {
  while (op && !isa<AIE::RuntimeSequenceOp>(op)) {
    for (Operation *previous = op->getPrevNode(); previous;
         previous = previous->getPrevNode()) {
      if (auto context = dyn_cast<SetDeviceOp>(previous))
        return context.getDeviceRefAttr();
      if (isa<NpuLoadPdiOp>(previous))
        return {};
    }
    op = op->getParentOp();
  }
  return {};
}

SmallVector<PacketRun> AIEX::packetRuns(Block &block, bool includePhases) {
  SmallVector<PacketRun> runs;
  PacketRun run;
  FlatSymbolRefAttr device = configuredDeviceAt(block.getParentOp());
  unsigned phase = 0;
  auto end = [&]() {
    if (!run.empty())
      runs.push_back(std::move(run));
    run.clear();
    phase = 0;
  };
  for (Operation &op : block) {
    if (auto barrier = dyn_cast<ControlPacketBarrierOp>(op);
        barrier && barrier.getCompilerOnly() && includePhases)
      continue;
    if (isa<SetDeviceOp, ControlPacketBarrierOp>(op)) {
      if (includePhases)
        ++phase;
      else
        end();
      if (auto context = dyn_cast<SetDeviceOp>(op))
        device = context.getDeviceRefAttr();
      continue;
    }
    if (auto packet = dyn_cast<NpuControlPacketOp>(op)) {
      run.push_back({packet, device, phase});
    } else {
      end();
      if (isa<NpuLoadPdiOp>(op))
        device = {};
    }
  }
  end();
  return runs;
}

SmallVector<PacketRun> AIEX::packetRuns(AIE::RuntimeSequenceOp seq,
                                        bool includePhases) {
  SmallVector<PacketRun> runs;
  seq.walk([&](Block *block) {
    llvm::append_range(runs, packetRuns(*block, includePhases));
  });
  return runs;
}

FailureOr<SmallVector<PacketRun>>
AIEX::deliveryAnalysisRuns(AIE::RuntimeSequenceOp sequence) {
  SmallVector<PacketRun> runs;
  if (sequence->getRegion(0).empty())
    return runs;
  WalkResult nested = sequence.walk([&](Operation *operation) {
    if (!isa<NpuControlPacketOp, NpuLoadPdiOp, NpuWriteConfigOp,
             NpuWriteTeardownOp>(operation) ||
        operation->getParentOp() == sequence.getOperation())
      return WalkResult::advance();
    operation->emitOpError(
        "requires flat control-packet execution for delivery analysis");
    return WalkResult::interrupt();
  });
  if (nested.wasInterrupted())
    return failure();
  if (!llvm::hasSingleElement(sequence->getRegion(0))) {
    sequence.emitOpError("requires a single block for delivery analysis");
    return failure();
  }
  PacketRun run;
  PacketClassifier classifier(AIE::getTargetModel(sequence));
  ModuleOp module = sequence->getParentOfType<ModuleOp>();
  ConfiguredDevices devices(module);
  OverlayMulticasts multicasts(module, false);
  DenseMap<Attribute, SmallVector<size_t>> live;
  FlatSymbolRefAttr device;
  NpuLoadPdiOp lastLoad;
  unsigned phase = 0;
  bool hostEffects = false;
  bool gatesInitialized = false;
  DenseMap<uint32_t, int32_t> initializers;
  DenseSet<uint32_t> initialized;
  auto resetGateProof = [&] {
    gatesInitialized = false;
    initializers.clear();
    initialized.clear();
    auto configured =
        device ? module.lookupSymbol<AIE::DeviceOp>(device) : AIE::DeviceOp{};
    if (!configured)
      return;
    const auto &target = configured.getTargetModel();
    for (AIE::LockOp lock : configured.getOps<AIE::LockOp>()) {
      AIE::TileID tile = lock.getTileOp().getTileID();
      if (!lock.getLockID()) {
        initializers.clear();
        return;
      }
      auto address = target.getLocalLockAddress(*lock.getLockID(), tile);
      if (!address || (!lock.getInit() && !lock.getTileOp().getCoreOp())) {
        initializers.clear();
        return;
      }
      initializers[*address |
                   (static_cast<uint32_t>(tile.col)
                    << target.getColumnShift()) |
                   (static_cast<uint32_t>(tile.row) << target.getRowShift())] =
          lock.getInit().value_or(0);
    }
  };
  auto finish = [&] {
    if (!run.empty())
      runs.push_back(std::move(run));
    run.clear();
    live.clear();
    phase = 0;
  };
  for (Operation &operation : sequence->getRegion(0).front()) {
    if (auto packet = dyn_cast<NpuControlPacketOp>(operation)) {
      auto write = classifier.classify(packet);
      if (failed(write))
        return failure();
      if (hostEffects && !initializers.empty() && packet.getOpcode() == 0 &&
          packet.getData()) {
        for (auto [offset, value] : llvm::enumerate(*packet.getData())) {
          auto found = initializers.find(packet.getAddress() + 4 * offset);
          if (found == initializers.end())
            continue;
          if (found->second == value)
            initialized.insert(found->first);
          else
            initialized.erase(found->first);
        }
        gatesInitialized = initialized.size() == initializers.size();
      }
      if (write->kind != WriteKind::Other)
        live[device].push_back(run.size());
      run.push_back({packet, device, phase});
      if (hostEffects && !gatesInitialized)
        run.back().hostPhase = phase;
      continue;
    }
    if (auto barrier = dyn_cast<ControlPacketBarrierOp>(operation);
        barrier && barrier.getCompilerOnly())
      continue;
    ++phase;
    if (auto context = dyn_cast<SetDeviceOp>(operation)) {
      device = context.getDeviceRefAttr();
      resetGateProof();
    } else if (auto load = dyn_cast<NpuLoadPdiOp>(operation)) {
      if (lastLoad) {
        bool distinctSymbols =
            lastLoad.getDeviceRefAttr() && load.getDeviceRefAttr() &&
            lastLoad.getDeviceRefAttr() != load.getDeviceRefAttr();
        bool distinctAddresses = !lastLoad.getDeviceRefAttr() &&
                                 !load.getDeviceRefAttr() &&
                                 lastLoad.getAddress() && load.getAddress() &&
                                 lastLoad.getAddress() != load.getAddress();
        if (distinctSymbols || distinctAddresses) {
          finish();
          hostEffects = false;
        }
      }
      lastLoad = load;
      device = {};
      resetGateProof();
    } else if (auto barrier = dyn_cast<ControlPacketBarrierOp>(operation)) {
      if (auto drained = barrier.getDrainsAttr()) {
        if (auto found = live.find(drained); found != live.end()) {
          llvm::erase_if(found->second, [&](size_t index) {
            ContextualPacket &entry = run[index];
            auto write = classifier.classify(entry.packet);
            assert(succeeded(write));
            int channel =
                write->kind == WriteKind::ChannelStart ? write->channel : -1;
            for (AIE::TileID tile :
                 tilesReached(entry.packet, multicasts.of(entry.device)))
              if (!devices.teardownStops(drained, {tile, channel}))
                return false;
            entry.streamEndPhase = phase;
            return true;
          });
          if (found->second.empty())
            live.erase(found);
        }
        if (live.empty())
          finish();
      }
    } else {
      if (operation.getNumRegions())
        lastLoad = {};
      if (!isMemoryEffectFree(&operation) && !isa<NpuMaskPollOp>(operation)) {
        hostEffects = true;
        resetGateProof();
        for (const auto &entry : live)
          for (size_t index : entry.second)
            run[index].hostPhase = std::min(run[index].hostPhase, phase);
      }
    }
  }
  finish();
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
  auto [it, fresh] =
      infos.try_emplace(device ? device.getAttr() : StringAttr{});
  if (fresh)
    it->second = std::make_unique<Info>();
  Info &info = *it->second;
  if (!fresh || !device)
    return info;
  auto d = module.lookupSymbol<AIE::DeviceOp>(device);
  if (!d)
    return info;
  for (auto core : d.getOps<AIE::CoreOp>())
    info.teardownSources.insert({core.getTileOp().getTileID(), -1});
  for (const AIE::DmaChannelProgram &program : AIE::collectDmaPrograms(d))
    if (program.dma.dir == AIE::DMAChannelDir::MM2S &&
        !program.op->getParentOfType<AIE::RuntimeSequenceOp>() &&
        !d.getTargetModel().isShimNOCorPLTile(program.dma.tile.col,
                                              program.dma.tile.row))
      info.teardownSources.insert({program.dma.tile, program.dma.channel});
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
                                          FlatSymbolRefAttr device,
                                          const OverlayMulticasts::Info *info,
                                          ConfiguredDevices &devices,
                                          const std::set<Arbiter> &held) {
  const OverlayMulticasts::Multicast *m =
      info ? info->multicastOf(packet) : nullptr;
  if (!m)
    return devices.controlArbiterIn(device, packet.getTileFromAddr(), held);
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
readMulticasts(AIE::DeviceOp overlay, bool traceRoutes) {
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
  if (!traceRoutes)
    return info;
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
      it->second = readMulticasts(overlay, traceRoutes);
  return it->second.get();
}

std::set<Arbiter> ConfiguredDevices::startedArbiters(FlatSymbolRefAttr device,
                                                     AIE::TileID tile,
                                                     const Classified &write,
                                                     bool useGating) {
  std::set<Arbiter> held;
  int channel = write.kind == WriteKind::ChannelStart ? write.channel : -1;
  if (write.kind == WriteKind::Other ||
      (useGating && startGated(device, {tile, channel})))
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
