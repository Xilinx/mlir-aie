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
#include "CtrlPacketRuns.h"

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
    if (!group ||
        (last && split.byKindAndDevice && !sameKindAndDevice(last, group)))
      end();
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
    if (!startedBy.isAttrPresent(op))
      return;
    if (auto core = dyn_cast<AIE::CoreOp>(op))
      info.gated.insert({core.getTileOp().getTileID(), -1});
    else if (auto start = dyn_cast<AIE::DMAStartOp>(op))
      info.gated.insert({cast<AIE::TileElement>(start->getParentOp())
                             .getTile()
                             .getDefiningOp<AIE::TileOp>()
                             .getTileID(),
                         start.getChannelIndex()});
  });
  return info;
}
