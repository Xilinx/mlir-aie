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

SmallVector<GroupRun> AIEX::groupRuns(Block &block) {
  SmallVector<GroupRun> runs;
  GroupRun run;
  auto end = [&]() {
    if (!run.empty())
      runs.push_back(std::move(run));
    run.clear();
  };
  for (Operation &op : block) {
    auto group = dyn_cast<ControlPacketGroupOp>(op);
    if (!group || (!run.empty() && !sameKindAndDevice(run.back().group, group)))
      end();
    if (group)
      for (auto packet : group.getBody().front().getOps<NpuControlPacketOp>())
        run.push_back({packet, group});
  }
  end();
  return runs;
}

SmallVector<GroupRun> AIEX::groupRuns(AIE::RuntimeSequenceOp seq) {
  SmallVector<GroupRun> runs;
  seq.walk([&](Block *block) {
    if (!isa<ControlPacketGroupOp>(block->getParentOp()))
      llvm::append_range(runs, groupRuns(*block));
  });
  return runs;
}
