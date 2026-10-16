//===- ControlPacketClassifier.h --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Classifies control packets by the registers they write, for the passes that
// reorder and merge control packets.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TRANSFORMS_CONTROLPACKETCLASSIFIER_H
#define AIEX_TRANSFORMS_CONTROLPACKETCLASSIFIER_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>
#include <string>

namespace xilinx::AIEX {

enum class Part { Other, ChannelStart, CoreEnable };

struct Classified {
  Part part;
  // The MM2S channel a Part::ChannelStart packet starts.
  int channel = -1;
  // Whether the packet writes a DMA channel's start queue (MM2S or S2MM) or the
  // core's control register. The tile's other writes have to land before such a
  // write, whatever their address.
  bool startsSomething = false;
};

// Classifies control packets by the register they write, from the target's
// register database.
class PacketClassifier {
public:
  explicit PacketClassifier(const AIE::AIETargetModel &tm) : tm(tm) {}

  mlir::FailureOr<Classified> classify(NpuControlPacketOp op) {
    auto data = op.getData();
    if (op.getOpcode() != 0 || !data)
      return Classified{Part::Other};
    AIE::TileID tile{static_cast<int>(op.getColumnFromAddr()),
                     static_cast<int>(op.getRowFromAddr())};
    const TileRegisters *regs = getRegisters(op, tile);
    if (!regs)
      return mlir::failure();
    uint32_t base = op.getAddress() & ((1u << tm.getRowShift()) - 1);
    Classified c{Part::Other};
    for (auto [i, word] : llvm::enumerate(*data)) {
      uint32_t offset = base + 4 * i;
      if (regs->queues.contains(offset) ||
          (regs->coreControl && offset == *regs->coreControl))
        c.startsSomething = true;
      if (auto it = regs->mm2sQueues.find(offset);
          it != regs->mm2sQueues.end() && c.part != Part::ChannelStart) {
        c.part = Part::ChannelStart;
        c.channel = it->second;
      }
      if (regs->coreControl && offset == *regs->coreControl &&
          (word & regs->coreEnableMask) && c.part == Part::Other)
        c.part = Part::CoreEnable;
    }
    return c;
  }

private:
  struct TileRegisters {
    // Queue register offset -> MM2S channel.
    llvm::DenseMap<uint32_t, int> mm2sQueues;
    // Every DMA channel's queue register.
    llvm::DenseSet<uint32_t> queues;
    std::optional<uint32_t> coreControl;
    uint32_t coreEnableMask = 0;
  };

  const TileRegisters *getRegisters(NpuControlPacketOp op, AIE::TileID tile) {
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

  const AIE::AIETargetModel &tm;
  llvm::DenseMap<AIE::TileID, TileRegisters> registers;
};

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CONTROLPACKETCLASSIFIER_H
