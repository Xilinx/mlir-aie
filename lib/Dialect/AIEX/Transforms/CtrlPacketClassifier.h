//===- CtrlPacketClassifier.h -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Classifies control packets by the registers they write.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TRANSFORMS_CTRLPACKETCLASSIFIER_H
#define AIEX_TRANSFORMS_CTRLPACKETCLASSIFIER_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace xilinx::AIEX {

// "(col, row)", for a diagnostic.
inline std::string tileName(AIE::TileID tile) {
  return "(" + std::to_string(tile.col) + ", " + std::to_string(tile.row) + ")";
}

enum class WriteKind { Other, ChannelStart, CoreEnable };

struct Classified {
  WriteKind kind;
  // The MM2S channel a WriteKind::ChannelStart packet starts.
  int channel = -1;
  // Whether the packet writes a DMA channel's start queue (MM2S or S2MM) or
  // sets the core's enable bit. Such a write has to keep its place among the
  // tile's other writes, whatever their address.
  bool startsSomething = false;
  // The tile-local word addresses the packet writes.
  llvm::SmallVector<uint32_t, 4> words;

  // Whether either order of two writes to one tile gives the same result.
  bool commutesWith(const Classified &other) const {
    return !startsSomething && !other.startsSomething &&
           llvm::none_of(words, [&](uint32_t w) {
             return llvm::is_contained(other.words, w);
           });
  }
};

class PacketClassifier {
public:
  explicit PacketClassifier(const AIE::AIETargetModel &tm) : tm(tm) {}

  // Fails when the register database lacks a register it needs.
  mlir::FailureOr<Classified> classify(NpuControlPacketOp op);

private:
  struct TileRegisters {
    // Queue register offset -> MM2S channel.
    llvm::DenseMap<uint32_t, int> mm2sQueues;
    // Every DMA channel's queue register.
    llvm::DenseSet<uint32_t> queues;
    std::optional<uint32_t> coreControl;
    uint32_t coreEnableMask = 0;
  };

  const TileRegisters *getRegisters(NpuControlPacketOp op);

  const AIE::AIETargetModel &tm;
  llvm::DenseMap<AIE::TileID, TileRegisters> registers;
};

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CTRLPACKETCLASSIFIER_H
