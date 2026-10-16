//===- AIEReorderControlPacketsByTile.cpp -----------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEREORDERCONTROLPACKETSBYTILE
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

enum class Part { Other, ChannelStart, CoreEnable };

// Classifies control packets by the register they write, from the target's
// register database.
class PacketClassifier {
public:
  explicit PacketClassifier(const AIE::AIETargetModel &tm) : tm(tm) {}

  FailureOr<Part> classify(NpuControlPacketOp op) {
    auto data = op.getData();
    if (op.getOpcode() != 0 || !data)
      return Part::Other;
    AIE::TileID tile{static_cast<int>(op.getColumnFromAddr()),
                     static_cast<int>(op.getRowFromAddr())};
    const TileRegisters *regs = getRegisters(op, tile);
    if (!regs)
      return failure();
    uint32_t base = op.getAddress() & ((1u << tm.getRowShift()) - 1);
    Part part = Part::Other;
    for (auto [i, word] : llvm::enumerate(*data)) {
      uint32_t offset = base + 4 * i;
      if (regs->mm2sQueues.contains(offset))
        return Part::ChannelStart;
      if (regs->coreControl && offset == *regs->coreControl &&
          (word & regs->coreEnableMask))
        part = Part::CoreEnable;
    }
    return part;
  }

private:
  struct TileRegisters {
    llvm::DenseSet<uint32_t> mm2sQueues;
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
      // A shim tile's MM2S channels are started through a task queue, the
      // others' through a start queue.
      const char *queue =
          tm.isShimNOCTile(tile.col, tile.row) ? "_Task_Queue" : "_Start_Queue";
      for (int channel = 0;; ++channel) {
        std::string name = "DMA_MM2S_" + std::to_string(channel) + queue;
        const AIE::RegisterInfo *reg =
            tm.lookupRegister(name, tile, /*isMem=*/true);
        if (!reg) {
          if (channel > 0)
            break;
          op.emitOpError("cannot reorder control packets: the register "
                         "database has no ")
              << name << " register";
          return nullptr;
        }
        regs.mm2sQueues.insert(reg->offset);
      }
    }
    if (tm.isCoreTile(tile.col, tile.row)) {
      const AIE::RegisterInfo *reg = tm.lookupRegister("Core_Control", tile);
      const AIE::BitFieldInfo *enable = reg ? reg->getField("Enable") : nullptr;
      std::optional<uint32_t> mask =
          enable ? tm.getFieldMask(*enable) : std::nullopt;
      if (!mask) {
        op.emitOpError("cannot reorder control packets: the register "
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

struct AIEReorderControlPacketsByTilePass
    : xilinx::AIEX::impl::AIEReorderControlPacketsByTileBase<
          AIEReorderControlPacketsByTilePass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    PacketClassifier classifier(device.getTargetModel());

    SmallVector<SmallVector<NpuControlPacketOp>> runs;
    for (auto seq : device.getOps<AIE::RuntimeSequenceOp>()) {
      seq.walk([&](Block *block) {
        SmallVector<NpuControlPacketOp> run;
        for (Operation &op : *block) {
          if (auto packet = dyn_cast<NpuControlPacketOp>(op)) {
            run.push_back(packet);
            continue;
          }
          if (run.size() > 1)
            runs.push_back(std::move(run));
          run.clear();
        }
        if (run.size() > 1)
          runs.push_back(std::move(run));
      });
    }

    for (SmallVector<NpuControlPacketOp> &run : runs)
      if (failed(reorder(run, classifier)))
        return signalPassFailure();
  }

  static LogicalResult reorder(SmallVector<NpuControlPacketOp> &run,
                               PacketClassifier &classifier) {
    // Part, then column, then row; each tile's packets of one part keep their
    // order.
    using Key = std::tuple<Part, uint32_t, uint32_t>;
    SmallVector<std::pair<Key, NpuControlPacketOp>> keyed;
    for (NpuControlPacketOp packet : run) {
      FailureOr<Part> part = classifier.classify(packet);
      if (failed(part))
        return failure();
      keyed.push_back(
          {{*part, packet.getColumnFromAddr(), packet.getRowFromAddr()},
           packet});
    }
    std::stable_sort(
        keyed.begin(), keyed.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });
    Operation *next = run.back()->getNextNode();
    Block *block = run.back()->getBlock();
    for (auto &[key, packet] : keyed) {
      if (next)
        packet->moveBefore(next);
      else
        packet->moveBefore(block, block->end());
    }
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEReorderControlPacketsByTilePass() {
  return std::make_unique<AIEReorderControlPacketsByTilePass>();
}
