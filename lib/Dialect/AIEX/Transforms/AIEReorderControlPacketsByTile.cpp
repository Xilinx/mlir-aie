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
#include <limits>
#include <set>
#include <tuple>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEREORDERCONTROLPACKETSBYTILE
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

enum class Part { Other, ChannelStart, CoreEnable };

struct Classified {
  Part part;
  // The MM2S channel a Part::ChannelStart packet starts.
  int channel = -1;
};

// Classifies control packets by the register they write, from the target's
// register database.
class PacketClassifier {
public:
  explicit PacketClassifier(const AIE::AIETargetModel &tm) : tm(tm) {}

  FailureOr<Classified> classify(NpuControlPacketOp op) {
    auto data = op.getData();
    if (op.getOpcode() != 0 || !data)
      return Classified{Part::Other};
    AIE::TileID tile{static_cast<int>(op.getColumnFromAddr()),
                     static_cast<int>(op.getRowFromAddr())};
    const TileRegisters *regs = getRegisters(op, tile);
    if (!regs)
      return failure();
    uint32_t base = op.getAddress() & ((1u << tm.getRowShift()) - 1);
    Part part = Part::Other;
    for (auto [i, word] : llvm::enumerate(*data)) {
      uint32_t offset = base + 4 * i;
      if (auto it = regs->mm2sQueues.find(offset); it != regs->mm2sQueues.end())
        return Classified{Part::ChannelStart, it->second};
      if (regs->coreControl && offset == *regs->coreControl &&
          (word & regs->coreEnableMask))
        part = Part::CoreEnable;
    }
    return Classified{part};
  }

private:
  struct TileRegisters {
    // Queue register offset -> MM2S channel.
    llvm::DenseMap<uint32_t, int> mm2sQueues;
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
        regs.mm2sQueues[reg->offset] = channel;
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

// The MM2S channels (channel >= 0) and cores (channel -1)
// aie-check-core-start-gating proved gated, per device.
class GatedChannels {
public:
  explicit GatedChannels(ModuleOp module) : module(module) {}

  bool contains(FlatSymbolRefAttr device, AIE::TileID tile, int channel) {
    if (!device)
      return false;
    auto it = gated.find(device.getAttr());
    if (it == gated.end()) {
      std::set<std::tuple<int, int, int>> channels;
      AIEXDialect::CtrlPktGatedAttrHelper stamp(module.getContext());
      if (auto d = module.lookupSymbol<AIE::DeviceOp>(device))
        d.walk([&](AIE::DMAStartOp start) {
          auto index = dyn_cast<IntegerAttr>(start.getChannel());
          if (!start.isSend() || !index || !stamp.isAttrPresent(start))
            return;
          auto t = cast<AIE::TileOp>(
              start->getParentOp()->getOperand(0).getDefiningOp());
          channels.insert(
              {t.getCol(), t.getRow(), static_cast<int>(index.getInt())});
        });
      if (auto d = module.lookupSymbol<AIE::DeviceOp>(device))
        d.walk([&](AIE::CoreOp core) {
          if (stamp.isAttrPresent(core))
            channels.insert(
                {core.getTileOp().getCol(), core.getTileOp().getRow(), -1});
        });
      it = gated.try_emplace(device.getAttr(), std::move(channels)).first;
    }
    return it->second.count({tile.col, tile.row, channel});
  }

private:
  ModuleOp module;
  llvm::DenseMap<StringAttr, std::set<std::tuple<int, int, int>>> gated;
};

struct AIEReorderControlPacketsByTilePass
    : xilinx::AIEX::impl::AIEReorderControlPacketsByTileBase<
          AIEReorderControlPacketsByTilePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    GatedChannels gated(module);
    for (auto device : module.getOps<AIE::DeviceOp>()) {
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
            if (!run.empty())
              runs.push_back(std::move(run));
            run.clear();
          }
          if (!run.empty())
            runs.push_back(std::move(run));
        });
      }

      for (SmallVector<NpuControlPacketOp> &run : runs)
        if (failed(reorder(run, classifier, gated)))
          return signalPassFailure();
    }
  }

  LogicalResult reorder(SmallVector<NpuControlPacketOp> &run,
                        PacketClassifier &classifier, GatedChannels &gated) {
    AIEXDialect::CtrlPktDeviceAttrHelper deviceStamp(&getContext());
    AIEXDialect::CtrlPktEnablePhaseAttrHelper enablePhase(&getContext());
    // Part, then column, then row (by-tile only); each tile's packets of one
    // part keep their order.
    using Key = std::tuple<Part, uint32_t, uint32_t>;
    SmallVector<std::pair<Key, NpuControlPacketOp>> keyed;
    for (NpuControlPacketOp packet : run) {
      FailureOr<Classified> c = classifier.classify(packet);
      if (failed(c))
        return failure();
      AIE::TileID tile{static_cast<int>(packet.getColumnFromAddr()),
                       static_cast<int>(packet.getRowFromAddr())};
      Part part = c->part;
      // With gated-enable, a proved core's enable goes last among its
      // column's other writes.
      uint32_t row = tile.row;
      if (part == Part::ChannelStart &&
          gated.contains(deviceStamp.getAttr(packet), tile, c->channel))
        part = Part::Other;
      else if (part == Part::CoreEnable && gatedEnable &&
               gated.contains(deviceStamp.getAttr(packet), tile, -1)) {
        part = Part::Other;
        row = std::numeric_limits<uint32_t>::max();
      }
      if (part != Part::Other)
        enablePhase.setAttr(packet, UnitAttr::get(&getContext()));
      else if (enablePhase.isAttrPresent(packet))
        enablePhase.removeAttr(packet);
      Key key{part, 0, 0};
      if (byTile)
        key = {part, tile.col, row};
      keyed.push_back({key, packet});
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

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEReorderControlPacketsByTilePass() {
  return std::make_unique<AIEReorderControlPacketsByTilePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEReorderControlPacketsByTilePass(
    const AIEReorderControlPacketsByTileOptions &options) {
  return std::make_unique<AIEReorderControlPacketsByTilePass>(options);
}
