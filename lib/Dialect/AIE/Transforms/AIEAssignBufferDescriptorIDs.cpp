//===- AIEAssignBufferDescriptorIDs.cpp -------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEASSIGNBUFFERDESCRIPTORIDS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-assign-bd-ids"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

BdIdGenerator::BdIdGenerator(int col, int row,
                             const AIETargetModel &targetModel)
    : col(col), row(row), targetModel(targetModel) {}

std::optional<uint32_t> BdIdGenerator::nextBdId(int channelIndex) {
  uint32_t bd_id = 0;
  uint32_t max_bd_id = targetModel.getNumBDs(col, row) - 1;
  // Find the next free BD ID. This is not an efficient algorithm, but doesn't
  for (; (bdIdAlreadyAssigned(bd_id) ||
          !targetModel.isBdChannelAccessible(col, row, bd_id, channelIndex)) &&
         bd_id <= max_bd_id;
       bd_id++)
    ;
  if (bd_id > max_bd_id) {
    return std::nullopt;
  }
  assignBdId(bd_id);
  return std::optional<uint32_t>(bd_id);
}

void BdIdGenerator::assignBdId(uint32_t bdId) {
  assert(!alreadyAssigned.count(bdId) && "bdId has already been assigned");
  alreadyAssigned.insert(bdId);
}

bool BdIdGenerator::bdIdAlreadyAssigned(uint32_t bdId) {
  return alreadyAssigned.count(bdId);
}

void BdIdGenerator::freeBdId(uint32_t bdId) { alreadyAssigned.erase(bdId); }

// Read the head buffer-descriptor id and (biased) START_QUEUE repeat count of
// the resident DMA channel (dir, channel) on `tile`, off the placed
// aie.mem / aie.memtile_dma chain, exactly as the load-time start-queue push
// does. Returns false if there is no such resident channel or its head BD id is
// not yet assigned. Only the block form (aie.dma_start) is scanned; the
// objectFIFO transform always emits that form.
static bool resolveResidentChannel(DeviceOp device, TileOp tile,
                                   DMAChannelDir dir, int channel,
                                   uint32_t &headBdId, int32_t &repeatCount) {
  auto scanRegion = [&](Region &region) -> bool {
    for (Block &block : region)
      for (DMAStartOp startOp : block.getOps<DMAStartOp>()) {
        if (startOp.getChannelDir() != dir ||
            startOp.getChannelIndex() != channel)
          continue;
        Block *dest = startOp.getDest();
        if (!dest)
          return false;
        auto bds = dest->getOps<DMABDOp>();
        if (bds.empty())
          return false;
        DMABDOp head = *bds.begin();
        if (!head.getBdId().has_value())
          return false;
        headBdId = head.getBdId().value();
        repeatCount = startOp.getRepeatCount();
        return true;
      }
    return false;
  };
  for (MemOp mem : device.getOps<MemOp>())
    if (mem.getTile() == tile.getResult())
      return scanRegion(mem->getRegion(0));
  for (MemTileDMAOp mem : device.getOps<MemTileDMAOp>())
    if (mem.getTile() == tile.getResult())
      return scanRegion(mem->getRegion(0));
  return false;
}

// Populate each objectFIFO re-arm binding's head_bd_ids / repeat_counts now
// that BD ids are assigned, so the later aiex.dma_channel_reset_for lowering
// reads them straight off the binding instead of re-scanning the emitted chain.
// A binding that already carries them (hand-authored) is left untouched; one
// whose endpoints cannot all be resolved is left unpopulated for the lowering
// to diagnose.
static void populateRearmBindings(DeviceOp device) {
  for (auto binding : device.getOps<ObjectFifoRearmBindingOp>()) {
    if (binding.getHeadBdIds().has_value() ||
        binding.getRepeatCounts().has_value())
      continue;
    ValueRange tiles = binding.getChannelTiles();
    ArrayRef<int32_t> dirs = binding.getChannelDirs();
    ArrayRef<int32_t> chans = binding.getChannelIndices();
    SmallVector<int32_t> headBdIds, repeatCounts;
    bool resolvedAll = true;
    for (unsigned i = 0; i < tiles.size(); ++i) {
      auto tileOp = tiles[i].getDefiningOp<TileOp>();
      DMAChannelDir dir =
          dirs[i] == 0 ? DMAChannelDir::S2MM : DMAChannelDir::MM2S;
      uint32_t headBdId = 0;
      int32_t repeatCount = 0;
      if (!tileOp || !resolveResidentChannel(device, tileOp, dir, chans[i],
                                             headBdId, repeatCount)) {
        resolvedAll = false;
        break;
      }
      headBdIds.push_back(static_cast<int32_t>(headBdId));
      repeatCounts.push_back(repeatCount);
    }
    if (!resolvedAll)
      continue;
    OpBuilder b(binding);
    binding.setHeadBdIdsAttr(b.getDenseI32ArrayAttr(headBdIds));
    binding.setRepeatCountsAttr(b.getDenseI32ArrayAttr(repeatCounts));
  }
}

struct AIEAssignBufferDescriptorIDsPass
    : xilinx::AIE::impl::AIEAssignBufferDescriptorIDsBase<
          AIEAssignBufferDescriptorIDsPass> {
  void runOnOperation() override {
    DeviceOp targetOp = getOperation();
    const AIETargetModel &targetModel = targetOp.getTargetModel();

    auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
    llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
    llvm::append_range(memOps, targetOp.getOps<ShimDMAOp>());
    for (TileElement memOp : memOps) {
      int col = memOp.getTileID().col;
      int row = memOp.getTileID().row;

      BdIdGenerator gen(col, row, targetModel);
      memOp->walk<WalkOrder::PreOrder>([&](DMABDOp bd) {
        if (bd.getBdId().has_value())
          gen.assignBdId(bd.getBdId().value());
      });

      auto dmaOps = memOp.getOperation()->getRegion(0).getOps<DMAOp>();
      if (!dmaOps.empty()) {
        for (auto dmaOp : dmaOps) {
          auto bdRegions = dmaOp.getBds();
          for (auto &bdRegion : bdRegions) {
            auto &block = bdRegion.getBlocks().front();
            DMABDOp bd = *block.getOps<DMABDOp>().begin();
            if (bd.getBdId().has_value()) {
              assert(
                  gen.bdIdAlreadyAssigned(bd.getBdId().value()) &&
                  "bdId assigned by user but not found during previous walk");
            } else {
              std::optional<int32_t> next_id =
                  gen.nextBdId(dmaOp.getChannelIndex());
              if (!next_id) {
                int channelIndex = dmaOp.getChannelIndex();
                bd.emitOpError()
                    << "Allocator exhausted available BD IDs (maximum "
                    << targetModel.getNumBDsForChannel(col, row, channelIndex)
                    << " available for channel " << channelIndex << ").";
                return signalPassFailure();
              }
              bd.setBdId(*next_id);
            }
          }
        }
      } else {
        DenseMap<Block *, int> blockChannelMap;
        // Associate with each block the channel index specified by the
        // dma_start
        for (Block &block : memOp.getOperation()->getRegion(0))
          for (auto op : block.getOps<DMAStartOp>()) {
            int chNum = op.getChannelIndex();
            blockChannelMap[&block] = chNum;
            Block *dest = op.getDest();
            while (dest) {
              blockChannelMap[dest] = chNum;
              if (dest->hasNoSuccessors())
                break;
              dest = dest->getSuccessors()[0];
              if (blockChannelMap.contains(dest))
                dest = nullptr;
            }
          }

        for (Block &block : memOp.getOperation()->getRegion(0)) {
          if (block.getOps<DMABDOp>().empty())
            continue;
          assert(blockChannelMap.count(&block));
          DMABDOp bd = (*block.getOps<DMABDOp>().begin());
          if (bd.getBdId().has_value()) {
            assert(gen.bdIdAlreadyAssigned(bd.getBdId().value()) &&
                   "bdId assigned by user but not found during previous walk");
          } else {
            std::optional<int32_t> next_id =
                gen.nextBdId(blockChannelMap[&block]);
            if (!next_id) {
              int channelIndex = blockChannelMap[&block];
              bd.emitOpError()
                  << "Allocator exhausted available BD IDs (maximum "
                  << targetModel.getNumBDsForChannel(col, row, channelIndex)
                  << " available for channel " << channelIndex << ").";
              return signalPassFailure();
            }
            bd.setBdId(*next_id);
          }
        }
      }
    }
    for (TileElement memOp : memOps) {
      auto dmaOps = memOp.getOperation()->getRegion(0).getOps<DMAOp>();
      if (!dmaOps.empty()) {
        for (auto dmaOp : dmaOps) {
          auto bdRegions = dmaOp.getBds();
          for (auto *bdRegionIt = bdRegions.begin();
               bdRegionIt != bdRegions.end();) {
            auto &block = bdRegionIt->getBlocks().front();
            DMABDOp bd = *block.getOps<DMABDOp>().begin();
            std::optional<int> nextBdId;
            if (++bdRegionIt != bdRegions.end())
              nextBdId =
                  (*bdRegionIt->getBlocks().front().getOps<DMABDOp>().begin())
                      .getBdId();
            else if (dmaOp.getLoop())
              nextBdId = (*bdRegions.front()
                               .getBlocks()
                               .front()
                               .getOps<DMABDOp>()
                               .begin())
                             .getBdId();
            bd.setNextBdId(nextBdId);
          }
        }
      } else {
        DenseMap<Block *, int> blockBdIdMap;
        for (Block &block : memOp.getOperation()->getRegion(0)) {
          if (block.getOps<DMABDOp>().empty())
            continue;
          DMABDOp bd = *block.getOps<DMABDOp>().begin();
          assert(bd.getBdId().has_value() &&
                 "DMABDOp should have bd_id assigned by now");
          blockBdIdMap[&block] = bd.getBdId().value();
        }

        for (Block &block : memOp.getOperation()->getRegion(0)) {
          if (block.getOps<DMABDOp>().empty())
            continue;
          DMABDOp bd = *block.getOps<DMABDOp>().begin();
          std::optional<int> nextBdId;
          if (block.getNumSuccessors()) {
            assert(llvm::range_size(block.getSuccessors()) == 1 &&
                   "should have only one successor block");
            Block *nextBlock = block.getSuccessor(0);
            if (!blockBdIdMap.contains(nextBlock))
              assert(nextBlock->getOperations().size() == 1 &&
                     isa<EndOp>(nextBlock->getOperations().front()) &&
                     "bb that's not in blockMap can only have aie.end");
            else
              nextBdId = blockBdIdMap[nextBlock];
            bd.setNextBdId(nextBdId);
          }
        }
      }
    }

    populateRearmBindings(targetOp);
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferDescriptorIDsPass() {
  return std::make_unique<AIEAssignBufferDescriptorIDsPass>();
}