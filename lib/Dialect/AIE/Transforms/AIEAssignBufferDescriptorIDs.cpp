//===- AIEAssignBufferDescriptorIDs.cpp -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"

#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "aie-assign-bd-ids"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

BdIdGenerator::BdIdGenerator(int col, int row,
                             const AIETargetModel &targetModel)
    : col(col), row(row), targetModel(targetModel) {}

std::optional<uint32_t> BdIdGenerator::nextBdId(int channelIndex) {
  uint32_t bd_id = 0;
  // Find the next free BD ID. This is not an efficient algorithm, but doesn't need to be since BD IDs are small numbers.
  // FIXME: Specify WireBundle
  for(; bdIdAlreadyAssigned(bd_id) || !targetModel.bdCanAccessChannel(col, row, bd_id, channelIndex); bd_id++);
  if(bd_id >= targetModel.getNumBDs(col, row)) {
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

void BdIdGenerator::freeBdId(uint32_t bdId) {
  alreadyAssigned.erase(bdId);
}

struct AIEAssignBufferDescriptorIDsPass
    : AIEAssignBufferDescriptorIDsBase<AIEAssignBufferDescriptorIDsPass> {
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
              std::optional<int32_t> next_id = gen.nextBdId(dmaOp.getChannelIndex());
              if(!next_id) {
                bd.emitOpError() << "Allocator exhausted available BD IDs (maximum " << targetModel.getNumBDs(col, row) << " available).";
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
            std::optional<int32_t> next_id = gen.nextBdId(blockChannelMap[&block]);
            if(!next_id) {
              bd.emitOpError() << "Allocator exhausted available BD IDs (maximum " << targetModel.getNumBDs(col, row) << " available).";
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
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferDescriptorIDsPass() {
  return std::make_unique<AIEAssignBufferDescriptorIDsPass>();
}