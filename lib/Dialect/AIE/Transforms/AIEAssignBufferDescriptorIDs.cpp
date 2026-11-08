//===- AIEAssignBufferDescriptorIDs.cpp -------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>

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

std::optional<uint32_t> BdIdGenerator::nextBdId(int channelIndex,
                                                uint32_t alignment) {
  uint32_t bdId = 0;
  uint32_t maxBdId = targetModel.getNumBDs(col, row) - 1;
  // Find the next free BD ID. This is not an efficient algorithm, but doesn't
  for (; (bdIdAlreadyAssigned(bdId) || reserved.count(bdId) ||
          !targetModel.isBdChannelAccessible(col, row, bdId, channelIndex)) &&
         bdId <= maxBdId;
       bdId += alignment)
    ;
  if (bdId > maxBdId) {
    return std::nullopt;
  }
  assignBdId(bdId);
  return std::optional<uint32_t>(bdId);
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
    if (failed(verifyDMAChannelsResolved(targetOp)))
      return signalPassFailure();
    const AIETargetModel &targetModel = targetOp.getTargetModel();

    auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
    llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
    llvm::append_range(memOps, targetOp.getOps<ShimDMAOp>());

    // BD IDs are per-tile, but a tile can be described by more than one op
    // here -- key the generator by tile, not by op, or two ops on the same
    // tile hand out the same ids and silently collide.
    std::map<std::pair<int, int>, BdIdGenerator> gens;
    for (TileElement memOp : memOps) {
      int col = memOp.getTileID().col;
      int row = memOp.getTileID().row;

      auto emplaced =
          gens.try_emplace(std::make_pair(col, row), col, row, targetModel);
      BdIdGenerator &gen = emplaced.first->second;
      if (emplaced.second) {
        for (TileOp tile : targetOp.getOps<TileOp>()) {
          if (tile.getCol() != col || tile.getRow() != row)
            continue;
          if (auto reserved = tile->getAttrOfType<DenseI32ArrayAttr>(
                  "aiex.reserved_bd_ids"))
            for (int32_t bdId : reserved.asArrayRef()) {
              auto id = static_cast<uint32_t>(bdId);
              if (!gen.bdIdAlreadyAssigned(id))
                gen.assignBdId(id);
            }
        }
      }
      bool bdIdCollision = false;
      memOp->walk<WalkOrder::PreOrder>([&](DMABDOp bd) {
        if (!bd.getBdId().has_value())
          return;
        uint32_t bdId = bd.getBdId().value();
        if (gen.bdIdAlreadyAssigned(bdId)) {
          bd.emitOpError() << "assigned bd_id " << bdId
                           << " is already used by another BD on this tile";
          bdIdCollision = true;
          return;
        }
        gen.assignBdId(bdId);
      });
      if (bdIdCollision)
        return signalPassFailure();
    }

    struct Assignment {
      DMABDOp bd;
      int col, row, channel;
    };
    SmallVector<Assignment> assignments;
    std::map<int32_t, SmallVector<size_t>> receiversOfTag;
    for (TileElement memOp : memOps) {
      DenseMap<Block *, int> channels;
      Region &region = memOp->getRegion(0);
      for (Block &block : region)
        for (auto start : block.getOps<DMAStartOp>()) {
          Block *dest = start.getDest();
          while (dest && !channels.contains(dest)) {
            channels[dest] = start.getChannelIndex();
            dest = dest->hasNoSuccessors() ? nullptr : dest->getSuccessor(0);
          }
        }
      memOp->walk([&](DMABDOp bd) {
        auto dma = bd->getParentOfType<DMAOp>();
        int channel =
            dma ? dma.getChannelIndex() : channels.lookup(bd->getBlock());
        if (auto tag = bd.getBdTag())
          receiversOfTag[*tag].push_back(assignments.size());
        assignments.push_back(
            {bd, memOp.getTileID().col, memOp.getTileID().row, channel});
      });
    }

    // Broadcast destinations must agree on a hardware ID. Reserve their
    // common IDs before ordinary allocation can consume them on one tile.
    for (auto &[tag, indices] : receiversOfTag) {
      if (indices.size() < 2)
        continue;
      std::optional<uint32_t> fixed;
      for (size_t i : indices) {
        if (auto id = assignments[i].bd.getBdId()) {
          if (fixed && *fixed != static_cast<uint32_t>(*id)) {
            assignments[i].bd.emitOpError(
                "broadcast receiving buffers must have matching bd_id values "
                "for bd_tag ")
                << tag;
            return signalPassFailure();
          }
          fixed = id;
        }
      }
      auto available = [&, &indices = indices](uint32_t id) {
        std::set<std::pair<int, int>> tiles;
        for (size_t i : indices) {
          auto &a = assignments[i];
          auto tile = std::make_pair(a.col, a.row);
          if (!tiles.insert(tile).second ||
              id >= targetModel.getNumBDs(a.col, a.row) ||
              !targetModel.isBdChannelAccessible(a.col, a.row, id, a.channel) ||
              (!a.bd.getBdId() && gens.at(tile).bdIdAlreadyAssigned(id)))
            return false;
        }
        return true;
      };
      uint32_t id = fixed.value_or(0);
      auto &first = assignments[indices.front()];
      uint32_t limit = targetModel.getNumBDs(first.col, first.row);
      if (!fixed)
        while (id < limit && !available(id))
          ++id;
      if (id >= limit || !available(id)) {
        first.bd.emitOpError("cannot allocate a common bd_id for broadcast "
                             "receiving buffers with bd_tag ")
            << tag;
        return signalPassFailure();
      }
      for (size_t i : indices) {
        auto &a = assignments[i];
        if (!a.bd.getBdId()) {
          gens.at({a.col, a.row}).assignBdId(id);
          a.bd.setBdId(id);
        }
      }
    }
    for (auto &a : assignments) {
      auto &gen = gens.at({a.col, a.row});
      if (auto id = a.bd.getBdId()) {
        if (!targetModel.isBdChannelAccessible(a.col, a.row, *id, a.channel)) {
          a.bd.emitOpError()
              << "assigned bd_id " << *id << " is not accessible from channel "
              << a.channel << " on this tile";
          return signalPassFailure();
        }
      } else {
        auto nextId = gen.nextBdId(a.channel);
        if (!nextId) {
          a.bd.emitOpError()
              << "Allocator exhausted available BD IDs (maximum "
              << targetModel.getNumBDsForChannel(a.col, a.row, a.channel)
              << " available for channel " << a.channel << ").";
          return signalPassFailure();
        }
        a.bd.setBdId(*nextId);
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
          auto bdId = bd.getBdId();
          assert(bdId.has_value() &&
                 "DMABDOp should have bd_id assigned by now");
          blockBdIdMap[&block] = *bdId;
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
            if (blockBdIdMap.contains(nextBlock)) {
              nextBdId = blockBdIdMap[nextBlock];
            } else if (nextBlock->getOperations().size() != 1 ||
                       !isa<EndOp>(nextBlock->getOperations().front())) {
              bd.emitOpError("must be followed by a block with a dma_bd or "
                             "by a block with only aie.end");
              return signalPassFailure();
            }
            bd.setNextBdId(nextBdId);
          }
        }
      }
    }

    if (failed(resolveOutOfOrderTags(targetOp)))
      return signalPassFailure();
    populateRearmBindings(targetOp);
  }

  /// Resolve temporary receiving-BD tags after hardware ID assignment.
  LogicalResult resolveOutOfOrderTags(DeviceOp device) {
    DenseMap<int32_t, int> idOfTag;
    WalkResult receivers = device.walk([&](DMABDOp bd) {
      std::optional<uint32_t> id = bd.getBdId();
      if (std::optional<int32_t> tag = bd.getBdTag(); tag && id) {
        auto [it, inserted] = idOfTag.try_emplace(*tag, *id);
        if (!inserted && it->second != static_cast<int>(*id)) {
          bd.emitOpError("broadcast receiving buffers must have matching bd_id "
                         "values for bd_tag ")
              << *tag;
          return WalkResult::interrupt();
        }
      }
      bd.removeBdTagAttr();
      return WalkResult::advance();
    });
    if (receivers.wasInterrupted())
      return failure();
    WalkResult walked = device.walk([&](DMABDOp bd) {
      std::optional<int32_t> tag = bd.getOutOfOrderTag();
      if (!tag)
        return WalkResult::advance();
      auto it = idOfTag.find(*tag);
      if (it == idOfTag.end()) {
        bd.emitOpError("sends out of order into a descriptor that no "
                       "receiving tile holds");
        return WalkResult::interrupt();
      }
      bd.setOutOfOrderId(it->second);
      bd.removeOutOfOrderTagAttr();
      return WalkResult::advance();
    });
    return failure(walked.wasInterrupted());
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferDescriptorIDsPass() {
  return std::make_unique<AIEAssignBufferDescriptorIDsPass>();
}
