//===- AIEDMAChannelAnalysis.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

Value DMAChannelAnalysis::getTileKey(Value tile) {
  auto tileOp = cast<TileLike>(tile.getDefiningOp());
  auto col = tileOp.tryGetCol();
  auto row = tileOp.tryGetRow();
  if (!col || !row)
    return tile;
  return tilesByCoordinate.try_emplace({*col, *row}, tile).first->second;
}

DMAChannelAnalysis::DMAChannelAnalysis(DeviceOp &device) {
  for (auto program : device.getOps<DmaBody>()) {
    for (Block &block : program.getDmaBody()) {
      for (auto start : block.getOps<DMAStartOp>()) {
        usedChannels.try_emplace(std::make_tuple(getTileKey(program.getTile()),
                                                 start.getChannelDir(),
                                                 start.getChannelIndex()),
                                 start.getOperation());
      }
    }
  }

  for (auto flowOp : device.getOps<FlowOp>()) {
    if (flowOp.getSourceBundle() == WireBundle::Core) {
      usedStreams.insert({getTileKey(flowOp.getSource()), DMAChannelDir::MM2S,
                          flowOp.getSourceChannel()});
    }
    if (flowOp.getDestBundle() == WireBundle::Core) {
      usedStreams.insert({getTileKey(flowOp.getDest()), DMAChannelDir::S2MM,
                          flowOp.getDestChannel()});
    }
  }

  // Shim allocations reserve channels outside the DMA bodies above.
  for (auto allocOp : device.getOps<ShimDMAAllocationOp>()) {
    usedChannels.try_emplace(std::make_tuple(getTileKey(allocOp.getTile()),
                                             allocOp.getChannelDir(),
                                             (int)allocOp.getChannelIndex()),
                             allocOp.getOperation());
  }
}

int DMAChannelAnalysis::getDMAChannelLimit(
    TileLike tile, DMAChannelDir dir, bool requiresAdjacentTileAccessChannels) {
  int maxChannelNum = (dir == DMAChannelDir::MM2S)
                          ? tile.getNumSourceConnections(WireBundle::DMA)
                          : tile.getNumDestConnections(WireBundle::DMA);

  if (!requiresAdjacentTileAccessChannels)
    return maxChannelNum;

  std::optional<int> col = tile.tryGetCol();
  std::optional<int> row = tile.tryGetRow();
  const auto &targetModel = getTargetModel(tile);
  if (col && row)
    return std::min<int>(
        maxChannelNum,
        targetModel.getMaxChannelNumForAdjacentMemTile(*col, *row));

  // Assignment before placement must be safe at every compatible position,
  // not assume that unresolved coordinates remove neighbor restrictions.
  bool foundPosition = false;
  for (int c = 0; c < targetModel.columns(); ++c) {
    if (col && c != *col)
      continue;
    for (int r = 0; r < targetModel.rows(); ++r) {
      if ((row && r != *row) ||
          targetModel.getTileType(c, r) != tile.getTileType())
        continue;
      foundPosition = true;
      maxChannelNum = std::min<int>(
          maxChannelNum, targetModel.getMaxChannelNumForAdjacentMemTile(c, r));
    }
  }

  return foundPosition ? maxChannelNum : 0;
}

int DMAChannelAnalysis::getDMAChannelIndex(
    TileLike tile, DMAChannelDir dir, bool requiresAdjacentTileAccessChannels,
    Operation *owner) {
  int limit = getDMAChannelLimit(tile, dir, requiresAdjacentTileAccessChannels);
  for (int i = 0; i < limit; i++) {
    if (reservePinnedChannel(tile, dir, i, owner) >= 0) {
      return i;
    }
  }
  return -1;
}

int DMAChannelAnalysis::reservePinnedChannel(TileLike tile, DMAChannelDir dir,
                                             int channel, Operation *owner) {
  int maxChannelNum = getDMAChannelLimit(tile, dir, false);
  if (channel < 0 || channel >= maxChannelNum) {
    return -1;
  }
  return usedChannels
                 .try_emplace(std::make_tuple(getTileKey(tile->getResult(0)),
                                              dir, channel),
                              owner)
                 .second
             ? channel
             : -1;
}

Operation *DMAChannelAnalysis::getDMAChannelOwner(TileLike tile,
                                                  DMAChannelDir dir,
                                                  int channel) {
  return usedChannels.lookup({getTileKey(tile->getResult(0)), dir, channel});
}

void DMAChannelAnalysis::checkAIEStreamIndex(TileLike tile, DMAChannel chan) {
  if (usedStreams
          .insert(
              {getTileKey(tile->getResult(0)), chan.direction, chan.channel})
          .second) {
    return;
  }
  if (chan.direction == DMAChannelDir::MM2S) {
    tile->emitOpError("number of output Core channels exceeded!");
  } else {
    tile->emitOpError("number of input Core channels exceeded!");
  }
}
