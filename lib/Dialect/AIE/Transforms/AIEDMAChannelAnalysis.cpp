//===- AIEDMAChannelAnalysis.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

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
        // The route endpoint it names claims the channel; see assignChannels.
        std::optional<int32_t> index = start.getChannelIndex();
        if (!index)
          continue;
        usedChannels.try_emplace(std::make_tuple(getTileKey(program.getTile()),
                                                 start.getChannelDir(), *index),
                                 start.getOperation());
      }
    }
  }

  // A flow ending at a tile's DMA claims that channel's stream even when no
  // DMA body here starts it: its BDs may come from the runtime sequence alone.
  // First-free assignment must skip it, but a pinned request may name it --
  // that is the DMA feeding or draining the flow, as when allocation reruns
  // over flows it lowered itself.
  auto streamDMA = [&](Value tile, DMAChannelDir dir, int channel,
                       Operation *owner) {
    streamedChannels.try_emplace(
        std::make_tuple(getTileKey(tile), dir, channel), owner);
  };
  for (auto flowOp : device.getOps<FlowOp>()) {
    if (flowOp.getSourceBundle() == WireBundle::Core) {
      usedStreams.insert({getTileKey(flowOp.getSource()), DMAChannelDir::MM2S,
                          flowOp.getSourceChannel()});
    }
    if (flowOp.getDestBundle() == WireBundle::Core) {
      usedStreams.insert({getTileKey(flowOp.getDest()), DMAChannelDir::S2MM,
                          flowOp.getDestChannel()});
    }
    if (flowOp.getSourceBundle() == WireBundle::DMA)
      streamDMA(flowOp.getSource(), DMAChannelDir::MM2S,
                flowOp.getSourceChannel(), flowOp);
    if (flowOp.getDestBundle() == WireBundle::DMA)
      streamDMA(flowOp.getDest(), DMAChannelDir::S2MM, flowOp.getDestChannel(),
                flowOp);
  }
  for (auto packetFlow : device.getOps<PacketFlowOp>()) {
    for (auto source : packetFlow.getOps<PacketSourceOp>())
      if (source.getBundle() == WireBundle::DMA)
        streamDMA(source.getTile(), DMAChannelDir::MM2S, source.getChannel(),
                  packetFlow);
    for (auto dest : packetFlow.getOps<PacketDestOp>())
      if (dest.getBundle() == WireBundle::DMA)
        streamDMA(dest.getTile(), DMAChannelDir::S2MM, dest.getChannel(),
                  packetFlow);
  }

  // A task the runtime sequence configures on a channel by index programs that
  // channel, as a DMA body would.
  device.walk([&](AIEX::DMAConfigureTaskOp task) {
    usedChannels.try_emplace(std::make_tuple(getTileKey(task.getTile()),
                                             task.getDirection(),
                                             (int)task.getChannel()),
                             task.getOperation());
  });

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
    if (isChannelFree(tile, dir, i) &&
        reservePinnedChannel(tile, dir, i, owner) >= 0) {
      return i;
    }
  }
  return -1;
}

bool DMAChannelAnalysis::isChannelFree(TileLike tile, DMAChannelDir dir,
                                       int channel) {
  auto key = std::make_tuple(getTileKey(tile->getResult(0)), dir, channel);
  return !usedChannels.contains(key) && !streamedChannels.contains(key);
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
  auto key = std::make_tuple(getTileKey(tile->getResult(0)), dir, channel);
  if (Operation *owner = usedChannels.lookup(key))
    return owner;
  return streamedChannels.lookup(key);
}

LogicalResult DMAChannelAnalysis::checkAIEStreamIndex(TileLike tile,
                                                      DMAChannel chan,
                                                      bool diagnose) {
  if (usedStreams
          .insert(
              {getTileKey(tile->getResult(0)), chan.direction, chan.channel})
          .second) {
    return success();
  }
  if (!diagnose)
    return failure();
  if (chan.direction == DMAChannelDir::MM2S)
    return tile->emitOpError("number of output Core channels exceeded!");
  return tile->emitOpError("number of input Core channels exceeded!");
}
