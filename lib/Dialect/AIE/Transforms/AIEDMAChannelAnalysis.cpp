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

DMAChannelAnalysis::DMAChannelAnalysis(DeviceOp &device) {
  // go over the channels used for each tile and update channel map
  for (auto memOp : device.getOps<MemOp>()) {
    Region &r = memOp.getBody();
    for (auto &bl : r.getBlocks()) {
      for (auto op : bl.getOps<DMAStartOp>()) {
        channelsPerTile[{memOp.getTile(), op.getChannelDir(),
                         op.getChannelIndex()}] = 1;
      }
    }
  }
  for (auto memOp : device.getOps<MemTileDMAOp>()) {
    Region &r = memOp.getBody();
    for (auto &bl : r.getBlocks()) {
      for (auto op : bl.getOps<DMAStartOp>()) {
        channelsPerTile[{memOp.getTile(), op.getChannelDir(),
                         op.getChannelIndex()}] = 1;
      }
    }
  }
  for (auto memOp : device.getOps<ShimDMAOp>()) {
    Region &r = memOp.getBody();
    for (auto &bl : r.getBlocks()) {
      for (auto op : bl.getOps<DMAStartOp>()) {
        channelsPerTile[{memOp.getTile(), op.getChannelDir(),
                         op.getChannelIndex()}] = 1;
      }
    }
  }
  for (auto flowOp : device.getOps<FlowOp>()) {
    if (flowOp.getSourceBundle() == WireBundle::Core)
      aieStreamsPerTile[{flowOp.getSource(), DMAChannelDir::MM2S,
                         flowOp.getSourceChannel()}] = 1;
    if (flowOp.getDestBundle() == WireBundle::Core)
      aieStreamsPerTile[{flowOp.getDest(), DMAChannelDir::S2MM,
                         flowOp.getDestChannel()}] = 1;
  }
  // Scan ShimDMAAllocationOps so that channels already claimed (e.g. by
  // the control packet overlay) are marked used in channelsPerTile and are
  // therefore skipped by getDMAChannelIndex when it auto-assigns channels
  // for objectFIFO lowering.
  for (auto allocOp : device.getOps<ShimDMAAllocationOp>()) {
    auto tile = allocOp.getTileOp();
    if (!tile)
      continue;
    channelsPerTile[{tile.getResult(), allocOp.getChannelDir(),
                     (int)allocOp.getChannelIndex()}] = 1;
  }
}

/// Given a tile and DMAChannelDir, returns next usable channel index for
/// that tile.
int DMAChannelAnalysis::getDMAChannelIndex(
    TileLike tile, DMAChannelDir dir, bool requiresAdjacentTileAccessChannels) {
  int maxChannelNum = 0;
  if (dir == DMAChannelDir::MM2S)
    maxChannelNum = tile.getNumSourceConnections(WireBundle::DMA);
  else
    maxChannelNum = tile.getNumDestConnections(WireBundle::DMA);

  // Reaching a neighbour's memory restricts the range, and which neighbour a
  // tile has is only known once it is placed.
  std::optional<int> col = tile.tryGetCol();
  std::optional<int> row = tile.tryGetRow();
  if (requiresAdjacentTileAccessChannels && col && row) {
    const auto &targetModel = getTargetModel(tile);
    maxChannelNum = std::min<int>(
        maxChannelNum,
        targetModel.getMaxChannelNumForAdjacentMemTile(*col, *row));
  }

  Value result = tile->getResult(0);
  for (int i = 0; i < maxChannelNum; i++) {
    if (int usageCnt = channelsPerTile[{result, dir, i}]; usageCnt == 0) {
      channelsPerTile[{result, dir, i}] = 1;
      return i;
    }
  }
  return -1;
}

/// Reserve a user-pinned DMA channel for (tileOp, dir). Returns the channel
/// on success; returns -1 if the channel is out of range for the tile or is
/// already in use (the caller emits a diagnostic). Reserving up-front ensures
/// first-free auto-assignment never steals a pinned channel.
int DMAChannelAnalysis::reservePinnedChannel(TileLike tile, DMAChannelDir dir,
                                             int channel) {
  int maxChannelNum = (dir == DMAChannelDir::MM2S)
                          ? tile.getNumSourceConnections(WireBundle::DMA)
                          : tile.getNumDestConnections(WireBundle::DMA);
  if (channel < 0 || channel >= maxChannelNum)
    return -1;
  Value result = tile->getResult(0);
  if (channelsPerTile[{result, dir, channel}] != 0)
    return -1;
  channelsPerTile[{result, dir, channel}] = 1;
  return channel;
}

/// Given a tile and DMAChannel, adds entry to aieStreamsPerTile or
/// throws an error if the stream is already used.
void DMAChannelAnalysis::checkAIEStreamIndex(TileLike tile, DMAChannel chan) {
  Value result = tile->getResult(0);
  if (aieStreamsPerTile.find({result, chan.direction, chan.channel}) ==
      aieStreamsPerTile.end()) {
    aieStreamsPerTile[{result, chan.direction, chan.channel}] = 1;
  } else {
    if (chan.direction == DMAChannelDir::MM2S)
      tile->emitOpError("number of output Core channels exceeded!");
    else
      tile->emitOpError("number of input Core channels exceeded!");
  }
}
