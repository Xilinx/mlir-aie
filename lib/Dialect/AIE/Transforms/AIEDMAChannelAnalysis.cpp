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
  for (auto program : device.getOps<DmaBody>()) {
    for (Block &block : program.getDmaBody()) {
      for (auto start : block.getOps<DMAStartOp>()) {
        usedChannels.insert({program.getTile(), start.getChannelDir(),
                             start.getChannelIndex()});
      }
    }
  }

  for (auto flowOp : device.getOps<FlowOp>()) {
    if (flowOp.getSourceBundle() == WireBundle::Core) {
      usedStreams.insert(
          {flowOp.getSource(), DMAChannelDir::MM2S, flowOp.getSourceChannel()});
    }
    if (flowOp.getDestBundle() == WireBundle::Core) {
      usedStreams.insert(
          {flowOp.getDest(), DMAChannelDir::S2MM, flowOp.getDestChannel()});
    }
  }

  // Shim allocations reserve channels outside the DMA bodies above.
  for (auto allocOp : device.getOps<ShimDMAAllocationOp>()) {
    auto tile = allocOp.getTileOp();
    if (!tile) {
      continue;
    }
    usedChannels.insert({tile.getResult(), allocOp.getChannelDir(),
                         (int)allocOp.getChannelIndex()});
  }
}

/// Given a tile and DMAChannelDir, returns next usable channel index for
/// that tile.
int DMAChannelAnalysis::getDMAChannelIndex(
    TileLike tile, DMAChannelDir dir, bool requiresAdjacentTileAccessChannels) {
  int maxChannelNum = (dir == DMAChannelDir::MM2S)
                          ? tile.getNumSourceConnections(WireBundle::DMA)
                          : tile.getNumDestConnections(WireBundle::DMA);

  // Reaching a neighbor's memory restricts the range, and which neighbor a
  // tile has is only known once it is placed.
  std::optional<int> col = tile.tryGetCol();
  std::optional<int> row = tile.tryGetRow();
  if (requiresAdjacentTileAccessChannels && col && row) {
    const auto &targetModel = getTargetModel(tile);
    maxChannelNum = std::min<int>(
        maxChannelNum,
        targetModel.getMaxChannelNumForAdjacentMemTile(*col, *row));
  }

  for (int i = 0; i < maxChannelNum; i++) {
    if (reservePinnedChannel(tile, dir, i) >= 0) {
      return i;
    }
  }
  return -1;
}

/// Reserve `channel`, returning -1 when it is unavailable. The caller owns the
/// endpoint-specific diagnostic.
int DMAChannelAnalysis::reservePinnedChannel(TileLike tile, DMAChannelDir dir,
                                             int channel) {
  int maxChannelNum = (dir == DMAChannelDir::MM2S)
                          ? tile.getNumSourceConnections(WireBundle::DMA)
                          : tile.getNumDestConnections(WireBundle::DMA);
  if (channel < 0 || channel >= maxChannelNum) {
    return -1;
  }
  return usedChannels.insert({tile->getResult(0), dir, channel}).second
             ? channel
             : -1;
}

/// Claims a raw stream port, reporting on `tile` when it is already taken.
void DMAChannelAnalysis::checkAIEStreamIndex(TileLike tile, DMAChannel chan) {
  if (usedStreams.insert({tile->getResult(0), chan.direction, chan.channel})
          .second) {
    return;
  }
  if (chan.direction == DMAChannelDir::MM2S) {
    tile->emitOpError("number of output Core channels exceeded!");
  } else {
    tile->emitOpError("number of input Core channels exceeded!");
  }
}
