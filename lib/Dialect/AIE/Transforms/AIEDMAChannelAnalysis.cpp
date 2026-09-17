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
        usedChannels[std::make_tuple(program.getTile(), start.getChannelDir(),
                                     start.getChannelIndex())] =
            ChannelOccupant::Circuit;
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
    bool isPkt = (bool)allocOp.getPacket();
    usedChannels[std::make_tuple(tile.getResult(), allocOp.getChannelDir(),
                                 (int)allocOp.getChannelIndex())] =
        isPkt ? ChannelOccupant::Packet : ChannelOccupant::Circuit;
  }
}

int DMAChannelAnalysis::getDMAChannelIndex(
    TileLike tile, DMAChannelDir dir, bool requiresAdjacentTileAccessChannels,
    bool isPacket) {
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

  // First-free, not load-balanced: for an unpinned PACKET endpoint this
  // returns the first channel that admits it, so once a channel already
  // holds a Packet occupant, subsequent unpinned packet legs co-tenant that
  // same channel instead of spreading out to a still-free one. Currently
  // unreachable (no in-tree design has two unpinned packet shim legs on one
  // tile); flagged for anyone enabling data-data packet sharing.
  for (int i = 0; i < maxChannelNum; i++) {
    if (reservePinnedChannel(tile, dir, i, isPacket) >= 0) {
      return i;
    }
  }
  return -1;
}

int DMAChannelAnalysis::reservePinnedChannel(TileLike tile, DMAChannelDir dir,
                                             int channel, bool isPacket) {
  int maxChannelNum = (dir == DMAChannelDir::MM2S)
                          ? tile.getNumSourceConnections(WireBundle::DMA)
                          : tile.getNumDestConnections(WireBundle::DMA);
  if (channel < 0 || channel >= maxChannelNum) {
    return -1;
  }
  auto key = std::make_tuple(tile->getResult(0), dir, channel);
  auto it = usedChannels.find(key);
  if (it == usedChannels.end()) {
    usedChannels[key] =
        isPacket ? ChannelOccupant::Packet : ChannelOccupant::Circuit;
    return channel;
  }
  // Occupied: only packet-onto-packet co-tenancy is allowed.
  // Reachability: this fires only when TWO design packet legs are assigned to
  // the same shim (tile, dir, channel) through assignChannels. No in-tree
  // design reaches it today -- the resident control overlay shares a design
  // channel via the overlay generator's own sharedWithData path at
  // overlay-generation time, which is upstream of and independent from this
  // analysis, and AIEAutoPacketizeControlIngress flips at most one leg per
  // column, so two design packet legs never land on one channel. This is a
  // correct enabler for future data-data packet sharing; it is exercised
  // today only by test/dialect/AIE/dma-channel-packet-cotenancy.mlir.
  if (isPacket && it->second == ChannelOccupant::Packet) {
    return channel;
  }
  return -1;
}

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
