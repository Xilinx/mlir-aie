//===- AIEDMAChannelAnalysis.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"

#include <functional>
#include <numeric>

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
        if (start.getEndpoint())
          continue;
        usedChannels.try_emplace(std::make_tuple(getTileKey(program.getTile()),
                                                 start.getChannelDir(),
                                                 start.getChannelIndex()),
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
      usedStreams.try_emplace({getTileKey(flowOp.getSource()),
                               DMAChannelDir::MM2S, flowOp.getSourceChannel()},
                              StreamClaim{flowOp, false});
    }
    if (flowOp.getDestBundle() == WireBundle::Core) {
      usedStreams.try_emplace({getTileKey(flowOp.getDest()),
                               DMAChannelDir::S2MM, flowOp.getDestChannel()},
                              StreamClaim{flowOp, false});
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

  for (auto flow : device.getOps<PacketFlowOp>()) {
    Block &ports = flow.getPorts().front();
    for (auto source : ports.getOps<PacketSourceOp>())
      if (source.getBundle() == WireBundle::Core)
        usedStreams.try_emplace({getTileKey(source.getTile()),
                                 DMAChannelDir::MM2S, source.channelIndex()},
                                StreamClaim{source, true});
    for (auto dest : ports.getOps<PacketDestOp>())
      if (dest.getBundle() == WireBundle::Core)
        usedStreams.try_emplace({getTileKey(dest.getTile()),
                                 DMAChannelDir::S2MM, dest.channelIndex()},
                                StreamClaim{dest, true});
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
                                                      Operation *user,
                                                      bool packet) {
  auto [it, inserted] = usedStreams.try_emplace(
      {getTileKey(tile->getResult(0)), chan.direction, chan.channel},
      StreamClaim{user, packet});
  if (inserted || (packet && it->second.packet)) {
    return success();
  }
  auto diag = user->emitOpError("Core ")
              << (chan.direction == DMAChannelDir::MM2S ? "output " : "input ")
              << chan.channel
              << " is already in use on this tile; a core stream port carries "
                 "one circuit or only packet flows";
  diag.attachNote(it->second.owner->getLoc()) << "the other stream is here";
  return diag;
}

std::optional<SmallVector<int>>
DMAChannelAnalysis::planBDChannels(const AIETargetModel &target, int col,
                                   int row, ArrayRef<BDPlanEnd> ends,
                                   function_ref<bool(DMAChannelDir, int)> free,
                                   function_ref<int64_t(int)> spent) {
  uint32_t numBDs = target.getNumBDs(col, row);
  int numChannels = std::max(
      target.getNumSourceSwitchboxConnections(col, row, WireBundle::DMA),
      target.getNumDestSwitchboxConnections(col, row, WireBundle::DMA));

  // Channels whose BD ids overlap at all share one pool of ids, as bdRoom in
  // aie-objectfifo-allocate counts them.
  struct Group {
    llvm::BitVector ids;
    SmallVector<int> channels;
    int64_t room = 0;
  };
  SmallVector<Group> groups;
  for (int channel = 0; channel < numChannels; ++channel) {
    llvm::BitVector ids(numBDs);
    for (uint32_t bd = 0; bd < numBDs; ++bd)
      if (target.isBdChannelAccessible(col, row, bd, channel))
        ids.set(bd);
    Group merged{ids, {channel}};
    for (auto *it = groups.begin(); it != groups.end();) {
      if (!it->ids.anyCommon(merged.ids)) {
        ++it;
        continue;
      }
      merged.ids |= it->ids;
      llvm::append_range(merged.channels, it->channels);
      it = groups.erase(it);
    }
    groups.push_back(std::move(merged));
  }
  for (Group &group : groups) {
    llvm::sort(group.channels);
    group.room = group.ids.count();
    for (int channel : group.channels)
      group.room -= spent(channel);
  }

  // Each end goes to a group, the most demanding first; a choice stands when
  // every group's ends fit its ids and, within a direction, take distinct free
  // channels below their bounds: the tightest bound first, each the lowest
  // channel left, which finds a matching whenever one exists since the
  // allowed channels of any two ends nest.
  SmallVector<unsigned> order(ends.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(
      order, [&](unsigned a, unsigned b) { return ends[a].bds > ends[b].bds; });
  SmallVector<unsigned> groupOf(ends.size());
  SmallVector<int> channelOf(ends.size(), -1);
  auto match = [&]() {
    SmallVector<unsigned> byBound(ends.size());
    std::iota(byBound.begin(), byBound.end(), 0);
    llvm::stable_sort(byBound, [&](unsigned a, unsigned b) {
      return ends[a].limit < ends[b].limit;
    });
    llvm::DenseSet<std::pair<int, int>> taken;
    for (unsigned end : byBound) {
      channelOf[end] = -1;
      for (int channel : groups[groupOf[end]].channels) {
        int dir = static_cast<int>(ends[end].dir);
        if (channel < ends[end].limit && free(ends[end].dir, channel) &&
            taken.insert({dir, channel}).second) {
          channelOf[end] = channel;
          break;
        }
      }
      if (channelOf[end] < 0)
        return false;
    }
    return true;
  };
  std::function<bool(size_t)> place = [&](size_t next) {
    if (next == order.size())
      return match();
    unsigned end = order[next];
    for (unsigned g = 0; g < groups.size(); ++g) {
      if (groups[g].room < ends[end].bds)
        continue;
      groupOf[end] = g;
      groups[g].room -= ends[end].bds;
      bool placed = place(next + 1);
      groups[g].room += ends[end].bds;
      if (placed)
        return true;
    }
    return false;
  };
  if (!place(0))
    return std::nullopt;
  return channelOf;
}
