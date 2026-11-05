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
#include <map>
#include <numeric>
#include <set>

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
        if (usedChannels
                .try_emplace(std::make_tuple(getTileKey(program.getTile()),
                                             start.getChannelDir(),
                                             start.getChannelIndex()),
                             start.getOperation())
                .second)
          claimingStarts.push_back(start);
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
  // The switchbox direction is reversed relative to the DMA direction.
  int maxChannelNum = (dir == DMAChannelDir::MM2S)
                          ? tile.getNumDestConnections(WireBundle::DMA)
                          : tile.getNumSourceConnections(WireBundle::DMA);

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

SmallVector<Block *> DMAChannelAnalysis::chainOf(DMAStartOp start) {
  SmallVector<Block *> chain;
  SmallPtrSet<Block *, 8> seen;
  SmallVector<Block *> work{start.getDest()};
  while (!work.empty()) {
    Block *block = work.pop_back_val();
    if (!seen.insert(block).second || block->getOps<DMABDOp>().empty())
      continue;
    chain.push_back(block);
    if (auto next = dyn_cast<NextBDOp>(block->getTerminator()))
      work.push_back(next.getDest());
  }
  return chain;
}

int64_t DMAChannelAnalysis::countBDs(DMAStartOp start) {
  int64_t bds = 0;
  for (Block *block : chainOf(start))
    bds += llvm::range_size(block->getOps<DMABDOp>());
  return bds;
}

DMAChannelAnalysis::TakenBDs::TakenBDs(DeviceOp device) {
  for (auto program : device.getOps<DmaBody>())
    for (Block &block : program.getDmaBody())
      for (auto start : block.getOps<DMAStartOp>())
        if (!start.getEndpointOp())
          if (Operation *tile = program.getTile().getDefiningOp())
            fixed[tile][start.getChannelIndex()] += countBDs(start);
  unsigned order = 0;
  device.walk([&](AIEX::DMAConfigureTaskOp task) {
    Operation *tile = task.getTile().getDefiningOp();
    if (!tile)
      return;
    task.getBody().walk([&](DMABDOp bd) {
      if (std::optional<int32_t> id = bd.getBdId())
        pinned[tile].try_emplace(
            *id, Pin{order++, static_cast<int>(task.getChannel())});
    });
  });
}

DenseMap<int, int64_t>
DMAChannelAnalysis::TakenBDs::at(const AIETargetModel &target, int col, int row,
                                 ArrayRef<Operation *> tiles) const {
  auto here = [&](Operation *tile) {
    auto like = dyn_cast<TileLike>(tile);
    return llvm::is_contained(tiles, tile) ||
           (like && like.tryGetCol() == col && like.tryGetRow() == row);
  };
  DenseMap<int, int64_t> taken;
  for (auto &[tile, channels] : fixed)
    if (here(tile))
      for (auto [channel, bds] : channels)
        taken[channel] += bds;
  std::map<int32_t, Pin> ids;
  for (auto &[tile, pins] : pinned)
    if (here(tile))
      for (auto [id, pin] : pins)
        if (auto [it, added] = ids.try_emplace(id, pin);
            !added && pin.order < it->second.order)
          it->second = pin;
  for (auto [id, pin] : ids)
    for (int channel = 0; channel <= pin.channel; ++channel)
      if (target.isBdChannelAccessible(col, row, id, channel)) {
        ++taken[channel];
        break;
      }
  return taken;
}

SmallVector<std::pair<int, int>>
DMAChannelAnalysis::TakenBDs::positions() const {
  std::set<std::pair<int, int>> at;
  auto add = [&](Operation *tile) {
    auto like = dyn_cast<TileLike>(tile);
    if (!like)
      return;
    if (auto col = like.tryGetCol(), row = like.tryGetRow(); col && row)
      at.insert({*col, *row});
  };
  for (auto &entry : fixed)
    add(entry.first);
  for (auto &entry : pinned)
    add(entry.first);
  return {at.begin(), at.end()};
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
  auto shape = [&](unsigned end) {
    return std::make_tuple(-ends[end].bds, static_cast<int>(ends[end].dir),
                           ends[end].limit);
  };
  llvm::stable_sort(
      order, [&](unsigned a, unsigned b) { return shape(a) < shape(b); });
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
  // Within a group and direction, ends take distinct free channels below
  // their bounds, which nest; so they fit exactly when, for every bound, no
  // more of them have that bound or a tighter one than free channels lie below
  // it. Checked as ends are placed, every completed choice matches, and no
  // group ever holds more ends than it has channels.
  SmallVector<int> bounds;
  for (const BDPlanEnd &end : ends)
    bounds.push_back(end.limit);
  llvm::sort(bounds);
  bounds.erase(llvm::unique(bounds), bounds.end());
  size_t slots = 2 * bounds.size();
  SmallVector<int64_t> below(groups.size() * slots, 0);
  for (auto [g, group] : llvm::enumerate(groups))
    for (DMAChannelDir dir : {DMAChannelDir::S2MM, DMAChannelDir::MM2S})
      for (auto [i, bound] : llvm::enumerate(bounds))
        for (int channel : group.channels)
          below[g * slots + static_cast<int>(dir) * bounds.size() + i] +=
              channel < bound && free(dir, channel);
  SmallVector<int64_t> held(groups.size() * slots, 0);
  auto fits = [&](unsigned g, DMAChannelDir dir) {
    size_t base = g * slots + static_cast<int>(dir) * bounds.size();
    int64_t tighter = 0;
    for (size_t i = 0; i < bounds.size(); ++i) {
      tighter += held[base + i];
      if (tighter > below[base + i])
        return false;
    }
    return true;
  };
  // Over every group at once, too many ends of a direction and bound fail
  // before any is placed.
  for (DMAChannelDir dir : {DMAChannelDir::S2MM, DMAChannelDir::MM2S}) {
    int64_t tighter = 0;
    for (size_t i = 0; i < bounds.size(); ++i) {
      int64_t room = 0;
      for (size_t g = 0; g < groups.size(); ++g)
        room += below[g * slots + static_cast<int>(dir) * bounds.size() + i];
      tighter += llvm::count_if(ends, [&](const BDPlanEnd &end) {
        return end.dir == dir && end.limit == bounds[i];
      });
      if (tighter > room)
        return std::nullopt;
    }
  }
  // Ends of one shape are interchangeable, so each takes a group no lower
  // than the one before it of its shape.
  std::function<bool(size_t)> place = [&](size_t next) {
    if (next == order.size())
      return match();
    unsigned end = order[next];
    size_t slot = static_cast<int>(ends[end].dir) * bounds.size() +
                  (llvm::lower_bound(bounds, ends[end].limit) - bounds.begin());
    unsigned first = next > 0 && shape(order[next - 1]) == shape(end)
                         ? groupOf[order[next - 1]]
                         : 0;
    for (unsigned g = first; g < groups.size(); ++g) {
      if (groups[g].room < ends[end].bds)
        continue;
      ++held[g * slots + slot];
      if (fits(g, ends[end].dir)) {
        groupOf[end] = g;
        groups[g].room -= ends[end].bds;
        bool placed = place(next + 1);
        groups[g].room += ends[end].bds;
        if (placed) {
          --held[g * slots + slot];
          return true;
        }
      }
      --held[g * slots + slot];
    }
    return false;
  };
  if (!place(0))
    return std::nullopt;
  return channelOf;
}
