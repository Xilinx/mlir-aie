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
#include <set>
#include <vector>

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
    (AIEDialect::ShareableWithAttrHelper(allocOp->getContext())
             .isAttrPresent(allocOp)
         ? shareableChannels
         : usedChannels)
        .try_emplace(std::make_tuple(getTileKey(allocOp.getTile()),
                                     allocOp.getChannelDir(),
                                     (int)allocOp.getChannelIndex()),
                     allocOp.getOperation());
  }
}

int DMAChannelAnalysis::getDMAChannelLimit(
    TileLike tile, DMAChannelDir dir, bool requiresAdjacentTileAccessChannels) {
  std::optional<int> col = tile.tryGetCol();
  std::optional<int> row = tile.tryGetRow();
  const auto &targetModel = getTargetModel(tile);
  // A tile that is not fully placed stands for any tile of its type.
  TileID id =
      col && row
          ? TileID{*col, *row}
          : cast<LogicalTileOp>(tile.getOperation()).getCanonicalTileID();
  int maxChannelNum = targetModel.getNumDMAChannels(id.col, id.row, dir);

  if (!requiresAdjacentTileAccessChannels)
    return maxChannelNum;

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
  return !usedChannels.contains(key) && !streamedChannels.contains(key) &&
         !shareableChannels.contains(key);
}

int DMAChannelAnalysis::reservePinnedChannel(TileLike tile, DMAChannelDir dir,
                                             int channel, Operation *owner,
                                             bool shareable) {
  int maxChannelNum = getDMAChannelLimit(tile, dir, false);
  if (channel < 0 || channel >= maxChannelNum) {
    return -1;
  }
  auto key = std::make_tuple(getTileKey(tile->getResult(0)), dir, channel);
  if (auto *reservation = shareableChannels.lookup(key)) {
    if (!shareable)
      return -1;
    auto allowed =
        AIEDialect::ShareableWithAttrHelper(reservation->getContext())
            .getAttr(reservation);
    auto name =
        owner ? owner->getAttrOfType<StringAttr>("sym_name") : StringAttr();
    if (!allowed || !name || llvm::none_of(allowed, [&](Attribute attr) {
          auto symbol = dyn_cast<FlatSymbolRefAttr>(attr);
          return symbol && symbol.getAttr() == name;
        }))
      return -1;
  }
  if (usedChannels.try_emplace(key, owner).second) {
    if (shareable)
      sharedChannels.insert(key);
    return channel;
  }
  // Packet-switched sources share a channel: the runtime writes its
  // descriptors one transfer at a time, each under its own header.
  if (shareable && sharedChannels.contains(key))
    return channel;
  return -1;
}

Operation *DMAChannelAnalysis::getDMAChannelOwner(TileLike tile,
                                                  DMAChannelDir dir,
                                                  int channel) {
  auto key = std::make_tuple(getTileKey(tile->getResult(0)), dir, channel);
  if (Operation *owner = usedChannels.lookup(key))
    return owner;
  if (Operation *owner = streamedChannels.lookup(key))
    return owner;
  return shareableChannels.lookup(key);
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

std::optional<unsigned> DMAChannelAnalysis::nextPoolToWalk(
    ArrayRef<BDPlanEnd> ends, ArrayRef<unsigned> poolOf,
    ArrayRef<BDWalkPool> pools, ArrayRef<bool> whole) {
  SmallVector<int64_t> bds(pools.size(), 0);
  for (auto [end, pool] : llvm::zip_equal(ends, poolOf))
    bds[pool] += end.bds;
  auto key = [&](unsigned pool) {
    return std::make_tuple(pools[pool].depth, -pools[pool].objectBytes,
                           bds[pool]);
  };
  std::optional<unsigned> next;
  for (unsigned pool = 0, e = pools.size(); pool < e; ++pool)
    if (pools[pool].walkable && !whole[pool] &&
        (!next || key(pool) > key(*next)))
      next = pool;
  return next;
}

DMAChannelAnalysis::BDWalkSets DMAChannelAnalysis::walkSetsThatFit(
    const AIETargetModel &target, int col, int row, ArrayRef<BDPlanEnd> ends,
    ArrayRef<unsigned> poolOf, ArrayRef<BDWalkPool> pools,
    function_ref<bool(DMAChannelDir, int)> free,
    function_ref<int64_t(int)> spent) {
  BDWalkSets result;
  auto none = [](int) -> int64_t { return 0; };
  auto fitsWalking = [&](ArrayRef<bool> whole, bool weighted) {
    SmallVector<BDPlanEnd> planned;
    for (auto [end, pool] : llvm::zip_equal(ends, poolOf))
      planned.push_back({end.dir,
                         !weighted     ? 0
                         : whole[pool] ? 1
                                       : end.bds,
                         end.limit});
    return planBDChannels(target, col, row, planned, free,
                          weighted ? spent : function_ref<int64_t(int)>(none))
        .has_value();
  };
  SmallVector<bool> whole(pools.size(), false);
  if (!fitsWalking(whole, /*weighted=*/false)) {
    result.channelShort = true;
    return result;
  }
  if (fitsWalking(whole, /*weighted=*/true))
    return result;

  SmallVector<unsigned> ranked;
  while (std::optional<unsigned> next =
             nextPoolToWalk(ends, poolOf, pools, whole)) {
    ranked.push_back(*next);
    whole[*next] = true;
  }
  if (!fitsWalking(whole, /*weighted=*/true)) {
    result.fits = false;
    return result;
  }
  // Walking a pool only lowers its ends' BD ids, so a set holding one that
  // fits fits too, without asking planBDChannels.
  unsigned n = ranked.size();
  assert(n < 32 && "each walkable pool draws a channel the plan above fit");
  SmallVector<uint32_t> masks;
  llvm::BitVector fitsMask(uint32_t(1) << n);
  for (uint32_t mask = 1; mask < (uint32_t(1) << n); ++mask) {
    bool fits = llvm::any_of(llvm::seq<unsigned>(0, n), [&](unsigned bit) {
      return (mask >> bit & 1) && fitsMask.test(mask & ~(uint32_t(1) << bit));
    });
    if (!fits) {
      llvm::fill(whole, false);
      for (unsigned bit = 0; bit < n; ++bit)
        if (mask >> bit & 1)
          whole[ranked[bit]] = true;
      fits = fitsWalking(whole, /*weighted=*/true);
    }
    if (fits) {
      fitsMask.set(mask);
      masks.push_back(mask);
    }
  }
  auto bytes = [&](uint32_t mask) {
    int64_t total = 0;
    for (unsigned bit = 0; bit < n; ++bit)
      if (mask >> bit & 1)
        total += pools[ranked[bit]].depth * pools[ranked[bit]].objectBytes;
    return total;
  };
  llvm::stable_sort(masks, [&](uint32_t a, uint32_t b) {
    int64_t aBytes = bytes(a), bBytes = bytes(b);
    if (aBytes != bBytes)
      return aBytes < bBytes;
    if (llvm::popcount(a) != llvm::popcount(b))
      return llvm::popcount(a) < llvm::popcount(b);
    // Of two sets alike in size, the one holding the first-ranked pool that
    // only one holds.
    uint32_t differ = a ^ b;
    return differ && (a & differ & -differ);
  });
  for (uint32_t mask : masks) {
    SmallVector<unsigned> set;
    for (unsigned bit = 0; bit < n; ++bit)
      if (mask >> bit & 1)
        set.push_back(ranked[bit]);
    result.sets.push_back(std::move(set));
    result.bytes.push_back(bytes(mask));
  }
  return result;
}

DMAChannelAnalysis::WalkChoice DMAChannelAnalysis::searchWalkProduct(
    ArrayRef<SmallVector<int64_t>> bytes,
    function_ref<bool(ArrayRef<unsigned>)> places, unsigned maxTries) {
  using Choice = std::vector<unsigned>;
  auto total = [&](const Choice &choice) {
    int64_t sum = 0;
    for (auto [tile, set] : llvm::enumerate(choice))
      sum += bytes[tile][set];
    return sum;
  };
  // Each tile's sets come cheapest first, so a choice's successors (one tile
  // a set further) never cost less: popped in order, the first that places
  // is the cheapest that does.
  std::set<std::pair<int64_t, Choice>> frontier;
  std::set<Choice> seen;
  Choice first(bytes.size(), 0);
  frontier.insert({total(first), first});
  seen.insert(first);
  WalkChoice result;
  result.sets.assign(first.begin(), first.end());
  for (unsigned tries = 0; !frontier.empty(); ++tries) {
    if (tries == maxTries) {
      result.stopped = true;
      return result;
    }
    Choice choice = frontier.begin()->second;
    frontier.erase(frontier.begin());
    if (places(choice)) {
      result.sets.assign(choice.begin(), choice.end());
      result.placed = true;
      return result;
    }
    for (unsigned tile = 0; tile < choice.size(); ++tile) {
      if (choice[tile] + 1 >= bytes[tile].size())
        continue;
      Choice next = choice;
      ++next[tile];
      if (seen.insert(next).second)
        frontier.insert({total(next), next});
    }
  }
  return result;
}
