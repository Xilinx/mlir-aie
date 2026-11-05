//===- AIEBufferAllocation.h ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEBUFFERALLOCATION_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEBUFFERALLOCATION_H

#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <optional>
#include <utility>

namespace xilinx::AIE {

/// A memtile is reached by DMA rather than by core vector load and store, so
/// its bus width also covers the vector-alignment requirement.
struct TileMemoryLimits {
  int64_t maxDataMemorySize;
  uint32_t tileAlignBitWidth;
  uint32_t maxVecAlignBits;
};
inline TileMemoryLimits tileMemoryLimits(const AIETargetModel &targetModel,
                                         AIETileType tileType) {
  if (tileType == AIETileType::MemTile) {
    return {targetModel.getMemTileSize(),
            targetModel.getMemTileLoadStoreBusWidth(),
            targetModel.getMemTileLoadStoreBusWidth()};
  }
  return {targetModel.getLocalMemorySize(),
          targetModel.getComputeTileLoadStoreBusWidth(),
          targetModel.getComputeTileMaxVectorAlignBits()};
}

/// Return the alignment (in bits) a buffer of `sizeBytes` must satisfy.
///
/// Bus width alone is insufficient: from AIE2P on, a full-width vector access
/// needs 512-bit alignment while the bus is 256 bits wide. An externally
/// compiled kernel may perform such an access, so a buffer large enough to hold
/// a full-width vector gets the stricter alignment. A smaller buffer keeps the
/// bus width and costs no padding.
inline uint32_t requiredBufferAlignBits(uint32_t busAlignBits,
                                        uint32_t maxVecAlignBits,
                                        int64_t sizeBytes) {
  if (maxVecAlignBits <= busAlignBits)
    return busAlignBits;
  return sizeBytes * 8 >= static_cast<int64_t>(maxVecAlignBits)
             ? maxVecAlignBits
             : busAlignBits;
}

struct BufferAllocation {
  int64_t size;
  int64_t alignment;
  std::optional<int64_t> address;
};

/// Buffers are laid out, and given a tile, largest buffer first: by the address
/// layout below, the objectFIFO allocator, and the SA placer's estimate of it.
/// A pool walked whole is one buffer of all its objects. Equal sizes keep their
/// order.
template <typename Range, typename BytesFn>
void sortLargestBufferFirst(Range &&buffers, BytesFn bytes) {
  llvm::stable_sort(buffers, [&](const auto &a, const auto &b) {
    return bytes(a) > bytes(b);
  });
}

/// The basic address allocator and resource planner share this layout: retain
/// address pins, then place unpinned buffers largest-first, aligning each start
/// and skipping occupied ranges. Return the high-water mark, without padding
/// after the final buffer. Entries retain their input order.
inline int64_t
assignSequentialBufferAddresses(llvm::MutableArrayRef<BufferAllocation> buffers,
                                int64_t start = 0) {
  llvm::SmallVector<std::pair<int64_t, int64_t>> pinned;
  llvm::SmallVector<BufferAllocation *> pending;
  int64_t highWater = start;
  for (auto &buffer : buffers) {
    if (buffer.address) {
      pinned.emplace_back(*buffer.address, buffer.size);
      highWater = std::max(highWater, *buffer.address + buffer.size);
    } else {
      pending.push_back(&buffer);
    }
  }
  llvm::stable_sort(pinned, [](auto a, auto b) { return a.first < b.first; });
  sortLargestBufferFirst(pending, [](auto *buffer) { return buffer->size; });
  auto *current = pinned.begin();
  int64_t address = start;
  for (auto *buffer : pending) {
    address = llvm::alignTo(address, buffer->alignment);
    while (current != pinned.end() && address + buffer->size > current->first) {
      address =
          llvm::alignTo(current->first + current->second, buffer->alignment);
      ++current;
    }
    buffer->address = address;
    address += buffer->size;
  }
  return std::max(highWater, address);
}

/// A mem tile pool as aie-objectfifo-allocate packs it: `count` buffers of
/// `bytes` homed on `home`, and the DMA ends it has there in its busier
/// direction.
struct SpillPool {
  int64_t bytes;
  int64_t count;
  unsigned home;
  int demand;
};

/// aie-objectfifo-allocate's demand-first packing order of `pools`, given in
/// its size-first order, as indices into them: each home's pools by demand,
/// then buffer size, merged by buffer size, then demand, then home in order of
/// first appearance, so that small pools are not put first over large buffers
/// on unrelated tiles.
inline llvm::SmallVector<unsigned>
demandFirstOrder(llvm::ArrayRef<SpillPool> pools) {
  llvm::SmallVector<unsigned> homes;
  llvm::SmallVector<llvm::SmallVector<unsigned>> perHome;
  for (auto [index, pool] : llvm::enumerate(pools)) {
    auto *at = llvm::find(homes, pool.home);
    if (at == homes.end()) {
      homes.push_back(pool.home);
      perHome.emplace_back();
      at = std::prev(homes.end());
    }
    perHome[at - homes.begin()].push_back(index);
  }
  for (auto &list : perHome)
    llvm::stable_sort(list, [&](unsigned a, unsigned b) {
      if (pools[a].demand != pools[b].demand)
        return pools[a].demand > pools[b].demand;
      return pools[a].bytes > pools[b].bytes;
    });
  llvm::SmallVector<unsigned> order;
  llvm::SmallVector<size_t> next(perHome.size(), 0);
  for (size_t slot = 0; slot < pools.size(); ++slot) {
    std::optional<size_t> best;
    for (size_t home = 0; home < perHome.size(); ++home) {
      if (next[home] >= perHome[home].size())
        continue;
      const SpillPool &pool = pools[perHome[home][next[home]]];
      const SpillPool *top =
          best ? &pools[perHome[*best][next[*best]]] : nullptr;
      if (!top || pool.bytes > top->bytes ||
          (pool.bytes == top->bytes && pool.demand > top->demand))
        best = home;
    }
    order.push_back(perHome[*best][next[*best]++]);
  }
  return order;
}

/// The tiles a mem tile buffer homed at `home` may spill to: either side of it
/// in its row, where home shares memory with them.
inline llvm::SmallVector<TileID, 2>
memTileSpillNeighbours(const AIETargetModel &target, TileID home) {
  llvm::SmallVector<TileID, 2> neighbours;
  for (int col : {home.col - 1, home.col + 1}) {
    if (col < 0 || col >= target.columns())
      continue;
    TileID neighbour{col, home.row};
    auto shared = target.getSharedMemory(home, neighbour);
    if (shared == AIETargetModel::SharedMemory::Second ||
        shared == AIETargetModel::SharedMemory::Either)
      neighbours.push_back(neighbour);
  }
  return neighbours;
}

/// The mem tiles a mem tile pool's buffers go to: aie-objectfifo-allocate
/// places them by this search, and the SA placer runs it on each placement it
/// weighs. A tile's use is its layout by assignSequentialBufferAddresses. Each
/// buffer in turn goes home when it fits there, else to a neighbour home shares
/// memory with, least used first; when one fits nowhere, earlier choices are
/// revisited, latest first, until kRetries alternatives have been tried.
class MemTileSpillSearch {
public:
  struct Tile {
    int64_t capacity;
    llvm::SmallVector<BufferAllocation> fixed;
  };
  /// A buffer of `bytes` homed on `home`, which it may take unless `atHome`
  /// is false, and the `neighbours` it may spill to, in the order tried when
  /// they are equally used.
  struct Buffer {
    int64_t bytes;
    unsigned home;
    bool atHome;
    llvm::SmallVector<unsigned, 2> neighbours;
  };

  MemTileSpillSearch(llvm::SmallVector<Tile> tiles, int64_t alignment)
      : tiles(std::move(tiles)), planned(this->tiles.size()),
        alignment(alignment) {
    weighed.assign(this->tiles.size(), false);
    plannedBytes.assign(this->tiles.size(), 0);
    misaligned.assign(this->tiles.size(), 0);
    for (const Tile &tile : this->tiles) {
      bool packed = llvm::all_of(tile.fixed, [&](const BufferAllocation &b) {
        return !b.address && b.size % alignment == 0;
      });
      int64_t bytes = 0;
      for (const BufferAllocation &buffer : tile.fixed)
        bytes += buffer.size;
      fixedBytes.push_back(packed ? std::optional<int64_t>(bytes)
                                  : std::nullopt);
    }
  }

  /// The bytes `tile` spans with what it holds and `count` more of `bytes`.
  int64_t used(unsigned tile, int64_t bytes = 0, int count = 0) const {
    // Unpinned buffers whose sizes all keep the alignment leave no gaps, so
    // the layout spans their sum.
    if (fixedBytes[tile] && !misaligned[tile] && bytes % alignment == 0)
      return *fixedBytes[tile] + plannedBytes[tile] + count * bytes;
    llvm::SmallVector<BufferAllocation> layout;
    for (int64_t size : planned[tile])
      layout.push_back({size, alignment, std::nullopt});
    for (int i = 0; i < count; ++i)
      layout.push_back({bytes, alignment, std::nullopt});
    llvm::append_range(layout, tiles[tile].fixed);
    return assignSequentialBufferAddresses(layout);
  }
  bool fits(unsigned tile, int64_t bytes, int count = 1) const {
    return used(tile, bytes, count) <= tiles[tile].capacity;
  }
  void reserve(unsigned tile, int64_t bytes, int count = 1) {
    for (int i = 0; i < count; ++i)
      add(tile, bytes);
  }

  /// Alternatives the search tries before it gives up.
  static constexpr int kRetries = 256;

  /// Places `buffers` in order on top of what the tiles hold. On success each
  /// buffer's tile is in `chosen`; otherwise `failed` is the buffer that fit
  /// nowhere with the fewest buffers left after it, and `unplacedBytes` the
  /// bytes of it and those.
  bool place(llvm::ArrayRef<Buffer> buffers) {
    chosen.clear();
    failed = buffers.size();
    unplacedBytes = 0;
    int retries = kRetries;
    return placeFrom(buffers, 0, retries);
  }
  llvm::SmallVector<unsigned> chosen;
  size_t failed = 0;
  int64_t unplacedBytes = 0;
  /// Homes a spill from was weighed: aie-objectfifo-allocate gives their
  /// neighbours tile ops, as later passes may use them.
  llvm::SmallVector<bool> weighed;

private:
  llvm::SmallVector<unsigned> spillTargets(const Buffer &buffer) {
    weighed[buffer.home] = true;
    llvm::SmallVector<unsigned, 2> neighbours(buffer.neighbours);
    llvm::stable_sort(
        neighbours, [&](unsigned a, unsigned b) { return used(a) < used(b); });
    llvm::SmallVector<unsigned> fitting;
    for (unsigned neighbour : neighbours)
      if (fits(neighbour, buffer.bytes))
        fitting.push_back(neighbour);
    return fitting;
  }

  bool placeFrom(llvm::ArrayRef<Buffer> buffers, size_t index, int &retries) {
    if (index == buffers.size())
      return true;
    const Buffer &buffer = buffers[index];
    bool fitsHome = buffer.atHome && fits(buffer.home, buffer.bytes);
    llvm::SmallVector<unsigned> candidates;
    if (fitsHome)
      candidates.push_back(buffer.home);
    else
      candidates = spillTargets(buffer);
    if (candidates.empty() && index < failed) {
      failed = index;
      unplacedBytes = 0;
      for (const Buffer &left : buffers.drop_front(index))
        unplacedBytes += left.bytes;
    }
    auto tryTile = [&](unsigned tile) {
      add(tile, buffer.bytes);
      chosen.push_back(tile);
      if (placeFrom(buffers, index + 1, retries))
        return true;
      removeLast(tile);
      chosen.pop_back();
      return false;
    };
    for (auto [i, tile] : llvm::enumerate(candidates)) {
      if (i > 0 && retries-- <= 0)
        return false;
      if (tryTile(tile))
        return true;
    }
    if (!fitsHome)
      return false;
    for (unsigned tile : spillTargets(buffer)) {
      if (retries-- <= 0)
        return false;
      if (tryTile(tile))
        return true;
    }
    return false;
  }

  void add(unsigned tile, int64_t bytes) {
    planned[tile].push_back(bytes);
    plannedBytes[tile] += bytes;
    misaligned[tile] += bytes % alignment != 0;
  }
  void removeLast(unsigned tile) {
    int64_t bytes = planned[tile].pop_back_val();
    plannedBytes[tile] -= bytes;
    misaligned[tile] -= bytes % alignment != 0;
  }

  llvm::SmallVector<Tile> tiles;
  llvm::SmallVector<llvm::SmallVector<int64_t>> planned;
  int64_t alignment;
  /// What the fixed buffers span when unpinned and aligned in size, and the
  /// planned buffers' bytes and how many of them are not.
  llvm::SmallVector<std::optional<int64_t>> fixedBytes;
  llvm::SmallVector<int64_t> plannedBytes;
  llvm::SmallVector<int> misaligned;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEBUFFERALLOCATION_H
