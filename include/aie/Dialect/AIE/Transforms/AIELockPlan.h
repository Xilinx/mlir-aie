//===- AIELockPlan.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIELOCKPLAN_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIELOCKPLAN_H

#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace xilinx::AIE {

/// Lock placement shared by aie-objectfifo-allocate and the SA placer, on
/// plain records.
class LockPlan {
public:
  /// A tile's candidate positions, and the one placement chose, once it has.
  struct Tile {
    llvm::SmallVector<TileID, 1> positions;
    std::optional<TileID> at;
  };
  /// A tile using the locks, `localOnly` when only its own (dmaLocalOnly).
  struct User {
    unsigned tile;
    bool localOnly;
  };
  struct Group {
    int count;
    llvm::SmallVector<unsigned> candidates;
    llvm::SmallVector<User> users;
  };
  struct Placed {
    std::optional<unsigned> tile;
    /// Some candidate every user reaches.
    bool reachable = false;
    /// The last such candidate, when all were full.
    std::optional<unsigned> full;
  };

private:
  enum class Packing { Found, None, Stopped };

public:
  /// Packings by what the packing reads, the least recently used dropped
  /// first beyond `capacity`. A packing found records each group's tile as
  /// an offset into its eligible tiles.
  class Memo {
  public:
    explicit Memo(size_t capacity) : capacity(capacity) {}

  private:
    friend class LockPlan;
    struct Outcome {
      Packing packing;
      llvm::SmallVector<unsigned> blocked;
      llvm::SmallVector<unsigned> offsets;
    };
    struct Entry {
      std::vector<int64_t> key;
      size_t hash;
      Outcome outcome;
    };
    using Entries = std::list<Entry>;

    const Outcome *lookup(const std::vector<int64_t> &key, size_t hash) {
      auto [first, last] = index.equal_range(hash);
      for (auto hit = first; hit != last; ++hit)
        if (hit->second->key == key) {
          entries.splice(entries.begin(), entries, hit->second);
          return &hit->second->outcome;
        }
      return nullptr;
    }
    void record(std::vector<int64_t> key, size_t hash, Outcome outcome) {
      entries.push_front({std::move(key), hash, std::move(outcome)});
      index.emplace(hash, entries.begin());
      if (entries.size() > capacity) {
        auto oldest = std::prev(entries.end());
        auto [first, last] = index.equal_range(oldest->hash);
        index.erase(std::find_if(first, last, [&](const auto &hit) {
          return hit.second == oldest;
        }));
        entries.pop_back();
      }
    }

    size_t capacity;
    Entries entries;
    std::unordered_multimap<size_t, Entries::iterator> index;
  };

  explicit LockPlan(const AIETargetModel &target) : target(target) {}

  /// The lock groups `owners` lock owners of a pool of `depth` objects take,
  /// by count. With semaphore locks each owner is a segment taking a pair,
  /// one lock counting free objects and one full ones; otherwise the one
  /// owner is the pool, taking a binary lock per object.
  static llvm::SmallVector<int> groupCounts(const AIETargetModel &target,
                                            int64_t depth, int owners) {
    if (target.hasProperty(AIETargetModel::UsesSemaphoreLocks))
      return llvm::SmallVector<int>(owners, 2);
    if (!owners)
      return {};
    return {static_cast<int>(depth)};
  }

  /// Whether a DMA on a tile of `type` uses only its own tile's locks: every
  /// DMA does but a mem tile's.
  static bool dmaLocalOnly(AIETileType type) {
    return type != AIETileType::MemTile;
  }

  unsigned addTile(Tile tile) {
    tiles.push_back(std::move(tile));
    return tiles.size() - 1;
  }

  /// Takes `count` locks on `tile`, or returns false when it has too few
  /// left. A tile not yet placed is not counted.
  bool reserve(unsigned tile, int count) {
    if (!tiles[tile].at)
      return true;
    TileID at = *tiles[tile].at;
    int &taken = used[at];
    if (taken + count > static_cast<int>(target.getNumLocks(at.col, at.row)))
      return false;
    taken += count;
    return true;
  }

  /// Whether a tile at one of `user` reaches memory at one of `memory`; with
  /// `localOnly`, only at its own position.
  static bool reaches(const AIETargetModel &target, llvm::ArrayRef<TileID> user,
                      llvm::ArrayRef<TileID> memory, bool localOnly) {
    for (TileID u : user)
      for (TileID m : memory) {
        if (localOnly && u != m)
          continue;
        auto shared = target.getSharedMemory(u, m);
        if (shared == AIETargetModel::SharedMemory::Second ||
            shared == AIETargetModel::SharedMemory::Either)
          return true;
      }
    return false;
  }
  bool reaches(const User &user, unsigned tile) const {
    return user.tile == tile || reaches(target, tiles[user.tile].positions,
                                        tiles[tile].positions, user.localOnly);
  }

  /// A group of `count` locks a pool's objects are counted by: on `home`,
  /// else, unless the pool must stay `local`, on a tile holding its buffers,
  /// one of its users' tiles, then any tile in `ranked` order.
  static Group group(int count, unsigned home, bool local,
                     llvm::ArrayRef<unsigned> buffers,
                     llvm::ArrayRef<User> users,
                     llvm::ArrayRef<unsigned> ranked) {
    Group group{count, {home}, llvm::to_vector(users)};
    if (!local) {
      llvm::append_range(group.candidates, buffers);
      for (const User &user : users)
        group.candidates.push_back(user.tile);
      llvm::append_range(group.candidates, ranked);
    }
    return group;
  }

  Placed place(const Group &group) {
    Placed placed;
    for (unsigned tile : group.candidates) {
      if (!llvm::all_of(group.users,
                        [&](const User &user) { return reaches(user, tile); }))
        continue;
      placed.reachable = true;
      if (reserve(tile, group.count)) {
        placed.tile = tile;
        placed.full.reset();
        return placed;
      }
      placed.full = tile;
    }
    return placed;
  }

  struct Assignment {
    llvm::SmallVector<std::optional<unsigned>> tiles;
    /// The first group no assignment finds room for: first fit's miss, or,
    /// after packing or matching run, the first one still left over.
    std::optional<size_t> failed;
    Placed failure;
    /// How many groups no assignment finds room for.
    size_t unassigned = 0;
    /// The exact packing ran out of kPackBudget steps before it decided.
    bool stopped = false;
  };

  /// Places `groups` first fit: with `callerOrder`, in order on each one's
  /// candidates, then in canonical order (canonicalFirstFit); without, in
  /// canonical order only. On a miss: exact bipartite matching when every
  /// group takes one count, else exact packing, which leaves the first-fit
  /// placement standing if no packing exists or the search runs out of
  /// kPackBudget steps. The SA places in canonical order only, so it fits
  /// only where allocation's own-order-first pass fits too.
  Assignment assign(llvm::ArrayRef<Group> groups, Memo *memo = nullptr,
                    bool callerOrder = true);

  /// Steps the exact packing of groups of different counts may take on each
  /// connected component.
  static constexpr int64_t kPackBudget = 1000000;

private:
  /// Places each group, larger counts first, then by the positions it may
  /// take, on the first of `eligible` with room: a tile not yet placed, then
  /// placed ones by position. The order comes from the groups' counts and
  /// positions alone, and groups alike in both are interchangeable, so every
  /// caller listing the same groups gets the same verdict. Returns each
  /// group's index and placement, in the order placed.
  llvm::SmallVector<std::pair<size_t, Placed>>
  canonicalFirstFit(llvm::ArrayRef<Group> groups,
                    llvm::ArrayRef<llvm::SmallVector<unsigned>> eligible);

  /// Puts each group in `chosen` on a tile in `eligible` with room for it,
  /// from the room left `before` any group; a group that may use a tile not
  /// yet placed takes it. The rest is solved one connected component of
  /// positions at a time by bin completion: the open position with the fewest
  /// groups to choose from takes every subset of them wasting no more room
  /// than the component can spare, then closes. A state whose groups do not
  /// fit even when split is cut off, as is one whose open positions must waste
  /// more than it can spare. Groups alike in count and positions are
  /// interchangeable, so a subset takes how many of each. Each component may
  /// take kPackBudget steps; one that does not decide leaves the packing
  /// Stopped unless another has none, whose groups `blocked` then holds.
  Packing pack(llvm::ArrayRef<Group> groups,
               llvm::ArrayRef<llvm::SmallVector<unsigned>> eligible,
               const llvm::DenseMap<TileID, int> &before,
               llvm::SmallVectorImpl<std::optional<unsigned>> &chosen,
               llvm::SmallVectorImpl<unsigned> &blocked) const;

  const AIETargetModel &target;
  llvm::SmallVector<Tile> tiles;
  llvm::DenseMap<TileID, int> used;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIELOCKPLAN_H
