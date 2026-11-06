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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <optional>
#include <set>
#include <vector>

namespace xilinx::AIE {

/// Where aie-objectfifo-allocate puts locks, on plain records the placers
/// also run: each group, in order, takes its locks on the first of its
/// candidates that every user reaches and that has room.
class LockPlan {
public:
  /// The positions a tile may take, one once placed, and that position.
  struct Tile {
    llvm::SmallVector<TileID, 1> positions;
    std::optional<TileID> at;
  };
  /// A tile using the locks. A compute tile's DMA uses only its own.
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

  explicit LockPlan(const AIETargetModel &target) : target(target) {}

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
    /// A group no assignment finds room for, and why: the first placing them
    /// in order misses, or, after the matching, the first it leaves out.
    std::optional<size_t> failed;
    Placed failure;
    /// How many groups no assignment finds room for.
    size_t unassigned = 0;
  };

  /// Places `groups` in order. When one finds no room and all take the same
  /// count (always on AIE2; on AIE1 when every pool has one depth), a tile
  /// holds a whole number of groups, so they go to tiles as a maximum
  /// bipartite matching instead: whether every group fits then does not
  /// depend on their order. The matching augments the in-order placement, so
  /// a group that found room there keeps its tile unless another needs it.
  /// Groups of different counts (AIE1 pools of different depths) are packed
  /// by an exact search instead, largest first, within kPackBudget steps; when
  /// it finds no packing the in-order placement stands.
  Assignment assign(llvm::ArrayRef<Group> groups) {
    Assignment result;
    llvm::DenseMap<TileID, int> before = used;
    for (auto [index, group] : llvm::enumerate(groups)) {
      Placed placed = place(group);
      result.tiles.push_back(placed.tile);
      if (!placed.tile && !result.failed) {
        result.failed = index;
        result.failure = placed;
      }
    }
    if (!result.failed)
      return result;

    llvm::SmallVector<llvm::SmallVector<unsigned>> eligible;
    for (const Group &group : groups) {
      llvm::SmallVector<unsigned> &tilesOf = eligible.emplace_back();
      for (unsigned tile : group.candidates)
        if (!llvm::is_contained(tilesOf, tile) &&
            llvm::all_of(group.users,
                         [&](const User &user) { return reaches(user, tile); }))
          tilesOf.push_back(tile);
    }
    int count = groups.front().count;
    if (llvm::any_of(groups,
                     [&](const Group &g) { return g.count != count; })) {
      if (std::optional<llvm::SmallVector<std::optional<unsigned>>> packed =
              pack(groups, eligible, before)) {
        result.tiles = std::move(*packed);
        result.failed.reset();
        used = before;
        for (auto [group, tile] : llvm::zip(groups, result.tiles))
          if (tiles[*tile].at)
            used[*tiles[*tile].at] += group.count;
        return result;
      }
      result.unassigned = llvm::count(result.tiles, std::nullopt);
      return result;
    }
    llvm::DenseMap<TileID, llvm::SmallVector<size_t>> members;
    auto assignTo = [&](size_t group, unsigned tile) {
      result.tiles[group] = tile;
      if (tiles[tile].at)
        members[*tiles[tile].at].push_back(group);
    };
    auto unassign = [&](size_t group) {
      if (std::optional<TileID> at = tiles[*result.tiles[group]].at)
        llvm::erase(members[*at], group);
      result.tiles[group].reset();
    };
    for (auto [group, tile] : llvm::enumerate(result.tiles))
      if (tile && tiles[*tile].at)
        members[*tiles[*tile].at].push_back(group);
    auto slots = [&](TileID at) {
      return (static_cast<int>(target.getNumLocks(at.col, at.row)) -
              before.lookup(at)) /
             count;
    };
    llvm::DenseSet<TileID> visited;
    std::function<bool(size_t)> augment = [&](size_t group) {
      for (unsigned tile : eligible[group]) {
        std::optional<TileID> at = tiles[tile].at;
        if (!at) {
          assignTo(group, tile);
          return true;
        }
        if (!visited.insert(*at).second)
          continue;
        if (static_cast<int>(members[*at].size()) < slots(*at)) {
          assignTo(group, tile);
          return true;
        }
        for (size_t other : llvm::SmallVector<size_t>(members[*at])) {
          unsigned held = *result.tiles[other];
          unassign(other);
          if (augment(other)) {
            assignTo(group, tile);
            return true;
          }
          assignTo(other, held);
        }
      }
      return false;
    };
    for (size_t group = 0; group < groups.size(); ++group) {
      if (result.tiles[group])
        continue;
      visited.clear();
      if (!augment(group))
        ++result.unassigned;
    }
    used = before;
    for (std::optional<unsigned> tile : result.tiles)
      if (tile && tiles[*tile].at)
        used[*tiles[*tile].at] += count;
    result.failed.reset();
    for (auto [group, tile] : llvm::enumerate(result.tiles)) {
      if (tile)
        continue;
      result.failed = group;
      result.failure = {std::nullopt, !eligible[group].empty(),
                        eligible[group].empty()
                            ? std::nullopt
                            : std::optional(eligible[group].back())};
      break;
    }
    return result;
  }

  /// Steps the exact packing of groups of different counts may take.
  static constexpr int64_t kPackBudget = 100000;

private:
  /// Each group on a tile in `eligible` with room for it, from the room left
  /// `before` any group, or nothing when no such packing exists within
  /// kPackBudget steps. Largest groups go first; a state (next group, room
  /// on each position) that failed once is not tried again.
  std::optional<llvm::SmallVector<std::optional<unsigned>>>
  pack(llvm::ArrayRef<Group> groups,
       llvm::ArrayRef<llvm::SmallVector<unsigned>> eligible,
       const llvm::DenseMap<TileID, int> &before) const {
    auto order = llvm::to_vector(
        llvm::seq<unsigned>(0, static_cast<unsigned>(groups.size())));
    llvm::stable_sort(order, [&](unsigned a, unsigned b) {
      return groups[a].count > groups[b].count;
    });
    llvm::SmallVector<TileID> positions;
    for (const auto &tilesOf : eligible)
      for (unsigned tile : tilesOf)
        if (tiles[tile].at && !llvm::is_contained(positions, *tiles[tile].at))
          positions.push_back(*tiles[tile].at);
    std::vector<int> room;
    for (TileID at : positions)
      room.push_back(static_cast<int>(target.getNumLocks(at.col, at.row)) -
                     before.lookup(at));
    llvm::SmallVector<std::optional<unsigned>> chosen(groups.size());
    std::set<std::pair<size_t, std::vector<int>>> failed;
    int64_t steps = kPackBudget;
    std::function<bool(size_t)> place = [&](size_t next) {
      if (next == order.size())
        return true;
      if (steps-- <= 0 || failed.count({next, room}))
        return false;
      unsigned group = order[next];
      for (unsigned tile : eligible[group]) {
        std::optional<TileID> at = tiles[tile].at;
        int *left = nullptr;
        if (at) {
          left = &room[llvm::find(positions, *at) - positions.begin()];
          if (*left < groups[group].count)
            continue;
          *left -= groups[group].count;
        }
        chosen[group] = tile;
        if (place(next + 1))
          return true;
        if (left)
          *left += groups[group].count;
        if (steps <= 0)
          return false;
      }
      chosen[group].reset();
      failed.insert({next, room});
      return false;
    };
    if (!place(0))
      return std::nullopt;
    return chosen;
  }

  const AIETargetModel &target;
  llvm::SmallVector<Tile> tiles;
  llvm::DenseMap<TileID, int> used;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIELOCKPLAN_H
