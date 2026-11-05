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
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <limits>
#include <optional>

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
    /// The first group placing them in order finds no room for, and why.
    std::optional<size_t> failed;
    Placed failure;
    /// How many groups no assignment finds room for.
    size_t unassigned = 0;
  };

  /// Places `groups` in order. When one finds no room and all take the same
  /// count, a tile holds a whole number of groups, so they go to tiles as a
  /// maximum bipartite matching instead: whether every group fits then does
  /// not depend on their order. The matching augments the in-order
  /// placement, so a group that found room there keeps its tile unless
  /// another needs it.
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
    int count = groups.front().count;
    if (llvm::any_of(groups,
                     [&](const Group &g) { return g.count != count; })) {
      result.unassigned = llvm::count(result.tiles, std::nullopt);
      return result;
    }

    llvm::SmallVector<llvm::SmallVector<unsigned>> eligible;
    for (const Group &group : groups) {
      llvm::SmallVector<unsigned> &tilesOf = eligible.emplace_back();
      for (unsigned tile : group.candidates)
        if (!llvm::is_contained(tilesOf, tile) &&
            llvm::all_of(group.users,
                         [&](const User &user) { return reaches(user, tile); }))
          tilesOf.push_back(tile);
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
    if (!result.unassigned)
      result.failed.reset();
    return result;
  }

  /// How many of `groups` groups of `count` locks each, every group of a
  /// class taking any of `tiles` all its users reach, no assignment finds
  /// room for: the groups less a maximum flow, which is the size of the
  /// matching assign() falls back to.
  size_t
  shortfall(llvm::ArrayRef<std::pair<int, llvm::SmallVector<User>>> classes,
            llvm::ArrayRef<unsigned> candidates, int count) {
    size_t sources = classes.size(), sinks = candidates.size();
    llvm::SmallVector<int> slots;
    for (unsigned tile : candidates) {
      std::optional<TileID> at = tiles[tile].at;
      slots.push_back(
          at ? (static_cast<int>(target.getNumLocks(at->col, at->row)) -
                used.lookup(*at)) /
                   count
             : std::numeric_limits<int>::max() / 2);
    }
    llvm::SmallVector<llvm::SmallVector<bool>> reach(sources);
    for (auto [c, entry] : llvm::enumerate(classes))
      for (unsigned tile : candidates)
        reach[c].push_back(llvm::all_of(entry.second, [&](const User &user) {
          return reaches(user, tile);
        }));
    // flow[c][t]: groups of class c on tile t; residual capacities follow.
    llvm::SmallVector<llvm::SmallVector<int>> flow(
        sources, llvm::SmallVector<int>(sinks, 0));
    llvm::SmallVector<int> left, room(slots);
    for (auto &entry : classes)
      left.push_back(entry.first);
    size_t placed = 0;
    while (true) {
      // Breadth-first over classes and tiles for a path from a class with
      // groups left to a tile with room.
      llvm::SmallVector<int> fromClass(sinks, -1), fromTile(sources, -1);
      llvm::SmallVector<unsigned> queue;
      llvm::SmallVector<bool> seen(sources, false);
      for (size_t c = 0; c < sources; ++c)
        if (left[c] > 0) {
          seen[c] = true;
          queue.push_back(c);
        }
      std::optional<size_t> end;
      for (size_t i = 0; i < queue.size() && !end; ++i) {
        unsigned c = queue[i];
        for (size_t t = 0; t < sinks && !end; ++t) {
          if (!reach[c][t] || fromClass[t] >= 0)
            continue;
          fromClass[t] = c;
          if (room[t] > 0) {
            end = t;
            break;
          }
          for (size_t d = 0; d < sources; ++d)
            if (!seen[d] && flow[d][t] > 0) {
              seen[d] = true;
              fromTile[d] = t;
              queue.push_back(d);
            }
        }
      }
      if (!end)
        break;
      size_t t = *end;
      --room[t];
      while (true) {
        int c = fromClass[t];
        ++flow[c][t];
        if (fromTile[c] < 0) {
          --left[c];
          break;
        }
        t = fromTile[c];
        --flow[c][t];
      }
      ++placed;
    }
    size_t total = 0;
    for (auto &entry : classes)
      total += entry.first;
    return total - placed;
  }

private:
  const AIETargetModel &target;
  llvm::SmallVector<Tile> tiles;
  llvm::DenseMap<TileID, int> used;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIELOCKPLAN_H
