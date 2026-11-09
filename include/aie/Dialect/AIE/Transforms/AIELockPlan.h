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

#include <algorithm>
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
    /// in order misses (when packing, in a component no packing exists for),
    /// or, after the matching, the first it leaves out.
    std::optional<size_t> failed;
    Placed failure;
    /// How many groups no assignment finds room for.
    size_t unassigned = 0;
    /// The exact packing ran out of kPackBudget steps before it decided.
    bool stopped = false;
  };

  /// Places `groups` in order. When one finds no room and all take the same
  /// count (always on AIE2; on AIE1 when every pool has one depth), a tile
  /// holds a whole number of groups, so they go to tiles as a maximum
  /// bipartite matching instead: whether every group fits then does not
  /// depend on their order. The matching augments the in-order placement, so
  /// a group that found room there keeps its tile unless another needs it.
  /// Groups of different counts (AIE1 pools of different depths) are packed
  /// by an exact search instead; when no packing exists, or the search
  /// stops after kPackBudget steps, the in-order placement stands.
  Assignment assign(llvm::ArrayRef<Group> groups) {
    Assignment result;
    llvm::DenseMap<TileID, int> before = used;
    llvm::SmallVector<Placed> misses;
    for (auto [index, group] : llvm::enumerate(groups)) {
      Placed placed = place(group);
      result.tiles.push_back(placed.tile);
      misses.push_back(placed);
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
      llvm::SmallVector<std::optional<unsigned>> packed;
      llvm::SmallVector<unsigned> blocked;
      Packing packing = pack(groups, eligible, before, packed, blocked);
      if (packing == Packing::Found) {
        result.tiles = std::move(packed);
        result.failed.reset();
        used = before;
        for (auto [group, tile] : llvm::zip(groups, result.tiles))
          if (tiles[*tile].at)
            used[*tiles[*tile].at] += group.count;
        return result;
      }
      result.stopped = packing == Packing::Stopped;
      for (unsigned group : blocked)
        if (!result.tiles[group]) {
          result.failed = group;
          result.failure = misses[group];
          break;
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

  /// Steps the exact packing of groups of different counts may take on each
  /// connected component.
  static constexpr int64_t kPackBudget = 100000;

private:
  enum class Packing { Found, None, Stopped };

  /// Puts each group in `chosen` on a tile in `eligible` with room for it,
  /// from the room left `before` any group. A group that may use a tile not
  /// yet placed takes it. The others are solved one connected component of
  /// positions at a time by bin completion: the open position with the
  /// fewest groups to choose from is filled with every subset of them that
  /// wastes no more room than the component can spare, then closed. A state
  /// whose groups do not fit even when split is cut off. Groups of one count
  /// and one set of positions are interchangeable, so a subset takes how many
  /// of each. Each component may take kPackBudget steps; one that does not
  /// decide in them leaves the packing Stopped unless another has none. With
  /// None, `blocked` holds the groups of the component that has none.
  Packing pack(llvm::ArrayRef<Group> groups,
               llvm::ArrayRef<llvm::SmallVector<unsigned>> eligible,
               const llvm::DenseMap<TileID, int> &before,
               llvm::SmallVectorImpl<std::optional<unsigned>> &chosen,
               llvm::SmallVectorImpl<unsigned> &blocked) const {
    chosen.assign(groups.size(), std::nullopt);
    blocked.clear();
    llvm::SmallVector<TileID> positions;
    llvm::SmallVector<llvm::SmallVector<unsigned>> reach(groups.size());
    for (auto [group, tilesOf] : llvm::enumerate(eligible)) {
      if (tilesOf.empty()) {
        blocked.push_back(group);
        return Packing::None;
      }
      for (unsigned tile : tilesOf) {
        if (!tiles[tile].at) {
          chosen[group] = tile;
          break;
        }
        unsigned position =
            llvm::find(positions, *tiles[tile].at) - positions.begin();
        if (position == positions.size())
          positions.push_back(*tiles[tile].at);
        if (!llvm::is_contained(reach[group], position))
          reach[group].push_back(position);
      }
      llvm::sort(reach[group]);
    }

    llvm::SmallVector<unsigned> component(positions.size());
    for (unsigned position : llvm::seq<unsigned>(0, positions.size()))
      component[position] = position;
    std::function<unsigned(unsigned)> root = [&](unsigned position) {
      return component[position] == position
                 ? position
                 : component[position] = root(component[position]);
    };
    for (auto [group, own] : llvm::enumerate(reach))
      if (!chosen[group])
        for (unsigned position : own)
          component[root(position)] = root(own.front());

    bool anyStopped = false;
    for (unsigned top : llvm::seq<unsigned>(0, positions.size())) {
      if (root(top) != top)
        continue;
      int64_t steps = kPackBudget;
      bool stopped = false;
      llvm::SmallVector<unsigned> local;
      for (unsigned position : llvm::seq<unsigned>(0, positions.size()))
        if (root(position) == top)
          local.push_back(position);
      std::vector<int> room;
      for (unsigned position : local) {
        TileID at = positions[position];
        room.push_back(
            std::max(0, static_cast<int>(target.getNumLocks(at.col, at.row)) -
                            before.lookup(at)));
      }
      // Interchangeable groups, largest count first: count, the local
      // positions they reach, and their members.
      struct Kind {
        int count;
        llvm::SmallVector<unsigned> reach;
        llvm::SmallVector<unsigned> members;
      };
      llvm::SmallVector<Kind> kinds;
      for (auto [group, own] : llvm::enumerate(reach)) {
        if (chosen[group] || own.empty() || root(own.front()) != top)
          continue;
        llvm::SmallVector<unsigned> at;
        for (unsigned position : own)
          at.push_back(llvm::find(local, position) - local.begin());
        auto kind = llvm::find_if(kinds, [&](const Kind &k) {
          return k.count == groups[group].count && k.reach == at;
        });
        if (kind == kinds.end()) {
          kinds.push_back({groups[group].count, at, {}});
          kind = std::prev(kinds.end());
        }
        kind->members.push_back(group);
      }
      llvm::stable_sort(kinds, [](const Kind &a, const Kind &b) {
        return a.count > b.count;
      });

      // The state is which positions are open and how many of each kind
      // are taken; taken members are always a kind's first ones.
      std::vector<int> open(local.size(), 1), taken(kinds.size(), 0);
      llvm::SmallVector<llvm::SmallVector<unsigned>> where(kinds.size());
      std::set<std::pair<std::vector<int>, std::vector<int>>> failed;
      // Whether the open positions' room holds what is left when groups
      // may be split: a maximum flow from kinds to the positions they reach.
      std::vector<std::vector<int>> flow;
      std::vector<int> spare;
      llvm::SmallVector<bool> visited;
      std::function<int(unsigned, int)> augment = [&](unsigned kind,
                                                      int amount) {
        for (unsigned position : kinds[kind].reach) {
          if (!open[position] || visited[position])
            continue;
          visited[position] = true;
          if (int direct = std::min(amount, spare[position])) {
            spare[position] -= direct;
            flow[kind][position] += direct;
            return direct;
          }
          for (unsigned other : llvm::seq<unsigned>(0, kinds.size())) {
            if (other == kind || !flow[other][position])
              continue;
            if (int moved =
                    augment(other, std::min(amount, flow[other][position]))) {
              flow[other][position] -= moved;
              flow[kind][position] += moved;
              return moved;
            }
          }
        }
        return 0;
      };
      auto covers = [&]() {
        flow.assign(kinds.size(), std::vector<int>(local.size(), 0));
        spare = room;
        for (auto [kind, k] : llvm::enumerate(kinds)) {
          int supply = (k.members.size() - taken[kind]) * k.count;
          while (supply) {
            visited.assign(local.size(), false);
            int moved = augment(kind, supply);
            if (!moved)
              return false;
            supply -= moved;
          }
        }
        return true;
      };
      std::function<bool()> complete = [&]() {
        if (steps-- <= 0) {
          stopped = true;
          return false;
        }
        int demand = 0;
        llvm::SmallVector<bool> useful(local.size(), false);
        for (auto [kind, k] : llvm::enumerate(kinds)) {
          int left = k.members.size() - taken[kind];
          if (!left)
            continue;
          demand += left * k.count;
          bool fits = false;
          for (unsigned position : k.reach)
            if (open[position]) {
              useful[position] = true;
              fits |= room[position] >= k.count;
            }
          if (!fits)
            return false;
        }
        if (!demand)
          return true;
        if (failed.count({open, taken}))
          return false;
        int capacity = 0;
        for (unsigned position : llvm::seq<unsigned>(0, local.size()))
          if (useful[position])
            capacity += room[position];
        int slack = capacity - demand;
        if (slack < 0 || !covers()) {
          failed.insert({open, taken});
          return false;
        }

        // A kind whose only open position is this one must all go there.
        auto onlyAt = [&](const Kind &k, unsigned position) {
          return llvm::all_of(k.reach, [&](unsigned other) {
            return other == position || !open[other];
          });
        };
        std::optional<unsigned> best;
        int bestChoices = 0;
        for (unsigned position : llvm::seq<unsigned>(0, local.size())) {
          if (!useful[position])
            continue;
          int choices = 0;
          for (auto [kind, k] : llvm::enumerate(kinds))
            if (taken[kind] < static_cast<int>(k.members.size()) &&
                llvm::is_contained(k.reach, position) && !onlyAt(k, position))
              choices += k.members.size() - taken[kind];
          if (!best || choices < bestChoices) {
            best = position;
            bestChoices = choices;
          }
        }
        unsigned position = *best;
        int fill = room[position];
        llvm::SmallVector<unsigned> free;
        std::vector<int> held = taken;
        for (auto [kind, k] : llvm::enumerate(kinds)) {
          int left = k.members.size() - taken[kind];
          if (!left || !llvm::is_contained(k.reach, position))
            continue;
          if (onlyAt(k, position)) {
            fill -= left * k.count;
            taken[kind] += left;
          } else {
            free.push_back(kind);
          }
        }
        int maxWaste = std::min(slack, room[position]);
        open[position] = 0;
        std::function<bool(size_t, int)> choose = [&](size_t next, int left) {
          if (steps-- <= 0) {
            stopped = true;
            return false;
          }
          if (left < 0)
            return false;
          if (next == free.size())
            return left <= maxWaste && complete();
          unsigned kind = free[next];
          const Kind &k = kinds[kind];
          int most =
              std::min<int>(k.members.size() - taken[kind], left / k.count);
          for (int n = most; n >= 0; --n) {
            taken[kind] += n;
            bool done = choose(next + 1, left - n * k.count);
            taken[kind] -= n;
            if (done) {
              where[kind].append(n, local[position]);
              return true;
            }
            if (stopped)
              return false;
          }
          return false;
        };
        bool done = choose(0, fill);
        open[position] = 1;
        if (done) {
          for (auto [kind, k] : llvm::enumerate(kinds))
            where[kind].append(taken[kind] - held[kind], local[position]);
        }
        taken = std::move(held);
        if (!done && !stopped)
          failed.insert({open, taken});
        return done;
      };
      if (!complete()) {
        if (stopped) {
          anyStopped = true;
          continue;
        }
        for (auto [group, own] : llvm::enumerate(reach))
          if (!chosen[group] && !own.empty() && root(own.front()) == top)
            blocked.push_back(group);
        return Packing::None;
      }
      for (auto [kind, k] : llvm::enumerate(kinds))
        for (auto [group, position] : llvm::zip(k.members, where[kind]))
          chosen[group] = *llvm::find_if(eligible[group], [&](unsigned tile) {
            return tiles[tile].at == positions[position];
          });
    }
    return anyStopped ? Packing::Stopped : Packing::Found;
  }

  const AIETargetModel &target;
  llvm::SmallVector<Tile> tiles;
  llvm::DenseMap<TileID, int> used;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIELOCKPLAN_H
