//===- AIELockPlan.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIELockPlan.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#include <functional>
#include <set>

using namespace xilinx::AIE;

LockPlan::Assignment LockPlan::assign(llvm::ArrayRef<Group> groups, Memo *memo,
                                      bool callerOrder) {
  Assignment result;
  llvm::DenseMap<TileID, int> before = used;
  llvm::SmallVector<Placed> misses;
  // Each group's candidates every user reaches, found once on first need.
  llvm::SmallVector<llvm::SmallVector<unsigned>> eligible;
  auto findEligible = [&] {
    if (!eligible.empty() || groups.empty())
      return;
    for (const Group &group : groups) {
      llvm::SmallVector<unsigned> &tilesOf = eligible.emplace_back();
      for (unsigned tile : group.candidates)
        if (!llvm::is_contained(tilesOf, tile) &&
            llvm::all_of(group.users,
                         [&](const User &user) { return reaches(user, tile); }))
          tilesOf.push_back(tile);
    }
  };
  auto record = [&](size_t index, const Placed &at) {
    result.tiles[index] = at.tile;
    misses[index] = at;
    if (!at.tile && !result.failed) {
      result.failed = index;
      result.failure = at;
    }
  };
  auto firstFit = [&](bool canonical) {
    used = before;
    result = {};
    result.tiles.assign(groups.size(), std::nullopt);
    misses.assign(groups.size(), {});
    if (!canonical) {
      for (auto [index, group] : llvm::enumerate(groups))
        record(index, place(group));
      return;
    }
    findEligible();
    for (auto &[index, at] : canonicalFirstFit(groups, eligible))
      record(index, at);
  };
  int count = groups.empty() ? 0 : groups.front().count;
  bool mixed =
      llvm::any_of(groups, [&](const Group &g) { return g.count != count; });
  firstFit(/*canonical=*/!callerOrder);
  if (!result.failed)
    return result;
  if (callerOrder && mixed) {
    Assignment own = result;
    llvm::SmallVector<Placed> ownMisses = misses;
    llvm::DenseMap<TileID, int> ownUsed = used;
    firstFit(/*canonical=*/true);
    if (!result.failed)
      return result;
    result = std::move(own);
    misses = std::move(ownMisses);
    used = std::move(ownUsed);
  }
  findEligible();

  if (mixed) {
    llvm::SmallVector<std::optional<unsigned>> packed;
    llvm::SmallVector<unsigned> blocked;
    // What the packing reads: each group's count and the positions of
    // its eligible tiles in order, with the locks taken there before.
    std::vector<int64_t> key;
    size_t hash = 0;
    if (memo) {
      size_t size = 0;
      for (const llvm::SmallVector<unsigned> &tilesOf : eligible)
        size += 2 + 3 * tilesOf.size();
      key.reserve(size);
      for (auto [group, tilesOf] : llvm::zip(groups, eligible)) {
        llvm::append_values(key, group.count, tilesOf.size());
        for (unsigned tile : tilesOf) {
          if (std::optional<TileID> at = tiles[tile].at)
            llvm::append_values(key, at->col, at->row, before.lookup(*at));
          else
            llvm::append_values(key, -1, -1, -1);
        }
      }
      hash = llvm::hash_combine_range(key.begin(), key.end());
    }
    const Memo::Outcome *recorded = memo ? memo->lookup(key, hash) : nullptr;
    Packing packing;
    if (recorded) {
      packing = recorded->packing;
      blocked = recorded->blocked;
      for (auto [tilesOf, offset] : llvm::zip(eligible, recorded->offsets))
        packed.push_back(tilesOf[offset]);
#ifdef EXPENSIVE_CHECKS
      llvm::SmallVector<std::optional<unsigned>> freshPacked;
      llvm::SmallVector<unsigned> freshBlocked;
      assert(pack(groups, eligible, before, freshPacked, freshBlocked) ==
                 packing &&
             freshBlocked == blocked &&
             (packing != Packing::Found || freshPacked == packed) &&
             "a recorded packing differs from a fresh one");
#endif
    } else {
      packing = pack(groups, eligible, before, packed, blocked);
      if (memo) {
        Memo::Outcome outcome{packing, blocked, {}};
        if (packing == Packing::Found)
          for (auto [tilesOf, tile] : llvm::zip(eligible, packed))
            outcome.offsets.push_back(llvm::find(tilesOf, *tile) -
                                      tilesOf.begin());
        memo->record(std::move(key), hash, std::move(outcome));
      }
    }
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

llvm::SmallVector<std::pair<size_t, LockPlan::Placed>>
LockPlan::canonicalFirstFit(
    llvm::ArrayRef<Group> groups,
    llvm::ArrayRef<llvm::SmallVector<unsigned>> eligible) {
  auto rank = [&](unsigned tile) {
    std::optional<TileID> at = tiles[tile].at;
    return at ? std::make_tuple(1, at->col, at->row) : std::make_tuple(0, 0, 0);
  };
  llvm::SmallVector<llvm::SmallVector<unsigned>> sorted;
  llvm::SmallVector<llvm::SmallVector<std::tuple<int, int, int>>> keys;
  for (const llvm::SmallVector<unsigned> &tilesOf : eligible) {
    llvm::SmallVector<unsigned> &order = sorted.emplace_back(tilesOf);
    llvm::stable_sort(
        order, [&](unsigned a, unsigned b) { return rank(a) < rank(b); });
    keys.push_back(llvm::map_to_vector(order, rank));
  }
  llvm::SmallVector<size_t> order =
      llvm::to_vector(llvm::seq<size_t>(0, groups.size()));
  llvm::stable_sort(order, [&](size_t a, size_t b) {
    if (groups[a].count != groups[b].count)
      return groups[a].count > groups[b].count;
    return keys[a] < keys[b];
  });
  llvm::SmallVector<std::pair<size_t, Placed>> placed;
  for (size_t index : order) {
    Placed at;
    at.reachable = !sorted[index].empty();
    for (unsigned tile : sorted[index]) {
      if (reserve(tile, groups[index].count)) {
        at.tile = tile;
        break;
      }
      at.full = tile;
    }
    if (at.tile)
      at.full.reset();
    placed.push_back({index, at});
  }
  return placed;
}

LockPlan::Packing
LockPlan::pack(llvm::ArrayRef<Group> groups,
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
      if (!tiles[tile].at)
        break;
      positions.push_back(*tiles[tile].at);
    }
  }
  // The search visits positions and kinds in an order of their own, so
  // whoever lists the groups (allocation or the SA) gets the same steps.
  llvm::sort(positions);
  positions.erase(llvm::unique(positions), positions.end());
  for (auto [group, tilesOf] : llvm::enumerate(eligible)) {
    for (unsigned tile : tilesOf) {
      if (!tiles[tile].at) {
        chosen[group] = tile;
        break;
      }
      unsigned position =
          llvm::find(positions, *tiles[tile].at) - positions.begin();
      if (!llvm::is_contained(reach[group], position))
        reach[group].push_back(position);
    }
    llvm::sort(reach[group]);
  }

  llvm::IntEqClasses component(positions.size());
  for (auto [group, own] : llvm::enumerate(reach))
    if (!chosen[group])
      for (unsigned position : own)
        component.join(position, own.front());
  component.compress();

  bool anyStopped = false;
  llvm::SmallVector<bool> seen(component.getNumClasses(), false);
  for (unsigned first : llvm::seq<unsigned>(0, positions.size())) {
    unsigned top = component[first];
    if (seen[top])
      continue;
    seen[top] = true;
    int64_t steps = kPackBudget;
    bool stopped = false;
    llvm::SmallVector<unsigned> local;
    for (unsigned position : llvm::seq<unsigned>(0, positions.size()))
      if (component[position] == top)
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
      if (chosen[group] || own.empty() || component[own.front()] != top)
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
      return a.count != b.count ? a.count > b.count : a.reach < b.reach;
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
    std::function<int(unsigned, int)> augment = [&](unsigned kind, int amount) {
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
      // What an open position cannot fill with the groups left that reach
      // it, those of count `skip` aside, is wasted wherever the others go.
      auto unfilled = [&](unsigned position, int skip) {
        std::vector<char> sums(room[position] + 1, 0);
        sums[0] = 1;
        for (auto [kind, k] : llvm::enumerate(kinds)) {
          if (k.count == skip || !llvm::is_contained(k.reach, position))
            continue;
          int copies = std::min<int>(k.members.size() - taken[kind],
                                     room[position] / k.count);
          for (int copy = 0; copy < copies; ++copy)
            for (int sum = room[position]; sum >= k.count; --sum)
              sums[sum] |= sums[sum - k.count];
        }
        int most = room[position];
        while (!sums[most])
          --most;
        return room[position] - most;
      };
      std::vector<int> least(local.size(), 0);
      int wasted = 0;
      for (unsigned position : llvm::seq<unsigned>(0, local.size()))
        if (useful[position]) {
          least[position] = unfilled(position, 0);
          wasted += least[position];
        }
      // Groups of one count fill at most as many positions as are left of
      // them, so only that many positions may do better with them.
      int bound = wasted;
      for (auto kind = kinds.begin(); kind != kinds.end();) {
        int count = kind->count, copies = 0;
        for (; kind != kinds.end() && kind->count == count; ++kind)
          copies += kind->members.size() - taken[kind - kinds.begin()];
        int without = 0;
        llvm::SmallVector<int> gains;
        for (unsigned position : llvm::seq<unsigned>(0, local.size())) {
          if (!useful[position])
            continue;
          int lost = unfilled(position, count);
          without += lost;
          if (lost > least[position])
            gains.push_back(lost - least[position]);
        }
        if (copies >= static_cast<int>(gains.size()))
          continue;
        llvm::sort(gains, std::greater<int>());
        for (int gain : llvm::ArrayRef(gains).take_front(copies))
          without -= gain;
        bound = std::max(bound, without);
      }
      if (slack < bound || !covers()) {
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
      int maxWaste = std::min(slack - wasted + least[position], room[position]);
      // The fills the free kinds from each on can still make.
      std::vector<std::vector<char>> later(
          free.size() + 1, std::vector<char>(room[position] + 1, 0));
      later.back()[0] = 1;
      for (size_t next = free.size(); next-- > 0;) {
        const Kind &k = kinds[free[next]];
        later[next] = later[next + 1];
        int copies = std::min<int>(k.members.size() - taken[free[next]],
                                   room[position] / k.count);
        for (int copy = 0; copy < copies; ++copy)
          for (int sum = room[position]; sum >= k.count; --sum)
            later[next][sum] |= later[next][sum - k.count];
      }
      open[position] = 0;
      std::function<bool(size_t, int)> choose = [&](size_t next, int left) {
        if (steps-- <= 0) {
          stopped = true;
          return false;
        }
        if (left < 0 || llvm::none_of(
                            llvm::seq(std::max(0, left - maxWaste), left + 1),
                            [&](int fill) { return later[next][fill]; }))
          return false;
        if (next == free.size())
          return complete();
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
        if (!chosen[group] && !own.empty() && component[own.front()] == top)
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
