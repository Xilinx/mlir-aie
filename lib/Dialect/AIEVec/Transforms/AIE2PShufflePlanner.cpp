//===- AIE2PShufflePlanner.cpp - Static shuffles on AIE2P -------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Plans static shuffles on 512-bit registers out of vshuffle modes and splats.
// Two bounded searches run: in rounds, where all registers go through the same
// mode (transposes, interleaves, reversals), and per result register, a few
// modes deep (repetitions, broadcasts, gathers). A plan that needs fewer modes
// can win at a higher shuffle count, since each mode costs a scalar register.
// A plan is used only if it beats LLVM's instruction count and running it on
// lane labels reproduces the requested result.
//===----------------------------------------------------------------------===//

#include "AIE2PShufflePlanner.h"

#include "aie/Dialect/AIEVec/IR/AIEVecShuffleModes.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "aie2p-shuffle-planner"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

namespace {

using Lanes = SmallVector<int32_t, 64>;
using Node = ShufflePlan::Node;

constexpr unsigned kInf = std::numeric_limits<unsigned>::max() / 4;
// Depth of the per-register search, and of the round search with all modes
// and with one pair of modes.
constexpr unsigned kMaxRegisterDepth = 3;
constexpr unsigned kMaxRounds = 3;
constexpr unsigned kMaxRoundsOnePair = 4;
// Bounds on the work of one search.
constexpr unsigned kMaxRoundStates = 50000;
constexpr unsigned kMaxSolveSteps = 200000;
// Registers in the round search.
constexpr unsigned kMaxRoundRegisters = 8;

std::string bytesOf(ArrayRef<int32_t> v) {
  return std::string(reinterpret_cast<const char *>(v.data()),
                     v.size() * sizeof(int32_t));
}

// A mode seen as a gather of whole lanes of `elemBits` bits: result lane i
// takes lane src[i] of lhs:rhs. Modes that would split a lane are left out.
struct LaneMode {
  ShuffleMode mode;
  bool twoInput;
  SmallVector<uint8_t, 64> src;
};

SmallVector<LaneMode> getLaneModes(unsigned elemBits) {
  SmallVector<LaneMode> modes;
  unsigned eb = elemBits / 8, lanes = 512 / elemBits;
  for (uint32_t m = 0; m <= getMaxEnumValForShuffleMode(); ++m) {
    auto mode = symbolizeShuffleMode(m);
    if (!mode)
      continue;
    std::array<uint8_t, 64> perm = getShuffleModeBytePermutation(*mode);
    LaneMode lm{*mode, getShuffleModeShape(*mode).isTwoInput(), {}};
    bool wholeLanes = true;
    for (unsigned i = 0; i < lanes && wholeLanes; ++i) {
      unsigned first = perm[i * eb];
      wholeLanes = first % eb == 0;
      for (unsigned b = 1; b < eb && wholeLanes; ++b)
        wholeLanes = perm[i * eb + b] == first + b;
      lm.src.push_back(first / eb);
    }
    if (!wholeLanes)
      continue;
    // A copy of one operand does nothing.
    bool copiesLhs = true, copiesRhs = true;
    for (unsigned i = 0; i < lanes; ++i) {
      copiesLhs &= lm.src[i] == i;
      copiesRhs &= lm.src[i] == lanes + i;
    }
    if (copiesLhs || copiesRhs)
      continue;
    modes.push_back(std::move(lm));
  }
  // Searches keep the first of equally good choices. Trying narrow modes
  // first makes related registers tend to use the same few modes.
  llvm::stable_sort(modes, [](const LaneMode &a, const LaneMode &b) {
    return getShuffleModeShape(a.mode).elemBits <
           getShuffleModeShape(b.mode).elemBits;
  });
  return modes;
}

// The registers both searches work on, interned by contents. A lane holds a
// label, or -1 when its contents are unknown.
class RegisterSearch {
public:
  // With `only`, the searches use those modes alone.
  RegisterSearch(const ShuffleGather &gather, ArrayRef<ShuffleMode> only = {},
                 unsigned maxRounds = kMaxRounds)
      : g(gather), lanes(512 / gather.elemBits),
        modes(getLaneModes(gather.elemBits)), maxRounds(maxRounds) {
    if (!only.empty())
      llvm::erase_if(modes, [&](const LaneMode &lm) {
        return !llvm::is_contained(only, lm.mode);
      });
    for (unsigned k = 0; k < g.sourceLanes.size(); ++k) {
      if (g.sourceIsSplat[k]) {
        sources.push_back(splat(g.sourceOffset(k)));
        continue;
      }
      int64_t n = g.sourceLanes[k];
      for (int64_t c = 0; c * lanes < std::max<int64_t>(n, 1); ++c) {
        Lanes l(lanes, -1);
        for (unsigned j = 0; j < lanes && c * lanes + j < n; ++j)
          l[j] = g.sourceOffset(k) + c * lanes + j;
        Node node{Node::Source};
        node.source = k;
        node.chunk = c;
        sources.push_back(intern(l, node));
      }
    }
    int64_t m = g.target.size();
    for (int64_t r = 0; r * lanes < m; ++r) {
      Lanes t(lanes, -1);
      for (unsigned j = 0; j < lanes && r * lanes + j < m; ++j)
        t[j] = g.target[r * lanes + j];
      targets.push_back(t);
    }
    // Modes paired as the lo and hi halves of one result.
    for (unsigned i = 0; i < modes.size(); ++i) {
      ShuffleModeShape a = getShuffleModeShape(modes[i].mode);
      if (!modes[i].twoInput || a.hi)
        continue;
      for (unsigned j = 0; j < modes.size(); ++j) {
        ShuffleModeShape b = getShuffleModeShape(modes[j].mode);
        if (b.hi && b.elemBits == a.elemBits && b.rows == a.rows &&
            b.cols == a.cols)
          loHi.push_back({i, j});
      }
    }
  }

  std::optional<ShufflePlan> searchRounds();
  std::optional<ShufflePlan> searchPerRegister();
  // The lo and hi modes that make up the two halves of one result.
  SmallVector<std::pair<ShuffleMode, ShuffleMode>> modePairs() const {
    SmallVector<std::pair<ShuffleMode, ShuffleMode>> pairs;
    for (auto [lo, hi] : loHi)
      pairs.push_back({modes[lo].mode, modes[hi].mode});
    return pairs;
  }
  // Result registers that no source register already holds; each costs at
  // least one instruction.
  unsigned lowerBound() const;

private:
  struct Register {
    Lanes lanes;
    Node node;
  };

  bool matches(ArrayRef<int32_t> have, ArrayRef<int32_t> want) const {
    for (unsigned i = 0; i < lanes; ++i)
      if (want[i] >= 0 && have[i] != want[i])
        return false;
    return true;
  }
  static bool isAny(ArrayRef<int32_t> spec) {
    return llvm::all_of(spec, [](int32_t l) { return l < 0; });
  }
  // The single label of `spec`, if it has exactly one.
  static std::optional<int32_t> singleLabel(ArrayRef<int32_t> spec) {
    std::optional<int32_t> label;
    for (int32_t l : spec) {
      if (l < 0)
        continue;
      if (label && *label != l)
        return std::nullopt;
      label = l;
    }
    return label;
  }
  bool isSplatSourceLabel(int32_t label) const {
    for (unsigned k = 0; k < g.sourceLanes.size(); ++k)
      if (g.sourceIsSplat[k] && g.sourceOffset(k) == label)
        return true;
    return false;
  }
  bool canSplat(int32_t label) const {
    return g.canSplat || isSplatSourceLabel(label);
  }

  unsigned intern(ArrayRef<int32_t> l, const Node &node) {
    auto [it, inserted] = index.try_emplace(bytesOf(l), registers.size());
    if (inserted)
      registers.push_back({Lanes(l.begin(), l.end()), node});
    return it->second;
  }
  unsigned splat(int32_t label) {
    Node node{Node::Splat};
    node.label = label;
    return intern(Lanes(lanes, label), node);
  }
  Lanes applyLanes(unsigned modeIdx, ArrayRef<int32_t> lhs,
                   ArrayRef<int32_t> rhs) const {
    const LaneMode &lm = modes[modeIdx];
    Lanes out(lanes);
    for (unsigned i = 0; i < lanes; ++i) {
      unsigned s = lm.src[i];
      out[i] = s < lanes ? lhs[s] : rhs[s - lanes];
    }
    return out;
  }
  unsigned apply(unsigned modeIdx, unsigned lhs, unsigned rhs) {
    const LaneMode &lm = modes[modeIdx];
    Lanes out = applyLanes(modeIdx, registers[lhs].lanes, registers[rhs].lanes);
    Node node{Node::Shuffle};
    node.mode = lm.mode;
    node.lhs = lhs;
    node.rhs = lm.twoInput ? rhs : lhs;
    return intern(out, node);
  }
  // The lanes of lhs and rhs that `spec` needs through mode `modeIdx`, or
  // false if two result lanes need one source lane to hold different labels.
  bool invert(unsigned modeIdx, ArrayRef<int32_t> spec, Lanes &lhs,
              Lanes &rhs) const {
    const LaneMode &lm = modes[modeIdx];
    lhs.assign(lanes, -1);
    rhs.assign(lanes, -1);
    for (unsigned i = 0; i < lanes; ++i) {
      if (spec[i] < 0)
        continue;
      unsigned s = lm.src[i];
      int32_t &slot = s < lanes ? lhs[s] : rhs[s - lanes];
      if (slot >= 0 && slot != spec[i])
        return false;
      slot = spec[i];
    }
    return true;
  }

  ShufflePlan extract(ArrayRef<int> results) const;

  // Per-register search. Among choices of equal cost, the one whose new
  // registers hold more distinct labels wins: such registers are more likely
  // to serve later result registers too. Then the one that needs fewer modes
  // the plan does not use yet, as each mode takes a scalar register.
  struct Choice {
    unsigned cost = kInf;
    unsigned info = 0;
    SmallVector<ShuffleMode, 4> newModes;
    enum { None, Existing, Splat, Mode } kind = None;
    unsigned reg = 0;
    int32_t label = -1;
    unsigned modeIdx = 0;
    // The depth the operands were searched with.
    unsigned sub = 0;
    // Merged: lhs and rhs are one register meeting both specs.
    bool merged = false;
    Lanes lhs, rhs;
    // The lanes the choice produces.
    Lanes result;
    bool isBetterThan(const Choice &other) const {
      return std::make_tuple(cost, -static_cast<int>(info), newModes.size()) <
             std::make_tuple(other.cost, -static_cast<int>(other.info),
                             other.newModes.size());
    }
  };
  const Choice &solve(ArrayRef<int32_t> spec, unsigned depth);
  unsigned build(ArrayRef<int32_t> spec, unsigned depth);

  const ShuffleGather &g;
  unsigned lanes;
  SmallVector<LaneMode> modes;
  unsigned maxRounds;
  SmallVector<std::pair<unsigned, unsigned>> loHi;
  std::vector<Register> registers;
  std::unordered_map<std::string, unsigned> index;
  SmallVector<unsigned> sources;
  SmallVector<Lanes, 4> targets;
  // Registers the per-register search has built so far, and their modes.
  SmallVector<unsigned> built;
  SmallVector<ShuffleMode> usedModes;
  std::unordered_map<std::string, Choice> memo;
  unsigned steps = 0;
};

unsigned RegisterSearch::lowerBound() const {
  unsigned n = 0;
  for (const Lanes &t : targets)
    if (!isAny(t) && llvm::none_of(sources, [&](unsigned s) {
          return registers[s].node.kind == Node::Source &&
                 matches(registers[s].lanes, t);
        }))
      ++n;
  return n;
}

// Collects the registers the results depend on, operands first, and counts
// the instructions among them.
ShufflePlan RegisterSearch::extract(ArrayRef<int> results) const {
  ShufflePlan plan;
  DenseMap<unsigned, int> newIndex;
  std::function<int(unsigned)> visit = [&](unsigned r) -> int {
    if (auto it = newIndex.find(r); it != newIndex.end())
      return it->second;
    Node node = registers[r].node;
    if (node.kind == Node::Shuffle) {
      node.lhs = visit(node.lhs);
      node.rhs = visit(node.rhs);
    }
    if (node.kind != Node::Source)
      ++plan.cost;
    plan.nodes.push_back(node);
    return newIndex[r] = plan.nodes.size() - 1;
  };
  for (int r : results)
    plan.results.push_back(r < 0 ? -1 : visit(r));
  return plan;
}

std::optional<ShufflePlan> RegisterSearch::searchRounds() {
  unsigned n = llvm::PowerOf2Ceil(std::max(sources.size(), targets.size()));
  if (n > kMaxRoundRegisters)
    return std::nullopt;
  unsigned bits = llvm::Log2_32(n);

  // A round, as a function from the old registers to the new ones.
  using State = SmallVector<unsigned, 8>;
  SmallVector<std::function<State(const State &)>> rounds;
  for (unsigned mi = 0; mi < modes.size(); ++mi) {
    if (modes[mi].twoInput)
      continue;
    rounds.push_back([this, mi](const State &s) {
      State t;
      for (unsigned r : s)
        t.push_back(apply(mi, r, r));
      return t;
    });
  }
  for (auto [lo, hi] : loHi) {
    // Each register with itself, all lo or all hi.
    for (unsigned mi : {lo, hi})
      rounds.push_back([this, mi](const State &s) {
        State t;
        for (unsigned r : s)
          t.push_back(apply(mi, r, r));
        return t;
      });
    for (unsigned b = 0; b < bits; ++b) {
      unsigned bit = 1u << b;
      // Registers i and i | bit as lhs:rhs; lo to i, hi to i | bit.
      rounds.push_back([this, lo, hi, bit](const State &s) {
        State t(s.size());
        for (unsigned i = 0; i < s.size(); ++i)
          if (!(i & bit)) {
            t[i] = apply(lo, s[i], s[i | bit]);
            t[i | bit] = apply(hi, s[i], s[i | bit]);
          }
        return t;
      });
      // Each register with itself; lo where the bit is clear, else hi.
      rounds.push_back([this, lo, hi, bit](const State &s) {
        State t;
        for (unsigned i = 0; i < s.size(); ++i)
          t.push_back(apply(i & bit ? hi : lo, s[i], s[i]));
        return t;
      });
    }
  }

  // The register of the state that holds each result, if all are there.
  auto resultsOf = [&](const State &s) -> std::optional<SmallVector<int>> {
    SmallVector<int> results;
    for (const Lanes &t : targets) {
      if (isAny(t)) {
        results.push_back(-1);
        continue;
      }
      auto it = llvm::find_if(
          s, [&](unsigned r) { return matches(registers[r].lanes, t); });
      if (it == s.end())
        return std::nullopt;
      results.push_back(*it);
    }
    return results;
  };

  State start;
  for (unsigned i = 0; i < n; ++i)
    start.push_back(sources[i % sources.size()]);
  std::optional<ShufflePlan> best;
  // Fewer distinct modes break ties: each takes a scalar register.
  auto numModes = [](const ShufflePlan &p) {
    SmallVector<ShuffleMode> used;
    for (const Node &node : p.nodes)
      if (node.kind == Node::Shuffle && !llvm::is_contained(used, node.mode))
        used.push_back(node.mode);
    return used.size();
  };
  auto consider = [&](const State &s) {
    if (auto results = resultsOf(s)) {
      ShufflePlan plan = extract(*results);
      if (!best || plan.cost < best->cost ||
          (plan.cost == best->cost && numModes(plan) < numModes(*best)))
        best = std::move(plan);
    }
  };
  consider(start);
  if (best)
    return best;

  std::unordered_set<std::string> seen{
      bytesOf(SmallVector<int32_t>(start.begin(), start.end()))};
  std::vector<State> frontier{start};
  unsigned states = 0;
  for (unsigned depth = 1; depth <= maxRounds && !best; ++depth) {
    std::vector<State> next;
    for (const State &s : frontier) {
      for (auto &round : rounds) {
        if (++states > kMaxRoundStates)
          return best;
        State t = round(s);
        if (!seen.insert(bytesOf(SmallVector<int32_t>(t.begin(), t.end())))
                 .second)
          continue;
        consider(t);
        if (depth < maxRounds)
          next.push_back(std::move(t));
      }
    }
    frontier = std::move(next);
  }
  return best;
}

static unsigned countLabels(ArrayRef<int32_t> lanes) {
  SmallVector<int32_t, 64> labels;
  for (int32_t l : lanes)
    if (l >= 0 && !llvm::is_contained(labels, l))
      labels.push_back(l);
  return labels.size();
}

// The cheapest way to build a register meeting `spec` from what is built
// already, splats and up to `depth` modes in a chain.
const RegisterSearch::Choice &RegisterSearch::solve(ArrayRef<int32_t> spec,
                                                    unsigned depth) {
  std::string key = bytesOf(spec);
  key.push_back(static_cast<char>(depth));
  if (auto it = memo.find(key); it != memo.end())
    return it->second;

  Choice best;
  for (unsigned r : built)
    if (matches(registers[r].lanes, spec)) {
      best.cost = 0;
      best.kind = Choice::Existing;
      best.reg = r;
      best.result = registers[r].lanes;
      break;
    }
  if (best.cost > 0)
    if (auto label = singleLabel(spec); label && canSplat(*label)) {
      best.cost = 1;
      best.info = 1;
      best.kind = Choice::Splat;
      best.label = *label;
      best.result.assign(lanes, *label);
    }
  if (depth > 0 && best.cost > 0) {
    Lanes lhs, rhs;
    for (unsigned mi = 0; mi < modes.size(); ++mi) {
      if (++steps > kMaxSolveSteps)
        break;
      if (!invert(mi, spec, lhs, rhs))
        continue;
      // An operand costs at least one instruction per mode in its chain, so
      // only chains shorter than the best cost can match it.
      unsigned sub = std::min(depth - 1, best.cost - 1);
      // Copies, since solve may grow the memo and move what it returned.
      auto consider = [&](const Choice &l, const Choice *r) {
        if (l.cost >= kInf || (r && r->cost >= kInf))
          return;
        Choice c;
        c.cost = 1 + l.cost + (r ? r->cost : 0);
        c.kind = Choice::Mode;
        c.modeIdx = mi;
        c.sub = sub;
        c.merged = !r;
        c.lhs = lhs;
        c.rhs = rhs;
        c.result = applyLanes(mi, l.result, r ? r->result : l.result);
        c.info = l.info + (r ? r->info : 0) + countLabels(c.result);
        c.newModes = l.newModes;
        if (r)
          for (ShuffleMode m : r->newModes)
            if (!llvm::is_contained(c.newModes, m))
              c.newModes.push_back(m);
        if (!llvm::is_contained(usedModes, modes[mi].mode) &&
            !llvm::is_contained(c.newModes, modes[mi].mode))
          c.newModes.push_back(modes[mi].mode);
        if (c.isBetterThan(best))
          best = std::move(c);
      };
      bool lhsAny = isAny(lhs), rhsAny = isAny(rhs);
      if (!modes[mi].twoInput || rhsAny) {
        Choice l = solve(lhs, sub);
        rhs = lhs;
        consider(l, nullptr);
        continue;
      }
      if (lhsAny) {
        Choice r = solve(rhs, sub);
        lhs = rhs;
        consider(r, nullptr);
        continue;
      }
      // One register for both operands, when their lanes agree.
      Lanes both = lhs;
      bool agree = true;
      for (unsigned i = 0; i < lanes && agree; ++i) {
        if (rhs[i] < 0)
          continue;
        agree = both[i] < 0 || both[i] == rhs[i];
        both[i] = rhs[i];
      }
      if (agree) {
        Lanes l = lhs, r = rhs;
        lhs = rhs = both;
        Choice m = solve(both, sub);
        consider(m, nullptr);
        lhs = l;
        rhs = r;
      }
      Choice l = solve(lhs, sub);
      if (1 + l.cost <= best.cost) {
        Choice r = solve(rhs, sub);
        consider(l, &r);
      }
    }
  }
  return memo[key] = std::move(best);
}

// Builds the register `solve` chose for `spec` and adds it to `built`.
unsigned RegisterSearch::build(ArrayRef<int32_t> spec, unsigned depth) {
  std::string key = bytesOf(spec);
  key.push_back(static_cast<char>(depth));
  Choice choice = memo.at(key);
  unsigned r;
  switch (choice.kind) {
  case Choice::Existing:
    return choice.reg;
  case Choice::Splat:
    r = splat(choice.label);
    break;
  case Choice::Mode: {
    unsigned lhs = build(choice.lhs, choice.sub);
    unsigned rhs = choice.merged ? lhs : build(choice.rhs, choice.sub);
    r = apply(choice.modeIdx, lhs, rhs);
    if (!llvm::is_contained(usedModes, modes[choice.modeIdx].mode))
      usedModes.push_back(modes[choice.modeIdx].mode);
    break;
  }
  case Choice::None:
    llvm_unreachable("building a register without a plan");
  }
  if (!llvm::is_contained(built, r))
    built.push_back(r);
  return r;
}

std::optional<ShufflePlan> RegisterSearch::searchPerRegister() {
  for (unsigned s : sources)
    if (registers[s].node.kind == Node::Source)
      built.push_back(s);
  SmallVector<int> results;
  for (const Lanes &t : targets) {
    if (isAny(t)) {
      results.push_back(-1);
      continue;
    }
    // What is cheapest depends on what is built, so start afresh. Nothing
    // beats a single instruction; otherwise a deeper search may find a plan
    // as cheap whose steps serve later registers too.
    memo.clear();
    steps = 0;
    unsigned depth = 0;
    while (solve(t, depth).cost > 1 && depth < kMaxRegisterDepth)
      ++depth;
    if (solve(t, depth).cost >= kInf)
      return std::nullopt;
    results.push_back(build(t, depth));
  }
  return extract(results);
}

} // namespace

int32_t ShuffleGather::sourceOffset(unsigned k) const {
  int32_t offset = 0;
  for (unsigned i = 0; i < k; ++i)
    offset += sourceIsSplat[i] ? 1 : sourceLanes[i];
  return offset;
}

std::string ShuffleGather::key() const {
  SmallVector<int32_t> v{static_cast<int32_t>(elemBits), canSplat,
                         static_cast<int32_t>(sourceLanes.size())};
  for (unsigned k = 0; k < sourceLanes.size(); ++k) {
    v.push_back(sourceLanes[k]);
    v.push_back(sourceIsSplat[k]);
  }
  v.append(target.begin(), target.end());
  return bytesOf(v);
}

unsigned xilinx::aievec::estimateLLVMShuffleCost(const ShuffleGather &g) {
  unsigned w = g.elemBits, lanes = 512 / w;
  int64_t m = g.target.size();
  int64_t half = std::min<int64_t>(256 / w, m);
  // The source and lane behind a label, for sources that are not splats.
  auto laneOf =
      [&](int32_t label) -> std::optional<std::pair<unsigned, int64_t>> {
    for (unsigned k = 0; k < g.sourceLanes.size(); ++k) {
      int64_t lane = label - g.sourceOffset(k);
      if (!g.sourceIsSplat[k] && lane >= 0 && lane < g.sourceLanes[k])
        return std::make_pair(k, lane);
    }
    return std::nullopt;
  };
  // Whether lanes [begin, end) are poison or consecutive lanes of one source
  // starting at a multiple of `align`. `start` receives the first source
  // lane, if any lane is not poison.
  auto isRun = [&](int64_t begin, int64_t end, int64_t align,
                   std::optional<int64_t> *start = nullptr) {
    std::optional<std::pair<unsigned, int64_t>> origin;
    for (int64_t i = begin; i < end; ++i) {
      if (g.target[i] < 0)
        continue;
      auto at = laneOf(g.target[i]);
      if (!at)
        return false;
      int64_t first = at->second - (i - begin);
      if (first < 0 || first % align != 0 ||
          (origin && *origin != std::make_pair(at->first, first)))
        return false;
      origin = {at->first, first};
    }
    if (start && origin)
      *start = origin->second;
    return true;
  };

  unsigned cost = 0;
  for (int64_t begin = 0; begin < m; begin += lanes) {
    int64_t end = std::min<int64_t>(begin + lanes, m);
    // A register equal to an earlier one is that register again.
    bool repeated = false;
    for (int64_t prev = 0; prev < begin && !repeated; prev += lanes)
      repeated = std::equal(g.target.begin() + begin, g.target.begin() + end,
                            g.target.begin() + prev);
    if (repeated)
      continue;
    SmallVector<int32_t> labels;
    unsigned used = 0;
    for (int64_t i = begin; i < end; ++i)
      if (g.target[i] >= 0) {
        ++used;
        if (!llvm::is_contained(labels, g.target[i]))
          labels.push_back(g.target[i]);
      }
    if (labels.empty())
      continue;
    // A splat of one lane.
    if (labels.size() == 1) {
      cost += 1;
      continue;
    }
    // Aligned 256-bit halves of the sources are subregisters. One that lands
    // in the other half of a 512-bit result takes a move.
    bool moves = true;
    unsigned crossings = 0;
    for (int64_t c = begin; c < end && moves; c += half) {
      std::optional<int64_t> start;
      moves = isRun(c, std::min(c + half, end), half, &start);
      if (moves && start && m >= lanes &&
          (c - begin) / half % 2 != *start / half % 2)
        ++crossings;
    }
    if (moves) {
      cost += crossings;
      continue;
    }
    // Consecutive lanes of one source: a shift.
    if (isRun(begin, end, 1)) {
      cost += 1;
      continue;
    }
    // Each lane from its own position in one of the sources: a select, or a
    // few lanes inserted into one source.
    unsigned misplaced = 0;
    SmallVector<unsigned> from;
    for (int64_t i = begin; i < end; ++i) {
      if (g.target[i] < 0)
        continue;
      auto at = laneOf(g.target[i]);
      if (!at || at->second != i) {
        ++misplaced;
        continue;
      }
      if (!llvm::is_contained(from, at->first))
        from.push_back(at->first);
    }
    if (misplaced == 0 && from.size() <= 2) {
      cost += 1;
      continue;
    }
    // Otherwise each lane is extracted once and pushed in.
    cost += std::min<unsigned>(used + labels.size(),
                               from.size() <= 1 ? 2 * misplaced : ~0u);
  }
  return cost;
}

bool xilinx::aievec::simulateShufflePlan(const ShuffleGather &g,
                                         const ShufflePlan &plan) {
  unsigned eb = g.elemBits / 8, lanes = 512 / g.elemBits;
  // Byte b of lane label l is l * eb + b; unknown bytes are -1.
  SmallVector<std::array<int32_t, 64>> regs;
  for (const Node &node : plan.nodes) {
    std::array<int32_t, 64> bytes;
    bytes.fill(-1);
    switch (node.kind) {
    case Node::Source: {
      if (node.source >= g.sourceLanes.size() || g.sourceIsSplat[node.source])
        return false;
      for (unsigned j = 0; j < lanes; ++j) {
        int64_t lane = node.chunk * lanes + j;
        if (lane < g.sourceLanes[node.source])
          for (unsigned b = 0; b < eb; ++b)
            bytes[j * eb + b] = (g.sourceOffset(node.source) + lane) * eb + b;
      }
      break;
    }
    case Node::Splat: {
      bool fromSplat = false;
      for (unsigned k = 0; k < g.sourceLanes.size(); ++k)
        fromSplat |= g.sourceIsSplat[k] && g.sourceOffset(k) == node.label;
      if (node.label < 0 || node.label >= g.numLabels() ||
          (!fromSplat && !g.canSplat))
        return false;
      for (unsigned i = 0; i < 64; ++i)
        bytes[i] = node.label * eb + i % eb;
      break;
    }
    case Node::Shuffle: {
      if (node.lhs < 0 || node.rhs < 0 ||
          static_cast<size_t>(std::max(node.lhs, node.rhs)) >= regs.size())
        return false;
      std::array<uint8_t, 64> perm = getShuffleModeBytePermutation(node.mode);
      for (unsigned i = 0; i < 64; ++i)
        bytes[i] = perm[i] < 64 ? regs[node.lhs][perm[i]]
                                : regs[node.rhs][perm[i] - 64];
      break;
    }
    }
    regs.push_back(bytes);
  }
  int64_t m = g.target.size();
  if (static_cast<int64_t>(plan.results.size()) * lanes < m)
    return false;
  for (int64_t i = 0; i < m; ++i) {
    if (g.target[i] < 0)
      continue;
    int r = plan.results[i / lanes];
    if (r < 0 || static_cast<size_t>(r) >= regs.size())
      return false;
    for (unsigned b = 0; b < eb; ++b)
      if (regs[r][(i % lanes) * eb + b] !=
          g.target[i] * int32_t(eb) + int32_t(b))
        return false;
  }
  return true;
}

SmallVector<ShuffleMode> ShufflePlan::modes() const {
  SmallVector<ShuffleMode> used;
  for (const Node &node : nodes)
    if (node.kind == Node::Shuffle && !llvm::is_contained(used, node.mode))
      used.push_back(node.mode);
  return used;
}

const SmallVector<ShufflePlan, 2> &
AIE2PShufflePlanner::candidates(const ShuffleGather &g) {
  auto [it, inserted] = cache.try_emplace(g.key());
  if (!inserted)
    return it->second;
  SmallVector<ShufflePlan, 2> found;
  // The search keeps its own account of lane contents; a plan is kept only if
  // running it through the mode table agrees.
  auto keep = [&](std::optional<ShufflePlan> p, StringRef what) {
    LLVM_DEBUG(if (p) {
      llvm::dbgs() << what << ": " << p->cost << " instructions, modes";
      for (ShuffleMode m : p->modes())
        llvm::dbgs() << " " << stringifyShuffleMode(m);
      llvm::dbgs() << "\n";
    });
    if (p && simulateShufflePlan(g, *p))
      found.push_back(std::move(*p));
  };
  RegisterSearch search(g);
  keep(search.searchPerRegister(), "per-register search");
  if (found.empty() || found.front().cost > search.lowerBound())
    keep(RegisterSearch(g).searchRounds(), "round search");
  // Plans that need fewer modes, which another plan may already have put in
  // scalar registers.
  for (auto [lo, hi] : search.modePairs()) {
    SmallVector<ShuffleMode> pair{lo, hi};
    keep(RegisterSearch(g, pair, kMaxRoundsOnePair).searchRounds(),
         "round search with one pair");
    keep(RegisterSearch(g, pair).searchPerRegister(),
         "per-register search with one pair");
  }
  // Keep the plans no other plan beats in both cost and modes.
  SmallVector<unsigned> kept;
  for (unsigned i = 0; i < found.size(); ++i) {
    SmallVector<ShuffleMode> mi = found[i].modes();
    bool dominated = false;
    for (unsigned j = 0; j < found.size() && !dominated; ++j) {
      if (i == j)
        continue;
      SmallVector<ShuffleMode> mj = found[j].modes();
      bool fewerModes = llvm::all_of(
          mj, [&](ShuffleMode m) { return llvm::is_contained(mi, m); });
      bool sameModes = fewerModes && mj.size() == mi.size();
      bool sameCost = found[j].cost == found[i].cost;
      dominated = found[j].cost <= found[i].cost && fewerModes &&
                  (!sameModes || !sameCost || j < i);
    }
    if (!dominated)
      kept.push_back(i);
  }
  SmallVector<ShufflePlan, 2> &plans = it->second;
  for (unsigned i : kept)
    plans.push_back(std::move(found[i]));
  llvm::stable_sort(plans, [](const ShufflePlan &a, const ShufflePlan &b) {
    return a.cost < b.cost;
  });
  return plans;
}

//===----------------------------------------------------------------------===//
// From and to vector.shuffle
//===----------------------------------------------------------------------===//

namespace {

// Builds the gather of a request. With `fold`, a vector.shuffle operand used
// only by the request's user is looked through, so the two become one gather.
// `leaves` receives the value behind each source.
std::optional<ShuffleGather> gatherOf(const ShuffleRequest &req, bool fold,
                                      SmallVector<Value> &leaves) {
  VectorType resTy = req.resultType;
  Type elemTy = resTy.getElementType();
  if (resTy.getRank() != 1 || !elemTy.isIntOrFloat())
    return std::nullopt;
  unsigned w = elemTy.getIntOrFloatBitWidth();
  if (w != 8 && w != 16 && w != 32 && w != 64)
    return std::nullopt;
  int64_t m = resTy.getNumElements();
  // emitPlan concatenates the result registers in pairs.
  if (m * w > 512 && (m * w % 512 != 0 || !llvm::isPowerOf2_64(m * w / 512)))
    return std::nullopt;

  ShuffleGather g;
  g.elemBits = w;
  g.canSplat = elemTy.isBF16() || elemTy.isF32() ||
               elemTy.isSignlessInteger(8) || elemTy.isSignlessInteger(16) ||
               elemTy.isSignlessInteger(32);
  leaves.clear();

  // Source and lane behind each lane of `v`, or -1 for a poison lane.
  std::function<std::optional<SmallVector<std::pair<int, int64_t>>>(
      Value, Operation *, unsigned)>
      lanesOf = [&](Value v, Operation *user, unsigned depth)
      -> std::optional<SmallVector<std::pair<int, int64_t>>> {
    auto vTy = cast<VectorType>(v.getType());
    if (vTy.getRank() != 1)
      return std::nullopt;
    int64_t n = vTy.getNumElements();
    SmallVector<std::pair<int, int64_t>> lanes;
    if (v.getDefiningOp<ub::PoisonOp>()) {
      lanes.assign(n, {-1, 0});
      return lanes;
    }
    auto producer = v.getDefiningOp<vector::ShuffleOp>();
    if (fold && depth > 0 && producer &&
        llvm::all_of(producer->getUsers(),
                     [&](Operation *u) { return u == user; })) {
      auto a = lanesOf(producer.getV1(), producer, depth - 1);
      auto b = lanesOf(producer.getV2(), producer, depth - 1);
      if (!a || !b)
        return std::nullopt;
      int64_t n1 = a->size();
      for (int64_t idx : producer.getMask())
        lanes.push_back(idx < 0    ? std::make_pair(-1, int64_t(0))
                        : idx < n1 ? (*a)[idx]
                                   : (*b)[idx - n1]);
      return lanes;
    }
    if (n * w > 512 && n * w % 512 != 0)
      return std::nullopt;
    int k = llvm::find(leaves, v) - leaves.begin();
    if (k == static_cast<int>(leaves.size())) {
      leaves.push_back(v);
      auto bcast = v.getDefiningOp<vector::BroadcastOp>();
      g.sourceIsSplat.push_back(g.canSplat && bcast &&
                                !isa<VectorType>(bcast.getSourceType()));
      g.sourceLanes.push_back(n);
    }
    for (int64_t i = 0; i < n; ++i)
      lanes.push_back({k, i});
    return lanes;
  };

  auto a = lanesOf(req.v1, req.user, 3);
  auto b = lanesOf(req.v2, req.user, 3);
  if (!a || !b)
    return std::nullopt;
  int64_t n1 = a->size();
  for (int64_t idx : req.mask) {
    std::pair<int, int64_t> src = idx < 0
                                      ? std::make_pair(-1, int64_t(0))
                                      : (idx < n1 ? (*a)[idx] : (*b)[idx - n1]);
    if (src.first < 0) {
      g.target.push_back(-1);
      continue;
    }
    g.target.push_back(g.sourceOffset(src.first) +
                       (g.sourceIsSplat[src.first] ? 0 : src.second));
  }
  return g;
}

bool isShuffleElemType(Type t, unsigned bits) {
  if (t.getIntOrFloatBitWidth() != bits)
    return false;
  return t.isSignlessInteger() || t.isBF16() || t.isF32();
}

Value emitPlan(ConversionPatternRewriter &rewriter, Location loc,
               const ShuffleGather &g, const ShufflePlan &plan,
               ArrayRef<Value> leaves, VectorType resTy) {
  Type elemTy = resTy.getElementType();
  unsigned w = g.elemBits;
  int64_t lanes = 512 / w;
  auto regTy = VectorType::get({lanes}, elemTy);
  SmallVector<Value> vals;
  for (const Node &node : plan.nodes) {
    Value v;
    switch (node.kind) {
    case Node::Source: {
      Value leaf = rewriter.getRemappedValue(leaves[node.source]);
      int64_t n = g.sourceLanes[node.source];
      if (n == lanes) {
        v = leaf;
        break;
      }
      // The 512-bit part, or the whole source padded with poison lanes.
      SmallVector<int64_t> mask;
      for (int64_t j = 0; j < lanes; ++j) {
        int64_t lane = node.chunk * lanes + j;
        mask.push_back(lane < n ? lane : vector::ShuffleOp::kPoisonIndex);
      }
      v = vector::ShuffleOp::create(rewriter, loc, leaf, leaf, mask);
      break;
    }
    case Node::Splat: {
      Value scalar;
      for (unsigned k = 0; k < g.sourceLanes.size() && !scalar; ++k) {
        int32_t lane = node.label - g.sourceOffset(k);
        if (lane < 0 || lane >= (g.sourceIsSplat[k] ? 1 : g.sourceLanes[k]))
          continue;
        if (g.sourceIsSplat[k]) {
          scalar = rewriter.getRemappedValue(
              leaves[k].getDefiningOp<vector::BroadcastOp>().getSource());
        } else {
          scalar = vector::ExtractOp::create(
              rewriter, loc, rewriter.getRemappedValue(leaves[k]), lane);
        }
      }
      v = aievec::BroadcastScalarOp::create(rewriter, loc, regTy, scalar);
      break;
    }
    case Node::Shuffle: {
      unsigned mw = getShuffleModeShape(node.mode).elemBits;
      Type opElemTy = isShuffleElemType(elemTy, mw)
                          ? elemTy
                          : Type(rewriter.getIntegerType(mw));
      auto opTy = VectorType::get({512 / mw}, opElemTy);
      auto cast = [&](Value x) -> Value {
        if (x.getType() == opTy)
          return x;
        return vector::BitCastOp::create(rewriter, loc, opTy, x);
      };
      Value lhs = cast(vals[node.lhs]);
      Value rhs = getShuffleModeShape(node.mode).isTwoInput()
                      ? cast(vals[node.rhs])
                      : Value();
      v = aievec::ShuffleOp::create(rewriter, loc, opTy, lhs, rhs, node.mode);
      if (opTy != regTy)
        v = vector::BitCastOp::create(rewriter, loc, regTy, v);
      break;
    }
    }
    vals.push_back(v);
  }

  SmallVector<Value> parts;
  for (int r : plan.results)
    parts.push_back(r >= 0 ? vals[r]
                           : ub::PoisonOp::create(rewriter, loc, regTy));
  int64_t m = resTy.getNumElements();
  if (m < lanes) {
    auto mask = llvm::to_vector(llvm::seq<int64_t>(0, m));
    return vector::ShuffleOp::create(rewriter, loc, parts[0], parts[0], mask);
  }
  while (parts.size() > 1) {
    SmallVector<Value> next;
    for (size_t i = 0; i < parts.size(); i += 2) {
      int64_t len = cast<VectorType>(parts[i].getType()).getNumElements();
      auto mask = llvm::to_vector(llvm::seq<int64_t>(0, 2 * len));
      next.push_back(vector::ShuffleOp::create(rewriter, loc, parts[i],
                                               parts[i + 1], mask));
    }
    parts = std::move(next);
  }
  return parts[0];
}

ShuffleRequest requestOf(vector::ShuffleOp op) {
  return {op.getV1(), op.getV2(), llvm::to_vector(op.getMask()),
          op.getResultVectorType(), op};
}

// vector.interleave as a shuffle of its two operands.
std::optional<ShuffleRequest> requestOf(vector::InterleaveOp op) {
  VectorType srcTy = op.getSourceVectorType();
  if (srcTy.getRank() != 1 || srcTy.isScalable())
    return std::nullopt;
  int64_t n = srcTy.getNumElements();
  SmallVector<int64_t> mask;
  for (int64_t i = 0; i < 2 * n; ++i)
    mask.push_back(i / 2 + (i % 2) * n);
  return ShuffleRequest{op.getLhs(), op.getRhs(), mask,
                        op.getResultVectorType(), op};
}

// One result of vector.deinterleave as a shuffle of its source.
std::optional<ShuffleRequest> requestOf(vector::DeinterleaveOp op,
                                        unsigned result) {
  VectorType srcTy = op.getSourceVectorType();
  if (srcTy.getRank() != 1 || srcTy.isScalable())
    return std::nullopt;
  SmallVector<int64_t> mask;
  for (int64_t i = result; i < srcTy.getNumElements(); i += 2)
    mask.push_back(i);
  return ShuffleRequest{op.getSource(), op.getSource(), mask,
                        op.getResultVectorType(), op};
}

Value lower(ConversionPatternRewriter &rewriter, AIE2PShufflePlanner &planner,
            const ShuffleRequest &req, const ShufflePlan &plan,
            const ShuffleGather &g, ArrayRef<Value> leaves) {
  Value v =
      emitPlan(rewriter, req.user->getLoc(), g, plan, leaves, req.resultType);
  LLVM_DEBUG({
    llvm::dbgs() << "lowered with " << plan.cost << " instructions, modes";
    for (ShuffleMode m : plan.modes())
      llvm::dbgs() << " " << stringifyShuffleMode(m);
    llvm::dbgs() << "\n";
  });
  planner.noteUsed(plan);
  return v;
}

// A static vector.shuffle as the plan the planner found for it.
struct LowerStaticShuffleAIE2pPattern : OpConversionPattern<vector::ShuffleOp> {
  LowerStaticShuffleAIE2pPattern(MLIRContext *ctx,
                                 std::shared_ptr<AIE2PShufflePlanner> planner)
      : OpConversionPattern(ctx, /*benefit=*/3), planner(std::move(planner)) {}

  LogicalResult
  matchAndRewrite(vector::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ShuffleRequest req = requestOf(op);
    ShuffleGather g;
    SmallVector<Value> leaves;
    std::optional<ShufflePlan> plan = planner->planFor(req, &g, &leaves);
    if (!plan)
      return rewriter.notifyMatchFailure(op, "no plan beats LLVM's lowering");
    rewriter.replaceOp(op, lower(rewriter, *planner, req, *plan, g, leaves));
    return success();
  }

  std::shared_ptr<AIE2PShufflePlanner> planner;
};

struct LowerInterleaveAIE2pPattern : OpConversionPattern<vector::InterleaveOp> {
  LowerInterleaveAIE2pPattern(MLIRContext *ctx,
                              std::shared_ptr<AIE2PShufflePlanner> planner)
      : OpConversionPattern(ctx), planner(std::move(planner)) {}

  LogicalResult
  matchAndRewrite(vector::InterleaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<ShuffleRequest> req = requestOf(op);
    ShuffleGather g;
    SmallVector<Value> leaves;
    std::optional<ShufflePlan> plan;
    if (req)
      plan = planner->planFor(*req, &g, &leaves);
    if (!plan)
      return rewriter.notifyMatchFailure(op, "no plan beats LLVM's lowering");
    rewriter.replaceOp(op, lower(rewriter, *planner, *req, *plan, g, leaves));
    return success();
  }

  std::shared_ptr<AIE2PShufflePlanner> planner;
};

// Each result of vector.deinterleave the planner has a plan for is lowered
// with it; the other becomes a vector.shuffle.
struct LowerDeinterleaveAIE2pPattern
    : OpConversionPattern<vector::DeinterleaveOp> {
  LowerDeinterleaveAIE2pPattern(MLIRContext *ctx,
                                std::shared_ptr<AIE2PShufflePlanner> planner)
      : OpConversionPattern(ctx), planner(std::move(planner)) {}

  LogicalResult
  matchAndRewrite(vector::DeinterleaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<ShuffleRequest, 2> reqs;
    SmallVector<std::optional<ShufflePlan>, 2> plans;
    SmallVector<ShuffleGather, 2> gathers(2);
    SmallVector<SmallVector<Value>, 2> leaves(2);
    for (unsigned r = 0; r < 2; ++r) {
      std::optional<ShuffleRequest> req = requestOf(op, r);
      if (!req)
        return failure();
      reqs.push_back(*req);
      plans.push_back(planner->planFor(*req, &gathers[r], &leaves[r]));
    }
    if (!plans[0] && !plans[1])
      return rewriter.notifyMatchFailure(op, "no plan beats LLVM's lowering");
    SmallVector<Value, 2> results;
    for (unsigned r = 0; r < 2; ++r)
      results.push_back(plans[r]
                            ? lower(rewriter, *planner, reqs[r], *plans[r],
                                    gathers[r], leaves[r])
                            : vector::ShuffleOp::create(
                                  rewriter, op.getLoc(), adaptor.getSource(),
                                  adaptor.getSource(), reqs[r].mask)
                                  .getResult());
    rewriter.replaceOp(op, results);
    return success();
  }

  std::shared_ptr<AIE2PShufflePlanner> planner;
};

// A vector.transpose as a vector.shuffle of the flattened vector, which the
// planner or LLVM then lowers.
struct LowerVectorTransposeToShuffleAIE2pPattern
    : OpConversionPattern<vector::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcTy = op.getSourceVectorType();
    VectorType resTy = op.getResultVectorType();
    if (srcTy.isScalable())
      return failure();
    ArrayRef<int64_t> perm = op.getPermutation();
    ArrayRef<int64_t> srcShape = srcTy.getShape();
    SmallVector<int64_t> srcStrides(srcShape.size(), 1);
    for (int64_t d = srcShape.size() - 2; d >= 0; --d)
      srcStrides[d] = srcStrides[d + 1] * srcShape[d + 1];
    int64_t n = srcTy.getNumElements();
    SmallVector<int64_t> mask;
    SmallVector<int64_t> idx(resTy.getRank(), 0);
    for (int64_t i = 0; i < n; ++i) {
      // Result index idx reads source index s with s[perm[d]] = idx[d].
      int64_t s = 0;
      for (int64_t d = 0; d < resTy.getRank(); ++d)
        s += idx[d] * srcStrides[perm[d]];
      mask.push_back(s);
      for (int64_t d = resTy.getRank() - 1; d >= 0; --d) {
        if (++idx[d] < resTy.getDimSize(d))
          break;
        idx[d] = 0;
      }
    }
    Location loc = op.getLoc();
    auto flatTy = VectorType::get({n}, srcTy.getElementType());
    Value flat =
        vector::ShapeCastOp::create(rewriter, loc, flatTy, adaptor.getVector());
    Value shuffled = vector::ShuffleOp::create(rewriter, loc, flat, flat, mask);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resTy, shuffled);
    return success();
  }
};

} // namespace

std::optional<ShufflePlan>
AIE2PShufflePlanner::planFor(const ShuffleRequest &req, ShuffleGather *gather,
                             SmallVector<Value> *leaves) {
  SmallVector<Value> asIsLeaves;
  std::optional<ShuffleGather> asIs = gatherOf(req, /*fold=*/false, asIsLeaves);
  if (!asIs)
    return std::nullopt;
  unsigned llvmCost = estimateLLVMShuffleCost(*asIs);
  if (llvmCost == 0)
    return std::nullopt;
  LLVM_DEBUG(llvm::dbgs() << "planning " << *req.user
                          << "\nLLVM estimate: " << llvmCost << "\n");

  // A mode costs one instruction to load into a scalar register, unless an
  // earlier plan has loaded it already. Then fewer new modes and fewer
  // instructions break ties.
  auto score = [&](const ShufflePlan &p) {
    unsigned fresh = llvm::count_if(p.modes(), [&](ShuffleMode m) {
      return !llvm::is_contained(usedModes, m);
    });
    return std::make_tuple(p.cost + fresh, fresh, p.cost);
  };
  std::optional<ShufflePlan> best;
  auto consider = [&](const ShuffleGather &g, SmallVector<Value> &ls) {
    for (const ShufflePlan &p : candidates(g)) {
      if (p.cost >= llvmCost || (best && score(p) >= score(*best)))
        continue;
      best = p;
      if (gather)
        *gather = g;
      if (leaves)
        *leaves = ls;
    }
  };
  SmallVector<Value> foldedLeaves;
  std::optional<ShuffleGather> folded =
      gatherOf(req, /*fold=*/true, foldedLeaves);
  if (folded && folded->key() != asIs->key())
    consider(*folded, foldedLeaves);
  consider(*asIs, asIsLeaves);
  return best;
}

void AIE2PShufflePlanner::noteUsed(const ShufflePlan &plan) {
  for (ShuffleMode m : plan.modes())
    if (!llvm::is_contained(usedModes, m))
      usedModes.push_back(m);
}

void xilinx::aievec::populateAIE2PShufflePlannerPatterns(
    RewritePatternSet &patterns, std::shared_ptr<AIE2PShufflePlanner> planner) {
  patterns.add<LowerStaticShuffleAIE2pPattern, LowerInterleaveAIE2pPattern,
               LowerDeinterleaveAIE2pPattern>(patterns.getContext(), planner);
  // Ahead of the AIE2 lowering, which takes 512-bit transposes only.
  patterns.add<LowerVectorTransposeToShuffleAIE2pPattern>(patterns.getContext(),
                                                          /*benefit=*/2);
}

void xilinx::aievec::configureAIE2PShufflePlannerLegality(
    ConversionTarget &target, std::shared_ptr<AIE2PShufflePlanner> planner) {
  target.addDynamicallyLegalOp<vector::ShuffleOp>(
      [planner](vector::ShuffleOp op) {
        return !planner->planFor(requestOf(op)).has_value();
      });
  target.addDynamicallyLegalOp<vector::InterleaveOp>(
      [planner](vector::InterleaveOp op) {
        std::optional<ShuffleRequest> req = requestOf(op);
        return !req || !planner->planFor(*req).has_value();
      });
  target.addDynamicallyLegalOp<vector::DeinterleaveOp>(
      [planner](vector::DeinterleaveOp op) {
        for (unsigned r = 0; r < 2; ++r) {
          std::optional<ShuffleRequest> req = requestOf(op, r);
          if (req && planner->planFor(*req))
            return false;
        }
        return true;
      });
}
