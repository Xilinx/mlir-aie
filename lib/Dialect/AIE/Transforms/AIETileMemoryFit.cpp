//===- AIETileMemoryFit.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIETileMemoryFit.h"
#include "aie/Dialect/AIE/Transforms/AIEBufferAllocation.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <set>
#include <tuple>

using namespace xilinx::AIE;

TileMemoryLayout::TileMemoryLayout(const AIETargetModel &target, TileID tile,
                                   bool hasCore)
    : numBanks(target.getNumBanks(tile.col, tile.row)),
      maxDataMemorySize(target.getTileType(tile.col, tile.row) ==
                                AIETileType::MemTile
                            ? target.getMemTileSize()
                            : target.getLocalMemorySize()),
      hasCore(hasCore) {
  int64_t bankSize = maxDataMemorySize / numBanks;
  for (int i = 0; i < numBanks; i++)
    bankLimits.push_back({bankSize * i, bankSize});
  // On npu2, the first word of the column-0 memtile reads back as 0x00CD0CD0
  // on every dispatch after the first, though nothing in the design writes it.
  // A buffer that DMA refills each dispatch is unaffected, but an initialized
  // one is loaded only once and keeps the bad word, so keep initialized buffers
  // off that word. Other buffers may still take it, and a buffer the user
  // pinned there keeps its address.
  //
  // TODO: Investigate. This is a low-confidence workaround for a cause nobody
  // has found: the word may be reserved by the firmware or driver, or clobbered
  // by something else. The value predates initialized buffers dropping their
  // locks (#3608 saw it first), so that change is not the cause.
  if (target.getTileType(tile.col, tile.row) == AIETileType::MemTile &&
      tile.col == 0 && target.hasProperty(AIETargetModel::IsNPU) &&
      target.getTargetArch() == AIEArch::AIE2p)
    initGuardBytes = target.getMemTileLoadStoreBusWidth() / 8;
}

int TileMemoryLayout::bankContaining(int64_t addr) const {
  for (int i = 0; i < numBanks; i++) {
    if (bankLimits[i].contains(addr)) {
      return i;
    }
  }
  return -1;
}

// Peano models a bank as an address space (`aiebase_resources.h`: a..d are
// 5..8, 9..14 are pairs) and schedules loads on the strength of the qualifier,
// while nothing placed the buffer to match. Reading it here makes that
// assumption true rather than merely asserted.
llvm::SmallVector<int, 2> xilinx::AIE::banksFromMemorySpace(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memref) {
    return {};
  }
  auto space =
      llvm::dyn_cast_or_null<mlir::IntegerAttr>(memref.getMemorySpace());
  if (!space) {
    return {};
  }
  int64_t value = space.getInt();
  if (value >= 5 && value <= 8) {
    return {static_cast<int>(value - 5)};
  }
  switch (value) {
  case 9:
    return {0, 1};
  case 10:
    return {0, 2};
  case 11:
    return {0, 3};
  case 12:
    return {1, 2};
  case 13:
    return {1, 3};
  case 14:
    return {2, 3};
  default:
    return {};
  }
}

int64_t xilinx::AIE::fitAlignBytes(bool aligned, uint32_t tileAlignBitWidth,
                                   uint32_t maxVecAlignBits, int64_t size,
                                   int64_t measuredAlign) {
  // Linker section alignment still applies when vector alignment is disabled.
  if (!aligned) {
    return measuredAlign;
  }
  return std::max<int64_t>(
      requiredBufferAlignBits(tileAlignBitWidth, maxVecAlignBits, size) / 8,
      measuredAlign);
}

llvm::SmallVector<unsigned>
xilinx::AIE::fitOrder(llvm::ArrayRef<FitBuffer> buffers) {
  auto order = llvm::to_vector(
      llvm::seq<unsigned>(0, static_cast<unsigned>(buffers.size())));
  llvm::stable_sort(order, [&](unsigned ai, unsigned bi) {
    const FitBuffer &a = buffers[ai], &b = buffers[bi];
    if (a.banks.empty() != b.banks.empty()) {
      return !a.banks.empty();
    }
    if (!a.banks.empty() && a.banks.size() != b.banks.size())
      return a.banks.size() < b.banks.size();
    return a.size > b.size;
  });
  return order;
}

namespace {
// Sentinel first key for an unscored fallback, so it sorts behind every
// scored candidate however those score.
constexpr int64_t kFallbackRank = std::numeric_limits<int64_t>::max();

// Candidate address ordering, lower ranks first:
//  1. banks touched, fewest first (each spanned bank costs DMA bandwidth);
//  2. largest free run left behind, biggest first, capped by contiguityCap --
//     the bytes still to place. A run wider than that serves nothing, so past
//     the cap every candidate ties and criterion 3 takes over;
//  3. round-robin distance from the cursor, nearest first (spreads for DMA);
//  4. slack, tightest first, which leaves the large runs unbroken;
//  5. address, lowest first, for determinism.
// A memtile has no core, so criterion 2 is neutralized there and round-robin
// governs throughout.
struct Placement {
  int64_t addr;
  int bank;
  int64_t banksTouched;
  int64_t negLargestRunLeft;
  int64_t rrDistance;
  int64_t slack;

  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> rank() const {
    return {banksTouched, negLargestRunLeft, rrDistance, slack, addr};
  }
};
} // namespace

// Every address `buffer` could take in the current occupancy, best first.
//
// The head of this list is the address a single-pass greedy would have chosen,
// so a design that placed before places identically; the tail is where the
// search goes when a later buffer is left with nowhere to sit.
static llvm::SmallVector<Placement>
rankedPlacements(const FitBuffer &buffer, const TileMemoryLayout &ctx,
                 int startBankIndex, const MemoryOccupancy &tileOccupancy,
                 int64_t contiguityCap) {
  assert(startBankIndex < ctx.numBanks &&
         "Unexpected input value for startBankIndex");
  // An initialized buffer sees the guarded bytes as taken, so none of its
  // candidates cover them. Every other buffer may still be placed there.
  std::optional<MemoryOccupancy> guarded;
  if (ctx.initGuardBytes > 0 && buffer.initialized) {
    guarded = tileOccupancy;
    guarded->markOccupied(0, std::min(ctx.initGuardBytes, guarded->size()));
  }
  const MemoryOccupancy &occupancy = guarded ? *guarded : tileOccupancy;
  int64_t size = buffer.size;
  int64_t alignBytes = buffer.alignBytes;
  llvm::SmallVector<Placement> out;

  // A requested bank set is a hard constraint, and the round-robin cursor stays
  // put. Only the two flush ends of each hole are worth trying, because an
  // interior start leaves strictly less behind.
  if (!buffer.banks.empty()) {
    auto &banks = buffer.banks;
    llvm::SmallVector<MemoryRun> ranges;
    for (int bank : banks)
      ranges.push_back(ctx.bankLimits[bank]);
    // Adjacent allowed banks can hold one spanning buffer; a disjoint pair
    // cannot use the intervening, forbidden bank.
    if (banks.size() == 2 && banks.back() == banks.front() + 1)
      ranges.push_back({ctx.bankLimits[banks.front()].start,
                        ctx.bankLimits[banks.front()].size +
                            ctx.bankLimits[banks.back()].size});
    for (MemoryRun limits : ranges) {
      for (MemoryRun gap : occupancy.gapsIn(limits.start, limits.end())) {
        int64_t low = llvm::alignTo(gap.start, alignBytes);
        if (low + size > gap.end()) {
          continue;
        }
        for (int64_t addr :
             {low, std::max<int64_t>(
                       llvm::alignDown(gap.end() - size, alignBytes), low)}) {
          MemoryOccupancy trial = occupancy;
          trial.markOccupied(addr, addr + size);
          int bank = ctx.bankContaining(addr);
          if (!llvm::is_contained(banks, bank))
            continue;
          int lastBank = size == 0 ? bank : ctx.bankContaining(addr + size - 1);
          out.push_back({addr, bank, /*banksTouched=*/lastBank - bank + 1,
                         -trial.largestGap(0, trial.size()), /*rrDistance=*/0,
                         /*slack=*/0});
        }
      }
    }
    // A zero-sized buffer needs no free byte, so every bank holds it.
    if (out.empty() && size == 0) {
      int bank = banks.front();
      out.push_back({ctx.bankLimits[bank].start, bank, 1, 0, 0, 0});
    }
  } else {
    int64_t bankedEnd = ctx.bankLimits.back().end();
    int64_t bankSize = bankedEnd / ctx.numBanks;
    // Computed once per buffer, not per candidate: every candidate splits one
    // run, so the largest run left elsewhere is one of these two. Named
    // separately, because C++17 cannot capture a structured binding in the
    // lambda below.
    std::pair<int64_t, int64_t> topGaps =
        occupancy.topTwoGaps(0, ctx.maxDataMemorySize);
    int64_t largestRun = topGaps.first, secondRun = topGaps.second;
    auto consider = [&](std::optional<int64_t> addr) {
      if (!addr) {
        return;
      }
      int bank = ctx.bankContaining(*addr);
      if (bank < 0) {
        return;
      }
      int64_t touched =
          size == 0 ? 1 : (*addr + size - 1) / bankSize - *addr / bankSize + 1;
      // This placement splits one run into the piece below and the piece above.
      // Every other run stays whole, so the biggest of those is whichever of
      // the top two this candidate does not consume.
      MemoryRun gap = occupancy.gapAt(*addr);
      int64_t elsewhere = gap.size == largestRun ? secondRun : largestRun;
      int64_t leftPiece = *addr - gap.start;
      int64_t rightPiece = gap.end() - (*addr + size);
      int64_t runLeft =
          std::min(std::max({elsewhere, leftPiece, rightPiece}), contiguityCap);
      out.push_back({*addr, bank, touched, ctx.hasCore ? -runLeft : 0,
                     (bank - startBankIndex + ctx.numBanks) % ctx.numBanks,
                     occupancy.slackAt(*addr, size)});
    };

    for (int i = 0; i < ctx.numBanks; i++) {
      consider(occupancy.findGap(ctx.bankLimits[i].start,
                                 ctx.bankLimits[i].end(), size, alignBytes));
    }
    // Allow straddling when nothing fits one bank. The banked region maps to a
    // bank, so the result always has one.
    consider(occupancy.findGap(0, bankedEnd, size, alignBytes));
    // Try bank-boundary starts only when a boundary also satisfies the buffer's
    // own alignment, because otherwise this searches on an unintended stride.
    if (bankSize % alignBytes == 0) {
      consider(occupancy.findGap(0, bankedEnd, size, bankSize));
    }
    // A zero-sized buffer needs no free byte, so a full tile still holds it.
    if (out.empty() && size == 0) {
      out.push_back({0, 0, 1, 0, 0, 0});
    }
  }

  llvm::stable_sort(out, [](const Placement &a, const Placement &b) {
    return a.rank() < b.rank();
  });

  // The probes above ask each hole only for its tightest fit, so a layout
  // needing some other position in one is unreachable -- and unreachable reads
  // as "no room". Both flush ends of every hole cover that. Appended after the
  // sort, never scored: ranking them would move addresses that already work.
  // Unpinned only; the mem_bank branch above already does this within its bank,
  // and widening it here would place the buffer outside that bank.
  if (size > 0 && buffer.banks.empty()) {
    for (MemoryRun gap : occupancy.gapsIn(0, ctx.maxDataMemorySize)) {
      int64_t low = llvm::alignTo(gap.start, alignBytes);
      if (low + size > gap.end()) {
        continue;
      }
      for (int64_t addr :
           {low, std::max<int64_t>(
                     llvm::alignDown(gap.end() - size, alignBytes), low)}) {
        int bank = ctx.bankContaining(addr);
        if (bank >= 0) {
          // Ranks last by construction, so it is only ever reached by
          // backtracking past every scored candidate.
          out.push_back({addr, bank, kFallbackRank, 0, 0, 0});
        }
      }
    }
  }

  // The probes overlap: a gap that is the tightest in its bank is often also
  // the tightest overall, and the fallbacks repeat whichever of those was a
  // flush end. Equal addresses are the same placement, and trying one twice
  // only costs budget. Stable, so the ranked entry survives its duplicate.
  llvm::stable_sort(out, [](const Placement &a, const Placement &b) {
    return a.addr < b.addr;
  });
  out.erase(
      llvm::unique(out, [](const Placement &a,
                           const Placement &b) { return a.addr == b.addr; }),
      out.end());
  llvm::stable_sort(out, [](const Placement &a, const Placement &b) {
    return a.rank() < b.rank();
  });
  return out;
}

// Depth-first placement over `order`, trying each buffer's addresses in rank
// order and undoing a choice that leaves a later buffer nowhere to go.
//
// The first descent is exactly what a single-pass greedy produces, so this only
// does extra work where that pass used to fail outright. On failure `deepest`
// names the buffer that ran out of addresses furthest into the order, which is
// the one whose demand the tile could not meet.
namespace {
struct PlacementSearch {
  llvm::ArrayRef<FitBuffer> order;
  const TileMemoryLayout &ctx;
  MemoryOccupancy &occupancy;
  llvm::SmallVector<std::pair<size_t, FitPlacement>> &placed;
  FitStats &stats;
  // How many placements the search may still try on this tile before giving
  // up and reporting the deepest failure it reached; the pass's
  // placement-budget option. Packing around fixed obstacles is NP-hard, so the
  // tree has no useful worst-case bound; this keeps a design the search cannot
  // solve to bounded compile time instead of exponential. A node count rather
  // than a time limit, so a build stays reproducible.
  int64_t budget;
  // Set when the node budget ran out with candidates still untried, so the
  // caller can say "gave up" rather than "no such layout exists".
  bool exhausted = false;

  // The furthest the search reached, kept so a failure can report the layout it
  // got closest with rather than the empty tile backtracking leaves behind.
  // First arrangement to reach a given depth wins, not the last: the first
  // comes down the highest-ranked path, so the map a user sees is the one the
  // ranking preferred rather than whichever branch happened to be explored
  // last.
  std::optional<size_t> deepest = std::nullopt;
  llvm::SmallVector<std::pair<size_t, FitPlacement>> deepestLayout = {};

  // Subproblems already searched in full without a layout, keyed by the next
  // buffer's index and the free runs left. The candidates a buffer gets depend
  // on nothing else -- the cursor only orders them -- so one of these can never
  // succeed. Same-sized buffers reach the same free runs in every order they
  // are placed in; without this the search retries each order.
  std::set<llvm::SmallVector<int64_t>> failedStates = {};

  bool run(size_t index, int startBankIndex, int64_t remaining) {
    if (index == order.size()) {
      return true;
    }
    llvm::SmallVector<int64_t> state = {static_cast<int64_t>(index)};
    for (MemoryRun gap : occupancy.gapsIn(0, occupancy.size())) {
      state.append({gap.start, gap.size});
    }
    if (failedStates.count(state)) {
      return false;
    }
    const FitBuffer &buffer = order[index];
    int64_t size = buffer.size;
    int64_t rest = remaining - size; // Placement's criterion 2 cap.

    llvm::SmallVector<Placement> candidates =
        rankedPlacements(buffer, ctx, startBankIndex, occupancy, rest);
    if (candidates.empty() && (!deepest || index > *deepest)) {
      recordDeepest(index);
    }
    for (const Placement &candidate : candidates) {
      if (budget <= 0) {
        // Out of budget before this buffer had an address: it is where the
        // search stopped, so it is the one to report.
        if (!deepest || index > *deepest) {
          recordDeepest(index);
        }
        exhausted = true;
        break;
      }
      --budget;
      occupancy.markOccupied(candidate.addr, candidate.addr + size);
      placed.push_back({index, {candidate.addr, candidate.bank}});
      // A pinned buffer leaves the cursor alone; see rankedPlacements.
      int nextBank = !buffer.banks.empty()
                         ? startBankIndex
                         : (candidate.bank + 1) % ctx.numBanks;
      if (run(index + 1, nextBank, rest)) {
        return true;
      }
      placed.pop_back();
      ++stats.backtracks;
      occupancy.markFree(candidate.addr, candidate.addr + size);
    }
    if (!exhausted) {
      failedStates.insert(std::move(state));
    }
    return false;
  }

  // Re-apply the best partial layout so the caller's memory map describes the
  // closest the search came, not the tile it unwound to.
  void restoreDeepest() {
    for (auto &[index, at] : deepestLayout) {
      occupancy.markOccupied(at.addr, at.addr + order[index].size);
      placed.push_back({index, at});
    }
  }

private:
  void recordDeepest(size_t index) {
    deepest = index;
    deepestLayout.assign(placed.begin(), placed.end());
  }
};
} // namespace

// Exhaustive placement, for when the ranked search fails. The ranked search
// only tries positions flush against what is already placed, so a buffer that
// must sit against one placed after it is out of its reach in any order.
//
// Any layout can be slid down, one buffer at a time in address order, until
// each buffer sits at the lowest address its alignment and banks allow above
// the one before it. Walking the free runs upward and choosing which buffer
// comes next therefore reaches every layout that exists, and a tile this
// rejects within budget has none. What can follow an address depends only on
// the address and the buffers left, so a failed pair is never tried twice,
// which collapses the orders of one set of buffers into a single visit.
namespace {
struct CompactionSearch {
  llvm::ArrayRef<FitBuffer> order;
  const TileMemoryLayout &ctx;
  llvm::SmallVector<MemoryRun> gaps;
  // Per buffer in `order`: size, alignment, the ranges it may lie in, and a
  // class shared with every buffer it could swap with.
  llvm::SmallVector<int64_t> sizes, aligns;
  llvm::SmallVector<llvm::SmallVector<MemoryRun, 2>> ranges;
  llvm::SmallVector<unsigned> classes;
  llvm::SmallVector<int64_t> addrs;
  // Free bytes and the largest run from each gap to the top of the tile.
  llvm::SmallVector<int64_t> freeAbove, runAbove;
  llvm::DenseSet<std::pair<int64_t, llvm::BitVector>> failed;
  int64_t budget;
  bool exhausted = false;

  CompactionSearch(llvm::ArrayRef<FitBuffer> order, const TileMemoryLayout &ctx,
                   const MemoryOccupancy &occupancy, int64_t budget)
      : order(order), ctx(ctx),
        gaps(occupancy.gapsIn(0, ctx.maxDataMemorySize)),
        addrs(order.size(), -1), budget(budget) {
    for (size_t i = 0; i < order.size(); ++i) {
      const FitBuffer &buffer = order[i];
      sizes.push_back(buffer.size);
      aligns.push_back(buffer.alignBytes);
      llvm::SmallVector<MemoryRun, 2> allowed;
      if (buffer.banks.empty()) {
        allowed.push_back({0, ctx.maxDataMemorySize});
      } else {
        auto &banks = buffer.banks;
        for (int bank : banks)
          allowed.push_back(ctx.bankLimits[bank]);
        if (banks.size() == 2 && banks.back() == banks.front() + 1)
          allowed = {{ctx.bankLimits[banks.front()].start,
                      ctx.bankLimits[banks.front()].size +
                          ctx.bankLimits[banks.back()].size}};
      }
      // An initialized buffer must dodge the guarded bytes at the foot of
      // the tile, the same restriction rankedPlacements enforces. Clamping the
      // ranges here, before the equivalence classes below are computed, keeps a
      // guarded buffer out of any class an unguarded one of the same size
      // could also join.
      if (ctx.initGuardBytes > 0 && buffer.initialized) {
        llvm::SmallVector<MemoryRun, 2> clamped;
        for (MemoryRun range : allowed) {
          int64_t start = std::max(range.start, ctx.initGuardBytes);
          if (start < range.end())
            clamped.push_back({start, range.end() - start});
        }
        allowed = std::move(clamped);
      }
      ranges.push_back(allowed);
      unsigned cls = i;
      for (size_t j = 0; j < i; ++j) {
        if (sizes[j] == sizes[i] && aligns[j] == aligns[i] &&
            llvm::equal(ranges[j], ranges[i], [](MemoryRun a, MemoryRun b) {
              return a.start == b.start && a.size == b.size;
            })) {
          cls = classes[j];
          break;
        }
      }
      classes.push_back(cls);
    }
    freeAbove.assign(gaps.size() + 1, 0);
    runAbove.assign(gaps.size() + 1, 0);
    for (size_t g = gaps.size(); g-- > 0;) {
      freeAbove[g] = freeAbove[g + 1] + gaps[g].size;
      runAbove[g] = std::max(runAbove[g + 1], gaps[g].size);
    }
  }

  // Lowest address at or above `from` in gap `g` where buffer `i` fits.
  std::optional<int64_t> lowest(size_t i, size_t g, int64_t from) const {
    std::optional<int64_t> best;
    for (MemoryRun range : ranges[i]) {
      int64_t at = llvm::alignTo(std::max(from, range.start), aligns[i]);
      if (at + sizes[i] <= std::min(gaps[g].end(), range.end()) &&
          ctx.bankContaining(at) >= 0 && (!best || at < *best))
        best = at;
    }
    return best;
  }

  bool run() {
    llvm::BitVector left(order.size());
    int64_t bytes = 0;
    for (size_t i = 0; i < order.size(); ++i) {
      if (sizes[i] == 0) {
        // Covers no byte, so it goes in the first gap its banks reach.
        for (size_t g = 0; g < gaps.size() && addrs[i] < 0; ++g)
          if (auto at = lowest(i, g, gaps[g].start))
            addrs[i] = *at;
        if (addrs[i] < 0)
          addrs[i] = ranges[i].front().start;
        continue;
      }
      left.set(i);
      bytes += sizes[i];
    }
    return gaps.empty() ? left.none()
                        : place(0, gaps.front().start, left, bytes);
  }

private:
  bool place(size_t g, int64_t from, llvm::BitVector &left, int64_t bytes) {
    if (left.none())
      return true;
    int64_t largest = 0;
    for (int i : left.set_bits())
      largest = std::max(largest, sizes[i]);
    int64_t here = gaps[g].end() - from;
    if (bytes > here + freeAbove[g + 1] ||
        largest > std::max(here, runAbove[g + 1]) ||
        failed.contains({from, left}))
      return false;
    llvm::SmallVector<unsigned> tried;
    for (int i : left.set_bits()) {
      if (llvm::is_contained(tried, classes[i]))
        continue;
      tried.push_back(classes[i]);
      std::optional<int64_t> at = lowest(i, g, from);
      if (!at)
        continue;
      if (budget-- <= 0) {
        exhausted = true;
        return false;
      }
      addrs[i] = *at;
      left.reset(i);
      bool done = place(g, *at + sizes[i], left, bytes - sizes[i]);
      left.set(i);
      if (done)
        return true;
      if (exhausted)
        return false;
    }
    if (g + 1 < gaps.size() && place(g + 1, gaps[g + 1].start, left, bytes))
      return true;
    if (!exhausted)
      failed.insert({from, left});
    return false;
  }
};
} // namespace

FitResult xilinx::AIE::fitTileBuffers(const TileMemoryLayout &layout,
                                      llvm::ArrayRef<FitBuffer> buffers,
                                      MemoryOccupancy &occupancy,
                                      int64_t budget, FitStats &stats) {
  FitResult result;
  int64_t remaining = 0;
  for (const FitBuffer &buffer : buffers) {
    remaining += buffer.size;
  }
  PlacementSearch search{buffers,       layout, occupancy,
                         result.placed, stats,  budget};
  int64_t before = stats.backtracks;
  bool solved = search.run(/*index=*/0, /*startBankIndex=*/0, remaining);
  if (stats.backtracks > before) {
    ++stats.tilesSearched;
  }
  if (solved) {
    result.solved = true;
    return result;
  }
  // The ranked search unwinds everything it tried, so the tile is back to its
  // pinned buffers alone.
  CompactionSearch complete(buffers, layout, occupancy, budget);
  if (complete.run()) {
    ++stats.tilesCompacted;
    for (auto [index, addr] : llvm::enumerate(complete.addrs)) {
      occupancy.markOccupied(addr, addr + buffers[index].size);
      result.placed.push_back({index, {addr, layout.bankContaining(addr)}});
    }
    result.solved = true;
    return result;
  }
  search.restoreDeepest();
  result.exhausted = complete.exhausted;
  if (complete.exhausted) {
    ++stats.budgetExhausted;
  }
  result.failed = search.deepest;
  return result;
}
