//===- AIETileMemoryFit.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIETILEMEMORYFIT_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIETILEMEMORYFIT_H

#include "aie/Dialect/AIE/IR/AIECoreMemory.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "mlir/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <optional>
#include <utility>

namespace xilinx::AIE {

// Which bytes of one tile's data memory are taken. Byte-granular, because a
// buffer marked `aligned = false` is packed at an unaligned offset.
class MemoryOccupancy {
public:
  explicit MemoryOccupancy(int64_t size) : occupied(size, false) {}

  int64_t size() const { return occupied.size(); }

  // True when [start, end) lies inside the tile and no byte of it is taken.
  bool isRangeFree(int64_t start, int64_t end) const {
    if (start < 0 || end > size() || start > end) {
      return false;
    }
    return start == end || occupied.find_first_in(start, end) == -1;
  }

  void markOccupied(int64_t start, int64_t end) {
    assert(start >= 0 && end <= size() && start <= end &&
           "range must lie inside the tile");
    if (start < end) {
      occupied.set(start, end);
    }
  }

  // Undo a markOccupied. The caller lifts the core's data-region placeholder
  // once every buffer is placed. No buffer lies inside the placeholder's range,
  // so clearing exactly that range is exact.
  void markFree(int64_t start, int64_t end) {
    assert(start >= 0 && end <= size() && start <= end &&
           "range must lie inside the tile");
    if (start < end) {
      occupied.reset(start, end);
    }
  }

  // Placement for `size` bytes in [lo, hi) that leaves the largest single free
  // run behind anywhere in [0, size()), or nullopt when nothing fits. An object
  // placed before the unconstrained buffers -- the core's data region, a
  // bank-pinned buffer -- keeps more contiguous space when it sits flush
  // against occupied space. Only the two flush positions of each hole are
  // candidates, because an interior placement leaves strictly less contiguous
  // space.
  std::optional<int64_t> findLeastFragmentingGap(int64_t lo, int64_t hi,
                                                 int64_t size,
                                                 int64_t alignBytes) const {
    assert(alignBytes > 0 && "alignment must be positive");
    std::optional<int64_t> best;
    int64_t bestRun = -1;
    auto consider = [&](int64_t start) {
      MemoryOccupancy trial = *this;
      trial.markOccupied(start, start + size);
      int64_t run = trial.largestGap(0, this->size());
      // Ties to the lowest address, for determinism.
      if (run > bestRun || (run == bestRun && best && start < *best)) {
        bestRun = run;
        best = start;
      }
    };
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      int64_t low = llvm::alignTo(gapStart, alignBytes);
      if (low + size > gapEnd) {
        return;
      }
      consider(low);
      // Flush to the top, aligned down but never below `low`.
      consider(
          std::max<int64_t>(llvm::alignDown(gapEnd - size, alignBytes), low));
    });
    return best;
  }

  // Start of the tightest gap in [lo, hi) that holds `size` bytes, or nullopt.
  // Each candidate start is aligned up before the fit test, so a misaligned
  // first free byte does not reject a gap.
  std::optional<int64_t> findGap(int64_t lo, int64_t hi, int64_t size,
                                 int64_t alignBytes) const {
    assert(alignBytes > 0 && "alignment must be positive");
    std::optional<int64_t> best;
    int64_t bestSlack = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      int64_t start = llvm::alignTo(gapStart, alignBytes);
      if (start + size <= gapEnd) {
        // Waste measured across the whole gap, so front-alignment padding
        // counts and a padded hole does not rank as a tight fit.
        int64_t slack = (gapEnd - gapStart) - size;
        if (!best || slack < bestSlack) {
          best = start;
          bestSlack = slack;
        }
      }
    });
    return best;
  }

  // Total free bytes in [lo, hi).
  int64_t freeBytes(int64_t lo, int64_t hi) const {
    int64_t total = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      total += gapEnd - gapStart;
    });
    return total;
  }

  // Bytes of the enclosing free run that `size` bytes at `addr` would leave
  // unused; placement scoring prefers a tight fit to keep large runs whole.
  int64_t slackAt(int64_t addr, int64_t size) const {
    int64_t slack = 0;
    forEachGap(0, this->size(), [&](int64_t gapStart, int64_t gapEnd) {
      if (addr >= gapStart && addr + size <= gapEnd) {
        slack = (gapEnd - gapStart) - size;
      }
    });
    return slack;
  }

  // The free run enclosing `addr`, or a zero-length run if it is taken.
  MemoryRun gapAt(int64_t addr) const {
    MemoryRun found;
    forEachGap(0, this->size(), [&](int64_t gapStart, int64_t gapEnd) {
      if (addr >= gapStart && addr < gapEnd) {
        found = {gapStart, gapEnd - gapStart};
      }
    });
    return found;
  }

  // The two largest free runs by size, duplicates kept, so two equal runs give
  // {n, n}. A placement consumes one run and leaves the other whole, which lets
  // a candidate score the run it leaves behind in constant time.
  std::pair<int64_t, int64_t> topTwoGaps(int64_t lo, int64_t hi) const {
    int64_t first = 0, second = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      int64_t run = gapEnd - gapStart;
      if (run > first) {
        second = first;
        first = run;
      } else if (run > second) {
        second = run;
      }
    });
    return {first, second};
  }

  // Every maximal free run in [lo, hi), lowest first. The search enumerates
  // candidate placements from these rather than taking the single best run.
  llvm::SmallVector<MemoryRun> gapsIn(int64_t lo, int64_t hi) const {
    llvm::SmallVector<MemoryRun> runs;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      runs.push_back({gapStart, gapEnd - gapStart});
    });
    return runs;
  }

  // Largest single free run in [lo, hi). A reservation needs one contiguous
  // run, which freeBytes does not measure.
  int64_t largestGap(int64_t lo, int64_t hi) const {
    int64_t best = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      best = std::max(best, gapEnd - gapStart);
    });
    return best;
  }

private:
  // Calls fn(gapStart, gapEnd) for every maximal free run in [lo, hi).
  template <typename Fn>
  void forEachGap(int64_t lo, int64_t hi, Fn fn) const {
    lo = std::max<int64_t>(lo, 0);
    hi = std::min(hi, size());
    for (int64_t cursor = lo; cursor < hi;) {
      int gapStart = occupied.find_first_unset_in(cursor, hi);
      if (gapStart == -1) {
        break;
      }
      int nextTaken = occupied.find_first_in(gapStart, hi);
      // gapStart is clear, so gapEnd > gapStart >= cursor: the cursor advances.
      int64_t gapEnd = nextTaken == -1 ? hi : nextTaken;
      fn(gapStart, gapEnd);
      cursor = gapEnd;
    }
  }

  llvm::BitVector occupied;
};

/// A tile's data memory as aie-assign-buffer-addresses lays it out: equal
/// banks over its whole size.
struct TileMemoryLayout {
  int numBanks;
  llvm::SmallVector<MemoryRun> bankLimits;
  int64_t maxDataMemorySize;
  /// Whether this tile has a core, and so compiled sections that need a
  /// contiguous region. A memtile has none.
  bool hasCore;
  /// Bytes at the start of the tile an initialized buffer must not cover.
  /// Other buffers may still use them.
  int64_t initGuardBytes = 0;

  TileMemoryLayout(const AIETargetModel &target, TileID tile, bool hasCore);

  /// Index of the bank owning `addr`, or -1 when it falls outside every bank.
  int bankContaining(int64_t addr) const;
};

/// A buffer the fit chooses an address for. `banks` are the banks it must
/// start in, empty when any will do.
struct FitBuffer {
  int64_t size;
  int64_t alignBytes;
  bool initialized = false;
  llvm::SmallVector<int, 2> banks;
};

struct FitPlacement {
  int64_t addr;
  int bank;
};

/// What the search cost, for --mlir-pass-statistics.
struct FitStats {
  int64_t backtracks = 0; // placements undone because a later buffer had none
  int64_t tilesSearched = 0; // tiles needing more than the ranked first try
  int64_t budgetExhausted = 0;
  int64_t tilesCompacted = 0; // placed only by the exhaustive search
};

struct FitResult {
  bool solved = false;
  /// The exhaustive search ran out of budget with arrangements untried.
  bool exhausted = false;
  /// On failure, the position in the order of the buffer the search came
  /// closest to placing last.
  std::optional<size_t> failed;
  /// Placements by position in the order: all of them on success, the
  /// closest partial layout on failure.
  llvm::SmallVector<std::pair<size_t, FitPlacement>> placed;
};

/// The banks a buffer's memory space pins it to, empty when it pins none.
llvm::SmallVector<int, 2> banksFromMemorySpace(mlir::Type type);
llvm::SmallVector<int, 2> banksOfMemorySpace(int64_t space);

/// The alignment, in bytes, a buffer of `size` is placed at.
int64_t fitAlignBytes(bool aligned, uint32_t tileAlignBitWidth,
                      uint32_t maxVecAlignBits, int64_t size,
                      int64_t measuredAlign);

/// The order buffers are fitted in: fewest allowed banks first, then largest
/// first, so a small buffer does not split the run a large one needs.
llvm::SmallVector<unsigned> fitOrder(llvm::ArrayRef<FitBuffer> buffers);

/// aie-assign-buffer-addresses' search for `buffers`, given in fit order, on
/// top of what `occupancy` holds: ranked depth-first, then exhaustive, each
/// within `budget` placements. `occupancy` ends holding `placed`.
FitResult fitTileBuffers(const TileMemoryLayout &layout,
                         llvm::ArrayRef<FitBuffer> buffers,
                         MemoryOccupancy &occupancy, int64_t budget,
                         FitStats &stats);

/// aie-assign-buffer-addresses' default placement budget.
constexpr int64_t kDefaultFitBudget = 100000;

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIETILEMEMORYFIT_H
