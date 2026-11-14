//===- AIEDMAChannelAnalysis.h ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <optional>

namespace xilinx::AIE {

/// Which DMA channels of each tile are already spoken for, so that a channel is
/// handed out at most once across everything that programs one.
class DMAChannelAnalysis {
  /// Fully resolved tile aliases name the same hardware resources. Unresolved
  /// tiles retain their SSA identity until placement establishes co-location.
  mlir::DenseMap<std::pair<int, int>, mlir::Value> tilesByCoordinate;
  mlir::Value getTileKey(mlir::Value tile);

  /// Keep the reserving operation so diagnostics retain its MLIR location.
  mlir::DenseMap<std::tuple<mlir::Value, DMAChannelDir, int>, mlir::Operation *>
      usedChannels;
  /// DMA channels an explicit flow routes. First-free assignment skips them;
  /// a pinned request may still claim one.
  mlir::DenseMap<std::tuple<mlir::Value, DMAChannelDir, int>, mlir::Operation *>
      streamedChannels;
  /// Channels whose claimants all share them. Several packet-switched sources
  /// coexist on one; a circuit claim is exclusive.
  mlir::DenseSet<std::tuple<mlir::Value, DMAChannelDir, int>> sharedChannels;
  mlir::DenseMap<std::tuple<mlir::Value, DMAChannelDir, int>, mlir::Operation *>
      shareableChannels;
  /// The first claimant of a core stream port, and whether only packets use
  /// it; circuit occupancy is exclusive.
  struct StreamClaim {
    mlir::Operation *owner;
    bool packet;
  };
  mlir::DenseMap<std::tuple<mlir::Value, DMAChannelDir, int>, StreamClaim>
      usedStreams;

public:
  /// A channel claimed by index: by a DMA program start, a runtime task or a
  /// shim allocation (`owner`), on `tile`.
  struct ChannelClaim {
    ChannelClaim(mlir::Value tile, DMAChannelDir dir, int channel,
                 mlir::Operation *owner)
        : tile(tile), dir(dir), channel(channel), owner(owner) {}
    mlir::Value tile;
    DMAChannelDir dir;
    int channel;
    mlir::Operation *owner;
  };

private:
  /// Every claim that took a channel, in the order they were made.
  llvm::SmallVector<ChannelClaim> claims;
  void claim(mlir::Value tile, DMAChannelDir dir, int channel,
             mlir::Operation *owner);

public:
  DMAChannelAnalysis(DeviceOp &device);

  /// The channels taken by index rather than through a route endpoint, in
  /// the order the claims were made. A shareable one may appear more than
  /// once.
  llvm::ArrayRef<ChannelClaim> getChannelClaims() const { return claims; }

  /// Exclusive upper bound on channel indices eligible for this transfer.
  static int getDMAChannelLimit(TileLike tile, DMAChannelDir dir,
                                bool requiresAdjacentTileAccessChannels);

  /// Next free channel of `tile` in `dir`, or -1 when the tile has none left.
  /// A channel an explicit flow routes is not free.
  /// A channel reaching an adjacent MemTile's memory must come from the
  /// target's restricted range at every compatible physical position.
  int getDMAChannelIndex(TileLike tile, DMAChannelDir dir,
                         bool requiresAdjacentTileAccessChannels,
                         mlir::Operation *owner = nullptr);

  /// Whether first-free assignment could hand out `channel`: nothing reserves
  /// it and no explicit flow routes it.
  bool isChannelFree(TileLike tile, DMAChannelDir dir, int channel);

  /// Claim `channel` for (`tile`, `dir`) so first-free assignment cannot take
  /// it. Returns the channel, or -1 when it is out of range or already
  /// claimed; the caller reports, since it knows which endpoint asked.
  /// `shareable` claimants join a channel other shareable ones hold.
  int reservePinnedChannel(TileLike tile, DMAChannelDir dir, int channel,
                           mlir::Operation *owner = nullptr,
                           bool shareable = false);

  /// Operation reserving or routing this channel; null if none.
  mlir::Operation *getDMAChannelOwner(TileLike tile, DMAChannelDir dir,
                                      int channel);

  /// A DMA end of one tile: its direction, the BD ids it takes, the
  /// exclusive upper bound on the channels it may take, and the channel it
  /// pins, if any, which every end pinning it shares.
  struct BDPlanEnd {
    BDPlanEnd(DMAChannelDir dir, int64_t bds, int limit,
              std::optional<int> pinned)
        : dir(dir), bds(bds), limit(limit), pinned(pinned) {}
    DMAChannelDir dir;
    int64_t bds;
    int limit;
    std::optional<int> pinned;
  };

  /// Channels among `0..numChannels-1` at (`col`, `row`), grouped by whichever
  /// BD ids any two in a group share, transitively.
  struct BDGroup {
    llvm::BitVector ids;
    llvm::SmallVector<int> channels;
  };
  static llvm::SmallVector<BDGroup> bdGroups(const AIETargetModel &target,
                                             int col, int row, int numChannels);

  /// A channel of the tile at (`col`, `row`) for each of `ends`: its pinned
  /// one, else one `free` leaves that no other end of its direction takes;
  /// and, for every set of channels sharing BD ids, the ends on them plus the
  /// ids `spent` on each channel within those ids fit those ids. Nothing when
  /// no assignment fits.
  static std::optional<llvm::SmallVector<int>>
  planBDChannels(const AIETargetModel &target, int col, int row,
                 llvm::ArrayRef<BDPlanEnd> ends,
                 llvm::function_ref<bool(DMAChannelDir, int)> free,
                 llvm::function_ref<int64_t(int)> spent);

  /// A pool whose ends a BD plan weighs: how many objects it holds, the bytes
  /// of one, and whether it can be walked whole, its ends then taking one BD
  /// id each.
  struct BDWalkPool {
    int64_t depth;
    int64_t objectBytes;
    bool walkable;
  };
  /// walkSetsThatFit's result; `channelShort` when the ends do not fit the
  /// tile's channels even taking no BD ids, `fits` false when no set does.
  struct BDWalkSets {
    llvm::SmallVector<llvm::SmallVector<unsigned>> sets;
    /// The bytes each set walks whole.
    llvm::SmallVector<int64_t> bytes;
    bool channelShort = false;
    bool fits = true;
  };

  /// The walkable pool not yet `whole` to walk next: the deepest, then the one
  /// with the smallest objects, then the one whose ends take the most BD ids;
  /// `pools` order decides only between pools alike in all three. Nothing
  /// when none is left.
  static std::optional<unsigned>
  nextPoolToWalk(llvm::ArrayRef<BDPlanEnd> ends,
                 llvm::ArrayRef<unsigned> poolOf,
                 llvm::ArrayRef<BDWalkPool> pools, llvm::ArrayRef<bool> whole);

  /// Every set of walkable `pools` whose walk lets planBDChannels fit `ends`
  /// (`poolOf` gives each end's pool), ordered by the bytes the set walks
  /// whole, then its size, then its members in nextPoolToWalk's order; at
  /// most `maxSets` of them, cheapest first (unbounded by default). Empty
  /// when the ends fit unwalked. At most 12 pools are walkable: each draws a
  /// channel of its own, and more ends than channels make `channelShort`.
  /// aie-objectfifo-allocate and the SA placer both walk by this; the SA only
  /// ever takes the first set, so it should pass `maxSets=1`.
  static BDWalkSets
  walkSetsThatFit(const AIETargetModel &target, int col, int row,
                  llvm::ArrayRef<BDPlanEnd> ends,
                  llvm::ArrayRef<unsigned> poolOf,
                  llvm::ArrayRef<BDWalkPool> pools,
                  llvm::function_ref<bool(DMAChannelDir, int)> free,
                  llvm::function_ref<int64_t(int)> spent,
                  unsigned maxSets = std::numeric_limits<unsigned>::max());

  /// One walkSetsThatFit set per mem tile of a spill component
  /// (memTileSpillComponents), by index; `placed` when its buffers place,
  /// `stopped` when the search gave up at `maxTries` tries.
  struct WalkChoice {
    llvm::SmallVector<unsigned> sets;
    bool placed = false;
    bool stopped = false;
  };
  /// Tries the choices of one set per tile, `bytes[tile]` giving each of its
  /// sets' bytes walked whole, in order of total bytes walked, then the
  /// tiles' set indices in turn, until `places` (the caller's buffer search
  /// over the component) accepts one; the first choice when none does.
  static WalkChoice
  searchWalkProduct(llvm::ArrayRef<llvm::SmallVector<int64_t>> bytes,
                    llvm::function_ref<bool(llvm::ArrayRef<unsigned>)> places,
                    unsigned maxTries);
  static constexpr unsigned kMaxWalkPlacements = 1024;

  /// Blocks of the BD chain `start` begins, following next_bd.
  static llvm::SmallVector<mlir::Block *> chainOf(DMAStartOp start);
  static int64_t countBDs(DMAStartOp start);

  /// The BD ids tiles' own programs hold before objectFIFOs take theirs: the
  /// chains DMA programs start by index, on their channels, and each id a
  /// runtime sequence pins (AIEX::forEachRuntimePinnedBdId), once per tile, on
  /// the lowest channel up to its own reaching it (aie-reserve-runtime-bd-ids
  /// keeps static BDs off those).
  class TakenBDs {
  public:
    TakenBDs() = default;
    explicit TakenBDs(DeviceOp device);
    /// By channel, the ids held at (`col`, `row`) by `tiles` and by every
    /// tile with those coordinates.
    llvm::DenseMap<int, int64_t>
    at(const AIETargetModel &target, int col, int row,
       llvm::ArrayRef<mlir::Operation *> tiles = {}) const;
    /// Coordinates of the placed tiles that hold BD ids.
    llvm::SmallVector<std::pair<int, int>> positions() const;

  private:
    struct Pin {
      unsigned order;
      std::optional<int> channel;
    };
    llvm::MapVector<mlir::Operation *, llvm::DenseMap<int, int64_t>> fixed;
    llvm::MapVector<mlir::Operation *, llvm::DenseMap<int32_t, Pin>> pinned;
  };

  /// Claim a raw stream port for `user`, reporting on it when the port is
  /// taken; packet users may share it with other packets.
  mlir::LogicalResult checkAIEStreamIndex(TileLike tile, DMAChannel chan,
                                          mlir::Operation *user,
                                          bool packet = false);
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H
