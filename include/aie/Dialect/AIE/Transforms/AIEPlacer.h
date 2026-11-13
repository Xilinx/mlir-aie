//===- AIEPlacer.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PLACER_H
#define AIE_PLACER_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIE/Transforms/AIEBufferAllocation.h"
#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"
#include "aie/Dialect/AIE/Transforms/AIELockPlan.h"
#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "llvm/ADT/MapVector.h"

#include <array>
#include <deque>
#include <list>
#include <map>
#include <random>
#include <tuple>
#include <unordered_map>

namespace xilinx::AIE {

/// A map that keeps its `capacity` most recently used entries.
template <typename Key, typename Value>
class LRUCache {
public:
  explicit LRUCache(size_t capacity) : capacity(capacity) {}
  const Value *find(const Key &key) {
    auto it = index.find(key);
    if (it == index.end())
      return nullptr;
    order.splice(order.begin(), order, it->second);
    return &it->second->second;
  }
  void clear() {
    order.clear();
    index.clear();
  }
  void insert(const Key &key, Value value) {
    order.emplace_front(key, std::move(value));
    index[key] = order.begin();
    if (index.size() > capacity) {
      index.erase(order.back().first);
      order.pop_back();
    }
  }

private:
  size_t capacity;
  std::list<std::pair<Key, Value>> order;
  std::map<Key, typename std::list<std::pair<Key, Value>>::iterator> index;
};

/// Placement algorithm type for pass option
enum class PlacerType { SequentialPlacer, SAPlacer };

/// Get DMA channel capacity (maxIn, maxOut) for a tile position: its S2MM and
/// MM2S channel counts.
inline std::pair<int, int> getDMACapacity(const AIETargetModel &tm,
                                          TileID tile) {
  return {tm.getNumDMAChannels(tile.col, tile.row, DMAChannelDir::S2MM),
          tm.getNumDMAChannels(tile.col, tile.row, DMAChannelDir::MM2S)};
}

// Streams through the North and South ports of a tile's stream switch.
struct StreamDemand {
  int northIn = 0, northOut = 0, southIn = 0, southOut = 0;

  StreamDemand &operator+=(const StreamDemand &other) {
    northIn += other.northIn;
    northOut += other.northOut;
    southIn += other.southIn;
    southOut += other.southOut;
    return *this;
  }
};

// maps logical tile operations to physical coordinates
using PlacementResult = llvm::DenseMap<mlir::Operation *, TileID>;

// Track available tiles and resource usage
struct TileAvailability {
  std::vector<TileID> compTiles;
  std::vector<TileID> nonCompTiles; // Memory and shim tiles

  // O(1) membership shadow of `compTiles` used by SequentialPlacer's
  // satisfiesComputePeer forward-look, which checks whether each of a
  // candidate's physical compute neighbors is still free. Without it the
  // forward-look would do an O(|compTiles|) std::find per direction per
  // candidate per LTO, which scales as O(N * M^2) on large arrays. Must be
  // kept in sync with `compTiles` by any code that mutates the vector
  // (initialize(), limitCoresPerColumn(), removeTile()).
  llvm::DenseSet<TileID> compTilesSet;
  // Core positions a physical aie.tile's own aie.core already holds.
  llvm::DenseSet<TileID> physicalCores;
  // Compute tiles cores-per-col leaves out. A core pinned by column or row
  // outranks the budget and may still take one.
  std::vector<TileID> overBudgetCompTiles;

  llvm::DenseMap<TileID, int> inputChannelsUsed;
  llvm::DenseMap<TileID, int> outputChannelsUsed;
  llvm::DenseMap<TileID, StreamDemand> streamsUsed;
  // The logical tiles placed here whose runtime tasks take BD ids.
  llvm::DenseMap<TileID, llvm::SmallVector<mlir::Operation *>> runtimeBdTiles;

  void removeTile(TileID tile, AIETileType type);
};

// Abstract placer interface
class Placer {
public:
  Placer() = default;
  virtual ~Placer() = default;

  virtual void initialize(DeviceOp device, bool controlOverlay,
                          bool adaptiveOverlay, bool packetSwitched);

  virtual mlir::LogicalResult place(DeviceOp device) = 0;

  virtual llvm::StringRef getName() const = 0;

  std::optional<TileID> getPlacement(mlir::Operation *logicalTile) const {
    auto it = result.find(logicalTile);
    if (it != result.end())
      return it->second;
    return std::nullopt;
  }

  // Buffer allocation redirects: (ObjectFifoCreateOp, delegate TileID)
  // Generated when a core tile's buffers exceed local memory and must
  // spill to a neighbor tile's memory module.
  using AllocateInfo = std::pair<mlir::Operation *, TileID>;
  llvm::SmallVector<AllocateInfo> &getAllocates() { return allocates; }

  // Per-LTO peer edges indexed by either endpoint. `tileToEdges` only
  // indexes `LogicalTileOp` endpoints; `TileOp` peers carry their own
  // coords.
  struct Adjacency {
    llvm::SmallVector<std::pair<TileLike, TileLike>, 4> edges;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
        tileToEdges;

    void addEdge(TileLike first, TileLike second);

    // Convenience for IR walkers: skip if either Value isn't a TileLike.
    void addEdgeFromValues(mlir::Value a, mlir::Value b);

    // True if `op` has at least one edge in this adjacency.
    bool hasEdges(mlir::Operation *op) const {
      return tileToEdges.count(op) != 0;
    }
  };

  // IR operations collected from a DeviceOp for placement.
  struct CollectedOps {
    llvm::SmallVector<LogicalTileOp> logicalTiles;
    llvm::SmallVector<ObjectFifoCreateOp> objectFifos;
    llvm::SmallVector<ObjectFifoLinkOp> objectFifoLinks;
    llvm::SmallVector<ObjectFifoAllocateOp> objectFifoAllocates;
    llvm::SmallVector<CascadeFlowOp> cascadeFlows;
    llvm::SmallVector<FlowOp> flows;
    llvm::SmallVector<PacketFlowOp> pktFlows;
    llvm::SmallVector<RouteOp> routes;
    // The device's top-level ops by StructuralGraph::rank, for ties.
    llvm::DenseMap<mlir::Operation *, unsigned> rank;
  };

  // Collect all placement-relevant operations from the device.
  static CollectedOps collectOperations(DeviceOp device);

  // The (input, output) DMA channels each tile's flows, packet flows, route
  // endpoints and channel claims (DMAChannelAnalysis's: program starts,
  // runtime tasks, shim allocations) take: one per endpoint however many
  // routes name it, and one per channel named by index. Tiles with both
  // coordinates are keyed by position. With `named`, the (output, index)
  // channels named by index, by tile; with `keys`, the tile each tile's
  // demand is keyed by.
  static llvm::MapVector<mlir::Operation *, std::pair<int, int>>
  flowChannelDemand(
      llvm::ArrayRef<FlowOp> flows, llvm::ArrayRef<PacketFlowOp> pktFlows,
      llvm::ArrayRef<RouteOp> routes,
      llvm::ArrayRef<DMAChannelAnalysis::ChannelClaim> claims,
      llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::pair<int, int>>>
          *named = nullptr,
      llvm::DenseMap<mlir::Operation *, mlir::Operation *> *keys = nullptr);

  // Check if a logical tile's col/row constraints are satisfied at a position.
  static bool satisfiesConstraints(mlir::Operation *tile, TileID pos);

  // Adjacency builders: pure IR walkers with no placer-specific state.

  // Edge: (cascade source, dest).
  static Adjacency
  buildCascadeAdjacency(llvm::ArrayRef<CascadeFlowOp> cascadeFlows);

  // Edge: (producer LTO, consumer LTO) for every ObjectFifo whose producer
  // AND consumer are CoreTile LTOs.
  static Adjacency
  buildComputePeerAdjacency(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Edge: (producer, consumer) per forcesSharedMemory() ObjectFifo with one
  // consumer.
  static Adjacency
  buildSharedMemAdjacency(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Edge: (producer, consumer_i) per fifo.
  static Adjacency
  buildObjectFifoAdjacency(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Edge: (src, dst) per `aie.flow`; cross-product per `aie.packet_flow`.
  static Adjacency buildFlowAdjacency(llvm::ArrayRef<FlowOp> flows,
                                      llvm::ArrayRef<PacketFlowOp> pktFlows);

  // A shim tile op's ends on one side, as ShimTransferSpans::plan reads them.
  struct ShimSide {
    llvm::SmallVector<mlir::StringAttr> ends;
    llvm::SmallVector<std::pair<int, llvm::SmallVector<mlir::StringAttr>>>
        shared;
    llvm::SmallVector<int> pinned;
  };
  // Each shim tile op's sides, [0] receiving and [1] sending.
  using ShareableShimEnds =
      llvm::DenseMap<mlir::Operation *, std::array<ShimSide, 2>>;
  // With `routeKeys` (flowChannelDemand's `keys`), a route endpoint's end
  // goes to the tile its channel demand is keyed by.
  static ShareableShimEnds collectShareableShimEnds(
      DeviceOp device, llvm::ArrayRef<ObjectFifoCreateOp> objectFifos,
      const llvm::DenseMap<mlir::Operation *, mlir::Operation *> &routeKeys =
          {});
  // The channels `sides` save in one direction below one per end, per
  // ShimTransferSpans::plan, with `extra` ends joining and objectFIFO ends
  // pinning `pinnedHere` there.
  static int shimChannelsSaved(const ShimTransferSpans &spans,
                               llvm::ArrayRef<const ShimSide *> sides,
                               llvm::ArrayRef<mlir::StringAttr> extra,
                               llvm::ArrayRef<int> pinnedHere);

  // The objectFIFO ends that pin one DMA channel of one tile share it, as
  // objectFIFO allocation lets them: sending ends off a shim or off a core
  // that pushes its tasks, told apart by packet id, and receiving ends that
  // take objects out of order.
  class PinnedChannels {
  public:
    // The end of `fifo` (its producer's when `consumer` is empty) on `tile`
    // pinning `channel`. `movesByDma`: the fifo moves by DMA wherever its
    // ends are placed (dmaReason(), or an end off a core tile); otherwise
    // neighbouring cores' shared memory may carry it, and its pin holds only
    // where it does not.
    struct End {
      ObjectFifoCreateOp fifo;
      std::optional<unsigned> consumer;
      mlir::Operation *tile;
      bool sends;
      int channel;
      bool movesByDma;
    };
    PinnedChannels() = default;
    // With `packetSwitched`, an end asking for no packet header gets one.
    PinnedChannels(const AIETargetModel &targetModel, bool packetSwitched)
        : targetModel(&targetModel), packetSwitched(packetSwitched) {}
    // The end of `fifo` that pins a DMA channel, if it does.
    static std::optional<End> endOf(ObjectFifoCreateOp fifo,
                                    std::optional<unsigned> consumer);
    // Why `a` and `b`, pinning one channel of one tile, cannot share it;
    // nothing when they can.
    std::optional<std::string> whyApart(End a, End b) const;
    // The channel `end` shares with the ends of its own tile op that took it
    // before, numbered by the first. An end that cannot share it with one
    // that, like it, moves by DMA wherever placed is diagnosed: its tile
    // holds both wherever it goes.
    unsigned share(End end);
    // Whether share() diagnosed an end, which fails the placement.
    bool failed() const { return diagnosed; }
    // Whether `end` may take turns on the channel of `claim`, as allocation
    // lets a pinned end: a shim allocation whose aie.shareable_with names it,
    // when it sends packet-switched or receives out of order.
    bool sharesClaim(End end,
                     const DMAChannelAnalysis::ChannelClaim &claim) const;

  private:
    const AIETargetModel *targetModel = nullptr;
    bool packetSwitched = false;
    bool diagnosed = false;
    llvm::DenseMap<std::tuple<mlir::Operation *, int, int>,
                   std::pair<llvm::SmallVector<End, 2>, unsigned>>
        taken;
  };

  // The pinned ends at each position, where the ends of tiles placed together
  // meet: one channel serves those that may share it, and two of different
  // tile ops that may not clash there. Two of one tile op clash wherever it
  // goes when both move by DMA wherever placed, which PinnedChannels::share
  // reports; otherwise they clash only where both are placed.
  class PinnedChannelUse {
  public:
    PinnedChannelUse() = default;
    explicit PinnedChannelUse(const PinnedChannels &rule) : rule(&rule) {}
    // Adds (`sign` 1) or removes (-1) `end` at `at`. Returns `sign` when the
    // end takes or frees its channel there, 0 when other ends hold it too.
    int place(TileID at, const PinnedChannels::End &end, int sign);
    // The channels at `at` holding two ends that cannot share them, or
    // claimed by two tile ops.
    int clashes(TileID at) const;
    // The (input, output) channels `ends`, one tile op's, pin that `at`
    // holds already; nothing when one of them clashes there.
    std::optional<std::pair<int, int>>
    heldAt(TileID at, llvm::ArrayRef<PinnedChannels::End> ends) const;
    // An end at `at` that `end` cannot share its channel with, if any.
    std::optional<PinnedChannels::End>
    clashWith(TileID at, const PinnedChannels::End &end) const;
    // The sending ends at `at`, by the MM2S channel they pin.
    llvm::DenseMap<int, llvm::SmallVector<PinnedChannels::End, 2>>
    sendingAt(TileID at) const;
    // Adds (`sign` 1) or removes (-1) `claim` at `at`. A claimed channel is
    // held from every pinned end there but those sharesClaim() lets share it.
    void claim(TileID at, const DMAChannelAnalysis::ChannelClaim &claim,
               int sign);
    // The claim at `at` that holds `end`'s channel from it, if any.
    std::optional<DMAChannelAnalysis::ChannelClaim>
    claimWith(TileID at, const PinnedChannels::End &end) const;
    // The channels ends at `at` pin in one direction.
    llvm::SmallVector<int> channelsAt(TileID at, bool sends) const;

  private:
    const PinnedChannels *rule = nullptr;
    std::map<std::tuple<int, int, int, int>,
             llvm::SmallVector<PinnedChannels::End, 2>>
        placed;
    std::map<std::tuple<int, int, int, int>,
             llvm::SmallVector<DMAChannelAnalysis::ChannelClaim, 1>>
        claimed;
  };

  // The objectFIFO end a runtime sequence names `symbol` for, before split:
  // the fifo itself, sending when its producer is a shim tile.
  static std::optional<ShimTransferSpans::End>
  shimEndOf(DeviceOp device, mlir::StringAttr symbol);

  // The gates objectFIFO allocation will share receiving shim ends by: the
  // #aie.consumed_by proofs aie-objectfifo-verify finds on a copy of the
  // design with every fifo between two tiles moved by DMA. Placing a fifo
  // on shared memory instead keeps each such proof (see the commit that adds
  // this), so these never promise a turn allocation refuses. Empty when the
  // copy does not verify.
  static llvm::DenseMap<mlir::StringAttr,
                        llvm::SmallVector<ShimTransferSpans::Gate>>
  provenShimGates(DeviceOp device);

  // The BD ids a shim tile's DMAs take, as aie-assign-bd-ids and
  // aie-assign-runtime-sequence-bd-ids hand them out: static BDs around the ids
  // runtime tasks and copies pin; each sequence's tasks their pinned ids, then
  // the lowest free (head aligned, descriptors split as
  // aie-decompose-large-dma-bd splits them), held from configure to the first
  // await or free; a copy holds its id to the sequence's end. Reclaim,
  // restarts, already-taken tokens, ids a copy forbids, BD chains and aiex.run
  // callees are not followed, so the count can err either way and the pass
  // checks exactly. Placed shims count where they sit.
  class RuntimeBdIds {
  public:
    // The most ids a shim's DMAs hold at once on a tile of their own, and
    // whether any pins an id or aligns its head.
    struct Peak {
      int live = 0;
      bool constrained = false;
    };
    RuntimeBdIds() = default;
    RuntimeBdIds(DeviceOp device, const AIETargetModel &targetModel);

    bool uses(mlir::Operation *logicalTile) const {
      return peaks.contains(logicalTile);
    }
    const llvm::DenseMap<mlir::Operation *, Peak> &users() const {
      return peaks;
    }

    // The ids the DMAs of `onTile` and of the shim already placed at `tile`
    // cannot get when they share it.
    int shortfall(llvm::ArrayRef<mlir::Operation *> onTile, TileID tile) const;

  private:
    struct Task {
      mlir::Operation *owner;
      // One entry per descriptor after the split: its pinned id, if any.
      llvm::SmallVector<std::optional<uint32_t>> bdIds;
      uint32_t headAlignment;
      // An aiex.npu.dma_memcpy_nd: its one id stays held to the end of the
      // sequence.
      bool copy = false;
    };
    struct StaticBds {
      llvm::SmallVector<uint32_t> pinned;
      int unpinned = 0;
    };
    llvm::SmallVector<Task> tasks;
    // Per runtime sequence, (task, configured or released) in program order.
    llvm::SmallVector<llvm::SmallVector<std::pair<unsigned, bool>>> sequences;
    llvm::DenseMap<mlir::Operation *, StaticBds> statics;
    llvm::DenseMap<mlir::Operation *, Peak> peaks;
    llvm::DenseMap<TileID, mlir::Operation *> placed;
    const AIETargetModel *targetModel = nullptr;
    // Every shim tile lays out its BD ids alike, so the answer depends only on
    // which shims share one.
    mutable std::map<std::vector<mlir::Operation *>, int> shortfallOf;
  };

protected:
  // Whether the control overlay takes an MM2S channel on `tile`.
  int overlayChannels(TileID tile) const {
    return controlOverlay && tile.row == 0 &&
           targetModel->getTileType(tile.col, tile.row) ==
               AIETileType::ShimNOCTile;
  }
  // The control overlay's MM2S channel on shim tile `at`, where
  // aie-generate-column-control-overlay puts it: channel 0, or with
  // `adaptiveOverlay` any one, that no circuit-switched end pins and that
  // overlayUses of the tile ops at `at` (`tilesHere`) finds no circuit on,
  // fewest packet-switched ends first. The pinned ends there share it when
  // ShimTransferSpans finds each apart from the overlay; otherwise allocation
  // refuses them, and the overlay does not `fit`. `extra` are a tile op's
  // pinned ends about to join `pins`'.
  struct OverlayFit {
    bool fits = true;
    // The MM2S channels it adds: none on a pinned channel it shares.
    int channels = 0;
    // Whether it holds a channel of its own, which unpinned ends may share.
    bool own = false;
  };
  OverlayFit overlayAt(TileID at, const PinnedChannelUse &pins,
                       llvm::ArrayRef<PinnedChannels::End> extra,
                       llvm::ArrayRef<mlir::Operation *> tilesHere,
                       const ShimTransferSpans *spans) const;
  // The shim MM2S channels each tile op's ends name, as the overlay's pass
  // weighs them: `device`'s, and those of the other devices the overlay
  // serves (its target's), at the tiles they sit on now.
  ShimMM2SUses overlayUses;
  void collectOverlayUses(DeviceOp device);
  // The tile ops overlayUses names.
  llvm::SmallVector<mlir::Operation *> overlayTileOps() const;

  // flowChannelDemand's `named`, by tile op.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::pair<int, int>>>
      namedChannels;
  // A tile op among `others` naming a channel `tile` names too, with that
  // channel. Two such tile ops cannot share a physical tile: their programs
  // or flows would drive one channel.
  std::optional<std::pair<mlir::Operation *, std::pair<int, int>>>
  namedClash(mlir::Operation *tile,
             llvm::ArrayRef<mlir::Operation *> others) const;
  // Reports that `tile` can take no tile of its type, because the one at
  // `at` with room for it holds `clash`, a tile op naming a channel it names.
  mlir::LogicalResult
  emitNamedClash(mlir::Operation *tile, TileID at,
                 std::pair<mlir::Operation *, std::pair<int, int>> clash) const;

  PlacementResult result;
  llvm::SmallVector<AllocateInfo> allocates;
  const AIETargetModel *targetModel = nullptr;
  bool controlOverlay = false;
  bool adaptiveOverlay = true;
  // Routes asking for no packet header carry one (packet-sw-objFifos).
  bool packetSwitched = false;
  TileAvailability availability;
};

// Invoke `fn(peer, thisIsFirst)` for each edge in `adjacency` that
// mentions `op`. `peer` is the OTHER endpoint of the edge; `thisIsFirst`
// is true when `op` is `edge.first`.
template <typename F>
inline void forEachPeer(mlir::Operation *op, const Placer::Adjacency &adjacency,
                        F &&fn) {
  auto it = adjacency.tileToEdges.find(op);
  if (it == adjacency.tileToEdges.end())
    return;
  for (unsigned idx : it->second) {
    auto [first, second] = adjacency.edges[idx];
    bool thisIsFirst = first.getOperation() == op;
    TileLike peer = thisIsFirst ? second : first;
    fn(peer, thisIsFirst);
  }
}

// Whether two compute tiles can share one memory module: unlike buffer
// ownership (where only the consumer must reach the owner's module),
// sharing has no owner/consumer asymmetry, so either tile reaching the
// other's module is enough. isLegalMemAffinity itself is directional (see
// AIETargetModel::getSharedMemory's None/First/Second/Either), so compute-
// peer neighbor-ness -- what `--aie-objectfifo-split`'s auto transport
// actually tests -- has to check both.
inline bool isSharedComputeMemory(const AIETargetModel &targetModel, TileID a,
                                  TileID b) {
  return targetModel.isLegalMemAffinity(a.col, a.row, b.col, b.row) ||
         targetModel.isLegalMemAffinity(b.col, b.row, a.col, a.row);
}

// Invoke `fn(TileID)` on every CoreTile neighbor of `at` that can share
// compute-peer memory with it per the target model.
template <typename F>
inline void forEachMemAffinityNeighbor(const AIETargetModel &targetModel,
                                       TileID at, F &&fn) {
  for (auto [dc, dr] :
       {std::pair{0, -1}, std::pair{0, 1}, std::pair{-1, 0}, std::pair{1, 0}}) {
    int nc = at.col + dc;
    int nr = at.row + dr;
    if (nc < 0 || nc >= targetModel.columns())
      continue;
    if (nr < 0 || nr >= targetModel.rows())
      continue;
    if (targetModel.getTileType(nc, nr) != AIETileType::CoreTile)
      continue;
    if (!isSharedComputeMemory(targetModel, at, TileID{nc, nr}))
      continue;
    fn(TileID{nc, nr});
  }
}

// Sequential placement algorithm
//
// Greedy, single-pass placer that maps each `aie.logical_tile` (LTO) to a
// physical (col, row). Four phases:
//
//   1. Collect LogicalTileOp / ObjectFifo* / Cascade* / Flow* / PacketFlow* /
//      Route from the device.
//   2. Build per-LTO constraints: buffer/cascade/compute-peer adjacencies,
//      channel requirements, and needNeighborIn/Out (the minimum number of
//      compute peers per LTO that MUST land on a shared-L1 neighbor to fit
//      the per-tile DMA budget). Group the compute LTOs by the non-core
//      tile each shares the most flows with (buildCoreGroups).
//   3. Place compute tiles. Make each core group contiguous, then sort by
//      constraint level (pinned -> partial -> unpinned) and by descending
//      placementPriority. For each LTO: validate buffer/cascade/compute-peer
//      constraints and channel budget, then claim the tile, trying first
//      the columns that keep its group together. Unpinned candidates are
//      tried in two passes: first the slots with column stream capacity and
//      not soft-reserved for another demand-bearing LTO's peers, then a
//      fallback accepting overloaded columns and reserved slots. Placing a
//      demand-bearing LTO soft-reserves its mem-affinity neighbor slots for
//      that LTO's compute peers.
//   4. Place each remaining non-core (mem/shim) LTO near the column
//      centroid of its placed-core peers. With mergeLogicalTiles == true
//      (default) several non-core LTOs may share one physical tile when
//      DMA capacity allows, preferring tiles whose North/South stream ports
//      and runtime-sequence BDs also fit; with mergeLogicalTiles == false
//      each non-core LTO claims its own physical tile.
//
// Greedy: once a tile is chosen the decision is final, with no backtracking.
// If a later LTO's constraints become unsatisfiable given prior placements,
// the placer emits an error pointing at the LTOs the user can manually pin
// to break the deadlock.
//
// Compute-to-compute ObjectFifos consume no DMA channel when their endpoints
// land on shared-L1 neighbor tiles, so they enter the placer as a separate
// compute-peer adjacency rather than as channel demand.
class SequentialPlacer : public Placer {
public:
  SequentialPlacer(std::optional<int> coresPerCol = std::nullopt,
                   bool mergeLogicalTiles = true,
                   bool spreadUnanchoredTiles = false)
      : coresPerCol(coresPerCol), mergeLogicalTiles(mergeLogicalTiles),
        spreadUnanchoredTiles(spreadUnanchoredTiles) {}

  void initialize(DeviceOp device, bool controlOverlay, bool adaptiveOverlay,
                  bool packetSwitched) override;

  mlir::LogicalResult place(DeviceOp device) override;

  llvm::StringRef getName() const override { return "sequential_placer"; }

private:
  std::optional<int> coresPerCol;
  bool mergeLogicalTiles;
  // When a non-core LTO has no CoreTile peers, rank candidates by DMA load
  // instead of by distance to column 0. See placeNonCoreTileByCentroid.
  bool spreadUnanchoredTiles;
  // Physical tiles already assigned to a non-core aie.logical_tile. Used
  // only when mergeLogicalTiles == false to forbid mapping a second
  // non-core aie.logical_tile onto a tile that already hosts one.
  llvm::DenseSet<TileID> assignedNonCoreTiles;
  int deviceCoresPerCol = 0; // Actual cores per column in device
  // The compute tiles cores-per-col leaves to unpinned logical cores.
  int placeableCores = 0;

  // Shim channels are counted one per end until a shim tile runs out; then
  // ends never in flight together count once, as allocation will share them.
  std::optional<ShimTransferSpans> shimSpans;
  ShareableShimEnds shareableShimEnds;
  llvm::DenseMap<TileID, llvm::SmallVector<mlir::Operation *>> shimTilesAt;
  // The tile ops placed at each position, whose overlayUses count there.
  llvm::DenseMap<TileID, llvm::SmallVector<mlir::Operation *>> overlayTilesAt;
  bool fitsSharingShim(TileID tile, mlir::Operation *logicalTile,
                       std::pair<int, int> required);
  void recordShimEnds(TileID tile, mlir::Operation *logicalTile);

  // Counts once in `requirements` each group of fifos between a core and one
  // mem tile that allocation's planTurnSharing will put on one channel.
  void countTurns(
      DeviceOp device, llvm::ArrayRef<ObjectFifoCreateOp> objectFifos,
      llvm::DenseMap<mlir::Operation *, std::pair<int, int>> &requirements);

  // DMA channel direction selector.
  enum class DmaDir { In, Out };

  // Whether an LTO's placement was pinned by the user (via col/row
  // attributes) or chosen by the placer.
  enum class PlacementOrigin { Pinned, Selected };

  /// Keeps at most `maxCoresPerCol` cores in each column, counting the
  /// `held` positions a physical or pinned core already takes.
  void limitCoresPerColumn(int maxCoresPerCol, int numColumns,
                           const llvm::DenseSet<TileID> &held);

  // targetCol == nullopt means no routing anchor is known, so candidates are
  // ranked by DMA load rather than by distance to a column. `withinBudgets`
  // also requires fitsStreamAndBdBudgets; `apartByName` requires that no
  // tile op there names a channel `logicalTile` names (namedClash).
  std::optional<TileID>
  findTileWithCapacity(std::optional<int> targetCol,
                       llvm::ArrayRef<TileID> tiles, int requiredInputChannels,
                       int requiredOutputChannels, AIETileType requestedType,
                       mlir::Operation *logicalTile, bool withinBudgets,
                       bool apartByName = true);

  // Diagnosis for a failed findTileWithCapacity() call. `findTileWithCapacity`
  // scans every tile of `requestedType` on the device -- `targetCol` orders
  // candidates, it does not exclude any -- so a failure is either (a) no
  // tile of this type has the requested channels free ANYWHERE (device-wide
  // DMA exhaustion: pinning a column cannot create capacity), or (b) at
  // least one tile has them free but is already claimed by a different
  // non-core logical tile (only possible under merge-logical-tiles=false).
  // `capacityExistsSomewhere` selects between the two; the totals let the
  // caller report the real budget instead of guessing at it.
  struct CapacityDiagnosis {
    bool capacityExistsSomewhere = false;
    int tilesOfType = 0;
    int inUsed = 0, inMax = 0, outUsed = 0, outMax = 0;
    // Both shortfalls belong to one tile, minimizing their sum.
    int closestShortIn = 0, closestShortOut = 0;
    // Of outUsed, the control overlay's channels; and the tiles where it has
    // none it can take.
    int overlayOut = 0, overlayUnfit = 0;
    // The tiles where a pinned end of the joining tile clashes with one
    // already there, and the first such pair.
    int clashTiles = 0;
    std::optional<std::tuple<TileID, PinnedChannels::End, PinnedChannels::End>>
        clash;
  };
  // `joining` is the logical tile that found no tile, whose pinned ends move
  // the control overlay.
  CapacityDiagnosis diagnoseCapacityExhaustion(AIETileType requestedType,
                                               int requiredInputChannels,
                                               int requiredOutputChannels,
                                               mlir::Operation *joining);

  void updateChannelUsage(TileID tile, DmaDir direction, int numChannels);

  // North/South streams of each tile's ObjectFifos: a memtile has 6
  // DMA inputs but only 4 streams from the core above, and no East/West.
  llvm::DenseMap<mlir::Operation *, StreamDemand> streamDemand;
  // Aggregate traffic between a column's cores and the tiles below them.
  llvm::DenseMap<int, StreamDemand> columnStreamsUsed;
  // Broadcast consumers in one column share a single incoming stream.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      incomingFifos;
  llvm::DenseMap<int, llvm::DenseSet<mlir::Operation *>> columnIncomingFifos;
  void recordColumnStreamUsage(mlir::Operation *tile, int col);
  // Extra broadcast destinations beyond the first, which uses one DMA channel.
  llvm::DenseMap<mlir::Operation *, int> broadcastFanout;
  llvm::DenseMap<TileID, int> broadcastFanoutUsed;
  void buildStreamDemand(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  RuntimeBdIds runtimeBdIds;

  // Pinned objectFIFO ends: the rule, their positions, and each tile op's own.
  PinnedChannels pinnedRule;
  PinnedChannelUse pinsAt;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<PinnedChannels::End>>
      pinnedEnds;
  // DMAChannelAnalysis::getChannelClaims(), grouped by the tile op owning each.
  llvm::DenseMap<mlir::Operation *,
                 llvm::SmallVector<DMAChannelAnalysis::ChannelClaim>>
      claimsOf;

  // True if `tile`'s stream ports and BDs still fit `logicalTile`'s demand on
  // top of what earlier placements use.
  bool fitsStreamAndBdBudgets(mlir::Operation *logicalTile, TileID tile) const;
  // Records the streams, BDs and named DMA channels `logicalTile` takes on
  // `tile`.
  void recordStreamAndBdUsage(mlir::Operation *logicalTile, TileID tile);

  // `joining` is the logical tile being placed on `tile`: its pinned ends and
  // named channels count there, as does its own use of the control overlay.
  bool hasAvailableChannels(TileID tile, int inputChannels, int outputChannels,
                            mlir::Operation *joining = nullptr);
  OverlayFit overlayWith(TileID tile, mlir::Operation *joining);

  // Notes on `diag` the first pinned end of `logicalTile` that clashes at
  // `at` and what it clashes with.
  void notePinClash(mlir::InFlightDiagnostic &diag, LogicalTileOp logicalTile,
                    TileID at);

  mlir::LogicalResult validateAndUpdateChannelUsage(
      LogicalTileOp logicalTile, TileID tile,
      const llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements,
      PlacementOrigin origin);

  llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
  buildChannelRequirements(
      llvm::SmallVector<ObjectFifoCreateOp> &objectFifos,
      llvm::SmallVector<ObjectFifoLinkOp> &objectFifoLinks);

  // Per-place() neighbor-demand maps and soft-reservation table.
  struct PlacementContext {
    const AIETargetModel &targetModel;
    const Adjacency &computePeerAdjacency;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborIn;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborOut;
    // Mutable: each placement of an LTO with neighbor demand reserves
    // its mem-affinity neighbor slots for that LTO's compute peers.
    llvm::DenseMap<TileID, llvm::SmallVector<mlir::Operation *, 2>>
        reservedFor{};

    int neighborDemand(mlir::Operation *op) const {
      return needNeighborIn.lookup(op) + needNeighborOut.lookup(op);
    }
    // Priority = max(self demand, highest peer demand). A peer of a
    // high-fanin Worker rises to the front of the sort so its neighbor
    // slots aren't consumed by unrelated LTOs.
    int placementPriority(mlir::Operation *op) const;
    // Soft-reserve `at`'s mem-affinity compute-tile neighbors for
    // `placedOp`'s peers, if `placedOp` has non-zero neighbor demand.
    void reserveNeighborSlots(mlir::Operation *placedOp, TileID at);
    // True if `candidate` is reserved by any LTO that is NOT a compute
    // peer of `lto` (i.e. would compete for the slot).
    bool isReservedForOther(mlir::Operation *lto, TileID candidate) const;
  };

  // A pairwise-legality adjacency constraint (buffer, cascade, ...)
  // bundled with everything the placer needs to check it and report a
  // useful diagnostic when it fails: the edge set, the predicate, the
  // peer-role label for notes, the short name for the error message,
  // and the educational hint text appended after the peer notes.
  struct AdjacencyKind {
    const Adjacency &adjacency;
    llvm::function_ref<bool(TileID firstPos, TileID secondPos)> pred;
    llvm::function_ref<llvm::StringRef(bool thisIsFirst)> peerLabel;
    llvm::StringRef name;           // e.g. "shared-L1 buffer", "cascade"
    llvm::StringRef constraintHint; // appended as a final attachNote
  };

  // Check `logicalTile` at `tile` against `kind`'s adjacency. On
  // failure, emit a fully-formed diagnostic (error + peer notes +
  // constraint hint) and return failure.
  mlir::LogicalResult enforceAdjacency(LogicalTileOp logicalTile, TileID tile,
                                       const AdjacencyKind &kind);

  // Record that `logicalTile` is placed at `tile`. For CoreTile LTOs
  // this also removes the tile from availability and asks `ctx` to
  // soft-reserve the tile's mem-affinity neighbor slots for the LTO's
  // compute peers. `debugLabel` ("pinned" / "unconstrained") names the
  // placement path in the LLVM_DEBUG trace.
  void recordPlacement(LogicalTileOp logicalTile, TileID tile,
                       PlacementContext &ctx, llvm::StringRef debugLabel);

  // Per-call inputs for `findUnconstrainedCoreCandidate`. Bundles the
  // adjacency + demand state that Phase 3 builds for each candidate
  // filter.
  struct UnpinnedPlacementInputs {
    const Adjacency &bufferAdjacency;
    llvm::function_ref<bool(TileID, TileID)> bufferPred;
    const Adjacency &cascadeAdjacency;
    llvm::function_ref<bool(TileID, TileID)> cascadePred;
    const Adjacency &sharedMemAdjacency;
    llvm::function_ref<bool(TileID, TileID)> sharedMemPred;
    const Adjacency &computePeerAdjacency;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborIn;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborOut;
    llvm::function_ref<bool(mlir::Operation *, TileID)> isReservedForOther;
    // Columns to try first, in order, to keep the LTO with its core group
    // (see buildCoreGroups). Ignored for LTOs with neighbor demand.
    llvm::ArrayRef<int> preferredCols;
  };

  // Outputs of `findUnconstrainedCoreCandidate`. `placement` is set on
  // success; on failure the three boolean flags describe which filter
  // rejected the most candidates so place() can construct an actionable
  // diagnostic.
  struct UnpinnedSearchResult {
    std::optional<TileID> placement;
    bool sawConstraintMatch = false;
    bool allConstraintMatchesFailedAdjacency = true;
    bool computePeerWasCause = false;
    // The first candidate passing every other check where a pin of the LTO
    // clashes with one placed there.
    std::optional<TileID> pinClash;
  };

  // Two-pass candidate search for an unpinned (or partially-pinned)
  // CoreTile LTO. First pass prefers candidates that aren't
  // soft-reserved by an unrelated demand-bearing LTO's peers; second
  // pass falls back to reserved candidates. Each candidate is filtered
  // by buffer adjacency, cascade adjacency, the compute-peer DMA
  // budget and the LTO's pinned channels. Does not mutate placer state.
  UnpinnedSearchResult findUnconstrainedCoreCandidate(
      LogicalTileOp logicalTile, std::optional<int> col, std::optional<int> row,
      const UnpinnedPlacementInputs &inputs, llvm::ArrayRef<TileID> candidates,
      llvm::ArrayRef<TileID> pastBudget);

  // Edge: (consumer LTO, owner tile). Predicate: `isLegalMemAffinity`.
  // SequentialPlacer-only: walks CoreOps and BufferOps, and takes each end of
  // an allocated fifo as a consumer of its delegate tile.
  Adjacency
  buildBufferAdjacency(llvm::ArrayRef<LogicalTileOp> logicalTiles,
                       llvm::ArrayRef<ObjectFifoAllocateOp> allocates);

  // Per-LTO peer-Value lists, one list per flow that mentions the LTO.
  // Built once and shared so each centroid lookup is O(#flows on the LTO).
  struct FlowMembership {
    llvm::DenseMap<mlir::Value,
                   llvm::SmallVector<llvm::SmallVector<mlir::Value>>>
        ltoFlows;
  };

  // Phase 4 driver: for every still-unplaced non-core LTO, place it near
  // the centroid column of its core peers, iterating non-core LTOs in
  // descending channel-demand order so the heaviest consumers get first
  // pick of physical tiles.
  mlir::LogicalResult placeNonCoreLogicalTiles(
      llvm::ArrayRef<LogicalTileOp> logicalTiles,
      const FlowMembership &flowIndex,
      const llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements);

  // Core LTOs grouped by the non-core LTO each shares the most flows with
  // (its hub), members in program order. A core with no non-core peer is a
  // group of its own. Placing a group in one column lets the hub's
  // centroid land on that column.
  struct CoreGroups {
    llvm::DenseMap<mlir::Operation *, unsigned> groupOf;
    llvm::SmallVector<llvm::SmallVector<LogicalTileOp>> members;
  };

  CoreGroups buildCoreGroups(llvm::ArrayRef<LogicalTileOp> logicalTiles,
                             const FlowMembership &flowIndex) const;

  // Columns to try first for `logicalTile`: those already holding members
  // of its group, most members first; for a group's first member, the
  // columns with a free core for every member, in column order. Empty
  // means no preference.
  llvm::SmallVector<int> preferredGroupColumns(LogicalTileOp logicalTile,
                                               const CoreGroups &groups) const;

  FlowMembership
  buildFlowMembership(llvm::ArrayRef<FlowOp> flows,
                      llvm::ArrayRef<PacketFlowOp> pktFlows,
                      llvm::ArrayRef<RouteOp> routes,
                      llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Pick the column that minimizes total routing cost across the LTO's
  // flows. See AIEPlacer.cpp for the per-flow cost formulas and tiebreak.
  // Returns nullopt when the LTO has no CoreTile peer to measure against.
  std::optional<int> computeCentroidColumn(LogicalTileOp logicalTile,
                                           const FlowMembership &flowIndex);

  mlir::LogicalResult placeNonCoreTileByCentroid(
      LogicalTileOp logicalTile, const FlowMembership &flowIndex,
      const llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements);

  // Pairwise legality check. `pred(firstPos, secondPos)` is evaluated for
  // every edge mentioning `logicalTile`; unplaced peers defer.
  bool satisfiesAdjacency(
      LogicalTileOp logicalTile, TileID candidate, const Adjacency &adjacency,
      llvm::function_ref<bool(TileID firstPos, TileID secondPos)> pred) const;

  // Compute-peer DMA budget check. Returns true if placing `logicalTile` at
  // `candidate` keeps the compute-peer neighbor demand satisfiable for BOTH
  // the LTO being placed AND every already-placed compute peer of the LTO.
  // A compute LTO with peer demand N (N compute-peer fifos must use shared
  // neighbor memory to fit DMA budget) has slack (totalComputePeers − N): it
  // can afford at most that many non-neighbor compute peers. The check also
  // forward-looks: unplaced peers still need a free physical compute-tile
  // neighbor of `candidate` to land on.
  bool satisfiesComputePeer(
      LogicalTileOp logicalTile, TileID candidate,
      const Adjacency &computePeerAdjacency,
      const llvm::DenseMap<mlir::Operation *, int> &needNeighborIn,
      const llvm::DenseMap<mlir::Operation *, int> &needNeighborOut) const;

  // Compute-peer in/out edge counts for `op`, looked up from a prebuilt
  // `computePeerAdjacency`. Returns {0, 0} if `op` has no entry. Producer
  // is edges[idx].first, consumer is .second (per buildComputePeerAdjacency).
  static std::pair<int, int> totalComputePeers(mlir::Operation *op,
                                               const Adjacency &adjacency);

  // Diagnostic peer notes. `labelPeer(thisIsFirst)` names the peer endpoint
  // role; the attached note reads "<label> peer placed at (col, row)".
  void attachPeerNotes(
      mlir::InFlightDiagnostic &diag, LogicalTileOp logicalTile,
      const Adjacency &adjacency,
      llvm::function_ref<llvm::StringRef(bool thisIsFirst)> labelPeer) const;
};

// Configuration for the SA placer's cost model and schedule.
//
// The SA placer minimizes a total cost with four components:
//
//   totalCost = HPWL + resourcePenalty + cascadePenalty + memoryPressure
//
// 1. HPWL (half-perimeter wire length): sum of bounding-box perimeters
//    across all nets.
//
// 2. Resource penalty (hard, blocks legality): penalizes MemTile buffer
//    overflow, core tile overflow, DMA channel overuse, BD count
//    overuse, and fifo ends that can't reach their allocate's delegate.
//
// 3. Cascade penalty (hard, blocks legality): penalizes cascade put/get
//    pairs that aren't adjacent.
//
// 4. Memory pressure (soft, guides optimization): quadratic penalty when
//    tile utilization exceeds a threshold.
struct SAConfig {
  // HPWL (cost 1)
  int multicastThreshold = 4;  // consumer count above which penalty applies
  int multicastMultiplier = 2; // HPWL multiplier for multicast nets

  // Resource penalty (cost 2)
  int memPenaltyPerKB = 30;       // per KB of unresolved memory overflow
  int dmaPenaltyPerChannel = 20;  // per DMA channel or BD over limit
  int delegateWeightPerDist = 30; // per distance from an allocate's delegate
  // Walk choices one step tries per spill component, short of allocation's
  // DMAChannelAnalysis::kMaxWalkPlacements: a state left unsettled counts as
  // unplaced, never as fitting.
  unsigned walkPlacementsPerStep = 1;

  // Cascade penalty (cost 3)
  int cascadeWeightPerDist = 30; // per Manhattan distance to valid position
  // A shared-memory pair apart, per distance past a neighbour: above a
  // fifo's worth of memory overflow, which a neighbour's spare room absorbs.
  int sharedMemWeightPerDist = 300;

  // Memory pressure (cost 4)
  int pressurePerKB = 3;           // quadratic pressure above threshold
  double pressureThreshold = 0.75; // utilization fraction triggering pressure

  // Adaptive cooling
  double coolFast = 0.995;        // accept ratio > 96%: cool faster
  double coolSteady = 0.998;      // accept ratio > 80%: steady progress
  double coolBalanced = 0.999;    // accept ratio > 15%: balanced exploration
  double coolStuck = 0.9998;      // accept ratio <= 15%: barely cool
  double greedyThreshold = 0.005; // switch to greedy (T=0) below this

  // Schedule sizing
  int minMovesPerIter = 100;
  int maxMovesPerIter = 2000;
  int minMaxIters = 10000;
  int greedyMultiplier = 50; // greedyIters = multiplier * numMovable
  double effort = 1.0;       // scales movesPerIter and greedyIters
  // assign-buffer-addresses' placement-budget, for core tile layouts.
  int64_t placementBudget = kDefaultPlacementBudget;
  double tempScaleEstimate = 10.0; // initTemp cap: scale * estimated T
  double tempScaleCost = 2.0;      // initTemp cap: scale * totalCost

  // Temperature estimation
  int tempSearchIters = 100;         // binary search iterations
  double tempSearchCeiling = 5.0;    // highT = ceiling * maxDelta
  double tempSearchEpsilon = 1e-10;  // zero guard in binary search
  double tempSearchTolerance = 1e-6; // convergence tolerance

  // SA loop intervals
  int debugInterval = 5000;          // iterations between debug log lines
  int fullRecomputeInterval = 10000; // iterations between full cost recompute

  // Move generation
  int maxMoveAttempts = 20; // retries per move before giving up

  // Acceptance tracking
  int windowSize = 100; // sliding window for acceptance ratio

  // ObjectFIFOs whose mem tile pools allocation walked whole.
  llvm::SmallVector<std::string> walkedFifos;
  // The data region each core keeps for its own sections, by the device's
  // core order, where a build measured it, for a core declaring neither
  // data_size nor measured_data_size: its bytes and the alignment its
  // sections start at.
  struct CoreDataCharge {
    CoreDataCharge(int64_t bytes, int64_t alignBytes)
        : bytes(bytes), alignBytes(alignBytes) {}
    int64_t bytes;
    int64_t alignBytes;
  };
  llvm::DenseMap<unsigned, CoreDataCharge> coreDataCharges;
};

// SA temperature schedule with windowed acceptance tracking.
class SASchedule {
public:
  SASchedule() = default;
  SASchedule(double startTemp, int movesPerIter, int maxIters, int greedyIters,
             const SAConfig &cfg = {});

  double getTemperature() const { return temperature; }
  int getIteration() const { return currIteration; }
  int getMovesPerIter() const { return movesPerIter; }
  bool isGreedy() const { return inGreedyStage; }
  bool limitReached() const;
  double getAcceptanceRatio() const;
  void record(bool accepted);
  void cool();

private:
  double temperature = 0.0;
  int currIteration = 0;
  int currGreedyIteration = 0;
  int movesPerIter = 100;
  int maxIters = 5000;
  int greedyIters = 100;
  int windowSize = 100;
  SAConfig config;
  bool inGreedyStage = false;
  std::deque<int> history;
  int acceptCount = 0;
  int rejectCount = 0;

  void trimWindow();
};

// Per-net bounding box state for incremental HPWL updates.
// countAt* tracks how many endpoints sit at each extreme so we can avoid
// full rescans when an endpoint moves away from a non-unique extreme.
struct NetBoundingBox {
  int minCol = 0, maxCol = 0, minRow = 0, maxRow = 0;
  int countAtMinCol = 0, countAtMaxCol = 0;
  int countAtMinRow = 0, countAtMaxRow = 0;
};

struct NetInfo {
  llvm::SmallVector<mlir::Operation *> endpoints; // producer + all consumers
  NetBoundingBox bb;
  bool isMulticast = false; // >4 consumers triggers 2x HPWL penalty
};

// Simulated annealing placement algorithm
//
// Uses HPWL (half-perimeter wire length) cost with incremental bounding-box
// updates. Nets are derived from ObjectFifo producer/consumer connectivity.
// All tile types (compute, mem, shim) participate in SA moves. After SA
// converges, a merge pass collapses mem/shim logical tiles that landed in the
// same column onto a shared physical tile when DMA capacity permits.
class SAPlacer : public Placer {
public:
  explicit SAPlacer(unsigned seed = 0, SAConfig config = {})
      : rngSeed(seed), config(config) {}

  mlir::LogicalResult place(DeviceOp device) override;
  llvm::StringRef getName() const override { return "sa_placer"; }
  int getFinalCost() const { return finalCost; }
  int getFinalLockGroups() const { return finalLockGroups; }
  int getFinalLockPenalty() const { return finalLockPenalty; }
  bool getFinalLockStopped() const { return finalLockStopped; }

private:
  // Configuration
  unsigned rngSeed;
  SAConfig config;
  int finalCost = 0;
  int finalLockGroups = 0, finalLockPenalty = 0;
  bool finalLockStopped = false;
  std::mt19937 rng;
  SASchedule schedule;

  // Collected IR state
  CollectedOps collected;
  llvm::DenseMap<mlir::Operation *, AIETileType> tileTypes;
  llvm::DenseMap<mlir::Operation *, int64_t> staticBufferSizes;
  llvm::DenseMap<mlir::Operation *, int64_t> stackSizes;
  // Locks the design already places on each logical tile, in the order
  // collectOperations ranks the tiles, and on each placed tile; placed tiles
  // take lock groups too.
  llvm::MapVector<mlir::Operation *, int> fixedLocks;
  llvm::DenseMap<TileID, int> placedLocks;
  llvm::SmallVector<TileID> placedTiles;
  // Each fifo's place among the ones its core first acquires, in the core's
  // program order.
  llvm::DenseMap<mlir::Operation *, unsigned> firstAcquire;
  // The lock groups of pools already in the input, read as
  // aie-objectfifo-allocate reads them: each segment without its locks, used
  // by the tiles of the endpoints selecting it (a compute tile's DMA reaching
  // only its own locks).
  struct InputLockGroup {
    mlir::Operation *home;
    llvm::SmallVector<std::pair<mlir::Operation *, bool>, 2> users;
    int locks;
  };
  llvm::SmallVector<InputLockGroup> inputLockGroups;

  // Net model
  struct FifoBufferInfo {
    mlir::Operation *fifoOp = nullptr;
    // Logical ends, null for a physical one.
    mlir::Operation *producer = nullptr;
    llvm::SmallVector<mlir::Operation *> consumers;
    // Every end's tile, physical ones included.
    mlir::Operation *producerTile = nullptr;
    llvm::SmallVector<mlir::Operation *> consumerTiles;
    mlir::Operation *delegate = nullptr;
    int64_t producerSizeBytes = 0;
    int64_t consumerSizeBytes = 0;
    // Each end's object bytes, unaligned, as addressing lays the buffers out.
    int64_t producerRawBytes = 0;
    int64_t consumerRawBytes = 0;
    // ObjectFifoCreateOp::sharedDepth() and dmaDepth() of each end.
    int sharedDepth = 0;
    int producerDepth = 0;
    llvm::SmallVector<int> consumerDepths;
    // ObjectFifoCreateOp::plannedDmaEnd of each end, and the index of its
    // entry in memTileEnds on a mem tile.
    std::optional<PlannedDmaEnd> producerEnd;
    llvm::SmallVector<std::optional<PlannedDmaEnd>> consumerEnds;
    std::optional<unsigned> producerMemTileEntry;
    llvm::SmallVector<std::optional<unsigned>> consumerMemTileEntries;
    // Whether an end on a link's tile holds the link's one pool, which the
    // split builds from getPoolOwner and getPoolDepth as the plan says, or
    // shares one another end holds.
    enum class LinkPool { None, Holds, Shares };
    LinkPool producerLinkPool = LinkPool::None;
    llvm::SmallVector<LinkPool> consumerLinkPools;
    std::optional<PlannedDmaEnd> producerLinkPlan;
    llvm::SmallVector<std::optional<PlannedDmaEnd>> consumerLinkPlans;
    // The channel each end that draws one pins, if it pins one.
    std::optional<PinnedChannels::End> producerPin;
    llvm::SmallVector<std::optional<PinnedChannels::End>> consumerPins;
    // dmaReason(): it moves by DMA wherever its ends sit.
    bool forcesDMA = false;
    // The tile of the link naming the fifo, if any.
    mlir::Operation *linkTile = nullptr;
    // With an aie.objectfifo.allocate, ObjectFifoCreateOp's
    // allocateFallsBackToDMA() and whether linkPoolMisfit() is None.
    bool allocateFallsBackToDMA = false;
    bool fitsLinkPool = true;
    bool linkSharedProd = false;
    // PoolLocks::needsLocks' facts for the pools the split makes, and which
    // ends' cores use the fifo.
    bool unsynchronized = false;
    int64_t initObjects = 0;
    bool rearmed = false;
    bool producerCore = false;
    llvm::SmallVector<bool> consumerCore;
    // The lock groups of the link pool an end holds: the link's segments, or
    // none when a participant disables synchronization.
    int producerLinkSegments = 0;
    llvm::SmallVector<int> consumerLinkSegments;
  };
  llvm::SmallVector<NetInfo> nets;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<size_t>> tileToNetIndices;
  llvm::SmallVector<FifoBufferInfo, 0> fifoBuffers;
  // Each fifo's end tile types, where known, as tileTypes gives them.
  struct FifoEndTypes {
    std::optional<AIETileType> producer;
    llvm::SmallVector<std::optional<AIETileType>, 4> consumers;
  };
  std::vector<FifoEndTypes> fifoEndTypes;

  // A route's DMA end on a logical tile: one channel, in or out, that moves
  // with the tile.
  struct RouteEndInfo {
    mlir::Operation *tile;
    bool output;
  };
  llvm::SmallVector<RouteEndInfo> routeEnds;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<size_t>> tileToRouteEnds;
  // The DMA ends of each mem tile, logical or physical, and the BD ids each
  // takes, as the split will make them.
  struct MemTileEnd {
    DMAChannelDir dir;
    PlannedDmaEnd end;
  };
  llvm::MapVector<mlir::Operation *, llvm::SmallVector<MemTileEnd>> memTileEnds;
  // A link whose pool's depth follows where tiles sit: a participant other
  // than the owner shares the pool through its allocate only where its ends,
  // its delegate and the link's tile let it (ObjectFifoLinkOp::getPoolPlan).
  // `tiles` are those tiles; `depth` is the pool's depth the participants'
  // plans hold now. `holds` are the participants' ends that hold the pool's
  // plan. A participant's ends depend on where tiles sit only through link
  // pools' depths, so `replans` keeps them, and the pool's plan, by the
  // depths every variable link has.
  struct VariableLink {
    struct Replan {
      llvm::SmallVector<llvm::SmallVector<std::optional<PlannedDmaEnd>>> ends;
      std::optional<PlannedDmaEnd> held;
    };
    ObjectFifoLinkOp link;
    LinkPoolPlan plan;
    int64_t depth;
    llvm::SmallVector<mlir::Operation *> tiles;
    llvm::SmallVector<size_t> fifos;
    llvm::SmallVector<std::pair<size_t, std::optional<unsigned>>> holds;
    std::map<std::vector<int64_t>, Replan> replans;
  };
  llvm::SmallVector<VariableLink, 0> variableLinks;
  // The variable links the move being weighed planned again, with their
  // depths before and after it.
  struct ReplannedLink {
    unsigned index;
    int64_t from;
    int64_t to;
  };
  llvm::SmallVector<ReplannedLink> replannedLinks;
  // The pools of memTileEnds that config.walkedFifos names.
  llvm::DenseSet<mlir::Operation *> chargedPools;

  // Placement state
  llvm::DenseMap<mlir::Operation *, TileID> currentPlacement;
  llvm::DenseMap<TileID, mlir::Operation *> physToLogical;
  llvm::SmallVector<mlir::Operation *> movableTiles;
  llvm::DenseSet<mlir::Operation *> constrainedTiles;

  // Resource tracking
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<size_t>>
      tileToFifoIndices;
  llvm::DenseMap<TileID, int64_t> currentMemUsage;
  // Lock packings the moves repeat, the costliest to search again.
  mutable LockPlan::Memo lockMemo{64};
  // Every position whose memory a lock user at a position reaches, by
  // LockPlan::reaches, in position order and its own included.
  llvm::DenseMap<TileID, llvm::SmallVector<TileID>> lockReach;
  // The locks the pools at each position take, and the move's changes.
  llvm::DenseMap<TileID, int> currentLockDemand;
  llvm::SmallVector<std::pair<TileID, int>> lockUndo;
  llvm::DenseMap<TileID, std::pair<int, int>> currentDMAUsage;
  // The rule pinned ends share channels by, and the pinned ends at each
  // position, whose channels count in currentDMAUsage once each.
  PinnedChannels pinnedRule;
  PinnedChannelUse pinUse;
  llvm::DenseMap<size_t, TileID> sharedMemDestination;
  int cachedResourcePenalty = 0;
  // The resource terms that read only some tiles' positions, kept so a move
  // recomputes one only when it moves a tile the term reads.
  int memSpilloverPenalty = 0, bdCountPenalty = 0, delegatePenalty = 0;
  // What each core tile op holds besides objectFIFO buffers, as
  // assign-buffer-addresses lays it out: its aie.buffers and unsplit pools,
  // the buffers its core's section reservations become, and the stack.
  struct CoreTileContents {
    llvm::SmallVector<TileLayoutBuffer> buffers;
    bool hasCore = false;
    MemoryRun stackRun;
    std::optional<int> stackBank;
    int64_t stackAlignBytes = 1;
  };
  llvm::DenseMap<mlir::Operation *, CoreTileContents> coreContents;
  // The objectFIFO buffers on each core tile by size, the changes to them
  // since the last accepted move, for undo, and their net change in the
  // current one.
  llvm::DenseMap<TileID, std::map<int64_t, int>> coreFifoBuffers;
  llvm::SmallVector<std::tuple<TileID, int64_t, int>> coreFifoLog;
  llvm::DenseMap<TileID, std::map<int64_t, int>> coreFifoChange;
  // coreUnplaced of each core tile, the tiles whose ops a move changed, and
  // what the last move overwrote.
  llvm::DenseMap<TileID, int64_t> coreUnplacedAt;
  llvm::DenseSet<TileID> coreDirty;
  llvm::SmallVector<std::pair<TileID, int64_t>> coreUndo;
  // The core overflow penalty, kept while no tile it reads changes, and the
  // overflowing tiles it was computed for.
  int coreOverflowPenalty = 0;
  llvm::SmallVector<TileID> coreOverflowTiles;
  // coreUnplaced by layout problem, and by tile ops and objectFIFO buffers;
  // a result equals a fresh layoutTile.
  struct KeyHash {
    size_t operator()(const std::vector<int64_t> &key) const {
      return llvm::hash_combine_range(key.begin(), key.end());
    }
  };
  mutable std::unordered_map<std::vector<int64_t>, int64_t, KeyHash>
      coreUnplacedOf, coreUnplacedByTile;
  // computeLockPenalty's penalty, and whether its packing stopped, by all it
  // reads; a result equals a fresh one.
  mutable std::unordered_map<std::vector<int64_t>, std::pair<int, bool>,
                             KeyHash>
      lockVerdicts;
  static constexpr size_t kLockVerdictBytes = 1 << 20;
  static constexpr unsigned kLockVerdictMissRun = 4096;
  mutable size_t lockVerdictBytes = 0;
  mutable unsigned lockVerdictMissRun = 0;
  llvm::DenseSet<mlir::Operation *> spillTiles, delegateTermTiles, pairedTiles;
  // The buffers each mem tile, logical or physical, already holds, as the
  // allocator lays them out.
  llvm::MapVector<mlir::Operation *, llvm::SmallVector<BufferAllocation>>
      memTileFixed;
  // The BD ids the mem tile ends at each position are short of, what the
  // last move overwrote in it, and the shortfall of each set of logical tiles
  // sharing a position, by its tiles in order. A variable link's depth
  // changes what ends its participants plan there, so every cache below that
  // reads a position's ends or buffers also keys on every variable link's
  // depth, not the tiles' positions alone.
  llvm::DenseMap<TileID, int64_t> bdShortfallAt;
  llvm::SmallVector<std::pair<TileID, int64_t>> bdUndo;
  using WalkSets =
      llvm::SmallVector<llvm::SmallVector<std::pair<mlir::Operation *, int>>>;
  struct BDShortfall {
    int64_t ids;
    WalkSets sets;
    llvm::SmallVector<int64_t> bytes;
  };
  LRUCache<
      std::tuple<TileID, std::vector<mlir::Operation *>, std::vector<int64_t>>,
      BDShortfall>
      bdShortfallOf{1u << 16};
  /// The bytes the spill search leaves unplaced in one spill component, by its
  /// tiles' contents and each mem tile's walk set (bdShortfallAt's comment).
  LRUCache<std::vector<uintptr_t>, int64_t> unplacedOf{1u << 16};
  /// The spill penalty, by where every mem tile's holders sit
  /// (bdShortfallAt's comment).
  LRUCache<std::vector<int64_t>, int> spillPenaltyOf{1u << 14};
  // For each mem tile position, the first position seen whose channels reach
  // the same BD ids, which the shortfalls are cached under.
  llvm::DenseMap<TileID, TileID> bdLayoutOf;
  // What aie-objectfifo-allocate plans a tile's BD ids around: the channels
  // and ids its own programs and pinned runtime ids already hold.
  std::optional<DMAChannelAnalysis> channelAnalysis;
  // The channels each tile op's claims take by index, which hold them from
  // pinned ends placed with it.
  llvm::DenseMap<mlir::Operation *,
                 llvm::SmallVector<DMAChannelAnalysis::ChannelClaim>>
      claimsOf;
  DMAChannelAnalysis::TakenBDs takenBDs;
  llvm::DenseMap<TileID, llvm::SmallVector<mlir::Operation *>> physicalTiles;
  // Every logical mem tile, objectFIFO ends or not: its programs hold BD ids
  // and channels wherever it is placed.
  llvm::SmallVector<mlir::Operation *> logicalMemTiles;
  int cachedPressure = 0;
  // The DMA-channel term per tile, and what the last move overwrote in it.
  int dmaPenalty = 0;
  llvm::DenseMap<TileID, int> dmaPenaltyAt;
  llvm::SmallVector<std::pair<TileID, int>> dmaUndo;
  // An over-full shim tile is charged only for the channels allocation
  // still needs once ends never in flight together share them.
  std::optional<ShimTransferSpans> shimSpans;
  ShareableShimEnds shareableShimEnds;
  // The tile ops overlayUses names, which the overlay's channel weighs where
  // they are.
  llvm::SmallVector<mlir::Operation *> overlayTiles;
  // A shim tile is also charged for the runtime BD ids its LTOs' tasks lack.
  RuntimeBdIds runtimeBdIds;

  // Cascade
  Adjacency cascadeAdjacency;
  // Ends of fifos that ask for shared memory, which must share a module.
  Adjacency sharedMemAdjacency;

  // SA loop state
  int totalCost = 0;
  int cascadePen = 0;
  int bestCost = INT_MAX;        // best legal (resPenalty==0 && cascade==0)
  int bestOverallCost = INT_MAX; // best regardless of legality (fallback)
  PlacementResult bestPlacement; // corresponds to bestCost
  PlacementResult bestOverallPlacement; // corresponds to bestOverallCost
  int deviceSlots = 0;
  int zeroDeltaMoves = 0, posDeltaMoves = 0, negDeltaMoves = 0;
  int acceptedUphill = 0, rejectedMoves = 0;
  std::chrono::steady_clock::time_point startTime;

  // Phase methods
  mlir::LogicalResult collectAndBuildModel(DeviceOp device);
  mlir::LogicalResult generateInitialPlacement();
  void initializeSAState();
  void runSAMainLoop();
  mlir::LogicalResult finalizePlacement(DeviceOp device);

  // Cost and resource methods
  void initResourceTracking();
  void addFifoContribution(size_t fifoIdx, int sign);
  void addRouteEndContribution(size_t endIdx, int sign);
  void drawChannel(TileID position, bool output,
                   const std::optional<PinnedChannels::End> &pin, int sign);
  int updateResourcePenalty(
      const llvm::SmallVector<std::pair<mlir::Operation *, TileID>>
          &oldPlacements);
  int getResourcePenalty() const { return cachedResourcePenalty; }
  /// The BD ids the ends of mem tiles `tiles`, at `position`, are short of
  /// once aie-objectfifo-allocate has planned their channels; with
  /// `walkSets`, DMAChannelAnalysis::walkSetsThatFit's sets (every walkable
  /// pool when none fits) and, with `walkBytes`, each one's bytes.
  int64_t bdShortfall(TileID position, llvm::ArrayRef<mlir::Operation *> tiles,
                      WalkSets *walkSets = nullptr,
                      llvm::SmallVector<int64_t> *walkBytes = nullptr);
  /// The mem tiles at `position`, in memTileEnds order.
  llvm::SmallVector<mlir::Operation *> memTilesAt(TileID position) const;
  int computeMemSpilloverPenalty();
  int computeCoreOverflowPenalty() const;
  /// Charges `count` objectFIFO buffers of `bytes` to the core tile at `pos`.
  void chargeCoreFifo(TileID pos, int64_t bytes, int count);
  /// The bytes assign-buffer-addresses' layout (layoutTile) leaves unplaced
  /// on the core tile at `pos`, with `added` objectFIFO buffers more and
  /// `removed` fewer.
  int64_t coreUnplaced(TileID pos, const std::map<int64_t, int> &added = {},
                       const std::map<int64_t, int> &removed = {}) const;
  /// Relocates the intratile fifos of the core tile at `pos` to neighbours
  /// whose layout still places everything with them and what `lent` already
  /// gave them, as generateAllocates' allocates do; records each in `moves`
  /// and returns the bytes the tile's layout still leaves unplaced.
  int64_t
  planCoreFit(TileID pos, llvm::DenseMap<TileID, std::map<int64_t, int>> &lent,
              llvm::SmallVectorImpl<std::pair<size_t, TileID>> *moves) const;
  /// Recomputes coreUnplacedAt for the core tiles whose contents changed, and
  /// returns them.
  llvm::SmallVector<TileID> refreshCoreUnplaced();
  /// The overflowing core tiles and their neighbours, which the core
  /// overflow penalty reads.
  llvm::DenseSet<TileID> coreOverflowReads() const;
  /// Fills `dmaPenaltyAt` and returns its sum.
  int computeDMAChannelPenalty();
  int computeTileDMAPenalty(TileID tilePos) const;
  /// Fills `bdShortfallAt` and returns the penalty it gives.
  int computeBDCountPenalty();
  int computeDelegatePenalty() const;
  /// Plans the pools of the links `depths` names for the depths it gives
  /// them, and their participants' ends; a link already at its given depth
  /// keeps its plans. Every other variable link's tiles must not have moved
  /// since its last plan.
  void replanLinks(llvm::ArrayRef<std::pair<unsigned, int64_t>> depths);
  /// Charges each lock LockPlan cannot place at the current positions (see
  /// lockGroups). Sets `stopped`, when given, to whether its search stopped.
  int computeLockPenalty(bool *stopped = nullptr) const;
  /// The lock groups aie-objectfifo-allocate gives the pools the split makes
  /// where the tiles sit now: how many locks each takes (two a segment, or a
  /// pool's depth on AIE1) and the tiles using them, a compute tile's DMA
  /// reaching only its own.
  llvm::SmallVector<
      std::pair<int, llvm::SmallVector<std::pair<TileID, bool>, 2>>>
  lockGroups() const;
  std::optional<TileID> positionOf(mlir::Operation *tile) const;
  int computeMemoryPressure() const;
  int computeAdjacencyPenalty(const Adjacency &adj,
                              llvm::ArrayRef<std::pair<int, int>> validOffsets,
                              int weight) const;
  /// Cascade neighbours placed apart, and shared-memory ends placed on
  /// tiles sharing no memory module, by how far.
  int computePairingPenalty() const;
  /// Core tiles over local memory, in position order.
  llvm::SmallVector<TileID> overflowingCoreTiles() const;
  void generateAllocates();

  // Move methods
  void tryMove(llvm::SmallVector<std::pair<mlir::Operation *, TileID>> &moves);
  bool
  generateMove(llvm::SmallVector<std::pair<mlir::Operation *, TileID>> &moves);
  int evaluateMove(
      mlir::Operation *tile, TileID newPos,
      llvm::SmallVector<std::pair<size_t, NetBoundingBox>> &backups);
  void
  revertMove(llvm::SmallVector<std::pair<size_t, NetBoundingBox>> &backups);

  // Utility methods
  void buildNetModel(llvm::SmallVector<ObjectFifoCreateOp> &objectFifos,
                     llvm::SmallVector<ObjectFifoLinkOp> &objectFifoLinks);
  void buildFifoBufferInfo(DeviceOp device,
                           llvm::ArrayRef<ObjectFifoCreateOp> objectFifos,
                           llvm::ArrayRef<ObjectFifoLinkOp> objectFifoLinks);
  void buildRouteModel(DeviceOp device, llvm::ArrayRef<FlowOp> flows,
                       llvm::ArrayRef<PacketFlowOp> pktFlows,
                       llvm::ArrayRef<RouteOp> routes,
                       llvm::ArrayRef<DMAChannelAnalysis::ChannelClaim> claims);
  int computeNetHPWL(const NetInfo &net) const;
  int computeTotalHPWL() const;
  void initBoundingBoxes();
  double estimateInitialTemperature(int numSamples);
  bool isLegalPosition(mlir::Operation *tile, TileID pos) const;
  // The tile ops at `pos`, but `tile` and `leaving`, that name a channel by
  // index, for namedClash().
  llvm::SmallVector<mlir::Operation *>
  namedAt(mlir::Operation *tile, TileID pos,
          mlir::Operation *leaving = nullptr) const;
  int64_t computeTileMemoryWeight(mlir::Operation *tile) const;
  void printPlacementStats(int64_t elapsedMs) const;
};

} // namespace xilinx::AIE

#endif // AIE_PLACER_H
