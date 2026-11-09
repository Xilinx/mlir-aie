//===- AIEObjectFifoAllocate.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEBufferAllocation.h"
#include "aie/Dialect/AIE/Transforms/AIEChannelTurns.h"
#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"
#include "aie/Dialect/AIE/Transforms/AIEPacketIdSpace.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPoolLocks.h"
#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "aie/Dialect/AIE/Transforms/AIEStructuralRank.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#include <map>
#include <memory>
#include <set>

#define DEBUG_TYPE "aie-objectfifo-allocate"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOALLOCATE
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

struct AIEObjectFifoAllocatePass
    : public xilinx::AIE::impl::AIEObjectFifoAllocateBase<
          AIEObjectFifoAllocatePass> {
  using Base::Base;

  DeviceOp device;
  OpBuilder builder{static_cast<MLIRContext *>(nullptr)};
  /// The last buffer or lock this pass placed on each tile.
  DenseMap<Value, Operation *> lastPlaced;
  /// Which pools get locks, and the resets re-arming each fifo.
  std::shared_ptr<PoolLocks> poolLocks;
  /// Flows already lowered. A route endpoint reads its direction off the
  /// flow naming it, so these outlive the walk that replaces them.
  SmallVector<Operation *> loweredFlows;
  /// The flows each route endpoint is the source of, by its symbol name,
  /// gathered before any flow is lowered.
  DenseMap<StringAttr, SmallVector<RouteOp>> sourcedFlows;
  llvm::MapVector<ObjectFifoDmaEndpointOp, llvm::StringMap<PacketInfoAttr>>
      dispatchHeaders;
  /// Ends that take another end's channel, by the end they take it from:
  /// shim ends never in flight together, and ends taking turns.
  llvm::MapVector<Operation *, Operation *> channelSharers;
  DenseMap<Operation *, SmallVector<Value>> bufferPlacements;
  DenseMap<Operation *, int> channelAssignments;
  DenseMap<Operation *, Value> localPools;
  DenseMap<Operation *, SmallVector<Value>> poolUsers;
  DenseMap<Operation *, SmallVector<Operation *>> lockUsers;
  DenseMap<Operation *, Value> lockPlacements;
  /// What runs on each channel the design programs itself, by the
  /// `aie.route_endpoint` naming it: dma_starts in its tile's DMA program and
  /// tasks the runtime sequence configures for it.
  DenseMap<Operation *, SmallVector<DMAStartOp>> endpointStarts;
  DenseMap<Operation *, SmallVector<AIEX::DMAConfigureTaskForOp>> endpointTasks;
  /// BDs the chains DMA programs start by index hold on each channel of a
  /// placed tile, by (col, row, channel). Channels are chosen from these and
  /// the endpoints' own chains alone, so an image built for one runtime
  /// sequence serves another's instruction stream.
  DenseMap<std::tuple<int, int, int>, int64_t> fixedBDs;
  /// The BD ids runtime tasks pin, which aie-reserve-runtime-bd-ids keeps
  /// static BDs off, each once per tile, by the lowest channel that reaches it.
  DenseMap<std::tuple<int, int, int>, int64_t> pinnedBDs;
  /// The most unpinned BDs one task the runtime sequence configures on each
  /// channel by index holds, by (col, row, channel).
  DenseMap<std::tuple<int, int, int>, int64_t> taskBDs;
  /// The BDs every channel the last assignment handed out needs at once,
  /// runtime tasks included.
  DenseMap<std::tuple<int, int, int>, int64_t> channelDemand;
  /// Where each tile, pool, endpoint and route falls in an order taken from
  /// the design's structure alone, which settles the choices op order would.
  DenseMap<Operation *, unsigned> structuralRank;
  RouteEndpoint channelFailure;
  // Whether channelFailure's channel has too few BD ids left, rather than no
  // channel being free.
  bool bdShortfall = false;
  /// Mem tiles once short of BD ids, whose ends since take their channels
  /// from DMAChannelAnalysis::planBDChannels.
  llvm::SetVector<Operation *> bdPlanned;
  /// The pool the buffer search could not place with the fewest left
  /// unplaced, i.e. where it got furthest.
  ObjectFifoPoolOp bufferFailure;
  /// Neighbors made for a spill planBuffers weighed. The ones no buffer or
  /// lock ends up on are erased once allocation is done.
  SmallVector<TileOp> spillTiles;
  Value lockFailure;
  Operation *lockAccessFailure = nullptr;
  /// Whether a search for which pools to walk whole stopped at its bound,
  /// DMAChannelAnalysis::kMaxWalkPlacements, before finding one that places.
  bool walkSearchStopped = false;

  bool sameTile(Value a, Value b) {
    if (a == b)
      return true;
    auto first = cast<TileLike>(a.getDefiningOp());
    auto second = cast<TileLike>(b.getDefiningOp());
    auto ac = first.tryGetCol(), ar = first.tryGetRow();
    auto bc = second.tryGetCol(), br = second.tryGetRow();
    return ac && ar && bc && br && ac == bc && ar == br;
  }

  /// The mem tiles a buffer search weighs, as MemTileSpillSearch indices: one
  /// per placed position, however many ops name it, with the positions a
  /// placed TileOp may spill to, and one per unplaced tile.
  DenseMap<TileID, unsigned> memTilePositions;
  DenseMap<Operation *, unsigned> unplacedMemTiles;
  SmallVector<std::optional<TileID>> memTileAt;

  std::optional<unsigned> memTileIndex(Value tile) {
    auto like = cast<TileLike>(tile.getDefiningOp());
    if (!like.isMemTile())
      return std::nullopt;
    auto col = like.tryGetCol(), row = like.tryGetRow();
    if (col && row)
      return memTilePositions.lookup({*col, *row});
    return unplacedMemTiles.lookup(like);
  }

  /// The search over the device's mem tiles, holding their existing buffers.
  MemTileSpillSearch memTileSearch() {
    const auto &target = device.getTargetModel();
    memTilePositions.clear();
    unplacedMemTiles.clear();
    memTileAt.clear();
    SmallVector<MemTileSpillSearch::Tile> tiles;
    auto add = [&](std::optional<TileID> at) {
      memTileAt.push_back(at);
      tiles.push_back({target.getMemTileSize(), {}});
      return static_cast<unsigned>(tiles.size() - 1);
    };
    for (auto tile : device.getOps<TileLike>()) {
      if (!tile.isMemTile())
        continue;
      auto col = tile.tryGetCol(), row = tile.tryGetRow();
      if (!col || !row) {
        unplacedMemTiles[tile] = add(std::nullopt);
        continue;
      }
      TileID at{*col, *row};
      if (!memTilePositions.count(at))
        memTilePositions[at] = add(at);
      if (isa<TileOp>(tile.getOperation()))
        for (TileID neighbour : memTileSpillNeighbours(target, at))
          if (!memTilePositions.count(neighbour))
            memTilePositions[neighbour] = add(neighbour);
    }
    int64_t alignment = target.getMemTileLoadStoreBusWidth() / 8;
    for (auto buffer : device.getOps<BufferOp>())
      if (std::optional<unsigned> index = memTileIndex(buffer.getTile()))
        tiles[*index].fixed.push_back({buffer.getAllocationSize(),
                                       buffer.getAligned() ? alignment : 1,
                                       buffer.getAddress()});
    return MemTileSpillSearch(std::move(tiles), alignment);
  }

  LogicalResult collectFixedMemory() {
    MemTileSpillSearch search = memTileSearch();
    for (auto tile : device.getOps<TileLike>()) {
      std::optional<unsigned> index = memTileIndex(tile->getResult(0));
      if (!index)
        continue;
      int64_t bytes = search.used(*index);
      int64_t capacity = device.getTargetModel().getMemTileSize();
      if (bytes > capacity)
        return tile->emitOpError("existing buffers require ")
               << bytes << " bytes, exceeding MemTile capacity of " << capacity
               << " bytes";
    }
    return success();
  }

  /// Coordinates `tile` may take: its own once placed, else every position
  /// of its type in the rows and columns it is pinned to.
  SmallVector<TileID> compatiblePositions(TileLike tile) {
    const auto &target = device.getTargetModel();
    SmallVector<TileID> compatible;
    auto col = tile.tryGetCol(), row = tile.tryGetRow();
    for (int c = 0; c < target.columns(); ++c) {
      if (col && c != *col)
        continue;
      for (int r = 0; r < target.rows(); ++r) {
        if ((!row || r == *row) &&
            target.getTileType(c, r) == tile.getTileType())
          compatible.push_back({c, r});
      }
    }
    return compatible;
  }

  bool canAccess(Value user, Value memory, bool localOnly = false) {
    if (user == memory)
      return true;
    auto memoryTile = dyn_cast<TileLike>(memory.getDefiningOp());
    return memoryTile &&
           canAccess(user, compatiblePositions(memoryTile), localOnly);
  }

  /// Before placement, defer only accesses that some compatible physical
  /// positions could satisfy. Known coordinates can already disprove affinity.
  bool canAccess(Value user, ArrayRef<TileID> memoryPositions,
                 bool localOnly = false) {
    auto userTile = dyn_cast<TileLike>(user.getDefiningOp());
    if (!userTile)
      return false;
    const auto &target = device.getTargetModel();
    for (TileID u : compatiblePositions(userTile))
      for (TileID m : memoryPositions) {
        if (localOnly && u != m)
          continue;
        auto shared = target.getSharedMemory(u, m);
        if (shared == AIETargetModel::SharedMemory::Second ||
            shared == AIETargetModel::SharedMemory::Either)
          return true;
      }
    return false;
  }

  Value lockPlacement(ObjectFifoPoolOp pool, Operation *group) {
    if (Value placed = lockPlacements.lookup(group))
      return placed;
    Value local = localPools.lookup(pool);
    return local ? local : pool.getTile();
  }

  /// A contiguous pool's objects are one buffer; others have one per object.
  static int bufferCount(ObjectFifoPoolOp pool) {
    return pool.getContiguous() ? 1 : pool.getDepth();
  }
  static int64_t bufferBytes(ObjectFifoPoolOp pool) {
    return pool.getObjectSizeInBytes() * pool.getObjectsPerBuffer();
  }

  /// FIXME: choosing which tile a buffer lives on is the buffer allocator's
  /// job, not this pass's; --aie-assign-buffer-addresses should instead be
  /// free to move buffers between tiles that share a memory module.
  ///
  /// Reserve mandatory-local objects before any other pool can spill into
  /// their memory. Existing buffers are fixed and counted by identity, not by
  /// how many pools or endpoints reference them. The others' mem tile buffers
  /// go where MemTileSpillSearch puts them, in pool order.
  LogicalResult planBuffers(ArrayRef<ObjectFifoPoolOp> pools) {
    bufferPlacements.clear();
    bufferFailure = nullptr;
    MemTileSpillSearch search = memTileSearch();
    auto reachable = [&](ObjectFifoPoolOp pool, auto memory) {
      return llvm::all_of(poolUsers[pool],
                          [&](Value user) { return canAccess(user, memory); });
    };
    for (auto pool : pools) {
      if (!localPools.contains(pool))
        continue;
      Value tile = localPools.lookup(pool);
      int count = pool.getBuffers() ? 0 : bufferCount(pool);
      int64_t size = bufferBytes(pool);
      std::optional<unsigned> index = memTileIndex(tile);
      if (!canAccess(pool.getTile(), tile) || !reachable(pool, tile) ||
          (index && !search.fits(*index, size, count))) {
        bufferFailure = pool;
        return failure();
      }
      if (index)
        search.reserve(*index, size, count);
    }
    const auto &target = device.getTargetModel();
    SmallVector<MemTileSpillSearch::Buffer> buffers;
    SmallVector<ObjectFifoPoolOp> owners;
    for (auto pool : pools) {
      if (pool.getBuffers()) {
        for (auto buffer : pool.getBufferOps()) {
          Value tile = buffer.getBufferTile();
          if (tile && ((localPools.contains(pool) &&
                        !sameTile(tile, localPools.lookup(pool))) ||
                       !reachable(pool, tile))) {
            bufferFailure = pool;
            return failure();
          }
          bufferPlacements[pool].push_back(tile);
        }
        continue;
      }
      if (pool.getTileLike().isShimTile())
        continue;
      if (localPools.contains(pool)) {
        bufferPlacements[pool].append(bufferCount(pool),
                                      localPools.lookup(pool));
        continue;
      }
      Value home = pool.getTile();
      bool atHome = reachable(pool, home);
      std::optional<unsigned> index = memTileIndex(home);
      if (!index) {
        if (!atHome) {
          bufferFailure = pool;
          return failure();
        }
        bufferPlacements[pool].append(bufferCount(pool), home);
        continue;
      }
      // Which tiles neighbor an unplaced one is not yet known, so its buffers
      // stay at home.
      SmallVector<unsigned, 2> neighbours;
      if (auto homeOp = dyn_cast<TileOp>(home.getDefiningOp()))
        for (TileID at :
             memTileSpillNeighbours(target, {homeOp.getCol(), homeOp.getRow()}))
          if (reachable(pool, ArrayRef<TileID>{at}))
            neighbours.push_back(memTilePositions.lookup(at));
      for (int i = 0, e = bufferCount(pool); i < e; ++i) {
        buffers.push_back({bufferBytes(pool), *index, atHome, neighbours});
        owners.push_back(pool);
      }
    }
    bool placed = search.place(buffers);
    // A spill weighed from a placed home makes its neighbors' tile ops; the
    // ones no buffer or lock ends up on are erased once allocation is done.
    for (auto [buffer, pool] : llvm::zip(buffers, owners)) {
      auto homeOp = dyn_cast<TileOp>(pool.getTile().getDefiningOp());
      if (!homeOp || !search.weighed[buffer.home])
        continue;
      for (int col : {homeOp.getCol() - 1, homeOp.getCol() + 1})
        if (col >= 0 && col < target.columns())
          spillTile(col, homeOp.getRow(), pool.getLoc());
    }
    if (!placed) {
      bufferFailure = owners[search.failed];
      return failure();
    }
    for (auto [buffer, tile, pool] : llvm::zip(buffers, search.chosen, owners))
      bufferPlacements[pool].push_back(
          tile == buffer.home ? pool.getTile()
                              : spillTile(memTileAt[tile]->col,
                                          memTileAt[tile]->row, pool.getLoc())
                                    .getResult());
    return success();
  }

  /// The tile op at (`col`, `row`), made if the device has none.
  TileOp spillTile(int col, int row, Location loc) {
    bool declared = llvm::any_of(device.getOps<TileOp>(), [&](TileOp t) {
      return t.getCol() == col && t.getRow() == row;
    });
    TileOp tile = TileOp::getOrCreate(builder, device, col, row, loc);
    if (!declared)
      spillTiles.push_back(tile);
    return tile;
  }

  /// Buffers and locks sit directly below the tile whose memory holds them,
  /// after whatever this pass has already put there.
  void setInsertionPointOn(Value tile) {
    Operation *after = lastPlaced.lookup(tile);
    if (!after) {
      after = tile.getDefiningOp();
    }
    if (after) {
      builder.setInsertionPointAfter(after);
    } else {
      builder.setInsertionPointToStart(device.getBody());
    }
  }

  void allocateBuffers(ObjectFifoPoolOp pool) {
    if (pool.getBuffers()) {
      return;
    }
    TileLike home = pool.getTileLike();
    if (home.isShimTile()) {
      return;
    }

    ArrayRef<Attribute> initValues;
    if (auto initAttr = pool.getInitValues()) {
      initValues = initAttr->getValue();
    }
    int filled = initValues.size();
    StringRef base = pool.getBaseName();

    SmallVector<Attribute> names;
    MemRefType type = pool.getElemType();
    ElementsAttr zeros;
    if (pool.getContiguous()) {
      int64_t objectElements = type.getNumElements();
      type = MemRefType::get({pool.getObjectsPerBuffer() * objectElements},
                             type.getElementType());
      auto tensor =
          RankedTensorType::get(type.getShape(), type.getElementType());
      Attribute zero = builder.getZeroAttr(type.getElementType());
      if (filled) {
        // The one buffer starts as its objects do, in order, between its zero
        // objects; objects with no initial contents start zero.
        auto zeroObjects = pool.getZeroObjects();
        SmallVector<Attribute> values(
            (zeroObjects ? (*zeroObjects)[0] : 0) * objectElements, zero);
        for (int i = 0; i < pool.getDepth(); i++) {
          if (i < filled)
            llvm::append_range(
                values,
                cast<DenseElementsAttr>(initValues[i]).getValues<Attribute>());
          else
            values.append(objectElements, zero);
        }
        values.append((zeroObjects ? (*zeroObjects)[1] : 0) * objectElements,
                      zero);
        zeros = DenseElementsAttr::get(tensor, values);
      } else if (pool.getZeroObjects()) {
        zeros = DenseElementsAttr::get(tensor, zero);
      }
    }
    // Initial contents may fill only the first objects of a deeper pool.
    for (int i = 0; i < bufferCount(pool); i++) {
      ElementsAttr init = !pool.getContiguous() && i < filled
                              ? cast<ElementsAttr>(initValues[i])
                              : zeros;
      std::string name = (base + "__buf" + std::to_string(i)).str();
      Value placement = bufferPlacements[pool][i];
      setInsertionPointOn(placement);
      lastPlaced[placement] = BufferOp::create(
          builder, pool.getLoc(), type, placement, builder.getStringAttr(name),
          /*address=*/nullptr, init,
          /*mem_bank=*/nullptr, /*core_data=*/nullptr);
      names.push_back(FlatSymbolRefAttr::get(builder.getContext(), name));
    }
    pool.setBuffersAttr(builder.getArrayAttr(names));
  }

  LockOp createLock(ObjectFifoPoolOp pool, Value tile, StringRef name,
                    int value) {
    setInsertionPointOn(tile);
    auto lock = LockOp::create(builder, pool.getLoc(), tile, value);
    lastPlaced[tile] = lock;
    lock->setAttr(SymbolTable::getSymbolAttrName(),
                  builder.getStringAttr(name));
    return lock;
  }

  LogicalResult planLocks(ArrayRef<ObjectFifoPoolOp> pools) {
    lockFailure = {};
    lockAccessFailure = nullptr;
    lockPlacements.clear();
    DenseMap<std::pair<int, int>, int64_t> used;
    auto reserve = [&](Value tile, int64_t count) {
      auto like = cast<TileLike>(tile.getDefiningOp());
      auto col = like.tryGetCol(), row = like.tryGetRow();
      if (!col || !row)
        return success();
      if (used[{*col, *row}] + count >
          device.getTargetModel().getNumLocks(*col, *row)) {
        lockFailure = tile;
        return failure();
      }
      used[{*col, *row}] += count;
      return success();
    };
    for (auto lock : device.getOps<LockOp>())
      if (failed(reserve(lock.getTile(), 1)))
        return failure();
    // Neighbors created to weigh a spill have no rank and come last.
    SmallVector<TileLike> tiles(device.getOps<TileLike>());
    llvm::stable_sort(tiles, [&](TileLike a, TileLike b) {
      auto aRank = structuralRank.find(a), bRank = structuralRank.find(b);
      if ((aRank == structuralRank.end()) != (bRank == structuralRank.end()))
        return bRank == structuralRank.end();
      return aRank != structuralRank.end() && aRank->second < bRank->second;
    });
    auto place = [&](ObjectFifoPoolOp pool, Operation *group, int count) {
      SmallVector<Value> candidates{lockPlacement(pool, group)};
      if (!localPools.contains(pool)) {
        llvm::append_range(candidates, bufferPlacements[pool]);
        for (Operation *user : lockUsers[group])
          candidates.push_back(user->getOperand(0));
        for (auto tile : tiles)
          candidates.push_back(tile->getResult(0));
      }
      bool reachable = false;
      for (Value tile : candidates) {
        if (!tile || !llvm::all_of(lockUsers[group], [&](Operation *user) {
              return canAccessLocks(user, tile);
            }))
          continue;
        reachable = true;
        if (succeeded(reserve(tile, count))) {
          lockPlacements[group] = tile;
          lockFailure = {};
          return success();
        }
      }
      if (!reachable && !lockUsers[group].empty()) {
        lockFailure = {};
        lockAccessFailure = lockUsers[group].front();
      }
      return failure();
    };
    for (auto pool : pools) {
      if (!poolLocks->needsLocks(pool))
        continue;
      if (device.getTargetModel().getTargetArch() == AIEArch::AIE1) {
        if (!pool.getLocks() && failed(place(pool, pool, pool.getDepth())))
          return failure();
      } else {
        for (auto segment : pool.getSegmentOps())
          if (!(segment.getProduceLock() && segment.getConsumeLock()) &&
              failed(place(pool, segment, 2)))
            return failure();
      }
    }
    auto check = [&](auto endpoint) {
      for (Value tile : endpointLockTiles(endpoint)) {
        if (!canAccessLocks(endpoint, tile)) {
          lockAccessFailure = endpoint;
          return failure();
        }
      }
      return success();
    };
    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>())
      if (failed(check(endpoint)))
        return failure();
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>())
      if (failed(check(endpoint)))
        return failure();
    return success();
  }

  /// AIE1 guards each buffer with one binary lock that rotates with it; AIE2
  /// gives each segment a counting pair, the producer's lock counting free
  /// objects and the consumer's counting full ones.
  void allocateLocks(ObjectFifoPoolOp pool) {
    if (!poolLocks->needsLocks(pool))
      return;
    StringRef base = pool.getBaseName();
    int depth = pool.getDepth();
    auto initValues = pool.getInitValues();
    int filled = initValues ? initValues->size() : 0;
    int repeat = pool.getRepeatCount().value_or(1);

    if (device.getTargetModel().getTargetArch() == AIEArch::AIE1) {
      if (pool.getLocks()) {
        return;
      }
      SmallVector<Attribute> names;
      for (int i = 0; i < depth; i++) {
        std::string name = (base + "__lock" + std::to_string(i)).str();
        createLock(pool, lockPlacement(pool, pool), name, i < filled ? 1 : 0);
        names.push_back(FlatSymbolRefAttr::get(builder.getContext(), name));
      }
      pool.setLocksAttr(builder.getArrayAttr(names));
      return;
    }

    for (auto [index, segment] : llvm::enumerate(pool.getSegmentOps())) {
      if (segment.getProduceLock() && segment.getConsumeLock()) {
        continue;
      }
      std::string produce = (base + "__prodlock" + std::to_string(index)).str();
      std::string consume = (base + "__conslock" + std::to_string(index)).str();
      Value tile = lockPlacement(pool, segment);
      createLock(pool, tile, produce, (depth - filled) * repeat);
      createLock(pool, tile, consume, filled * repeat);
      segment.setProduceLockAttr(
          FlatSymbolRefAttr::get(builder.getContext(), produce));
      segment.setConsumeLockAttr(
          FlatSymbolRefAttr::get(builder.getContext(), consume));
    }
  }

  bool canAccessLocks(Operation *endpoint, Value tile) {
    Value user = endpoint->getOperand(0);
    // Compute-tile DMA engines, unlike cores, only use their own lock module.
    if (isa<ObjectFifoDmaEndpointOp>(endpoint) &&
        !cast<TileLike>(user.getDefiningOp()).isMemTile())
      return canAccess(user, tile, /*localOnly=*/true);
    return canAccess(user, tile);
  }

  template <typename EndpointOp>
  void collectLockUsers(EndpointOp endpoint) {
    auto pool = endpoint.getPoolOp();
    if (device.getTargetModel().getTargetArch() == AIEArch::AIE1) {
      lockUsers[pool].push_back(endpoint);
      return;
    }
    for (auto segment : endpoint.getSelectedSegments())
      lockUsers[segment].push_back(endpoint);
  }

  /// Predict the locks core and DMA lowering will use, including hand-written
  /// locks and those allocation will create. Only selected segments matter.
  template <typename EndpointOp>
  SmallVector<Value> endpointLockTiles(EndpointOp endpoint) {
    SmallVector<Value> tiles;
    auto pool = endpoint.getPoolOp();
    if (pool.getDepth() == 0)
      return tiles;
    if (device.getTargetModel().getTargetArch() == AIEArch::AIE1) {
      if (poolLocks->needsLocks(pool) && !pool.getLocks())
        tiles.push_back(lockPlacement(pool, pool));
      else
        for (auto lock : pool.getLockOps())
          tiles.push_back(lock.getTile());
      return tiles;
    }
    for (auto segment : endpoint.getSelectedSegments()) {
      if (poolLocks->needsLocks(pool) &&
          !(segment.getProduceLock() && segment.getConsumeLock())) {
        tiles.push_back(lockPlacement(pool, segment));
        continue;
      }
      for (auto name :
           {segment.getProduceLockAttr(), segment.getConsumeLockAttr()})
        if (name)
          if (auto lock = lookupNamedOp<LockOp>(device, name.getAttr()))
            tiles.push_back(lock.getTile());
    }
    return tiles;
  }

  /// Both buffers and locks must be local to use the local-only channels.
  /// Distinct unresolved tiles may be neighbors after placement, so they
  /// cannot be assumed local when assigning a placed endpoint's channels.
  bool reachesAdjacentTile(RouteEndpoint endpoint) {
    if (auto route = dyn_cast<RouteEndpointOp>(endpoint.getOperation()))
      return programReachesAdjacentTile(route);
    auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation());
    if (!dma || !dma.getTileLike().isMemTile()) {
      return false;
    }
    ObjectFifoPoolOp pool = dma.getPoolOp();
    auto remote = [&](Value tile) {
      return tile && !sameTile(tile, endpoint.getTile());
    };
    return pool && (llvm::any_of(bufferPlacements[pool], remote) ||
                    llvm::any_of(endpointLockTiles(dma), remote));
  }

  TileLike tileOf(RouteEndpoint endpoint) {
    return dyn_cast<TileLike>(endpoint.getTile().getDefiningOp());
  }

  /// A DMA channel the design programs itself, by naming this endpoint from a
  /// dma_start or a runtime task, rather than one an objectFIFO lowers to.
  static bool isProgramOwned(RouteEndpoint endpoint) {
    auto route = dyn_cast<RouteEndpointOp>(endpoint.getOperation());
    return route && route.getBundle() == WireBundle::DMA &&
           !route.getTileLike().isShimTile();
  }

  /// Blocks of the BD chain `start` begins, following next_bd.
  static SmallVector<Block *> chainOf(DMAStartOp start) {
    SmallVector<Block *> chain;
    SmallPtrSet<Block *, 8> seen;
    SmallVector<Block *> work{start.getDest()};
    while (!work.empty()) {
      Block *block = work.pop_back_val();
      if (!seen.insert(block).second || block->getOps<DMABDOp>().empty())
        continue;
      chain.push_back(block);
      if (auto next = dyn_cast<NextBDOp>(block->getTerminator()))
        work.push_back(next.getDest());
    }
    return chain;
  }

  static int64_t countBDs(DMAStartOp start) {
    int64_t bds = 0;
    for (Block *block : chainOf(start))
      bds += llvm::range_size(block->getOps<DMABDOp>());
    return bds;
  }

  static int64_t countBDs(Region &taskBody) {
    int64_t bds = 0;
    taskBody.walk([&](DMABDOp) { bds++; });
    return bds;
  }

  /// BDs a program-owned endpoint's static chains hold; they stay configured.
  int64_t endpointBDs(Operation *endpoint) {
    int64_t bds = 0;
    for (DMAStartOp start : endpointStarts.lookup(endpoint))
      bds += countBDs(start);
    return bds;
  }

  /// BDs the largest runtime task on a program-owned endpoint needs.
  int64_t endpointTaskBDs(Operation *endpoint) {
    int64_t largest = 0;
    for (auto task : endpointTasks.lookup(endpoint))
      largest = std::max(largest, countBDs(task.getBody()));
    return largest;
  }

  /// Whether a program-owned MemTile channel's BDs touch a neighbor's buffers
  /// or locks, which only the lower channels can.
  bool programReachesAdjacentTile(RouteEndpointOp endpoint) {
    if (!endpoint.getTileLike().isMemTile())
      return false;
    auto remote = [&](Operation *op) {
      Value tile;
      if (auto bd = dyn_cast<DMABDOp>(op)) {
        if (auto buffer =
                dyn_cast_or_null<BufferOp>(bd.getBuffer().getDefiningOp()))
          tile = buffer.getTile();
      } else if (auto use = dyn_cast<UseLockOp>(op)) {
        if (auto lock = dyn_cast_or_null<LockOp>(use.getLock().getDefiningOp()))
          tile = lock.getTile();
      }
      return tile && !sameTile(tile, endpoint.getTile());
    };
    for (DMAStartOp start : endpointStarts.lookup(endpoint))
      for (Block *block : chainOf(start))
        if (llvm::any_of(block->getOperations(),
                         [&](Operation &op) { return remote(&op); }))
          return true;
    for (auto task : endpointTasks.lookup(endpoint)) {
      bool reaches = false;
      task.getBody().walk([&](Operation *op) { reaches |= remote(op); });
      if (reaches)
        return true;
    }
    return false;
  }

  /// Find what programs each endpoint-named channel, and the BDs already spoken
  /// for on channels named by index.
  void collectChannelPrograms() {
    endpointStarts.clear();
    endpointTasks.clear();
    fixedBDs.clear();
    pinnedBDs.clear();
    taskBDs.clear();
    auto key = [](Value tile,
                  int channel) -> std::optional<std::tuple<int, int, int>> {
      auto placed = cast<TileLike>(tile.getDefiningOp());
      auto col = placed.tryGetCol(), row = placed.tryGetRow();
      if (!col || !row)
        return std::nullopt;
      return std::make_tuple(*col, *row, channel);
    };
    for (auto program : device.getOps<DmaBody>())
      for (Block &block : program.getDmaBody())
        for (auto start : block.getOps<DMAStartOp>()) {
          if (auto endpoint = start.getEndpointOp()) {
            endpointStarts[endpoint].push_back(start);
          } else if (auto at =
                         key(program.getTile(), start.getChannelIndex())) {
            fixedBDs[*at] += countBDs(start);
          }
        }
    // Each id a task pins on a tile, with a channel reaching it.
    std::map<std::pair<int, int>, std::map<int32_t, int>> pinned;
    device.walk([&](Operation *op) {
      if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
        if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(
                SymbolTable::lookupNearestSymbolFrom(
                    device, task.getAlloc().getRootReference())))
          endpointTasks[endpoint].push_back(task);
      } else if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op)) {
        if (auto at = key(task.getTile(), task.getChannel())) {
          int64_t unpinned = 0;
          task.getBody().walk([&](DMABDOp bd) {
            if (std::optional<int32_t> id = bd.getBdId())
              pinned[{std::get<0>(*at), std::get<1>(*at)}].try_emplace(
                  *id, task.getChannel());
            else
              ++unpinned;
          });
          int64_t &largest = taskBDs[*at];
          largest = std::max(largest, unpinned);
        }
      }
    });
    const AIETargetModel &target = device.getTargetModel();
    for (auto &[at, ids] : pinned) {
      auto [col, row] = at;
      for (auto [id, reaching] : ids)
        for (int channel = 0; channel <= reaching; ++channel)
          if (target.isBdChannelAccessible(col, row, id, channel)) {
            ++pinnedBDs[{col, row, channel}];
            break;
          }
    }
  }

  /// Rank tiles, pools, endpoints and routes by their attributes and then, a
  /// round at a time until no class splits, by the ranks of what they connect
  /// to. Symbol names and op order play no part, so a permuted or renamed
  /// design ranks alike, and ops still tied cannot be told apart.
  void rankStructurally() {
    StructuralGraph graph;
    // A placed tile is one node however many ops name it.
    DenseMap<std::pair<int, int>, Operation *> placed;
    DenseMap<Operation *, Operation *> tileNode;
    for (auto tile : device.getOps<TileLike>()) {
      auto col = tile.tryGetCol(), row = tile.tryGetRow();
      if (!col || !row) {
        graph.add(tileNode[tile] = tile);
        continue;
      }
      auto [it, first] = placed.try_emplace({*col, *row}, tile);
      tileNode[tile] = it->second;
      if (first) {
        graph.nodes.push_back(tile);
        llvm::raw_string_ostream(graph.features[tile])
            << "tile " << *col << ' ' << *row << ' '
            << stringifyAIETileType(tile.getTileType());
      }
    }
    auto link = [&](Operation *from, Operation *to, StringRef label) {
      if (Operation *tile = tileNode.lookup(to))
        to = tile;
      graph.link(from, to, label);
    };
    for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
      llvm::raw_string_ostream os(graph.add(pool));
      for (auto segment : pool.getSegmentOps())
        os << " [" << segment.getOffset() << ", " << segment.getSize() << "]";
      link(pool, pool.getTile().getDefiningOp(), "tile");
    }
    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      llvm::raw_string_ostream os(graph.add(endpoint));
      for (auto segment : endpoint.getSelectedSegments())
        os << " @" << segment.getOffset();
      link(endpoint, endpoint.getTile().getDefiningOp(), "tile");
      link(endpoint, endpoint.getPoolOp(), "pool");
    }
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      llvm::raw_string_ostream os(graph.add(endpoint));
      if (auto dma =
              dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation())) {
        for (auto segment : dma.getSelectedSegments())
          os << " @" << segment.getOffset();
        link(endpoint, dma.getPoolOp(), "pool");
      }
      link(endpoint, endpoint.getTile().getDefiningOp(), "tile");
    }
    for (auto route : device.getOps<RouteOp>()) {
      graph.add(route);
      for (StringRef source : route.getSourceNames())
        link(route, lookupEndpoint(source), "source");
      for (StringRef dest : route.getDestinationNames())
        link(route, lookupEndpoint(dest), "dest");
    }
    DenseMap<Operation *, unsigned> rank = graph.rank();
    for (auto [tile, node] : tileNode)
      rank[tile] = rank[node];
    structuralRank = std::move(rank);
  }

  /// BDs `channel` of the tile at (`col`, `row`) can use that `demand` leaves
  /// over, counting every channel whose BD ids overlap its own. Negative when
  /// overcommitted.
  int64_t bdRoom(TileLike tile, int col, int row, int channel,
                 const DenseMap<std::tuple<int, int, int>, int64_t> &demand) {
    const AIETargetModel &target = device.getTargetModel();
    uint32_t numBDs = target.getNumBDs(col, row);
    int numChannels = std::max(tile.getNumSourceConnections(WireBundle::DMA),
                               tile.getNumDestConnections(WireBundle::DMA));
    auto sharesBDs = [&](int other) {
      for (uint32_t bd = 0; bd < numBDs; ++bd)
        if (target.isBdChannelAccessible(col, row, bd, channel) &&
            target.isBdChannelAccessible(col, row, bd, other))
          return true;
      return false;
    };
    int64_t room = target.getNumBDsForChannel(col, row, channel);
    for (int other = 0; other < numChannels; ++other)
      if (sharesBDs(other))
        if (auto it = demand.find({col, row, other}); it != demand.end())
          room -= it->second;
    return room;
  }

  /// On a tile whose BD ids are split between channels, the free channel whose
  /// BDs have the most room left once every channel sharing them has its
  /// demand; ties, and tiles without such a split, go to the lowest index.
  /// Falls back to first-free where coordinates are unknown.
  int chooseChannel(DMAChannelAnalysis &channels, RouteEndpoint endpoint,
                    const DenseMap<std::tuple<int, int, int>, int64_t> &demand,
                    bool adjacent) {
    TileLike tile = tileOf(endpoint);
    DMAChannelDir dir = endpoint.getRouteDirection();
    std::optional<int> col = tile.tryGetCol(), row = tile.tryGetRow();
    if (!col || !row)
      return channels.getDMAChannelIndex(tile, dir, adjacent,
                                         endpoint.getOperation());
    int best = -1;
    int64_t bestRoom = 0;
    int limit = DMAChannelAnalysis::getDMAChannelLimit(tile, dir, adjacent);
    for (int channel = 0; channel < limit; ++channel) {
      if (!channels.isChannelFree(tile, dir, channel))
        continue;
      int64_t room = bdRoom(tile, *col, *row, channel, demand);
      if (best < 0 || room > bestRoom) {
        best = channel;
        bestRoom = room;
      }
    }
    if (best < 0)
      return -1;
    return channels.reservePinnedChannel(tile, dir, best,
                                         endpoint.getOperation());
  }

  void noteChannelOwner(InFlightDiagnostic &diag, Operation *owner,
                        std::optional<int> channel = std::nullopt) {
    auto &note = diag.attachNote(owner->getLoc());
    if (auto endpoint = dyn_cast<RouteEndpoint>(owner)) {
      note << "DMA endpoint @" << cast<SymbolOpInterface>(owner).getName();
      if (auto fifo = owner->getAttrOfType<StringAttr>("fifoName"))
        note << " for ObjectFifo @" << fifo.getValue();
      if (reachesAdjacentTile(endpoint))
        note << " requires adjacent MemTile access";
      if (channel)
        note << "; occupies channel " << *channel;
    } else if (channel) {
      note << "pre-existing " << owner->getName() << " reserves DMA channel "
           << *channel;
    }
  }

  // No allocation can move a stream port the design names, so a clash fails
  // the pass before the search rather than surfacing inside it.
  LogicalResult checkStreamPorts() {
    DMAChannelAnalysis channels(device);
    DenseSet<Operation *> packetEndpoints;
    for (auto flow : device.getOps<RouteOp>()) {
      if (!flow.getPacket())
        continue;
      for (StringRef source : flow.getSourceNames())
        packetEndpoints.insert(lookupEndpoint(source).getOperation());
      for (StringRef dest : flow.getDestinationNames())
        packetEndpoints.insert(lookupEndpoint(dest).getOperation());
    }
    bool clash = false;
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      if (endpoint.getRouteBundle() != WireBundle::Core)
        continue;
      std::optional<int> channel = endpoint.getRouteChannel();
      if (!channel)
        return endpoint->emitOpError("a stream port names its own channel");
      clash |= failed(channels.checkAIEStreamIndex(
          tileOf(endpoint), {endpoint.getRouteDirection(), *channel},
          endpoint.getOperation(),
          packetEndpoints.contains(endpoint.getOperation())));
    }
    return failure(clash);
  }

  /// Whether `endpoint` draws a DMA channel of its own: a core's stream port
  /// is named by the design (checkStreamPorts has claimed it), and an end in
  /// channelSharers takes another's channel, so neither is counted where
  /// channels are planned.
  bool drawsOwnChannel(RouteEndpoint endpoint) {
    return endpoint.getRouteBundle() != WireBundle::Core &&
           !channelSharers.contains(endpoint.getOperation());
  }

  LogicalResult assignChannels(DMAChannelAnalysis &channels,
                               bool diagnose = true) {
    channelAssignments.clear();
    channelFailure = nullptr;
    bdShortfall = false;
    SmallVector<RouteEndpoint> pending;
    // BDs per placed (col, row, channel) as channels are handed out, for
    // choosing among a tile's channels by the BD ids they can use.
    DenseMap<std::tuple<int, int, int>, int64_t> demand = fixedBDs;
    for (auto &[at, bds] : pinnedBDs)
      demand[at] += bds;
    auto bdsOf = [&](RouteEndpoint endpoint) -> int64_t {
      if (auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation()))
        return dma.getNumBDs();
      if (isProgramOwned(endpoint))
        return endpointBDs(endpoint.getOperation());
      return 0;
    };
    auto noteDemand = [&](RouteEndpoint endpoint, int channel) {
      TileLike tile = tileOf(endpoint);
      auto col = tile.tryGetCol(), row = tile.tryGetRow();
      if (col && row)
        demand[{*col, *row, channel}] += bdsOf(endpoint);
    };
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      DMAChannelDir dir = endpoint.getRouteDirection();
      std::optional<int> channel = endpoint.getRouteChannel();
      if (!drawsOwnChannel(endpoint))
        continue;

      if (channel) {
        if (reachesAdjacentTile(endpoint) &&
            *channel >= DMAChannelAnalysis::getDMAChannelLimit(tileOf(endpoint),
                                                               dir, true)) {
          channelFailure = endpoint;
          if (!diagnose)
            return failure();
          return endpoint->emitOpError("pinned ")
                 << stringifyDMAChannelDir(dir) << " DMA channel " << *channel
                 << " cannot access adjacent MemTile buffers or locks";
        }
        if (channels.reservePinnedChannel(tileOf(endpoint), dir, *channel,
                                          endpoint.getOperation()) < 0) {
          if (!diagnose)
            return failure();
          auto diag = endpoint->emitOpError("pinned ");
          diag << stringifyDMAChannelDir(dir) << " DMA channel " << *channel
               << " is out of range or already in use on this tile";
          if (Operation *owner =
                  channels.getDMAChannelOwner(tileOf(endpoint), dir, *channel))
            noteChannelOwner(diag, owner, channel);
          return failure();
        }
        noteDemand(endpoint, *channel);
        continue;
      }
      pending.push_back(endpoint);
    }

    // Endpoints reaching a spilled buffer draw from a restricted channel
    // range, so they are served before the unrestricted ones. Channels the
    // design programs itself come after the objectFIFOs', whose BDs they are
    // then chosen around, the most demanding first. The structural rank
    // settles the rest, so the order ops were written in never does.
    DenseMap<Operation *, int64_t> programBDs;
    for (auto endpoint : pending)
      if (isProgramOwned(endpoint))
        programBDs[endpoint.getOperation()] =
            endpointBDs(endpoint.getOperation());
    llvm::stable_sort(pending, [&](RouteEndpoint a, RouteEndpoint b) {
      bool aAdjacent = reachesAdjacentTile(a);
      bool bAdjacent = reachesAdjacentTile(b);
      if (aAdjacent != bAdjacent)
        return aAdjacent;
      bool aProgram = isProgramOwned(a), bProgram = isProgramOwned(b);
      if (aProgram != bProgram)
        return bProgram;
      int64_t aBDs = programBDs.lookup(a.getOperation());
      int64_t bBDs = programBDs.lookup(b.getOperation());
      if (aBDs != bBDs)
        return aBDs > bBDs;
      return structuralRank.lookup(a.getOperation()) <
             structuralRank.lookup(b.getOperation());
    });

    // On a mem tile once short of BD ids, every end gets the channel a plan of
    // the tile's BD ids gives it, before any pool there is walked whole.
    DenseMap<Operation *, int> planned;
    for (Operation *tileOp : bdPlanned) {
      TileLike tile = cast<TileLike>(tileOp);
      std::optional<int> col = tile.tryGetCol(), row = tile.tryGetRow();
      if (!col || !row)
        continue;
      SmallVector<RouteEndpoint> onTile;
      SmallVector<DMAChannelAnalysis::BDPlanEnd> ends, unweighted;
      for (RouteEndpoint endpoint : pending) {
        if (tileOf(endpoint).getOperation() != tileOp)
          continue;
        DMAChannelDir dir = endpoint.getRouteDirection();
        int limit = DMAChannelAnalysis::getDMAChannelLimit(
            tile, dir, reachesAdjacentTile(endpoint));
        onTile.push_back(endpoint);
        ends.push_back({dir, bdsOf(endpoint), limit});
        unweighted.push_back({dir, 0, limit});
      }
      auto free = [&](DMAChannelDir dir, int channel) {
        return channels.isChannelFree(tile, dir, channel);
      };
      const AIETargetModel &target = device.getTargetModel();
      std::optional<SmallVector<int>> plan = DMAChannelAnalysis::planBDChannels(
          target, *col, *row, ends, free,
          [&](int channel) { return demand.lookup({*col, *row, channel}); });
      if (plan) {
        for (auto [endpoint, channel] : llvm::zip_equal(onTile, *plan))
          planned[endpoint.getOperation()] = channel;
        continue;
      }
      // Too few channels is reported, or shared, as it is anywhere.
      if (!DMAChannelAnalysis::planBDChannels(target, *col, *row, unweighted,
                                              free, [](int) { return 0; }))
        continue;
      RouteEndpoint deepest =
          *llvm::max_element(onTile, [&](RouteEndpoint a, RouteEndpoint b) {
            return bdsOf(a) < bdsOf(b);
          });
      channelFailure = deepest;
      bdShortfall = true;
      if (!diagnose)
        return failure();
      int64_t total = 0;
      for (const DMAChannelAnalysis::BDPlanEnd &end : ends)
        total += end.bds;
      return deepest->emitOpError("needs ")
             << bdsOf(deepest)
             << " buffer descriptors, but no choice of channels fits its "
                "tile's "
             << ends.size() << " unpinned DMA ends, " << total
             << " buffer descriptors in all, into the BD ids their channels "
                "share, and no pool on the tile is left to walk whole";
    }

    for (auto endpoint : pending) {
      DMAChannelDir dir = endpoint.getRouteDirection();
      TileLike tile = tileOf(endpoint);
      bool adjacent = reachesAdjacentTile(endpoint);
      if (auto it = planned.find(endpoint.getOperation());
          it != planned.end()) {
        channelAssignments[endpoint.getOperation()] =
            channels.reservePinnedChannel(tile, dir, it->second,
                                          endpoint.getOperation());
        noteDemand(endpoint, it->second);
        continue;
      }
      // An objectFIFO end takes the first free channel, unless that channel's
      // BD ids have too few left for it: then the one with the most left.
      bool roomy = true;
      auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation());
      std::optional<int> col = tile.tryGetCol(), row = tile.tryGetRow();
      if (dma && col && row) {
        int limit = DMAChannelAnalysis::getDMAChannelLimit(tile, dir, adjacent);
        for (int free = 0; free < limit; ++free) {
          if (!channels.isChannelFree(tile, dir, free))
            continue;
          roomy = bdRoom(tile, *col, *row, free, demand) >= dma.getNumBDs();
          break;
        }
      }
      int channel = isProgramOwned(endpoint) || !roomy
                        ? chooseChannel(channels, endpoint, demand, adjacent)
                        : channels.getDMAChannelIndex(tile, dir, adjacent,
                                                      endpoint.getOperation());
      if (channel >= 0 && dma && col && row && tile.isMemTile() &&
          bdRoom(tile, *col, *row, channel, demand) < dma.getNumBDs()) {
        channelFailure = endpoint;
        bdShortfall = true;
        if (!diagnose)
          return failure();
        return endpoint->emitOpError("needs ")
               << dma.getNumBDs() << " buffer descriptors, but channel "
               << channel << " of its tile has "
               << bdRoom(tile, *col, *row, channel, demand)
               << " left beside the channels sharing its BD ids";
      }
      if (channel < 0) {
        channelFailure = endpoint;
        if (!diagnose)
          return failure();
        int capacity =
            DMAChannelAnalysis::getDMAChannelLimit(tile, dir, adjacent);
        auto diag =
            tile.emitOpError(dir == DMAChannelDir::MM2S
                                 ? "number of output DMA channel exceeded!"
                                 : "number of input DMA channel exceeded!");
        diag << " requires at least " << capacity + 1 << " "
             << stringifyDMAChannelDir(dir) << " channels, but capacity is "
             << capacity;
        if (adjacent) {
          diag << " for adjacent MemTile access";
        }
        for (int i = 0; i < capacity; ++i)
          if (Operation *owner = channels.getDMAChannelOwner(tile, dir, i))
            noteChannelOwner(diag, owner, i);
        noteChannelOwner(diag, endpoint.getOperation());
        return failure();
      }
      channelAssignments[endpoint.getOperation()] = channel;
      noteDemand(endpoint, channel);
    }
    for (auto [sharer, owner] : channelSharers)
      channelAssignments[sharer] = channelAssignments.lookup(owner);
    // Unpinned runtime BDs did not steer the choice, so the image serves any
    // runtime sequence pinning the same ids, but they hold BDs while they
    // run, which the overcommit warning counts.
    for (auto &[at, bds] : taskBDs)
      demand[at] += bds;
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      TileLike tile = tileOf(endpoint);
      std::optional<int> col = tile.tryGetCol(), row = tile.tryGetRow();
      std::optional<int> channel = endpoint.getRouteChannel();
      if (auto it = channelAssignments.find(endpoint.getOperation());
          it != channelAssignments.end())
        channel = it->second;
      if (isProgramOwned(endpoint) && channel && col && row)
        demand[{*col, *row, *channel}] +=
            endpointTaskBDs(endpoint.getOperation());
    }
    channelDemand = std::move(demand);
    return success();
  }

  /// Gives one more shim end of `exhausted` the channel of ends it is never in
  /// flight with, in the order ShimTransferSpans groups them, so only as many
  /// ends share as the tile needs. Packet headers tell a shared channel's ends
  /// apart.
  bool planShimSharing(TileLike exhausted, DMAChannelDir dir) {
    // Without --aie-objectfifo-verify there are no gates.
    ShimTransferSpans spans(
        device,
        [&](StringAttr symbol) -> std::optional<ShimTransferSpans::End> {
          auto endpoint = SymbolTable::lookupNearestSymbolFrom<RouteEndpointOp>(
              device, symbol);
          if (!endpoint || !endpoint.getFifoName())
            return std::nullopt;
          return ShimTransferSpans::End{endpoint.getFifoNameAttr(),
                                        endpoint.getRouteDirection() ==
                                            DMAChannelDir::MM2S};
        },
        ShimTransferSpans::provenGates(device));
    SmallVector<Operation *> candidates;
    SmallVector<StringAttr> fifos;
    for (auto endpoint : device.getOps<RouteEndpointOp>()) {
      if (endpoint.getBundle() != WireBundle::DMA ||
          endpoint.getChannelIndex() || !endpoint.getFifoName() ||
          endpoint.getTile() != exhausted->getResult(0) ||
          endpoint.getRouteDirection() != dir)
        continue;
      candidates.push_back(endpoint);
      fifos.push_back(endpoint.getFifoNameAttr());
    }
    for (auto &members : spans.groups(fifos))
      for (unsigned sharer : llvm::drop_begin(members))
        if (channelSharers
                .insert({candidates[sharer], candidates[members.front()]})
                .second)
          return true;
    return false;
  }

  /// Whether `pool` can be walked whole (ObjectFifoPoolOp::canWalkWhole),
  /// its ends taking no turns but those in `regrouped`, whose turns are being
  /// chosen again.
  bool canWalkWhole(ObjectFifoPoolOp pool,
                    ArrayRef<ObjectFifoDmaEndpointOp> regrouped = {}) {
    return pool && pool.canWalkWhole() &&
           llvm::all_of(device.getOps<ObjectFifoDmaEndpointOp>(),
                        [&](ObjectFifoDmaEndpointOp end) {
                          return end.getPoolOp() != pool || !end.getTurns() ||
                                 llvm::is_contained(regrouped, end);
                        });
  }

  /// The DMA ends `dmas` of one mem tile that draw a channel of their own,
  /// and their pools in structural order, as a plan of the tile's BD ids
  /// weighs them with every buffer at home.
  struct HomeBDPlan {
    SmallVector<ObjectFifoPoolOp> pools;
    SmallVector<DMAChannelAnalysis::BDWalkPool> walkable;
    SmallVector<DMAChannelAnalysis::BDPlanEnd> ends;
    SmallVector<unsigned> poolOf;
  };
  HomeBDPlan homeBDPlan(ArrayRef<ObjectFifoDmaEndpointOp> dmas) {
    HomeBDPlan plan;
    for (auto dma : dmas)
      if (!llvm::is_contained(plan.pools, dma.getPoolOp()))
        plan.pools.push_back(dma.getPoolOp());
    llvm::stable_sort(plan.pools, [&](ObjectFifoPoolOp a, ObjectFifoPoolOp b) {
      return structuralRank.lookup(a) < structuralRank.lookup(b);
    });
    for (ObjectFifoPoolOp pool : plan.pools)
      plan.walkable.push_back(
          {pool.getDepth(), pool.getObjectSizeInBytes(), canWalkWhole(pool)});
    for (auto dma : dmas) {
      auto endpoint = cast<RouteEndpoint>(dma.getOperation());
      if (!drawsOwnChannel(endpoint))
        continue;
      DMAChannelDir dir = endpoint.getRouteDirection();
      plan.ends.push_back({dir, dma.getNumBDs(),
                           DMAChannelAnalysis::getDMAChannelLimit(
                               tileOf(endpoint), dir, false)});
      plan.poolOf.push_back(llvm::find(plan.pools, dma.getPoolOp()) -
                            plan.pools.begin());
    }
    return plan;
  }

  /// Walks the pools a plan of the BD ids at home needs (walkHomeBDPools);
  /// when that plan needs none, the pool on `tile`
  /// DMAChannelAnalysis::nextPoolToWalk picks, for what only the tile's real
  /// channels are short of. False when none is left.
  bool walkForBDShortfall(TileLike tile,
                          MutableArrayRef<ObjectFifoPoolOp> pools) {
    if (walkHomeBDPools(pools))
      return true;
    SmallVector<ObjectFifoDmaEndpointOp> dmas;
    for (auto dma : device.getOps<ObjectFifoDmaEndpointOp>())
      if (sameTile(dma.getTile(), tile->getResult(0)) && dma.getPoolOp())
        dmas.push_back(dma);
    HomeBDPlan plan = homeBDPlan(dmas);
    std::optional<unsigned> next = DMAChannelAnalysis::nextPoolToWalk(
        plan.ends, plan.poolOf, plan.walkable,
        SmallVector<bool>(plan.pools.size(), false));
    if (next)
      plan.pools[*next].setContiguous(true);
    return next.has_value();
  }

  /// Walks whole, on each placed mem tile, a set of
  /// DMAChannelAnalysis::walkSetsThatFit's, chosen by placing `pools` as the
  /// SA placer does; every walkable pool when none fits. Those tiles then take
  /// their channels from that plan. False when nothing is left to walk.
  bool walkHomeBDPools(MutableArrayRef<ObjectFifoPoolOp> pools) {
    std::map<std::pair<int, int>, SmallVector<ObjectFifoDmaEndpointOp>> ends;
    for (auto dma : device.getOps<ObjectFifoDmaEndpointOp>()) {
      TileLike tile = tileOf(cast<RouteEndpoint>(dma.getOperation()));
      auto col = tile.tryGetCol(), row = tile.tryGetRow();
      if (tile.isMemTile() && col && row && dma.getPoolOp())
        ends[{*col, *row}].push_back(dma);
    }
    // Walk sets per spill component, as the SA placer chooses them
    // (DMAChannelAnalysis::searchWalkProduct).
    struct Tile {
      TileID at;
      HomeBDPlan home;
      DMAChannelAnalysis::BDWalkSets found;
      SmallVector<ObjectFifoDmaEndpointOp> dmas;
      unsigned applied = 0;
    };
    SmallVector<Tile, 0> tiles;
    for (auto &[at, dmas] : ends) {
      HomeBDPlan home = homeBDPlan(dmas);
      DMAChannelAnalysis::BDWalkSets found =
          DMAChannelAnalysis::walkSetsThatFit(
              device.getTargetModel(), at.first, at.second, home.ends,
              home.poolOf, home.walkable,
              [](DMAChannelDir, int) { return true; },
              [](int) -> int64_t { return 0; });
      if (found.sets.empty() && !found.fits && !found.channelShort) {
        found.sets.emplace_back();
        found.bytes.push_back(0);
        for (auto [pool, walkable] : llvm::enumerate(home.walkable))
          if (walkable.walkable) {
            found.sets.back().push_back(pool);
            found.bytes.back() += walkable.depth * walkable.objectBytes;
          }
      }
      if (!found.sets.empty())
        tiles.push_back(
            {{at.first, at.second}, std::move(home), std::move(found), dmas});
    }
    auto setWalked = [&](Tile &tile, bool walked) {
      for (unsigned pool : tile.found.sets[tile.applied])
        tile.home.pools[pool].setContiguous(walked);
    };
    auto homeOf = [](ObjectFifoPoolOp pool) -> std::optional<TileID> {
      TileLike tile = pool.getTileLike();
      auto col = tile.tryGetCol(), row = tile.tryGetRow();
      if (!tile.isMemTile() || !col || !row)
        return std::nullopt;
      return TileID{*col, *row};
    };
    std::set<std::pair<int, int>> homeSet;
    for (ObjectFifoPoolOp pool : pools)
      if (std::optional<TileID> at = homeOf(pool))
        homeSet.insert({at->col, at->row});
    for (Tile &tile : tiles)
      homeSet.insert({tile.at.col, tile.at.row});
    SmallVector<TileID> homes;
    for (auto [col, row] : homeSet)
      homes.push_back({col, row});
    SmallVector<unsigned> componentIds =
        memTileSpillComponents(device.getTargetModel(), homes);
    std::map<std::pair<int, int>, unsigned> componentAt;
    for (auto [home, id] : llvm::zip_equal(homes, componentIds))
      componentAt[{home.col, home.row}] = id;

    bool walkedAny = false;
    std::set<unsigned> searched;
    for (Tile &first : tiles) {
      unsigned component = componentAt[{first.at.col, first.at.row}];
      if (!searched.insert(component).second)
        continue;
      SmallVector<Tile *> members;
      for (Tile &tile : tiles)
        if (componentAt[{tile.at.col, tile.at.row}] == component)
          members.push_back(&tile);
      SmallVector<ObjectFifoPoolOp> componentPools;
      for (ObjectFifoPoolOp pool : pools)
        if (std::optional<TileID> at = homeOf(pool);
            at && componentAt[{at->col, at->row}] == component)
          componentPools.push_back(pool);
      SmallVector<SmallVector<int64_t>> bytes;
      for (Tile *tile : members)
        bytes.push_back(tile->found.bytes);
      auto apply = [&](ArrayRef<unsigned> choice) {
        for (auto [tile, set] : llvm::zip_equal(members, choice)) {
          setWalked(*tile, false);
          tile->applied = set;
          setWalked(*tile, true);
        }
      };
      DMAChannelAnalysis::WalkChoice choice =
          DMAChannelAnalysis::searchWalkProduct(
              bytes,
              [&](ArrayRef<unsigned> choice) {
                apply(choice);
                SmallVector<ObjectFifoPoolOp> trial(componentPools);
                sizeOrder(trial);
                localPools.clear();
                return succeeded(planBuffers(trial));
              },
              DMAChannelAnalysis::kMaxWalkPlacements);
      apply(choice.sets);
      walkSearchStopped |= choice.stopped;
      for (Tile *tile : members) {
        if (tile->found.sets[tile->applied].empty())
          continue;
        walkedAny = true;
        for (auto dma : tile->dmas)
          bdPlanned.insert(
              tileOf(cast<RouteEndpoint>(dma.getOperation())).getOperation());
      }
    }
    return walkedAny;
  }

  /// The other end of the one route naming `endpoint`, when that route is a
  /// circuit between two objectFIFO DMA endpoints.
  /// Whether, in every runtime sequence, the host issues each transfer
  /// through `fifo` before any wait: a turn chain waiting on that fifo's
  /// objects then waits on nothing the host does after the array has moved.
  bool hostMovesFirst(StringAttr fifo) {
    for (auto seq : device.getOps<RuntimeSequenceOp>()) {
      auto names = [&](Operation *op) {
        auto uses = SymbolTable::getSymbolUses(op);
        return fifo && uses && llvm::any_of(*uses, [&](auto use) {
                 auto end = dyn_cast_or_null<RouteEndpoint>(
                     SymbolTable::lookupNearestSymbolFrom(seq,
                                                          use.getSymbolRef()));
                 return end && end.getFifoName() == fifo.getValue();
               });
      };
      auto waits = [](Operation *op) {
        return isa<AIEX::NpuDmaWaitOp, AIEX::DMAAwaitTaskOp, AIEX::NpuSyncOp,
                   AIEX::NpuMaskPollOp>(op);
      };
      bool waited = false;
      WalkResult walk = seq.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (waits(op)) {
          waited = true;
          return WalkResult::advance();
        }
        Operation *transfer = op;
        if (auto start = dyn_cast<AIEX::DMAStartTaskOp>(op))
          transfer = start.getTask().getDefiningOp();
        if (!transfer ||
            !isa<AIEX::NpuDmaMemcpyNdOp, AIEX::DMAConfigureTaskForOp,
                 AIEX::DMAStartBdChainForOp>(transfer) ||
            isa<AIEX::DMAConfigureTaskForOp>(op) || !names(transfer))
          return WalkResult::advance();
        if (waited)
          return WalkResult::interrupt();
        for (Operation *parent = op->getParentOp(); parent != seq;
             parent = parent->getParentOp())
          if (parent
                  ->walk([&](Operation *inner) {
                    return waits(inner) ? WalkResult::interrupt()
                                        : WalkResult::advance();
                  })
                  .wasInterrupted())
            return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (walk.wasInterrupted())
        return false;
    }
    return true;
  }

  ObjectFifoDmaEndpointOp circuitPeer(ObjectFifoDmaEndpointOp endpoint) {
    ObjectFifoDmaEndpointOp peer;
    StringRef name = endpoint.getSymName();
    for (RouteOp route : device.getOps<RouteOp>()) {
      SmallVector<StringRef> sources = llvm::to_vector(route.getSourceNames());
      SmallVector<StringRef> dests =
          llvm::to_vector(route.getDestinationNames());
      bool source = llvm::is_contained(sources, name);
      if (!source && !llvm::is_contained(dests, name))
        continue;
      if (peer || route.getPacket() || sources.size() != 1 || dests.size() != 1)
        return {};
      RouteEndpoint other = lookupEndpoint(source ? dests[0] : sources[0]);
      peer = other ? dyn_cast<ObjectFifoDmaEndpointOp>(other.getOperation())
                   : ObjectFifoDmaEndpointOp();
      if (!peer)
        return {};
    }
    return peer;
  }

  /// The one core endpoint working through `endpoint`'s pool from the other
  /// side.
  ObjectFifoCoreEndpointOp coreEndOf(ObjectFifoDmaEndpointOp endpoint) {
    ObjectFifoCoreEndpointOp found;
    for (auto core : device.getOps<ObjectFifoCoreEndpointOp>()) {
      if (core.getPool() != endpoint.getPool())
        continue;
      if (found || core.drains() == endpoint.drains())
        return {};
      found = core;
    }
    return found;
  }

  static bool canTakeTurns(ObjectFifoDmaEndpointOp endpoint) {
    return endpoint && !endpoint.getChannelIndex() && !endpoint.getDispatch() &&
           !endpoint.getPacketAttr() && !endpoint.getIterCount() &&
           endpoint.getRepeat() == 1 && !endpoint.walksStream();
  }

  /// Whether a chain moving `turns` over `ends` can walk each contiguous pool
  /// with one descriptor per turn of its unrolled passes, each stepping over
  /// the others' objects: they must divide the pool's objects.
  static bool contiguousWalkersFit(ArrayRef<ObjectFifoDmaEndpointOp> ends,
                                   ArrayRef<unsigned> turns) {
    SmallVector<std::pair<int64_t, int64_t>> counts;
    for (auto [i, constEnd] : llvm::enumerate(ends)) {
      ObjectFifoDmaEndpointOp end = constEnd;
      counts.push_back({llvm::count(turns, i), end.getPoolOp().getContiguous()
                                                   ? 0
                                                   : end.getNumDescriptors()});
    }
    int64_t passes = turnPasses(counts);
    for (auto [i, constEnd] : llvm::enumerate(ends)) {
      ObjectFifoDmaEndpointOp end = constEnd;
      if (end.getPoolOp().getContiguous() &&
          (end.walksStream() ||
           end.getPoolOp().getDepth() % (counts[i].first * passes)))
        return false;
    }
    return true;
  }

  /// Gives `ends` one channel, the k-th object belonging to `ends[order[k]]`.
  void takeTurns(MutableArrayRef<ObjectFifoDmaEndpointOp> ends,
                 ArrayRef<unsigned> order) {
    SmallVector<Attribute> names;
    for (unsigned i : order)
      names.push_back(FlatSymbolRefAttr::get(ends[i].getSymNameAttr()));
    ArrayAttr turns = builder.getArrayAttr(names);
    Operation *owner = ends[order.front()];
    channelSharers.erase(owner);
    for (ObjectFifoDmaEndpointOp end : ends) {
      end.setTurnsAttr(turns);
      if (end != owner)
        channelSharers[end] = owner;
    }
  }

  /// Gives one more end of `exhausted` the channel of others whose objects
  /// the same core needs in a fixed order, read off its program, so only as
  /// many ends share as the tile needs. A circuit carries one stream, so the
  /// ends at the routes' other ends take the same turns.
  bool planTurnSharing(TileLike exhausted, DMAChannelDir dir) {
    // `coreEndOf`/`circuitPeer` each rescan the whole device; neither's
    // domain (core endpoints, routes) changes within this function (only
    // pool `contiguous` flags and endpoints' `turns` attrs do), so caching
    // them once here turns every lookup below from O(device size) into O(1).
    DenseMap<Operation *, ObjectFifoCoreEndpointOp> coreEndCache;
    DenseMap<Operation *, ObjectFifoDmaEndpointOp> circuitPeerCache;
    DenseMap<Operation *, SmallVector<ObjectFifoDmaEndpointOp>> poolEnds;
    for (auto dma : device.getOps<ObjectFifoDmaEndpointOp>()) {
      coreEndCache[dma] = coreEndOf(dma);
      circuitPeerCache[dma] = circuitPeer(dma);
      poolEnds[dma.getPoolOp()].push_back(dma);
    }
    // Each circuit destination's one sender, a broadcast's included, and the
    // ends a circuit joins to the shim alone, which wait on the host only.
    DenseMap<Operation *, ObjectFifoDmaEndpointOp> senderOf;
    DenseSet<Operation *> byShim;
    auto onShim = [](RouteEndpoint end) {
      auto tile = end.getTile().getDefiningOp<TileLike>();
      return tile && tile.isShimTile();
    };
    for (RouteOp route : device.getOps<RouteOp>()) {
      SmallVector<StringRef> sources = llvm::to_vector(route.getSourceNames());
      RouteEndpoint source = sources.size() == 1 && !route.getPacket()
                                 ? lookupEndpoint(sources[0])
                                 : RouteEndpoint();
      if (!source)
        continue;
      auto sender = dyn_cast<ObjectFifoDmaEndpointOp>(source.getOperation());
      SmallVector<RouteEndpoint> dests;
      for (StringRef dest : route.getDestinationNames())
        dests.push_back(lookupEndpoint(dest));
      for (RouteEndpoint end : dests) {
        if (end && sender)
          senderOf[end.getOperation()] = sender;
        if (end && onShim(source))
          byShim.insert(end.getOperation());
      }
      if (llvm::all_of(dests,
                       [&](RouteEndpoint end) { return end && onShim(end); }))
        byShim.insert(source.getOperation());
    }
    auto cachedCoreEnd = [&](ObjectFifoDmaEndpointOp dma) {
      return coreEndCache.lookup(dma);
    };
    auto cachedCircuitPeer = [&](ObjectFifoDmaEndpointOp dma) {
      return circuitPeerCache.lookup(dma);
    };
    SmallVector<ObjectFifoDmaEndpointOp> candidates;
    for (auto dma : device.getOps<ObjectFifoDmaEndpointOp>())
      if (sameTile(dma.getTile(), exhausted->getResult(0)) &&
          dma.getRouteDirection() == dir && cachedCoreEnd(dma) &&
          canTakeTurns(dma.getTurns() ? turnEndpoints(dma).front() : dma) &&
          canTakeTurns(cachedCircuitPeer(dma)))
        candidates.push_back(dma);
    llvm::stable_sort(candidates, [&](Operation *a, Operation *b) {
      return structuralRank.lookup(a) < structuralRank.lookup(b);
    });
    LLVM_DEBUG({
      llvm::dbgs() << "turns on " << exhausted->getLoc() << ": "
                   << candidates.size() << " candidate ends\n";
      for (auto dma : device.getOps<ObjectFifoDmaEndpointOp>())
        if (sameTile(dma.getTile(), exhausted->getResult(0)) &&
            dma.getRouteDirection() == dir)
          llvm::dbgs() << "  " << dma.getSymName() << ": core end "
                       << bool(cachedCoreEnd(dma)) << ", takes turns "
                       << canTakeTurns(dma) << ", peer takes turns "
                       << canTakeTurns(cachedCircuitPeer(dma)) << "\n";
    });
    auto groupOf = [](ObjectFifoDmaEndpointOp end) {
      SmallVector<ObjectFifoDmaEndpointOp> group = turnEndpoints(end);
      return group.empty() ? SmallVector<ObjectFifoDmaEndpointOp>{end} : group;
    };
    for (auto [i, a] : llvm::enumerate(candidates)) {
      for (ObjectFifoDmaEndpointOp b : llvm::drop_begin(candidates, i + 1)) {
        SmallVector<ObjectFifoDmaEndpointOp> group = groupOf(a);
        if (llvm::is_contained(group, b))
          continue;
        llvm::append_range(group, groupOf(b));
        SmallVector<ObjectFifoCoreEndpointOp> cores;
        SmallVector<ObjectFifoDmaEndpointOp> peers;
        SmallVector<int64_t> depths;
        for (ObjectFifoDmaEndpointOp end : group) {
          cores.push_back(cachedCoreEnd(end));
          peers.push_back(cachedCircuitPeer(end));
          depths.push_back(end.getPoolOp().getDepth());
        }
        if (llvm::any_of(peers, [&](ObjectFifoDmaEndpointOp peer) {
              return !sameTile(peer.getTile(), peers.front().getTile()) ||
                     llvm::any_of(turnEndpoints(peer), [&](auto other) {
                       return !llvm::is_contained(peers, other);
                     });
            })) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  " << a.getSymName() << " + " << b.getSymName()
                     << ": senders on different tiles\n");
          continue;
        }
        CoreTurns turns;
        std::string reason;
        if (succeeded(turns.read(cores, reason)) && !turns.keepsUp(depths))
          reason = "would stall";
        // The peers' chain moves the same turns, so a core working through
        // their pools, or through the pools that feed them circuit by circuit,
        // must keep up with that order too, through all those pools' depths.
        // A peer reached from the shim waits on the host, so the host must
        // move it before it waits on anything; one whose feeding core the walk
        // cannot find refuses the turns.
        SmallVector<ObjectFifoCoreEndpointOp> peerCores;
        SmallVector<int64_t> peerDepths;
        SmallVector<unsigned> fed(peers.size(), ~0u);
        bool lost = false, hostWaits = false;
        for (auto [i, peer] : llvm::enumerate(peers)) {
          int64_t depth = peer.getPoolOp().getDepth();
          ObjectFifoCoreEndpointOp core = cachedCoreEnd(peer);
          SmallPtrSet<Operation *, 8> seen;
          for (ObjectFifoDmaEndpointOp at = peer; !core;) {
            SmallVector<ObjectFifoDmaEndpointOp> across;
            for (ObjectFifoDmaEndpointOp end : poolEnds.lookup(at.getPoolOp()))
              if (end.drains() != at.drains())
                across.push_back(end);
            if (!seen.insert(at).second || across.size() != 1) {
              lost = true;
              break;
            }
            if (byShim.contains(across.front())) {
              hostWaits |= !hostMovesFirst(across.front().getFifoNameAttr());
              break;
            }
            at = across.front().drains() ? cachedCircuitPeer(across.front())
                                         : senderOf.lookup(across.front());
            if (!at) {
              lost = true;
              break;
            }
            depth += at.getPoolOp().getDepth();
            core = cachedCoreEnd(at);
          }
          if (!core)
            continue;
          if (llvm::is_contained(peerCores, core)) {
            lost = true;
            continue;
          }
          fed[i] = peerCores.size();
          peerCores.push_back(core);
          peerDepths.push_back(depth);
        }
        if (reason.empty() && lost)
          reason = "a peer's feeding core is not known, or feeds two peers";
        if (reason.empty() && hostWaits)
          reason = "the host waits before it moves a peer fed by the shim";
        if (reason.empty() && !peerCores.empty()) {
          SmallVector<unsigned> peerOrder;
          for (unsigned turn : turns.turns())
            if (fed[turn] != ~0u)
              peerOrder.push_back(fed[turn]);
          CoreTurns peerTurns;
          if (llvm::any_of(peerCores, [&](ObjectFifoCoreEndpointOp c) {
                return !sameTile(c.getTile(), peerCores.front().getTile());
              }))
            reason = "several cores feed the peers";
          else if (succeeded(peerTurns.read(peerCores, reason)) &&
                   !peerTurns.keepsUp(peerOrder, peerDepths))
            reason = "the core at the peers' far end would stall";
        }
        // Over plain pools the chain unrolls to whole passes over each; past
        // the peers' channel BDs, walking them whole takes one a turn.
        bool overflows = false;
        if (reason.empty()) {
          SmallVector<std::pair<int64_t, int64_t>> counts;
          for (auto [i, constPeer] : llvm::enumerate(peers)) {
            ObjectFifoDmaEndpointOp peer = constPeer;
            counts.push_back({llvm::count(turns.turns(), i),
                              peer.getPoolOp().getContiguous()
                                  ? 0
                                  : peer.getNumDescriptors()});
          }
          TileLike tile = peers.front().getTileLike();
          std::optional<int> col = tile.tryGetCol(), row = tile.tryGetRow();
          overflows =
              col && row &&
              turnPasses(counts) * static_cast<int64_t>(turns.turns().size()) >
                  device.getTargetModel().getNumBDsForChannel(*col, *row, 0);
        }
        // Beside a pool walked whole, the others' descriptors must come round
        // each pass, which walking them whole as well lines up.
        // A pool whose other end already takes turns may be walked whole too,
        // where those turns still line up.
        SmallVector<ObjectFifoPoolOp> walked;
        if (reason.empty() &&
            (overflows || !contiguousWalkersFit(peers, turns.turns())))
          for (ObjectFifoDmaEndpointOp peer : peers) {
            ObjectFifoPoolOp pool = peer.getPoolOp();
            SmallVector<ObjectFifoDmaEndpointOp> others;
            for (auto end : device.getOps<ObjectFifoDmaEndpointOp>())
              if (end.getPoolOp() == pool && end.getTurns() &&
                  !llvm::is_contained(peers, end))
                others.push_back(end);
            SmallVector<ObjectFifoDmaEndpointOp> ignored(peers);
            llvm::append_range(ignored, others);
            if (!canWalkWhole(pool, ignored)) {
              LLVM_DEBUG(llvm::dbgs() << "  " << pool.getSymName()
                                      << " cannot be walked whole\n");
              continue;
            }
            pool.setContiguous(true);
            bool linedUp =
                llvm::all_of(others, [&](ObjectFifoDmaEndpointOp end) {
                  SmallVector<ObjectFifoDmaEndpointOp> ends =
                      turnEndpoints(end);
                  SmallVector<unsigned> order;
                  for (auto name :
                       end.getTurnsAttr().getAsRange<FlatSymbolRefAttr>())
                    order.push_back(
                        llvm::find_if(ends,
                                      [&](ObjectFifoDmaEndpointOp e) {
                                        return e.getSymNameAttr() ==
                                               name.getAttr();
                                      }) -
                        ends.begin());
                  return contiguousWalkersFit(ends, order);
                });
            LLVM_DEBUG(
                llvm::dbgs()
                << "  " << pool.getSymName() << " walked whole"
                << (linedUp ? "" : ", but its other turns would not line up")
                << "\n");
            if (linedUp)
              walked.push_back(pool);
            else
              pool.setContiguous(false);
          }
        if (reason.empty() && (!contiguousWalkersFit(group, turns.turns()) ||
                               !contiguousWalkersFit(peers, turns.turns()))) {
          reason = "a contiguous pool's objects do not divide among its turns";
          for (ObjectFifoPoolOp pool : walked)
            pool.setContiguous(false);
        }
        if (!reason.empty()) {
          LLVM_DEBUG(llvm::dbgs() << "  " << a.getSymName() << " + "
                                  << b.getSymName() << ": " << reason << "\n");
          continue;
        }
        takeTurns(group, turns.turns());
        takeTurns(peers, turns.turns());
        return true;
      }
    }
    return false;
  }

  /// A route reaching or leaving a shared shim channel carries a packet
  /// header, its id distinct from those of the routes it shares the channel
  /// with. Other shared channels reuse the same ids: the router keeps flows
  /// with one id apart wherever their destinations differ.
  LogicalResult headShimSharingRoutes() {
    DenseMap<Operation *, Operation *> channelOf;
    for (auto [sharer, owner] : channelSharers) {
      if (!tileOf(cast<RouteEndpoint>(owner)).isShimTile())
        continue;
      channelOf[sharer] = owner;
      channelOf[owner] = owner;
    }
    auto channelsOf = [&](RouteOp route) {
      SmallVector<Operation *> channels;
      auto note = [&](StringRef name) {
        if (auto endpoint = lookupEndpoint(name);
            endpoint && channelOf.count(endpoint.getOperation()))
          channels.push_back(channelOf.lookup(endpoint.getOperation()));
      };
      llvm::for_each(route.getSourceNames(), note);
      llvm::for_each(route.getDestinationNames(), note);
      return channels;
    };
    DenseMap<Operation *, DenseSet<int>> idsOn;
    for (auto route : device.getOps<RouteOp>())
      if (PacketInfoAttr packet = route.getPacketAttr();
          packet && packet.isAssigned())
        for (Operation *channel : channelsOf(route))
          idsOn[channel].insert(packet.assignedId());
    PacketIdSpace ids(device);
    SmallVector<int> reused;
    for (auto route : device.getOps<RouteOp>()) {
      SmallVector<Operation *> channels = channelsOf(route);
      if (route.getPacket() || channels.empty())
        continue;
      auto freeOnAll = [&](int id) {
        return llvm::none_of(channels, [&](Operation *channel) {
          return idsOn[channel].contains(id);
        });
      };
      std::optional<int> id;
      if (auto *it = llvm::find_if(reused, freeOnAll); it != reused.end())
        id = *it;
      else if ((id = ids.takeLowestFrom()))
        reused.push_back(*id);
      else
        return route.emitOpError("shares a shim channel, but no packet id is "
                                 "left to tell its transfers apart");
      for (Operation *channel : channels)
        idsOn[channel].insert(*id);
      route.setPacketAttr(PacketInfoAttr::get(builder.getContext(), 0,
                                              static_cast<uint16_t>(*id)));
    }
    return success();
  }

  /// Channels are chosen by BD room but never refused for lack of it: a task
  /// that frees its BDs before the next starts lets demand exceed the BDs.
  /// Otherwise BD id assignment fails, so say where.
  void warnOvercommittedBDs() {
    const AIETargetModel &target = device.getTargetModel();
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      auto it = channelAssignments.find(endpoint.getOperation());
      TileLike tile = tileOf(endpoint);
      std::optional<int> col = tile.tryGetCol(), row = tile.tryGetRow();
      if (it == channelAssignments.end() || !isProgramOwned(endpoint) || !col ||
          !row)
        continue;
      int channel = it->second;
      int64_t room = bdRoom(tile, *col, *row, channel, channelDemand);
      if (room >= 0)
        continue;
      int64_t numBDs = target.getNumBDsForChannel(*col, *row, channel);
      endpoint->emitWarning("channel ")
          << channel << " can use " << numBDs
          << " BDs, but it and the channels sharing them need " << numBDs - room
          << " at once; BD id assignment fails unless tasks free their BDs "
             "before others start";
    }
  }

  /// Keep a successful largest-first allocation unchanged. On channel
  /// exhaustion, try keeping an affected pool's resources local, cheapest
  /// first. Existing buffers and locks never move. Each DMA accesses every
  /// object in its pool, even when it selects only one segment. Reserving that
  /// pool once can therefore free several restricted channels. This is a
  /// bounded repair of the greedy buffer placement, not an exhaustive solver
  /// for tile placement or memory packing.
  LogicalResult planAllocation(ArrayRef<ObjectFifoPoolOp> pools,
                               std::set<std::vector<unsigned>> &tried) {
    std::vector<unsigned> key;
    for (auto [index, pool] : llvm::enumerate(pools)) {
      if (auto tile = localPools.lookup(pool)) {
        auto placed = cast<TileLike>(tile.getDefiningOp());
        auto col = placed.tryGetCol(), row = placed.tryGetRow();
        assert(col && row && "locality repair requires resolved coordinates");
        key.push_back(index);
        key.push_back(*col);
        key.push_back(*row);
      }
    }
    if (tried.size() >= 64 || !tried.insert(key).second ||
        failed(planBuffers(pools)) || failed(planLocks(pools)))
      return failure();
    DMAChannelAnalysis channels(device);
    if (succeeded(assignChannels(channels, /*diagnose=*/false)))
      return success();
    if (!channelFailure || !tileOf(channelFailure).isMemTile() ||
        !tileOf(channelFailure).tryGetCol() ||
        !tileOf(channelFailure).tryGetRow())
      return failure();

    Value localTile = channelFailure.getTile();
    SmallVector<ObjectFifoPoolOp> candidates;
    for (auto dma : device.getOps<ObjectFifoDmaEndpointOp>()) {
      auto endpoint = cast<RouteEndpoint>(dma.getOperation());
      auto pool = dma.getPoolOp();
      if (!pool || localPools.contains(pool) ||
          !sameTile(endpoint.getTile(), channelFailure.getTile()) ||
          endpoint.getRouteDirection() != channelFailure.getRouteDirection() ||
          (channelFailure.getRouteChannel() && endpoint != channelFailure) ||
          (endpoint.getRouteChannel() && endpoint != channelFailure) ||
          !reachesAdjacentTile(endpoint) ||
          llvm::is_contained(candidates, pool))
        continue;
      candidates.push_back(pool);
    }
    llvm::stable_sort(candidates, [&](ObjectFifoPoolOp a, ObjectFifoPoolOp b) {
      int64_t aBytes =
          a.getBuffers() ? 0 : a.getObjectSizeInBytes() * a.getDepth();
      int64_t bBytes =
          b.getBuffers() ? 0 : b.getObjectSizeInBytes() * b.getDepth();
      if (aBytes != bBytes)
        return aBytes < bBytes;
      return structuralRank.lookup(a) < structuralRank.lookup(b);
    });
    for (auto pool : candidates) {
      localPools[pool] = localTile;
      if (succeeded(planAllocation(pools, tried)))
        return success();
      localPools.erase(pool);
    }
    return failure();
  }

  /// A packet-switched flow shares the stream with others, so every buffer
  /// descriptor the source emits has to carry the packet header.
  /// One packet flow carries every source of the route: each stamps the same
  /// header, and their packets share the destination channels.
  LogicalResult lowerPacketFlow(RouteOp flow, ArrayRef<RouteEndpoint> sources,
                                PacketInfoAttr header) {
    int packetID = header.assignedId();
    for (auto [name, endpoint] : llvm::zip_equal(
             flow.getSources().getAsRange<FlatSymbolRefAttr>(), sources)) {
      RouteEndpoint source = endpoint;
      // Written out in turn order after the last route.
      if (auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(*source);
          dma && dma.getDispatch()) {
        dispatchHeaders[dma][*flow.getDestinationNames().begin()] = header;
        continue;
      }
      if (sourcedFlows.at(name.getAttr()).size() == 1)
        source.setRoutePacket(header);
    }

    builder.setInsertionPoint(flow);
    auto packetFlow = PacketFlowOp::create(
        builder, flow.getLoc(),
        builder.getIntegerAttr(builder.getI8Type(), packetID),
        flow.getKeepPktHeader() ? builder.getBoolAttr(true) : nullptr, nullptr);
    OpBuilder::InsertionGuard g(builder);
    Block &ports = packetFlow.getRegion().emplaceBlock();
    builder.setInsertionPointToStart(&ports);
    EndOp::create(builder, flow.getLoc());

    builder.setInsertionPointToStart(&ports);
    for (RouteEndpoint source : sources) {
      PacketSourceOp::create(builder, flow.getLoc(), source.getTile(),
                             source.getRouteBundle(), channelOf(source));
    }
    for (StringRef destName : flow.getDestinationNames()) {
      auto dest = lookupEndpoint(destName);
      PacketDestOp::create(builder, flow.getLoc(), dest.getTile(),
                           dest.getRouteBundle(), channelOf(dest));
    }
    return success();
  }

  /// Assigned before flows are lowered, so every endpoint has one by now.
  int channelOf(RouteEndpoint endpoint) {
    std::optional<int> channel = endpoint.getRouteChannel();
    assert(channel && "channels are assigned before flows are lowered");
    return *channel;
  }

  RouteEndpoint lookupEndpoint(StringRef name) {
    return dyn_cast_or_null<RouteEndpoint>(SymbolTable::lookupNearestSymbolFrom(
        device, StringAttr::get(device.getContext(), name)));
  }

  LogicalResult lowerFlows() {
    DenseSet<std::tuple<Value, int, Value, int>> circuits;
    for (RouteOp flow : device.getOps<RouteOp>())
      for (auto name : flow.getSources().getAsRange<FlatSymbolRefAttr>())
        sourcedFlows[name.getAttr()].push_back(flow);
    for (auto flow : device.getOps<RouteOp>()) {
      SmallVector<RouteEndpoint> sources;
      for (StringRef name : flow.getSourceNames()) {
        sources.push_back(lookupEndpoint(name));
      }
      loweredFlows.push_back(flow);

      // Which switching a route gets was settled upstream of this pass: the
      // header is on the route, and --aie-assign-packet-ids gave it its id.
      if (std::optional<PacketInfoAttr> packet = flow.getPacket()) {
        if (!packet->isAssigned()) {
          return flow.emitOpError("has a packet header with no pkt_id; run "
                                  "--aie-assign-packet-ids before this pass");
        }
        if (failed(lowerPacketFlow(flow, sources, *packet))) {
          return failure();
        }
        continue;
      }

      // The verifier keeps a circuit route to one source.
      RouteEndpoint source = sources.front();

      int sourceChannel = channelOf(source);
      builder.setInsertionPoint(flow);
      for (StringRef destName : flow.getDestinationNames()) {
        auto dest = lookupEndpoint(destName);
        // Ends taking turns share one circuit for all their routes.
        if (!circuits
                 .insert({source.getTile(), sourceChannel, dest.getTile(),
                          channelOf(dest)})
                 .second)
          continue;
        FlowOp::create(builder, flow.getLoc(), source.getTile(),
                       source.getRouteBundle(), sourceChannel, dest.getTile(),
                       dest.getRouteBundle(), channelOf(dest));
      }
    }
    for (auto &[endpoint, byDest] : dispatchHeaders) {
      SmallVector<Attribute> headers;
      for (StringRef dest :
           endpoint.getDispatchAttr().getAsValueRange<FlatSymbolRefAttr>())
        headers.push_back(byDest.lookup(dest));
      endpoint.setDispatchPacketsAttr(builder.getArrayAttr(headers));
    }
    return success();
  }

  /// An `aiex.dma_channel_reset_for` outlives the fifo it names, so record the
  /// channels and locks it has to re-arm and point it at that record. Shim
  /// endpoints are left out: the host re-pushes those itself.
  LogicalResult bindRearmTargets() {
    if (poolLocks->rearms().empty()) {
      return success();
    }

    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (auto &[fifoName, users] : poolLocks->rearms()) {
      SmallVector<Value> channelTiles, lockValues;
      SmallVector<int32_t> channelDirs, channelIndices, lockInits;
      // The record stands for these endpoints and pools: located at each.
      SmallVector<Location> locs;

      for (auto endpoint : device.getOps<RouteEndpoint>()) {
        std::optional<int> channel = endpoint.getRouteChannel();
        if (endpoint.getFifoName() != fifoName ||
            tileOf(endpoint).isShimTile() || !channel) {
          continue;
        }
        channelTiles.push_back(endpoint.getTile());
        channelDirs.push_back(
            static_cast<int32_t>(endpoint.getRouteDirection()));
        channelIndices.push_back(*channel);
        locs.push_back(endpoint->getLoc());
      }
      for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
        if (pool.getFifoName() != fifoName || pool.getTileLike().isShimTile()) {
          continue;
        }
        locs.push_back(pool.getLoc());
        for (LockOp lock : pool.getLockOps()) {
          lockValues.push_back(lock.getResult());
          lockInits.push_back(lock.getInit().value_or(0));
        }
      }

      if (channelTiles.empty() && lockValues.empty()) {
        for (AIEX::DmaChannelResetForOp user : users) {
          user.emitOpError() << "objectFIFO '" << fifoName
                             << "' has no resident core/mem DMA channels or "
                                "locks to re-arm";
        }
        return failure();
      }

      std::string name = (fifoName + "__rearm").str();
      for (unsigned suffix = 0; lookupNamedOpIn(device, StringRef(name));
           suffix++) {
        name = (fifoName + "__rearm" + std::to_string(suffix + 1)).str();
      }

      // head_bd_ids and repeat_counts are filled in by --aie-assign-bd-ids.
      ObjectFifoRearmBindingOp::create(
          builder, FusedLoc::get(builder.getContext(), locs),
          builder.getStringAttr(name), ValueRange(channelTiles),
          ValueRange(lockValues), builder.getDenseI32ArrayAttr(channelDirs),
          builder.getDenseI32ArrayAttr(channelIndices),
          builder.getDenseI32ArrayAttr(lockInits),
          /*head_bd_ids=*/DenseI32ArrayAttr(),
          /*repeat_counts=*/DenseI32ArrayAttr());
      auto target = FlatSymbolRefAttr::get(builder.getContext(), name);
      for (AIEX::DmaChannelResetForOp user : users) {
        user.setObjfifoAttr(target);
      }
    }
    return success();
  }

  /// Give each dma_start naming an endpoint the channel it was assigned.
  LogicalResult resolveStart(DMAStartOp start, RouteEndpointOp endpoint,
                             ArrayRef<RouteOp> flows, ArrayRef<int> ids,
                             int channel) {
    DMAChannelDir dir = endpoint.getRouteDirection();
    if (start.getChannelDir() != dir)
      return start.emitOpError("starts ")
             << stringifyDMAChannelDir(start.getChannelDir()) << " on "
             << start.getEndpoint() << ", but its route makes it "
             << stringifyDMAChannelDir(dir);
    // Only a runtime task's BDs are given the packet header, so a DMA
    // program states it in each BD; --aie-assign-packet-ids made it pin one.
    bool packet = !flows.empty() && RouteOp(flows.front()).getPacket();
    if (packet) {
      SmallVector<DMABDOp> bds;
      for (Block *block : chainOf(start))
        llvm::append_range(bds, block->getOps<DMABDOp>());
      if (failed(checkHeaders(endpoint, ids, bds)))
        return failure();
    }
    start.setChannelAttr(builder.getI32IntegerAttr(channel));
    return success();
  }

  /// A packet source that no one header is stamped on needs each of `bds` to
  /// state the header of one of its flows, whose pinned ids are `ids`.
  LogicalResult checkHeaders(RouteEndpointOp endpoint, ArrayRef<int> ids,
                             ArrayRef<DMABDOp> bds) {
    for (DMABDOp bd : bds) {
      std::optional<int> id;
      if (auto packet = bd.getPacket())
        id = packet->getPktId();
      else if (auto ops = bd->getBlock()->getOps<DMABDPACKETOp>(); !ops.empty())
        id = (*ops.begin()).getPacketId();
      if (id && llvm::is_contained(ids, *id))
        continue;
      return bd.emitOpError("sends on @")
             << endpoint.getSymName()
             << " without the packet header of one of its flows, ids " << ids;
    }
    return success();
  }

  /// Each runtime task or transfer that sends on a shared source must state
  /// one of its flows' headers; a DMA program's start is resolveStart's, and
  /// other users send nothing.
  LogicalResult checkSharedSourceUsers(RouteEndpointOp endpoint,
                                       ArrayRef<int> ids) {
    auto uses = SymbolTable::getSymbolUses(endpoint, device);
    if (!uses)
      return success();
    for (const SymbolTable::SymbolUse &use : *uses) {
      Operation *user = use.getUser();
      if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(user)) {
        SmallVector<DMABDOp> bds;
        task.getBody().walk([&](DMABDOp bd) { bds.push_back(bd); });
        if (failed(checkHeaders(endpoint, ids, bds)))
          return failure();
        continue;
      }
      if (auto start = dyn_cast<AIEX::DMAStartBdChainForOp>(user)) {
        auto chain = device.lookupSymbol<BDChainOp>(start.getSymbol());
        if (!chain)
          return start.emitOpError("symbol does not reference valid BD chain");
        SmallVector<DMABDOp> bds;
        chain.getBody().walk([&](DMABDOp bd) { bds.push_back(bd); });
        if (failed(checkHeaders(endpoint, ids, bds)))
          return failure();
        continue;
      }
      auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(user);
      if (!memcpy)
        continue;
      std::optional<PacketInfoAttr> packet = memcpy.getPacket();
      if (packet && llvm::is_contained(ids, packet->getPktId()))
        continue;
      return memcpy.emitOpError("sends on @")
             << endpoint.getSymName()
             << ", a source several packet routes share, which has no one "
                "header to stamp; it needs a packet header of its own, one of "
                "ids "
             << ids;
    }
    return success();
  }

  /// A runtime task naming an endpoint is configured on the endpoint's tile
  /// and channel directly.
  void resolveTask(AIEX::DMAConfigureTaskForOp task, RouteEndpointOp endpoint,
                   int channel) {
    builder.setInsertionPoint(task);
    auto configured = AIEX::DMAConfigureTaskOp::create(
        builder, task.getLoc(), builder.getIndexType(), endpoint.getTile(),
        DMAChannelDirAttr::get(builder.getContext(),
                               endpoint.getRouteDirection()),
        builder.getI32IntegerAttr(channel),
        builder.getBoolAttr(task.getIssueToken()),
        builder.getI32IntegerAttr(task.getRepeatCount()),
        task.getRepeatCountVal(), endpoint.getPacketAttr(),
        /*out_of_order=*/UnitAttr(), /*alloc=*/FlatSymbolRefAttr());
    task.getResult().replaceAllUsesWith(configured.getResult());
    configured.getBody().takeBody(task.getBody());
    task.erase();
  }

  /// Rewrite the programs and tasks naming an endpoint to its channel while
  /// the route giving its direction is still there. A shim end with a fifo
  /// name keeps its tasks, which emitShimAllocations points at the allocation.
  LogicalResult resolveEndpointUses() {
    for (auto endpoint : device.getOps<RouteEndpointOp>()) {
      auto starts = endpointStarts.lookup(endpoint);
      auto tasks = endpointTasks.lookup(endpoint);
      auto sourced = sourcedFlows.find(endpoint.getSymNameAttr());
      ArrayRef<RouteOp> flows;
      if (sourced != sourcedFlows.end())
        flows = sourced->second;
      SmallVector<int> ids;
      for (RouteOp flow : flows)
        if (std::optional<PacketInfoAttr> packet = flow.getPacket();
            packet && packet->isAssigned())
          ids.push_back(packet->assignedId());
      bool shared = flows.size() > 1;
      if (shared && failed(checkSharedSourceUsers(endpoint, ids)))
        return failure();
      if (starts.empty() && tasks.empty())
        continue;
      std::optional<int> channel = endpoint.getRouteChannel();
      assert(channel && "assignChannels gives every routed endpoint one");
      for (DMAStartOp start : starts)
        if (failed(resolveStart(start, endpoint, flows, ids, *channel)))
          return failure();
      if (endpoint.getTileLike().isShimTile() && endpoint.getFifoName())
        continue;
      for (auto task : tasks)
        resolveTask(task, endpoint, *channel);
    }
    return success();
  }

  /// A shim endpoint has no memory of its own, so the runtime needs its channel
  /// spelled out under the name the sequence refers to.
  void emitShimAllocations() {
    builder.setInsertionPoint(device.getBody()->getTerminator());
    // A fifo's first shim end takes the name the runtime sequence was pointed
    // at. Any further shim end of the same fifo, as when a core broadcasts to
    // two shim tiles, is a different tile and channel and needs a record of
    // its own; sharing the first would send the runtime to the wrong tile.
    llvm::StringMap<unsigned> shimEndsSeen;
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      std::optional<StringRef> fifoName = endpoint.getFifoName();
      std::optional<int> channel = endpoint.getRouteChannel();
      if (!tileOf(endpoint).isShimTile() || !fifoName || !channel) {
        continue;
      }
      std::string name = (*fifoName + "__shim_alloc").str();
      unsigned ordinal = shimEndsSeen[*fifoName]++;
      bool create = true;
      if (ordinal == 0) {
        // A record under this name may already be there, declared by hand.
        create = !SymbolTable::lookupNearestSymbolFrom<ShimDMAAllocationOp>(
            device, builder.getStringAttr(name));
      } else {
        for (unsigned suffix = ordinal - 1; device.lookupSymbol(name);
             suffix++) {
          name =
              (*fifoName + "__shim_alloc" + std::to_string(suffix + 1)).str();
        }
      }
      if (create) {
        // The proof names other shim endpoints, which replaceAllSymbolUses
        // renames to their allocations as each is reached.
        auto routeEndpoint = dyn_cast<RouteEndpointOp>(endpoint.getOperation());
        ShimDMAAllocationOp::create(
            builder, endpoint.getLoc(), builder.getStringAttr(name),
            endpoint.getTile(),
            DMAChannelDirAttr::get(builder.getContext(),
                                   endpoint.getRouteDirection()),
            builder.getI64IntegerAttr(*channel),
            builder.getBoolAttr(endpoint.getRouteBundle() == WireBundle::PLIO),
            endpoint->getAttrOfType<PacketInfoAttr>("packet"),
            routeEndpoint ? routeEndpoint.getConsumedByAttr() : ArrayAttr());
      }
      // The runtime sequence reaches the fifo through this record.
      (void)SymbolTable::replaceAllSymbolUses(
          endpoint, builder.getStringAttr(name), device);
    }
  }

  /// Size-first packing order: mem tile pools largest buffer first, equal
  /// ones by home coordinates, unplaced last, as the SA placer orders them, in
  /// the slots mem tile pools hold, the others left in place. A pool walked
  /// whole becomes one large buffer, so this is chosen again after every walk.
  void sizeOrder(MutableArrayRef<ObjectFifoPoolOp> pools) {
    SmallVector<size_t> memTileSlots;
    SmallVector<ObjectFifoPoolOp> memTilePools;
    for (auto [index, pool] : llvm::enumerate(pools)) {
      if (pool.getTileLike().isMemTile()) {
        memTileSlots.push_back(index);
        memTilePools.push_back(pool);
      }
    }
    llvm::stable_sort(memTilePools, [](ObjectFifoPoolOp a, ObjectFifoPoolOp b) {
      auto home = [](ObjectFifoPoolOp pool) {
        TileLike tile = pool.getTileLike();
        auto col = tile.tryGetCol(), row = tile.tryGetRow();
        return std::make_tuple(!col || !row, col.value_or(0), row.value_or(0));
      };
      return home(a) < home(b);
    });
    sortLargestBufferFirst(memTilePools, bufferBytes);
    for (auto [slot, pool] : llvm::zip(memTileSlots, memTilePools))
      pools[slot] = pool;
  }

  /// Alternative packing order (demandFirstOrder). The same buffer, lock and
  /// channel planner validates both orders.
  SmallVector<ObjectFifoPoolOp>
  demandOrderedPools(ArrayRef<ObjectFifoPoolOp> pools) {
    DenseMap<Operation *, std::pair<int, int>> poolChannelDemand;
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      ObjectFifoPoolOp pool = endpoint.getPoolOp();
      if (!sameTile(endpoint.getTile(), pool.getTile()))
        continue;
      auto &demand = poolChannelDemand[pool];
      if (endpoint.getRouteDirection() == DMAChannelDir::S2MM)
        ++demand.first;
      else
        ++demand.second;
    }
    SmallVector<size_t> memTileSlots;
    SmallVector<Value> homes;
    SmallVector<SpillPool> records;
    for (size_t index = 0, e = pools.size(); index < e; ++index) {
      ObjectFifoPoolOp pool = pools[index];
      if (!pool.getTileLike().isMemTile())
        continue;
      memTileSlots.push_back(index);
      auto *home = llvm::find_if(
          homes, [&](Value known) { return sameTile(pool.getTile(), known); });
      if (home == homes.end()) {
        homes.push_back(pool.getTile());
        home = std::prev(homes.end());
      }
      auto [input, output] = poolChannelDemand.lookup(pool.getOperation());
      records.push_back({bufferBytes(pool), bufferCount(pool),
                         static_cast<unsigned>(home - homes.begin()),
                         std::max(input, output)});
    }
    SmallVector<ObjectFifoPoolOp> ordered(pools);
    for (auto [slot, index] :
         llvm::zip(memTileSlots, demandFirstOrder(records)))
      ordered[slot] = pools[memTileSlots[index]];
    return ordered;
  }

  void runOnOperation() override {
    device = getOperation();
    builder = OpBuilder(device.getContext());
    // One pass instance serves every device in the module, and none of this
    // state means anything outside the device it was gathered from.
    lastPlaced.clear();
    loweredFlows.clear();
    sourcedFlows.clear();
    localPools.clear();
    poolUsers.clear();
    lockUsers.clear();
    lockPlacements.clear();
    spillTiles.clear();
    dispatchHeaders.clear();
    channelSharers.clear();
    bdPlanned.clear();
    bufferPlacements.clear();
    channelAssignments.clear();
    endpointStarts.clear();
    endpointTasks.clear();
    fixedBDs.clear();
    taskBDs.clear();
    channelDemand.clear();
    structuralRank.clear();
    channelFailure = nullptr;
    bufferFailure = nullptr;
    lockFailure = nullptr;
    lockAccessFailure = nullptr;
    walkSearchStopped = false;

    if (failed(collectFixedMemory()) || failed(checkStreamPorts()))
      return signalPassFailure();
    collectChannelPrograms();
    rankStructurally();

    // Preserve successful largest-first allocations, including their locality
    // repairs. Only try demand ordering when that search fails.
    SmallVector<ObjectFifoPoolOp> pools(device.getOps<ObjectFifoPoolOp>());
    llvm::stable_sort(pools, [&](ObjectFifoPoolOp a, ObjectFifoPoolOp b) {
      return structuralRank.lookup(a) < structuralRank.lookup(b);
    });
    sizeOrder(pools);

    poolLocks = std::make_shared<PoolLocks>(device);
    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      poolUsers[endpoint.getPoolOp()].push_back(endpoint.getTile());
      collectLockUsers(endpoint);
    }
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      poolUsers[endpoint.getPoolOp()].push_back(endpoint.getTile());
      collectLockUsers(endpoint);
    }
    for (auto &[group, users] : lockUsers)
      llvm::stable_sort(users, [&](Operation *a, Operation *b) {
        return structuralRank.lookup(a) < structuralRank.lookup(b);
      });

    std::set<std::vector<unsigned>> tried;
    LogicalResult allocated = planAllocation(pools, tried);
    if (failed(allocated)) {
      auto demandOrdered = demandOrderedPools(pools);
      if (!llvm::equal(pools, demandOrdered)) {
        // Memoized locality sets are specific to a packing order.
        localPools.clear();
        tried.clear();
        allocated = planAllocation(demandOrdered, tried);
      }
    }
    // Out of channels, ends that never need theirs at once take turns on one,
    // a tile and direction at a time: on a shim tile, ends never in flight
    // together; elsewhere, ends whose objects one core needs in a fixed order.
    // Short of BD ids on a mem tile channel, its deepest pool walked whole
    // takes one a side. Buffers that fit nowhere as split are placed again
    // with the pools walked whole that the tiles' BD ids need anyway.
    auto repair = [&] {
      if (channelFailure) {
        TileLike tile = tileOf(channelFailure);
        DMAChannelDir dir = channelFailure.getRouteDirection();
        if (bdShortfall) {
          if (bdPlanned.insert(tile.getOperation()) ||
              walkForBDShortfall(tile, pools))
            return true;
        } else if (tile.isShimTile() ? planShimSharing(tile, dir)
                                     : planTurnSharing(tile, dir)) {
          return true;
        }
      }
      localPools.clear();
      return failed(planBuffers(pools)) && walkHomeBDPools(pools);
    };
    while (failed(allocated) && repair()) {
      localPools.clear();
      tried.clear();
      sizeOrder(pools);
      allocated = planAllocation(pools, tried);
    }
    if (succeeded(allocated) && !channelSharers.empty() &&
        failed(headShimSharingRoutes()))
      return signalPassFailure();
    if (failed(allocated)) {
      localPools.clear();
      if (failed(planBuffers(pools))) {
        if (walkSearchStopped)
          bufferFailure.emitOpError("could not place buffers: the search for "
                                    "which pools to walk whole stopped after ")
              << DMAChannelAnalysis::kMaxWalkPlacements
              << " placements tried in one spill component";
        else
          bufferFailure.emitOpError(
              "could not place buffers in accessible memory with available "
              "capacity");
      } else if (failed(planLocks(pools))) {
        if (lockAccessFailure)
          lockAccessFailure->emitOpError("cannot access pool locks");
        else
          lockFailure.getDefiningOp()->emitOpError(
              "could not place locks within tile lock capacity");
      } else {
        DMAChannelAnalysis channels(device);
        (void)assignChannels(channels);
        device.emitRemark(
            "could not find a spill-aware allocation with size-first and "
            "demand-first packing and bounded local-pool retries; tile "
            "placement and buffer packing remain greedy");
      }
      return signalPassFailure();
    }
    for (ObjectFifoPoolOp pool : pools) {
      allocateBuffers(pool);
      allocateLocks(pool);
    }
    for (TileOp tile : spillTiles)
      if (tile->use_empty())
        tile->erase();
    for (auto endpoint : device.getOps<RouteEndpoint>())
      if (auto it = channelAssignments.find(endpoint.getOperation());
          it != channelAssignments.end())
        endpoint.setRouteChannel(it->second);
    warnOvercommittedBDs();

    if (failed(bindRearmTargets())) {
      return signalPassFailure();
    }
    if (failed(lowerFlows()) || failed(resolveEndpointUses())) {
      return signalPassFailure();
    }
    emitShimAllocations();
    for (Operation *flow : loweredFlows) {
      flow->erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoAllocatePass() {
  return std::make_unique<AIEObjectFifoAllocatePass>();
}
