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
#include "aie/Dialect/AIE/Transforms/AIELockPlan.h"
#include "aie/Dialect/AIE/Transforms/AIEPacketIdSpace.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPoolLocks.h"
#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "aie/Dialect/AIE/Transforms/AIEStructuralRank.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

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
  llvm::MapVector<Operation *, ShimDMAAllocationOp> allocationSharers;
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
  /// BD ids DMA programs and pinned runtime ids hold before objectFIFOs take
  /// theirs. Channels are chosen from these and the endpoints' own chains
  /// alone, so an image built for one runtime sequence serves another's
  /// instruction stream.
  DMAChannelAnalysis::TakenBDs takenBDs;
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
  /// Whether channelFailure's channel has too few BD ids left, rather than no
  /// channel being free.
  bool bdShortfall = false;
  /// Mem tiles once short of BD ids, whose ends since take their channels
  /// from planBDChannels.
  llvm::SetVector<Operation *> bdPlanned;
  /// Pools planTurnSharing walked whole (some walks are undone later).
  llvm::SetVector<Operation *> turnWalked;
  /// The pool the buffer search could not place with the fewest left
  /// unplaced, i.e. where it got furthest.
  ObjectFifoPoolOp bufferFailure;
  /// Neighbours made for a spill planBuffers weighed. The ones no buffer ends
  /// up on are erased once allocation is done.
  SmallVector<TileOp> spillTiles;
  Value lockFailure;
  Operation *lockAccessFailure = nullptr;
  /// The pool whose lock group the packing search was stopped on.
  ObjectFifoPoolOp lockSearchStoppedOn;
  /// See LockPlan::groupCounts for what this changes.
  bool semaphoreLocks = false;
  bool walkSearchStopped = false;
  bool lockSearchStopped = false;

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
    return userTile && LockPlan::reaches(device.getTargetModel(),
                                         compatiblePositions(userTile),
                                         memoryPositions, localOnly);
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
    // ones no buffer ends up on are erased once allocation is done.
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
    lockSearchStopped = false;
    lockPlacements.clear();
    LockPlan plan(device.getTargetModel());
    SmallVector<Value> tileValues;
    DenseMap<Value, unsigned> tileIndex;
    auto indexOf = [&](Value tile) {
      auto [at, fresh] = tileIndex.try_emplace(tile, tileValues.size());
      if (fresh) {
        auto like = cast<TileLike>(tile.getDefiningOp());
        auto col = like.tryGetCol(), row = like.tryGetRow();
        plan.addTile(
            {compatiblePositions(like),
             col && row ? std::optional<TileID>({*col, *row}) : std::nullopt});
        tileValues.push_back(tile);
      }
      return at->second;
    };
    for (auto lock : device.getOps<LockOp>())
      if (!plan.reserve(indexOf(lock.getTile()), 1)) {
        lockFailure = lock.getTile();
        return failure();
      }
    // Neighbours created to weigh a spill take no locks: the placers count
    // locks only on the tiles that exist before allocation.
    SmallVector<TileLike> tiles(device.getOps<TileLike>());
    llvm::stable_sort(tiles, [&](TileLike a, TileLike b) {
      auto aRank = structuralRank.find(a), bRank = structuralRank.find(b);
      if ((aRank == structuralRank.end()) != (bRank == structuralRank.end()))
        return bRank == structuralRank.end();
      return aRank != structuralRank.end() && aRank->second < bRank->second;
    });
    SmallVector<unsigned> ranked;
    for (auto tile : tiles)
      if (!llvm::is_contained(spillTiles, tile.getOperation()))
        ranked.push_back(indexOf(tile->getResult(0)));
    SmallVector<LockPlan::Group> groups;
    SmallVector<Operation *> groupOps;
    SmallVector<ObjectFifoPoolOp> groupPools;
    auto add = [&](ObjectFifoPoolOp pool, Operation *group, int count) {
      SmallVector<unsigned> buffers;
      for (Value tile : bufferPlacements[pool])
        if (tile && !llvm::is_contained(spillTiles, tile.getDefiningOp()))
          buffers.push_back(indexOf(tile));
      SmallVector<LockPlan::User> users;
      for (Operation *user : lockUsers[group]) {
        Value tile = user->getOperand(0);
        users.push_back(
            {indexOf(tile),
             isa<ObjectFifoDmaEndpointOp>(user) &&
                 LockPlan::dmaLocalOnly(
                     cast<TileLike>(tile.getDefiningOp()).getTileType())});
      }
      groups.push_back(
          LockPlan::group(count, indexOf(lockPlacement(pool, group)),
                          localPools.contains(pool), buffers, users, ranked));
      groupOps.push_back(group);
      groupPools.push_back(pool);
    };
    for (auto pool : pools) {
      if (!poolLocks->needsLocks(pool))
        continue;
      SmallVector<Operation *> owners = lockOwners(pool);
      for (auto [owner, count] : llvm::zip_equal(
               owners, LockPlan::groupCounts(device.getTargetModel(),
                                             pool.getDepth(), owners.size())))
        add(pool, owner, count);
    }
    LockPlan::Assignment assignment = plan.assign(groups);
    if (assignment.failed) {
      Operation *group = groupOps[*assignment.failed];
      const LockPlan::Placed &missed = assignment.failure;
      lockSearchStopped = assignment.stopped;
      lockSearchStoppedOn = groupPools[*assignment.failed];
      lockFailure = missed.full ? tileValues[*missed.full] : Value();
      if (!missed.reachable && !lockUsers[group].empty())
        lockAccessFailure = lockUsers[group].front();
      return failure();
    }
    for (auto [group, tile] : llvm::zip(groupOps, assignment.tiles))
      if (tile)
        lockPlacements[group] = tileValues[*tile];
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

  /// The ops whose locks allocation makes for `pool`: with semaphore locks,
  /// each segment without its pair; otherwise the pool, when it has none.
  SmallVector<Operation *> lockOwners(ObjectFifoPoolOp pool) {
    SmallVector<Operation *> owners;
    if (!semaphoreLocks) {
      if (!pool.getLocks())
        owners.push_back(pool);
      return owners;
    }
    for (auto segment : pool.getSegmentOps())
      if (!(segment.getProduceLock() && segment.getConsumeLock()))
        owners.push_back(segment);
    return owners;
  }

  void allocateLocks(ObjectFifoPoolOp pool) {
    if (!poolLocks->needsLocks(pool))
      return;
    StringRef base = pool.getBaseName();
    int depth = pool.getDepth();
    auto initValues = pool.getInitValues();
    int filled = initValues ? initValues->size() : 0;
    int repeat = pool.getRepeatCount().value_or(1);

    if (!semaphoreLocks) {
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
    if (isa<ObjectFifoDmaEndpointOp>(endpoint) &&
        LockPlan::dmaLocalOnly(
            cast<TileLike>(user.getDefiningOp()).getTileType()))
      return canAccess(user, tile, /*localOnly=*/true);
    return canAccess(user, tile);
  }

  template <typename EndpointOp>
  void collectLockUsers(EndpointOp endpoint) {
    auto pool = endpoint.getPoolOp();
    if (!semaphoreLocks) {
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
    if (!semaphoreLocks) {
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
  /// An end whose channel others take turns on reaches what they reach too.
  bool reachesAdjacentTile(RouteEndpoint endpoint) {
    if (ownReachesAdjacentTile(endpoint))
      return true;
    for (auto [sharer, owner] : channelSharers)
      if (owner == endpoint.getOperation() &&
          ownReachesAdjacentTile(cast<RouteEndpoint>(sharer)))
        return true;
    return false;
  }

  /// Whether `endpoint`'s own pool, locks or DMA program reach another tile,
  /// whatever ends take turns on its channel.
  bool ownReachesAdjacentTile(RouteEndpoint endpoint) {
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

  static int64_t countBDs(Region &taskBody) {
    int64_t bds = 0;
    taskBody.walk([&](DMABDOp) { bds++; });
    return bds;
  }

  /// BDs a program-owned endpoint's static chains hold; they stay configured.
  int64_t endpointBDs(Operation *endpoint) {
    int64_t bds = 0;
    for (DMAStartOp start : endpointStarts.lookup(endpoint))
      bds += DMAChannelAnalysis::countBDs(start);
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
      for (Block *block : DMAChannelAnalysis::chainOf(start))
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
    taskBDs.clear();
    takenBDs = DMAChannelAnalysis::TakenBDs(device);
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
        for (auto start : block.getOps<DMAStartOp>())
          if (auto endpoint = start.getEndpointOp())
            endpointStarts[endpoint].push_back(start);
    device.walk([&](Operation *op) {
      if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
        if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(
                SymbolTable::lookupNearestSymbolFrom(
                    device, task.getAlloc().getRootReference())))
          endpointTasks[endpoint].push_back(task);
      } else if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op)) {
        if (auto at = key(task.getTile(), task.getChannel())) {
          int64_t unpinned = 0;
          task.getBody().walk([&](DMABDOp bd) { unpinned += !bd.getBdId(); });
          int64_t &largest = taskBDs[*at];
          largest = std::max(largest, unpinned);
        }
      }
    });
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
    DenseMap<StringAttr, Operation *> coreEndpoints;
    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      coreEndpoints[endpoint.getSymNameAttr()] = endpoint;
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
    // Ops still tied fall to the order each core first acquires its ends in,
    // the core's program order: ranking again from the ranks found, with the
    // cores linked in, splits only ties.
    for (Operation *node : graph.nodes)
      graph.features[node] = llvm::formatv("{0,10}", rank[node]).str();
    for (auto core : device.getOps<CoreOp>()) {
      graph.add(core);
      llvm::SmallPtrSet<Operation *, 8> acquired;
      core.walk([&](ObjectFifoAcquireOp acquire) {
        Operation *endpoint =
            coreEndpoints.lookup(acquire.getObjFifoNameAttr().getAttr());
        if (endpoint && acquired.insert(endpoint).second)
          graph.link(core, endpoint,
                     llvm::formatv("acquire {0,10}", acquired.size()).str());
      });
    }
    rank = graph.rank();
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

  // The endpoints of the routes that switch packets.
  DenseSet<Operation *> packetEndpoints() {
    DenseSet<Operation *> endpoints;
    for (auto flow : device.getOps<RouteOp>()) {
      if (!flow.getPacket())
        continue;
      for (StringRef source : flow.getSourceNames())
        endpoints.insert(lookupEndpoint(source).getOperation());
      for (StringRef dest : flow.getDestinationNames())
        endpoints.insert(lookupEndpoint(dest).getOperation());
    }
    return endpoints;
  }

  // No allocation can move a stream port the design names, so a clash fails
  // the pass before the search rather than surfacing inside it.
  LogicalResult checkStreamPorts() {
    DMAChannelAnalysis channels(device);
    DenseSet<Operation *> packetEndpoints = this->packetEndpoints();
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

  /// Whether `endpoint` draws a DMA channel of its own, rather than being a
  /// core's stream port or sharing another end's channel.
  bool drawsOwnChannel(RouteEndpoint endpoint) {
    return endpoint.getRouteBundle() != WireBundle::Core &&
           !channelSharers.contains(endpoint.getOperation()) &&
           !allocationSharers.contains(endpoint.getOperation());
  }

  /// Whether this endpoint can share its pinned channel without a cyclic
  /// DMA program monopolizing it.
  bool sharesPinnedChannel(RouteEndpoint endpoint, DMAChannelDir dir) {
    // An out-of-order receiver takes each object into the buffer its header
    // names, so its channel keeps no order its fifos would have to agree on.
    auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation());
    if (dir == DMAChannelDir::S2MM)
      return !ObjectFifoCreateOp::sharedReceiveRefusal(
          tileOf(endpoint).isShimTile(), dma && dma.getOutOfOrder());
    if (!tileOf(endpoint).isShimTile() &&
        !(dma && dma.getReleaseStrategy() == ObjectFifoRelease::PushQueue))
      return false;
    bool hasPacketRoute = false;
    for (auto flow : device.getOps<RouteOp>()) {
      if (llvm::none_of(flow.getSourceNames(), [&](StringRef name) {
            return lookupEndpoint(name).getOperation() ==
                   endpoint.getOperation();
          }))
        continue;
      if (!flow.getPacket())
        return false;
      hasPacketRoute = true;
    }
    return hasPacketRoute;
  }

  LogicalResult assignChannels(DMAChannelAnalysis &channels,
                               bool diagnose = true) {
    channelAssignments.clear();
    channelFailure = nullptr;
    bdShortfall = false;
    SmallVector<RouteEndpoint> pending;
    // BDs per placed (col, row, channel) as channels are handed out, for
    // choosing among a tile's channels by the BD ids they can use.
    DenseMap<std::tuple<int, int, int>, int64_t> demand;
    for (auto [col, row] : takenBDs.positions())
      for (auto [channel, bds] : takenBDs.at(device.getTargetModel(), col, row))
        demand[{col, row, channel}] += bds;
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
        if (channels.reservePinnedChannel(
                tileOf(endpoint), dir, *channel, endpoint.getOperation(),
                sharesPinnedChannel(endpoint, dir)) < 0) {
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

    // A mem tile's pinned ends must fit the BD ids of the channels they pin,
    // as a plan of the tile's BD ids weighs every end.
    std::map<std::pair<int, int>, SmallVector<RouteEndpoint>> pinnedEnds;
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      TileLike tile = tileOf(endpoint);
      std::optional<int> col = tile.tryGetCol(), row = tile.tryGetRow();
      if (tile.isMemTile() && col && row && endpoint.getRouteChannel() &&
          drawsOwnChannel(endpoint))
        pinnedEnds[{*col, *row}].push_back(endpoint);
    }
    for (auto &[at, onTile] : pinnedEnds) {
      auto [col, row] = at;
      SmallVector<DMAChannelAnalysis::BDPlanEnd> ends;
      for (RouteEndpoint endpoint : onTile) {
        DMAChannelDir dir = endpoint.getRouteDirection();
        ends.push_back({dir, bdsOf(endpoint),
                        DMAChannelAnalysis::getDMAChannelLimit(tileOf(endpoint),
                                                               dir, false),
                        endpoint.getRouteChannel()});
      }
      DenseMap<int, int64_t> taken =
          takenBDs.at(device.getTargetModel(), col, row);
      if (DMAChannelAnalysis::planBDChannels(
              device.getTargetModel(), col, row, ends,
              [](DMAChannelDir, int) { return true; },
              [&](int channel) { return taken.lookup(channel); }))
        continue;
      int pinnedCol = col, pinnedRow = row;
      auto roomOf = [&](RouteEndpoint endpoint) {
        return bdRoom(tileOf(endpoint), pinnedCol, pinnedRow,
                      *endpoint.getRouteChannel(), demand);
      };
      // The ends short of BD ids, the most demanding first.
      SmallVector<RouteEndpoint> starved;
      for (RouteEndpoint endpoint : onTile)
        if (roomOf(endpoint) < 0)
          starved.push_back(endpoint);
      assert(!starved.empty() &&
             "bdRoom and planBDChannels weigh the same BD ids");
      llvm::stable_sort(starved, [&](RouteEndpoint a, RouteEndpoint b) {
        return std::make_tuple(bdsOf(b), structuralRank.lookup(a)) <
               std::make_tuple(bdsOf(a), structuralRank.lookup(b));
      });
      RouteEndpoint deepest = starved.front();
      channelFailure = deepest;
      bdShortfall = true;
      if (!diagnose)
        return failure();
      InFlightDiagnostic diag =
          deepest->emitOpError("needs ")
          << bdsOf(deepest) << " buffer descriptors, but pinned "
          << stringifyDMAChannelDir(deepest.getRouteDirection())
          << " DMA channel " << *deepest.getRouteChannel()
          << " of its tile has " << roomOf(deepest) + bdsOf(deepest)
          << " left beside the ends on the channels sharing its BD ids";
      for (RouteEndpoint other : llvm::drop_begin(starved))
        diag.attachNote(other->getLoc())
            << "this end, on pinned "
            << stringifyDMAChannelDir(other.getRouteDirection())
            << " DMA channel " << *other.getRouteChannel() << ", needs "
            << bdsOf(other) << " of them too";
      return diag;
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
        ends.push_back({dir, bdsOf(endpoint), limit, std::nullopt});
        unweighted.push_back({dir, 0, limit, std::nullopt});
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
      // Fails even with zero BDs: a channel shortage, handled below.
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
             << bdsOf(deepest) << " buffer descriptors, but the tile's "
             << ends.size() << " unpinned DMA ends need " << total
             << " in all, more than their channels' BD ids hold";
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
    DenseSet<Operation *> claimedAllocations;
    for (auto [endpoint, alloc] : allocationSharers) {
      auto end = cast<RouteEndpoint>(endpoint);
      int channel = alloc.getChannelIndex();
      if (claimedAllocations.insert(alloc).second &&
          channels.reservePinnedChannel(tileOf(end), alloc.getChannelDir(),
                                        channel, endpoint, true) < 0) {
        channelFailure = end;
        if (!diagnose)
          return failure();
        auto diag = endpoint->emitOpError("cannot share ");
        diag << stringifyDMAChannelDir(alloc.getChannelDir()) << " DMA channel "
             << channel << ": already in use or unavailable";
        if (Operation *owner = channels.getDMAChannelOwner(
                tileOf(end), alloc.getChannelDir(), channel))
          noteChannelOwner(diag, owner, channel);
        return failure();
      }
      channelAssignments[endpoint] = channel;
      noteDemand(end, channel);
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
    // Receiving ends share by the #aie.consumed_by proofs
    // aie-objectfifo-verify leaves; without them, by explicit permission only.
    ShimTransferSpans spans(
        device,
        [&](StringAttr symbol) {
          return ShimTransferSpans::endOf(device, symbol);
        },
        ShimTransferSpans::provenGates(device, ShimTransferSpans::keyOf));
    SmallVector<Operation *> candidates;
    SmallVector<StringAttr> fifos, names;
    for (auto endpoint : device.getOps<RouteEndpointOp>()) {
      if (endpoint.getBundle() != WireBundle::DMA ||
          endpoint.getChannelIndex() ||
          endpoint.getTile() != exhausted->getResult(0) ||
          endpoint.getRouteDirection() != dir)
        continue;
      candidates.push_back(endpoint);
      fifos.push_back(ShimTransferSpans::keyOf(endpoint));
      names.push_back(endpoint.getSymNameAttr());
    }
    SmallVector<ShimDMAAllocationOp> allocations;
    SmallVector<ShimTransferSpans::SharedChannel> shared;
    for (auto alloc : device.getOps<ShimDMAAllocationOp>()) {
      auto allowed = AIEDialect::ShareableWithAttrHelper(alloc->getContext())
                         .getAttr(alloc);
      if (!allowed || alloc.getTile() != exhausted->getResult(0) ||
          alloc.getChannelDir() != dir)
        continue;
      SmallVector<StringAttr> listed;
      for (Attribute attr : allowed)
        if (auto symbol = dyn_cast<FlatSymbolRefAttr>(attr))
          listed.push_back(symbol.getAttr());
      bool pinned =
          llvm::any_of(device.getOps<RouteEndpoint>(), [&](RouteEndpoint end) {
            return end.getRouteBundle() == WireBundle::DMA &&
                   sameTile(end.getTile(), alloc.getTile()) &&
                   end.getRouteDirection() == dir &&
                   end.getRouteChannel() == alloc.getChannelIndex();
          });
      allocations.push_back(alloc);
      shared.emplace_back(std::move(listed), pinned);
    }
    ShimTransferSpans::Plan plan = spans.plan(fifos, names, shared);
    for (unsigned channel : llvm::seq<unsigned>(0, shared.size()))
      for (auto [candidate, taken] : llvm::zip(candidates, plan.sharedChannel))
        if (taken == channel &&
            allocationSharers.try_emplace(candidate, allocations[channel])
                .second)
          return true;
    for (auto &members : plan.groups)
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
      plan.ends.push_back(
          {dir, dma.getNumBDs(),
           DMAChannelAnalysis::getDMAChannelLimit(tileOf(endpoint), dir, false),
           endpoint.getRouteChannel()});
      plan.poolOf.push_back(llvm::find(plan.pools, dma.getPoolOp()) -
                            plan.pools.begin());
    }
    return plan;
  }

  /// Walks whichever pools need it: first walkHomeBDPools' plan for `tile`'s
  /// BD ids at home; once that needs none, the one pool
  /// DMAChannelAnalysis::nextPoolToWalk picks for what only the tile's real
  /// channels are short of. False once none is left to walk.
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
    struct Tile {
      TileID at;
      HomeBDPlan home;
      DMAChannelAnalysis::BDWalkSets found;
      SmallVector<ObjectFifoDmaEndpointOp> dmas;
      unsigned applied = 0;
    };
    SmallVector<Tile, 0> tiles;
    // The tile's own programs and runtime tasks keep their channels and BD
    // ids, as the SA placer's plan of the tile does.
    DMAChannelAnalysis channels(device);
    for (auto &[at, dmas] : ends) {
      HomeBDPlan home = homeBDPlan(dmas);
      TileLike tile = tileOf(cast<RouteEndpoint>(dmas.front().getOperation()));
      DenseMap<int, int64_t> taken =
          takenBDs.at(device.getTargetModel(), at.first, at.second);
      DMAChannelAnalysis::BDWalkSets found =
          DMAChannelAnalysis::walkSetsThatFit(
              device.getTargetModel(), at.first, at.second, home.ends,
              home.poolOf, home.walkable,
              [&](DMAChannelDir dir, int channel) {
                return channels.isChannelFree(tile, dir, channel);
              },
              [&](int channel) { return taken.lookup(channel); });
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

  /// Whether, in every runtime sequence, the host issues each transfer
  /// through `fifo` before any wait: a turn chain waiting on that fifo's
  /// objects then waits on nothing the host does after the array has moved.
  bool hostMovesFirst(StringAttr fifo) {
    SymbolTableCollection symbols;
    for (auto seq : device.getOps<RuntimeSequenceOp>()) {
      auto names = [&](Operation *op) {
        return fifo && AIEX::namesFifo(op, fifo.getValue(), symbols);
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

  /// The other end of the one route naming `endpoint`, when that route is a
  /// circuit between two objectFIFO DMA endpoints.
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
    auto endsOf = [&](Operation *pool) -> ArrayRef<ObjectFifoDmaEndpointOp> {
      auto it = poolEnds.find(pool);
      return it == poolEnds.end() ? ArrayRef<ObjectFifoDmaEndpointOp>()
                                  : it->second;
    };
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
          (dma.getTurns() ? turnEndpoints(dma).front() : dma).canTakeTurns())
        if (ObjectFifoDmaEndpointOp peer = cachedCircuitPeer(dma);
            peer && peer.canTakeTurns())
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
                       << dma.canTakeTurns() << ", peer takes turns "
                       << (cachedCircuitPeer(dma) &&
                           cachedCircuitPeer(dma).canTakeTurns())
                       << "\n";
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
        // cannot find refuses the turns. Peers that split one pool between
        // them, draining it or filling it, each carry a part of what the core
        // at the pool's other end moves in one piece, so they count as one peer
        // of that core's when every segment on their side takes these turns.
        SmallVector<ObjectFifoCoreEndpointOp> peerCores;
        SmallVector<int64_t> peerDepths;
        SmallVector<unsigned> fed(peers.size(), ~0u);
        DenseMap<Operation *, unsigned> fedByPool;
        bool lost = false, hostWaits = false;
        auto wholePool = [&](ObjectFifoDmaEndpointOp peer) {
          bool drains = peer.drains();
          return peer.getSegments() &&
                 llvm::all_of(endsOf(peer.getPoolOp()), [&](auto end) {
                   return end.drains() != drains ||
                          llvm::is_contained(peers, end);
                 });
        };
        for (auto [i, peer] : llvm::enumerate(peers)) {
          if (wholePool(peer)) {
            auto segment = fedByPool.find(peer.getPoolOp());
            if (segment != fedByPool.end()) {
              fed[i] = segment->second;
              continue;
            }
          }
          int64_t depth = peer.getPoolOp().getDepth();
          ObjectFifoCoreEndpointOp core = cachedCoreEnd(peer);
          SmallPtrSet<Operation *, 8> seen;
          for (ObjectFifoDmaEndpointOp at = peer; !core;) {
            SmallVector<ObjectFifoDmaEndpointOp> across;
            for (ObjectFifoDmaEndpointOp end : endsOf(at.getPoolOp()))
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
          if (wholePool(peer))
            fedByPool[peer.getPoolOp()] = fed[i];
          peerCores.push_back(core);
          peerDepths.push_back(depth);
        }
        if (reason.empty() && lost)
          reason = "a peer's feeding core is not known, or feeds two peers";
        if (reason.empty() && hostWaits)
          reason = "the host waits before it moves a peer fed by the shim";
        // A turn on a segment past every object its pool's segments have
        // reached asks the core for the next one; the objects the segments
        // hold open at once must fit the pool, and every segment takes one
        // part of each object.
        SmallVector<unsigned> peerOrder;
        DenseMap<unsigned, int64_t> asked;
        SmallVector<int64_t> taken(peers.size(), 0);
        for (unsigned turn : turns.turns()) {
          if (fed[turn] == ~0u)
            continue;
          int64_t part = ++taken[turn];
          if (part > asked[fed[turn]]) {
            asked[fed[turn]] = part;
            peerOrder.push_back(fed[turn]);
          }
          ObjectFifoPoolOp pool = peers[turn].getPoolOp();
          int64_t fewest = part;
          for (auto [j, other] : llvm::enumerate(peers))
            if (other.getPoolOp() == pool)
              fewest = std::min(fewest, taken[j]);
          if (reason.empty() && asked[fed[turn]] - fewest > pool.getDepth())
            reason = "a pool's segments hold more objects open than it has";
        }
        for (auto [i, peer] : llvm::enumerate(peers))
          for (auto [j, other] : llvm::enumerate(peers))
            if (reason.empty() && other.getPoolOp() == peer.getPoolOp() &&
                taken[i] != taken[j])
              reason = "a pool's segments take different counts of turns";
        if (reason.empty() && !peerCores.empty()) {
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
            if (linedUp) {
              walked.push_back(pool);
              turnWalked.insert(pool);
            } else
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
    for (auto [endpoint, alloc] : allocationSharers)
      channelOf[endpoint] = alloc.getOperation();
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
      endpoint->emitWarning("")
          << stringifyDMAChannelDir(endpoint.getRouteDirection())
          << " DMA channel " << channel << " can use " << numBDs
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
          !ownReachesAdjacentTile(endpoint) ||
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

    // An out-of-order receiver reads the buffer it should fill out of the
    // header, so the header has to reach its DMA.
    bool keepHeader = flow.getKeepPktHeader();
    for (StringRef destName : flow.getDestinationNames())
      if (auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(
              lookupEndpoint(destName).getOperation()))
        keepHeader |= dma.getOutOfOrder();

    builder.setInsertionPoint(flow);
    auto packetFlow = PacketFlowOp::create(
        builder, flow.getLoc(),
        builder.getIntegerAttr(builder.getI8Type(), packetID),
        keepHeader ? builder.getBoolAttr(true) : nullptr, nullptr);
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
      // An out-of-order receiver matches on the header rather than walking a
      // chain, so its descriptors carry the id too.
      if (auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(dest.getOperation()))
        if (dma.getOutOfOrder())
          dest.setRoutePacket(header);
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
    // Independent core-pushed senders need different routing IDs even when
    // their descriptors are submitted at different times.
    DenseMap<std::tuple<Value, int, int>, Operation *> pushedPacketSources;
    for (RouteOp flow : device.getOps<RouteOp>()) {
      auto packet = flow.getPacket();
      if (!packet || !packet->isAssigned())
        continue;
      for (StringRef name : flow.getSourceNames()) {
        auto source = dyn_cast<ObjectFifoDmaEndpointOp>(
            lookupEndpoint(name).getOperation());
        if (!source ||
            source.getReleaseStrategy() != ObjectFifoRelease::PushQueue)
          continue;
        auto [it, inserted] = pushedPacketSources.try_emplace(
            std::make_tuple(source.getTile(), channelOf(source),
                            packet->assignedId()),
            source.getOperation());
        if (!inserted && it->second != source.getOperation()) {
          auto diag = source.emitOpError(
              "core-pushed senders sharing an MM2S channel require distinct "
              "packet IDs");
          diag.attachNote(it->second->getLoc()) << "other sender is here";
          return failure();
        }
      }
    }
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
        if (auto dma =
                dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation());
            dma && dma.getReleaseStrategy() == ObjectFifoRelease::PushQueue) {
          for (AIEX::DmaChannelResetForOp user : users)
            user.emitOpError(
                "does not support core-pushed DMA channels; reset requires "
                "coordination with the issuing core and all shared senders");
          return failure();
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
      for (Block *block : DMAChannelAnalysis::chainOf(start))
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
      if (!fifoName)
        if (auto named = dyn_cast<RouteEndpointOp>(endpoint.getOperation()))
          fifoName = named.getSymName();
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
    semaphoreLocks =
        device.getTargetModel().hasProperty(AIETargetModel::UsesSemaphoreLocks);
    // One pass instance serves every device in the module, and none of this
    // state means anything outside the device it was gathered from.
    channelSharers.clear();
    allocationSharers.clear();
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
    turnWalked.clear();
    bufferPlacements.clear();
    channelAssignments.clear();
    endpointStarts.clear();
    endpointTasks.clear();
    takenBDs = {};
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
    if (succeeded(allocated) &&
        (!channelSharers.empty() || !allocationSharers.empty()) &&
        failed(headShimSharingRoutes()))
      return signalPassFailure();
    if (failed(allocated)) {
      localPools.clear();
      if (failed(planBuffers(pools))) {
        InFlightDiagnostic error =
            walkSearchStopped
                ? bufferFailure.emitOpError(
                      "could not place buffers: the search for which pools "
                      "to make contiguous (one buffer, one BD per side) "
                      "stopped after ")
                      << DMAChannelAnalysis::kMaxWalkPlacements
                      << " placements tried in one group of neighbouring "
                         "mem tiles"
                : bufferFailure.emitOpError(kNoBufferRoom);
        if (std::optional<StringRef> fifo = bufferFailure.getFifoName())
          error << " for objectFIFO "
                << SymbolRefAttr::get(
                       device.getSymNameAttr(),
                       {FlatSymbolRefAttr::get(device.getContext(), *fifo)});
        SmallVector<Attribute> walked;
        for (Operation *op : turnWalked) {
          auto pool = cast<ObjectFifoPoolOp>(op);
          std::optional<StringRef> name = pool.getFifoName();
          if (pool.getContiguous() && name &&
              cast<TileLike>(pool.getTile().getDefiningOp()).isMemTile())
            walked.push_back(SymbolRefAttr::get(
                device.getSymNameAttr(),
                {FlatSymbolRefAttr::get(device.getContext(), *name)}));
        }
        if (!walked.empty()) {
          Diagnostic &note = error.attachNote();
          note << kTurnWalkedNote;
          llvm::interleaveComma(walked, note);
        }
      } else if (failed(planLocks(pools))) {
        if (lockSearchStopped) {
          InFlightDiagnostic error =
              lockSearchStoppedOn.emitOpError(
                  "could not decide whether the locks fit: the lock packing "
                  "search stopped after ")
              << LockPlan::kPackBudget << " steps";
          if (std::optional<StringRef> fifo = lockSearchStoppedOn.getFifoName())
            error << " for objectFIFO "
                  << SymbolRefAttr::get(
                         device.getSymNameAttr(),
                         {FlatSymbolRefAttr::get(device.getContext(), *fifo)});
        } else if (lockAccessFailure)
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

    // Routes still identify each sender's receiving buffers here, but hardware
    // BD IDs are assigned later. Preserve that pairing with temporary tags
    // before lowering erases the routes.
    llvm::EquivalenceClasses<Operation *> matchingPools;
    DenseMap<Operation *, int64_t> descriptorCounts;
    for (auto flow : device.getOps<RouteOp>()) {
      if (flow.getSources().size() != 1)
        continue;
      ObjectFifoPoolOp first;
      for (auto destName : flow.getDestinationNames()) {
        RouteEndpoint receiver = lookupEndpoint(destName);
        auto dst =
            receiver
                ? dyn_cast<ObjectFifoDmaEndpointOp>(receiver.getOperation())
                : nullptr;
        if (!dst || !dst.getOutOfOrder())
          continue;
        ObjectFifoPoolOp pool = dst.getPoolOp();
        if (!first)
          first = pool;
        matchingPools.unionSets(first.getOperation(), pool.getOperation());
        descriptorCounts[pool] = dst.getNumDescriptors();
      }
    }
    int32_t nextTag = 0;
    DenseMap<Operation *, DenseI32ArrayAttr> tagsOfGroup;
    for (ObjectFifoPoolOp pool : pools) {
      auto member = matchingPools.findLeader(pool.getOperation());
      if (member == matchingPools.member_end())
        continue;
      Operation *leader = *member;
      auto &tags = tagsOfGroup[leader];
      if (!tags) {
        SmallVector<int32_t> values;
        for (int64_t slot = 0; slot < descriptorCounts.lookup(pool); ++slot)
          values.push_back(nextTag++);
        tags = builder.getDenseI32ArrayAttr(values);
      }
      pool.setOutOfOrderTagsAttr(tags);
    }

    // A sender references the next hop's receiving slots, independently of
    // any receiving descriptors that fill its own pool.
    for (auto flow : device.getOps<RouteOp>()) {
      if (flow.getSources().size() != 1)
        continue;
      RouteEndpoint sender = lookupEndpoint(*flow.getSourceNames().begin());
      auto src = sender
                     ? dyn_cast<ObjectFifoDmaEndpointOp>(sender.getOperation())
                     : nullptr;
      if (!src || !src.getOutOfOrder())
        continue;
      for (auto destName : flow.getDestinationNames()) {
        RouteEndpoint receiver = lookupEndpoint(destName);
        auto dst =
            receiver
                ? dyn_cast<ObjectFifoDmaEndpointOp>(receiver.getOperation())
                : nullptr;
        if (dst && dst.getOutOfOrder()) {
          src.setOutOfOrderTagsAttr(dst.getPoolOp().getOutOfOrderTagsAttr());
          break;
        }
      }
    }

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
