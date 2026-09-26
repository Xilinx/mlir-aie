//===- AIEObjectFifoAllocate.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEBufferAllocation.h"
#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"

#include <map>
#include <set>

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
  /// Pools some endpoint writes into.
  DenseSet<Operation *> filledPools;
  /// The `aiex.dma_channel_reset_for` ops naming each fifo.
  llvm::StringMap<SmallVector<Operation *>> rearmUsers;
  /// Flows already lowered. A route endpoint reads its direction off the
  /// flow naming it, so these outlive the walk that replaces them.
  SmallVector<Operation *> loweredFlows;
  DenseMap<Value, SmallVector<int64_t>> plannedMemory;
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
  /// BDs spoken for on each channel of a placed tile, by (col, row, channel):
  /// the chains DMA programs start on it by index, plus the largest task the
  /// runtime sequence configures on it by index.
  std::map<std::tuple<int, int, int>, int64_t> fixedBDs;
  RouteEndpoint channelFailure;
  ObjectFifoPoolOp bufferFailure;
  Value lockFailure;
  Operation *lockAccessFailure = nullptr;

  bool sameTile(Value a, Value b) {
    if (a == b)
      return true;
    auto first = cast<TileLike>(a.getDefiningOp());
    auto second = cast<TileLike>(b.getDefiningOp());
    auto ac = first.tryGetCol(), ar = first.tryGetRow();
    auto bc = second.tryGetCol(), br = second.tryGetRow();
    return ac && ar && bc && br && ac == bc && ar == br;
  }

  int64_t memoryUsed(Value tile, int64_t extraSize = 0, int extraCount = 0) {
    SmallVector<BufferAllocation> layout;
    int64_t alignment =
        device.getTargetModel().getMemTileLoadStoreBusWidth() / 8;
    // Generated buffers are inserted immediately after their tile, before
    // existing buffers. Preserve that order for equal-sized aligned/unaligned
    // buffers, including coordinate-equivalent tile references.
    for (Operation &op : *device.getBody()) {
      if (auto placed = dyn_cast<TileLike>(op);
          placed && sameTile(tile, placed->getResult(0))) {
        Value value = placed->getResult(0);
        for (int64_t size : plannedMemory[value])
          layout.push_back({size, alignment, std::nullopt});
        if (value == tile)
          for (int i = 0; i < extraCount; ++i)
            layout.push_back({extraSize, alignment, std::nullopt});
      } else if (auto buffer = dyn_cast<BufferOp>(op);
                 buffer && sameTile(tile, buffer.getTile())) {
        layout.push_back({buffer.getAllocationSize(),
                          buffer.getAligned() ? alignment : 1,
                          buffer.getAddress()});
      }
    }
    return assignSequentialBufferAddresses(layout);
  }

  LogicalResult collectFixedMemory() {
    plannedMemory.clear();
    for (auto tile : device.getOps<TileLike>()) {
      if (!tile.isMemTile())
        continue;
      int64_t bytes = memoryUsed(tile->getResult(0));
      int64_t capacity = device.getTargetModel().getMemTileSize();
      if (bytes > capacity)
        return tile->emitOpError("existing buffers require ")
               << bytes << " bytes, exceeding MemTile capacity of " << capacity
               << " bytes";
    }
    return success();
  }

  bool canAccess(Value user, Value memory, bool localOnly = false) {
    if (user == memory)
      return true;
    auto userTile = dyn_cast<TileLike>(user.getDefiningOp());
    auto memoryTile = dyn_cast<TileLike>(memory.getDefiningOp());
    if (!userTile || !memoryTile)
      return false;
    const auto &target = device.getTargetModel();
    auto positions = [&](TileLike tile) {
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
    };
    // Before placement, defer only accesses that some compatible physical
    // positions could satisfy. Known coordinates can already disprove affinity.
    auto userPositions = positions(userTile);
    auto memoryPositions = positions(memoryTile);
    for (TileID u : userPositions)
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

  bool canPlace(ObjectFifoPoolOp pool, Value tile, int64_t sizeBytes,
                int count = 1) {
    if (cast<TileLike>(tile.getDefiningOp()).isMemTile() &&
        memoryUsed(tile, sizeBytes, count) >
            device.getTargetModel().getMemTileSize())
      return false;
    return llvm::all_of(poolUsers[pool],
                        [&](Value user) { return canAccess(user, tile); });
  }

  Value lockPlacement(ObjectFifoPoolOp pool, Operation *group) {
    if (Value placed = lockPlacements.lookup(group))
      return placed;
    Value local = localPools.lookup(pool);
    return local ? local : pool.getTile();
  }

  /// Reserve mandatory-local objects before any other pool can spill into
  /// their memory. Existing buffers are fixed and counted by identity, not by
  /// how many pools or endpoints reference them.
  LogicalResult planBuffers(ArrayRef<ObjectFifoPoolOp> pools) {
    plannedMemory.clear();
    bufferPlacements.clear();
    bufferFailure = nullptr;
    for (auto pool : pools) {
      if (!localPools.contains(pool))
        continue;
      Value tile = localPools.lookup(pool);
      int count = pool.getBuffers() ? 0 : pool.getDepth();
      int64_t size = pool.getObjectSizeInBytes();
      if (!canAccess(pool.getTile(), tile) ||
          !canPlace(pool, tile, size, count)) {
        bufferFailure = pool;
        return failure();
      }
      plannedMemory[tile].append(count, size);
    }
    SmallVector<ObjectFifoPoolOp> generated;
    for (auto pool : pools) {
      if (pool.getBuffers()) {
        for (auto buffer : pool.getBufferOps()) {
          Value tile = buffer.getBufferTile();
          if (tile && ((localPools.contains(pool) &&
                        !sameTile(tile, localPools.lookup(pool))) ||
                       !llvm::all_of(poolUsers[pool], [&](Value user) {
                         return canAccess(user, tile);
                       }))) {
            bufferFailure = pool;
            return failure();
          }
          bufferPlacements[pool].push_back(tile);
        }
        continue;
      }
      if (pool.getTileLike().isShimTile())
        continue;
      if (localPools.contains(pool))
        bufferPlacements[pool].append(pool.getDepth(), localPools.lookup(pool));
      else
        generated.append(pool.getDepth(), pool);
    }
    int retries = kSpillRetries;
    return success(placeBuffers(generated, retries));
  }

  /// The first layout tried is the greedy one. When a buffer fits nowhere,
  /// an earlier buffer may have spilled to the one neighbor it could use, so
  /// earlier spills are revisited, latest first, a bounded number of times.
  static constexpr int kSpillRetries = 256;

  bool placeBuffers(ArrayRef<ObjectFifoPoolOp> buffers, int &retries) {
    if (buffers.empty())
      return true;
    ObjectFifoPoolOp pool = buffers.front();
    int64_t size = pool.getObjectSizeInBytes();
    SmallVector<Value> candidates = placementsFor(pool, size);
    if (candidates.empty() && !bufferFailure)
      bufferFailure = pool;
    for (auto [index, tile] : llvm::enumerate(candidates)) {
      if (index > 0 && retries-- <= 0)
        break;
      plannedMemory[tile].push_back(size);
      bufferPlacements[pool].push_back(tile);
      if (placeBuffers(buffers.drop_front(), retries))
        return true;
      plannedMemory[tile].pop_back();
      bufferPlacements[pool].pop_back();
    }
    return false;
  }

  /// FIXME: choosing which tile a buffer lives on is the buffer allocator's
  /// job, not this pass's. Tracking used memory here to make that choice
  /// breaks the separation of concerns; --aie-assign-buffer-addresses should
  /// instead be free to move buffers between tiles that share a memory module.
  ///
  /// A MemTile buffer that does not fit at home spills to a neighbor reachable
  /// by its DMAs, preferring the emptier one so adjacent MemTiles
  /// keep room for their own spills. Which tiles neighbor an unplaced one is
  /// not yet known, so its buffers stay at home.
  SmallVector<Value> placementsFor(ObjectFifoPoolOp pool, int64_t sizeBytes) {
    TileLike home = pool.getTileLike();
    auto &target = device.getTargetModel();
    Value homeTile = home->getResult(0);
    if (canPlace(pool, homeTile, sizeBytes)) {
      return {homeTile};
    }
    if (!home.isMemTile())
      return {};

    auto homeOp = dyn_cast<TileOp>(home.getOperation());
    if (!homeOp) {
      return {};
    }

    SmallVector<TileOp> neighbors;
    for (int col : {homeOp.getCol() - 1, homeOp.getCol() + 1}) {
      if (col < 0 || col >= target.columns()) {
        continue;
      }
      TileOp neighbor = TileOp::getOrCreate(builder, device, col,
                                            homeOp.getRow(), pool.getLoc());
      using SharedMemory = AIETargetModel::SharedMemory;
      SharedMemory shared = sharedMemory(homeOp, neighbor);
      if (shared == SharedMemory::Second || shared == SharedMemory::Either) {
        neighbors.push_back(neighbor);
      }
    }
    llvm::stable_sort(neighbors, [&](TileOp a, TileOp b) {
      return memoryUsed(a.getResult()) < memoryUsed(b.getResult());
    });
    SmallVector<Value> fitting;
    for (TileOp neighbor : neighbors) {
      if (canPlace(pool, neighbor.getResult(), sizeBytes)) {
        fitting.push_back(neighbor.getResult());
      }
    }
    return fitting;
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

    auto initValues = pool.getInitValues();
    StringRef base = pool.getBaseName();

    SmallVector<Attribute> names;
    for (int i = 0; i < pool.getDepth(); i++) {
      ElementsAttr init =
          initValues ? cast<ElementsAttr>((*initValues)[i]) : nullptr;
      std::string name = (base + "_buff_" + std::to_string(i)).str();
      Value placement = bufferPlacements[pool][i];
      setInsertionPointOn(placement);
      lastPlaced[placement] = BufferOp::create(
          builder, pool.getLoc(), pool.getElemType(), placement,
          builder.getStringAttr(name), /*address=*/nullptr, init,
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

  bool needsLocks(ObjectFifoPoolOp pool) {
    if (pool.getDisableSynchronization()) {
      return false;
    }

    int depth = pool.getDepth();
    auto initValues = pool.getInitValues();
    int filled = initValues ? initValues->size() : 0;

    // A pool that starts full and is never refilled holds constants: its
    // readers have nothing to wait for. Locks would also let it be read only
    // as many times as they count, so a second launch would hang, unless a
    // `dma_channel_reset_for` re-arms them each launch.
    std::optional<StringRef> fifo = pool.getFifoName();
    return filled != depth || filled <= 0 || filledPools.contains(pool) ||
           (fifo && rearmUsers.contains(*fifo));
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
    auto place = [&](ObjectFifoPoolOp pool, Operation *group, int count) {
      SmallVector<Value> candidates{lockPlacement(pool, group)};
      if (!localPools.contains(pool)) {
        llvm::append_range(candidates, bufferPlacements[pool]);
        for (Operation *user : lockUsers[group])
          candidates.push_back(user->getOperand(0));
        for (auto tile : device.getOps<TileLike>())
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
      if (!needsLocks(pool))
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
    if (!needsLocks(pool))
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
        std::string name = (base + "_lock_" + std::to_string(i)).str();
        createLock(pool, lockPlacement(pool, pool), name, filled ? 1 : 0);
        names.push_back(FlatSymbolRefAttr::get(builder.getContext(), name));
      }
      pool.setLocksAttr(builder.getArrayAttr(names));
      return;
    }

    for (auto [index, segment] : llvm::enumerate(pool.getSegmentOps())) {
      if (segment.getProduceLock() && segment.getConsumeLock()) {
        continue;
      }
      std::string produce =
          (base + "_prod_lock_" + std::to_string(index)).str();
      std::string consume =
          (base + "_cons_lock_" + std::to_string(index)).str();
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
      if (needsLocks(pool) && !pool.getLocks())
        tiles.push_back(lockPlacement(pool, pool));
      else
        for (auto lock : pool.getLockOps())
          tiles.push_back(lock.getTile());
      return tiles;
    }
    for (auto segment : endpoint.getSelectedSegments()) {
      if (needsLocks(pool) &&
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

  /// BDs a program-owned endpoint's channel needs at once: its static chains,
  /// which stay configured, and the largest of its runtime tasks.
  int64_t endpointBDs(Operation *endpoint) {
    int64_t bds = 0, largestTask = 0;
    for (DMAStartOp start : endpointStarts.lookup(endpoint))
      bds += countBDs(start);
    for (auto task : endpointTasks.lookup(endpoint))
      largestTask = std::max(largestTask, countBDs(task.getBody()));
    return bds + largestTask;
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
    std::map<std::tuple<int, int, int>, int64_t> largestTask;
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
          } else if (auto at = key(program.getTile(), start.getChannel())) {
            fixedBDs[*at] += countBDs(start);
          }
        }
    device.walk([&](Operation *op) {
      if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
        if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(
                SymbolTable::lookupNearestSymbolFrom(
                    device, task.getAlloc().getRootReference())))
          endpointTasks[endpoint].push_back(task);
      } else if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op)) {
        if (auto at = key(task.getTile(), task.getChannel())) {
          int64_t &largest = largestTask[*at];
          largest = std::max(largest, countBDs(task.getBody()));
        }
      }
    });
    for (auto &[at, bds] : largestTask)
      fixedBDs[at] += bds;
  }

  /// On a tile whose BD ids are split between channels, the free channel whose
  /// BDs have the most room left once every channel sharing them has its
  /// demand; ties, and tiles without such a split, go to the lowest index.
  /// Falls back to first-free where coordinates are unknown.
  int chooseChannel(DMAChannelAnalysis &channels, RouteEndpoint endpoint,
                    const std::map<std::tuple<int, int, int>, int64_t> &demand,
                    bool adjacent) {
    TileLike tile = tileOf(endpoint);
    DMAChannelDir dir = endpoint.getRouteDirection();
    auto col = tile.tryGetCol(), row = tile.tryGetRow();
    if (!col || !row)
      return channels.getDMAChannelIndex(tile, dir, adjacent,
                                         endpoint.getOperation());
    const AIETargetModel &target = device.getTargetModel();
    uint32_t numBDs = target.getNumBDs(*col, *row);
    int numChannels = std::max(tile.getNumSourceConnections(WireBundle::DMA),
                               tile.getNumDestConnections(WireBundle::DMA));
    auto sharesBDs = [&](int a, int b) {
      for (uint32_t bd = 0; bd < numBDs; ++bd)
        if (target.isBdChannelAccessible(*col, *row, bd, a) &&
            target.isBdChannelAccessible(*col, *row, bd, b))
          return true;
      return false;
    };
    int best = -1;
    int64_t bestRoom = 0;
    int limit = DMAChannelAnalysis::getDMAChannelLimit(tile, dir, adjacent);
    for (int channel = 0; channel < limit; ++channel) {
      if (!channels.isChannelFree(tile, dir, channel))
        continue;
      int64_t room = target.getNumBDsForChannel(*col, *row, channel);
      for (int other = 0; other < numChannels; ++other)
        if (sharesBDs(channel, other))
          if (auto it = demand.find({*col, *row, other}); it != demand.end())
            room -= it->second;
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

  LogicalResult assignChannels(DMAChannelAnalysis &channels,
                               bool diagnose = true) {
    channelAssignments.clear();
    channelFailure = nullptr;
    SmallVector<RouteEndpoint> pending;
    // BDs per placed (col, row, channel) as channels are handed out, for
    // choosing among a tile's channels by the BD ids they can use.
    std::map<std::tuple<int, int, int>, int64_t> demand = fixedBDs;
    auto noteDemand = [&](RouteEndpoint endpoint, int channel) {
      TileLike tile = tileOf(endpoint);
      auto col = tile.tryGetCol(), row = tile.tryGetRow();
      if (!col || !row)
        return;
      int64_t bds = 0;
      if (auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation()))
        bds = dma.getNumBDs();
      else if (isProgramOwned(endpoint))
        bds = endpointBDs(endpoint.getOperation());
      demand[{*col, *row, channel}] += bds;
    };
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      DMAChannelDir dir = endpoint.getRouteDirection();
      std::optional<int> channel = endpoint.getRouteChannel();
      // A core's stream port is named by the design, not drawn from the tile's
      // DMA channels.
      if (endpoint.getRouteBundle() == WireBundle::Core) {
        if (!channel) {
          if (!diagnose)
            return failure();
          return endpoint->emitOpError("a stream port names its own channel");
        }
        if (failed(channels.checkAIEStreamIndex(tileOf(endpoint),
                                                {dir, *channel}, diagnose)))
          return failure();
        continue;
      }

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
    // then chosen around, the most demanding first.
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
      return aProgram && programBDs.lookup(a.getOperation()) >
                             programBDs.lookup(b.getOperation());
    });

    for (auto endpoint : pending) {
      DMAChannelDir dir = endpoint.getRouteDirection();
      int channel =
          isProgramOwned(endpoint)
              ? chooseChannel(channels, endpoint, demand,
                              reachesAdjacentTile(endpoint))
              : channels.getDMAChannelIndex(tileOf(endpoint), dir,
                                            reachesAdjacentTile(endpoint),
                                            endpoint.getOperation());
      if (channel < 0) {
        channelFailure = endpoint;
        if (!diagnose)
          return failure();
        TileLike tile = tileOf(endpoint);
        bool adjacent = reachesAdjacentTile(endpoint);
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
    return success();
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
    llvm::stable_sort(candidates, [](ObjectFifoPoolOp a, ObjectFifoPoolOp b) {
      int64_t aBytes =
          a.getBuffers() ? 0 : a.getObjectSizeInBytes() * a.getDepth();
      int64_t bBytes =
          b.getBuffers() ? 0 : b.getObjectSizeInBytes() * b.getDepth();
      return aBytes < bBytes;
    });
    for (auto pool : candidates) {
      localPools[pool] = localTile;
      if (succeeded(planAllocation(pools, tried)))
        return success();
      localPools.erase(pool);
    }
    return failure();
  }

  /// FIXME: assigning packet IDs does not belong in this pass. The shape it
  /// wants is an `%id = aie.packet_id` value that `aie.packet_flow` takes as an
  /// argument, concretized by a pass of its own; this set then becomes that
  /// pass's analysis.
  ///
  /// Packet IDs already spoken for, by an existing packet flow or by a flow
  /// that pinned one.
  llvm::SmallDenseSet<int> takenPacketIDs() {
    llvm::SmallDenseSet<int> taken;
    device.walk([&](PacketFlowOp flow) { taken.insert(flow.IDInt()); });
    for (auto flow : device.getOps<RouteOp>()) {
      if (auto pinned = flow.getPacketId()) {
        taken.insert(*pinned);
      }
    }
    return taken;
  }

  /// A packet-switched flow shares the stream with others, so every buffer
  /// descriptor the source emits has to carry the packet header.
  LogicalResult lowerPacketFlow(RouteOp flow, RouteEndpoint source,
                                int packetID) {

    auto info =
        PacketInfoAttr::get(builder.getContext(), /*pkt_type=*/0, packetID);
    source.setRoutePacket(info);

    builder.setInsertionPoint(flow);
    auto packetFlow = PacketFlowOp::create(
        builder, flow.getLoc(),
        builder.getIntegerAttr(builder.getI8Type(), packetID), nullptr,
        nullptr);
    OpBuilder::InsertionGuard g(builder);
    Block &ports = packetFlow.getRegion().emplaceBlock();
    builder.setInsertionPointToStart(&ports);
    EndOp::create(builder, flow.getLoc());

    builder.setInsertionPointToStart(&ports);
    PacketSourceOp::create(builder, flow.getLoc(), source.getTile(),
                           source.getRouteBundle(), channelOf(source));
    for (auto destName :
         flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
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

  RouteEndpoint lookupEndpoint(FlatSymbolRefAttr name) {
    return dyn_cast_or_null<RouteEndpoint>(
        SymbolTable::lookupNearestSymbolFrom(device, name.getAttr()));
  }

  LogicalResult lowerFlows() {
    int maxPacketID =
        static_cast<int>(device.getTargetModel().getMaxPacketId());
    llvm::SmallDenseSet<int> taken = takenPacketIDs();
    int nextFree = 0;
    for (auto flow : device.getOps<RouteOp>()) {
      auto source = lookupEndpoint(flow.getSourceAttr());
      loweredFlows.push_back(flow);

      // The pass flag is a default for flows that express no preference, so a
      // device may mix circuit- and packet-switched connections.
      if (flow.getPacket() || clPacketSwObjectFifos) {
        int packetID;
        if (auto pinned = flow.getPacketId()) {
          packetID = *pinned;
          if (packetID > maxPacketID) {
            return flow.emitOpError("packet_id ")
                   << packetID << " is out of range (max " << maxPacketID
                   << ")";
          }
        } else {
          while (taken.contains(nextFree)) {
            nextFree++;
          }
          if (nextFree > maxPacketID) {
            return flow.emitOpError("max number of packet IDs reached");
          }
          packetID = nextFree;
          taken.insert(packetID);
        }
        if (failed(lowerPacketFlow(flow, source, packetID))) {
          return failure();
        }
        continue;
      }

      int sourceChannel = channelOf(source);
      builder.setInsertionPoint(flow);
      for (auto destName :
           flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
        auto dest = lookupEndpoint(destName);
        FlowOp::create(builder, flow.getLoc(), source.getTile(),
                       source.getRouteBundle(), sourceChannel, dest.getTile(),
                       dest.getRouteBundle(), channelOf(dest));
      }
    }
    return success();
  }

  /// Write each endpoint-named dma_start's channel into it, so every later
  /// pass sees an index.
  LogicalResult resolveStarts() {
    for (auto endpoint : device.getOps<RouteEndpointOp>()) {
      for (DMAStartOp start : endpointStarts.lookup(endpoint)) {
        DMAChannelDir dir = endpoint.getRouteDirection();
        if (start.getChannelDir() != dir)
          return start.emitOpError("starts ")
                 << stringifyDMAChannelDir(start.getChannelDir()) << " on @"
                 << endpoint.getSymName() << ", but its route makes it "
                 << stringifyDMAChannelDir(dir);
        // Allocation stamps the header on runtime tasks only.
        if (endpoint.getPacket())
          return start.emitOpError("names @")
                 << endpoint.getSymName()
                 << ", the source of a packet-switched route; a DMA program's "
                    "BDs would not carry its header";
        start.setChannelIndex(channelOf(endpoint));
        start.removeEndpointAttr();
      }
    }
    return success();
  }

  /// A runtime task naming an endpoint that gets no shim allocation is
  /// configured on the endpoint's tile and channel directly.
  void rewriteTasks() {
    for (auto endpoint : device.getOps<RouteEndpointOp>()) {
      if (endpoint.getTileLike().isShimTile() && endpoint.getFifoName())
        continue;
      for (auto task : endpointTasks.lookup(endpoint)) {
        builder.setInsertionPoint(task);
        auto configured = AIEX::DMAConfigureTaskOp::create(
            builder, task.getLoc(), builder.getIndexType(), endpoint.getTile(),
            DMAChannelDirAttr::get(builder.getContext(),
                                   endpoint.getRouteDirection()),
            builder.getI32IntegerAttr(channelOf(endpoint)),
            builder.getBoolAttr(task.getIssueToken()),
            builder.getI32IntegerAttr(task.getRepeatCount()),
            task.getRepeatCountVal(), endpoint.getPacketAttr());
        task.getResult().replaceAllUsesWith(configured.getResult());
        configured.getBody().takeBody(task.getBody());
        task.erase();
      }
    }
  }

  void collectRearmUsers() {
    device.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "aiex.dma_channel_reset_for") {
        return;
      }
      auto sym = op->getAttrOfType<FlatSymbolRefAttr>("objfifo");
      if (!sym) {
        return;
      }
      // Split may already have pointed this at the fifo's shim endpoint.
      StringRef name = sym.getValue();
      if (auto endpoint = lookupEndpoint(sym)) {
        if (auto fifoName = endpoint.getFifoName()) {
          name = *fifoName;
        }
      }
      rearmUsers[name].push_back(op);
    });
  }

  /// An `aiex.dma_channel_reset_for` outlives the fifo it names, so record the
  /// channels and locks it has to re-arm and point it at that record. Shim
  /// endpoints are left out: the host re-pushes those itself.
  LogicalResult bindRearmTargets() {
    if (rearmUsers.empty()) {
      return success();
    }

    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (auto &[fifoName, users] : rearmUsers) {
      SmallVector<Value> channelTiles, lockValues;
      SmallVector<int32_t> channelDirs, channelIndices, lockInits;

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
      }
      for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
        if (pool.getFifoName() != fifoName || pool.getTileLike().isShimTile()) {
          continue;
        }
        for (LockOp lock : pool.getLockOps()) {
          lockValues.push_back(lock.getResult());
          lockInits.push_back(lock.getInit().value_or(0));
        }
      }

      if (channelTiles.empty() && lockValues.empty()) {
        for (Operation *user : users) {
          user->emitOpError() << "objectFIFO '" << fifoName
                              << "' has no resident core/mem DMA channels or "
                                 "locks to re-arm";
        }
        return failure();
      }

      std::string name = (fifoName + "_rearm").str();
      for (unsigned suffix = 0; lookupNamedOpIn(device, StringRef(name));
           suffix++) {
        name = (fifoName + "_rearm_" + std::to_string(suffix)).str();
      }

      // head_bd_ids and repeat_counts are filled in by --aie-assign-bd-ids.
      ObjectFifoRearmBindingOp::create(
          builder, device.getLoc(), builder.getStringAttr(name),
          ValueRange(channelTiles), ValueRange(lockValues),
          builder.getDenseI32ArrayAttr(channelDirs),
          builder.getDenseI32ArrayAttr(channelIndices),
          builder.getDenseI32ArrayAttr(lockInits),
          /*head_bd_ids=*/DenseI32ArrayAttr(),
          /*repeat_counts=*/DenseI32ArrayAttr());
      auto target = FlatSymbolRefAttr::get(builder.getContext(), name);
      for (Operation *user : users) {
        user->setAttr("objfifo", target);
      }
    }
    return success();
  }

  /// A shim endpoint has no memory of its own, so the runtime needs its channel
  /// spelled out under the name the sequence refers to.
  void emitShimAllocations() {
    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      std::optional<StringRef> fifoName = endpoint.getFifoName();
      std::optional<int> channel = endpoint.getRouteChannel();
      if (!tileOf(endpoint).isShimTile() || !fifoName || !channel) {
        continue;
      }
      std::string name = (*fifoName + "_shim_alloc").str();
      if (!SymbolTable::lookupNearestSymbolFrom<ShimDMAAllocationOp>(
              device, builder.getStringAttr(name))) {
        ShimDMAAllocationOp::create(
            builder, endpoint.getLoc(), builder.getStringAttr(name),
            endpoint.getTile(),
            DMAChannelDirAttr::get(builder.getContext(),
                                   endpoint.getRouteDirection()),
            builder.getI64IntegerAttr(*channel),
            builder.getBoolAttr(endpoint.getRouteBundle() == WireBundle::PLIO),
            endpoint->getAttrOfType<PacketInfoAttr>("packet"));
      }
      // The runtime sequence reaches the fifo through this record.
      (void)SymbolTable::replaceAllSymbolUses(
          endpoint, builder.getStringAttr(name), device);
    }
  }

  /// Alternative packing order: prioritize DMA demand within each home tile,
  /// but merge those lists by object size to avoid globally prioritizing small
  /// pools over large objects on unrelated tiles. This is only a heuristic;
  /// the same buffer, lock and channel planner validates both orders.
  SmallVector<ObjectFifoPoolOp>
  demandOrderedPools(ArrayRef<ObjectFifoPoolOp> pools) {
    SmallVector<ObjectFifoPoolOp> ordered(pools);
    SmallVector<size_t> memTileSlots;
    SmallVector<Value> memTileHomes;
    DenseMap<Value, SmallVector<ObjectFifoPoolOp>> memTilePools;
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
    for (auto [index, pool] : llvm::enumerate(ordered)) {
      if (pool.getTileLike().isMemTile()) {
        memTileSlots.push_back(index);
        Value home = pool.getTile();
        for (Value known : memTileHomes)
          if (sameTile(home, known)) {
            home = known;
            break;
          }
        if (!memTilePools.count(home))
          memTileHomes.push_back(home);
        memTilePools[home].push_back(pool);
      }
    }
    DenseMap<Value, size_t> nextPool;
    auto demand = [&](ObjectFifoPoolOp pool) {
      auto [input, output] = poolChannelDemand.lookup(pool.getOperation());
      return std::max(input, output);
    };
    for (Value home : memTileHomes) {
      auto &tilePools = memTilePools[home];
      llvm::stable_sort(tilePools, [&](ObjectFifoPoolOp a, ObjectFifoPoolOp b) {
        if (demand(a) != demand(b))
          return demand(a) > demand(b);
        return a.getObjectSizeInBytes() > b.getObjectSizeInBytes();
      });
    }
    for (size_t slot : memTileSlots) {
      Value bestHome;
      ObjectFifoPoolOp bestPool;
      for (Value home : memTileHomes) {
        auto &tilePools = memTilePools[home];
        size_t index = nextPool[home];
        if (index >= tilePools.size())
          continue;
        ObjectFifoPoolOp pool = tilePools[index];
        if (!bestPool ||
            pool.getObjectSizeInBytes() > bestPool.getObjectSizeInBytes() ||
            (pool.getObjectSizeInBytes() == bestPool.getObjectSizeInBytes() &&
             demand(pool) > demand(bestPool))) {
          bestHome = home;
          bestPool = pool;
        }
      }
      ordered[slot] = bestPool;
      ++nextPool[bestHome];
    }
    return ordered;
  }

  void runOnOperation() override {
    device = getOperation();
    builder = OpBuilder(device.getContext());
    // One pass instance serves every device in the module, and none of this
    // state means anything outside the device it was gathered from.
    lastPlaced.clear();
    filledPools.clear();
    rearmUsers.clear();
    loweredFlows.clear();
    localPools.clear();
    poolUsers.clear();
    lockUsers.clear();
    lockPlacements.clear();

    if (failed(collectFixedMemory()))
      return signalPassFailure();
    collectChannelPrograms();

    // Preserve successful largest-first allocations, including their locality
    // repairs. Only try demand ordering when that search fails.
    SmallVector<ObjectFifoPoolOp> pools(device.getOps<ObjectFifoPoolOp>());
    SmallVector<size_t> memTileSlots;
    SmallVector<ObjectFifoPoolOp> memTilePools;
    for (auto [index, pool] : llvm::enumerate(pools)) {
      if (pool.getTileLike().isMemTile()) {
        memTileSlots.push_back(index);
        memTilePools.push_back(pool);
      }
    }
    llvm::stable_sort(memTilePools, [](ObjectFifoPoolOp a, ObjectFifoPoolOp b) {
      return a.getObjectSizeInBytes() > b.getObjectSizeInBytes();
    });
    for (auto [slot, pool] : llvm::zip(memTileSlots, memTilePools)) {
      pools[slot] = pool;
    }

    collectRearmUsers();
    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      poolUsers[endpoint.getPoolOp()].push_back(endpoint.getTile());
      collectLockUsers(endpoint);
      if (!endpoint.drains()) {
        filledPools.insert(endpoint.getPoolOp());
      }
    }
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      poolUsers[endpoint.getPoolOp()].push_back(endpoint.getTile());
      collectLockUsers(endpoint);
      if (!endpoint.drains()) {
        filledPools.insert(endpoint.getPoolOp());
      }
    }

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
    if (failed(allocated)) {
      localPools.clear();
      if (failed(planBuffers(pools))) {
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
    for (auto endpoint : device.getOps<RouteEndpoint>())
      if (auto it = channelAssignments.find(endpoint.getOperation());
          it != channelAssignments.end())
        endpoint.setRouteChannel(it->second);

    if (failed(bindRearmTargets())) {
      return signalPassFailure();
    }
    if (failed(lowerFlows())) {
      return signalPassFailure();
    }
    // After lowerFlows, which stamps a packet source's header.
    if (failed(resolveStarts())) {
      return signalPassFailure();
    }
    rewriteTasks();
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

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoAllocatePass(bool packetSwitched) {
  AIEObjectFifoAllocateOptions options;
  options.clPacketSwObjectFifos = packetSwitched;
  return std::make_unique<AIEObjectFifoAllocatePass>(options);
}
