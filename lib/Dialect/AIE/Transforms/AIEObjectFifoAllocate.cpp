//===- AIEObjectFifoAllocate.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"

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
  /// Flows already lowered. A route endpoint reads its direction off the
  /// flow naming it, so these outlive the walk that replaces them.
  SmallVector<Operation *> loweredFlows;
  /// Passes the longest-running drainer of each pool makes over it.
  DenseMap<Operation *, int> drainerIterations;
  DenseMap<Value, int64_t> fixedMemory;
  DenseMap<Value, int64_t> plannedMemory;
  DenseMap<Operation *, SmallVector<Value>> bufferPlacements;
  DenseMap<Operation *, int> channelAssignments;
  DenseMap<Operation *, Value> localPools;
  DenseMap<Operation *, SmallVector<Value>> poolUsers;
  RouteEndpoint channelFailure;
  ObjectFifoPoolOp bufferFailure;
  Value lockFailure;

  bool sameTile(Value a, Value b) {
    if (a == b)
      return true;
    auto first = cast<TileLike>(a.getDefiningOp());
    auto second = cast<TileLike>(b.getDefiningOp());
    auto ac = first.tryGetCol(), ar = first.tryGetRow();
    auto bc = second.tryGetCol(), br = second.tryGetRow();
    return ac && ar && bc && br && ac == bc && ar == br;
  }

  int64_t memoryUsed(Value tile) {
    int64_t bytes = 0;
    for (auto [placed, size] : plannedMemory)
      if (sameTile(tile, placed))
        bytes += size;
    return bytes;
  }

  LogicalResult collectFixedMemory() {
    fixedMemory.clear();
    for (auto buffer : device.getOps<BufferOp>())
      fixedMemory[buffer.getTile()] += buffer.getAllocationSize();
    plannedMemory = fixedMemory;
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

  bool canAccess(Value user, Value memory) {
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
        auto shared = target.getSharedMemory(u, m);
        if (shared == AIETargetModel::SharedMemory::Second ||
            shared == AIETargetModel::SharedMemory::Either)
          return true;
      }
    return false;
  }

  bool canPlace(ObjectFifoPoolOp pool, Value tile, int64_t sizeBytes) {
    if (pool.getTileLike().isMemTile() &&
        memoryUsed(tile) + sizeBytes > device.getTargetModel().getMemTileSize())
      return false;
    return llvm::all_of(poolUsers[pool],
                        [&](Value user) { return canAccess(user, tile); });
  }

  Value lockPlacement(ObjectFifoPoolOp pool) {
    Value local = localPools.lookup(pool);
    return local ? local : pool.getTile();
  }

  /// Reserve mandatory-local objects before any other pool can spill into
  /// their memory. Existing buffers are fixed and counted by identity, not by
  /// how many pools or endpoints reference them.
  LogicalResult planBuffers(ArrayRef<ObjectFifoPoolOp> pools) {
    plannedMemory = fixedMemory;
    bufferPlacements.clear();
    bufferFailure = nullptr;
    for (auto pool : pools) {
      if (!localPools.contains(pool))
        continue;
      Value tile = localPools.lookup(pool);
      int64_t bytes =
          pool.getBuffers() ? 0 : pool.getObjectSizeInBytes() * pool.getDepth();
      if (!canAccess(pool.getTile(), tile) || !canPlace(pool, tile, bytes)) {
        bufferFailure = pool;
        return failure();
      }
      plannedMemory[tile] += bytes;
    }
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
      for (int i = 0; i < pool.getDepth(); ++i) {
        Value tile = localPools.contains(pool)
                         ? localPools.lookup(pool)
                         : placementFor(pool, pool.getObjectSizeInBytes());
        if (!tile) {
          bufferFailure = pool;
          return failure();
        }
        bufferPlacements[pool].push_back(tile);
        if (!localPools.contains(pool))
          plannedMemory[tile] += pool.getObjectSizeInBytes();
      }
    }
    return success();
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
  Value placementFor(ObjectFifoPoolOp pool, int64_t sizeBytes) {
    TileLike home = pool.getTileLike();
    auto &target = device.getTargetModel();
    Value homeTile = home->getResult(0);
    if (canPlace(pool, homeTile, sizeBytes)) {
      return homeTile;
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
      TileOp neighbor =
          TileOp::getOrCreate(builder, device, col, homeOp.getRow());
      using SharedMemory = AIETargetModel::SharedMemory;
      SharedMemory shared = sharedMemory(homeOp, neighbor);
      if (shared == SharedMemory::Second || shared == SharedMemory::Either) {
        neighbors.push_back(neighbor);
      }
    }
    llvm::stable_sort(neighbors, [&](TileOp a, TileOp b) {
      return memoryUsed(a.getResult()) < memoryUsed(b.getResult());
    });
    for (TileOp neighbor : neighbors) {
      if (canPlace(pool, neighbor.getResult(), sizeBytes)) {
        return neighbor.getResult();
      }
    }
    return {};
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

  LockOp createLock(ObjectFifoPoolOp pool, StringRef name, int value) {
    Value tile = lockPlacement(pool);
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

    // A pool that starts full, is never refilled and is read more than once
    // holds constants: its readers have nothing to wait for.
    //
    // FIXME: revisit whether the pass count belongs in that test. Nothing
    // refills this pool however often it is read, so the locks look like dead
    // weight either way; dropping the clause also frees every `init_values`
    // fifo of its locks, which wants looking at on its own.
    return filled != depth || filled <= 0 || filledPools.contains(pool) ||
           drainerIterations.lookup(pool) <= 1;
  }

  LogicalResult planLocks(ArrayRef<ObjectFifoPoolOp> pools) {
    lockFailure = {};
    DenseMap<std::pair<int, int>, int64_t> used;
    auto reserve = [&](Value tile, int64_t count) {
      auto like = cast<TileLike>(tile.getDefiningOp());
      auto col = like.tryGetCol(), row = like.tryGetRow();
      if (!like.isMemTile() || !col || !row)
        return success();
      if ((used[{*col, *row}] += count) >
          device.getTargetModel().getNumLocks(*col, *row)) {
        lockFailure = tile;
        return failure();
      }
      return success();
    };
    for (auto lock : device.getOps<LockOp>())
      if (failed(reserve(lock.getTile(), 1)))
        return failure();
    for (auto pool : pools) {
      if (!pool.getTileLike().isMemTile() || !needsLocks(pool))
        continue;
      int64_t count =
          2 * llvm::count_if(pool.getSegmentOps(), [](auto segment) {
            return !(segment.getProduceLock() && segment.getConsumeLock());
          });
      if (failed(reserve(lockPlacement(pool), count)))
        return failure();
    }
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
        createLock(pool, name, filled ? 1 : 0);
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
      createLock(pool, produce, (depth - filled) * repeat);
      createLock(pool, consume, filled * repeat);
      segment.setProduceLockAttr(
          FlatSymbolRefAttr::get(builder.getContext(), produce));
      segment.setConsumeLockAttr(
          FlatSymbolRefAttr::get(builder.getContext(), consume));
    }
  }

  /// MemTiles use counting locks. Predict the locks DMA lowering will use,
  /// including hand-written locks and the pairs allocation will create.
  SmallVector<Value> dmaLockTiles(ObjectFifoDmaEndpointOp endpoint) {
    SmallVector<Value> tiles;
    auto pool = endpoint.getPoolOp();
    if (pool.getDepth() == 0)
      return tiles;
    for (auto segment : endpoint.getSelectedSegments()) {
      if (needsLocks(pool) &&
          !(segment.getProduceLock() && segment.getConsumeLock())) {
        tiles.push_back(lockPlacement(pool));
        continue;
      }
      auto acquire = endpoint.drains() ? segment.getConsumeLockAttr()
                                       : segment.getProduceLockAttr();
      auto release = endpoint.drains() ? segment.getProduceLockAttr()
                                       : segment.getConsumeLockAttr();
      if (!acquire)
        continue;
      for (auto name : {acquire, release})
        if (name)
          if (auto lock =
                  SymbolTable::lookupNearestSymbolFrom<LockOp>(device, name))
            tiles.push_back(lock.getTile());
    }
    return tiles;
  }

  /// Both buffers and locks must be local to use the local-only channels.
  /// Distinct unresolved tiles may be neighbors after placement, so they
  /// cannot be assumed local when assigning a placed endpoint's channels.
  bool reachesAdjacentTile(RouteEndpoint endpoint) {
    auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation());
    if (!dma || !dma.getTileLike().isMemTile()) {
      return false;
    }
    ObjectFifoPoolOp pool = dma.getPoolOp();
    auto remote = [&](Value tile) {
      return tile && !sameTile(tile, endpoint.getTile());
    };
    return pool && (llvm::any_of(bufferPlacements[pool], remote) ||
                    llvm::any_of(dmaLockTiles(dma), remote));
  }

  TileLike tileOf(RouteEndpoint endpoint) {
    return dyn_cast<TileLike>(endpoint.getTile().getDefiningOp());
  }

  LogicalResult assignChannels(DMAChannelAnalysis &channels,
                               bool diagnose = true) {
    channelAssignments.clear();
    channelFailure = nullptr;
    SmallVector<RouteEndpoint> pending;
    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      DMAChannelDir dir = endpoint.getRouteDirection();
      std::optional<int> channel = endpoint.getRouteChannel();
      if (auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation());
          dma && dma.getTileLike().isMemTile() &&
          !llvm::all_of(dmaLockTiles(dma), [&](Value tile) {
            return canAccess(endpoint.getTile(), tile);
          })) {
        channelFailure = endpoint;
        if (!diagnose)
          return failure();
        return endpoint->emitOpError("cannot access pool locks");
      }

      // A core's stream port is named by the design, not drawn from the tile's
      // DMA channels.
      if (endpoint.getRouteBundle() == WireBundle::Core) {
        if (!channel) {
          if (!diagnose)
            return failure();
          return endpoint->emitOpError("a stream port names its own channel");
        }
        channels.checkAIEStreamIndex(tileOf(endpoint), {dir, *channel});
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
        if (channels.reservePinnedChannel(tileOf(endpoint), dir, *channel) <
            0) {
          if (!diagnose)
            return failure();
          return endpoint->emitOpError("pinned ")
                 << stringifyDMAChannelDir(dir) << " DMA channel " << *channel
                 << " is out of range or already in use on this tile";
        }
        continue;
      }
      pending.push_back(endpoint);
    }

    // Endpoints reaching a spilled buffer draw from a restricted channel
    // range, so they are served before the unrestricted ones.
    llvm::stable_sort(pending, [&](RouteEndpoint a, RouteEndpoint b) {
      return reachesAdjacentTile(a) && !reachesAdjacentTile(b);
    });

    for (auto endpoint : pending) {
      DMAChannelDir dir = endpoint.getRouteDirection();
      int channel = channels.getDMAChannelIndex(tileOf(endpoint), dir,
                                                reachesAdjacentTile(endpoint));
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
        for (auto contributor : device.getOps<RouteEndpoint>()) {
          if (!sameTile(contributor.getTile(), endpoint.getTile()) ||
              contributor.getRouteBundle() != WireBundle::DMA ||
              contributor.getRouteDirection() != dir) {
            continue;
          }
          auto &note = diag.attachNote(contributor.getLoc());
          note << "DMA endpoint @"
               << cast<SymbolOpInterface>(contributor.getOperation()).getName();
          if (auto fifo = contributor->getAttrOfType<StringAttr>("fifoName")) {
            note << " for ObjectFifo @" << fifo.getValue();
          }
          if (reachesAdjacentTile(contributor)) {
            note << " requires adjacent MemTile access";
          }
        }
        return failure();
      }
      channelAssignments[endpoint.getOperation()] = channel;
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

  /// An `aiex.dma_channel_reset_for` outlives the fifo it names, so record the
  /// channels and locks it has to re-arm and point it at that record. Shim
  /// endpoints are left out: the host re-pushes those itself.
  LogicalResult bindRearmTargets() {
    llvm::StringMap<SmallVector<Operation *>> usersByFifo;
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
      usersByFifo[name].push_back(op);
    });
    if (usersByFifo.empty()) {
      return success();
    }

    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (auto &[fifoName, users] : usersByFifo) {
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
      for (unsigned suffix = 0; device.lookupSymbol(name); suffix++) {
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
    loweredFlows.clear();
    drainerIterations.clear();
    localPools.clear();
    poolUsers.clear();

    if (failed(collectFixedMemory()))
      return signalPassFailure();

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

    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      poolUsers[endpoint.getPoolOp()].push_back(endpoint.getTile());
      if (!endpoint.drains()) {
        filledPools.insert(endpoint.getPoolOp());
      }
    }
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      poolUsers[endpoint.getPoolOp()].push_back(endpoint.getTile());
      if (!endpoint.drains()) {
        filledPools.insert(endpoint.getPoolOp());
        continue;
      }
      int &iterations = drainerIterations[endpoint.getPoolOp()];
      iterations = std::max(iterations, endpoint.getIterCount().value_or(1));
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
        lockFailure.getDefiningOp()->emitOpError(
            "could not place locks within MemTile lock capacity");
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
