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

  /// Bytes already committed to `tile` by every buffer placed so far.
  int64_t usedMemory(Value tile) {
    int64_t total = 0;
    for (auto buffer : device.getOps<BufferOp>()) {
      if (buffer.getTile() == tile) {
        total += buffer.getAllocationSize();
      }
    }
    return total;
  }

  /// FIXME: choosing which tile a buffer lives on is the buffer allocator's
  /// job, not this pass's. Tracking used memory here to make that choice
  /// breaks the separation of concerns; --aie-assign-buffer-addresses should
  /// instead be free to move buffers between tiles that share a memory module.
  ///
  /// A MemTile buffer that does not fit at home spills to whichever neighbor
  /// its DMAs can still reach, preferring the emptier one so adjacent MemTiles
  /// keep room for their own spills. Which tiles neighbor an unplaced one is
  /// not yet known, so its buffers stay at home.
  Value placementFor(TileLike home, int64_t sizeBytes) {
    auto &target = device.getTargetModel();
    Value homeTile = home->getResult(0);
    if (!home.isMemTile() ||
        usedMemory(homeTile) + sizeBytes <= target.getMemTileSize()) {
      return homeTile;
    }

    auto homeOp = dyn_cast<TileOp>(home.getOperation());
    if (!homeOp) {
      return homeTile;
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
      return usedMemory(a.getResult()) < usedMemory(b.getResult());
    });
    for (TileOp neighbor : neighbors) {
      if (usedMemory(neighbor.getResult()) + sizeBytes <=
          target.getMemTileSize()) {
        return neighbor.getResult();
      }
    }
    return homeTile;
  }

  /// Buffers and locks live directly below the tile declarations.
  /// Buffers and locks live directly below the tile declarations.
  ///
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
    int64_t sizeBytes = pool.getObjectSizeInBytes();
    StringRef base = pool.getBaseName();

    SmallVector<Attribute> names;
    for (int i = 0; i < pool.getDepth(); i++) {
      ElementsAttr init =
          initValues ? cast<ElementsAttr>((*initValues)[i]) : nullptr;
      std::string name = (base + "_buff_" + std::to_string(i)).str();
      Value placement = placementFor(home, sizeBytes);
      setInsertionPointOn(placement);
      lastPlaced[placement] = BufferOp::create(
          builder, pool.getLoc(), pool.getElemType(), placement,
          builder.getStringAttr(name), /*address=*/nullptr, init,
          /*mem_bank=*/nullptr, /*aligned=*/nullptr);
      names.push_back(FlatSymbolRefAttr::get(builder.getContext(), name));
    }
    pool.setBuffersAttr(builder.getArrayAttr(names));
  }

  LockOp createLock(ObjectFifoPoolOp pool, StringRef name, int value) {
    setInsertionPointOn(pool.getTile());
    auto lock = LockOp::create(builder, pool.getLoc(), pool.getTile(), value);
    lastPlaced[pool.getTile()] = lock;
    lock->setAttr(SymbolTable::getSymbolAttrName(),
                  builder.getStringAttr(name));
    return lock;
  }

  /// AIE1 guards each buffer with one binary lock that rotates with it; AIE2
  /// gives each segment a counting pair, the producer's lock counting free
  /// objects and the consumer's counting full ones.
  void allocateLocks(ObjectFifoPoolOp pool) {
    if (pool.getDisableSynchronization()) {
      return;
    }

    StringRef base = pool.getBaseName();
    int depth = pool.getDepth();
    auto initValues = pool.getInitValues();
    int filled = initValues ? initValues->size() : 0;
    int repeat = pool.getRepeatCount().value_or(1);

    // A pool that starts full and is re-read each outer iteration is never
    // written again, so nothing needs synchronizing.
    // FIXME: this stands in for "no endpoint fills this pool", which is what
    // actually decides it. Asking that question directly is a follow-on
    // change; see the follow-up list on the PR that introduced these passes.
    auto iterCount = pool.getIterCount();
    if (filled == depth && filled > 0 && iterCount && *iterCount > 1) {
      return;
    }

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

    auto segments = pool.getSegments();
    if (!segments) {
      return;
    }

    SmallVector<Attribute> updated;
    for (auto [index, segment] :
         llvm::enumerate(segments->getAsRange<ObjectFifoSegmentAttr>())) {
      if (segment.getProduceLock() && segment.getConsumeLock()) {
        updated.push_back(segment);
        continue;
      }
      std::string produce =
          (base + "_prod_lock_" + std::to_string(index)).str();
      std::string consume =
          (base + "_cons_lock_" + std::to_string(index)).str();
      createLock(pool, produce, (depth - filled) * repeat);
      createLock(pool, consume, filled * repeat);
      updated.push_back(ObjectFifoSegmentAttr::get(
          builder.getContext(), segment.getOffset(), segment.getSize(),
          FlatSymbolRefAttr::get(builder.getContext(), produce),
          FlatSymbolRefAttr::get(builder.getContext(), consume)));
    }
    pool.setSegmentsAttr(builder.getArrayAttr(updated));
  }

  /// A pool whose buffers spilled onto a neighbor can only be reached by the
  /// channels that see that neighbor's memory.
  bool reachesAdjacentTile(ObjectFifoFlowEndpoint endpoint) {
    auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation());
    if (!dma) {
      return false;
    }
    ObjectFifoPoolOp pool = dma.getPoolOp();
    return pool && llvm::any_of(pool.getBufferOps(), [&](BufferLike buffer) {
             Value tile = buffer.getBufferTile();
             return tile && tile != pool.getTile();
           });
  }

  TileLike tileOf(ObjectFifoFlowEndpoint endpoint) {
    return dyn_cast<TileLike>(endpoint.getTile().getDefiningOp());
  }

  LogicalResult assignChannels(DMAChannelAnalysis &channels) {
    SmallVector<ObjectFifoFlowEndpoint> pending;
    for (auto endpoint : device.getOps<ObjectFifoFlowEndpoint>()) {
      DMAChannelDir dir = endpoint.getFlowDirection();
      std::optional<int> channel = endpoint.getFlowChannel();

      // A core's stream port is named by the design, not drawn from the tile's
      // DMA channels.
      if (endpoint.getFlowBundle() == WireBundle::Core) {
        if (!channel) {
          return endpoint->emitOpError("a stream port names its own channel");
        }
        channels.checkAIEStreamIndex(tileOf(endpoint), {dir, *channel});
        continue;
      }

      if (channel) {
        if (channels.reservePinnedChannel(tileOf(endpoint), dir, *channel) <
            0) {
          return endpoint->emitOpError("pinned ")
                 << stringifyDMAChannelDir(dir) << " DMA channel " << *channel
                 << " is out of range or already in use on this tile";
        }
        continue;
      }
      pending.push_back(endpoint);
    }

    // Endpoints reaching a spilled buffer draw from the restricted low half of
    // the range, so they are served before the unrestricted ones.
    llvm::stable_sort(
        pending, [&](ObjectFifoFlowEndpoint a, ObjectFifoFlowEndpoint b) {
          return reachesAdjacentTile(a) && !reachesAdjacentTile(b);
        });

    for (auto endpoint : pending) {
      DMAChannelDir dir = endpoint.getFlowDirection();
      int channel = channels.getDMAChannelIndex(tileOf(endpoint), dir,
                                                reachesAdjacentTile(endpoint));
      if (channel < 0) {
        return tileOf(endpoint).emitOpError(
            dir == DMAChannelDir::MM2S
                ? "number of output DMA channel exceeded!"
                : "number of input DMA channel exceeded!");
      }
      endpoint.setFlowChannel(channel);
    }
    return success();
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
    for (auto flow : device.getOps<ObjectFifoFlowOp>()) {
      if (auto pinned = flow.getPacketId()) {
        taken.insert(*pinned);
      }
    }
    return taken;
  }

  /// A packet-switched flow shares the stream with others, so every buffer
  /// descriptor the source emits has to carry the packet header.
  LogicalResult lowerPacketFlow(ObjectFifoFlowOp flow,
                                ObjectFifoFlowEndpoint source, int packetID) {

    auto info =
        PacketInfoAttr::get(builder.getContext(), /*pkt_type=*/0, packetID);
    source.setFlowPacket(info);

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
                           source.getFlowBundle(), channelOf(source));
    for (auto destName :
         flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
      auto dest = lookupEndpoint(destName);
      PacketDestOp::create(builder, flow.getLoc(), dest.getTile(),
                           dest.getFlowBundle(), channelOf(dest));
    }
    return success();
  }

  /// Assigned before flows are lowered, so every endpoint has one by now.
  int channelOf(ObjectFifoFlowEndpoint endpoint) {
    std::optional<int> channel = endpoint.getFlowChannel();
    assert(channel && "channels are assigned before flows are lowered");
    return *channel;
  }

  ObjectFifoFlowEndpoint lookupEndpoint(FlatSymbolRefAttr name) {
    return dyn_cast_or_null<ObjectFifoFlowEndpoint>(
        SymbolTable::lookupNearestSymbolFrom(device, name.getAttr()));
  }

  LogicalResult lowerFlows() {
    // A packet header carries five bits of id.
    constexpr int maxPacketID = 31;
    llvm::SmallDenseSet<int> taken = takenPacketIDs();
    int nextFree = 0;
    SmallVector<Operation *> toErase;
    for (auto flow : device.getOps<ObjectFifoFlowOp>()) {
      auto source = lookupEndpoint(flow.getSourceAttr());
      toErase.push_back(flow);

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
                       source.getFlowBundle(), sourceChannel, dest.getTile(),
                       dest.getFlowBundle(), channelOf(dest));
      }
    }
    for (Operation *op : toErase) {
      op->erase();
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

      for (auto endpoint : device.getOps<ObjectFifoFlowEndpoint>()) {
        std::optional<int> channel = endpoint.getFlowChannel();
        if (endpoint.getFifoName() != fifoName ||
            tileOf(endpoint).isShimTile() || !channel) {
          continue;
        }
        channelTiles.push_back(endpoint.getTile());
        channelDirs.push_back(
            static_cast<int32_t>(endpoint.getFlowDirection()));
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
    for (auto endpoint : device.getOps<ObjectFifoFlowEndpoint>()) {
      std::optional<StringRef> fifoName = endpoint.getFifoName();
      std::optional<int> channel = endpoint.getFlowChannel();
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
                                   endpoint.getFlowDirection()),
            builder.getI64IntegerAttr(*channel),
            builder.getBoolAttr(endpoint.getFlowBundle() == WireBundle::PLIO),
            endpoint->getAttrOfType<PacketInfoAttr>("packet"));
      }
      // The runtime sequence reaches the fifo through this record.
      (void)SymbolTable::replaceAllSymbolUses(
          endpoint, builder.getStringAttr(name), device);
    }
  }

  void runOnOperation() override {
    device = getOperation();
    builder = OpBuilder(device.getContext());

    // MemTile pools are served largest-first so the big buffers claim home
    // placement before smaller ones consume the neighbors they would spill to.
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

    for (ObjectFifoPoolOp pool : pools) {
      allocateBuffers(pool);
      allocateLocks(pool);
    }

    DMAChannelAnalysis channels(device);
    if (failed(assignChannels(channels))) {
      return signalPassFailure();
    }

    if (failed(bindRearmTargets())) {
      return signalPassFailure();
    }
    if (failed(lowerFlows())) {
      return signalPassFailure();
    }
    emitShimAllocations();
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
