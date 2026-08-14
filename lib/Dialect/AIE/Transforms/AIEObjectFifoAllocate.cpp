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

int64_t objectSizeInBytes(ObjectFifoPoolOp pool) {
  auto elemType = pool.getElemType();
  DataLayout layout = DataLayout::closest(pool);
  return elemType.getNumElements() *
         layout.getTypeSizeInBits(elemType.getElementType()) / 8;
}

struct AIEObjectFifoAllocatePass
    : public xilinx::AIE::impl::AIEObjectFifoAllocateBase<
          AIEObjectFifoAllocatePass> {
  using Base::Base;

  DeviceOp device;
  OpBuilder builder{static_cast<MLIRContext *>(nullptr)};

  /// Bytes already committed to `tile` by every buffer placed so far.
  int64_t usedMemory(Value tile) {
    int64_t total = 0;
    for (auto buffer : device.getOps<BufferOp>())
      if (buffer.getTile() == tile)
        total += buffer.getAllocationSize();
    return total;
  }

  TileOp findOrCreateTile(TileLike host, int col, int row) {
    for (auto tile : device.getOps<TileOp>())
      if (tile.getCol() == col && tile.getRow() == row)
        return tile;

    OpBuilder::InsertionGuard g(builder);
    Operation *insertAfter = host.getOperation();
    while (isa_and_nonnull<BufferOp>(insertAfter->getNextNode()))
      insertAfter = insertAfter->getNextNode();
    builder.setInsertionPointAfter(insertAfter);
    return TileOp::create(builder, host.getLoc(), col, row);
  }

  /// A MemTile buffer that does not fit at home spills to whichever neighbour
  /// its DMAs can still reach, preferring the emptier one so adjacent MemTiles
  /// keep room for their own spills. Which tiles neighbour an unplaced one is
  /// not yet known, so its buffers stay at home.
  Value placementFor(TileLike home, int64_t sizeBytes) {
    auto &target = device.getTargetModel();
    Value homeTile = home->getResult(0);
    if (!home.isMemTile() ||
        usedMemory(homeTile) + sizeBytes <= target.getMemTileSize())
      return homeTile;

    auto homeOp = dyn_cast<TileOp>(home.getOperation());
    if (!homeOp)
      return homeTile;

    SmallVector<TileOp> neighbours;
    for (int col : {homeOp.getCol() - 1, homeOp.getCol() + 1}) {
      if (col < 0 || col >= target.columns())
        continue;
      TileOp neighbour = findOrCreateTile(home, col, homeOp.getRow());
      int direction = 0;
      if (isSharedMemory(homeOp, neighbour, &direction) &&
          (direction == 1 || direction == 2))
        neighbours.push_back(neighbour);
    }
    llvm::stable_sort(neighbours, [&](TileOp a, TileOp b) {
      return usedMemory(a.getResult()) < usedMemory(b.getResult());
    });
    for (TileOp neighbour : neighbours)
      if (usedMemory(neighbour.getResult()) + sizeBytes <=
          target.getMemTileSize())
        return neighbour.getResult();
    return homeTile;
  }

  /// Buffers and locks live directly below the tile declarations.
  void setInsertionPointBelowTiles() {
    Operation *lastTile = nullptr;
    for (Operation &op : *device.getBody())
      if (isa<TileLike>(op))
        lastTile = &op;
    if (lastTile)
      builder.setInsertionPointAfter(lastTile);
    else
      builder.setInsertionPointToStart(device.getBody());
  }

  void allocateBuffers(ObjectFifoPoolOp pool) {
    if (pool.getBuffers())
      return;
    TileLike home = pool.getTileLike();
    if (home.isShimTile())
      return;

    setInsertionPointBelowTiles();
    auto initValues = pool.getInitValues();
    int64_t sizeBytes = objectSizeInBytes(pool);
    StringRef base = pool.getBaseName();

    SmallVector<Attribute> names;
    for (int i = 0; i < pool.getDepth(); i++) {
      ElementsAttr init =
          initValues ? cast<ElementsAttr>((*initValues)[i]) : nullptr;
      std::string name = (base + "_buff_" + std::to_string(i)).str();
      BufferOp::create(builder, pool.getLoc(), pool.getElemType(),
                       placementFor(home, sizeBytes),
                       builder.getStringAttr(name), /*address=*/nullptr, init,
                       /*mem_bank=*/nullptr, /*aligned=*/nullptr);
      names.push_back(FlatSymbolRefAttr::get(builder.getContext(), name));
    }
    pool.setBuffersAttr(builder.getArrayAttr(names));
  }

  LockOp createLock(ObjectFifoPoolOp pool, StringRef name, int value) {
    auto lock = LockOp::create(builder, pool.getLoc(), pool.getTile(), value);
    lock->setAttr(SymbolTable::getSymbolAttrName(),
                  builder.getStringAttr(name));
    return lock;
  }

  /// AIE1 guards each buffer with one binary lock that rotates with it; AIE2
  /// gives each segment a counting pair, the producer's lock counting free
  /// objects and the consumer's counting full ones.
  void allocateLocks(ObjectFifoPoolOp pool) {
    if (pool.getDisableSynchronization())
      return;

    StringRef base = pool.getBaseName();
    int depth = pool.getDepth();
    auto initValues = pool.getInitValues();
    int filled = initValues ? initValues->size() : 0;
    int repeat = pool.getRepeatCount().value_or(1);

    // A pool that starts full and is re-read each outer iteration is never
    // written again, so nothing needs synchronising.
    auto iterCount = pool.getIterCount();
    if (filled == depth && filled > 0 && iterCount && *iterCount > 1)
      return;

    if (device.getTargetModel().getTargetArch() == AIEArch::AIE1) {
      if (pool.getLocks())
        return;
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
    if (!segments)
      return;

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

  /// Which way an endpoint moves data. A shim endpoint has no pool and so no
  /// role, but its side of the flow says the same thing: a source sends memory
  /// out onto the stream, a destination writes what the stream delivers.
  DenseMap<StringRef, DMAChannelDir> directions;

  void collectDirections() {
    for (auto flow : device.getOps<ObjectFifoFlowOp>()) {
      directions[flow.getSource()] = DMAChannelDir::MM2S;
      for (auto dest : flow.getDestinations().getAsRange<FlatSymbolRefAttr>())
        directions[dest.getValue()] = DMAChannelDir::S2MM;
    }
  }

  DMAChannelDir directionOf(ObjectFifoDmaEndpointOp endpoint) {
    auto found = directions.find(endpoint.getSymName());
    if (found != directions.end())
      return found->second;
    return endpoint.drains() ? DMAChannelDir::MM2S : DMAChannelDir::S2MM;
  }

  /// A pool whose buffers spilled onto a neighbour can only be reached by the
  /// channels that see that neighbour's memory.
  bool reachesAdjacentTile(ObjectFifoDmaEndpointOp endpoint) {
    ObjectFifoPoolOp pool = endpoint.getPoolOp();
    if (!pool)
      return false;
    return llvm::any_of(pool.getObjects(), [&](Value object) {
      auto buffer = dyn_cast<BufferOp>(object.getDefiningOp());
      return buffer && buffer.getTile() != pool.getTile();
    });
  }

  LogicalResult assignChannels(DMAChannelAnalysis &channels) {
    SmallVector<ObjectFifoDmaEndpointOp> pending;
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      if (endpoint.getChannel())
        continue;
      DMAChannelDir dir = directionOf(endpoint);
      // A Core stream port is named by the user, not drawn from the tile's
      // DMA channels.
      if (auto port = endpoint.getStreamPort()) {
        channels.checkAIEStreamIndex(endpoint.getTileLike(), {dir, *port});
        endpoint.setChannelAttr(
            ObjectFifoChannelAttr::get(builder.getContext(), dir, *port));
        continue;
      }
      if (auto pinned = endpoint.getPinnedChannel()) {
        int channel =
            channels.reservePinnedChannel(endpoint.getTileLike(), dir, *pinned);
        if (channel < 0)
          return endpoint.emitOpError("pinned ")
                 << stringifyDMAChannelDir(dir) << " DMA channel " << *pinned
                 << " is out of range or already in use on this tile";
        endpoint.setChannelAttr(
            ObjectFifoChannelAttr::get(builder.getContext(), dir, channel));
        continue;
      }
      pending.push_back(endpoint);
    }

    // Endpoints reaching a spilled buffer draw from the restricted low half of
    // the range, so they are served before the unrestricted ones.
    llvm::stable_sort(
        pending, [&](ObjectFifoDmaEndpointOp a, ObjectFifoDmaEndpointOp b) {
          return reachesAdjacentTile(a) && !reachesAdjacentTile(b);
        });

    for (auto endpoint : pending) {
      DMAChannelDir dir = directionOf(endpoint);
      int channel = channels.getDMAChannelIndex(endpoint.getTileLike(), dir,
                                                reachesAdjacentTile(endpoint));
      if (channel < 0)
        return endpoint.getTileLike().emitOpError(
            dir == DMAChannelDir::MM2S
                ? "number of output DMA channel exceeded!"
                : "number of input DMA channel exceeded!");
      endpoint.setChannelAttr(
          ObjectFifoChannelAttr::get(builder.getContext(), dir, channel));
    }
    return success();
  }

  /// Which wire an endpoint's channel sits on: a raw stream port on the core,
  /// PLIO at a shim boundary, or an ordinary DMA channel.
  WireBundle wireFor(ObjectFifoDmaEndpointOp endpoint) {
    if (endpoint.getStreamPort())
      return WireBundle::Core;
    return endpoint.getPlio() && endpoint.getTileLike().isShimTile()
               ? WireBundle::PLIO
               : WireBundle::DMA;
  }

  /// Packet IDs already spoken for elsewhere in the device.
  int nextPacketID() {
    int next = 0;
    device.walk(
        [&](PacketFlowOp flow) { next = std::max<int>(next, flow.IDInt() + 1); });
    return next;
  }

  /// A packet-switched flow shares the stream with others, so every buffer
  /// descriptor the source emits has to carry the packet header.
  LogicalResult lowerPacketFlow(ObjectFifoFlowOp flow,
                                ObjectFifoDmaEndpointOp source, int packetID) {
    if (packetID > 31)
      return device.emitOpError("max number of packet IDs reached");


    auto info =
        PacketInfoAttr::get(builder.getContext(), /*pkt_type=*/0, packetID);
    source.setPacketAttr(info);

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
                           WireBundle::DMA, channelOf(source).getIndex());
    for (auto destName :
         flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
      auto dest = lookupEndpoint(destName);
      PacketDestOp::create(builder, flow.getLoc(), dest.getTile(),
                           WireBundle::DMA, channelOf(dest).getIndex());
    }
    return success();
  }

  /// Assigned before flows are lowered, so every endpoint has one by now.
  ObjectFifoChannelAttr channelOf(ObjectFifoDmaEndpointOp endpoint) {
    std::optional<ObjectFifoChannelAttr> channel = endpoint.getChannel();
    assert(channel && "channels are assigned before flows are lowered");
    return *channel;
  }

  ObjectFifoDmaEndpointOp lookupEndpoint(FlatSymbolRefAttr name) {
    return SymbolTable::lookupNearestSymbolFrom<ObjectFifoDmaEndpointOp>(device,
                                                                         name);
  }

  LogicalResult lowerFlows() {
    int packetID = nextPacketID();
    SmallVector<Operation *> toErase;
    for (auto flow : device.getOps<ObjectFifoFlowOp>()) {
      auto source = lookupEndpoint(flow.getSourceAttr());
      toErase.push_back(flow);

      if (clPacketSwObjectFifos) {
        if (failed(lowerPacketFlow(flow, source, packetID++)))
          return failure();
        continue;
      }

      auto sourceChannel = channelOf(source);
      builder.setInsertionPoint(flow);
      for (auto destName :
           flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
        auto dest = lookupEndpoint(destName);
        auto destChannel = channelOf(dest);
        FlowOp::create(builder, flow.getLoc(), source.getTile(),
                       wireFor(source), sourceChannel.getIndex(),
                       dest.getTile(), wireFor(dest), destChannel.getIndex());
      }
    }
    for (Operation *op : toErase)
      op->erase();
    return success();
  }

  /// An `aiex.dma_channel_reset_for` outlives the fifo it names, so record the
  /// channels and locks it has to re-arm and point it at that record. Shim
  /// endpoints are left out: the host re-pushes those itself.
  LogicalResult bindRearmTargets() {
    llvm::StringMap<SmallVector<Operation *>> usersByFifo;
    device.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "aiex.dma_channel_reset_for")
        return;
      auto sym = op->getAttrOfType<FlatSymbolRefAttr>("objfifo");
      if (!sym)
        return;
      // Split may already have pointed this at the fifo's shim endpoint.
      StringRef name = sym.getValue();
      if (auto endpoint = lookupEndpoint(sym))
        if (auto fifoName = endpoint.getFifoName())
          name = *fifoName;
      usersByFifo[name].push_back(op);
    });
    if (usersByFifo.empty())
      return success();

    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (auto &[fifoName, users] : usersByFifo) {
      SmallVector<Value> channelTiles, lockValues;
      SmallVector<int32_t> channelDirs, channelIndices, lockInits;

      for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
        std::optional<ObjectFifoChannelAttr> channel = endpoint.getChannel();
        if (endpoint.getFifoName() != fifoName ||
            endpoint.getTileLike().isShimTile() || !channel)
          continue;
        channelTiles.push_back(endpoint.getTile());
        channelDirs.push_back(static_cast<int32_t>(channel->getDirection()));
        channelIndices.push_back(channel->getIndex());
      }
      for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
        if (pool.getFifoName() != fifoName || pool.getTileLike().isShimTile())
          continue;
        for (LockOp lock : pool.getLockOps()) {
          lockValues.push_back(lock.getResult());
          lockInits.push_back(lock.getInit().value_or(0));
        }
      }

      if (channelTiles.empty() && lockValues.empty()) {
        for (Operation *user : users)
          user->emitOpError() << "objectFIFO '" << fifoName
                              << "' has no resident core/mem DMA channels or "
                                 "locks to re-arm";
        return failure();
      }

      std::string name = (fifoName + "_rearm").str();
      for (unsigned suffix = 0; device.lookupSymbol(name); suffix++)
        name = (fifoName + "_rearm_" + std::to_string(suffix)).str();

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
      for (Operation *user : users)
        user->setAttr("objfifo", target);
    }
    return success();
  }

  /// A shim endpoint has no memory of its own, so the runtime needs its channel
  /// spelled out under the name the sequence refers to.
  void emitShimAllocations() {
    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      std::optional<StringRef> fifoName = endpoint.getFifoName();
      std::optional<ObjectFifoChannelAttr> channel = endpoint.getChannel();
      if (!endpoint.getTileLike().isShimTile() || !fifoName || !channel)
        continue;
      std::string name = (*fifoName + "_shim_alloc").str();
      if (!SymbolTable::lookupNearestSymbolFrom<ShimDMAAllocationOp>(
              device, builder.getStringAttr(name)))
        ShimDMAAllocationOp::create(
            builder, endpoint.getLoc(), builder.getStringAttr(name),
            endpoint.getTile(),
            DMAChannelDirAttr::get(builder.getContext(),
                                   channel->getDirection()),
            builder.getI64IntegerAttr(channel->getIndex()),
            builder.getBoolAttr(endpoint.getPlio()), endpoint.getPacketAttr());
      // The runtime sequence reaches the fifo through this record.
      (void)SymbolTable::replaceAllSymbolUses(
          endpoint, builder.getStringAttr(name), device);
    }
  }

  void runOnOperation() override {
    device = getOperation();
    builder = OpBuilder(device.getContext());

    // MemTile pools are served largest-first so the big buffers claim home
    // placement before smaller ones consume the neighbours they would spill to.
    SmallVector<ObjectFifoPoolOp> pools(device.getOps<ObjectFifoPoolOp>());
    SmallVector<size_t> memTileSlots;
    SmallVector<ObjectFifoPoolOp> memTilePools;
    for (auto [index, pool] : llvm::enumerate(pools))
      if (pool.getTileLike().isMemTile()) {
        memTileSlots.push_back(index);
        memTilePools.push_back(pool);
      }
    llvm::stable_sort(memTilePools, [](ObjectFifoPoolOp a, ObjectFifoPoolOp b) {
      return objectSizeInBytes(a) > objectSizeInBytes(b);
    });
    for (auto [slot, pool] : llvm::zip(memTileSlots, memTilePools))
      pools[slot] = pool;

    for (ObjectFifoPoolOp pool : pools) {
      setInsertionPointBelowTiles();
      allocateBuffers(pool);
      allocateLocks(pool);
    }

    DMAChannelAnalysis channels(device);
    collectDirections();
    if (failed(assignChannels(channels)))
      return signalPassFailure();

    if (failed(bindRearmTargets()))
      return signalPassFailure();
    if (failed(lowerFlows()))
      return signalPassFailure();
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
