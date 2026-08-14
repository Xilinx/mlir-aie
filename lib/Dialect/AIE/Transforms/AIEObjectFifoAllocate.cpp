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

/// A pool's symbol carries the fifo it came from plus an end marker; buffers
/// and locks are named after the fifo alone.
StringRef poolBaseName(ObjectFifoPoolOp pool) {
  StringRef name = pool.getSymName();
  return name.consume_back("_pool") ? name : pool.getSymName();
}

int64_t objectSizeInBytes(ObjectFifoPoolOp pool) {
  auto elemType = pool.getElemType();
  DataLayout layout = DataLayout::closest(pool);
  return elemType.getNumElements() *
         layout.getTypeSizeInBits(elemType.getElementType()) / 8;
}

struct AIEObjectFifoAllocatePass
    : public xilinx::AIE::impl::AIEObjectFifoAllocateBase<
          AIEObjectFifoAllocatePass> {

  DeviceOp device;
  OpBuilder builder{static_cast<MLIRContext *>(nullptr)};

  /// Bytes already committed to `tile` by every buffer placed so far.
  int64_t usedMemory(TileOp tile) {
    int64_t total = 0;
    for (auto buffer : device.getOps<BufferOp>())
      if (buffer.getTile() == tile.getResult())
        total += buffer.getAllocationSize();
    return total;
  }

  TileOp findOrCreateTile(TileOp host, int col, int row) {
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
  /// keep room for their own spills.
  TileOp placementFor(TileOp home, int64_t sizeBytes) {
    auto &target = device.getTargetModel();
    if (!home.isMemTile() ||
        usedMemory(home) + sizeBytes <= target.getMemTileSize())
      return home;

    SmallVector<TileOp> neighbours;
    for (int col : {home.getCol() - 1, home.getCol() + 1}) {
      if (col < 0 || col >= target.columns())
        continue;
      TileOp neighbour = findOrCreateTile(home, col, home.getRow());
      int direction = 0;
      if (isSharedMemory(home, neighbour, &direction) &&
          (direction == 1 || direction == 2))
        neighbours.push_back(neighbour);
    }
    llvm::stable_sort(neighbours, [&](TileOp a, TileOp b) {
      return usedMemory(a) < usedMemory(b);
    });
    for (TileOp neighbour : neighbours)
      if (usedMemory(neighbour) + sizeBytes <= target.getMemTileSize())
        return neighbour;
    return home;
  }

  /// Buffers and locks live directly below the tile declarations.
  void setInsertionPointBelowTiles() {
    Operation *lastTile = nullptr;
    for (auto tile : device.getBody()->getOps<TileOp>())
      lastTile = tile.getOperation();
    if (lastTile)
      builder.setInsertionPointAfter(lastTile);
    else
      builder.setInsertionPointToStart(device.getBody());
  }

  void allocateBuffers(ObjectFifoPoolOp pool) {
    if (pool.getBuffers())
      return;
    TileOp home = pool.getTileOp();
    if (home.isShimTile())
      return;

    setInsertionPointBelowTiles();
    auto initValues = pool.getInitValues();
    int64_t sizeBytes = objectSizeInBytes(pool);
    StringRef base = poolBaseName(pool);

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
    auto lock = LockOp::create(builder, pool.getLoc(), pool.getTileOp(), value);
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
    TileOp home = pool.getTileOp();
    if (home.isShimTile())
      return;

    StringRef base = poolBaseName(pool);
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
    if (!pool || !pool.getBuffers())
      return false;
    for (auto name : pool.getBuffers()->getAsRange<FlatSymbolRefAttr>()) {
      auto buffer =
          SymbolTable::lookupNearestSymbolFrom<BufferOp>(device, name);
      if (buffer && buffer.getTile() != pool.getTile())
        return true;
    }
    return false;
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
        channels.checkAIEStreamIndex(endpoint.getTileOp(), {dir, *port});
        endpoint.setChannelAttr(
            ObjectFifoChannelAttr::get(builder.getContext(), dir, *port));
        continue;
      }
      if (auto pinned = endpoint.getPinnedChannel()) {
        int channel =
            channels.reservePinnedChannel(endpoint.getTileOp(), dir, *pinned);
        if (channel < 0)
          return endpoint.emitOpError("pinned DMA channel ")
                 << *pinned << " is out of range or already in use";
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
      int channel = channels.getDMAChannelIndex(endpoint.getTileOp(), dir,
                                                reachesAdjacentTile(endpoint));
      if (channel < 0)
        return endpoint.getTileOp().emitOpError(
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
    return endpoint.getPlio() && endpoint.getTileOp().isShimTile()
               ? WireBundle::PLIO
               : WireBundle::DMA;
  }

  void lowerFlows() {
    SmallVector<Operation *> toErase;
    for (auto flow : device.getOps<ObjectFifoFlowOp>()) {
      auto source =
          SymbolTable::lookupNearestSymbolFrom<ObjectFifoDmaEndpointOp>(
              device, flow.getSourceAttr());
      auto sourceChannel = *source.getChannel();
      builder.setInsertionPoint(flow);
      for (auto destName :
           flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
        auto dest =
            SymbolTable::lookupNearestSymbolFrom<ObjectFifoDmaEndpointOp>(
                device, destName);
        auto destChannel = *dest.getChannel();
        FlowOp::create(builder, flow.getLoc(), source.getTile(),
                       wireFor(source), sourceChannel.getIndex(),
                       dest.getTile(), wireFor(dest), destChannel.getIndex());
      }
      toErase.push_back(flow);
    }
    for (Operation *op : toErase)
      op->erase();
  }

  /// A shim endpoint has no memory of its own, so the runtime needs its channel
  /// spelled out under the name the sequence refers to.
  void emitShimAllocations() {
    builder.setInsertionPoint(device.getBody()->getTerminator());
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      if (!endpoint.getTileOp().isShimTile() || !endpoint.getFifoName())
        continue;
      auto channel = endpoint.getChannel();
      if (!channel)
        continue;
      std::string name = (*endpoint.getFifoName() + "_shim_alloc").str();
      if (SymbolTable::lookupNearestSymbolFrom<ShimDMAAllocationOp>(
              device, builder.getStringAttr(name)))
        continue;
      ShimDMAAllocationOp::create(
          builder, endpoint.getLoc(), builder.getStringAttr(name),
          endpoint.getTile(),
          DMAChannelDirAttr::get(builder.getContext(), channel->getDirection()),
          builder.getI64IntegerAttr(channel->getIndex()),
          builder.getBoolAttr(endpoint.getPlio()), endpoint.getPacketAttr());
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
      if (pool.getTileOp().isMemTile()) {
        memTileSlots.push_back(index);
        memTilePools.push_back(pool);
      }
    llvm::stable_sort(memTilePools, [](ObjectFifoPoolOp a, ObjectFifoPoolOp b) {
      return objectSizeInBytes(a) > objectSizeInBytes(b);
    });
    for (auto [slot, pool] : llvm::zip(memTileSlots, memTilePools))
      pools[slot] = pool;

    for (ObjectFifoPoolOp pool : pools) {
      allocateBuffers(pool);
      allocateLocks(pool);
    }

    DMAChannelAnalysis channels(device);
    collectDirections();
    if (failed(assignChannels(channels)))
      return signalPassFailure();

    lowerFlows();
    emitShimAllocations();
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoAllocatePass() {
  return std::make_unique<AIEObjectFifoAllocatePass>();
}
