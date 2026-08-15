//===- AIEObjectFifoSplit.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOSPLIT
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

/// Where the buffers of a fifo end live, and which of a shared pool's segments
/// that end occupies.
struct PoolRef {
  ObjectFifoPoolOp pool;
  std::optional<int32_t> segment;
};

std::optional<ObjectFifoLinkOp> getOptionalLinkOp(ObjectFifoCreateOp op) {
  auto device = op->getParentOfType<DeviceOp>();
  for (ObjectFifoLinkOp linkOp : device.getOps<ObjectFifoLinkOp>()) {
    for (ObjectFifoCreateOp in : linkOp.getInputObjectFifos()) {
      if (in == op) {
        return linkOp;
      }
    }
    for (ObjectFifoCreateOp out : linkOp.getOutputObjectFifos()) {
      if (out == op) {
        return linkOp;
      }
    }
  }
  return {};
}

/// The aie.objectfifo.allocate naming `op`, if any. Reports when several claim
/// the same fifo, since only one delegate tile can hold its objects.
std::optional<ObjectFifoAllocateOp>
getOptionalAllocateOp(ObjectFifoCreateOp op) {
  std::optional<ObjectFifoAllocateOp> found;
  for (auto alloc :
       op->getParentOfType<DeviceOp>().getOps<ObjectFifoAllocateOp>()) {
    if (alloc.getObjFifoName() != op.name().getValue()) {
      continue;
    }
    if (found) {
      op.emitOpError("has more than one allocate operation");
    }
    found = alloc;
  }
  return found;
}

/// Whether `delegate`'s memory module is reachable from both ends of `op`.
bool delegateReachesBothEnds(ObjectFifoCreateOp op, TileOp delegate) {
  auto consumerTileOp = cast<TileOp>(op.getConsumerTiles()[0].getDefiningOp());
  int toProducer = 0, toConsumer = 0;
  isSharedMemory(delegate, op.getProducerTileOp(), &toProducer);
  isSharedMemory(delegate, consumerTileOp, &toConsumer);
  return (toProducer == -1 || toProducer == 2) &&
         (toConsumer == -1 || toConsumer == 2);
}

/// A fifo needs DMAs unless both ends reach one memory module and neither end
/// asks the DMA to reshape the data on the way.
bool requiresDMAs(ObjectFifoCreateOp createOp, int &shareDirection) {
  if (createOp.getVia_DMA() || createOp.getRepeatCount() ||
      createOp.getAieStream() || createOp.getConsumerElemType()) {
    return true;
  }

  bool hasSharedMemory = false;
  if (createOp.getConsumerTiles().size() == 1 &&
      createOp.getDimensionsToStream().empty()) {
    auto consumerTileOp =
        cast<TileOp>(createOp.getConsumerTiles()[0].getDefiningOp());
    hasSharedMemory = isSharedMemory(createOp.getProducerTileOp(),
                                     consumerTileOp, &shareDirection);
  }

  if (!hasSharedMemory) {
    return true;
  }

  for (BDDimLayoutArrayAttr dims :
       createOp.getDimensionsFromStreamPerConsumer()) {
    if (!dims.empty()) {
      return true;
    }
  }

  // A link passes objects between two fifos through a tile's DMAs, unless an
  // aie.objectfifo.allocate names a delegate tile whose memory both ends of
  // this fifo can reach directly.
  if (getOptionalLinkOp(createOp)) {
    if (auto alloc = getOptionalAllocateOp(createOp)) {
      return !delegateReachesBothEnds(createOp, alloc->getDelegateTileOp());
    }
    return true;
  }

  return false;
}

/// Objects a fifo needs on `tile`: enough to cover the largest acquire made
/// there, plus one so a core can hold an object while the next arrives.
int objectCountOn(DeviceOp device, Value tile, ObjectFifoCreateOp objFifo) {
  if (objFifo.size() == 0) {
    return 0;
  }

  auto tileOp = cast<TileOp>(tile.getDefiningOp());
  if (tileOp.isMemTile()) {
    return objFifo.size();
  }

  if (tileOp.isShimTile()) {
    for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>()) {
      if (regOp.getTile() == tile) {
        return regOp.getExternalBuffers().size();
      }
    }
  }

  int maxAcquire = 0;
  for (auto coreOp : device.getOps<CoreOp>()) {
    if (coreOp.getTile() == tile) {
      coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
        if (acqOp.getObjectFifo() == objFifo) {
          maxAcquire = std::max(maxAcquire, acqOp.acqNumber());
        }
      });
    }
  }

  if (maxAcquire == 0) {
    return objFifo.size();
  }
  if (maxAcquire == 1 && objFifo.size() == 1) {
    return 1;
  }
  return maxAcquire + 1;
}

bool hasCoreAccess(DeviceOp device, Value tile, ObjectFifoCreateOp objFifo,
                   ObjectFifoPort port) {
  for (auto coreOp : device.getOps<CoreOp>()) {
    if (coreOp.getTile() != tile) {
      continue;
    }
    bool found = false;
    coreOp.walk([&](Operation *op) {
      if (auto acq = dyn_cast<ObjectFifoAcquireOp>(op)) {
        found |= acq.getObjectFifo() == objFifo && acq.getPort() == port;
      } else if (auto rel = dyn_cast<ObjectFifoReleaseOp>(op)) {
        found |= rel.getObjectFifo() == objFifo && rel.getPort() == port;
      }
    });
    if (found) {
      return true;
    }
  }
  return false;
}

struct AIEObjectFifoSplitPass
    : public xilinx::AIE::impl::AIEObjectFifoSplitBase<AIEObjectFifoSplitPass> {

  DeviceOp device;
  OpBuilder builder{static_cast<MLIRContext *>(nullptr)};

  ObjectFifoPoolOp createPool(Location loc, StringRef name, Value tile,
                              int depth, MemRefType elemType,
                              ObjectFifoCreateOp from,
                              ArrayRef<std::pair<int64_t, int64_t>> extents,
                              bool holdsInitialContents,
                              std::optional<int> repeatCount) {
    SmallVector<Attribute> segments;
    for (auto [offset, size] : extents) {
      segments.push_back(ObjectFifoSegmentAttr::get(
          builder.getContext(), offset, size, nullptr, nullptr));
    }

    std::optional<int32_t> iterCount = from.getIterCount();
    return ObjectFifoPoolOp::create(
        builder, loc, name, tile, depth, elemType, /*buffers=*/ArrayAttr(),
        builder.getArrayAttr(segments), /*locks=*/ArrayAttr(),
        repeatCount ? builder.getI32IntegerAttr(*repeatCount) : IntegerAttr(),
        iterCount ? builder.getI32IntegerAttr(*iterCount) : IntegerAttr(),
        from.getDisableSynchronization(),
        builder.getStringAttr(from.name().getValue()),
        holdsInitialContents ? from.getInitValuesAttr() : ArrayAttr());
  }

  ObjectFifoCoreEndpointOp createCoreEndpoint(Location loc, StringRef name,
                                              Value tile, PoolRef ref,
                                              ObjectFifoRole role) {
    return ObjectFifoCoreEndpointOp::create(
        builder, loc, name, tile, role, ref.pool.getSymName(),
        ref.segment ? builder.getDenseI32ArrayAttr({*ref.segment})
                    : DenseI32ArrayAttr());
  }

  ObjectFifoDmaEndpointOp createDmaEndpoint(
      Location loc, StringRef name, Value tile, std::optional<PoolRef> ref,
      ObjectFifoRole role, ObjectFifoCreateOp from, BDDimLayoutArrayAttr dims,
      std::optional<int> pinnedChannel, std::optional<int> streamPort,
      std::optional<int> acqRelCount, std::optional<int> repeatCount,
      std::optional<int> transferSize) {
    // Only a MemTile's DMA runs its chain a fixed number of times, so the
    // count is recorded only where it is honored.
    std::optional<int32_t> iterCount;
    if (auto tileLike = dyn_cast<TileLike>(tile.getDefiningOp())) {
      if (tileLike.isMemTile()) {
        iterCount = from.getIterCount();
      }
    }
    return ObjectFifoDmaEndpointOp::create(
        builder, loc, builder.getStringAttr(name), tile,
        ref ? ObjectFifoRoleAttr::get(builder.getContext(), role)
            : ObjectFifoRoleAttr(),
        ref ? FlatSymbolRefAttr::get(builder.getContext(),
                                     ref->pool.getSymName())
            : FlatSymbolRefAttr(),
        ref && ref->segment ? builder.getDenseI32ArrayAttr({*ref->segment})
                            : DenseI32ArrayAttr(),
        // A stream port is not drawn from the tile's DMA channels: the
        // index is the port the fifo named, and the direction follows the side
        // this end sits on, so it is settled here rather than in allocation.
        streamPort ? ObjectFifoChannelAttr::get(builder.getContext(),
                                                role == ObjectFifoRole::Drain
                                                    ? DMAChannelDir::MM2S
                                                    : DMAChannelDir::S2MM,
                                                *streamPort)
                   : ObjectFifoChannelAttr(),
        pinnedChannel ? builder.getI32IntegerAttr(*pinnedChannel)
                      : IntegerAttr(),
        dims && !dims.empty() ? dims : BDDimLayoutArrayAttr(),
        from.getPadDimensionsAttr(),
        from.getPadValue() ? builder.getI32IntegerAttr(from.getPadValue())
                           : IntegerAttr(),
        from.getRepeatCount() && repeatCount
            ? builder.getI32IntegerAttr(*repeatCount)
            : IntegerAttr(),
        iterCount ? builder.getI32IntegerAttr(*iterCount) : IntegerAttr(),
        acqRelCount && *acqRelCount > 1
            ? builder.getI32IntegerAttr(*acqRelCount)
            : IntegerAttr(),
        transferSize ? builder.getI32IntegerAttr(*transferSize) : IntegerAttr(),
        /*streamPort=*/
        streamPort ? builder.getI32IntegerAttr(*streamPort) : IntegerAttr(),
        from.getPlio() ? builder.getBoolAttr(true) : BoolAttr(),
        /*packet=*/PacketInfoAttr(),
        builder.getStringAttr(from.name().getValue()));
  }

  /// External buffers registered against a fifo's end on `tile` become the
  /// objects of a pool there. A shim end holds its objects in DDR rather than
  /// on the tile, so the pool names external buffers where the design
  /// registered them and stays empty where the runtime supplies the address at
  /// dispatch; either way the end has a pool, like every other end.
  PoolRef shimPool(ObjectFifoCreateOp fifo, Value tile, StringRef name,
                   MemRefType elemType) {
    SmallVector<Attribute> names;
    for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>()) {
      if (regOp.getTile() != tile || regOp.getObjectFifo() != fifo) {
        continue;
      }
      for (Value buffer : regOp.getExternalBuffers()) {
        auto external = cast<ExternalBufferOp>(buffer.getDefiningOp());
        names.push_back(SymbolRefAttr::get(external.getSymNameAttr()));
        // The registered buffer's own extent is what the DMA moves, which need
        // not be the fifo's object size.
        elemType = cast<MemRefType>(external.getType());
      }
    }

    auto pool = ObjectFifoPoolOp::create(
        builder, fifo.getLoc(), name, tile,
        names.empty() ? fifo.size() : (int)names.size(), elemType,
        names.empty() ? ArrayAttr() : builder.getArrayAttr(names),
        builder.getArrayAttr({ObjectFifoSegmentAttr::get(
            builder.getContext(), 0, elemType.getNumElements(), nullptr,
            nullptr)}),
        /*locks=*/ArrayAttr(), /*repeatCount=*/IntegerAttr(),
        /*iterCount=*/IntegerAttr(), fifo.getDisableSynchronization(),
        builder.getStringAttr(fifo.name().getValue()),
        /*initValues=*/ArrayAttr());
    return PoolRef{pool, std::nullopt};
  }

  /// A link's two ends may disagree on an object's size. On a compute tile the
  /// fifo moves its own; a MemTile moves whichever is larger, so that padding
  /// applied on the way out has room.
  std::optional<int> transferSizeInto(PoolRef ref, MemRefType elemType) {
    if (ref.segment) {
      return std::nullopt;
    }
    int64_t own = elemType.getNumElements();
    int64_t pooled = ref.pool.getObjectSize();
    if (own == pooled) {
      return std::nullopt;
    }
    if (ref.pool.getTileLike().isMemTile() && own < pooled) {
      return std::nullopt;
    }
    return own;
  }

  /// Point a core's accesses at the endpoint it works through.
  void retargetCoreAccesses(ObjectFifoCreateOp fifo, ObjectFifoPort port,
                            Value tile, StringRef endpointName) {
    auto name = FlatSymbolRefAttr::get(builder.getContext(), endpointName);
    // Each access moves to the endpoint for the port it named, so this cannot
    // be a blanket replaceAllSymbolUses: the two ports of one fifo become two
    // different symbols.
    for (auto coreOp : device.getOps<CoreOp>()) {
      if (coreOp.getTile() != tile) {
        continue;
      }
      coreOp.walk([&](Operation *op) {
        if (auto acq = dyn_cast<ObjectFifoAcquireOp>(op)) {
          if (acq.getObjectFifo() == fifo && acq.getPort() == port) {
            acq.setObjFifoNameAttr(name);
            acq.removePortAttr();
          }
        } else if (auto rel = dyn_cast<ObjectFifoReleaseOp>(op)) {
          if (rel.getObjectFifo() == fifo && rel.getPort() == port) {
            rel.setObjFifoNameAttr(name);
            rel.removePortAttr();
          }
        }
      });
    }
  }

  void runOnOperation() override;

  /// Every pool sits on a concrete tile, so the fifo's ends must be placed.
  LogicalResult verifyTilesArePlaced() {
    for (auto fifo : device.getOps<ObjectFifoCreateOp>()) {
      auto placed = [](Value tile) {
        return isa_and_nonnull<TileOp>(tile.getDefiningOp());
      };
      if (!placed(fifo.getProducerTile())) {
        return fifo.emitOpError("producer tile is not a placed aie.tile; run "
                                "--aie-place-tiles before this pass");
      }
      for (Value consumer : fifo.getConsumerTiles()) {
        if (!placed(consumer)) {
          return fifo.emitOpError("consumer tile is not a placed aie.tile; run "
                                  "--aie-place-tiles before this pass");
        }
      }
    }
    return success();
  }

  /// A delegate tile can only hold a fifo's objects if both ends reach its
  /// memory module. A fifo that meets a link elsewhere has no objects of its
  /// own to place, so its delegate is moot.
  LogicalResult verifyAllocateDelegates() {
    for (auto fifo : device.getOps<ObjectFifoCreateOp>()) {
      if (getOptionalLinkOp(fifo) && !linkPoolOwner.contains(fifo)) {
        continue;
      }
      auto alloc = getOptionalAllocateOp(fifo);
      if (alloc && !delegateReachesBothEnds(fifo, alloc->getDelegateTileOp())) {
        return alloc->emitOpError("objectfifo has no shared memory access to "
                                  "delegate tile's memory module");
      }
    }
    return success();
  }

  /// A fifo end wired to a Core stream port holds no objects, so there is
  /// nothing for a core to acquire or release there.
  LogicalResult verifyStreamPortAccesses() {
    auto check = [&](Operation *op, ObjectFifoCreateOp fifo,
                     std::optional<ObjectFifoPort> port, StringRef verb) {
      if (!fifo || !port || !fifo.getAieStream()) {
        return success();
      }
      int streamEnd = *fifo.getAieStream();
      int end = *port == ObjectFifoPort::Produce ? 0 : 1;
      if (streamEnd != 2 && streamEnd != end) {
        return success();
      }
      return LogicalResult(op->emitOpError("cannot ")
                           << verb << " objectfifo stream port");
    };
    for (auto coreOp : device.getOps<CoreOp>()) {
      auto result = coreOp.walk([&](Operation *op) {
        LogicalResult ok = success();
        if (auto acq = dyn_cast<ObjectFifoAcquireOp>(op)) {
          ok = check(op, acq.getObjectFifo(), acq.getPort(), "acquire from");
        } else if (auto rel = dyn_cast<ObjectFifoReleaseOp>(op)) {
          ok = check(op, rel.getObjectFifo(), rel.getPort(), "release from");
        }
        return failed(ok) ? WalkResult::interrupt() : WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        return failure();
      }
    }
    return success();
  }

  /// Ends of a linked fifo that resolve to the link's shared pool.
  DenseMap<Operation *, PoolRef> linkedProducerEnd;
  DenseMap<Operation *, PoolRef> linkedConsumerEnd;
  /// Fifos whose own objects a link's shared pool holds.
  DenseSet<Operation *> linkPoolOwner;
  /// Shim endpoint each fifo is driven through, where it has one.
  DenseMap<Operation *, std::string> shimEndpointName;

  void createLinkPools();
};

/// A link's participants meet on one pool on the shared tile: the fifo carrying
/// whole objects owns its size, and each of the other side's fifos occupies one
/// segment.
void AIEObjectFifoSplitPass::createLinkPools() {
  for (auto linkOp : device.getOps<ObjectFifoLinkOp>()) {
    auto sharedTile = linkOp.getOptionalSharedTile();
    if (!sharedTile) {
      continue;
    }

    std::vector<ObjectFifoCreateOp> ins = linkOp.getInputObjectFifos();
    std::vector<ObjectFifoCreateOp> outs = linkOp.getOutputObjectFifos();
    bool isJoin = linkOp.isJoin();
    bool isDistribute = linkOp.isDistribute();

    ObjectFifoCreateOp owner = isJoin ? outs[0] : ins[0];
    if (!isJoin && !isDistribute) {
      auto inType = cast<MemRefType>(
          cast<AIEObjectFifoType>(ins[0].getElemType()).getElementType());
      auto outType = cast<MemRefType>(
          cast<AIEObjectFifoType>(outs[0].getElemType()).getElementType());
      // Padding is applied as the objects leave, so the pool holds what
      // arrives.
      if (outs[0].getInitValues() ||
          (outType.getNumElements() > inType.getNumElements() &&
           !outs[0].getPadDimensions())) {
        owner = outs[0];
      }
    }

    auto elemType = cast<MemRefType>(
        cast<AIEObjectFifoType>(owner.getElemType()).getElementType());
    builder.setInsertionPoint(owner);

    SmallVector<std::pair<int64_t, int64_t>> extents;
    if (isJoin || isDistribute) {
      ArrayAttr offsets =
          isJoin ? linkOp.getSrcOffsets() : linkOp.getDstOffsets();
      std::vector<ObjectFifoCreateOp> &side = isJoin ? ins : outs;
      for (auto [index, participant] : llvm::enumerate(side)) {
        int64_t offset = *getConstantIntValue(offsets[index]);
        int64_t next = index + 1 < side.size()
                           ? *getConstantIntValue(offsets[index + 1])
                           : elemType.getNumElements();
        int64_t size = next - offset;
        if (size <= 0) {
          size = cast<MemRefType>(
                     cast<AIEObjectFifoType>(participant.getElemType())
                         .getElementType())
                     .getNumElements();
        }
        extents.emplace_back(offset, size);
      }
    } else {
      extents.emplace_back(0, elemType.getNumElements());
    }

    // The pool is one end of the owning fifo, and is named and sized as that
    // end would be, so the owner's other end keeps its own name.
    bool ownerIsOutput = llvm::is_contained(outs, owner);
    std::string name;
    int depth;
    if (ownerIsOutput) {
      name = (owner.name().getValue() + "_pool").str();
      depth = owner.size();
    } else {
      auto consumers = owner.getConsumerTiles();
      int index =
          std::distance(consumers.begin(), llvm::find(consumers, *sharedTile));
      name = (owner.name().getValue() +
              (consumers.size() > 1 ? "_" + std::to_string(index) + "_cons_pool"
                                    : "_cons_pool"))
                 .str();
      depth = isa<ArrayAttr>(owner.getElemNumber())
                  ? owner.size(index + 1)
                  : objectCountOn(device, *sharedTile, owner);
    }

    auto pool = createPool(
        linkOp.getLoc(), name, *sharedTile, depth, elemType, owner, extents,
        /*holdsInitialContents=*/ownerIsOutput, linkOp.getRepeatCount());
    linkPoolOwner.insert(owner);

    for (auto [index, in] : llvm::enumerate(ins)) {
      linkedConsumerEnd[in] =
          PoolRef{pool, isJoin ? std::optional<int32_t>(index) : std::nullopt};
    }
    for (auto [index, out] : llvm::enumerate(outs)) {
      linkedProducerEnd[out] = PoolRef{
          pool, isDistribute ? std::optional<int32_t>(index) : std::nullopt};
    }
  }
}

void AIEObjectFifoSplitPass::runOnOperation() {
  device = getOperation();
  builder = OpBuilder(device.getContext());
  builder.setInsertionPoint(device.getBody()->getTerminator());

  SmallVector<ObjectFifoCreateOp> fifos(device.getOps<ObjectFifoCreateOp>());

  if (failed(verifyTilesArePlaced()) || failed(verifyStreamPortAccesses())) {
    return signalPassFailure();
  }

  createLinkPools();

  if (failed(verifyAllocateDelegates())) {
    return signalPassFailure();
  }

  for (ObjectFifoCreateOp fifo : fifos) {
    builder.setInsertionPoint(fifo);
    Location loc = fifo.getLoc();
    StringRef fifoName = fifo.name().getValue();
    auto fifoType = cast<AIEObjectFifoType>(fifo.getElemType());
    auto elemType = cast<MemRefType>(fifoType.getElementType());
    auto consElemType = cast<MemRefType>(
        cast<AIEObjectFifoType>(fifo.getConsumerElemTypeOrDefault())
            .getElementType());

    int shareDirection = 0;
    bool shared = !requiresDMAs(fifo, shareDirection);

    if (shared) {
      PoolRef ref;
      if (auto linked = linkedProducerEnd.find(fifo);
          linked != linkedProducerEnd.end()) {
        ref = linked->second;
      } else if (auto linked = linkedConsumerEnd.find(fifo);
                 linked != linkedConsumerEnd.end()) {
        ref = linked->second;
      } else {
        Value tile = shareDirection == 1 ? fifo.getConsumerTiles()[0]
                                         : fifo.getProducerTile();
        // An aie.objectfifo.allocate names the tile whose memory is to hold the
        // objects, in place of either end's own.
        if (auto alloc = getOptionalAllocateOp(fifo)) {
          if (!delegateReachesBothEnds(fifo, alloc->getDelegateTileOp())) {
            alloc->emitOpError("objectfifo has no shared memory access to "
                               "delegate tile's memory module");
            return signalPassFailure();
          }
          tile = alloc->getDelegateTile();
        }
        ref = PoolRef{
            createPool(loc, (fifoName + "_pool").str(), tile, fifo.size(),
                       elemType, fifo, {{0, elemType.getNumElements()}},
                       /*holdsInitialContents=*/true, fifo.getRepeatCount()),
            std::nullopt};
      }

      if (hasCoreAccess(device, fifo.getProducerTile(), fifo,
                        ObjectFifoPort::Produce)) {
        auto name = (fifoName + "_prod").str();
        createCoreEndpoint(loc, name, fifo.getProducerTile(), ref,
                           ObjectFifoRole::Fill);
        retargetCoreAccesses(fifo, ObjectFifoPort::Produce,
                             fifo.getProducerTile(), name);
      }
      if (hasCoreAccess(device, fifo.getConsumerTiles()[0], fifo,
                        ObjectFifoPort::Consume)) {
        auto name = (fifoName + "_cons").str();
        createCoreEndpoint(loc, name, fifo.getConsumerTiles()[0], ref,
                           ObjectFifoRole::Drain);
        retargetCoreAccesses(fifo, ObjectFifoPort::Consume,
                             fifo.getConsumerTiles()[0], name);
      }
      continue;
    }

    // Producer end.
    Value prodTile = fifo.getProducerTile();
    // A fifo end wired straight to a Core stream port has no objects of its
    // own: whatever the core writes goes out on the port.
    int streamEnd = fifo.getAieStream().value_or(-1);
    std::optional<int> prodStreamPort;
    std::optional<int> consStreamPort;
    if (streamEnd == 0 || streamEnd == 2) {
      prodStreamPort = fifo.getAieStreamPort();
    }
    if (streamEnd == 1 || streamEnd == 2) {
      consStreamPort = fifo.getAieStreamPort();
    }

    bool prodIsShim = cast<TileOp>(prodTile.getDefiningOp()).isShimTile();
    std::optional<PoolRef> prodRef;
    // Objects entering a repeating link arrive that many at a time.
    std::optional<int> prodAcqRel;
    std::optional<int> consAcqRel;
    std::optional<int> prodTransfer;
    std::optional<int> consTransfer;
    if (auto linked = linkedProducerEnd.find(fifo);
        linked != linkedProducerEnd.end()) {
      prodRef = linked->second;
      prodTransfer = transferSizeInto(linked->second, elemType);
    } else if (prodIsShim) {
      prodRef = shimPool(fifo, prodTile, (fifoName + "_pool").str(), elemType);
    } else if (!prodStreamPort) {
      int depth = fifo.getInitValues() ? fifo.size()
                                       : objectCountOn(device, prodTile, fifo);
      auto pool =
          createPool(loc, (fifoName + "_pool").str(), prodTile, depth, elemType,
                     fifo, {{0, elemType.getNumElements()}},
                     /*holdsInitialContents=*/true, fifo.getRepeatCount());
      prodRef = PoolRef{pool, std::nullopt};
    }

    if (prodRef &&
        hasCoreAccess(device, prodTile, fifo, ObjectFifoPort::Produce)) {
      auto name = (fifoName + "_prod").str();
      createCoreEndpoint(loc, name, prodTile, *prodRef, ObjectFifoRole::Fill);
      retargetCoreAccesses(fifo, ObjectFifoPort::Produce, prodTile, name);
    }

    // A link that repeats gathers that many objects each time the pool is
    // filled; draining it stays one object at a time.
    std::optional<int> linkRepeat;
    if (auto linkOp = getOptionalLinkOp(fifo)) {
      linkRepeat = linkOp->getRepeatCount();
    }
    auto prodDma = createDmaEndpoint(
        loc, (fifoName + "_prod_dma").str(), prodTile, prodRef,
        ObjectFifoRole::Drain, fifo, fifo.getDimensionsToStreamAttr(),
        fifo.getProdDmaChannel(), prodStreamPort, prodAcqRel,
        fifo.getRepeatCount(), prodTransfer);
    if (prodIsShim) {
      shimEndpointName[fifo] = prodDma.getSymName().str();
    }

    // Consumer ends.
    SmallVector<Attribute> destinations;
    ArrayRef<BDDimLayoutArrayAttr> consumerDims =
        fifo.getDimensionsFromStreamPerConsumer();
    int consumerIndex = 0;
    for (Value consumerTile : fifo.getConsumerTiles()) {
      std::string suffix = fifo.getConsumerTiles().size() > 1
                               ? ("_" + std::to_string(consumerIndex) + "_cons")
                               : "_cons";
      bool consIsShim = cast<TileOp>(consumerTile.getDefiningOp()).isShimTile();

      std::optional<PoolRef> consRef;
      if (auto linked = linkedConsumerEnd.find(fifo);
          linked != linkedConsumerEnd.end() &&
          consumerTile == linked->second.pool.getTile()) {
        consRef = linked->second;
        consAcqRel = linkRepeat;
        consTransfer = transferSizeInto(linked->second, consElemType);
      } else if (consIsShim) {
        consRef = shimPool(fifo, consumerTile,
                           (fifoName + suffix + "_pool").str(), consElemType);
      } else if (!consStreamPort) {
        int depth = isa<ArrayAttr>(fifo.getElemNumber())
                        ? fifo.size(consumerIndex + 1)
                        : objectCountOn(device, consumerTile, fifo);
        auto pool = createPool(loc, (fifoName + suffix + "_pool").str(),
                               consumerTile, depth, consElemType, fifo,
                               {{0, consElemType.getNumElements()}},
                               /*holdsInitialContents=*/false,
                               /*repeatCount=*/std::nullopt);
        consRef = PoolRef{pool, std::nullopt};
      }

      if (consRef &&
          hasCoreAccess(device, consumerTile, fifo, ObjectFifoPort::Consume)) {
        auto name = (fifoName + suffix).str();
        createCoreEndpoint(loc, name, consumerTile, *consRef,
                           ObjectFifoRole::Drain);
        retargetCoreAccesses(fifo, ObjectFifoPort::Consume, consumerTile, name);
      }

      std::optional<int> pinned;
      if (auto consChannels = fifo.getConsDmaChannels();
          consChannels && consumerIndex < (int)consChannels->size() &&
          (*consChannels)[consumerIndex] >= 0) {
        pinned = (*consChannels)[consumerIndex];
      }

      auto consDma =
          createDmaEndpoint(loc, (fifoName + suffix + "_dma").str(),
                            consumerTile, consRef, ObjectFifoRole::Fill, fifo,
                            consumerDims.empty() ? BDDimLayoutArrayAttr()
                                                 : consumerDims[consumerIndex],
                            pinned, consStreamPort, consAcqRel,
                            /*repeatCount=*/std::nullopt, consTransfer);
      destinations.push_back(
          FlatSymbolRefAttr::get(builder.getContext(), consDma.getSymName()));
      if (consIsShim) {
        shimEndpointName[fifo] = consDma.getSymName().str();
      }
      consumerIndex++;
    }

    ObjectFifoFlowOp::create(
        builder, loc,
        FlatSymbolRefAttr::get(builder.getContext(), prodDma.getSymName()),
        builder.getArrayAttr(destinations),
        fifo.getPacket() ? builder.getUnitAttr() : UnitAttr(),
        fifo.getPacketIdAttr());
  }

  SmallVector<Operation *> toErase;
  for (ObjectFifoCreateOp fifo : fifos) {
    // The runtime sequence drives a fifo through its shim end.
    auto shim = shimEndpointName.find(fifo);
    if (shim != shimEndpointName.end()) {
      (void)SymbolTable::replaceAllSymbolUses(
          fifo, builder.getStringAttr(shim->second), device);
    }
    toErase.push_back(fifo);
  }
  for (auto linkOp : device.getOps<ObjectFifoLinkOp>()) {
    toErase.push_back(linkOp);
  }
  for (auto allocOp : device.getOps<ObjectFifoAllocateOp>()) {
    toErase.push_back(allocOp);
  }
  for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>()) {
    toErase.push_back(regOp);
  }
  for (Operation *op : toErase) {
    op->erase();
  }
}

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoSplitPass() {
  return std::make_unique<AIEObjectFifoSplitPass>();
}
