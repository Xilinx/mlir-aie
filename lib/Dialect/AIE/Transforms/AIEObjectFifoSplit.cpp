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

/// Which memory module two tiles share:
///   2 if either may be used, -1 for a's, 1 for b's, 0 if none.
bool isSharedMemory(TileOp a, TileOp b, int *shareDirection) {
  const auto &targetModel = getTargetModel(a.getOperation());

  bool aIsShim = a.isShimTile(), bIsShim = b.isShimTile();
  bool aIsMem = targetModel.isMemTile(a.getCol(), a.getRow());
  bool bIsMem = targetModel.isMemTile(b.getCol(), b.getRow());
  if (aIsShim != bIsShim || aIsMem != bIsMem) {
    *shareDirection = 0;
    return false;
  }

  bool rightShared = targetModel.isLegalMemAffinity(a.colIndex(), a.rowIndex(),
                                                    b.colIndex(), b.rowIndex());
  bool leftShared = targetModel.isLegalMemAffinity(b.colIndex(), b.rowIndex(),
                                                   a.colIndex(), a.rowIndex());

  if (leftShared && rightShared)
    *shareDirection = 2;
  else if (leftShared)
    *shareDirection = -1;
  else if (rightShared)
    *shareDirection = 1;
  else
    *shareDirection = 0;

  return leftShared || rightShared;
}

std::optional<ObjectFifoLinkOp> getOptionalLinkOp(ObjectFifoCreateOp op) {
  auto device = op->getParentOfType<DeviceOp>();
  for (ObjectFifoLinkOp linkOp : device.getOps<ObjectFifoLinkOp>()) {
    for (ObjectFifoCreateOp in : linkOp.getInputObjectFifos())
      if (in == op)
        return linkOp;
    for (ObjectFifoCreateOp out : linkOp.getOutputObjectFifos())
      if (out == op)
        return linkOp;
  }
  return {};
}

/// A fifo needs DMAs unless both ends reach one memory module and neither end
/// asks the DMA to reshape the data on the way.
bool requiresDMAs(ObjectFifoCreateOp createOp, int &shareDirection) {
  if (createOp.getVia_DMA() || createOp.getRepeatCount() ||
      createOp.getAieStream() || createOp.getConsumerElemType())
    return true;

  bool hasSharedMemory = false;
  if (createOp.getConsumerTiles().size() == 1 &&
      createOp.getDimensionsToStream().empty()) {
    auto consumerTileOp =
        cast<TileOp>(createOp.getConsumerTiles()[0].getDefiningOp());
    hasSharedMemory = isSharedMemory(createOp.getProducerTileOp(),
                                     consumerTileOp, &shareDirection);
  }

  if (!hasSharedMemory)
    return true;

  for (BDDimLayoutArrayAttr dims :
       createOp.getDimensionsFromStreamPerConsumer())
    if (!dims.empty())
      return true;

  // A link point on a compute tile is reached through its DMAs, so the two ends
  // cannot share one buffer set even when the tiles are adjacent.
  if (auto linkOp = getOptionalLinkOp(createOp))
    if (auto sharedTile = linkOp->getOptionalSharedTile())
      if (!cast<TileOp>(sharedTile->getDefiningOp()).isMemTile())
        return true;

  return false;
}

/// Objects a fifo needs on `tile`: enough to cover the largest acquire made
/// there, plus one so a core can hold an object while the next arrives.
int objectCountOn(DeviceOp device, Value tile, ObjectFifoCreateOp objFifo) {
  if (objFifo.size() == 0)
    return 0;

  auto tileOp = cast<TileOp>(tile.getDefiningOp());
  if (tileOp.isMemTile())
    return objFifo.size();

  if (tileOp.isShimTile())
    for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>())
      if (regOp.getTile() == tile)
        return regOp.getExternalBuffers().size();

  int maxAcquire = 0;
  for (auto coreOp : device.getOps<CoreOp>())
    if (coreOp.getTile() == tile)
      coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
        if (acqOp.getObjectFifo() == objFifo)
          maxAcquire = std::max(maxAcquire, acqOp.acqNumber());
      });

  if (maxAcquire == 0)
    return objFifo.size();
  if (maxAcquire == 1 && objFifo.size() == 1)
    return 1;
  return maxAcquire + 1;
}

bool hasCoreAccess(DeviceOp device, Value tile, ObjectFifoCreateOp objFifo,
                   ObjectFifoPort port) {
  for (auto coreOp : device.getOps<CoreOp>()) {
    if (coreOp.getTile() != tile)
      continue;
    bool found = false;
    coreOp.walk([&](Operation *op) {
      if (auto acq = dyn_cast<ObjectFifoAcquireOp>(op))
        found |= acq.getObjectFifo() == objFifo && acq.getPort() == port;
      else if (auto rel = dyn_cast<ObjectFifoReleaseOp>(op))
        found |= rel.getObjectFifo() == objFifo && rel.getPort() == port;
    });
    if (found)
      return true;
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
                              ArrayRef<std::pair<int64_t, int64_t>> extents) {
    SmallVector<Attribute> segments;
    for (auto [offset, size] : extents)
      segments.push_back(ObjectFifoSegmentAttr::get(
          builder.getContext(), offset, size, nullptr, nullptr));

    return ObjectFifoPoolOp::create(
        builder, loc, name, tile, depth, elemType, /*buffers=*/ArrayAttr(),
        builder.getArrayAttr(segments), /*locks=*/ArrayAttr(),
        from.getDisableSynchronization(), from.getInitValuesAttr());
  }

  ObjectFifoCoreEndpointOp createCoreEndpoint(Location loc, StringRef name,
                                              Value tile, PoolRef ref,
                                              ObjectFifoRole role) {
    return ObjectFifoCoreEndpointOp::create(
        builder, loc, name, tile, role, ref.pool.getSymName(),
        ref.segment ? builder.getDenseI32ArrayAttr({*ref.segment})
                    : DenseI32ArrayAttr());
  }

  ObjectFifoDmaEndpointOp
  createDmaEndpoint(Location loc, StringRef name, Value tile,
                    std::optional<PoolRef> ref, ObjectFifoRole role,
                    ObjectFifoCreateOp from, BDDimLayoutArrayAttr dims) {
    return ObjectFifoDmaEndpointOp::create(
        builder, loc, builder.getStringAttr(name), tile,
        ref ? ObjectFifoRoleAttr::get(builder.getContext(), role)
            : ObjectFifoRoleAttr(),
        ref ? FlatSymbolRefAttr::get(builder.getContext(),
                                     ref->pool.getSymName())
            : FlatSymbolRefAttr(),
        ref && ref->segment ? builder.getDenseI32ArrayAttr({*ref->segment})
                            : DenseI32ArrayAttr(),
        /*channel=*/ObjectFifoChannelAttr(),
        dims && !dims.empty() ? dims : BDDimLayoutArrayAttr(),
        from.getPadDimensionsAttr(),
        from.getPadValue() ? builder.getI32IntegerAttr(from.getPadValue())
                           : IntegerAttr(),
        from.getRepeatCount()
            ? builder.getI32IntegerAttr(*from.getRepeatCount())
            : IntegerAttr(),
        from.getIterCount() ? builder.getI32IntegerAttr(*from.getIterCount())
                            : IntegerAttr(),
        /*streamPort=*/IntegerAttr(),
        from.getPlio() ? builder.getBoolAttr(true) : BoolAttr(),
        /*packet=*/PacketInfoAttr(),
        builder.getStringAttr(from.name().getValue()));
  }

  /// Point a core's accesses at the endpoint it works through.
  void retargetCoreAccesses(ObjectFifoCreateOp fifo, ObjectFifoPort port,
                            Value tile, StringRef endpointName) {
    auto name = FlatSymbolRefAttr::get(builder.getContext(), endpointName);
    for (auto coreOp : device.getOps<CoreOp>()) {
      if (coreOp.getTile() != tile)
        continue;
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

  /// Ends of a linked fifo that resolve to the link's shared pool.
  DenseMap<Operation *, PoolRef> linkedProducerEnd;
  DenseMap<Operation *, PoolRef> linkedConsumerEnd;
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
    if (!sharedTile)
      continue;

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
      if (outs[0].getInitValues() ||
          outType.getNumElements() > inType.getNumElements())
        owner = outs[0];
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
        if (size <= 0)
          size = cast<MemRefType>(
                     cast<AIEObjectFifoType>(participant.getElemType())
                         .getElementType())
                     .getNumElements();
        extents.emplace_back(offset, size);
      }
    } else {
      extents.emplace_back(0, elemType.getNumElements());
    }

    // The pool is one end of the owning fifo, and is named as such so that the
    // owner's other end keeps its own name.
    bool ownerIsOutput = llvm::is_contained(outs, owner);
    std::string name =
        (owner.name().getValue() + (ownerIsOutput ? "_pool" : "_cons_pool"))
            .str();

    auto pool = createPool(linkOp.getLoc(), name, *sharedTile, owner.size(),
                           elemType, owner, extents);

    for (auto [index, in] : llvm::enumerate(ins))
      linkedConsumerEnd[in] =
          PoolRef{pool, isJoin ? std::optional<int32_t>(index) : std::nullopt};
    for (auto [index, out] : llvm::enumerate(outs))
      linkedProducerEnd[out] = PoolRef{
          pool, isDistribute ? std::optional<int32_t>(index) : std::nullopt};
  }
}

void AIEObjectFifoSplitPass::runOnOperation() {
  device = getOperation();
  builder = OpBuilder(device.getContext());
  builder.setInsertionPoint(device.getBody()->getTerminator());

  SmallVector<ObjectFifoCreateOp> fifos(device.getOps<ObjectFifoCreateOp>());

  createLinkPools();

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
      Value tile = shareDirection == 1 ? fifo.getConsumerTiles()[0]
                                       : fifo.getProducerTile();
      auto pool = createPool(loc, (fifoName + "_pool").str(), tile, fifo.size(),
                             elemType, fifo, {{0, elemType.getNumElements()}});
      PoolRef ref{pool, std::nullopt};

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
    bool prodIsShim = cast<TileOp>(prodTile.getDefiningOp()).isShimTile();
    std::optional<PoolRef> prodRef;
    if (auto linked = linkedProducerEnd.find(fifo);
        linked != linkedProducerEnd.end()) {
      prodRef = linked->second;
    } else if (!prodIsShim) {
      int depth = fifo.getInitValues() ? fifo.size()
                                       : objectCountOn(device, prodTile, fifo);
      auto pool = createPool(loc, (fifoName + "_pool").str(), prodTile, depth,
                             elemType, fifo, {{0, elemType.getNumElements()}});
      prodRef = PoolRef{pool, std::nullopt};
    }

    if (prodRef &&
        hasCoreAccess(device, prodTile, fifo, ObjectFifoPort::Produce)) {
      auto name = (fifoName + "_prod").str();
      createCoreEndpoint(loc, name, prodTile, *prodRef, ObjectFifoRole::Fill);
      retargetCoreAccesses(fifo, ObjectFifoPort::Produce, prodTile, name);
    }

    auto prodDma = createDmaEndpoint(loc, (fifoName + "_prod_dma").str(),
                                     prodTile, prodRef, ObjectFifoRole::Drain,
                                     fifo, fifo.getDimensionsToStreamAttr());
    if (prodIsShim)
      shimEndpointName[fifo] = prodDma.getSymName().str();

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
          linked != linkedConsumerEnd.end() && consumerIndex == 0) {
        consRef = linked->second;
      } else if (!consIsShim) {
        int depth = isa<ArrayAttr>(fifo.getElemNumber())
                        ? fifo.size(consumerIndex + 1)
                        : objectCountOn(device, consumerTile, fifo);
        auto pool = createPool(loc, (fifoName + suffix + "_pool").str(),
                               consumerTile, depth, consElemType, fifo,
                               {{0, consElemType.getNumElements()}});
        consRef = PoolRef{pool, std::nullopt};
      }

      if (consRef &&
          hasCoreAccess(device, consumerTile, fifo, ObjectFifoPort::Consume)) {
        auto name = (fifoName + suffix).str();
        createCoreEndpoint(loc, name, consumerTile, *consRef,
                           ObjectFifoRole::Drain);
        retargetCoreAccesses(fifo, ObjectFifoPort::Consume, consumerTile, name);
      }

      auto consDma =
          createDmaEndpoint(loc, (fifoName + suffix + "_dma").str(),
                            consumerTile, consRef, ObjectFifoRole::Fill, fifo,
                            consumerDims.empty() ? BDDimLayoutArrayAttr()
                                                 : consumerDims[consumerIndex]);
      destinations.push_back(
          FlatSymbolRefAttr::get(builder.getContext(), consDma.getSymName()));
      if (consIsShim)
        shimEndpointName[fifo] = consDma.getSymName().str();
      consumerIndex++;
    }

    ObjectFifoFlowOp::create(
        builder, loc,
        FlatSymbolRefAttr::get(builder.getContext(), prodDma.getSymName()),
        builder.getArrayAttr(destinations));
  }

  SmallVector<Operation *> toErase;
  for (ObjectFifoCreateOp fifo : fifos) {
    // The runtime sequence drives a fifo through its shim end.
    auto shim = shimEndpointName.find(fifo);
    if (shim != shimEndpointName.end())
      (void)SymbolTable::replaceAllSymbolUses(
          fifo, builder.getStringAttr(shim->second), device);
    toErase.push_back(fifo);
  }
  for (auto linkOp : device.getOps<ObjectFifoLinkOp>())
    toErase.push_back(linkOp);
  for (auto allocOp : device.getOps<ObjectFifoAllocateOp>())
    toErase.push_back(allocOp);
  for (Operation *op : toErase)
    op->erase();
}

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoSplitPass() {
  return std::make_unique<AIEObjectFifoSplitPass>();
}
