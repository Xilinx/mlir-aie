//===- AIEObjectFifoLowerDMAs.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOLOWERDMAS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

/// One buffer descriptor's worth of work: which object, which slice of it, and
/// the locks guarding that slice.
struct Descriptor {
  Value buffer;
  int64_t offset;
  int64_t size;
  BDDimLayoutArrayAttr dimensions;
  BDPadLayoutArrayAttr padding;
  FlatSymbolRefAttr acquireLock;
  FlatSymbolRefAttr releaseLock;
  /// Index of the buffer's own binary lock, where the device uses those.
  int64_t binaryLock;
  /// Lock units taken and given back, where they are not one object's.
  std::optional<int64_t> acquireValue = std::nullopt;
  std::optional<int64_t> releaseValue = std::nullopt;
  /// How the address steps on each run of a descriptor that walks a
  /// contiguous pool's objects in turn.
  BDIterationAttr iteration = nullptr;
};

int64_t paddedSize(BDDimLayoutArrayAttr dimensions,
                   BDPadLayoutArrayAttr padding, int64_t fallback) {
  if (!dimensions || dimensions.empty() || !padding || padding.empty()) {
    return fallback;
  }
  int64_t size = 1;
  for (auto [dimension, pad] : llvm::zip(dimensions, padding)) {
    size *=
        dimension.getSize() + pad.getConstPadBefore() + pad.getConstPadAfter();
  }
  return size;
}

Block *findEndOpBlock(Region &region) {
  Block *endBlock = nullptr;
  for (Block &block : region) {
    if (!block.getOps<EndOp>().empty()) {
      endBlock = &block;
    }
  }
  return endBlock;
}

struct AIEObjectFifoLowerDMAsPass
    : public xilinx::AIE::impl::AIEObjectFifoLowerDMAsBase<
          AIEObjectFifoLowerDMAsPass> {

  DeviceOp device;
  OpBuilder builder{static_cast<MLIRContext *>(nullptr)};

  /// The DMA program of `tile`, created empty if the tile has none yet.
  DmaBody dmaProgramFor(TileLike tile, Location loc) {
    Value wanted = tile->getResult(0);
    for (auto program : device.getOps<DmaBody>()) {
      if (program.getTile() == wanted) {
        return program;
      }
    }

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(device.getBody()->getTerminator());
    DmaBody program;
    if (tile.isShimTile()) {
      program = ShimDMAOp::create(builder, loc, builder.getIndexType(), wanted);
    } else if (tile.isMemTile()) {
      program = MemTileDMAOp::create(builder, loc, wanted);
    } else {
      program = MemOp::create(builder, loc, wanted);
    }

    builder.setInsertionPointToStart(&program.getDmaBody().emplaceBlock());
    EndOp::create(builder, loc);
    return program;
  }

  /// Objects the endpoint moves.
  SmallVector<Value> buffersOf(ObjectFifoPoolOp pool) {
    SmallVector<Value> buffers;
    for (BufferLike buffer : pool.getBufferOps()) {
      buffers.push_back(buffer.getBuffer());
    }
    return buffers;
  }

  /// A contiguous pool's one buffer holds its objects side by side. Its
  /// filling end steps through them with one descriptor; its draining end
  /// walks the stream with one descriptor per output, each taking the objects
  /// up to the newest it reads and giving back those no later one reads.
  SmallVector<Descriptor>
  contiguousDescriptors(ObjectFifoDmaEndpointOp endpoint, Value buffer,
                        bool drains) {
    ObjectFifoPoolOp pool = endpoint.getPoolOp();
    ObjectFifoSegmentOp segment = endpoint.getSelectedSegments().front();
    FlatSymbolRefAttr take =
        drains ? segment.getConsumeLockAttr() : segment.getProduceLockAttr();
    FlatSymbolRefAttr give =
        drains ? segment.getProduceLockAttr() : segment.getConsumeLockAttr();
    MLIRContext *context = builder.getContext();
    if (!endpoint.walksStream()) {
      BDDimLayoutArrayAttr dims;
      if (auto all = endpoint.getDimensions(); all && !all->empty())
        dims = (*all)[0];
      int64_t object = pool.getObjectSize();
      int64_t first = 0;
      if (auto zeros = pool.getZeroObjects())
        first = (*zeros)[0] * object;
      return {{buffer, first + segment.getOffset(), segment.getSize(), dims,
               BDPadLayoutArrayAttr(), take, give, 0, std::nullopt,
               std::nullopt,
               BDIterationAttr::get(context, pool.getDepth(), object, 0)}};
    }

    StreamWindow window = *endpoint.getStreamWindow();
    BDDimLayoutArrayAttr dims =
        BDDimLayoutArrayAttr::get(context, window.inner);
    SmallVector<Descriptor> descriptors;
    for (StreamWindowStep step : streamWindowSteps(window))
      descriptors.push_back({buffer, step.first * window.objectSize,
                             window.transfer(), dims, BDPadLayoutArrayAttr(),
                             take, give, 0, step.acquire, step.release,
                             nullptr});
    return descriptors;
  }

  /// Buffer-major, segment-minor: the DMA walks the objects in turn, and within
  /// each object the slices this endpoint is responsible for.
  SmallVector<Descriptor> descriptorsFor(ObjectFifoDmaEndpointOp endpoint,
                                         ArrayRef<Value> buffers, bool drains) {
    if (endpoint.getPoolOp().getContiguous())
      return contiguousDescriptors(endpoint, buffers.front(), drains);
    std::vector<ObjectFifoSegmentOp> segments = endpoint.getSelectedSegments();
    SmallVector<BDDimLayoutArrayAttr> dimensions;
    if (auto allDimensions = endpoint.getDimensions()) {
      llvm::append_range(dimensions, *allDimensions);
    }
    SmallVector<BDPadLayoutArrayAttr> padding;
    if (auto allPadding = endpoint.getPadDimensions()) {
      llvm::append_range(padding, *allPadding);
    }

    SmallVector<Descriptor> descriptors;
    for (auto [index, buffer] : llvm::enumerate(buffers)) {
      for (auto [position, segment] : llvm::enumerate(segments)) {
        descriptors.push_back(
            {buffer, segment.getOffset(), segment.getSize(),
             position < dimensions.size() ? dimensions[position]
                                          : BDDimLayoutArrayAttr(),
             position < padding.size() ? padding[position]
                                       : BDPadLayoutArrayAttr(),
             drains ? segment.getConsumeLockAttr()
                    : segment.getProduceLockAttr(),
             drains ? segment.getProduceLockAttr()
                    : segment.getConsumeLockAttr(),
             static_cast<int64_t>(index)});
      }
    }
    return descriptors;
  }

  void emitDescriptor(ObjectFifoDmaEndpointOp endpoint, ObjectFifoPoolOp pool,
                      Descriptor &descriptor, Block *successor,
                      bool binaryLocks, bool drains) {
    Location loc = endpoint.getLoc();
    int count = drains ? 1 : pool.getRepeatCount().value_or(1);

    LockOp acquireLock, releaseLock;
    int acquireValue = count, releaseValue = count;
    auto action = LockAction::AcquireGreaterEqual;
    if (binaryLocks) {
      // A binary lock's value encodes which side owns the buffer.
      SmallVector<LockOp> locks = pool.getLockOps();
      if (descriptor.binaryLock < (int64_t)locks.size()) {
        acquireLock = releaseLock = locks[descriptor.binaryLock];
        acquireValue = drains ? 1 : 0;
        releaseValue = drains ? 0 : 1;
        action = LockAction::Acquire;
      }
    } else if (descriptor.acquireLock) {
      acquireLock =
          lookupNamedOp<LockOp>(device, descriptor.acquireLock.getAttr());
      releaseLock =
          lookupNamedOp<LockOp>(device, descriptor.releaseLock.getAttr());
    }

    if (descriptor.acquireValue)
      acquireValue = *descriptor.acquireValue;
    if (descriptor.releaseValue)
      releaseValue = *descriptor.releaseValue;
    if (acquireLock) {
      UseLockOp::create(builder, loc, acquireLock, action, acquireValue);
    }
    if (auto packet = endpoint.getPacket()) {
      DMABDPACKETOp::create(builder, loc, packet->getPktType(),
                            packet->getPktId());
    }

    DMABDOp bd;
    if (descriptor.dimensions && drains && descriptor.padding) {
      bd = DMABDOp::create(builder, loc, descriptor.buffer, descriptor.offset,
                           paddedSize(descriptor.dimensions, descriptor.padding,
                                      descriptor.size),
                           descriptor.dimensions, descriptor.padding);
    } else if (descriptor.dimensions) {
      bd = DMABDOp::create(builder, loc, descriptor.buffer, descriptor.offset,
                           descriptor.size, descriptor.dimensions);
    } else {
      bd = DMABDOp::create(builder, loc, descriptor.buffer, descriptor.offset,
                           descriptor.size);
    }
    if (descriptor.iteration && descriptor.iteration.getSize() > 1)
      bd.setIterationAttr(descriptor.iteration);

    if (releaseLock) {
      UseLockOp::create(builder, loc, releaseLock, LockAction::Release,
                        releaseValue);
    }
    NextBDOp::create(builder, loc, successor);
  }

  LogicalResult lowerEndpoint(ObjectFifoDmaEndpointOp endpoint, int channel) {
    bool drains = endpoint.drains();
    ObjectFifoPoolOp pool = endpoint.getPoolOp();
    SmallVector<Value> buffers = buffersOf(pool);
    if (buffers.empty()) {
      return success();
    }

    SmallVector<Descriptor> descriptors =
        descriptorsFor(endpoint, buffers, drains);
    if (descriptors.empty()) {
      return success();
    }
    // Allocation budgeted the channel's BDs by getNumBDs.
    if (static_cast<int64_t>(descriptors.size()) !=
        endpoint.getNumDescriptors())
      return endpoint.emitOpError()
             << "lowers to " << descriptors.size()
             << " descriptors per pass, but its channel was allocated for "
             << endpoint.getNumDescriptors();

    Location loc = endpoint.getLoc();
    // emitDescriptor scales the filling end's lock count instead.
    int64_t repeat = endpoint.getRepeat();
    std::optional<int32_t> iterCount = endpoint.getIterCount();

    bool repeatInHardware = endpoint.repeatsInHardware();
    int taskCount =
        iterCount ? *iterCount - 1 : (repeatInHardware ? repeat - 1 : 0);
    int64_t copies = repeatInHardware ? 1 : repeat;

    DmaBody program = dmaProgramFor(endpoint.getTileLike(), loc);
    Block *endBlock = findEndOpBlock(program.getDmaBody());
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    builder.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(builder, loc, endpoint.getRouteDirection(), channel,
                       taskCount, endpoint.getPadValue(),
                       /*out_of_order=*/false, bdBlock, endBlock);
    if (lastDmaBlock) {
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);
    }

    bool binaryLocks = !device.getTargetModel().hasProperty(
        AIETargetModel::UsesSemaphoreLocks);
    size_t total = descriptors.size() * copies;
    size_t emitted = 0;
    Block *current = bdBlock;
    for (Descriptor &descriptor : descriptors) {
      for (int64_t copy = 0; copy < copies; copy++) {
        Block *successor;
        if (emitted + 1 < total) {
          successor = builder.createBlock(endBlock);
        } else if (iterCount) {
          // A bounded chain exits after its final iteration.
          successor = builder.createBlock(endBlock);
          builder.setInsertionPointToStart(successor);
          EndOp::create(builder, loc);
        } else {
          successor = bdBlock;
        }
        builder.setInsertionPointToStart(current);
        emitDescriptor(endpoint, pool, descriptor, successor, binaryLocks,
                       drains);
        current = successor;
        emitted++;
      }
    }
    return success();
  }

  void runOnOperation() override {
    device = getOperation();
    builder = OpBuilder(device.getContext());

    SmallVector<ObjectFifoDmaEndpointOp> endpoints(
        device.getOps<ObjectFifoDmaEndpointOp>());
    // A tile's chains, and so the BD ids they get, follow its channels rather
    // than the order the endpoints were written in.
    DenseMap<Value, size_t> tileOrder;
    for (ObjectFifoDmaEndpointOp endpoint : endpoints)
      tileOrder.try_emplace(endpoint.getTile(), tileOrder.size());
    llvm::stable_sort(
        endpoints, [&](ObjectFifoDmaEndpointOp a, ObjectFifoDmaEndpointOp b) {
          return std::make_tuple(tileOrder.lookup(a.getTile()),
                                 a.getRouteDirection(),
                                 a.getChannelIndex().value_or(-1)) <
                 std::make_tuple(tileOrder.lookup(b.getTile()),
                                 b.getRouteDirection(),
                                 b.getChannelIndex().value_or(-1));
        });
    for (ObjectFifoDmaEndpointOp endpoint : endpoints) {
      std::optional<int> channel = endpoint.getChannelIndex();
      if (!channel) {
        endpoint.emitOpError("has no channel; run --aie-objectfifo-allocate");
        return signalPassFailure();
      }
      if (failed(lowerEndpoint(endpoint, *channel)))
        return signalPassFailure();
      endpoint.erase();
    }

    // Nothing to program at the far end of these; the routes they asked for
    // have been drawn.
    for (auto endpoint :
         llvm::make_early_inc_range(device.getOps<RouteEndpointOp>())) {
      endpoint.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoLowerDMAsPass() {
  return std::make_unique<AIEObjectFifoLowerDMAsPass>();
}
