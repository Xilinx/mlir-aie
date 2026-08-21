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

  /// Buffer-major, segment-minor: the DMA walks the objects in turn, and within
  /// each object the slices this endpoint is responsible for.
  SmallVector<Descriptor> descriptorsFor(ObjectFifoDmaEndpointOp endpoint,
                                         ArrayRef<Value> buffers, bool drains) {
    SmallVector<ObjectFifoSegmentAttr> segments =
        endpoint.getSelectedSegments();
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
             drains ? segment.getConsumeLock() : segment.getProduceLock(),
             drains ? segment.getProduceLock() : segment.getConsumeLock(),
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
      acquireLock = SymbolTable::lookupNearestSymbolFrom<LockOp>(
          device, descriptor.acquireLock);
      releaseLock = SymbolTable::lookupNearestSymbolFrom<LockOp>(
          device, descriptor.releaseLock);
    }

    if (acquireLock) {
      UseLockOp::create(builder, loc, acquireLock, action, acquireValue);
    }
    if (auto packet = endpoint.getPacket()) {
      DMABDPACKETOp::create(builder, loc, packet->getPktType(),
                            packet->getPktId());
    }

    if (descriptor.dimensions && drains && descriptor.padding) {
      DMABDOp::create(builder, loc, descriptor.buffer, descriptor.offset,
                      paddedSize(descriptor.dimensions, descriptor.padding,
                                 descriptor.size),
                      descriptor.dimensions, descriptor.padding);
    } else if (descriptor.dimensions) {
      DMABDOp::create(builder, loc, descriptor.buffer, descriptor.offset,
                      descriptor.size, descriptor.dimensions);
    } else {
      DMABDOp::create(builder, loc, descriptor.buffer, descriptor.offset,
                      descriptor.size);
    }

    if (releaseLock) {
      UseLockOp::create(builder, loc, releaseLock, LockAction::Release,
                        releaseValue);
    }
    NextBDOp::create(builder, loc, successor);
  }

  void lowerEndpoint(ObjectFifoDmaEndpointOp endpoint, int channel) {
    bool drains = endpoint.drains();
    ObjectFifoPoolOp pool = endpoint.getPoolOp();
    SmallVector<Value> buffers = buffersOf(pool);
    if (buffers.empty()) {
      return;
    }

    SmallVector<Descriptor> descriptors =
        descriptorsFor(endpoint, buffers, drains);
    if (descriptors.empty()) {
      return;
    }

    Location loc = endpoint.getLoc();
    // Only the draining end replays; the filling end covers the batch in one
    // acquire, which is why emitDescriptor scales its lock count instead.
    int repeat = drains ? pool.getRepeatCount().value_or(1) : 1;
    std::optional<int32_t> iterCount = endpoint.getIterCount();

    // FIXME: repeat_count and iter_count are tangled here. Deriving one from
    // the other is confusing: each should mean one thing and be honored or
    // rejected, not reinterpreted.
    //
    // A single-descriptor chain can use the DMA start queue's repeat count.
    bool repeatInHardware = repeat > 1 && descriptors.size() == 1 && !iterCount;
    int taskCount =
        iterCount ? *iterCount - 1 : (repeatInHardware ? repeat - 1 : 0);
    int copies = repeatInHardware ? 1 : repeat;

    DmaBody program = dmaProgramFor(endpoint.getTileLike(), loc);
    Block *endBlock = findEndOpBlock(program.getDmaBody());
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    builder.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(builder, loc, endpoint.getFlowDirection(), channel,
                       taskCount, endpoint.getPadValue(), bdBlock, endBlock);
    if (lastDmaBlock) {
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);
    }

    bool binaryLocks = !device.getTargetModel().hasProperty(
        AIETargetModel::UsesSemaphoreLocks);
    size_t total = descriptors.size() * copies;
    size_t emitted = 0;
    Block *current = bdBlock;
    for (Descriptor &descriptor : descriptors) {
      for (int copy = 0; copy < copies; copy++) {
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
  }

  void runOnOperation() override {
    device = getOperation();
    builder = OpBuilder(device.getContext());

    SmallVector<ObjectFifoDmaEndpointOp> endpoints(
        device.getOps<ObjectFifoDmaEndpointOp>());
    for (ObjectFifoDmaEndpointOp endpoint : endpoints) {
      std::optional<int> channel = endpoint.getChannelIndex();
      if (!channel) {
        endpoint.emitOpError("has no channel; run --aie-objectfifo-allocate");
        return signalPassFailure();
      }
      lowerEndpoint(endpoint, *channel);
      endpoint.erase();
    }

    // Nothing to program at the far end of these; the flows they asked for
    // have been drawn.
    for (auto dangling : llvm::make_early_inc_range(
             device.getOps<ObjectFifoDanglingEndpointOp>())) {
      dangling.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoLowerDMAsPass() {
  return std::make_unique<AIEObjectFifoLowerDMAsPass>();
}
