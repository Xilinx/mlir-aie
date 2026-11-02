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
  /// Which of the pool's objects this moves; also the index of the object's
  /// own binary lock, where the device uses those.
  int64_t object;
  /// Lock units taken and given back, where they are not one object's.
  std::optional<int64_t> acquireValue = std::nullopt;
  std::optional<int64_t> releaseValue = std::nullopt;
  /// How the address steps on each run of a descriptor that walks a
  /// contiguous pool's objects in turn.
  BDIterationAttr iteration = nullptr;
};

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

  /// The DMA program of `tile`, created empty if the tile has none yet. A
  /// program shared by several endpoints is located at all of them.
  DmaBody dmaProgramFor(TileLike tile, Location loc) {
    Value wanted = tile->getResult(0);
    for (auto program : device.getOps<DmaBody>()) {
      if (program.getTile() == wanted) {
        program->setLoc(
            FusedLoc::get(loc.getContext(), {program->getLoc(), loc}));
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
      int64_t walkers = endpoint.getNumDescriptors();
      SmallVector<Descriptor> descriptors;
      for (int64_t j = 0; j < walkers; j++)
        descriptors.push_back(
            {buffer, first + j * object + segment.getOffset(),
             segment.getSize(), dims, BDPadLayoutArrayAttr(), take, give, j,
             std::nullopt, std::nullopt,
             BDIterationAttr::get(context, pool.getDepth() / walkers,
                                  walkers * object, 0)});
      return descriptors;
    }

    StreamWindow window;
    if (failed(endpoint.getStreamWindow(window)))
      return {};
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
                      Descriptor &descriptor, PacketInfoAttr packet,
                      Block *successor, bool binaryLocks, bool drains) {
    Location loc = endpoint.getLoc();
    int count = drains ? 1 : pool.getRepeatCount().value_or(1);

    LockOp acquireLock, releaseLock;
    int acquireValue = count, releaseValue = count;
    auto action = LockAction::AcquireGreaterEqual;
    if (binaryLocks) {
      // A binary lock's value encodes which side owns the buffer.
      SmallVector<LockOp> locks = pool.getLockOps();
      if (descriptor.object < (int64_t)locks.size()) {
        acquireLock = releaseLock = locks[descriptor.object];
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
    if (packet) {
      DMABDPACKETOp::create(builder, loc, packet.getPktType(),
                            packet.assignedId());
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

  /// A `dma_start` of `channel` on the endpoint's tile, chained after the
  /// tile's others; returns the first BD's block and the program's end block.
  std::pair<Block *, Block *> startChain(ObjectFifoDmaEndpointOp endpoint,
                                         int channel, int taskCount) {
    Location loc = endpoint.getLoc();
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
    return {bdBlock, endBlock};
  }

  /// Endpoints taking turns on one channel share one chain: the k-th BD moves
  /// the next object of the endpoint `turns` names k-th, through its own pool.
  LogicalResult lowerTurns(ObjectFifoDmaEndpointOp first, int channel) {
    SmallVector<ObjectFifoDmaEndpointOp> ends = turnEndpoints(first);
    SmallVector<SmallVector<Descriptor>> descriptors;
    for (ObjectFifoDmaEndpointOp end : ends) {
      descriptors.push_back(
          descriptorsFor(end, buffersOf(end.getPoolOp()), end.drains()));
      if (static_cast<int64_t>(descriptors.back().size()) !=
              end.getNumDescriptors() ||
          descriptors.back().empty())
        return end.emitOpError("takes turns, but lowers to ")
               << descriptors.back().size() << " descriptors per pass for "
               << end.getNumDescriptors();
    }
    SmallVector<unsigned> turns;
    for (auto name : first.getTurnsAttr().getAsRange<FlatSymbolRefAttr>())
      turns.push_back(llvm::find_if(ends,
                                    [&](ObjectFifoDmaEndpointOp end) {
                                      return end.getSymNameAttr() ==
                                             name.getAttr();
                                    }) -
                      ends.begin());

    auto [bdBlock, endBlock] = startChain(first, channel, /*taskCount=*/0);
    bool binaryLocks = !device.getTargetModel().hasProperty(
        AIETargetModel::UsesSemaphoreLocks);
    SmallVector<size_t> next(ends.size(), 0);
    int64_t total = first.getNumBDs();
    Block *current = bdBlock;
    for (int64_t k = 0; k < total; k++) {
      unsigned i = turns[k % turns.size()];
      Block *successor =
          k + 1 < total ? builder.createBlock(endBlock) : bdBlock;
      builder.setInsertionPointToStart(current);
      emitDescriptor(ends[i], ends[i].getPoolOp(), descriptors[i][next[i]],
                     PacketInfoAttr(), successor, binaryLocks,
                     ends[i].drains());
      next[i] = (next[i] + 1) % descriptors[i].size();
      current = successor;
    }
    return success();
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
    // A dispatching chain runs until its turns line up with the pool's
    // objects again, so one run covers that many passes.
    int64_t passes = endpoint.getDispatchPasses();
    int taskCount = iterCount ? *iterCount / passes - 1
                              : (repeatInHardware ? repeat - 1 : 0);
    int64_t copies = repeatInHardware ? 1 : repeat;
    SmallVector<PacketInfoAttr> turns;
    if (std::optional<ArrayAttr> headers = endpoint.getDispatchPackets()) {
      llvm::append_range(turns, headers->getAsRange<PacketInfoAttr>());
    } else if (endpoint.getDispatch()) {
      return endpoint.emitOpError(
          "has no dispatchPackets; run --aie-objectfifo-allocate");
    }
    // A contiguous pool's descriptor steps through every object, so it has no
    // one object for a turn to follow.
    if (!turns.empty() && pool.getContiguous())
      return endpoint.emitOpError(
          "dispatches from a contiguous pool, which has no per-object BDs");
    int64_t depth = pool.getDepth();

    auto [bdBlock, endBlock] = startChain(endpoint, channel, taskCount);
    bool binaryLocks = !device.getTargetModel().hasProperty(
        AIETargetModel::UsesSemaphoreLocks);
    size_t total = descriptors.size() * copies * passes;
    size_t emitted = 0;
    Block *current = bdBlock;
    for (int64_t pass = 0; pass < passes; pass++) {
      for (Descriptor &descriptor : descriptors) {
        PacketInfoAttr packet = endpoint.getPacketAttr();
        if (!turns.empty())
          packet = turns[(pass * depth + descriptor.object) % turns.size()];
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
          emitDescriptor(endpoint, pool, descriptor, packet, successor,
                         binaryLocks, drains);
          current = successor;
          emitted++;
        }
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
      SmallVector<ObjectFifoDmaEndpointOp> turns = turnEndpoints(endpoint);
      if (turns.empty() ? failed(lowerEndpoint(endpoint, *channel))
                        : turns.front() == endpoint &&
                              failed(lowerTurns(endpoint, *channel)))
        return signalPassFailure();
    }
    for (ObjectFifoDmaEndpointOp endpoint : endpoints)
      endpoint.erase();

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
