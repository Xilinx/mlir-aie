//===- AIEObjectFifoInsertCoreQueuePushes.cpp -------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Push onto an out-of-order channel's task queue as the core consumes.
//
// A channel receiving out of order walks no chain of buffer descriptors, so
// nothing carries it round: it takes the objects its queue was pushed for and
// stops. A fifo that names no iter_count receives objects without end, and
// only the core on the receiving tile can push while the program runs.
//
// One push covers as many objects as a task's repeat count holds, so the core
// counts the objects it frees and pushes once per run of that many, rather
// than once per object.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOINSERTCOREQUEUEPUSHES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct AIEObjectFifoInsertCoreQueuePushesPass
    : public xilinx::AIE::impl::AIEObjectFifoInsertCoreQueuePushesBase<
          AIEObjectFifoInsertCoreQueuePushesPass> {
  using Base = xilinx::AIE::impl::AIEObjectFifoInsertCoreQueuePushesBase<
      AIEObjectFifoInsertCoreQueuePushesPass>;
  using Base::Base;

  /// A counter in the core's own memory, set to `initial`.
  Value makeCounter(CoreOp core, int32_t initial) {
    OpBuilder builder(core.getContext());
    Block &entry = core.getBody().front();
    builder.setInsertionPointToStart(&entry);
    Location loc = core.getLoc();
    Type i32 = builder.getI32Type();
    Value slot =
        memref::AllocaOp::create(builder, loc, MemRefType::get({}, i32));
    Value start = arith::ConstantOp::create(builder, loc, i32,
                                            builder.getI32IntegerAttr(initial));
    memref::StoreOp::create(builder, loc, start, slot, ValueRange{});
    return slot;
  }

  /// Count `released` objects against `counter`, and push once the count runs
  /// out. `perPush` objects make up one push, whose task repeats `repeat`
  /// times over the descriptor each object names.
  void emitCountAndPush(OpBuilder &builder, Location loc, Value counter,
                        int32_t released, int32_t perPush, int32_t repeat,
                        int channel) {
    Type i32 = builder.getI32Type();
    Value left = memref::LoadOp::create(builder, loc, counter, ValueRange{});
    Value step = arith::ConstantOp::create(builder, loc, i32,
                                           builder.getI32IntegerAttr(released));
    Value zero =
        arith::ConstantOp::create(builder, loc, i32, builder.getI32IntegerAttr(0));
    Value run = arith::ConstantOp::create(builder, loc, i32,
                                          builder.getI32IntegerAttr(perPush));
    Value rest = arith::SubIOp::create(builder, loc, left, step);
    Value spent = arith::CmpIOp::create(builder, loc,
                                        arith::CmpIPredicate::sle, rest, zero);
    auto push = scf::IfOp::create(builder, loc, spent, /*withElse=*/false);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(push.thenBlock());
      // A push onto a full queue is discarded, which strands whoever waits on
      // the objects it would have covered.
      if (checkQueueSpace)
        DmaPollForQueueSpaceOp::create(builder, loc, DMAChannelDir::S2MM,
                                       channel);
      DmaPushQueueOp::create(builder, loc, DMAChannelDir::S2MM, channel,
                             /*bd_id=*/0, repeat, /*out_of_order=*/true);
    }
    // Carry what the run overshot by, so the pushes stay in step with the
    // objects however many a release covers.
    Value renewed = arith::AddIOp::create(builder, loc, rest, run);
    Value next = arith::SelectOp::create(builder, loc, spent, renewed, rest);
    memref::StoreOp::create(builder, loc, next, counter, ValueRange{});
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    const AIETargetModel &targetModel = device.getTargetModel();

    // The receiving endpoints that go on receiving: out of order, and with no
    // iter_count to say when they stop.
    DenseMap<StringRef, ObjectFifoDmaEndpointOp> receiverOfPool;
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>())
      if (endpoint.getOutOfOrder() && !endpoint.drains() &&
          !endpoint.getIterCount())
        receiverOfPool[endpoint.getPool()] = endpoint;
    if (receiverOfPool.empty())
      return;

    int32_t repeat = static_cast<int32_t>(targetModel.getMaxRepeatCount());
    int32_t perPush = repeat + 1;

    for (CoreOp core : device.getOps<CoreOp>()) {
      // Each channel takes one count, since every object the core frees on it
      // belongs to the same queue however many fifos share the channel.
      DenseMap<int, SmallVector<std::pair<ObjectFifoReleaseOp, int32_t>>>
          releasesOfChannel;
      WalkResult walked = core.walk([&](ObjectFifoReleaseOp release) {
        auto endpoint = SymbolTable::lookupNearestSymbolFrom<
            ObjectFifoCoreEndpointOp>(device, release.getObjFifoNameAttr());
        if (!endpoint || !endpoint.drains())
          return WalkResult::advance();
        auto it = receiverOfPool.find(endpoint.getPool());
        if (it == receiverOfPool.end())
          return WalkResult::advance();
        std::optional<int32_t> channel = it->second.getChannelIndex();
        if (!channel) {
          it->second.emitOpError("receives out of order without a channel; "
                                 "run --aie-objectfifo-allocate first");
          return WalkResult::interrupt();
        }
        releasesOfChannel[*channel].emplace_back(release,
                                                release.relNumber());
        return WalkResult::advance();
      });
      if (walked.wasInterrupted())
        return signalPassFailure();
      if (releasesOfChannel.empty())
        continue;

      for (auto &[channel, releases] : releasesOfChannel) {
        // The channel holds a full run of objects before the core frees any,
        // so the count starts at one: the first release pushes the run that
        // follows the one the configuration pushed.
        Value counter = makeCounter(core, 1);
        for (auto [release, size] : releases) {
          OpBuilder builder(release.getContext());
          builder.setInsertionPointAfter(release);
          emitCountAndPush(builder, release.getLoc(), counter, size, perPush,
                           repeat, channel);
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoInsertCoreQueuePushesPass() {
  return std::make_unique<AIEObjectFifoInsertCoreQueuePushesPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoInsertCoreQueuePushesPass(
    const AIEObjectFifoInsertCoreQueuePushesOptions &options) {
  return std::make_unique<AIEObjectFifoInsertCoreQueuePushesPass>(options);
}
