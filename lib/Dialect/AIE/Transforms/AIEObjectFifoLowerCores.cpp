//===- AIEObjectFifoLowerCores.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Mem2Reg.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include <numeric>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOLOWERCORES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

// Marks the `memref.alloca`s emitted here for bookkeeping only (objects held,
// current object index). Memrefs thread through control flow more easily than
// SSA values; mem2reg turns them back into SSA at the end of the pass, and the
// marker keeps that sweep off allocas this pass did not create.
constexpr llvm::StringLiteral kBookkeepingSlotAttrName =
    "aie.objectfifo.bookkeeping_slot";

/// Build an `scf.index_switch` on `idx` with `n` cases (0..n-1) plus a default;
/// case k (and the default) yields `elem(k)`. The runtime selection folds to a
/// constant once the enclosing loops are unrolled. Leaves the builder after the
/// switch and returns its result.
Value buildRotatingSwitch(OpBuilder &builder, Location loc, Value idx,
                          Type resultTy, int n,
                          llvm::function_ref<Value(int)> elem) {
  SmallVector<int64_t, 4> caseValues;
  for (int c = 0; c < n; ++c) {
    caseValues.push_back(c);
  }
  auto cases = DenseI64ArrayAttr::get(builder.getContext(), caseValues);
  auto switchOp = scf::IndexSwitchOp::create(builder, loc, TypeRange{resultTy},
                                             idx, cases, n);
  builder.createBlock(&switchOp.getDefaultRegion());
  builder.setInsertionPointToStart(&switchOp.getDefaultBlock());
  scf::YieldOp::create(builder, loc, elem(0));
  for (int c = 0; c < n; ++c) {
    builder.createBlock(&switchOp.getCaseRegions()[c]);
    builder.setInsertionPointToStart(&switchOp.getCaseBlock(c));
    scf::YieldOp::create(builder, loc, elem(c));
  }
  builder.setInsertionPointAfter(switchOp);
  return switchOp.getResult(0);
}

/// Advance a rotating object-index counter in its slot:
/// counter = (counter + released) mod depth.
void emitAdvanceObjectIndex(OpBuilder &builder, Location loc, Value slot,
                            int depth, int released) {
  Value size =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(depth));
  Value old = memref::LoadOp::create(builder, loc, slot, ValueRange{});
  Value step = arith::ConstantOp::create(builder, loc,
                                         builder.getI32IntegerAttr(released));
  Value sum = arith::AddIOp::create(builder, loc, old, step);
  Value wrapping =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, sum, size);
  // The counter stays in [0, depth), so releasing a single object can only ever
  // wrap to exactly 0.
  Value wrapped = released == 1
                      ? Value(arith::ConstantOp::create(
                            builder, loc, builder.getI32IntegerAttr(0)))
                      : Value(arith::SubIOp::create(builder, loc, sum, size));
  Value next = arith::SelectOp::create(builder, loc, wrapping, wrapped, sum);
  memref::StoreOp::create(builder, loc, next, slot, ValueRange{});
}

/// Locks a core endpoint toggles for `action`, one per segment it works: a
/// filler waits on free objects and hands back full ones, a drainer the
/// reverse.
SmallVector<FlatSymbolRefAttr>
segmentLocksFor(ObjectFifoCoreEndpointOp endpoint, LockAction action) {
  bool acquiring = action == LockAction::AcquireGreaterEqual;
  bool wantsProduce = acquiring != endpoint.drains();

  SmallVector<FlatSymbolRefAttr> locks;
  for (ObjectFifoSegmentAttr segment : endpoint.getSelectedSegments()) {
    if (auto lock = wantsProduce ? segment.getProduceLock()
                                 : segment.getConsumeLock()) {
      locks.push_back(lock);
    }
  }
  return locks;
}

/// Everything a rewrite pattern needs that is not local to its op, resolved
/// before the driver runs so that no matchAndRewrite scans the IR.
struct LoweringContext {
  DeviceOp device;
  bool usesSemaphoreLocks;
  /// Per-(core, endpoint) runtime bookkeeping: the rotating object index and,
  /// where locks count, the number of objects currently held.
  DenseMap<std::pair<Operation *, Operation *>, memref::AllocaOp> objectIndex{};
  DenseMap<std::pair<Operation *, Operation *>, memref::AllocaOp> heldCount{};
  bool sawError = false;

  ObjectFifoCoreEndpointOp endpointOf(Operation *op, StringRef name) {
    return SymbolTable::lookupNearestSymbolFrom<ObjectFifoCoreEndpointOp>(
        device, StringAttr::get(op->getContext(), name));
  }

  LockOp lockOf(FlatSymbolRefAttr name) {
    return SymbolTable::lookupNearestSymbolFrom<LockOp>(device, name);
  }
};

/// Emit the rotating UseLocks for a binary-lock (AIE1) acquire or release. One
/// lock travels with each object, and the starting index is only known at
/// runtime, so each of the `count` locks is selected by an scf.index_switch on
/// the rotation counter -- folded to a concrete lock once the loops unroll.
void emitBinaryUseLocks(OpBuilder &builder, Location loc,
                        MutableArrayRef<LockOp> locks, int depth, bool drains,
                        Value counterSlot, int count, LockAction action) {
  if (count == 0 || locks.empty()) {
    return;
  }
  // The value carries the direction: 1 for a filler's release or a drainer's
  // acquire, 0 for the opposite.
  bool full = (!drains && action == LockAction::Release) ||
              (drains && action == LockAction::Acquire);
  Value counter =
      memref::LoadOp::create(builder, loc, counterSlot, ValueRange{});
  Value index =
      arith::IndexCastOp::create(builder, loc, builder.getIndexType(), counter);
  Type lockTy = locks[0].getType();
  for (int i = 0; i < count; i++) {
    Value lock =
        buildRotatingSwitch(builder, loc, index, lockTy, depth, [&](int c) {
          return locks[(i + c) % depth].getResult();
        });
    UseLockOp::create(builder, loc, lock, action, full ? 1 : 0);
  }
}

struct LowerRelease : OpRewritePattern<ObjectFifoReleaseOp> {
  LoweringContext &ctx;
  LowerRelease(MLIRContext *context, LoweringContext &ctx)
      : OpRewritePattern(context), ctx(ctx) {}

  LogicalResult matchAndRewrite(ObjectFifoReleaseOp releaseOp,
                                PatternRewriter &rewriter) const override {
    auto endpoint = ctx.endpointOf(releaseOp, releaseOp.getObjFifoName());
    if (!endpoint) {
      return failure();
    }
    ObjectFifoPoolOp pool = endpoint.getPoolOp();
    CoreOp core = releaseOp->getParentOfType<CoreOp>();
    auto key = std::make_pair(core.getOperation(), endpoint.getOperation());
    memref::AllocaOp index = ctx.objectIndex.lookup(key);

    int count = releaseOp.relNumber() * pool.getRepeatCount().value_or(1);
    rewriter.setInsertionPointAfter(releaseOp);
    Location loc = releaseOp.getLoc();

    if (!ctx.usesSemaphoreLocks) {
      // The rotation counter is created by the matching acquire, so a release
      // without one is releasing objects never acquired on this tile.
      if (!index) {
        releaseOp->emitOpError(
            "objectFifo release has no corresponding acquire on this tile");
        ctx.sawError = true;
        return failure();
      }
      SmallVector<LockOp> locks = pool.getLockOps();
      emitBinaryUseLocks(rewriter, loc, locks, pool.getDepth(),
                         endpoint.drains(), index.getResult(), count,
                         LockAction::Release);
    } else {
      Value amount = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(count));
      for (FlatSymbolRefAttr name :
           segmentLocksFor(endpoint, LockAction::Release)) {
        UseLockOp::create(rewriter, loc, ctx.lockOf(name), LockAction::Release,
                          amount);
      }

      memref::AllocaOp held = ctx.heldCount.lookup(key);
      assert(held && "a held counter is created for every released endpoint");
      Value current =
          memref::LoadOp::create(rewriter, loc, held.getResult(), ValueRange{});
      Value next = arith::SubIOp::create(rewriter, loc, current, amount);
      memref::StoreOp::create(rewriter, loc, next, held.getResult(),
                              ValueRange{});
    }

    if (index) {
      emitAdvanceObjectIndex(rewriter, loc, index.getResult(), pool.getDepth(),
                             releaseOp.getSize());
    }
    return success();
  }
};

struct LowerAcquire : OpRewritePattern<ObjectFifoAcquireOp> {
  LoweringContext &ctx;
  LowerAcquire(MLIRContext *context, LoweringContext &ctx)
      : OpRewritePattern(context), ctx(ctx) {}

  LogicalResult matchAndRewrite(ObjectFifoAcquireOp acquireOp,
                                PatternRewriter &rewriter) const override {
    auto endpoint = ctx.endpointOf(acquireOp, acquireOp.getObjFifoName());
    if (!endpoint) {
      return failure();
    }
    ObjectFifoPoolOp pool = endpoint.getPoolOp();
    CoreOp core = acquireOp->getParentOfType<CoreOp>();
    auto key = std::make_pair(core.getOperation(), endpoint.getOperation());

    int wanted = acquireOp.acqNumber();
    int repeat = pool.getRepeatCount().value_or(1);
    rewriter.setInsertionPointAfter(acquireOp);
    Location loc = acquireOp.getLoc();

    if (!ctx.usesSemaphoreLocks) {
      // Each object has its own rotating binary lock; while the loops are
      // rolled the offset of the held locks is unknown, so acquire the whole
      // window [counter, counter + wanted). Redundant re-acquires of
      // still-held locks are pruned after unrolling.
      memref::AllocaOp index = ctx.objectIndex.lookup(key);
      assert(index && "a rotation counter is created for every acquire");
      SmallVector<LockOp> locks = pool.getLockOps();
      emitBinaryUseLocks(rewriter, loc, locks, pool.getDepth(),
                         endpoint.drains(), index.getResult(), wanted * repeat,
                         LockAction::Acquire);
      return replaceWithObjects(acquireOp, endpoint, key, rewriter);
    }

    // An acquire names every object the core wants to hold, so only the ones
    // not already held are taken.
    memref::AllocaOp held = ctx.heldCount.lookup(key);
    assert(held && "a held counter is created for every acquire");
    Value current =
        memref::LoadOp::create(rewriter, loc, held.getResult(), ValueRange{});
    Value target = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(wanted));
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value shortfall = arith::SubIOp::create(rewriter, loc, target, current);
    Value delta = arith::MaxSIOp::create(rewriter, loc, shortfall, zero);
    if (repeat > 1) {
      Value repeatVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(repeat));
      delta = arith::MulIOp::create(rewriter, loc, delta, repeatVal);
    }
    for (FlatSymbolRefAttr name :
         segmentLocksFor(endpoint, LockAction::AcquireGreaterEqual)) {
      UseLockOp::create(rewriter, loc, ctx.lockOf(name),
                        LockAction::AcquireGreaterEqual, delta);
    }

    Value next = arith::AddIOp::create(rewriter, loc, current, delta);
    memref::StoreOp::create(rewriter, loc, next, held.getResult(),
                            ValueRange{});
    return replaceWithObjects(acquireOp, endpoint, key, rewriter);
  }

  /// Hand back the objects the rotating index selects, one per result.
  LogicalResult replaceWithObjects(ObjectFifoAcquireOp acquireOp,
                                   ObjectFifoCoreEndpointOp endpoint,
                                   std::pair<Operation *, Operation *> key,
                                   PatternRewriter &rewriter) const {
    ObjectFifoPoolOp pool = endpoint.getPoolOp();
    SmallVector<Value> buffers;
    for (BufferLike buffer : pool.getBufferOps()) {
      buffers.push_back(buffer.getBuffer());
    }
    if (buffers.empty()) {
      return success();
    }
    memref::AllocaOp index = ctx.objectIndex.lookup(key);
    assert(index && "a rotation counter is created for every acquire");

    Location loc = acquireOp.getLoc();

    // An endpoint holding one side of a join or distribute reaches only its
    // own run of each object.
    MemRefType accessType = endpoint.getAccessType();
    if (accessType != pool.getElemType()) {
      auto [offset, size] = endpoint.getExtent();
      for (Value &buffer : buffers) {
        buffer = memref::SubViewOp::create(rewriter, loc, accessType, buffer,
                                           ArrayRef<OpFoldResult>{
                                               rewriter.getIndexAttr(offset)},
                                           {rewriter.getIndexAttr(size)},
                                           {rewriter.getIndexAttr(1)});
      }
    }
    Value counter =
        memref::LoadOp::create(rewriter, loc, index.getResult(), ValueRange{});
    Value idx = arith::IndexCastOp::create(rewriter, loc,
                                           rewriter.getIndexType(), counter);
    int depth = pool.getDepth();
    for (auto [base, result] : llvm::enumerate(acquireOp.getObjects())) {
      Value object = buildRotatingSwitch(
          rewriter, loc, idx, buffers[0].getType(), depth,
          [&, base = base](int c) { return buffers[(base + c) % depth]; });
      rewriter.replaceAllUsesWith(result, object);
    }
    return success();
  }
};

struct AIEObjectFifoLowerCoresPass
    : public xilinx::AIE::impl::AIEObjectFifoLowerCoresBase<
          AIEObjectFifoLowerCoresPass> {

  /// A rank-0 `memref.alloca` initialized to 0, which mem2reg later threads
  /// through the enclosing loops as an iter_arg.
  memref::AllocaOp makeSlot(OpBuilder &builder, Location loc, Value zero) {
    auto scalarTy =
        MemRefType::get(SmallVector<int64_t>{}, builder.getI32Type());
    auto slot = memref::AllocaOp::create(builder, loc, scalarTy);
    slot->setAttr(kBookkeepingSlotAttrName, builder.getUnitAttr());
    memref::StoreOp::create(builder, loc, zero, slot.getResult(), ValueRange{});
    return slot;
  }

  void emitCounters(CoreOp coreOp, LoweringContext &ctx, OpBuilder &builder) {
    builder.setInsertionPointToStart(&coreOp.getBody().front());
    Operation *core = coreOp.getOperation();
    Value zero;

    auto record = [&](DenseMap<std::pair<Operation *, Operation *>,
                               memref::AllocaOp> &slots,
                      StringRef name) {
      auto endpoint = ctx.endpointOf(coreOp, name);
      if (!endpoint) {
        return;
      }
      auto key = std::make_pair(core, endpoint.getOperation());
      if (slots.count(key)) {
        return;
      }
      if (!zero) {
        zero = arith::ConstantOp::create(builder, coreOp.getLoc(),
                                         builder.getI32IntegerAttr(0));
      }
      slots[key] = makeSlot(builder, coreOp.getLoc(), zero);
    };

    coreOp.walk([&](ObjectFifoAcquireOp a) {
      record(ctx.objectIndex, a.getObjFifoName());
    });
    if (!ctx.usesSemaphoreLocks) {
      return;
    }
    coreOp.walk([&](ObjectFifoAcquireOp a) {
      record(ctx.heldCount, a.getObjFifoName());
    });
    coreOp.walk([&](ObjectFifoReleaseOp r) {
      record(ctx.heldCount, r.getObjFifoName());
    });
  }

  /// Tell `--aie-objectFifo-unroll` how far a loop must be unrolled for the
  /// rotating indices inside it to fold: the least common multiple of the
  /// depths it touches. Accesses nested in a child loop drive that child, not
  /// this one, so an ancestor is not over-unrolled.
  void annotateUnrollHints(DeviceOp device, LoweringContext &ctx,
                           OpBuilder &builder) {
    for (auto coreOp : device.getOps<CoreOp>()) {
      coreOp.walk([&](scf::ForOp forOp) {
        int64_t period = 1;
        bool touched = false;
        auto account = [&](Operation *op, StringRef name) {
          if (op->getParentOfType<scf::ForOp>() != forOp) {
            return;
          }
          auto endpoint = ctx.endpointOf(op, name);
          if (!endpoint) {
            return;
          }
          touched = true;
          period = std::lcm(period, (int64_t)endpoint.getPoolOp().getDepth());
        };
        forOp.getBody()->walk(
            [&](ObjectFifoAcquireOp a) { account(a, a.getObjFifoName()); });
        forOp.getBody()->walk(
            [&](ObjectFifoReleaseOp r) { account(r, r.getObjFifoName()); });
        if (touched) {
          forOp->setAttr(kObjectFifoUnrollHintAttrName,
                         builder.getI64IntegerAttr(period));
        }
      });
    }
  }

  /// The bookkeeping slots exist only to reach SSA form; nothing downstream
  /// should see them.
  LogicalResult promoteBookkeepingSlots(DeviceOp device) {
    SmallVector<PromotableAllocationOpInterface> allocators;
    device.walk([&](memref::AllocaOp allocaOp) {
      if (allocaOp->hasAttr(kBookkeepingSlotAttrName)) {
        allocators.push_back(
            cast<PromotableAllocationOpInterface>(allocaOp.getOperation()));
      }
    });
    if (allocators.empty()) {
      return success();
    }

    DataLayout dataLayout = DataLayout::closest(device);
    DominanceInfo dominance(device);
    OpBuilder builder(device.getContext());
    (void)tryToPromoteMemorySlots(allocators, builder, dataLayout, dominance);

    WalkResult leftover = device.walk([&](memref::AllocaOp allocaOp) {
      if (!allocaOp->hasAttr(kBookkeepingSlotAttrName)) {
        return WalkResult::advance();
      }
      allocaOp.emitOpError()
          << "objectFifo bookkeeping slot could not be promoted to SSA "
             "(mem2reg left it in place); the objectFifo lowering requires "
             "all bookkeeping counters to become loop-carried SSA values";
      return WalkResult::interrupt();
    });
    return failure(leftover.wasInterrupted());
  }

  /// A loop body that releases more than it acquires underflows the held count
  /// as it repeats, whatever the trip count. Accesses split across nesting
  /// levels are excluded: balancing them needs the inner trip counts. This runs
  /// before lowering because afterwards the imbalance survives only as
  /// loop-carried lock values, which no scan can decide.
  LogicalResult verifyOverRelease(DeviceOp device) {
    for (auto coreOp : device.getOps<CoreOp>()) {
      WalkResult result = coreOp.walk([&](scf::ForOp forOp) {
        auto directlyIn = [&](Operation *op) {
          return op->getParentOfType<scf::ForOp>() == forOp;
        };
        DenseMap<StringRef, int64_t> acquired;
        DenseMap<StringRef, int64_t> released;
        DenseMap<StringRef, Operation *> blame;
        DenseSet<StringRef> spansNestedLoop;

        forOp.getBody()->walk([&](ObjectFifoAcquireOp a) {
          StringRef key = a.getObjFifoName();
          if (!directlyIn(a)) {
            spansNestedLoop.insert(key);
          } else {
            acquired[key] += a.acqNumber();
            blame.try_emplace(key, a);
          }
        });
        forOp.getBody()->walk([&](ObjectFifoReleaseOp r) {
          StringRef key = r.getObjFifoName();
          if (!directlyIn(r)) {
            spansNestedLoop.insert(key);
          } else {
            released[key] += r.relNumber();
            blame.try_emplace(key, r);
          }
        });

        for (auto &[key, count] : released) {
          if (spansNestedLoop.contains(key) || count <= acquired.lookup(key)) {
            continue;
          }
          blame.lookup(key)->emitOpError(
              "cannot release more elements than are already acquired");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        return failure();
      }
    }
    return success();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device.getContext());

    if (failed(verifyOverRelease(device))) {
      return signalPassFailure();
    }

    LoweringContext ctx{device, device.getTargetModel().hasProperty(
                                    AIETargetModel::UsesSemaphoreLocks)};

    annotateUnrollHints(device, ctx, builder);
    for (auto coreOp : device.getOps<CoreOp>()) {
      emitCounters(coreOp, ctx, builder);
    }

    RewritePatternSet patterns(device.getContext());
    patterns.insert<LowerAcquire, LowerRelease>(device.getContext(), ctx);
    walkAndApplyPatterns(device, std::move(patterns));
    if (ctx.sawError) {
      return signalPassFailure();
    }

    SmallVector<Operation *> toErase;
    device.walk([&](Operation *op) {
      if (isa<ObjectFifoAcquireOp, ObjectFifoReleaseOp>(op)) {
        toErase.push_back(op);
      }
    });
    for (Operation *op : toErase) {
      op->dropAllUses();
      op->erase();
    }

    for (auto endpoint : llvm::make_early_inc_range(
             device.getOps<ObjectFifoCoreEndpointOp>())) {
      endpoint.erase();
    }

    if (failed(promoteBookkeepingSlots(device))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoLowerCoresPass() {
  return std::make_unique<AIEObjectFifoLowerCoresPass>();
}
