//===- AIEObjectFifoVerify.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOVERIFY
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

struct SegmentActors {
  SmallVector<Operation *> fillers;
  SmallVector<Operation *> drainers;
};

struct AIEObjectFifoVerifyPass
    : public xilinx::AIE::impl::AIEObjectFifoVerifyBase<
          AIEObjectFifoVerifyPass> {

  /// Every segment needs an actor at each end, so that what one writes another
  /// reads. A segment may go unfilled when the pool's objects start full, and
  /// may name no endpoint at an end its `manualFill` or `manualDrain` hands to
  /// a DMA program the author writes.
  LogicalResult verifyActors(ObjectFifoPoolOp pool,
                             ArrayRef<SegmentActors> actors) {
    bool external = llvm::any_of(pool.getBufferOps(), [](BufferLike buffer) {
      return isa<ExternalBufferOp>(buffer.getOperation());
    });
    std::vector<ObjectFifoSegmentOp> segmentOps = pool.getSegmentOps();
    for (auto [index, segment] : llvm::enumerate(actors)) {
      ObjectFifoSegmentOp segmentOp = segmentOps[index];
      bool manualFill = segmentOp.getManualFill();
      bool manualDrain = segmentOp.getManualDrain();
      if (segment.fillers.size() > 1 ||
          (manualFill && !segment.fillers.empty())) {
        return pool.emitOpError("segment ")
               << index << " is filled by more than one endpoint";
      }
      if (segment.drainers.size() > 1 ||
          (manualDrain && !segment.drainers.empty())) {
        return pool.emitOpError("segment ")
               << index << " is drained by more than one endpoint";
      }
      // The host works an external pool's buffers, so it stands in for the end
      // this device does not program.
      bool hostDrains = external && !segment.fillers.empty();
      bool hostFills = external && !segment.drainers.empty();
      if (segment.drainers.empty() && !hostDrains && !manualDrain) {
        return pool.emitOpError("segment ") << index << " has no drainer";
      }
      if (segment.fillers.empty() && !pool.getInitValues() && !hostFills &&
          !manualFill) {
        return pool.emitOpError("segment ") << index << " has no filler";
      }
    }
    return success();
  }

  /// An endpoint that is not connected carries data nowhere.
  LogicalResult verifyFlows(DeviceOp device) {
    DenseMap<StringRef, int> appearances;
    for (auto flow : device.getOps<RouteOp>()) {
      appearances[flow.getSource()]++;
      for (auto dest : flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
        appearances[dest.getValue()]++;
      }
    }

    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      int count =
          appearances.lookup(cast<SymbolOpInterface>(*endpoint).getName());
      if (count == 0) {
        return endpoint->emitOpError("is not connected by any flow");
      }
      // An aie.route_endpoint's own verifier says which flows may share it.
      if (count > 1 && !isa<RouteEndpointOp>(endpoint.getOperation())) {
        return endpoint->emitOpError(
                   "drives one channel, so at most one flow may name it, but "
                   "it is named ")
               << count << " times";
      }
    }
    return success();
  }

  /// A loop body that releases more than it acquires underflows the held count
  /// as it repeats, whatever the trip count. Accesses split across nesting
  /// levels are excluded: balancing them needs the inner trip counts. This runs
  /// before lowering because afterwards the imbalance survives only as
  /// loop-carried lock values, which no scan can decide.
  LogicalResult verifyOverRelease(DeviceOp device) {
    for (auto coreOp : device.getOps<CoreOp>()) {
      WalkResult result = coreOp.walk([&](Operation *loop) {
        if (!isObjectFifoLoop(loop)) {
          return WalkResult::advance();
        }
        auto directlyIn = [&](Operation *op) {
          return getObjectFifoLoopParent(op) == loop;
        };
        DenseMap<StringRef, int64_t> acquired;
        DenseMap<StringRef, int64_t> released;
        DenseMap<StringRef, Operation *> blame;
        DenseSet<StringRef> spansNestedLoop;

        loop->walk([&](ObjectFifoAcquireOp a) {
          StringRef key = a.getObjFifoName();
          if (!directlyIn(a)) {
            spansNestedLoop.insert(key);
          } else {
            acquired[key] += a.acqNumber();
            blame.try_emplace(key, a);
          }
        });
        loop->walk([&](ObjectFifoReleaseOp r) {
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

  /// An out-of-order sender names the buffer each object lands in, and one BD
  /// carries one such name, so the ends have to hold the same buffers in the
  /// same order. Naming a slice of a larger buffer would need the sender split
  /// into one descriptor per slice, which nothing does yet.
  LogicalResult verifyOutOfOrderPools(DeviceOp device) {
    auto lookup = [&](FlatSymbolRefAttr name) {
      return dyn_cast_or_null<ObjectFifoDmaEndpointOp>(
          SymbolTable::lookupNearestSymbolFrom(device, name.getAttr()));
    };
    auto objectBytes = [](ObjectFifoPoolOp pool) -> int64_t {
      auto type = dyn_cast<MemRefType>(pool.getElemType());
      if (!type)
        return 0;
      return (type.getNumElements() * type.getElementTypeBitWidth() + 7) / 8;
    };
    for (auto flow : device.getOps<RouteOp>()) {
      ObjectFifoDmaEndpointOp src = lookup(flow.getSourceAttr());
      if (!src || !src.getOutOfOrder())
        continue;
      for (auto destName :
           flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
        ObjectFifoDmaEndpointOp dst = lookup(destName);
        if (!dst)
          continue;
        ObjectFifoPoolOp from = src.getPoolOp(), to = dst.getPoolOp();
        if (!from || !to)
          continue;
        if (src.getNumDescriptors() != dst.getNumDescriptors() ||
            objectBytes(from) != objectBytes(to))
          return src.emitOpError("sends out of order from ")
                 << src.getNumDescriptors() << " buffers of "
                 << objectBytes(from) << " bytes into "
                 << dst.getNumDescriptors() << " of " << objectBytes(to)
                 << "; an out-of-order fifo needs the same buffers at both "
                    "ends, because one descriptor names one whole buffer";
      }
    }
    return success();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();

    DenseMap<Operation *, SmallVector<SegmentActors>> actorsPerPool;
    for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
      actorsPerPool[pool].resize(pool.getSegmentOps().size());
    }

    auto record = [&](auto endpoint) {
      ObjectFifoPoolOp pool = endpoint.getPoolOp();
      if (!pool) {
        return;
      }
      auto &actors = actorsPerPool[pool];
      std::vector<ObjectFifoSegmentOp> all = pool.getSegmentOps();
      for (ObjectFifoSegmentOp segment : endpoint.getSelectedSegments()) {
        size_t index = llvm::find(all, segment) - all.begin();
        (endpoint.drains() ? actors[index].drainers : actors[index].fillers)
            .push_back(endpoint);
      }
    };

    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      record(endpoint);
    }
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      record(endpoint);
    }

    for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
      if (failed(verifyActors(pool, actorsPerPool[pool]))) {
        return signalPassFailure();
      }
    }

    if (failed(verifyFlows(device)) || failed(verifyOverRelease(device)) ||
        failed(verifyOutOfOrderPools(device))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoVerifyPass() {
  return std::make_unique<AIEObjectFifoVerifyPass>();
}
