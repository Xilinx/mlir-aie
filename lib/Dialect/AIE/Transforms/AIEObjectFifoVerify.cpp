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

  /// Segments must tile the element type: a gap has no filler, and an overlap
  /// gives two actors the same bytes under different locks.
  LogicalResult verifyCoverage(ObjectFifoPoolOp pool) {
    int64_t expected = 0;
    for (ObjectFifoSegmentAttr segment : pool.getSegmentAttrs()) {
      if (static_cast<int64_t>(segment.getOffset()) != expected) {
        return pool.emitOpError("segments must be contiguous, but segment at "
                                "offset ")
               << segment.getOffset() << " follows " << expected;
      }
      expected += segment.getSize();
    }
    if (expected != pool.getObjectSize()) {
      return pool.emitOpError("segments cover ")
             << expected << " of " << pool.getObjectSize() << " elements";
    }
    return success();
  }

  /// Every segment needs an actor at each end, so that what one writes another
  /// reads. A segment may go unfilled when the pool's objects start full.
  LogicalResult verifyActors(ObjectFifoPoolOp pool,
                             ArrayRef<SegmentActors> actors) {
    bool external = llvm::any_of(pool.getBufferOps(), [](BufferLike buffer) {
      return isa<ExternalBufferOp>(buffer.getOperation());
    });
    for (auto [index, segment] : llvm::enumerate(actors)) {
      if (segment.fillers.size() > 1) {
        return pool.emitOpError("segment ")
               << index << " is filled by more than one endpoint";
      }
      if (segment.drainers.size() > 1) {
        return pool.emitOpError("segment ")
               << index << " is drained by more than one endpoint";
      }
      if (segment.drainers.empty() && !(external && !segment.fillers.empty())) {
        return pool.emitOpError("segment ") << index << " has no drainer";
      }
      if (segment.fillers.empty() && !pool.getInitValues() &&
          !(external && !segment.drainers.empty())) {
        return pool.emitOpError("segment ") << index << " has no filler";
      }
    }
    return success();
  }

  /// An endpoint that is not connected carries data nowhere.
  LogicalResult verifyFlows(DeviceOp device) {
    DenseMap<StringRef, int> appearances;
    for (auto flow : device.getOps<ObjectFifoFlowOp>()) {
      appearances[flow.getSource()]++;
      for (auto dest : flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
        appearances[dest.getValue()]++;
      }
    }

    for (auto endpoint : device.getOps<ObjectFifoFlowEndpoint>()) {
      int count =
          appearances.lookup(cast<SymbolOpInterface>(*endpoint).getName());
      if (count == 0) {
        return endpoint->emitOpError("is not connected by any flow");
      }
      if (count > 1) {
        return endpoint->emitOpError("is named ")
               << count << " times by flows; an endpoint drives one channel";
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

    DenseMap<Operation *, SmallVector<SegmentActors>> actorsPerPool;
    for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
      if (failed(verifyCoverage(pool))) {
        return signalPassFailure();
      }
      actorsPerPool[pool].resize(pool.getSegmentAttrs().size());
    }

    auto record = [&](auto endpoint) {
      ObjectFifoPoolOp pool = endpoint.getPoolOp();
      if (!pool) {
        return;
      }
      auto &actors = actorsPerPool[pool];
      SmallVector<ObjectFifoSegmentAttr> all = pool.getSegmentAttrs();
      for (ObjectFifoSegmentAttr segment : endpoint.getSelectedSegments()) {
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

    if (failed(verifyFlows(device)) || failed(verifyOverRelease(device))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoVerifyPass() {
  return std::make_unique<AIEObjectFifoVerifyPass>();
}
