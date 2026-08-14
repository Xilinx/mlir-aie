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
    for (auto [index, segment] : llvm::enumerate(actors)) {
      if (segment.fillers.size() > 1) {
        return pool.emitOpError("segment ")
               << index << " is filled by more than one endpoint";
      }
      if (segment.drainers.size() > 1) {
        return pool.emitOpError("segment ")
               << index << " is drained by more than one endpoint";
      }
      if (segment.drainers.empty()) {
        return pool.emitOpError("segment ") << index << " has no drainer";
      }
      if (segment.fillers.empty() && !pool.getInitValues()) {
        return pool.emitOpError("segment ") << index << " has no filler";
      }
    }
    return success();
  }

  /// A DMA endpoint that is not connected carries data nowhere.
  LogicalResult verifyFlows(DeviceOp device) {
    DenseMap<StringRef, int> appearances;
    for (auto flow : device.getOps<ObjectFifoFlowOp>()) {
      appearances[flow.getSource()]++;
      for (auto dest : flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
        appearances[dest.getValue()]++;
      }
    }

    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      int count = appearances.lookup(endpoint.getSymName());
      if (count == 0) {
        return endpoint.emitOpError("is not connected by any flow");
      }
      if (count > 1) {
        return endpoint.emitOpError("appears in ")
               << count << " flows; a DMA endpoint drives one channel";
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

    if (failed(verifyFlows(device))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoVerifyPass() {
  return std::make_unique<AIEObjectFifoVerifyPass>();
}
