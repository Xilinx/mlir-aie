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

#include <limits>
#include <optional>

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

/// A DMA endpoint with an iteration count moves this many objects through its
/// pool, then stops.
struct TransferBudget {
  ObjectFifoDmaEndpointOp endpoint;
  int32_t iterations;
  int64_t objects;
};

/// A loop the frontend means to run forever has a bound no design counts to:
/// IRON writes sys.maxsize, hand-written designs 2^32 - 1. From the AIE's
/// 24-bit BD-loop saturation value up, a trip count is read as "forever"; below
/// it, as a real bound.
static constexpr int64_t kRunsForever = (int64_t{1} << 24) - 2;

static std::optional<int64_t> getStaticTripCount(scf::ForOp forOp) {
  if (forOp.getSingleLowerBound() && forOp.getSingleUpperBound() &&
      forOp.getSingleStep()) {
    if (std::optional<llvm::APInt> tc = forOp.getStaticTripCount()) {
      return tc->getSExtValue();
    }
  }
  return std::nullopt;
}

/// How many times `op` runs when its core runs to completion, when static loop
/// bounds alone decide that. Anything conditional between the op and the core,
/// an `scf.if`, a loop with dynamic bounds or one meant to run forever, a block
/// reached by branching, makes it unknown.
static std::optional<int64_t> staticExecutions(Operation *op) {
  int64_t count = 1;
  for (Operation *cur = op; !isa<CoreOp>(cur); cur = cur->getParentOp()) {
    if (cur->getBlock() != &cur->getParentRegion()->front()) {
      return std::nullopt;
    }
    Operation *parent = cur->getParentOp();
    if (isa<CoreOp>(parent)) {
      break;
    }
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp) {
      return std::nullopt;
    }
    std::optional<int64_t> trip = getStaticTripCount(forOp);
    if (!trip || *trip >= kRunsForever ||
        (*trip != 0 && count > std::numeric_limits<int64_t>::max() / *trip)) {
      return std::nullopt;
    }
    count *= *trip;
  }
  return count;
}

struct AIEObjectFifoVerifyPass
    : public xilinx::AIE::impl::AIEObjectFifoVerifyBase<
          AIEObjectFifoVerifyPass> {

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
      // The host works an external pool's buffers, so it stands in for the end
      // this device does not program.
      bool hostDrains = external && !segment.fillers.empty();
      bool hostFills = external && !segment.drainers.empty();
      if (segment.drainers.empty() && !hostDrains) {
        return pool.emitOpError("segment ") << index << " has no drainer";
      }
      if (segment.fillers.empty() && !pool.getInitValues() && !hostFills) {
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
      if (count > 1) {
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

  /// A DMA endpoint with an iteration count stops after `iterCount * depth`
  /// objects, so a core that must release more through the same pool waits
  /// forever for the rest. Only releases static loop bounds force are counted,
  /// so what is reported is certain; a conditional release, or one in a loop
  /// meant to run forever, is left out, since such a core stalling once the
  /// data stops is how those designs end.
  LogicalResult verifyTransferBudgets(DeviceOp device) {
    DenseMap<Operation *, TransferBudget> budgets;
    for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>()) {
      std::optional<int32_t> iterations = endpoint.getIterCount();
      ObjectFifoPoolOp pool = endpoint.getPoolOp();
      if (!iterations || !pool) {
        continue;
      }
      budgets[pool] = {endpoint, *iterations,
                       int64_t{*iterations} * pool.getDepth()};
    }
    if (budgets.empty()) {
      return success();
    }

    DenseMap<StringRef, int64_t> released;
    DenseMap<StringRef, ObjectFifoReleaseOp> blame;
    device.walk([&](ObjectFifoReleaseOp release) {
      std::optional<int64_t> runs = staticExecutions(release);
      if (!runs) {
        return;
      }
      StringRef key = release.getObjFifoName();
      released[key] += *runs * release.relNumber();
      blame.try_emplace(key, release);
    });

    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      // A segment-selecting endpoint sees a share of the pool, not the whole.
      auto budget = budgets.find(endpoint.getPoolOp());
      if (budget == budgets.end() || endpoint.getSegments()) {
        continue;
      }
      StringRef name = endpoint.getSymName();
      int64_t count = released.lookup(name);
      if (count <= budget->second.objects) {
        continue;
      }
      ObjectFifoDmaEndpointOp dma = budget->second.endpoint;
      InFlightDiagnostic diag =
          blame.lookup(name).emitOpError("releases ")
          << count << " objects through @" << name
          << " over the run, but the pool's DMA endpoint stops after "
          << budget->second.objects << " (iterCount "
          << budget->second.iterations << " x depth "
          << endpoint.getPoolOp().getDepth()
          << "), so the core would wait forever for the rest";
      diag.attachNote(dma.getLoc()) << "the DMA endpoint is here";
      return failure();
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
        failed(verifyTransferBudgets(device))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoVerifyPass() {
  return std::make_unique<AIEObjectFifoVerifyPass>();
}
