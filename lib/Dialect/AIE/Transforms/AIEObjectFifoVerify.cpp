//===- AIEObjectFifoVerify.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEChannelTurns.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPoolLocks.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-objectfifo-verify"

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

/// IRON's forever loop is an `scf.while` on a constant true condition. A
/// hand-written `scf.for` meant to run forever has a bound no design counts
/// to, such as sys.maxsize or 2^32 - 1. From the AIE's 24-bit BD-loop
/// saturation value up, a trip count is read as "forever"; below it, as a real
/// bound.
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

/// How many times `op` runs when its core, or runtime sequence, runs to
/// completion, when static loop bounds alone decide that. Anything conditional
/// between the op and the core, an `scf.if`, a loop with dynamic bounds or one
/// meant to run forever, a block reached by branching, makes it unknown.
static std::optional<int64_t> staticExecutions(Operation *op) {
  int64_t count = 1;
  for (Operation *cur = op; !isa<CoreOp, RuntimeSequenceOp>(cur);
       cur = cur->getParentOp()) {
    if (cur->getBlock() != &cur->getParentRegion()->front()) {
      return std::nullopt;
    }
    Operation *parent = cur->getParentOp();
    if (isa<CoreOp, RuntimeSequenceOp>(parent)) {
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

/// Objects of the fifo `name` still held after `block` runs, entered holding
/// `held`: an acquire holds at least its count, a release gives its count
/// back. Unknown when an access sits under anything but a static loop, or a
/// loop's count has not settled after two passes.
static std::optional<int64_t> heldAfter(Block &block, StringRef name,
                                        int64_t held) {
  for (Operation &op : block) {
    if (auto acquire = dyn_cast<ObjectFifoAcquireOp>(op)) {
      if (acquire.getObjFifoName() == name)
        held = std::max<int64_t>(held, acquire.acqNumber());
      continue;
    }
    if (auto release = dyn_cast<ObjectFifoReleaseOp>(op)) {
      if (release.getObjFifoName() == name)
        held -= release.relNumber();
      if (held < 0)
        return std::nullopt;
      continue;
    }
    bool touches = op.getNumRegions() != 0 &&
                   op.walk([&](Operation *inner) {
                       auto acquire = dyn_cast<ObjectFifoAcquireOp>(inner);
                       auto release = dyn_cast<ObjectFifoReleaseOp>(inner);
                       if ((acquire && acquire.getObjFifoName() == name) ||
                           (release && release.getObjFifoName() == name))
                         return WalkResult::interrupt();
                       return WalkResult::advance();
                     }).wasInterrupted();
    if (!touches)
      continue;
    auto forOp = dyn_cast<scf::ForOp>(op);
    std::optional<int64_t> trip =
        forOp ? getStaticTripCount(forOp) : std::nullopt;
    if (!trip)
      return std::nullopt;
    if (*trip == 0)
      continue;
    std::optional<int64_t> once = heldAfter(*forOp.getBody(), name, held);
    if (!once)
      return std::nullopt;
    if (*trip > 1 && heldAfter(*forOp.getBody(), name, *once) != once)
      return std::nullopt;
    held = *once;
  }
  return held;
}

struct AIEObjectFifoVerifyPass
    : public xilinx::AIE::impl::AIEObjectFifoVerifyBase<
          AIEObjectFifoVerifyPass> {

  /// Which pools get locks. Without locks nothing waits on a release, and
  /// nothing a core holds stays taken.
  std::shared_ptr<PoolLocks> poolLocks;

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
  LogicalResult verifyFlows(DeviceOp device,
                            SymbolTableCollection &symbolTables) {
    DenseMap<StringRef, int> appearances;
    for (auto flow : device.getOps<RouteOp>()) {
      for (StringRef source : flow.getSourceNames()) {
        appearances[source]++;
      }
      for (StringRef dest : flow.getDestinationNames()) {
        appearances[dest]++;
      }
    }

    for (auto endpoint : device.getOps<RouteEndpoint>()) {
      int count =
          appearances.lookup(cast<SymbolOpInterface>(*endpoint).getName());
      if (count == 0) {
        return endpoint->emitOpError("is not connected by any flow");
      }
      if (auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(*endpoint);
          dma && dma.getDispatch()) {
        if (failed(verifyDispatch(device, dma, symbolTables)))
          return failure();
        continue;
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

  /// A dispatching endpoint is the one source of one packet-switched route per
  /// destination it takes turns between, and of nothing else. A second route to
  /// the same destination is refused there, by the count every endpoint gets.
  LogicalResult verifyDispatch(DeviceOp device,
                               ObjectFifoDmaEndpointOp endpoint,
                               SymbolTableCollection &symbolTables) {
    StringRef name = endpoint.getSymName();
    llvm::SmallSetVector<StringRef, 4> turns;
    for (StringRef dest :
         endpoint.getDispatchAttr().getAsValueRange<FlatSymbolRefAttr>())
      turns.insert(dest);
    llvm::SmallDenseSet<StringRef, 4> reached;
    for (auto flow : device.getOps<RouteOp>()) {
      if (flow.namesAsDestination(name))
        return flow.emitOpError("names dispatching endpoint @")
               << name << " as a destination";
      if (!flow.namesAsSource(name))
        continue;
      if (flow.getSources().size() != 1 || flow.getDestinations().size() != 1)
        return flow.emitOpError("leaves dispatching endpoint @")
               << name
               << " for one destination, so it has one source and "
                  "one destination";
      if (!flow.getPacket())
        return flow.emitOpError("leaves dispatching endpoint @")
               << name << " and needs a packet header to tell its turns apart";
      StringRef dest = *flow.getDestinationNames().begin();
      if (!turns.contains(dest))
        return flow.emitOpError("reaches @")
               << dest << ", which dispatching endpoint @" << name
               << " never takes a turn for";
      reached.insert(dest);
    }
    for (StringRef dest : turns)
      if (!reached.contains(dest))
        return endpoint.emitOpError("dispatches to @")
               << dest << ", but no route from it reaches there";

    // Every turn moves the same chain through the dispatching endpoint, one
    // object per turn, so each turn's destination must move as many elements
    // as the dispatching endpoint itself does.
    int64_t elements = transferElements(endpoint);
    for (StringRef dest : turns) {
      auto destEndpoint = symbolTables.lookupSymbolIn<ObjectFifoDmaEndpointOp>(
          device, StringAttr::get(device.getContext(), dest));
      if (!destEndpoint)
        continue;
      int64_t destElements = transferElements(destEndpoint);
      if (destElements != elements)
        return endpoint.emitOpError("dispatches, so every turn moves ")
               << elements << " elements, but @" << dest << " moves "
               << destElements;
    }
    return success();
  }

  /// The elements one pass of an endpoint's chain moves: each selected segment
  /// once, through its transform where it has one.
  static int64_t transferElements(ObjectFifoDmaEndpointOp endpoint) {
    int64_t elements = 0;
    std::optional<ArrayRef<BDDimLayoutArrayAttr>> dimensions =
        endpoint.getDimensions();
    for (auto [index, segment] :
         llvm::enumerate(endpoint.getSelectedSegments())) {
      BDDimLayoutArrayAttr dims;
      if (dimensions && index < dimensions->size()) {
        dims = (*dimensions)[index];
      }
      if (dims && !dims.empty()) {
        int64_t product = 1;
        for (BDDimLayoutAttr dim : dims) {
          product *= dim.getSize();
        }
        elements += product;
      } else {
        elements += segment.getSize();
      }
    }
    return elements;
  }

  /// A fan-in's destination fills its next object with whatever packet
  /// arrives, and a packet is one pass of its sender's chain, so every end has
  /// to move the same number of elements or two senders share an object. A
  /// route endpoint's transfers are the runtime's and are not checked here.
  LogicalResult verifyFanIn(DeviceOp device,
                            SymbolTableCollection &symbolTables) {
    for (auto flow : device.getOps<RouteOp>()) {
      if (!flow.isFanIn()) {
        continue;
      }
      std::optional<std::pair<StringRef, int64_t>> first;
      auto check = [&](StringRef name) -> LogicalResult {
        auto endpoint = symbolTables.lookupSymbolIn<ObjectFifoDmaEndpointOp>(
            device, StringAttr::get(device.getContext(), name));
        if (!endpoint) {
          return success();
        }
        int64_t elements = transferElements(endpoint);
        if (!first) {
          first = {name, elements};
          return success();
        }
        if (elements != first->second) {
          return flow.emitOpError("fans in, so every end moves one object of "
                                  "the same size, but @")
                 << first->first << " moves " << first->second
                 << " elements and @" << name << " moves " << elements;
        }
        return success();
      };
      for (StringRef name : flow.getSourceNames()) {
        if (failed(check(name))) {
          return failure();
        }
      }
      for (StringRef name : flow.getDestinationNames()) {
        if (failed(check(name))) {
          return failure();
        }
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

  /// A DMA endpoint with an iteration count stops after `iterCount * depth`
  /// objects, so a core that must release more through the same pool waits
  /// forever for the rest, or leaves objects it filled unsent. Only
  /// releases static loop bounds force are counted, so what is reported is
  /// certain; a conditional release, or one in a loop meant to run forever, is
  /// left out, since such a core stalling once the data stops is how those
  /// designs end. When all of a core's releases are counted, the DMA endpoint
  /// is bounded too: it cannot drain more than the core hands over, nor fill
  /// more than the pool's depth plus what the core releases.
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
    DenseSet<StringRef> uncounted;
    device.walk([&](ObjectFifoReleaseOp release) {
      StringRef key = release.getObjFifoName();
      std::optional<int64_t> runs = staticExecutions(release);
      if (!runs) {
        uncounted.insert(key);
        return;
      }
      released[key] += *runs * release.relNumber();
      blame.try_emplace(key, release);
    });
    DenseSet<StringRef> acquired;
    device.walk([&](ObjectFifoAcquireOp acquire) {
      acquired.insert(acquire.getObjFifoName());
    });

    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      // A segment-selecting endpoint sees a share of the pool, not the whole.
      auto budget = budgets.find(endpoint.getPoolOp());
      if (budget == budgets.end() || endpoint.getSegments()) {
        continue;
      }
      StringRef name = endpoint.getSymName();
      int64_t count = released.lookup(name);
      ObjectFifoDmaEndpointOp dma = budget->second.endpoint;
      ObjectFifoPoolOp pool = endpoint.getPoolOp();
      int64_t depth = pool.getDepth();
      if (count > budget->second.objects) {
        InFlightDiagnostic diag =
            blame.lookup(name).emitOpError("releases ")
            << count << " objects through @" << name
            << " over the run, but the pool's DMA endpoint stops after "
            << budget->second.objects << " (iterCount "
            << budget->second.iterations << " x depth " << depth << ")";
        // A producer finishes while the pool still has free objects to fill.
        if (!endpoint.drains() && count <= budget->second.objects + depth)
          diag << ", leaving " << count - budget->second.objects
               << " in the pool unsent";
        else
          diag << ", so the core would wait forever for the rest";
        diag.attachNote(dma.getLoc()) << "the DMA endpoint is here";
        return failure();
      }

      if (!acquired.contains(name) || uncounted.contains(name) ||
          dma.getSegments() || pool.getInitValues() ||
          dma.drains() == endpoint.drains() || !poolLocks->needsLocks(pool)) {
        continue;
      }
      int64_t reachable = endpoint.drains() ? count + depth : count;
      if (budget->second.objects <= reachable) {
        continue;
      }
      InFlightDiagnostic diag =
          dma.emitOpError(dma.drains() ? "drains " : "fills ")
          << budget->second.objects << " objects (iterCount "
          << budget->second.iterations << " x depth " << depth
          << ") before it stops, but the core releases exactly " << count
          << " through @" << name;
      if (endpoint.drains())
        diag << ", leaving room for only " << reachable;
      diag << ", so the DMA endpoint would wait forever for the rest";
      diag.attachNote(endpoint.getLoc()) << "the core endpoint is here";
      return failure();
    }
    return success();
  }

  /// A core that reaches its end still holding objects never gives their locks
  /// back, and locks keep their counts until a PDI load, so the next run of
  /// the configuration finds those objects taken. Only cores certain to end
  /// are checked: one block, and no loop that may run forever. A fifo an
  /// `aiex.dma_channel_reset_for` re-arms gets its locks back each launch.
  LogicalResult verifyHeldAtEnd(DeviceOp device) {
    for (auto core : device.getOps<CoreOp>()) {
      Region &body = core.getBody();
      if (!body.hasOneBlock())
        continue;
      bool mayNotEnd =
          core.walk([](Operation *op) {
                auto forOp = dyn_cast<scf::ForOp>(op);
                std::optional<int64_t> trip =
                    forOp ? getStaticTripCount(forOp) : std::nullopt;
                if (isa<scf::WhileOp>(op) || (trip && *trip >= kRunsForever))
                  return WalkResult::interrupt();
                return WalkResult::advance();
              })
              .wasInterrupted();
      if (mayNotEnd)
        continue;
      llvm::MapVector<StringRef, ObjectFifoAcquireOp> firstAcquire;
      core.walk([&](ObjectFifoAcquireOp acquire) {
        firstAcquire.insert({acquire.getObjFifoName(), acquire});
      });
      for (auto &[name, acquire] : firstAcquire) {
        auto endpoint =
            SymbolTable::lookupNearestSymbolFrom<ObjectFifoCoreEndpointOp>(
                device, acquire.getObjFifoNameAttr());
        ObjectFifoPoolOp pool = endpoint ? endpoint.getPoolOp() : nullptr;
        std::optional<StringRef> fifo =
            pool ? pool.getFifoName() : std::nullopt;
        if (pool && (!poolLocks->needsLocks(pool) ||
                     (fifo && poolLocks->rearms().contains(*fifo))))
          continue;
        std::optional<int64_t> held = heldAfter(body.front(), name, 0);
        if (!held || *held == 0)
          continue;
        InFlightDiagnostic diag =
            core.emitOpError("ends with ")
            << *held << " acquired from @" << name
            << " never released, so their locks stay taken into the next run "
               "of this configuration";
        diag.attachNote(acquire.getLoc()) << "acquired here";
        return failure();
      }
    }
    return success();
  }

  /// A core whose loop repeats forever, and whose every repetition acquires
  /// objects of a pool but releases none, keeps the same objects for good. The
  /// end opposite it then stops once the pool is full, if it does not stop by
  /// itself first: checked where that end is a DMA endpoint without an
  /// iteration count, or a core that moves objects each time its own loop
  /// repeats forever. A core holding an object acquired before its loop, for
  /// the whole configuration, is left alone.
  LogicalResult verifyAdvances(DeviceOp device) {
    auto repeatsForever = [](Operation *loop) {
      for (Operation *op = loop; op && !isa<CoreOp>(op);
           op = op->getParentOp()) {
        auto whileOp = dyn_cast<scf::WhileOp>(op);
        auto forOp = dyn_cast<scf::ForOp>(op);
        std::optional<int64_t> trip =
            forOp ? getStaticTripCount(forOp) : std::nullopt;
        if ((whileOp && isForeverLoop(whileOp)) ||
            (trip && *trip >= kRunsForever))
          return true;
      }
      return false;
    };

    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      ObjectFifoPoolOp pool = endpoint.getPoolOp();
      if (!pool || pool.getInitValues() || endpoint.getSegments() ||
          !poolLocks->needsLocks(pool))
        continue;
      Operation *keepsMoving = nullptr;
      for (auto dma : device.getOps<ObjectFifoDmaEndpointOp>())
        if (dma.getPoolOp() == pool && dma.drains() != endpoint.drains() &&
            !dma.getIterCount() && !dma.getSegments())
          keepsMoving = dma;
      for (auto other : device.getOps<ObjectFifoCoreEndpointOp>()) {
        if (keepsMoving || other.getPoolOp() != pool ||
            other.drains() == endpoint.drains() || other.getSegments())
          continue;
        CoreTurns otherTurns;
        std::string otherReason;
        if (succeeded(otherTurns.read({other}, otherReason)) &&
            repeatsForever(otherTurns.periodLoop()))
          keepsMoving = other;
      }
      if (!keepsMoving)
        continue;

      CoreTurns turns;
      std::string reason;
      if (succeeded(turns.read({endpoint}, reason)) || !turns.movesNone() ||
          !repeatsForever(turns.periodLoop()))
        continue;
      auto acquire =
          dyn_cast_or_null<ObjectFifoAcquireOp>(turns.periodAcquire());
      if (!acquire)
        continue;
      InFlightDiagnostic diag =
          acquire.emitOpError("takes from @")
          << endpoint.getSymName()
          << " each time its loop repeats, but the loop releases none of "
             "them, so ";
      if (endpoint.drains())
        diag << "the core keeps the same objects forever and the end "
                "filling the pool stops once it is full";
      else
        diag << "nothing the core writes there is ever sent";
      diag.attachNote(keepsMoving->getLoc()) << "the other end is here";
      return failure();
    }
    return success();
  }

  /// Bytes one run of `sequence` moves through the shim endpoint `name`, when
  /// every use of it is a DMA task whose executions, repeats and lengths are
  /// static.
  static std::optional<int64_t> hostBytes(RuntimeSequenceOp sequence,
                                          StringAttr name) {
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(name, &sequence.getBody());
    if (!uses)
      return std::nullopt;
    int64_t bytes = 0;
    for (const SymbolTable::SymbolUse &use : *uses) {
      auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(use.getUser());
      std::optional<int64_t> runs =
          task ? staticExecutions(task) : std::nullopt;
      if (!runs || task.getRepeatCountVal())
        return std::nullopt;
      int64_t perRun = 0;
      WalkResult result = task.getBody().walk([&](DMABDOp bd) {
        if (bd.getLengthParameter() || !bd.getConstantLen())
          return WalkResult::interrupt();
        perRun += bd.getLenInBytes();
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return std::nullopt;
      bytes += *runs * (task.getRepeatCount() + 1) * perRun;
    }
    return bytes;
  }

  /// Bytes one object of `endpoint` carries over its channel.
  static int64_t objectBytes(ObjectFifoDmaEndpointOp endpoint) {
    auto type = cast<MemRefType>(endpoint.getPoolOp().getElemType());
    return transferElements(endpoint) * type.getElementTypeBitWidth() / 8;
  }

  /// Objects the DMA endpoint `endpoint` moves in one run of `sequence`, traced
  /// to the shim directly or through one pool with one DMA endpoint toward the
  /// shim.
  std::optional<int64_t> hostObjects(DeviceOp device,
                                     RuntimeSequenceOp sequence,
                                     ObjectFifoDmaEndpointOp endpoint,
                                     bool throughPool) {
    if (endpoint.getIterCount() || endpoint.getDispatch())
      return std::nullopt;
    StringRef name = endpoint.getSymName();
    SmallVector<StringRef> peers;
    for (auto route : device.getOps<RouteOp>()) {
      if (endpoint.drains() ? route.namesAsSource(name)
                            : route.namesAsDestination(name)) {
        auto names = endpoint.drains() ? route.getDestinationNames()
                                       : route.getSourceNames();
        peers.append(names.begin(), names.end());
      }
    }
    if (peers.size() != 1)
      return std::nullopt;
    Operation *peer = SymbolTable::lookupNearestSymbolFrom(
        device, StringAttr::get(device.getContext(), peers.front()));
    // A draining endpoint sends each object `getRepeat` times.
    int64_t sent = objectBytes(endpoint) * endpoint.getRepeat();
    int64_t perObject = endpoint.drains() ? sent : objectBytes(endpoint);

    int64_t bytes;
    if (auto shim = dyn_cast_or_null<RouteEndpointOp>(peer)) {
      std::optional<int64_t> host = hostBytes(sequence, shim.getSymNameAttr());
      if (!host)
        return std::nullopt;
      bytes = *host;
    } else if (auto other = dyn_cast_or_null<ObjectFifoDmaEndpointOp>(peer);
               other && throughPool && other.drains() != endpoint.drains()) {
      ObjectFifoDmaEndpointOp toward;
      for (auto candidate : device.getOps<ObjectFifoDmaEndpointOp>()) {
        if (candidate == other || candidate.getPoolOp() != other.getPoolOp() ||
            candidate.drains() == other.drains())
          continue;
        if (toward)
          return std::nullopt;
        toward = candidate;
      }
      if (!toward)
        return std::nullopt;
      std::optional<int64_t> objects =
          hostObjects(device, sequence, toward, false);
      if (!objects)
        return std::nullopt;
      bytes =
          *objects * (other.drains() ? objectBytes(other) * other.getRepeat()
                                     : objectBytes(other));
    } else {
      return std::nullopt;
    }
    if (perObject == 0 || bytes % perObject)
      return std::nullopt;
    return bytes / perObject;
  }

  /// A Worker whose forever loop waits for each dispatch's parameters runs
  /// one pass of that loop per dispatch, so the objects one pass moves through
  /// a fifo must be the objects one run of the runtime sequence moves through
  /// it, or the core drifts out of step with the dispatches. Checked when one
  /// runtime sequence syncs the parameters, so every dispatch runs it, and its
  /// transfers on the fifo are static.
  LogicalResult verifyDispatchVolumes(DeviceOp device) {
    RuntimeSequenceOp sequence;
    int64_t syncs = 0;
    for (auto candidate : device.getOps<RuntimeSequenceOp>()) {
      auto ops = candidate.getBody()
                     .getOps<AIEX::SyncScratchpadParametersFromHostOp>();
      int64_t count = std::distance(ops.begin(), ops.end());
      // With a core waiting, --aie-lower-scratchpad-parameters syncs once in a
      // sequence that has no sync of its own, unless the sequence opts out.
      if (count == 0) {
        BoolAttr preamble = candidate.getEmitParameterSyncPreambleAttr();
        count = !preamble || preamble.getValue();
      }
      if (count == 0)
        continue;
      if (sequence)
        return success();
      sequence = candidate;
      syncs = count;
    }
    if (!sequence)
      return success();

    for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>()) {
      ObjectFifoPoolOp pool = endpoint.getPoolOp();
      if (!pool || pool.getInitValues() || endpoint.getSegments() ||
          !poolLocks->needsLocks(pool))
        continue;
      ObjectFifoDmaEndpointOp dma;
      bool single = true;
      for (auto candidate : device.getOps<ObjectFifoDmaEndpointOp>()) {
        if (candidate.getPoolOp() != pool ||
            candidate.drains() == endpoint.drains())
          continue;
        single = !dma;
        dma = candidate;
      }
      if (!dma || !single || dma.getSegments())
        continue;

      CoreTurns turns;
      std::string reason;
      if (failed(turns.read({endpoint}, reason)))
        continue;
      auto loop = dyn_cast<scf::WhileOp>(turns.periodLoop());
      if (!loop || !isForeverLoop(loop) ||
          loop.getAfterBody()
              ->getOps<AIEX::WaitScratchpadParametersOp>()
              .empty())
        continue;
      std::optional<int64_t> objects = hostObjects(device, sequence, dma, true);
      int64_t perDispatch = syncs * static_cast<int64_t>(turns.turns().size());
      LLVM_DEBUG(llvm::dbgs()
                 << "dispatch volume of @" << endpoint.getSymName()
                 << ": the core moves " << perDispatch << ", @"
                 << sequence.getSymName() << " moves "
                 << (objects ? std::to_string(*objects) : "an unknown count")
                 << "\n");
      if (!objects || *objects == perDispatch)
        continue;

      ObjectFifoAcquireOp acquire;
      loop.walk([&](ObjectFifoAcquireOp op) {
        if (!acquire && op.getObjFifoName() == endpoint.getSymName())
          acquire = op;
      });
      if (!acquire)
        continue;
      InFlightDiagnostic diag =
          acquire.emitOpError()
          << (endpoint.drains() ? "takes " : "hands over ") << perDispatch
          << " objects through @" << endpoint.getSymName()
          << " for each dispatch, one pass of a loop that waits for each "
             "dispatch's parameters";
      if (syncs > 1)
        diag << " (synced " << syncs << " times per run)";
      diag << ", but each run of runtime sequence @" << sequence.getSymName()
           << " moves " << *objects
           << ", so the core drifts out of step with the dispatches";
      diag.attachNote(sequence.getLoc()) << "the runtime sequence is here";
      return failure();
    }
    return success();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    poolLocks = std::make_shared<PoolLocks>(device);

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

    SymbolTableCollection symbolTables;
    if (failed(verifyFlows(device, symbolTables)) ||
        failed(verifyFanIn(device, symbolTables)) ||
        failed(verifyOverRelease(device)) ||
        failed(verifyTransferBudgets(device)) ||
        failed(verifyHeldAtEnd(device)) || failed(verifyAdvances(device)) ||
        failed(verifyDispatchVolumes(device))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoVerifyPass() {
  return std::make_unique<AIEObjectFifoVerifyPass>();
}
