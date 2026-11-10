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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
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

/// Whether `op` is a loop meant to run forever.
static bool runsForever(Operation *op) {
  auto whileOp = dyn_cast<scf::WhileOp>(op);
  auto forOp = dyn_cast<scf::ForOp>(op);
  std::optional<int64_t> trip =
      forOp ? getStaticTripCount(forOp) : std::nullopt;
  return (whileOp && isForeverLoop(whileOp)) || (trip && *trip >= kRunsForever);
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

/// The trip count of `forOp`: static, or an upper bound the host writes as
/// one constant for every dispatch. That is a word of an `aie.buffer` of
/// 32-bit words the cores only load, which every runtime sequence of the
/// device writes with the same constant (`aiex.npu.rtp_write`), directly in
/// its body, before it sets a lock (`aiex.set_lock`) the core waits on ahead
/// of the load. The lock starts at 0 and each release gives back no more
/// than the acquire just before it took, so every load follows a write.
static std::optional<int64_t> tripCount(scf::ForOp forOp, DeviceOp device) {
  if (std::optional<int64_t> trip = getStaticTripCount(forOp))
    return trip;
  std::optional<int64_t> lower = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  Value bound = forOp.getUpperBound();
  if (auto cast = bound.getDefiningOp<arith::IndexCastOp>())
    bound = cast.getIn();
  auto load = bound.getDefiningOp<memref::LoadOp>();
  if (!lower || !step || *step <= 0 || !load || load.getIndices().size() != 1)
    return std::nullopt;
  std::optional<int64_t> index = getConstantIntValue(load.getIndices().front());
  auto buffer = load.getMemRef().getDefiningOp<BufferOp>();
  // An RTP write's index counts 32-bit words, so only in a buffer of them is
  // that the element the load reads.
  auto type = buffer ? dyn_cast<MemRefType>(buffer.getType()) : MemRefType();
  if (!index || !type || !type.getElementType().isInteger(32) ||
      !type.getLayout().isIdentity() || !buffer.getSymName() ||
      buffer.getInitialValue() ||
      llvm::any_of(buffer->getUsers(),
                   [](Operation *user) { return !isa<memref::LoadOp>(user); }))
    return std::nullopt;

  UseLockOp wait;
  for (Operation *prev = load->getPrevNode(); prev && !wait;
       prev = prev->getPrevNode())
    if (auto use = dyn_cast<UseLockOp>(prev); use && !use.release())
      wait = use;
  std::optional<int64_t> waitsFor =
      wait ? getConstantIntValue(wait.getValue()) : std::nullopt;
  LockOp lock = wait ? wait.getLock().getDefiningOp<LockOp>() : LockOp();
  if (!lock || !wait.acquireGE() || !waitsFor || *waitsFor < 1 ||
      !wait.getAcqEn() ||
      wait.getBlocking().value_or(LockBlocking::Blocking) !=
          LockBlocking::Blocking ||
      lock.getInit().value_or(0) != 0)
    return std::nullopt;
  for (Operation *user : lock->getUsers()) {
    if (isa<AIEX::SetLockOp>(user))
      continue;
    auto use = dyn_cast<UseLockOp>(user);
    if (!use)
      return std::nullopt;
    if (!use.release())
      continue;
    std::optional<int64_t> given = getConstantIntValue(use.getValue());
    UseLockOp before;
    for (Operation *prev = use->getPrevNode(); prev && !before;
         prev = prev->getPrevNode())
      if (auto other = dyn_cast<UseLockOp>(prev);
          other && other.getLock() == use.getLock())
        before = other;
    std::optional<int64_t> taken = before && !before.release()
                                       ? getConstantIntValue(before.getValue())
                                       : std::nullopt;
    if (!given || !taken || *given > *taken)
      return std::nullopt;
  }

  TileOp tile = buffer.getTileOp();
  const AIETargetModel &tm = device.getTargetModel();
  auto reachesTile = [&](FlatSymbolRefAttr named, std::optional<int32_t> col,
                         std::optional<int32_t> row,
                         std::optional<int64_t> address) {
    if (named)
      return named.getValue() == *buffer.getSymName();
    if (!col || !row) {
      if (!address)
        return true;
      col = *address >> tm.getColumnShift();
      row = (*address >> tm.getRowShift()) &
            ((1 << (tm.getColumnShift() - tm.getRowShift())) - 1);
    }
    return *col == tile.getCol() && *row == tile.getRow();
  };

  std::optional<int64_t> value;
  for (auto sequence : device.getOps<RuntimeSequenceOp>()) {
    bool written = false;
    WalkResult walk = sequence.walk([&](Operation *op) {
      bool rawWrite =
          llvm::TypeSwitch<Operation *, bool>(op)
              .Case<AIEX::NpuWrite32Op, AIEX::NpuMaskWrite32Op>([&](auto w) {
                return reachesTile(w.getBufferAttr(), w.getColumn(), w.getRow(),
                                   getConstantIntValue(w.getAddress()));
              })
              .Case([&](AIEX::NpuBlockWriteOp w) {
                return reachesTile(w.getBufferAttr(), w.getColumn(), w.getRow(),
                                   w.getAddress());
              })
              .Case([&](AIEX::NpuPushQueueOp push) {
                return reachesTile({}, push.getColumn(), push.getRow(), {});
              })
              .Default(false);
      if (rawWrite)
        return WalkResult::interrupt();
      if (auto write = dyn_cast<AIEX::NpuWriteRTPOp>(op)) {
        if (write.getBuffer() != *buffer.getSymName() ||
            write.getIndex() != *index)
          return WalkResult::advance();
        // A write under a branch or a loop may never run.
        std::optional<int64_t> word = getConstantIntValue(write.getValue());
        if (write->getParentOp() != sequence || !word ||
            (value && *value != *word))
          return WalkResult::interrupt();
        value = word;
        written = true;
        return WalkResult::advance();
      }
      auto set = dyn_cast<AIEX::SetLockOp>(op);
      if (set && set.getLock() == lock.getResult() && !written)
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return std::nullopt;
  }
  if (!value)
    return std::nullopt;
  return *value > *lower ? (*value - *lower + *step - 1) / *step : 0;
}

/// The ratio (in, out) such that a core, once it has released r objects of
/// `out`, has taken at least r * in / out objects of `in`, at every point.
///
/// Each region of the core is summarized by two sets of (taken, released)
/// pairs: for a ratio k, the smallest taken - k * released over `end` bounds
/// that value where the region ends, and over `low` at every point inside it,
/// on every path. A branch may take either arm and a loop whose trip count
/// `tripCount` cannot tell may run any number of times, so the ratio holds
/// whatever decides them. Unknown when a branch's arms, or the passes of a
/// loop whose count is unknown, leave different objects held, or when `in` or
/// `out` is used under any other region.
class TakeRatio {
public:
  TakeRatio(DeviceOp device, StringRef in, StringRef out)
      : device(device), in(in), out(out) {}

  std::optional<std::pair<int64_t, int64_t>> of(CoreOp core) {
    core.walk([&](Operation *op) {
      if (fifoOf(op) != in && fifoOf(op) != out)
        return;
      for (Operation *parent = op->getParentOp(); parent != core;
           parent = parent->getParentOp())
        onUsePath.insert(parent);
      used = true;
    });
    Region &body = core.getBody();
    if (!used || !body.hasOneBlock())
      return std::nullopt;
    Held held;
    std::optional<Effect> effect = analyze(body.front(), held);
    if (!effect)
      return std::nullopt;
    std::optional<Pair> least;
    for (const Pair &pair : llvm::concat<const Pair>(effect->low, bounds))
      if (pair.released > 0 && (!least || pair.taken * least->released <
                                              least->taken * pair.released))
        least = pair;
    if (!least || least->taken == 0)
      return std::nullopt;
    int64_t common = std::gcd(least->taken, least->released);
    return std::make_pair(least->taken / common, least->released / common);
  }

private:
  struct Pair {
    int64_t taken, released;
  };
  struct Effect {
    SmallVector<Pair> end = {{0, 0}}, low = {{0, 0}};
  };
  struct Held {
    int64_t in = 0, out = 0;
    bool operator==(const Held &other) const {
      return in == other.in && out == other.out;
    }
  };
  static constexpr size_t maxPairs = 64;
  static constexpr int64_t maxPeeledPasses = 4096;

  static StringRef fifoOf(Operation *op) {
    if (auto acquire = dyn_cast<ObjectFifoAcquireOp>(op))
      return acquire.getObjFifoName();
    if (auto release = dyn_cast<ObjectFifoReleaseOp>(op))
      return release.getObjFifoName();
    return {};
  }

  // Keeps the pairs no other bounds from below: one with no more taken and no
  // fewer released gives a smaller taken - k * released for every k >= 0.
  std::optional<SmallVector<Pair>> prune(ArrayRef<Pair> pairs) {
    SmallVector<Pair> kept;
    for (auto [index, pair] : llvm::enumerate(pairs)) {
      bool dominated = false;
      for (auto [other, by] : llvm::enumerate(pairs))
        dominated |= other != index && by.taken <= pair.taken &&
                     by.released >= pair.released &&
                     (by.taken != pair.taken || by.released != pair.released ||
                      other < index);
      if (!dominated)
        kept.push_back(pair);
    }
    if (kept.size() > maxPairs)
      return std::nullopt;
    return kept;
  }

  std::optional<SmallVector<Pair>> sum(ArrayRef<Pair> a, ArrayRef<Pair> b) {
    SmallVector<Pair> sums;
    for (const Pair &x : a)
      for (const Pair &y : b) {
        Pair both;
        if (llvm::AddOverflow(x.taken, y.taken, both.taken) ||
            llvm::AddOverflow(x.released, y.released, both.released))
          return std::nullopt;
        sums.push_back(both);
      }
    return prune(sums);
  }

  std::optional<SmallVector<Pair>> scale(ArrayRef<Pair> pairs, int64_t n) {
    SmallVector<Pair> scaled;
    for (const Pair &pair : pairs) {
      Pair times;
      if (llvm::MulOverflow(pair.taken, n, times.taken) ||
          llvm::MulOverflow(pair.released, n, times.released))
        return std::nullopt;
      scaled.push_back(times);
    }
    return scaled;
  }

  std::optional<Effect> then(const Effect &first, const Effect &second) {
    std::optional<SmallVector<Pair>> end = sum(first.end, second.end);
    std::optional<SmallVector<Pair>> late = sum(first.end, second.low);
    if (!end || !late)
      return std::nullopt;
    std::optional<SmallVector<Pair>> low =
        prune(llvm::to_vector(llvm::concat<const Pair>(first.low, *late)));
    if (!low)
      return std::nullopt;
    return Effect{*end, *low};
  }

  std::optional<Effect> either(const Effect &a, const Effect &b) {
    std::optional<SmallVector<Pair>> end =
        prune(llvm::to_vector(llvm::concat<const Pair>(a.end, b.end)));
    std::optional<SmallVector<Pair>> low =
        prune(llvm::to_vector(llvm::concat<const Pair>(a.low, b.low)));
    if (!end || !low)
      return std::nullopt;
    return Effect{*end, *low};
  }

  // `n` >= 1 passes of `pass`, each starting with the objects it ends with
  // held: the value at a point of pass j is j times a pass's end plus a point
  // of one pass, smallest at j = 0 or j = n - 1.
  std::optional<Effect> repeat(const Effect &pass, int64_t n) {
    std::optional<SmallVector<Pair>> end = scale(pass.end, n);
    std::optional<SmallVector<Pair>> before = scale(pass.end, n - 1);
    std::optional<SmallVector<Pair>> late =
        before ? sum(*before, pass.low) : std::nullopt;
    if (!end || !late)
      return std::nullopt;
    std::optional<SmallVector<Pair>> low =
        prune(llvm::to_vector(llvm::concat<const Pair>(pass.low, *late)));
    std::optional<SmallVector<Pair>> ends = prune(*end);
    if (!low || !ends)
      return std::nullopt;
    return Effect{*ends, *low};
  }

  // Any number of passes: the ratio must keep each pass's end at or above
  // its start, recorded in `bounds`; then a point of pass j is no lower than
  // a point of one pass, and the end no lower than the start.
  Effect anyRepeat(const Effect &pass) {
    llvm::append_range(bounds, pass.end);
    Effect any;
    llvm::append_range(any.low, pass.low);
    return any;
  }

  std::optional<Effect> analyze(Block &block, Held &held) {
    Effect effect;
    for (Operation &op : block) {
      std::optional<Effect> step = analyzeOp(op, held);
      if (!step)
        return std::nullopt;
      std::optional<Effect> next = then(effect, *step);
      if (!next)
        return std::nullopt;
      effect = *next;
    }
    return effect;
  }

  std::optional<Effect> analyzeOp(Operation &op, Held &held) {
    StringRef fifo = fifoOf(&op);
    if (fifo == in || fifo == out) {
      int64_t &count = fifo == in ? held.in : held.out;
      Effect effect;
      if (auto acquire = dyn_cast<ObjectFifoAcquireOp>(op)) {
        int64_t wanted = acquire.acqNumber();
        if (fifo == in)
          effect.end = {{std::max<int64_t>(0, wanted - count), 0}};
        count = std::max(count, wanted);
        return effect;
      }
      int64_t released = cast<ObjectFifoReleaseOp>(op).relNumber();
      if (released > count)
        return std::nullopt;
      count -= released;
      if (fifo == out) {
        effect.end = {{0, released}};
        effect.low.push_back({0, released});
      }
      return effect;
    }
    if (!onUsePath.contains(&op))
      return Effect();
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      std::optional<int64_t> trip = tripCount(forOp, device);
      Held entry = held;
      if (!trip) {
        std::optional<Effect> pass = analyze(*forOp.getBody(), held);
        if (!pass || !(held == entry))
          return std::nullopt;
        return anyRepeat(*pass);
      }
      Effect total;
      for (int64_t done = 0; done < *trip; ++done) {
        Held before = held;
        std::optional<Effect> pass = analyze(*forOp.getBody(), held);
        std::optional<Effect> rest = pass;
        // Any number of passes covers this many, where counting overflows.
        if (pass && held == before) {
          rest = repeat(*pass, *trip - done);
          if (!rest)
            rest = anyRepeat(*pass);
        }
        std::optional<Effect> next = rest ? then(total, *rest) : std::nullopt;
        if (!next || (!(held == before) && done >= maxPeeledPasses))
          return std::nullopt;
        total = *next;
        if (held == before)
          break;
      }
      return total;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      Held otherwise = held;
      std::optional<Effect> taken = analyze(ifOp.getThenRegion().front(), held);
      std::optional<Effect> skipped =
          ifOp.getElseRegion().empty()
              ? Effect()
              : analyze(ifOp.getElseRegion().front(), otherwise);
      if (!taken || !skipped || !(held == otherwise))
        return std::nullopt;
      return either(*taken, *skipped);
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      Held entry = held;
      std::optional<Effect> check = analyze(*whileOp.getBeforeBody(), held);
      Held afterCheck = held;
      std::optional<Effect> body =
          check ? analyze(*whileOp.getAfterBody(), held) : std::nullopt;
      if (!body || !(held == entry))
        return std::nullopt;
      held = afterCheck;
      std::optional<Effect> pass = then(*body, *check);
      return pass ? then(*check, anyRepeat(*pass)) : std::nullopt;
    }
    return std::nullopt;
  }

  DeviceOp device;
  StringRef in, out;
  llvm::SmallPtrSet<Operation *, 16> onUsePath;
  bool used = false;
  // Pass ends of loops that may run any number of times.
  SmallVector<Pair> bounds;
};

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
  /// `aiex.dma_channel_reset_for` re-arms in every runtime sequence that moves
  /// data through it gets its locks back each launch.
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
        auto rearms =
            fifo ? poolLocks->rearms().find(*fifo) : poolLocks->rearms().end();
        // The sequences that fill or drain the fifo by name must each re-arm
        // it; where none names it (a link fills it), every sequence must.
        SmallVector<RuntimeSequenceOp> launches;
        for (auto seq : device.getOps<RuntimeSequenceOp>()) {
          auto uses = SymbolTable::getSymbolUses(seq.getOperation());
          if (fifo && uses && llvm::any_of(*uses, [&](auto use) {
                auto end = dyn_cast_or_null<RouteEndpoint>(
                    SymbolTable::lookupNearestSymbolFrom(seq,
                                                         use.getSymbolRef()));
                return end && end.getFifoName() == *fifo;
              }))
            launches.push_back(seq);
        }
        if (launches.empty())
          llvm::append_range(launches, device.getOps<RuntimeSequenceOp>());
        bool rearmedEveryLaunch =
            rearms != poolLocks->rearms().end() &&
            llvm::all_of(launches, [&](RuntimeSequenceOp seq) {
              return llvm::any_of(rearms->second, [&](auto reset) {
                return reset->getParentOp() == seq;
              });
            });
        if (pool && (!poolLocks->needsLocks(pool) || rearmedEveryLaunch))
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
  /// the whole configuration, is left alone, as is a loop the core may never
  /// reach.
  LogicalResult verifyAdvances(DeviceOp device) {
    // Whether `loop` runs, and repeats forever: it or a loop around it runs
    // forever, and nothing on the way from the core may skip it.
    auto repeatsForever = [](Operation *loop) {
      bool forever = false;
      for (Operation *op = loop; !isa<CoreOp>(op); op = op->getParentOp()) {
        Block *block = op->getBlock();
        if (block != &op->getParentRegion()->front() ||
            llvm::any_of(
                llvm::make_range(block->begin(), op->getIterator()),
                [&](Operation &before) { return runsForever(&before); }))
          return false;
        auto forOp = dyn_cast<scf::ForOp>(op);
        auto whileOp = dyn_cast<scf::WhileOp>(op);
        std::optional<int64_t> trip =
            forOp ? getStaticTripCount(forOp) : std::nullopt;
        if ((forOp && (!trip || *trip == 0)) ||
            (whileOp && !isForeverLoop(whileOp)))
          return false;
        forever |= runsForever(op);
        Operation *parent = op->getParentOp();
        if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
          bool inThen = op->getParentRegion() == &ifOp.getThenRegion();
          if (!matchPattern(ifOp.getCondition(), inThen ? m_One() : m_Zero()))
            return false;
        } else if (auto parentWhile = dyn_cast<scf::WhileOp>(parent)) {
          if (op->getParentRegion() != &parentWhile.getAfter())
            return false;
        } else if (!isa<scf::ForOp, CoreOp>(parent)) {
          return false;
        }
      }
      return forever;
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
  /// every use of it is a DMA task whose starts, repeats and lengths are
  /// static.
  static std::optional<int64_t> hostBytes(RuntimeSequenceOp sequence,
                                          StringAttr name) {
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(name, &sequence.getBody());
    if (!uses)
      return std::nullopt;
    std::optional<int64_t> bytes = 0;
    for (const SymbolTable::SymbolUse &use : *uses) {
      auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(use.getUser());
      std::optional<int64_t> runs =
          task ? staticExecutions(task) : std::nullopt;
      std::optional<int64_t> streamed =
          runs ? task.getStaticBytesStreamed() : std::nullopt;
      if (!streamed)
        return std::nullopt;
      std::optional<int64_t> moved = llvm::checkedMul(*runs, *streamed);
      bytes = moved ? llvm::checkedAdd(*bytes, *moved) : std::nullopt;
      if (!bytes)
        return std::nullopt;
    }
    return bytes;
  }

  /// Bytes one object of `endpoint` carries over its channel, its elements
  /// sized as a buffer descriptor's length counts them.
  static int64_t objectBytes(ObjectFifoDmaEndpointOp endpoint) {
    auto type = cast<MemRefType>(endpoint.getPoolOp().getElemType());
    return transferElements(endpoint) *
           static_cast<int64_t>(DataLayout::closest(endpoint).getTypeSize(
               type.getElementType()));
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

  /// A Worker whose forever loop waits once for each dispatch's parameters
  /// runs one pass of that loop per dispatch, so the objects one pass moves
  /// through a fifo must be the objects one run of the runtime sequence moves
  /// through it, or the core drifts out of step with the dispatches. A pass
  /// runs the fifo's period, an inner loop where that is the innermost loop
  /// around its accesses, as often as the static loops between them repeat.
  /// Checked when one runtime sequence syncs the parameters, so every
  /// dispatch runs it, and its transfers on the fifo are static. A sync sets
  /// the core's lock rather than adding to it, so where a run syncs twice the
  /// core may take only one, and nothing is checked.
  LogicalResult verifyDispatchVolumes(DeviceOp device) {
    RuntimeSequenceOp sequence;
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
      if (sequence || count > 1)
        return success();
      sequence = candidate;
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
      Operation *loop = turns.periodLoop();
      std::optional<int64_t> periods = 1;
      for (; periods && !isa<CoreOp>(loop) && !runsForever(loop);
           loop = loop->getParentOp()) {
        auto forOp = dyn_cast<scf::ForOp>(loop);
        std::optional<int64_t> trip =
            forOp ? getStaticTripCount(forOp) : std::nullopt;
        periods = trip && loop->getBlock() == &loop->getParentRegion()->front()
                      ? llvm::checkedMul(*periods, *trip)
                      : std::nullopt;
      }
      if (!periods || isa<CoreOp>(loop))
        continue;
      // Two waits a pass serve two dispatches, and a wait under a branch or an
      // inner loop may serve none or several.
      auto whileLoop = dyn_cast<scf::WhileOp>(loop);
      Block *body = whileLoop ? whileLoop.getAfterBody()
                              : cast<scf::ForOp>(loop).getBody();
      int64_t waits = 0;
      loop->walk([&](AIEX::WaitScratchpadParametersOp) { waits++; });
      if (waits != 1 ||
          body->getOps<AIEX::WaitScratchpadParametersOp>().empty())
        continue;
      std::optional<int64_t> perDispatch = llvm::checkedMul(
          *periods, static_cast<int64_t>(turns.turns().size()));
      if (!perDispatch)
        continue;
      std::optional<int64_t> objects = hostObjects(device, sequence, dma, true);
      LLVM_DEBUG(llvm::dbgs()
                 << "dispatch volume of @" << endpoint.getSymName()
                 << ": the core moves " << *perDispatch << ", @"
                 << sequence.getSymName() << " moves "
                 << (objects ? std::to_string(*objects) : "an unknown count")
                 << "\n");
      if (!objects || *objects == *perDispatch)
        continue;

      ObjectFifoAcquireOp acquire;
      loop->walk([&](ObjectFifoAcquireOp op) {
        if (!acquire && op.getObjFifoName() == endpoint.getSymName())
          acquire = op;
      });
      if (!acquire)
        continue;
      InFlightDiagnostic diag =
          acquire.emitOpError()
          << (endpoint.drains() ? "takes " : "hands over ") << *perDispatch
          << " objects through @" << endpoint.getSymName()
          << " for each dispatch, one pass of a loop that waits for each "
             "dispatch's parameters, but each run of runtime sequence @"
          << sequence.getSymName() << " moves " << *objects
          << ", so the core drifts out of step with the dispatches";
      diag.attachNote(sequence.getLoc()) << "the runtime sequence is here";
      return failure();
    }
    return success();
  }

  using PoolActors = DenseMap<Operation *, SmallVector<SegmentActors>>;

  /// A place the bytes some endpoint sends reach: once `far` has taken some
  /// of them, the endpoint has sent at least `inBytes` for every `outBytes`
  /// taken. With a nonzero `lagObject`, they pass a later segment of a join
  /// on the way, whose drainer takes `lagObject` bytes an object and has taken
  /// at least `lagIn` for every `lagOut` `far` took: that holds in whole
  /// objects of the join only (see recordConsumption's granule).
  struct Reach {
    Operation *far;
    int64_t inBytes, outBytes;
    int64_t lagObject = 0, lagIn = 0, lagOut = 0;
  };

  static int64_t segmentBytes(ObjectFifoPoolOp pool,
                              ArrayRef<ObjectFifoSegmentOp> segments) {
    int64_t elements = 0;
    for (ObjectFifoSegmentOp segment : segments)
      elements += segment.getSize();
    auto type = cast<MemRefType>(pool.getElemType());
    return elements *
           static_cast<int64_t>(
               DataLayout::closest(pool).getTypeSize(type.getElementType()));
  }

  /// Whether `dma` moves each object's bytes once, neither padding, repeating
  /// nor dispatching them.
  static bool passesOn(ObjectFifoDmaEndpointOp dma) {
    return !dma.getPadDimensions() && !dma.getDispatch() &&
           dma.getRepeat() == 1;
  }

  /// Where the bytes the endpoint `from` sends end up, scaled by `scale`: a
  /// shim's route endpoint, or a core endpoint draining a pool on the way.
  /// Every destination of a route is reached. A pool is entered through a
  /// fill that alone fills its segments and left through each drainer of the
  /// same segments, of the fill's last segment alone (a split), or of all
  /// segments when the fill fills the first (a join): segments release once
  /// written, so only those drainers' progress bounds the whole of the fill's.
  /// Bytes pass only through DMAs that neither pad, repeat nor dispatch them,
  /// counted as each DMA's transform moves them, and only to a destination no
  /// other route merges into, with no packet header kept in its bytes.
  void carries(DeviceOp device, StringAttr from, const PoolActors &actors,
               Reach scale, llvm::SmallPtrSetImpl<Operation *> &onPath,
               SmallVectorImpl<Reach> &reaches) {
    RouteOp route;
    for (auto candidate : device.getOps<RouteOp>()) {
      if (!candidate.namesAsSource(from.getValue()))
        continue;
      if (route)
        return;
      route = candidate;
    }
    if (!route || route.getSources().size() != 1 || route.getKeepPktHeader())
      return;
    for (StringRef name : route.getDestinationNames()) {
      if (llvm::any_of(device.getOps<RouteOp>(), [&](RouteOp other) {
            return other != route && other.namesAsDestination(name);
          }))
        continue;
      Operation *dest = SymbolTable::lookupNearestSymbolFrom(
          device, StringAttr::get(device.getContext(), name));
      if (auto shim = dyn_cast_or_null<RouteEndpointOp>(dest)) {
        Reach at = scale;
        at.far = shim;
        if (shim.getBundle() == WireBundle::DMA &&
            shim.getTileLike().isShimTile())
          reaches.push_back(at);
        continue;
      }
      auto fill = dyn_cast_or_null<ObjectFifoDmaEndpointOp>(dest);
      if (!fill || fill.drains() || !passesOn(fill))
        continue;
      ObjectFifoPoolOp pool = fill.getPoolOp();
      if (!pool || pool.getInitValues() || pool.getContiguous() ||
          !poolLocks->needsLocks(pool) || onPath.contains(pool))
        continue;
      auto it = actors.find(pool);
      if (it == actors.end())
        continue;
      std::vector<ObjectFifoSegmentOp> all = pool.getSegmentOps();
      std::vector<ObjectFifoSegmentOp> filled = fill.getSelectedSegments();
      auto indexOf = [&](ObjectFifoSegmentOp segment) {
        return llvm::find(all, segment) - all.begin();
      };
      if (filled.empty() || llvm::any_of(filled, [&](auto segment) {
            return it->second[indexOf(segment)].fillers.size() != 1;
          }))
        continue;
      llvm::SetVector<Operation *> drainers;
      for (const SegmentActors &segment : it->second)
        drainers.insert(segment.drainers.begin(), segment.drainers.end());
      onPath.insert(pool);
      for (Operation *drainer : drainers) {
        auto core = dyn_cast<ObjectFifoCoreEndpointOp>(drainer);
        auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(drainer);
        if (dma && !passesOn(dma))
          continue;
        std::vector<ObjectFifoSegmentOp> drained =
            core ? core.getSelectedSegments() : dma.getSelectedSegments();
        if (drained.empty())
          continue;
        bool split = drained.size() == 1 && drained.front() == filled.back();
        bool join = filled.size() == 1 && filled.front() == drained.front();
        // A later segment of a join is taken only after those ahead of it in
        // each object (a join's drainer reads its segments in order, any
        // transform staying within each), so its progress shows in whole
        // objects alone; one such segment per path.
        bool laterJoin = dma && filled.size() == 1 && !join &&
                         llvm::is_contained(drained, filled.front());
        if ((drained != filled && !split && !join && !laterJoin) ||
            (laterJoin && scale.lagObject))
          continue;
        int64_t in = objectBytes(fill);
        int64_t out = core ? segmentBytes(pool, drained) : objectBytes(dma);
        Reach next = scale;
        next.far = drainer;
        if (in <= 0 || out <= 0 ||
            llvm::MulOverflow(scale.inBytes, in, next.inBytes) ||
            llvm::MulOverflow(scale.outBytes, out, next.outBytes) ||
            (scale.lagObject &&
             (llvm::MulOverflow(scale.lagIn, in, next.lagIn) ||
              llvm::MulOverflow(scale.lagOut, out, next.lagOut))))
          continue;
        int64_t common = std::gcd(next.inBytes, next.outBytes);
        next.inBytes /= common;
        next.outBytes /= common;
        if (laterJoin)
          next = {next.far, next.inBytes, next.outBytes, out, 1, 1};
        if (next.lagObject) {
          int64_t lagCommon = std::gcd(next.lagIn, next.lagOut);
          next.lagIn /= lagCommon;
          next.lagOut /= lagCommon;
        }
        if (core)
          reaches.push_back(next);
        else
          carries(device, dma.getSymNameAttr(), actors, next, onPath, reaches);
      }
      onPath.erase(pool);
    }
  }

  /// Records on each shim input the shim outputs whose progress proves it
  /// consumed (#aie.consumed_by). Its bytes reach such an output; or they
  /// reach a core that, on every path through its program (TakeRatio), takes
  /// its share of them before each release into a fifo whose bytes reach the
  /// output, or another such core. A shim input sends a byte before anything
  /// downstream can take it, and the host receives a byte only after the
  /// cores released it, so once the output has received R bytes the input's
  /// first R * in / out have been sent; with a granule (see carries), R
  /// rounded down to a multiple of it.
  void recordConsumption(DeviceOp device, const PoolActors &actors) {
    MLIRContext *ctx = device.getContext();
    struct Share {
      int64_t in, out, granule;
    };
    for (auto input : device.getOps<RouteEndpointOp>()) {
      if (input.getBundle() != WireBundle::DMA ||
          !input.getTileLike().isShimTile())
        continue;
      // Per output, one proof: one without a granule where there is one, else
      // the most bytes for every `out` received, then the smallest granule.
      llvm::MapVector<Operation *, Share> proofs;
      auto prove = [&](RouteEndpointOp output, const Reach &reach) {
        Share share{reach.inBytes, reach.outBytes, 0};
        if (reach.lagObject) {
          // The smallest R with R * lagIn a multiple of lagOut * lagObject.
          int64_t whole;
          if (llvm::MulOverflow(reach.lagOut, reach.lagObject, whole))
            return;
          share.granule = whole / std::gcd(whole, reach.lagIn);
        }
        auto [it, inserted] = proofs.try_emplace(output, share);
        const Share &old = it->second;
        int64_t lhs, rhs;
        if (inserted || llvm::MulOverflow(share.in, old.out, lhs) ||
            llvm::MulOverflow(old.in, share.out, rhs))
          return;
        if ((old.granule && !share.granule) ||
            (!old.granule == !share.granule &&
             (lhs > rhs || (lhs == rhs && share.granule < old.granule))))
          it->second = share;
      };
      SmallVector<Reach> reaches;
      llvm::SmallPtrSet<Operation *, 8> onPath;
      carries(device, input.getSymNameAttr(), actors, {nullptr, 1, 1}, onPath,
              reaches);
      // A core passes the input on through each fifo it fills, perhaps to
      // another core, and so on, a few hops deep.
      constexpr int maxCoreHops = 4;
      SmallVector<std::pair<Reach, int>> work;
      for (const Reach &reach : reaches)
        work.push_back({reach, 0});
      while (!work.empty()) {
        auto [reach, hops] = work.pop_back_val();
        if (auto output = dyn_cast<RouteEndpointOp>(reach.far)) {
          prove(output, reach);
          continue;
        }
        if (hops == maxCoreHops)
          continue;
        auto consumer = cast<ObjectFifoCoreEndpointOp>(reach.far);
        CoreOp core;
        for (Operation *user : consumer.getTile().getUsers())
          if (auto candidate = dyn_cast<CoreOp>(user))
            core = candidate;
        if (!core)
          continue;
        int64_t takenBytes =
            segmentBytes(consumer.getPoolOp(), consumer.getSelectedSegments());
        for (auto producer : device.getOps<ObjectFifoCoreEndpointOp>()) {
          ObjectFifoPoolOp pool = producer.getPoolOp();
          if (producer.drains() || producer.getSegments() ||
              producer.getTile() != consumer.getTile() || !pool ||
              pool.getInitValues() || !poolLocks->needsLocks(pool))
            continue;
          auto it = actors.find(pool);
          if (it == actors.end() || it->second.size() != 1 ||
              it->second.front().fillers.size() != 1 ||
              it->second.front().drainers.size() != 1)
            continue;
          Operation *drainer = it->second.front().drainers.front();
          std::optional<std::pair<int64_t, int64_t>> ratio =
              TakeRatio(device, consumer.getSymName(), producer.getSymName())
                  .of(core);
          if (!ratio)
            continue;
          // Once the next place has taken X bytes, the drainer sent X *
          // next.inBytes / next.outBytes, the core released at least that
          // much of the producer and so took `taken` bytes of the consumer
          // for every `released`, of which the input sent reach's share.
          // `released` counts what the drainer moves of each object.
          auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(drainer);
          int64_t taken, released;
          if ((dma && !passesOn(dma)) ||
              llvm::MulOverflow(ratio->first, takenBytes, taken) ||
              llvm::MulOverflow(
                  ratio->second,
                  dma ? objectBytes(dma)
                      : segmentBytes(pool, producer.getSelectedSegments()),
                  released) ||
              released <= 0)
            continue;
          // A core sharing the pool's memory takes what this one released.
          SmallVector<Reach> nexts;
          if (!dma) {
            nexts.push_back({drainer, 1, 1, 0, 0, 0});
          } else {
            llvm::SmallPtrSet<Operation *, 8> outPath;
            carries(device, dma.getSymNameAttr(), actors, {nullptr, 1, 1},
                    outPath, nexts);
          }
          for (const Reach &next : nexts) {
            if (reach.lagObject && next.lagObject)
              continue;
            Reach composed = next;
            if (llvm::MulOverflow(next.inBytes, taken, composed.inBytes) ||
                llvm::MulOverflow(composed.inBytes, reach.inBytes,
                                  composed.inBytes) ||
                llvm::MulOverflow(next.outBytes, released, composed.outBytes) ||
                llvm::MulOverflow(composed.outBytes, reach.outBytes,
                                  composed.outBytes))
              continue;
            int64_t common = std::gcd(composed.inBytes, composed.outBytes);
            composed.inBytes /= common;
            composed.outBytes /= common;
            // A join before the core: its drainer took at least what the
            // core took, scaled on to the next place as the input's share is.
            if (reach.lagObject) {
              composed.lagObject = reach.lagObject;
              if (llvm::MulOverflow(reach.lagIn, taken, composed.lagIn) ||
                  llvm::MulOverflow(composed.lagIn, next.inBytes,
                                    composed.lagIn) ||
                  llvm::MulOverflow(reach.lagOut, released, composed.lagOut) ||
                  llvm::MulOverflow(composed.lagOut, next.outBytes,
                                    composed.lagOut))
                continue;
              int64_t lagCommon = std::gcd(composed.lagIn, composed.lagOut);
              composed.lagIn /= lagCommon;
              composed.lagOut /= lagCommon;
            }
            work.push_back({composed, hops + 1});
          }
        }
      }
      if (proofs.empty()) {
        input.removeConsumedByAttr();
        continue;
      }
      SmallVector<Attribute> attrs;
      for (auto [output, share] : proofs) {
        int64_t common = std::gcd(share.in, share.out);
        attrs.push_back(ConsumedByAttr::get(
            ctx,
            FlatSymbolRefAttr::get(
                cast<RouteEndpointOp>(output).getSymNameAttr()),
            share.in / common, share.out / common, share.granule,
            /*device=*/FlatSymbolRefAttr()));
      }
      input.setConsumedByAttr(ArrayAttr::get(ctx, attrs));
    }
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
      if (flow.getSources().size() != 1)
        continue;
      ObjectFifoDmaEndpointOp src =
          lookup(cast<FlatSymbolRefAttr>(flow.getSources()[0]));
      ObjectFifoDmaEndpointOp firstReceiver;
      for (auto destName :
           flow.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
        ObjectFifoDmaEndpointOp dst = lookup(destName);
        if (!dst || !dst.getOutOfOrder())
          continue;
        if (!firstReceiver)
          firstReceiver = dst;
        if (firstReceiver.getNumDescriptors() != dst.getNumDescriptors() ||
            objectBytes(firstReceiver.getPoolOp()) !=
                objectBytes(dst.getPoolOp()))
          return flow.emitOpError(
              "out-of-order broadcast destinations require matching buffer "
              "counts and object sizes");
        if (!src || !src.getOutOfOrder())
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
        failed(verifyDispatchVolumes(device)) ||
        failed(verifyOutOfOrderPools(device))) {
      return signalPassFailure();
    }
    recordConsumption(device, actorsPerPool);
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoVerifyPass() {
  return std::make_unique<AIEObjectFifoVerifyPass>();
}
