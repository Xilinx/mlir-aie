//===- AIEChannelTurns.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEChannelTurns.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <map>
#include <set>

#define DEBUG_TYPE "aie-channel-turns"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {
/// Events past this many in a period are not worth one chain.
constexpr size_t maxPeriodEvents = 4096;
/// Periods simulated before giving up on a steady state.
constexpr int maxPeriods = 256;
} // namespace

LogicalResult CoreTurns::read(ArrayRef<ObjectFifoCoreEndpointOp> ends,
                              std::string &reason) {
  assert(!ends.empty() && "turns between no endpoints");
  ObjectFifoCoreEndpointOp first = ends.front();
  Value tile = first.getTile();
  if (llvm::any_of(ends, [&](ObjectFifoCoreEndpointOp end) {
        return end.drains() != first.drains() || end.getTile() != tile;
      })) {
    reason = "the endpoints are not all one core's, one way";
    return failure();
  }
  CoreOp core;
  for (Operation *user : tile.getUsers())
    if (auto candidate = dyn_cast<CoreOp>(user))
      core = candidate;
  if (!core) {
    reason = "no core works through them";
    return failure();
  }
  SmallVector<StringAttr> names;
  for (ObjectFifoCoreEndpointOp end : ends)
    names.push_back(end.getSymNameAttr());
  return read(core, names, first.drains(), reason);
}

LogicalResult CoreTurns::read(CoreOp core, ArrayRef<StringAttr> names,
                              bool coreDrains, std::string &reason) {
  this->coreDrains = coreDrains;
  ends = names.size();
  DenseMap<StringAttr, unsigned> index;
  for (auto [i, name] : llvm::enumerate(names))
    index[name] = i;
  // An objectFIFO's acquires name its port; an endpoint's are one way.
  ObjectFifoPort port =
      coreDrains ? ObjectFifoPort::Consume : ObjectFifoPort::Produce;
  auto endOf = [&](Operation *op) -> std::optional<unsigned> {
    FlatSymbolRefAttr name;
    std::optional<ObjectFifoPort> used;
    if (auto acquire = dyn_cast<ObjectFifoAcquireOp>(op)) {
      name = acquire.getObjFifoNameAttr();
      used = acquire.getPort();
    } else if (auto release = dyn_cast<ObjectFifoReleaseOp>(op)) {
      name = release.getObjFifoNameAttr();
      used = release.getPort();
    }
    if (!name || (used && *used != port))
      return std::nullopt;
    auto it = index.find(name.getAttr());
    if (it == index.end())
      return std::nullopt;
    return it->second;
  };

  SmallVector<Operation *> uses;
  core.walk([&](Operation *op) {
    if (endOf(op))
      uses.push_back(op);
  });
  if (uses.empty()) {
    reason = "the core never acquires them";
    return failure();
  }
  // The period: the innermost loop around every use.
  auto loopsAround = [&](Operation *op) {
    SmallVector<Operation *> loops;
    for (Operation *parent = op->getParentOp(); parent != core;
         parent = parent->getParentOp())
      if (isa<LoopLikeOpInterface>(parent))
        loops.push_back(parent);
    return loops;
  };
  SmallVector<Operation *> candidates;
  for (Operation *candidate : loopsAround(uses.front()))
    if (llvm::all_of(uses, [&](Operation *use) {
          return candidate->isProperAncestor(use);
        }))
      candidates.push_back(candidate);
  if (candidates.empty()) {
    reason = "no loop repeats the core's use of them";
    return failure();
  }

  // The counters of the unrolled loops, so that branches on them fold.
  DenseMap<Value, Attribute> counters;
  std::function<Attribute(Value)> evaluate = [&](Value value) -> Attribute {
    Attribute attr = counters.lookup(value);
    Operation *def = value.getDefiningOp();
    if (attr || !def || matchPattern(value, m_Constant(&attr)))
      return attr;
    if (!isPure(def) || def->getNumResults() != 1)
      return {};
    SmallVector<Attribute> operands;
    for (Value operand : def->getOperands())
      if (!operands.emplace_back(evaluate(operand)))
        return {};
    SmallVector<OpFoldResult> folded;
    if (failed(def->fold(operands, folded)) || folded.size() != 1)
      return {};
    if (auto result = dyn_cast<Attribute>(folded.front()))
      return result;
    return evaluate(cast<Value>(folded.front()));
  };

  std::function<LogicalResult(Block &)> unroll = [&](Block &block) {
    for (Operation &op : block) {
      if (std::optional<unsigned> end = endOf(&op)) {
        if (auto acquire = dyn_cast<ObjectFifoAcquireOp>(op))
          period.push_back({*end, acquire.acqNumber(), true});
        else
          period.push_back(
              {*end, cast<ObjectFifoReleaseOp>(op).relNumber(), false});
        if (period.size() > maxPeriodEvents) {
          reason = "the core's period uses them too many times";
          return failure();
        }
        continue;
      }
      if (!llvm::any_of(
              uses, [&](Operation *use) { return op.isProperAncestor(use); }))
        continue;
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (auto taken =
                dyn_cast_or_null<BoolAttr>(evaluate(ifOp.getCondition()))) {
          for (Block &branch :
               taken.getValue() ? ifOp.getThenRegion() : ifOp.getElseRegion())
            if (failed(unroll(branch)))
              return failure();
          continue;
        }
      }
      auto forOp = dyn_cast<scf::ForOp>(op);
      std::optional<APInt> trips =
          forOp ? forOp.getStaticTripCount() : std::nullopt;
      if (!trips) {
        reason = "the core uses them under control flow that depends on data";
        return failure();
      }
      if (trips->getSExtValue() > static_cast<int64_t>(maxPeriodEvents)) {
        reason = "the core's period uses them too many times";
        return failure();
      }
      std::optional<int64_t> lower = getConstantIntValue(forOp.getLowerBound());
      std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
      for (int64_t trip = 0; trip < trips->getSExtValue(); trip++) {
        if (lower && step)
          counters[forOp.getInductionVar()] = IntegerAttr::get(
              forOp.getInductionVar().getType(), *lower + trip * *step);
        if (failed(unroll(*forOp.getBody())))
          return failure();
      }
    }
    return success();
  };
  // The innermost loop whose body unrolls, branches folded, is the period. An
  // scf.while's condition region holds no uses.
  Operation *loop = nullptr;
  for (Operation *candidate : candidates) {
    period.clear();
    counters.clear();
    if (llvm::all_of(cast<LoopLikeOpInterface>(candidate).getLoopRegions(),
                     [&](Region *region) {
                       return llvm::all_of(*region, [&](Block &block) {
                         return succeeded(unroll(block));
                       });
                     })) {
      loop = candidate;
      reason.clear();
      break;
    }
  }
  if (!loop)
    return failure();
  LLVM_DEBUG({
    llvm::dbgs() << "period of " << names.size() << " ends in "
                 << loop->getName() << " at " << loop->getLoc() << ": "
                 << period.size() << " events:";
    for (const Event &event : period)
      llvm::dbgs() << " " << (event.acquire ? "a" : "r") << event.end << "x"
                   << event.count;
    llvm::dbgs() << "\n";
  });

  // Objects the channel moves in a period: those each acquire takes beyond the
  // ones held, where the channel fills; those released, where it drains. The
  // first periods may differ while the core builds up what it holds, so the
  // order is that of the first period starting from a state seen before.
  SmallVector<int64_t> held(ends, 0);
  std::map<SmallVector<int64_t>, int> seen;
  for (int round = 0; round < maxPeriods; round++) {
    order.clear();
    SmallVector<int64_t> start = held;
    for (const Event &event : period) {
      int64_t &h = held[event.end];
      if (event.acquire) {
        if (coreDrains)
          order.append(std::max<int64_t>(0, event.count - h), event.end);
        h = std::max(h, event.count);
      } else {
        if (event.count > h) {
          reason = "the core releases objects it does not hold";
          return failure();
        }
        h -= event.count;
        if (!coreDrains)
          order.append(event.count, event.end);
      }
    }
    if (!seen.emplace(start, round).second) {
      if (order.empty()) {
        reason = "the core's period moves none of their objects";
        return failure();
      }
      return success();
    }
  }
  reason = "the core never settles into a steady order";
  return failure();
}

bool CoreTurns::keepsUp(ArrayRef<int64_t> depths) {
  assert(depths.size() == ends && "one depth per endpoint");
  // The shortest run of the period's order, in the same proportions, that the
  // channel can repeat, as a loop's steps do; the core's first and last steps
  // may move them in another order, which the pools' depths then absorb.
  SmallVector<int64_t> counts(ends, 0);
  for (unsigned end : order)
    counts[end]++;
  for (size_t length = 1; length < order.size(); length++) {
    if (order.size() % length)
      continue;
    size_t repeats = order.size() / length;
    if (llvm::any_of(counts, [&](int64_t c) { return c % repeats; }))
      continue;
    std::set<SmallVector<unsigned>> tried;
    for (size_t start = 0; start + length <= order.size(); start++) {
      SmallVector<unsigned> run(order.begin() + start,
                                order.begin() + start + length);
      SmallVector<int64_t> runCounts(ends, 0);
      for (unsigned end : run)
        runCounts[end]++;
      bool proportional = true;
      for (size_t i = 0; i < ends; i++)
        proportional &=
            runCounts[i] * static_cast<int64_t>(repeats) == counts[i];
      if (!proportional || !tried.insert(run).second || !keepsUp(run, depths))
        continue;
      order = run;
      return true;
    }
  }
  return keepsUp(order, depths);
}

bool CoreTurns::keepsUp(ArrayRef<unsigned> order,
                        ArrayRef<int64_t> depths) const {
  // Objects ready for the core (full where it drains, free where it fills)
  // and for the channel (the other kind); the core holds what it acquired.
  SmallVector<int64_t> forCore(ends, 0), forChannel(ends, 0), held(ends, 0);
  for (size_t i = 0; i < ends; i++)
    (coreDrains ? forChannel : forCore)[i] = depths[i];
  size_t turn = 0, event = 0;
  std::map<SmallVector<int64_t>, int> seen;
  for (int round = 0; round < maxPeriods * 4;) {
    bool moved = false;
    while (forChannel[order[turn]] > 0) {
      forChannel[order[turn]]--;
      forCore[order[turn]]++;
      turn = (turn + 1) % order.size();
      moved = true;
    }
    while (true) {
      const Event &e = period[event];
      if (e.acquire) {
        int64_t take = std::max<int64_t>(0, e.count - held[e.end]);
        if (forCore[e.end] < take)
          break;
        forCore[e.end] -= take;
        held[e.end] = std::max(held[e.end], e.count);
      } else {
        held[e.end] -= e.count;
        forChannel[e.end] += e.count;
      }
      moved = true;
      if (++event == period.size()) {
        event = 0;
        round++;
        SmallVector<int64_t> state{static_cast<int64_t>(turn)};
        llvm::append_range(state, forCore);
        llvm::append_range(state, forChannel);
        llvm::append_range(state, held);
        if (!seen.emplace(state, round).second)
          return true;
      }
    }
    if (!moved)
      return false;
  }
  return false;
}
