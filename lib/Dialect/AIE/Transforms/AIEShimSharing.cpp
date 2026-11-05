//===- AIEShimSharing.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <tuple>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

ShimTransferSpans::ShimTransferSpans(
    DeviceOp device, function_ref<std::optional<End>(StringAttr)> endOf,
    std::optional<DenseMap<StringAttr, SmallVector<Gate>>> gates)
    : gates(std::move(gates)) {
  for (auto sequence : device.getOps<RuntimeSequenceOp>()) {
    DenseMap<Operation *, int64_t> position, end;
    int64_t next = 0;
    std::function<void(Operation *)> number = [&](Operation *op) {
      position[op] = next++;
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &nested : block)
            number(&nested);
      end[op] = next - 1;
    };
    number(sequence);

    auto &byFifo = spans.emplace_back();
    DenseMap<Value, StringAttr> tasks;
    DenseMap<Value, bool> tokens;
    DenseMap<StringAttr, std::optional<int64_t>> moved;
    // An end's transfers not yet awaited, over the loops around them: nullopt
    // once one is issued that no await can take (no completion token, under a
    // branch, or in a loop of unknown trips). An await under a branch, or in a
    // loop that may not run, takes nothing; an issue there still spans the
    // whole branch or loop.
    DenseMap<StringAttr, std::optional<int64_t>> pending;
    auto note = [&](Operation *op, StringAttr symbol, bool issues, bool awaits,
                    bool tokened = false) {
      std::optional<End> shimEnd = endOf(symbol);
      if (!shimEnd)
        return;
      int64_t from = position[op], to = position[op];
      std::optional<int64_t> runs = 1;
      for (Operation *parent = op->getParentOp(); parent != sequence;
           parent = parent->getParentOp()) {
        from = position[parent];
        to = end[parent];
        auto loop = dyn_cast<LoopLikeOpInterface>(parent);
        std::optional<APInt> trips =
            loop ? loop.getStaticTripCount() : std::nullopt;
        runs = runs && trips && !trips->isZero()
                   ? llvm::checkedMul(
                         *runs, static_cast<int64_t>(trips->getZExtValue()))
                   : std::nullopt;
      }
      Span &span = byFifo[shimEnd->fifo];
      span.sends |= shimEnd->sends;
      span.first = std::min(span.first, from);
      span.last = std::max(span.last, to);
      std::optional<int64_t> &waiting =
          pending.try_emplace(shimEnd->fifo, 0).first->second;
      if (issues) {
        span.lastIssue = std::max(span.lastIssue, to);
        waiting = waiting && runs && tokened ? llvm::checkedAdd(*waiting, *runs)
                                             : std::nullopt;
      } else if (runs) {
        span.lastDone = std::max(span.lastDone, to);
        if (awaits && waiting) {
          waiting = std::max<int64_t>(*waiting - *runs, 0);
          if (*waiting == 0)
            span.lastAwaited = std::max(span.lastAwaited, to);
        }
      }
    };
    auto noteTask = [&](Operation *op, Value task, bool issues, bool awaits) {
      if (StringAttr symbol = tasks.lookup(task))
        note(op, symbol, issues, awaits, tokens.lookup(task));
    };
    // Adds the bytes `op` moves each time it runs, times the trip counts of
    // the loops around it, to its end's total; nullopt where any is unknown.
    auto count = [&](Operation *op, StringAttr symbol,
                     std::optional<int64_t> bytes) {
      std::optional<End> shimEnd = endOf(symbol);
      if (!shimEnd)
        return;
      for (Operation *parent = op->getParentOp(); bytes && parent != sequence;
           parent = parent->getParentOp()) {
        auto loop = dyn_cast<LoopLikeOpInterface>(parent);
        std::optional<APInt> trips =
            loop ? loop.getStaticTripCount() : std::nullopt;
        bytes = trips ? llvm::checkedMul(
                            *bytes, static_cast<int64_t>(trips->getZExtValue()))
                      : std::nullopt;
      }
      std::optional<int64_t> &total =
          moved.try_emplace(shimEnd->fifo, 0).first->second;
      total = total && bytes ? llvm::checkedAdd(*total, *bytes) : std::nullopt;
    };
    sequence.walk([&](Operation *op) {
      if (auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op)) {
        StringAttr symbol = memcpy.getMetadata().getRootReference();
        note(op, symbol, true, false, memcpy.getIssueToken());
        count(op, symbol, memcpy.getStaticBytesStreamed());
      } else if (auto wait = dyn_cast<AIEX::NpuDmaWaitOp>(op))
        note(op, wait.getSymbolAttr().getAttr(), false, true);
      else if (auto configure = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
        StringAttr symbol = configure.getAlloc().getRootReference();
        tasks[configure.getResult()] = symbol;
        tokens[configure.getResult()] = configure.getIssueToken();
        count(op, symbol, configure.getStaticBytesStreamed());
      } else if (auto chain = dyn_cast<AIEX::DMAStartBdChainForOp>(op)) {
        tasks[chain.getResult()] = chain.getAllocAttr().getAttr();
        tokens[chain.getResult()] = chain.getIssueToken();
        note(op, chain.getAllocAttr().getAttr(), true, false,
             chain.getIssueToken());
        count(op, chain.getAllocAttr().getAttr(), std::nullopt);
      } else if (auto start = dyn_cast<AIEX::DMAStartTaskOp>(op))
        noteTask(op, start.getTask(), true, false);
      else if (auto await = dyn_cast<AIEX::DMAAwaitTaskOp>(op))
        noteTask(op, await.getTask(), false, true);
      // Freeing a task lets its BD ids be reused, which is only safe once its
      // transfers are done.
      else if (auto free = dyn_cast<AIEX::DMAFreeTaskOp>(op))
        noteTask(op, free.getTask(), false, false);
    });
    for (auto &[fifo, bytes] : moved)
      if (auto it = byFifo.find(fifo); it != byFifo.end())
        it->second.bytes = bytes;
  }
}

SmallVector<const ShimTransferSpans::Gate *>
ShimTransferSpans::usableGates(StringAttr end) const {
  bool receives = llvm::any_of(spans, [&](const auto &byFifo) {
    auto it = byFifo.find(end);
    return it != byFifo.end() && !it->second.sends;
  });
  if (!gates || !receives)
    return {nullptr};
  SmallVector<const Gate *> usable;
  if (auto it = gates->find(end); it != gates->end())
    for (const Gate &gate : it->second)
      if (contained(end, gate))
        usable.push_back(&gate);
  return usable;
}

bool ShimTransferSpans::before(const DenseMap<StringAttr, Span> &byFifo,
                               const Span &x, const Span &y,
                               const Gate *gate) const {
  if (x.sends || x.last >= y.first || x.lastDone <= x.lastIssue)
    return false;
  if (!gate)
    return true;
  auto input = byFifo.find(gate->input);
  return x.lastAwaited > x.lastIssue && input != byFifo.end() &&
         input->second.first > x.lastAwaited;
}

bool ShimTransferSpans::contained(StringAttr end, const Gate &gate) const {
  return llvm::all_of(spans, [&](const DenseMap<StringAttr, Span> &byFifo) {
    auto in = byFifo.find(gate.input), out = byFifo.find(end);
    if (out == byFifo.end())
      return in == byFifo.end();
    const Span &drain = out->second;
    std::optional<int64_t> filled = in == byFifo.end() ? 0 : in->second.bytes;
    if (!filled || !drain.bytes || drain.lastAwaited <= drain.lastIssue ||
        (gate.granuleBytes && *drain.bytes % gate.granuleBytes))
      return false;
    std::optional<int64_t> sent = llvm::checkedMul(*filled, gate.outputBytes);
    std::optional<int64_t> taken =
        llvm::checkedMul(*drain.bytes, gate.inputBytes);
    return sent && taken && *sent <= *taken;
  });
}

bool ShimTransferSpans::apart(StringAttr a, StringAttr b) const {
  if (a == b)
    return false;
  // An end no sequence names is driven some other way, so nothing is known
  // about when it is in flight.
  for (StringAttr end : {a, b})
    if (llvm::none_of(spans,
                      [&](const auto &byFifo) { return byFifo.contains(end); }))
      return false;
  // One gate per end serves every sequence, for its fill bound and its turn.
  for (const Gate *ga : usableGates(a))
    for (const Gate *gb : usableGates(b))
      if (llvm::all_of(spans, [&](const DenseMap<StringAttr, Span> &byFifo) {
            auto sa = byFifo.find(a), sb = byFifo.find(b);
            return sa == byFifo.end() || sb == byFifo.end() ||
                   before(byFifo, sa->second, sb->second, gb) ||
                   before(byFifo, sb->second, sa->second, ga);
          }))
        return true;
  return false;
}

SmallVector<SmallVector<unsigned>>
ShimTransferSpans::groups(ArrayRef<StringAttr> ends) const {
  // By each end's whole span, all apart() reads, not by name: ends this
  // cannot order relate alike to every other, so either may go first.
  using Key = std::tuple<int64_t, int64_t, int64_t, int64_t, bool>;
  SmallVector<SmallVector<Key>> when;
  for (StringAttr end : ends) {
    auto &key = when.emplace_back();
    for (auto &byFifo : spans) {
      auto it = byFifo.find(end);
      Span span = it == byFifo.end() ? Span() : it->second;
      key.push_back(
          {span.first, span.last, span.lastIssue, span.lastDone, span.sends});
    }
  }
  auto order = llvm::to_vector(llvm::seq<unsigned>(0, ends.size()));
  llvm::stable_sort(order,
                    [&](unsigned x, unsigned y) { return when[x] < when[y]; });
  SmallVector<SmallVector<unsigned>> result;
  for (unsigned end : order) {
    auto *group = llvm::find_if(result, [&](ArrayRef<unsigned> members) {
      return llvm::all_of(
          members, [&](unsigned m) { return apart(ends[m], ends[end]); });
    });
    if (group == result.end())
      result.push_back({end});
    else
      group->push_back(end);
  }
  return result;
}
