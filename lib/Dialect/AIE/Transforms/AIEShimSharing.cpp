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
    DenseMap<StringAttr, std::optional<int64_t>> moved;
    auto note = [&](Operation *op, StringAttr symbol, bool issues) {
      std::optional<End> shimEnd = endOf(symbol);
      if (!shimEnd)
        return;
      int64_t from = position[op], to = position[op];
      for (Operation *parent = op->getParentOp(); parent != sequence;
           parent = parent->getParentOp())
        if (isa<LoopLikeOpInterface>(parent)) {
          from = position[parent];
          to = end[parent];
        }
      Span &span = byFifo[shimEnd->fifo];
      span.sends |= shimEnd->sends;
      span.first = std::min(span.first, from);
      span.last = std::max(span.last, to);
      int64_t &mark = issues ? span.lastIssue : span.lastDone;
      mark = std::max(mark, to);
    };
    auto noteTask = [&](Operation *op, Value task, bool issues) {
      if (StringAttr symbol = tasks.lookup(task))
        note(op, symbol, issues);
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
        if (!loop)
          continue;
        std::optional<APInt> trips = loop.getStaticTripCount();
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
        note(op, symbol, true);
        std::optional<int64_t> elems = 1;
        for (OpFoldResult size : memcpy.getMixedSizes()) {
          std::optional<int64_t> extent = getConstantIntValue(size);
          elems = elems && extent ? llvm::checkedMul(*elems, *extent)
                                  : std::nullopt;
        }
        int64_t bits = memcpy.getElementTypeBitwidth();
        count(op, symbol,
              elems && bits % 8 == 0 ? llvm::checkedMul(*elems, bits / 8)
                                     : std::nullopt);
      } else if (auto wait = dyn_cast<AIEX::NpuDmaWaitOp>(op))
        note(op, wait.getSymbolAttr().getAttr(), false);
      else if (auto configure = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
        StringAttr symbol = configure.getAlloc().getRootReference();
        tasks[configure.getResult()] = symbol;
        count(op, symbol, configure.getStaticBytesStreamed());
      } else if (auto chain = dyn_cast<AIEX::DMAStartBdChainForOp>(op)) {
        tasks[chain.getResult()] = chain.getAllocAttr().getAttr();
        note(op, chain.getAllocAttr().getAttr(), true);
        count(op, chain.getAllocAttr().getAttr(), std::nullopt);
      } else if (auto start = dyn_cast<AIEX::DMAStartTaskOp>(op))
        noteTask(op, start.getTask(), true);
      else if (auto await = dyn_cast<AIEX::DMAAwaitTaskOp>(op))
        noteTask(op, await.getTask(), false);
      // Freeing a task lets its BD ids be reused, which is only safe once its
      // transfers are done.
      else if (auto free = dyn_cast<AIEX::DMAFreeTaskOp>(op))
        noteTask(op, free.getTask(), false);
    });
    for (auto &[fifo, bytes] : moved)
      if (auto it = byFifo.find(fifo); it != byFifo.end())
        it->second.bytes = bytes;
  }
}

bool ShimTransferSpans::gatedAfter(const DenseMap<StringAttr, Span> &byFifo,
                                   StringAttr end, int64_t after) const {
  if (!gates)
    return true;
  return llvm::any_of(gates->lookup(end), [&](const Gate &gate) {
    auto input = byFifo.find(gate.input);
    return input != byFifo.end() && input->second.first > after;
  });
}

bool ShimTransferSpans::contained(StringAttr end) const {
  if (!gates)
    return true;
  return llvm::any_of(gates->lookup(end), [&](const Gate &gate) {
    return llvm::all_of(spans, [&](const DenseMap<StringAttr, Span> &byFifo) {
      auto in = byFifo.find(gate.input), out = byFifo.find(end);
      if (out == byFifo.end())
        return in == byFifo.end();
      const Span &drain = out->second;
      std::optional<int64_t> filled = in == byFifo.end() ? 0 : in->second.bytes;
      if (!filled || !drain.bytes || drain.lastDone <= drain.lastIssue)
        return false;
      std::optional<int64_t> sent = llvm::checkedMul(*filled, gate.outputBytes);
      std::optional<int64_t> taken =
          llvm::checkedMul(*drain.bytes, gate.inputBytes);
      return sent && taken && *sent <= *taken;
    });
  });
}

bool ShimTransferSpans::apart(StringAttr a, StringAttr b) const {
  if (a == b)
    return false;
  // An end no sequence names is driven some other way, so nothing is known
  // about when it is in flight.
  bool aUsed = false, bUsed = false, receives = false;
  for (auto &byFifo : spans) {
    auto sa = byFifo.find(a), sb = byFifo.find(b);
    aUsed |= sa != byFifo.end();
    bUsed |= sb != byFifo.end();
    if (sa == byFifo.end() || sb == byFifo.end())
      continue;
    auto before = [&](StringAttr xName, const Span &x, StringAttr yName,
                      const Span &y) {
      return !x.sends && x.last < y.first && x.lastDone > x.lastIssue &&
             gatedAfter(byFifo, yName, x.lastDone);
    };
    if (!before(a, sa->second, b, sb->second) &&
        !before(b, sb->second, a, sa->second))
      return false;
  }
  for (auto &byFifo : spans)
    for (StringAttr end : {a, b})
      if (auto it = byFifo.find(end); it != byFifo.end())
        receives |= !it->second.sends;
  return aUsed && bUsed && (!receives || (contained(a) && contained(b)));
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
