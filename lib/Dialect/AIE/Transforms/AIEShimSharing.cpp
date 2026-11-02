//===- AIEShimSharing.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <tuple>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

ShimTransferSpans::ShimTransferSpans(
    DeviceOp device, function_ref<std::optional<End>(StringAttr)> endOf,
    StringAttr configuration) {
  analyze(
      device,
      [&](DeviceOp owner, StringAttr symbol) {
        return owner == device ? endOf(symbol) : std::nullopt;
      },
      configuration);
}

ShimTransferSpans::ShimTransferSpans(
    ModuleOp module,
    function_ref<std::optional<End>(DeviceOp, StringAttr)> endOf,
    StringAttr configuration) {
  analyze(module, endOf, configuration);
}

void ShimTransferSpans::analyze(
    Operation *root,
    function_ref<std::optional<End>(DeviceOp, StringAttr)> endOf,
    StringAttr configuration) {
  using Ends = DenseSet<StringAttr>;
  DenseMap<Operation *, Ends> summaries;
  DenseSet<Operation *> active;
  NamedOpTableCollection names;
  auto conflict = [&](StringAttr a, StringAttr b) {
    overlaps.insert({a, b});
    overlaps.insert({b, a});
  };
  std::function<Ends(RuntimeSequenceOp)> summarize =
      [&](RuntimeSequenceOp sequence) -> Ends {
    if (active.contains(sequence)) {
      unknown = true;
      return {};
    }
    if (auto found = summaries.find(sequence); found != summaries.end())
      return found->second;
    active.insert(sequence);
    DeviceOp device = sequence->getParentOfType<DeviceOp>();
    Ends ends;
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
    auto interval = [&](Operation *op) {
      int64_t from = position[op], to = position[op];
      for (Operation *parent = op->getParentOp(); parent != sequence;
           parent = parent->getParentOp())
        if (isa<LoopLikeOpInterface>(parent)) {
          from = position[parent];
          to = end[parent];
        }
      return std::make_pair(from, to);
    };

    struct Call {
      Ends ends;
      int64_t first, last;
      size_t phase;
    };
    SmallVector<Call> calls;
    SmallVector<DenseMap<StringAttr, Span>> spans(1);
    size_t phase = 0;
    DenseMap<std::tuple<int, int, int, int>, StringAttr> physicalEnds;
    DenseMap<Value, AIEX::DMAConfigureTaskForOp> tasks;
    DenseMap<StringAttr, unsigned> pendingConfiguration;
    auto note = [&](Operation *op, StringAttr symbol, bool issues) {
      std::optional<End> shimEnd = endOf(device, symbol);
      if (!shimEnd)
        return;
      ends.insert(shimEnd->fifo);
      auto [from, to] = interval(op);
      if (shimEnd->channel)
        physicalEnds[*shimEnd->channel] = symbol;
      Span &span = spans[phase][shimEnd->fifo];
      span.sends |= shimEnd->sends;
      span.first = std::min(span.first, from);
      span.last = std::max(span.last, to);
      if (shimEnd->fifo == configuration) {
        if (issues)
          ++pendingConfiguration[symbol];
        else {
          auto pending = pendingConfiguration.find(symbol);
          if (pending != pendingConfiguration.end() && --pending->second == 0)
            pendingConfiguration.erase(pending);
          if (!pendingConfiguration.empty())
            return;
        }
      }
      int64_t &mark = issues ? span.lastIssue : span.lastDone;
      mark = std::max(mark, to);
    };
    auto noteTask = [&](Operation *op, Value task, bool issues) {
      if (auto configure = tasks.lookup(task))
        note(op, configure.getAlloc().getRootReference(), issues);
    };
    sequence.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto run = dyn_cast<AIEX::RunOp>(op)) {
        if (auto callee = run.getCalleeRuntimeSequenceOp(&names)) {
          Ends called = summarize(callee);
          ends.insert(called.begin(), called.end());
          auto [from, to] = interval(op);
          calls.push_back({std::move(called), from, to, phase});
        } else {
          unknown = true;
        }
        return;
      }
      if (configuration && isa<AIEX::ConfigureOp>(op)) {
        ends.insert(configuration);
        auto [from, to] = interval(op);
        calls.push_back({Ends{configuration}, from, to, phase});
        return;
      }
      if (configuration &&
          isa<AIEX::NpuLoadPdiOp, AIEX::NpuWriteConfigOp,
              AIEX::NpuWriteTeardownOp>(op) &&
          op->getParentOp() == sequence) {
        ends.insert(configuration);
        calls.push_back(
            {Ends{configuration}, position[op], position[op], ++phase});
        spans.emplace_back();
        ++phase;
        spans.emplace_back();
        physicalEnds.clear();
        pendingConfiguration.clear();
        return;
      }
      if (configuration &&
          isa<AIEX::NpuLoadPdiOp, AIEX::NpuWriteConfigOp,
              AIEX::NpuWriteTeardownOp, AIEX::NpuControlPacketOp>(op)) {
        ends.insert(configuration);
        Span &span = spans[phase][configuration];
        span.first = std::min(span.first, position[op]);
        // Control-packet lowering synchronizes each batch before the next
        // non-packet operation.
        span.lastIssue = position[op] - 1;
        span.last = position[op];
        if (pendingConfiguration.empty())
          span.lastDone = position[op];
        if (op->getParentOp() != sequence) {
          span.sends = true;
          span.first = 0;
          span.last = end[sequence];
          span.lastIssue = span.last;
        }
        return;
      }
      if (auto sync = dyn_cast<AIEX::NpuSyncOp>(op)) {
        SmallVector<int> values;
        for (Value operand : sync->getOperands()) {
          APInt value;
          if (!matchPattern(operand, m_ConstantInt(&value)))
            return;
          values.push_back(value.getSExtValue());
        }
        if (values[4] == 1 && values[5] == 1)
          if (auto symbol = physicalEnds.lookup(
                  {values[0], values[1], values[2], values[3]}))
            note(op, symbol, false);
        return;
      }
      if (auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op))
        note(op, memcpy.getMetadata().getRootReference(), true);
      else if (auto wait = dyn_cast<AIEX::NpuDmaWaitOp>(op))
        note(op, wait.getSymbolAttr().getAttr(), false);
      else if (auto configure = dyn_cast<AIEX::DMAConfigureTaskForOp>(op))
        tasks[configure.getResult()] = configure;
      else if (auto start = dyn_cast<AIEX::DMAStartTaskOp>(op))
        noteTask(op, start.getTask(), true);
      else if (auto await = dyn_cast<AIEX::DMAAwaitTaskOp>(op))
        noteTask(op, await.getTask(), false);
      else if (auto free = dyn_cast<AIEX::DMAFreeTaskOp>(op))
        noteTask(op, free.getTask(), false);
    });

    auto last = [](const Span &span) {
      return !span.sends && span.lastDone > span.lastIssue
                 ? span.last
                 : std::numeric_limits<int64_t>::max();
    };
    for (const auto &byEnd : spans)
      for (const auto &[a, x] : byEnd)
        for (const auto &[b, y] : byEnd)
          if (last(x) >= y.first && last(y) >= x.first)
            conflict(a, b);
    for (const Call &call : calls) {
      // A call completes its own transfers, but cannot retire caller work.
      for (const auto &[a, span] : spans[call.phase])
        if (span.first <= call.last && last(span) >= call.first)
          for (StringAttr b : call.ends)
            conflict(a, b);
      for (const Call &other : calls)
        if (call.phase == other.phase && &call != &other &&
            call.first <= other.last && other.first <= call.last)
          for (StringAttr a : call.ends)
            for (StringAttr b : other.ends)
              conflict(a, b);
    }
    active.erase(sequence);
    used.insert(ends.begin(), ends.end());
    summaries[sequence] = ends;
    return ends;
  };
  if (configuration)
    used.insert(configuration);
  root->walk([&](RuntimeSequenceOp sequence) { summarize(sequence); });
}

bool ShimTransferSpans::isUsed(StringAttr endpoint) const {
  return used.contains(endpoint);
}

bool ShimTransferSpans::apart(StringAttr a, StringAttr b) const {
  return !unknown && a != b && isUsed(a) && isUsed(b) &&
         !overlaps.contains({a, b});
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
