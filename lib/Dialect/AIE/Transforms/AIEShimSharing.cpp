//===- AIEShimSharing.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

ShimTransferSpans::ShimTransferSpans(
    DeviceOp device, function_ref<StringAttr(StringAttr)> fifoOf) {
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
    DenseMap<Value, AIEX::DMAConfigureTaskForOp> tasks;
    auto note = [&](Operation *op, StringAttr symbol, bool issues) {
      StringAttr fifo = fifoOf(symbol);
      if (!fifo)
        return;
      int64_t from = position[op], to = position[op];
      for (Operation *parent = op->getParentOp(); parent != sequence;
           parent = parent->getParentOp())
        if (isa<LoopLikeOpInterface>(parent)) {
          from = position[parent];
          to = end[parent];
        }
      Span &span = byFifo[fifo];
      span.first = std::min(span.first, from);
      span.last = std::max(span.last, to);
      int64_t &mark = issues ? span.lastIssue : span.lastDone;
      mark = std::max(mark, to);
    };
    auto noteTask = [&](Operation *op, Value task, bool issues) {
      if (auto configure = tasks.lookup(task))
        note(op, configure.getAlloc().getRootReference(), issues);
    };
    sequence.walk([&](Operation *op) {
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
      // Freeing a task lets its BD ids be reused, which is only safe once its
      // transfers are done.
      else if (auto free = dyn_cast<AIEX::DMAFreeTaskOp>(op))
        noteTask(op, free.getTask(), false);
    });
  }
}

bool ShimTransferSpans::apart(StringAttr a, StringAttr b) const {
  if (a == b)
    return false;
  // An end no sequence names is driven some other way, so nothing is known
  // about when it is in flight.
  bool aUsed = false, bUsed = false;
  for (auto &byFifo : spans) {
    auto sa = byFifo.find(a), sb = byFifo.find(b);
    aUsed |= sa != byFifo.end();
    bUsed |= sb != byFifo.end();
    if (sa == byFifo.end() || sb == byFifo.end())
      continue;
    auto before = [](const Span &x, const Span &y) {
      return x.last < y.first && x.lastDone > x.lastIssue;
    };
    if (!before(sa->second, sb->second) && !before(sb->second, sa->second))
      return false;
  }
  return aUsed && bUsed;
}

SmallVector<SmallVector<unsigned>>
ShimTransferSpans::groups(ArrayRef<StringAttr> ends) const {
  auto order = llvm::to_vector(llvm::seq<unsigned>(0, ends.size()));
  llvm::stable_sort(order, [&](unsigned x, unsigned y) {
    return ends[x].getValue() < ends[y].getValue();
  });
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
