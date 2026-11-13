//===- AIEShimSharing.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <tuple>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEANNOTATESHIMSHARING
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

ShimTransferSpans::ShimTransferSpans(
    DeviceOp device, function_ref<std::optional<End>(StringAttr)> endOf,
    std::optional<Gates> gates, bool plannedOverlay)
    : gates(std::move(gates)), context(device.getContext()),
      overlayColumns(plannedOverlay ? device.getTargetModel().columns() : 0) {
  auto inDevice = [&](DeviceOp owner, StringAttr symbol) {
    return owner == device ? endOf(symbol) : std::nullopt;
  };
  for (int column = 0; column < overlayColumns; ++column)
    used.insert(overlayEnd(column));
  analyze(device, inDevice);
  if (this->gates)
    analyzeGates(device, inDevice);
}

ShimTransferSpans::ShimTransferSpans(
    ModuleOp module,
    function_ref<std::optional<End>(DeviceOp, StringAttr)> endOf,
    std::optional<Gates> gates)
    : gates(std::move(gates)), context(module.getContext()) {
  analyze(module, endOf);
  if (this->gates)
    analyzeGates(module, endOf);
}

void ShimTransferSpans::analyze(
    Operation *root,
    function_ref<std::optional<End>(DeviceOp, StringAttr)> endOf) {
  struct Span {
    StringAttr endpoint;
    int64_t first, last;
  };
  struct Summary {
    SmallVector<Span> spans;
    int64_t duration = 0;
  };
  DenseMap<Operation *, Summary> summaries;
  DenseSet<Operation *> active;
  NamedOpTableCollection names;
  std::function<Summary(RuntimeSequenceOp)> summarize =
      [&](RuntimeSequenceOp sequence) -> Summary {
    if (active.contains(sequence)) {
      unknown = true;
      return {};
    }
    if (auto found = summaries.find(sequence); found != summaries.end())
      return found->second;
    active.insert(sequence);
    DeviceOp device = sequence->getParentOfType<DeviceOp>();
    Summary result;
    DenseMap<Operation *, int64_t> position, end;
    std::function<void(Operation *)> number = [&](Operation *op) {
      position[op] = result.duration++;
      if (auto run = dyn_cast<AIEX::RunOp>(op)) {
        if (auto callee = run.getCalleeRuntimeSequenceOp(&names))
          result.duration += summarize(callee).duration;
        else
          unknown = true;
      }
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &nested : block)
            number(&nested);
      end[op] = result.duration++;
    };
    number(sequence);
    auto interval = [&](Operation *op, int64_t from, int64_t to) {
      for (Operation *parent = op->getParentOp(); parent != sequence;
           parent = parent->getParentOp())
        if (isa<LoopLikeOpInterface>(parent)) {
          from = position[parent];
          to = end[parent];
        }
      return std::make_pair(from, to);
    };
    // Completion names a specific start, so a callee cannot retire caller work.
    auto note = [&](StringAttr endpoint, int64_t at,
                    std::optional<size_t> completion = std::nullopt) {
      if (completion) {
        result.spans[*completion].last = at;
        return *completion;
      }
      used.insert(endpoint);
      result.spans.push_back({endpoint, at, end[sequence]});
      return result.spans.size() - 1;
    };
    DenseMap<StringAttr, SmallVector<size_t>> pending;
    DenseMap<Value, StringAttr> tasks;
    DenseMap<size_t, Value> owners;
    auto transfer = [&](Operation *op, StringAttr symbol, bool starts,
                        bool all = false, Value owner = {}) {
      auto endpoint = endOf(device, symbol);
      if (!endpoint)
        return;
      auto [from, to] = interval(op, position[op], position[op]);
      auto &queue = pending[symbol];
      if (starts) {
        size_t start = note(endpoint->fifo, from);
        owners[start] = owner;
        queue.push_back(start);
      } else if (!endpoint->sends) {
        for (auto it = queue.begin(); it != queue.end();) {
          if (owner && owners.lookup(*it) != owner) {
            ++it;
            continue;
          }
          note(endpoint->fifo, to, *it);
          it = queue.erase(it);
          if (!all)
            break;
        }
      }
    };
    auto synchronous = [&](Operation *op, StringAttr endpoint, int64_t from,
                           int64_t to) {
      std::tie(from, to) = interval(op, from, to);
      auto start = note(endpoint, from);
      note(endpoint, to, start);
    };
    auto configuration = [&](Operation *op, DeviceOp configured,
                             std::optional<int> column = {}) {
      if (!configured)
        return;
      for (int planned = 0; planned < overlayColumns; ++planned)
        if (!column || planned == *column)
          synchronous(op, overlayEnd(planned), position[op], position[op]);
      for (auto alloc : configured.getOps<ShimDMAAllocationOp>()) {
        if (!AIEDialect::IsCtrlPktOverlayAttrHelper(root->getContext())
                 .isAttrPresent(alloc) ||
            (column && alloc.getTileOp().colIndex() != *column))
          continue;
        if (auto endpoint = endOf(configured, alloc.getSymNameAttr()))
          synchronous(op, endpoint->fifo, position[op], position[op]);
      }
    };
    sequence.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto run = dyn_cast<AIEX::RunOp>(op)) {
        if (auto callee = run.getCalleeRuntimeSequenceOp(&names))
          for (const Span &span : summarize(callee).spans)
            synchronous(op, span.endpoint, position[op] + 1 + span.first,
                        position[op] + 1 + span.last);
      } else if (auto configure = dyn_cast<AIEX::ConfigureOp>(op)) {
        configuration(op, configure.getReferencedDeviceOp());
      } else if (isa<AIEX::NpuLoadPdiOp, AIEX::NpuWriteConfigOp,
                     AIEX::NpuWriteTeardownOp>(op)) {
        if (isa<AIEX::NpuLoadPdiOp>(op) && op->getParentOp() == sequence) {
          for (auto &[symbol, queue] : pending) {
            for (size_t start : queue)
              note(result.spans[start].endpoint, position[op] - 1, start);
            queue.clear();
          }
        }
        auto ref = op->getAttrOfType<FlatSymbolRefAttr>("device_ref");
        configuration(
            op, ref ? SymbolTable::lookupNearestSymbolFrom<DeviceOp>(op, ref)
                    : device);
      } else if (auto packet = dyn_cast<AIEX::NpuControlPacketOp>(op)) {
        configuration(op, device, packet.getColumnFromAddr());
      } else if (auto sync = dyn_cast<AIEX::NpuSyncOp>(op)) {
        SmallVector<int> values;
        for (Value operand : sync->getOperands()) {
          APInt value;
          if (!matchPattern(operand, m_ConstantInt(&value)))
            return;
          values.push_back(value.getSExtValue());
        }
        if (values[4] != 1 || values[5] != 1)
          return;
        StringAttr symbol;
        size_t first = std::numeric_limits<size_t>::max();
        for (const auto &[name, queue] : pending) {
          if (queue.empty() || queue.front() >= first)
            continue;
          Operation *owner = device.lookupSymbol(name);
          TileOp tile;
          std::optional<int> channel;
          DMAChannelDir direction;
          if (auto alloc = dyn_cast_or_null<ShimDMAAllocationOp>(owner)) {
            tile = alloc.getTileOp();
            channel = alloc.getChannelIndex();
            direction = alloc.getChannelDir();
          } else if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(owner)) {
            tile = endpoint.getTile().getDefiningOp<TileOp>();
            channel = endpoint.getChannelIndex();
            direction = endpoint.getRouteDirection();
          }
          if (tile && channel && tile.colIndex() == values[0] &&
              tile.rowIndex() == values[1] &&
              static_cast<int>(direction) == values[2] &&
              *channel == values[3]) {
            first = queue.front();
            symbol = name;
          }
        }
        if (symbol)
          transfer(op, symbol, false);
      } else if (auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op)) {
        transfer(op, memcpy.getMetadata().getRootReference(), true);
      } else if (auto wait = dyn_cast<AIEX::NpuDmaWaitOp>(op)) {
        transfer(op, wait.getSymbolAttr().getAttr(), false, true);
      } else if (auto configure = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
        tasks[configure.getResult()] = configure.getAlloc().getRootReference();
      } else if (auto chain = dyn_cast<AIEX::DMAStartBdChainForOp>(op)) {
        StringAttr symbol = chain.getAllocAttr().getAttr();
        tasks[chain.getResult()] = symbol;
        transfer(op, symbol, true, false, chain.getResult());
      } else if (isa<AIEX::DMAStartTaskOp, AIEX::DMAAwaitTaskOp,
                     AIEX::DMAFreeTaskOp>(op)) {
        if (auto symbol = tasks.lookup(op->getOperand(0)))
          transfer(op, symbol, isa<AIEX::DMAStartTaskOp>(op),
                   isa<AIEX::DMAFreeTaskOp>(op),
                   isa<AIEX::DMAAwaitTaskOp>(op) ? Value{} : op->getOperand(0));
      }
    });
    llvm::stable_sort(result.spans, [](const Span &a, const Span &b) {
      return a.first < b.first;
    });
    DenseMap<StringAttr, int64_t> live;
    for (const Span &span : result.spans) {
      lifetimes[span.endpoint].emplace_back(summaries.size(), span.first,
                                            span.last);
      for (const auto &[other, last] : live) {
        if (last < span.first)
          continue;
        overlaps.insert({span.endpoint, other});
        overlaps.insert({other, span.endpoint});
      }
      live[span.endpoint] = std::max(live.lookup(span.endpoint), span.last);
    }
    active.erase(sequence);
    summaries[sequence] = result;
    return result;
  };
  root->walk([&](Operation *op) {
    if (!isa<ObjectFifoCreateOp, RouteEndpointOp, ShimDMAAllocationOp>(op))
      return;
    DeviceOp device = op->getParentOfType<DeviceOp>();
    auto endpoint = endOf(device, SymbolTable::getSymbolName(op));
    if (!endpoint)
      return;
    bool sends = endpoint->sends;
    if (auto alloc = dyn_cast<ShimDMAAllocationOp>(op)) {
      sends = alloc.getChannelDir() == DMAChannelDir::MM2S;
      used.insert(endpoint->fifo);
    } else if (auto allowed =
                   AIEDialect::ShareableWithAttrHelper(op->getContext())
                       .getAttr(op)) {
      auto &peers = permissions[endpoint->fifo];
      for (Attribute attr : allowed)
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(attr))
          if (auto other = endOf(device, ref.getAttr()))
            peers.push_back(other->fifo);
    }
    if (auto fifo = dyn_cast<ObjectFifoCreateOp>(op))
      if (llvm::any_of(fifo.getConsumerTiles(), [](Value tile) {
            return cast<TileLike>(tile.getDefiningOp()).isShimTile();
          }))
        sends = false;
    // Receive ordering does not constrain when independent producers send.
    if (!sends)
      receivers.insert(endpoint->fifo);
  });
  root->walk([&](RuntimeSequenceOp sequence) { summarize(sequence); });
}

void ShimTransferSpans::analyzeGates(
    Operation *root,
    function_ref<std::optional<End>(DeviceOp, StringAttr)> endOf) {
  root->walk([&](RuntimeSequenceOp sequence) {
    DeviceOp device = sequence->getParentOfType<DeviceOp>();
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
      std::optional<End> shimEnd = endOf(device, symbol);
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
      std::optional<End> shimEnd = endOf(device, symbol);
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
  });
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

bool ShimTransferSpans::gated(StringAttr a, StringAttr b) const {
  if (!gates || unknown || overlaps.contains({a, b}))
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

StringAttr ShimTransferSpans::keyOf(RouteEndpointOp endpoint) {
  return endpoint.getFifoNameAttr() ? endpoint.getFifoNameAttr()
                                    : endpoint.getSymNameAttr();
}

std::optional<ShimTransferSpans::End>
ShimTransferSpans::endOf(DeviceOp device, StringAttr symbol) {
  Operation *op = SymbolTable::lookupNearestSymbolFrom(device, symbol);
  if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(op)) {
    if (endpoint.getBundle() != WireBundle::DMA ||
        !endpoint.getTileLike().isShimTile())
      return std::nullopt;
    return End{keyOf(endpoint),
               endpoint.getRouteDirection() == DMAChannelDir::MM2S};
  }
  auto fifo = dyn_cast_or_null<ObjectFifoCreateOp>(op);
  if (!fifo)
    return std::nullopt;
  auto producer =
      dyn_cast_or_null<TileLike>(fifo.getProducerTile().getDefiningOp());
  return End{symbol, producer && producer.isShimTile()};
}

ShimTransferSpans::Gates ShimTransferSpans::provenGates(
    DeviceOp device, function_ref<StringAttr(RouteEndpointOp)> keyOf) {
  Gates gates;
  for (auto input : device.getOps<RouteEndpointOp>()) {
    StringAttr inputKey = input.getConsumedBy() ? keyOf(input) : StringAttr();
    if (!inputKey)
      continue;
    for (auto proof : input.getConsumedByAttr().getAsRange<ConsumedByAttr>()) {
      auto output = SymbolTable::lookupNearestSymbolFrom<RouteEndpointOp>(
          device, proof.getOutput());
      if (StringAttr outputKey = output ? keyOf(output) : StringAttr())
        gates[outputKey].push_back({inputKey, proof.getInputBytes(),
                                    proof.getOutputBytes(),
                                    proof.getGranuleBytes()});
    }
  }
  return gates;
}

bool ShimTransferSpans::isUsed(StringAttr endpoint) const {
  return used.contains(endpoint);
}

StringAttr ShimTransferSpans::overlayEnd(int column) const {
  // Typed, so no symbol name (an untyped StringAttr) is ever this end.
  return StringAttr::get("ctrlpkt_col" + Twine(column),
                         IndexType::get(context));
}

bool ShimTransferSpans::apart(StringAttr a, StringAttr b) const {
  auto permits = [&](StringAttr from, StringAttr to) {
    if (auto it = permissions.find(from); it != permissions.end())
      return llvm::is_contained(it->second, to);
    if (receivers.contains(from))
      return gated(from, to);
    return !unknown && isUsed(from) && isUsed(to) &&
           !overlaps.contains({from, to});
  };
  return a != b && permits(a, b) && permits(b, a);
}

SmallVector<SmallVector<unsigned>>
ShimTransferSpans::groups(ArrayRef<StringAttr> ends) const {
  auto when = llvm::map_to_vector(
      ends, [&](StringAttr end) { return lifetimes.lookup(end); });
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

ShimTransferSpans::Plan
ShimTransferSpans::plan(ArrayRef<StringAttr> ends, ArrayRef<StringAttr> names,
                        ArrayRef<SharedChannel> shared) const {
  Plan result;
  result.sharedChannel.assign(ends.size(), std::nullopt);
  for (auto [index, channel] : llvm::enumerate(shared)) {
    if (channel.pinned)
      continue;
    SmallVector<unsigned> taken;
    for (unsigned end : llvm::seq<unsigned>(0, ends.size()))
      if (!result.sharedChannel[end] &&
          llvm::is_contained(channel.listed, names[end]) &&
          llvm::all_of(taken, [&](unsigned other) {
            return apart(ends[end], ends[other]);
          })) {
        result.sharedChannel[end] = index;
        taken.push_back(end);
      }
  }
  SmallVector<unsigned> rest;
  SmallVector<StringAttr> restEnds;
  for (unsigned end : llvm::seq<unsigned>(0, ends.size()))
    if (!result.sharedChannel[end]) {
      rest.push_back(end);
      restEnds.push_back(ends[end]);
    }
  for (auto &group : groups(restEnds)) {
    auto &members = result.groups.emplace_back();
    for (unsigned member : group)
      members.push_back(rest[member]);
  }
  return result;
}

namespace {
struct AIEAnnotateShimSharingPass
    : xilinx::AIE::impl::AIEAnnotateShimSharingBase<
          AIEAnnotateShimSharingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    AIEDialect::IsCtrlPktOverlayAttrHelper overlay(&getContext());
    AIEDialect::ShareableWithAttrHelper sharing(&getContext());
    DenseMap<Operation *, StringAttr> keys;
    auto key = [&](Operation *op) {
      auto [it, inserted] = keys.try_emplace(op);
      if (inserted)
        it->second =
            StringAttr::get(&getContext(), std::to_string(keys.size()));
      return it->second;
    };
    ShimTransferSpans::Gates gates;
    for (DeviceOp device : module.getOps<DeviceOp>())
      for (auto &[output, proven] : ShimTransferSpans::provenGates(
               device, [&](RouteEndpointOp endpoint) {
                 return endpoint.getBundle() == WireBundle::DMA &&
                                endpoint.getTileLike().isShimTile()
                            ? key(endpoint)
                            : StringAttr();
               }))
        llvm::append_range(gates[output], proven);
    ShimTransferSpans spans(
        module,
        [&](DeviceOp device,
            StringAttr symbol) -> std::optional<ShimTransferSpans::End> {
          Operation *op = device.lookupSymbol(symbol);
          if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(op)) {
            if (endpoint.getBundle() != WireBundle::DMA ||
                !endpoint.getTileLike().isShimTile())
              return std::nullopt;
            return ShimTransferSpans::End{key(endpoint),
                                          endpoint.getRouteDirection() ==
                                              DMAChannelDir::MM2S};
          }
          auto alloc = dyn_cast_or_null<ShimDMAAllocationOp>(op);
          if (!alloc)
            return std::nullopt;
          if (overlay.isAttrPresent(alloc)) {
            if (auto ref = device.getOverlayAttr())
              if (auto definition =
                      module.lookupSymbol<DeviceOp>(ref.getValue()))
                if (Operation *original = definition.lookupSymbol(symbol))
                  op = original;
            return ShimTransferSpans::End{key(op), false};
          }
          return ShimTransferSpans::End{key(op), alloc.getChannelDir() ==
                                                     DMAChannelDir::MM2S};
        },
        std::move(gates));
    for (DeviceOp device : module.getOps<DeviceOp>()) {
      auto annotate = [&](Operation *op, Value tile, DMAChannelDir direction,
                          StringAttr identity) {
        if (sharing.isAttrPresent(op))
          return;
        SmallVector<Attribute> allowed;
        for (auto other : device.getOps<RouteEndpointOp>())
          if (other != op && other.getBundle() == WireBundle::DMA &&
              other.getTile() == tile &&
              other.getRouteDirection() == direction &&
              spans.apart(identity, keys.lookup(other)))
            allowed.push_back(FlatSymbolRefAttr::get(other.getSymNameAttr()));
        sharing.setAttr(op, ArrayAttr::get(&getContext(), allowed));
      };
      for (auto endpoint : device.getOps<RouteEndpointOp>())
        if (endpoint.getBundle() == WireBundle::DMA &&
            endpoint.getTileLike().isShimTile())
          annotate(endpoint, endpoint.getTile(), endpoint.getRouteDirection(),
                   keys.lookup(endpoint));
      for (auto alloc : device.getOps<ShimDMAAllocationOp>())
        if (overlay.isAttrPresent(alloc)) {
          Operation *original = alloc;
          if (auto ref = device.getOverlayAttr())
            if (auto definition = module.lookupSymbol<DeviceOp>(ref.getValue()))
              if (Operation *source =
                      definition.lookupSymbol(alloc.getSymName()))
                original = source;
          annotate(alloc, alloc.getTile(), alloc.getChannelDir(),
                   keys.lookup(original));
        }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
AIE::createAIEAnnotateShimSharingPass() {
  return std::make_unique<AIEAnnotateShimSharingPass>();
}
