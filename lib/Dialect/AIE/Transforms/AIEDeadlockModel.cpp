//===- AIEDeadlockModel.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEDeadlockModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <deque>
#include <set>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

constexpr uint64_t wordBytes = 4;
// A loop at least this long in a core is how a design says "forever".
constexpr int64_t foreverTrips = int64_t{1} << 30;
// Loops are unrolled up to this many events.
constexpr size_t maxEvents = 1 << 20;

uint64_t wordsOf(uint64_t bytes) { return (bytes + wordBytes - 1) / wordBytes; }

enum class Flow { Next, Forever, Outside };

/// Runs a core body or a runtime sequence on the integers it can know,
/// turning the ops `event` handles into events. A value read from memory, or
/// computed from one, is unknown; control flow that depends on it around an
/// event is outside the model. A loop that never exits is found by its
/// loop-carried state repeating, and leaves `cycleFrom` at the first event of
/// the repeating part.
class Interpreter {
public:
  /// Appends `op`'s events and returns true if it is an event op; with
  /// `probe`, only answers whether it is one.
  using EventFn = function_ref<bool(Operation *op, bool probe)>;

  Interpreter(EventFn event, function_ref<size_t()> eventCount)
      : event(event), eventCount(eventCount) {}

  std::optional<int64_t> value(Value v) const {
    auto it = env.find(v);
    if (it == env.end())
      return std::nullopt;
    return it->second;
  }

  Flow run(Block &block) {
    SmallVector<std::optional<int64_t>> yielded;
    return runBlock(block, yielded);
  }

  std::string reason;
  Operation *where = nullptr;
  std::optional<size_t> cycleFrom;

private:
  Flow outside(Operation *op, const Twine &why) {
    where = op;
    reason = why.str();
    return Flow::Outside;
  }

  void bind(ValueRange values, ArrayRef<std::optional<int64_t>> ints) {
    for (auto [v, i] : llvm::zip(values, ints)) {
      if (i)
        env[v] = *i;
      else
        env.erase(v);
    }
  }

  void unknown(ValueRange values) {
    for (Value v : values)
      env.erase(v);
  }

  bool hasEvents(Operation *op) {
    bool found = false;
    op->walk([&](Operation *x) {
      if (x != op && event(x, true))
        found = true;
    });
    return found;
  }

  Flow runBlock(Block &block, SmallVector<std::optional<int64_t>> &yielded) {
    for (Operation &op : block) {
      if (isa<scf::YieldOp, scf::ConditionOp>(op)) {
        yielded.clear();
        for (Value v : op.getOperands())
          yielded.push_back(value(v));
        return Flow::Next;
      }
      if (Flow f = runOp(&op); f != Flow::Next)
        return f;
    }
    return Flow::Next;
  }

  Flow runOp(Operation *op) {
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
        env[c.getResult()] = i.getValue().getSExtValue();
      else
        unknown(op->getResults());
      return Flow::Next;
    }
    if (std::optional<std::optional<int64_t>> r = arith(op)) {
      bind(op->getResults(), {*r});
      return Flow::Next;
    }
    if (event(op, false))
      return Flow::Next;
    if (auto loop = dyn_cast<scf::ForOp>(op))
      return runFor(loop);
    if (auto loop = dyn_cast<scf::WhileOp>(op))
      return runWhile(loop);
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      std::optional<int64_t> c = value(branch.getCondition());
      if (!c) {
        if (hasEvents(op))
          return outside(op, "tokens under data-dependent control");
        unknown(op->getResults());
        return Flow::Next;
      }
      Region &region = *c ? branch.getThenRegion() : branch.getElseRegion();
      SmallVector<std::optional<int64_t>> yielded;
      if (!region.empty())
        if (Flow f = runBlock(region.front(), yielded); f != Flow::Next)
          return f;
      bind(op->getResults(), yielded);
      return Flow::Next;
    }
    if (auto sw = dyn_cast<scf::IndexSwitchOp>(op)) {
      std::optional<int64_t> sel = value(sw.getArg());
      if (!sel) {
        if (hasEvents(op))
          return outside(op, "tokens under data-dependent control");
        unknown(op->getResults());
        return Flow::Next;
      }
      Region *region = &sw.getDefaultRegion();
      for (auto [i, c] : llvm::enumerate(sw.getCases()))
        if (c == *sel)
          region = &sw.getCaseRegions()[i];
      SmallVector<std::optional<int64_t>> yielded;
      if (Flow f = runBlock(region->front(), yielded); f != Flow::Next)
        return f;
      bind(op->getResults(), yielded);
      return Flow::Next;
    }
    if (isa<AIE::EndOp, cf::AssertOp, func::CallOp>(op) ||
        op->getNumRegions() == 0) {
      // Data, not tokens: memory accesses, calls, values the model need not
      // know. An op of the AIE dialects that moves tokens is an event above.
      if (isa<AIEDialect, AIEX::AIEXDialect>(op->getDialect()) &&
          !isa<AIE::EndOp, AIEX::NpuWriteRTPOp>(op))
        return outside(op, llvm::formatv("{0}", op->getName()).str());
      unknown(op->getResults());
      return Flow::Next;
    }
    return outside(op, llvm::formatv("{0}", op->getName()).str());
  }

  std::optional<std::optional<int64_t>> arith(Operation *op) {
    auto binary = [&](auto fn) -> std::optional<int64_t> {
      std::optional<int64_t> a = value(op->getOperand(0));
      std::optional<int64_t> b = value(op->getOperand(1));
      if (!a || !b)
        return std::nullopt;
      return fn(*a, *b);
    };
    auto div = [](int64_t a, int64_t b) -> std::optional<int64_t> {
      if (b == 0)
        return std::nullopt;
      return a / b;
    };
    auto rem = [](int64_t a, int64_t b) -> std::optional<int64_t> {
      if (b == 0)
        return std::nullopt;
      return a % b;
    };
    return llvm::TypeSwitch<Operation *, std::optional<std::optional<int64_t>>>(
               op)
        .Case([&](arith::AddIOp) {
          return binary([](int64_t a, int64_t b) { return a + b; });
        })
        .Case([&](arith::SubIOp) {
          return binary([](int64_t a, int64_t b) { return a - b; });
        })
        .Case([&](arith::MulIOp) {
          return binary([](int64_t a, int64_t b) { return a * b; });
        })
        .Case<arith::DivSIOp, arith::DivUIOp>([&](auto) {
          std::optional<int64_t> a = value(op->getOperand(0));
          std::optional<int64_t> b = value(op->getOperand(1));
          return a && b ? div(*a, *b) : std::nullopt;
        })
        .Case<arith::RemSIOp, arith::RemUIOp>([&](auto) {
          std::optional<int64_t> a = value(op->getOperand(0));
          std::optional<int64_t> b = value(op->getOperand(1));
          return a && b ? rem(*a, *b) : std::nullopt;
        })
        .Case<arith::MaxSIOp, arith::MaxUIOp>([&](auto) {
          return binary([](int64_t a, int64_t b) { return std::max(a, b); });
        })
        .Case<arith::MinSIOp, arith::MinUIOp>([&](auto) {
          return binary([](int64_t a, int64_t b) { return std::min(a, b); });
        })
        .Case([&](arith::AndIOp) {
          return binary([](int64_t a, int64_t b) { return a & b; });
        })
        .Case([&](arith::OrIOp) {
          return binary([](int64_t a, int64_t b) { return a | b; });
        })
        .Case([&](arith::XOrIOp) {
          return binary([](int64_t a, int64_t b) { return a ^ b; });
        })
        .Case([&](arith::ShLIOp) {
          return binary([](int64_t a, int64_t b) { return a << b; });
        })
        .Case<arith::ShRSIOp, arith::ShRUIOp>([&](auto) {
          return binary([](int64_t a, int64_t b) { return a >> b; });
        })
        .Case<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
              arith::ExtUIOp, arith::TruncIOp>([&](auto) {
          return std::optional<std::optional<int64_t>>(
              value(op->getOperand(0)));
        })
        .Case([&](arith::CmpIOp cmp) -> std::optional<std::optional<int64_t>> {
          std::optional<int64_t> a = value(cmp.getLhs());
          std::optional<int64_t> b = value(cmp.getRhs());
          if (!a || !b)
            return std::optional<int64_t>();
          bool r = false;
          switch (cmp.getPredicate()) {
          case arith::CmpIPredicate::eq:
            r = *a == *b;
            break;
          case arith::CmpIPredicate::ne:
            r = *a != *b;
            break;
          case arith::CmpIPredicate::slt:
          case arith::CmpIPredicate::ult:
            r = *a < *b;
            break;
          case arith::CmpIPredicate::sle:
          case arith::CmpIPredicate::ule:
            r = *a <= *b;
            break;
          case arith::CmpIPredicate::sgt:
          case arith::CmpIPredicate::ugt:
            r = *a > *b;
            break;
          case arith::CmpIPredicate::sge:
          case arith::CmpIPredicate::uge:
            r = *a >= *b;
            break;
          }
          return std::optional<int64_t>(r);
        })
        .Case(
            [&](arith::SelectOp sel) -> std::optional<std::optional<int64_t>> {
              std::optional<int64_t> c = value(sel.getCondition());
              if (!c)
                return std::optional<int64_t>();
              return value(*c ? sel.getTrueValue() : sel.getFalseValue());
            })
        .Default([](Operation *) { return std::nullopt; });
  }

  Flow runFor(scf::ForOp loop) {
    std::optional<int64_t> lb = value(loop.getLowerBound());
    std::optional<int64_t> ub = value(loop.getUpperBound());
    std::optional<int64_t> step = value(loop.getStep());
    SmallVector<std::optional<int64_t>> carried;
    for (Value v : loop.getInitArgs())
      carried.push_back(value(v));
    if (!lb || !ub || !step || *step <= 0) {
      if (hasEvents(loop))
        return outside(loop,
                       "a loop whose bounds are not known, around tokens");
      unknown(loop.getResults());
      return Flow::Next;
    }
    int64_t trips = std::max<int64_t>(0, (*ub - *lb + *step - 1) / *step);
    if (!hasEvents(loop) && carried.empty())
      return Flow::Next;
    Block &body = *loop.getBody();
    bool forever = trips >= foreverTrips;
    DenseMap<ArrayRef<int64_t>, size_t> seen;
    std::deque<SmallVector<int64_t>> keys;
    for (int64_t i = 0; forever || i < trips; ++i) {
      if (forever &&
          llvm::all_of(carried, [](auto c) { return c.has_value(); })) {
        SmallVector<int64_t> &k = keys.emplace_back();
        for (auto c : carried)
          k.push_back(*c);
        auto [it, fresh] = seen.try_emplace(ArrayRef<int64_t>(k), eventCount());
        if (!fresh) {
          cycleFrom = it->second;
          return Flow::Forever;
        }
      }
      env[body.getArgument(0)] = *lb + i * *step;
      bind(body.getArguments().drop_front(), carried);
      if (Flow f = runBlock(body, carried); f != Flow::Next)
        return f;
      if (eventCount() > maxEvents)
        return outside(loop, "a loop too long to unroll");
    }
    bind(loop.getResults(), carried);
    return Flow::Next;
  }

  Flow runWhile(scf::WhileOp loop) {
    SmallVector<std::optional<int64_t>> carried;
    for (Value v : loop.getInits())
      carried.push_back(value(v));
    DenseMap<ArrayRef<int64_t>, size_t> seen;
    std::deque<SmallVector<int64_t>> keys;
    while (true) {
      if (llvm::all_of(carried, [](auto c) { return c.has_value(); })) {
        SmallVector<int64_t> &k = keys.emplace_back();
        for (auto c : carried)
          k.push_back(*c);
        auto [it, fresh] = seen.try_emplace(ArrayRef<int64_t>(k), eventCount());
        if (!fresh) {
          cycleFrom = it->second;
          return Flow::Forever;
        }
      }
      Block &before = loop.getBefore().front();
      bind(before.getArguments(), carried);
      SmallVector<std::optional<int64_t>> cond;
      if (Flow f = runBlock(before, cond); f != Flow::Next)
        return f;
      if (cond.empty() || !cond.front())
        return outside(loop, "a while loop whose condition is not known");
      SmallVector<std::optional<int64_t>> forwarded(cond.begin() + 1,
                                                    cond.end());
      if (!*cond.front()) {
        bind(loop.getResults(), forwarded);
        return Flow::Next;
      }
      Block &after = loop.getAfter().front();
      bind(after.getArguments(), forwarded);
      if (Flow f = runBlock(after, carried); f != Flow::Next)
        return f;
      if (eventCount() > maxEvents)
        return outside(loop, "a loop too long to unroll");
    }
  }

  EventFn event;
  function_ref<size_t()> eventCount;
  DenseMap<Value, int64_t> env;
};

std::optional<TileID> tileOf(Value tile) {
  if (auto t = dyn_cast_or_null<TileOp>(tile.getDefiningOp()))
    return TileID{t.getCol(), t.getRow()};
  return std::nullopt;
}

} // namespace

std::string DeadlockModel::describe(const TileDMAChannel &c) const {
  return llvm::formatv("({0}, {1}) {2} {3}", c.tile.col, c.tile.row,
                       stringifyDMAChannelDir(c.dir), c.channel);
}

DeadlockModel::DeadlockModel(DeviceOp device, RuntimeSequenceOp sequence) {
  DenseMap<Operation *, unsigned> lockIndex;
  device.walk([&](LockOp lock) {
    lockIndex[lock] = lockNames.size();
    std::optional<int> id = lock.getLockID();
    lockNames.push_back(
        lock.hasName() ? lock.name().str()
                       : llvm::formatv("lock {0} of ({1}, {2})", id ? *id : -1,
                                       lock.colIndex(), lock.rowIndex())
                             .str());
    lockInit.push_back(lock.getInit() ? *lock.getInit() : 0);
  });
  auto note = [&](Operation *op, const Twine &why) {
    outsideNotes.push_back({op, why.str()});
  };
  auto lockEvent =
      [&](UseLockOp use,
          std::optional<int64_t> amount) -> std::optional<LockEvent> {
    auto lock = dyn_cast_or_null<LockOp>(use.getLock().getDefiningOp());
    if (!lock || !amount) {
      note(use, "a lock or lock value the model cannot know");
      return std::nullopt;
    }
    return LockEvent{lockIndex.lookup(lock), use.getAction(), *amount, use};
  };

  // A BD block: its acquire, its transfer and header, its release.
  auto bdOf = [&](Block &block) -> std::optional<BD> {
    BD bd{0, std::nullopt, std::nullopt, std::nullopt, nullptr};
    std::optional<int> header;
    for (Operation &op : block) {
      if (auto use = dyn_cast<UseLockOp>(op)) {
        FailureOr<int32_t> v = use.getConstantValue();
        std::optional<LockEvent> ev = lockEvent(
            use, succeeded(v) ? std::optional<int64_t>(*v) : std::nullopt);
        if (!ev)
          continue;
        (use.getAction() == LockAction::Release ? bd.release : bd.acquire) = ev;
      } else if (auto p = dyn_cast<DMABDPACKETOp>(op)) {
        header = p.getPacketId();
      } else if (auto dma = dyn_cast<DMABDOp>(op)) {
        bd.op = dma;
        bd.words = wordsOf(dma.getLenInBytes());
        if (std::optional<PacketInfoAttr> p = dma.getPacket())
          header = p->assignedId();
      }
    }
    if (!bd.op)
      return std::nullopt;
    bd.packet = header;
    return bd;
  };
  auto chainFrom = [&](Block *first) {
    Chain chain;
    SmallVector<Block *> seen;
    for (Block *b = first; b;) {
      if (auto *it = llvm::find(seen, b); it != seen.end()) {
        chain.loopTo = it - seen.begin();
        break;
      }
      seen.push_back(b);
      if (std::optional<BD> bd = bdOf(*b))
        chain.bds.push_back(*bd);
      Block *next = nullptr;
      if (auto nb = dyn_cast<NextBDOp>(b->getTerminator()))
        next = nb.getDest();
      b = next;
    }
    return chain;
  };

  // DMA programs the configuration starts.
  for (Operation &op : device.getBody()->getOperations()) {
    if (!isa<MemOp, MemTileDMAOp, ShimDMAOp>(op)) {
      if (isa<DMAOp>(op))
        note(&op, "aie.dma");
      continue;
    }
    std::optional<TileID> tile =
        tileOf(cast<TileElement>(op).getTileOp().getResult());
    op.walk([&](Operation *x) {
      if (isa<DMAOp>(x))
        note(x, "aie.dma");
      auto start = dyn_cast<DMAStartOp>(x);
      if (!start || !tile)
        return;
      if (start.getEndpoint()) {
        note(start, "a channel allocation has not resolved");
        return;
      }
      Chain chain = chainFrom(start.getDest());
      chain.passes = start.getRepeatCount() + 1;
      chain.op = start;
      statics[{*tile, start.getChannelDir(), start.getChannelIndex()}] = chain;
    });
  }

  // Streams.
  std::map<TileDMAChannel, SmallVector<std::pair<TileDMAChannel, int>>> into;
  auto dmaEnd = [&](Operation *op, Value tile, WireBundle bundle, int channel,
                    DMAChannelDir dir) -> std::optional<TileDMAChannel> {
    std::optional<TileID> t = tileOf(tile);
    if (!t || bundle != WireBundle::DMA) {
      note(op, "a stream end that is not a DMA channel");
      return std::nullopt;
    }
    return TileDMAChannel{*t, dir, channel};
  };
  for (auto flow : device.getOps<FlowOp>()) {
    if (flow.getSourceBundle() == WireBundle::Trace)
      continue;
    std::optional<TileDMAChannel> src =
        dmaEnd(flow, flow.getSource(), flow.getSourceBundle(),
               flow.getSourceChannel(), DMAChannelDir::MM2S);
    std::optional<TileDMAChannel> dst =
        dmaEnd(flow, flow.getDest(), flow.getDestBundle(),
               flow.getDestChannel(), DMAChannelDir::S2MM);
    if (src && dst) {
      sends[{*src, -1}].push_back(*dst);
      into[*dst].push_back({*src, -1});
    }
  }
  for (auto flow : device.getOps<PacketFlowOp>()) {
    SmallVector<TileDMAChannel> srcs, dsts;
    bool trace = false;
    for (Operation &x : flow.getPorts().front()) {
      if (auto s = dyn_cast<PacketSourceOp>(x)) {
        if (s.getBundle() == WireBundle::Trace) {
          trace = true;
          continue;
        }
        if (auto e = dmaEnd(flow, s.getTile(), s.getBundle(), s.getChannel(),
                            DMAChannelDir::MM2S))
          srcs.push_back(*e);
      } else if (auto d = dyn_cast<PacketDestOp>(x)) {
        if (auto e = dmaEnd(flow, d.getTile(), d.getBundle(), d.getChannel(),
                            DMAChannelDir::S2MM))
          dsts.push_back(*e);
      }
    }
    if (trace)
      continue;
    int id = flow.getID();
    for (const TileDMAChannel &s : srcs)
      for (const TileDMAChannel &d : dsts) {
        sends[{s, id}].push_back(d);
        into[d].push_back({s, id});
      }
    for (const TileDMAChannel &d : dsts)
      keepsHeader[d] = flow.getKeepPktHeader().value_or(false);
  }
  for (auto &[dst, srcs] : into) {
    std::set<TileDMAChannel> channels;
    for (auto &[s, id] : srcs)
      channels.insert(s);
    if (channels.size() > 1)
      note(device, "a receiver more than one channel sends to (" +
                       describe(dst) + "): their order is not proven");
  }

  // Cores.
  for (auto coreOp : device.getOps<CoreOp>()) {
    std::optional<TileID> tile = tileOf(coreOp.getTile());
    if (!tile)
      continue;
    Core core{*tile, {}, {}, coreOp};
    SmallVector<LockEvent> events;
    Interpreter *self = nullptr;
    auto event = [&](Operation *op, bool probe) {
      auto use = dyn_cast<UseLockOp>(op);
      if (!use)
        return false;
      if (!probe)
        if (std::optional<LockEvent> ev =
                lockEvent(use, self->value(use.getValue())))
          events.push_back(*ev);
      return true;
    };
    auto count = [&] { return events.size(); };
    Interpreter interp(event, count);
    self = &interp;
    if (coreOp.getBody().getBlocks().size() != 1) {
      note(coreOp, "a core with more than one block");
      continue;
    }
    Flow f = interp.run(coreOp.getBody().front());
    if (f == Flow::Outside) {
      note(interp.where, interp.reason);
      continue;
    }
    size_t from = f == Flow::Forever ? *interp.cycleFrom : events.size();
    core.prefix.assign(events.begin(), events.begin() + from);
    core.cycle.assign(events.begin() + from, events.end());
    cores.push_back(core);
  }

  // The host.
  if (sequence) {
    DenseMap<Value, unsigned> tasks;
    std::map<unsigned, TileDMAChannel> channelOfTask;
    Interpreter *self = nullptr;
    auto taskChain = [&](Operation *op, Region &body, bool token,
                         uint64_t repeat) {
      Chain chain = chainFrom(&body.front());
      chain.passes = repeat + 1;
      chain.token = token;
      chain.op = op;
      return chain;
    };
    auto event = [&](Operation *op, bool probe) {
      if (!isa<AIEX::AIEXDialect>(op->getDialect()) ||
          isa<AIEX::NpuWriteRTPOp>(op))
        return false;
      if (probe)
        return true;
      if (auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op)) {
        auto alloc = ShimDMAAllocationOp::getForSymbol(
            device, memcpy.getMetadata().getRootReference());
        std::optional<TileID> t =
            alloc ? tileOf(alloc.getTile()) : std::nullopt;
        if (!t) {
          note(op, "a transfer the model cannot place");
          return true;
        }
        SmallVector<int64_t> sizes;
        for (OpFoldResult s : memcpy.getMixedSizes()) {
          std::optional<int64_t> v;
          if (auto a = dyn_cast<Attribute>(s))
            v = cast<IntegerAttr>(a).getInt();
          else
            v = self->value(cast<Value>(s));
          if (!v) {
            note(op, "a runtime transfer of runtime size");
            return true;
          }
          sizes.push_back(*v);
        }
        uint64_t elements = 1;
        for (int64_t s : sizes)
          elements *= s;
        uint64_t runs = std::max<int64_t>(sizes.front(), 1);
        TileDMAChannel ch{*t, alloc.getChannelDir(),
                          static_cast<int>(alloc.getChannelIndex())};
        BD bd{wordsOf(elements / runs * memcpy.getElementTypeBitwidth() / 8),
              std::nullopt, std::nullopt, std::nullopt, op};
        if (std::optional<PacketInfoAttr> p = memcpy.getPacket())
          bd.packet = p->assignedId();
        else if (std::optional<PacketInfoAttr> p = alloc.getPacket())
          bd.packet = p->assignedId();
        Chain chain;
        chain.bds.push_back(bd);
        chain.passes = runs;
        chain.token = ch.dir == DMAChannelDir::S2MM || memcpy.getIssueToken();
        chain.op = op;
        pushed.push_back(chain);
        host.push_back({HostOp::Kind::Push, ch,
                        static_cast<unsigned>(pushed.size() - 1), 0, 0, op});
      } else if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
        auto alloc = ShimDMAAllocationOp::getForSymbol(
            device, task.getAlloc().getRootReference());
        std::optional<TileID> t =
            alloc ? tileOf(alloc.getTile()) : std::nullopt;
        if (!t || task.getRepeatCountVal()) {
          note(op, "a task the model cannot place or count");
          return true;
        }
        Chain chain = taskChain(op, task.getBody(), task.getIssueToken(),
                                task.getRepeatCount());
        if (std::optional<PacketInfoAttr> p = alloc.getPacket())
          for (BD &bd : chain.bds)
            if (!bd.packet)
              bd.packet = p->assignedId();
        pushed.push_back(chain);
        tasks[task.getResult()] = pushed.size() - 1;
        channelOfTask[pushed.size() - 1] = {
            *t, alloc.getChannelDir(),
            static_cast<int>(alloc.getChannelIndex())};
      } else if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op)) {
        std::optional<TileID> t = tileOf(task.getTile());
        if (!t || task.getRepeatCountVal()) {
          note(op, "a task the model cannot place or count");
          return true;
        }
        Chain chain = taskChain(op, task.getBody(), task.getIssueToken(),
                                task.getRepeatCount());
        if (std::optional<PacketInfoAttr> p = task.getPacket())
          for (BD &bd : chain.bds)
            if (!bd.packet)
              bd.packet = p->assignedId();
        pushed.push_back(chain);
        tasks[task.getResult()] = pushed.size() - 1;
        channelOfTask[pushed.size() - 1] = {
            *t, task.getDirection(), static_cast<int>(task.getChannel())};
      } else if (auto start = dyn_cast<AIEX::DMAStartTaskOp>(op)) {
        auto it = tasks.find(start.getTask());
        if (it == tasks.end()) {
          note(op, "a started task the model cannot see");
          return true;
        }
        host.push_back({HostOp::Kind::Push, channelOfTask[it->second],
                        it->second, 0, 0, op});
      } else if (auto await = dyn_cast<AIEX::DMAAwaitTaskOp>(op)) {
        auto it = tasks.find(await.getTask());
        if (it == tasks.end()) {
          note(op, "an awaited task the model cannot see");
          return true;
        }
        host.push_back({HostOp::Kind::Sync, channelOfTask[it->second],
                        it->second, 0, 0, op});
      } else if (auto free = dyn_cast<AIEX::DMAFreeTaskOp>(op)) {
        auto it = tasks.find(free.getTask());
        if (it == tasks.end()) {
          note(op, "a freed task the model cannot see");
          return true;
        }
        host.push_back({HostOp::Kind::Free, channelOfTask[it->second],
                        it->second, 0, 0, op});
      } else if (auto wait = dyn_cast<AIEX::NpuDmaWaitOp>(op)) {
        auto alloc =
            ShimDMAAllocationOp::getForSymbol(device, wait.getSymbol());
        std::optional<TileID> t =
            alloc ? tileOf(alloc.getTile()) : std::nullopt;
        if (!t) {
          note(op, "a wait the model cannot place");
          return true;
        }
        host.push_back({HostOp::Kind::Sync,
                        {*t, alloc.getChannelDir(),
                         static_cast<int>(alloc.getChannelIndex())},
                        0,
                        0,
                        0,
                        op});
      } else if (auto set = dyn_cast<AIEX::SetLockOp>(op)) {
        auto lock = dyn_cast_or_null<LockOp>(set.getLock().getDefiningOp());
        if (!lock) {
          note(op, "a lock the model cannot see");
          return true;
        }
        std::optional<int64_t> v = self->value(set.getValue());
        if (!v) {
          note(op, "a lock value the model cannot know");
          return true;
        }
        host.push_back(
            {HostOp::Kind::Set, {}, 0, lockIndex.lookup(lock), *v, op});
      } else {
        note(op, llvm::formatv("{0} in a runtime sequence", op->getName()));
      }
      return true;
    };
    auto count = [&] { return host.size(); };
    Interpreter interp(event, count);
    self = &interp;
    Flow f = interp.run(sequence.getBody().front());
    if (f == Flow::Outside)
      note(interp.where, interp.reason);
    else if (f == Flow::Forever)
      note(sequence, "a runtime sequence that never ends");
  }

  for (const HostOp &op : host)
    if (op.kind == HostOp::Kind::Sync)
      waited.insert(op.channel);

  // Which agents acquire each lock.
  std::map<unsigned, std::set<std::string>> acquirers;
  for (const Core &core : cores)
    for (const LockEvent &ev :
         llvm::concat<const LockEvent>(core.prefix, core.cycle))
      if (ev.action != LockAction::Release)
        acquirers[ev.lock].insert(
            llvm::formatv("core ({0}, {1})", core.tile.col, core.tile.row));
  auto chainAcquires = [&](const Chain &chain, const TileDMAChannel &ch) {
    for (const BD &bd : chain.bds)
      if (bd.acquire)
        acquirers[bd.acquire->lock].insert(describe(ch));
  };
  for (auto &[ch, chain] : statics)
    chainAcquires(chain, ch);
  for (const HostOp &op : host) {
    if (op.kind != HostOp::Kind::Push)
      continue;
    chainAcquires(pushed[op.chain], op.channel);
    if (statics.count(op.channel))
      note(op.op, "a runtime task on a channel the configuration already "
                  "programs");
  }
  for (auto &[lock, agents] : acquirers)
    if (agents.size() > 1)
      note(device, "a lock more than one agent acquires (" + lockNames[lock] +
                       "): which gets it first is not proven");
  for (const HostOp &op : host)
    if (op.kind == HostOp::Kind::Set)
      note(op.op, "the host sets a lock: the order against the agents "
                  "using it is not proven");

  // Every stream end has a program.
  auto programmed = [&](const TileDMAChannel &ch) {
    if (statics.count(ch))
      return true;
    return llvm::any_of(host, [&](const HostOp &op) {
      return op.kind == HostOp::Kind::Push && op.channel == ch;
    });
  };
  for (auto &[send, dsts] : sends) {
    if (!programmed(send.first) &&
        !llvm::any_of(device.getOps<ShimDMAAllocationOp>(), [&](auto a) {
          return tileOf(a.getTile()) == std::optional(send.first.tile) &&
                 a.getChannelDir() == send.first.dir &&
                 a.getChannelIndex() == send.first.channel;
        }))
      note(device, "a stream from " + describe(send.first) +
                       ", which the design does not program");
    for (const TileDMAChannel &d : dsts)
      if (!programmed(d) &&
          !llvm::any_of(device.getOps<ShimDMAAllocationOp>(), [&](auto a) {
            return tileOf(a.getTile()) == std::optional(d.tile) &&
                   a.getChannelDir() == d.dir &&
                   a.getChannelIndex() == d.channel;
          }))
        note(device, "a stream into " + describe(d) +
                         ", which the design does not program");
  }
}

namespace {

/// Words a sending channel has read and not yet sent, bound for one stream.
struct Segment {
  const SmallVector<TileDMAChannel> *receivers;
  uint64_t words;
  bool header;
};

struct ChannelState {
  const DeadlockModel::Chain *chain = nullptr;
  /// The push the running chain came from, or -1 for a static program.
  int push = -1;
  size_t bd = 0;
  int phase = 0;
  uint64_t done = 0;
  uint64_t pass = 0;
  std::deque<std::pair<const DeadlockModel::Chain *, int>> queue;
  std::deque<Segment> readAhead;
  uint64_t readAheadWords = 0;
  uint64_t tokens = 0;
  /// Words buffered in front of this receiving channel.
  uint64_t buffered = 0;
};

} // namespace

DeadlockModel::Outcome DeadlockModel::run(uint64_t buffering,
                                          uint64_t shimBuffering) const {
  SmallVector<int64_t> locks(lockInit);
  SmallVector<size_t> corePc(cores.size(), 0);
  std::map<TileDMAChannel, ChannelState> channels;
  for (auto &[ch, chain] : statics)
    channels[ch].chain = &chain;
  for (auto &[send, dsts] : sends) {
    channels[send.first];
    for (const TileDMAChannel &d : dsts)
      channels[d];
  }
  for (const HostOp &op : host)
    if (op.kind == HostOp::Kind::Push || op.kind == HostOp::Kind::Sync)
      channels[op.channel];
  SmallVector<bool> finished(host.size(), false);
  std::map<unsigned, int> lastPush;
  size_t hostPc = 0;
  SmallVector<Note> freedEarly;

  auto tryLock = [&](const LockEvent &ev) {
    int64_t &v = locks[ev.lock];
    switch (ev.action) {
    case LockAction::AcquireGreaterEqual:
      if (v < ev.value)
        return false;
      v -= ev.value;
      return true;
    case LockAction::Acquire:
      return v == ev.value;
    case LockAction::Release:
      v += ev.value;
      return true;
    }
    return false;
  };
  auto bdWords = [](const BD &bd, bool header) {
    return bd.words + (header ? 1 : 0);
  };
  // The receivers a BD's words reach, and whether its header travels as a
  // packet header (dropped unless a receiver keeps it).
  auto streamOf = [&](const TileDMAChannel &ch, const BD &bd)
      -> std::pair<const SmallVector<TileDMAChannel> *, bool> {
    if (bd.packet)
      if (auto it = sends.find({ch, *bd.packet}); it != sends.end())
        return {&it->second, true};
    if (auto it = sends.find({ch, -1}); it != sends.end())
      return {&it->second, false};
    return {nullptr, false};
  };
  auto taking = [&](const TileDMAChannel &r) -> uint64_t {
    const ChannelState &s = channels[r];
    if (!s.chain || s.phase != 1 || s.chain->bds.empty())
      return 0;
    return s.chain->bds[s.bd].words - s.done;
  };
  auto keeps = [&](const TileDMAChannel &r) {
    auto it = keepsHeader.find(r);
    return it != keepsHeader.end() && it->second;
  };
  // Moves up to `words` words, the first a header if `header`, onto
  // `receivers`; returns how many moved.
  auto deliver = [&](const SmallVector<TileDMAChannel> &receivers,
                     uint64_t words, bool header) -> uint64_t {
    uint64_t k = words;
    for (const TileDMAChannel &r : receivers) {
      ChannelState &rs = channels[r];
      uint64_t room = buffering + taking(r);
      room = room > rs.buffered ? room - rs.buffered : 0;
      uint64_t dropped = header && !keeps(r) ? 1 : 0;
      k = std::min(k, room + dropped);
    }
    if (k == 0)
      return 0;
    for (const TileDMAChannel &r : receivers)
      channels[r].buffered += k - (header && !keeps(r) ? 1 : 0);
    return k;
  };

  auto finishChain = [&](ChannelState &s) {
    if (s.chain->token)
      ++s.tokens;
    if (s.push >= 0)
      finished[s.push] = true;
    s.chain = nullptr;
    s.push = -1;
  };
  auto advanceBd = [&](ChannelState &s) {
    s.phase = 0;
    s.done = 0;
    if (++s.bd < s.chain->bds.size())
      return;
    if (s.chain->loopTo) {
      s.bd = *s.chain->loopTo;
      return;
    }
    s.bd = 0;
    if (++s.pass == s.chain->passes)
      finishChain(s);
  };

  auto stepChannel = [&](const TileDMAChannel &ch, ChannelState &s) {
    bool moved = false;
    // Read-ahead words leave first, in order.
    while (!s.readAhead.empty()) {
      Segment &seg = s.readAhead.front();
      uint64_t k = deliver(*seg.receivers, seg.words, seg.header);
      if (k == 0)
        break;
      moved = true;
      seg.words -= k;
      seg.header = false;
      s.readAheadWords -= k;
      if (seg.words == 0)
        s.readAhead.pop_front();
    }
    while (true) {
      if (!s.chain) {
        if (s.queue.empty())
          return moved;
        auto [chain, push] = s.queue.front();
        s.queue.pop_front();
        s.chain = chain;
        s.push = push;
        s.bd = s.done = s.pass = 0;
        s.phase = 0;
        moved = true;
        continue;
      }
      if (s.chain->bds.empty()) {
        finishChain(s);
        moved = true;
        continue;
      }
      const BD &bd = s.chain->bds[s.bd];
      if (s.phase == 0) {
        if (bd.acquire && !tryLock(*bd.acquire))
          return moved;
        s.phase = 1;
        s.done = 0;
        moved = true;
        continue;
      }
      if (s.phase == 2) {
        if (bd.release)
          tryLock(*bd.release);
        advanceBd(s);
        moved = true;
        continue;
      }
      if (ch.dir == DMAChannelDir::S2MM) {
        uint64_t k = std::min(s.buffered, bd.words - s.done);
        if (k == 0)
          return moved;
        s.buffered -= k;
        s.done += k;
        if (s.done == bd.words)
          s.phase = 2;
        moved = true;
        continue;
      }
      auto [receivers, isPacket] = streamOf(ch, bd);
      if (!receivers)
        return moved;
      bool header = isPacket && s.done == 0;
      uint64_t total = bdWords(bd, isPacket);
      uint64_t k = 0;
      if (s.readAhead.empty())
        k = deliver(*receivers, total - s.done, header);
      s.done += k;
      uint64_t cap = ch.tile.row == 0 ? shimBuffering : 0;
      if (s.done < total && cap > s.readAheadWords) {
        uint64_t r = std::min(total - s.done, cap - s.readAheadWords);
        s.readAhead.push_back({receivers, r, header && k == 0});
        s.readAheadWords += r;
        s.done += r;
        k += r;
      }
      if (k == 0)
        return moved;
      moved = true;
      if (s.done == total)
        s.phase = 2;
    }
  };

  auto stepCore = [&](size_t i) {
    const Core &core = cores[i];
    bool moved = false;
    while (true) {
      size_t pc = corePc[i];
      const LockEvent *ev = nullptr;
      if (pc < core.prefix.size())
        ev = &core.prefix[pc];
      else if (!core.cycle.empty())
        ev = &core.cycle[(pc - core.prefix.size()) % core.cycle.size()];
      if (!ev || !tryLock(*ev))
        return moved;
      ++corePc[i];
      moved = true;
    }
  };

  auto stepHost = [&]() {
    bool moved = false;
    while (hostPc < host.size()) {
      const HostOp &op = host[hostPc];
      if (op.kind == HostOp::Kind::Push) {
        ChannelState &s = channels[op.channel];
        if (s.queue.size() + (s.chain ? 1 : 0) >= 4)
          return moved;
        s.queue.push_back({&pushed[op.chain], static_cast<int>(hostPc)});
        lastPush[op.chain] = hostPc;
      } else if (op.kind == HostOp::Kind::Sync) {
        ChannelState &s = channels[op.channel];
        if (s.tokens == 0)
          return moved;
        --s.tokens;
      } else if (op.kind == HostOp::Kind::Free) {
        auto it = lastPush.find(op.chain);
        if (it == lastPush.end() || !finished[it->second])
          freedEarly.push_back(
              {op.op, "frees a task that may still be running"});
      } else {
        locks[op.lock] = op.value;
      }
      ++hostPc;
      moved = true;
    }
    return moved;
  };

  // The whole state, to find a run that cycles without ever settling.
  auto encode = [&]() {
    std::vector<int64_t> v(locks.begin(), locks.end());
    v.push_back(hostPc);
    for (auto [i, core] : llvm::enumerate(cores)) {
      size_t pc = corePc[i];
      if (pc >= core.prefix.size() && !core.cycle.empty())
        pc = core.prefix.size() + (pc - core.prefix.size()) % core.cycle.size();
      v.push_back(pc);
    }
    for (auto &[ch, s] : channels) {
      v.push_back(reinterpret_cast<intptr_t>(s.chain));
      v.insert(v.end(),
               {s.push, static_cast<int64_t>(s.bd), s.phase,
                static_cast<int64_t>(s.done), static_cast<int64_t>(s.pass),
                static_cast<int64_t>(s.tokens),
                static_cast<int64_t>(s.buffered),
                static_cast<int64_t>(s.queue.size()),
                static_cast<int64_t>(s.readAhead.size())});
      for (auto &[chain, push] : s.queue)
        v.insert(v.end(), {reinterpret_cast<intptr_t>(chain), push});
      for (const Segment &seg : s.readAhead)
        v.insert(v.end(), {reinterpret_cast<intptr_t>(seg.receivers),
                           static_cast<int64_t>(seg.words), seg.header});
    }
    return v;
  };
  std::set<std::vector<int64_t>> seen;
  bool cycling = false;
  for (bool progress = true; progress;) {
    progress = stepHost();
    for (size_t i = 0; i < cores.size(); ++i)
      progress |= stepCore(i);
    for (auto &[ch, s] : channels)
      progress |= stepChannel(ch, s);
    if (progress && !seen.insert(encode()).second) {
      cycling = true;
      break;
    }
  }

  Outcome outcome;
  outcome.deadlock = hostPc < host.size();
  auto lockName = [&](const LockEvent &ev) {
    return llvm::formatv("{0} (holds {1}, wants {2})", lockNames[ev.lock],
                         locks[ev.lock], ev.value)
        .str();
  };
  if (outcome.deadlock) {
    if (cycling)
      outcome.blocked.push_back(
          {nullptr, "the array cycles without ever letting the host on"});
    const HostOp &op = host[hostPc];
    if (op.kind == HostOp::Kind::Push)
      outcome.blocked.push_back({op.op, "the host waits for room in " +
                                            describe(op.channel) +
                                            "'s task queue"});
    else
      outcome.blocked.push_back(
          {op.op, "the host waits for a completion token from " +
                      describe(op.channel)});
    for (auto [i, core] : llvm::enumerate(cores)) {
      size_t pc = corePc[i];
      const LockEvent *ev = nullptr;
      if (pc < core.prefix.size())
        ev = &core.prefix[pc];
      else if (!core.cycle.empty())
        ev = &core.cycle[(pc - core.prefix.size()) % core.cycle.size()];
      if (ev && pc >= core.prefix.size() && core.cycle.size() &&
          (pc - core.prefix.size()) % core.cycle.size() == 0)
        continue;
      if (ev)
        outcome.blocked.push_back(
            {ev->op, llvm::formatv("core ({0}, {1}) waits to acquire {2}",
                                   core.tile.col, core.tile.row, lockName(*ev))
                         .str()});
    }
    for (auto &[ch, s] : channels) {
      if (!s.chain || s.chain->bds.empty())
        continue;
      const BD &bd = s.chain->bds[s.bd];
      if (s.phase == 0 && bd.acquire) {
        outcome.blocked.push_back({bd.op, describe(ch) + " waits to acquire " +
                                              lockName(*bd.acquire)});
      } else if (s.phase == 1 && ch.dir == DMAChannelDir::MM2S) {
        outcome.blocked.push_back(
            {bd.op,
             llvm::formatv("{0} waits to send {1} more words", describe(ch),
                           bdWords(bd, bd.packet.has_value()) - s.done)
                 .str()});
      } else if (s.phase == 1 && s.done > 0) {
        outcome.blocked.push_back(
            {bd.op, llvm::formatv("{0} waits for {1} more words", describe(ch),
                                  bd.words - s.done)
                        .str()});
      }
    }
    return outcome;
  }
  outcome.inFlight = freedEarly;
  if (cycling)
    outcome.inFlight.push_back(
        {nullptr, "the array keeps moving data after the dispatch ends"});
  for (auto &[ch, s] : channels) {
    if (!s.queue.empty() || (s.chain && s.push >= 0))
      outcome.inFlight.push_back(
          {s.chain ? s.chain->op : s.queue.front().first->op,
           describe(ch) + " is still running a task"});
    else if (s.chain && !s.chain->bds.empty() && s.phase == 1 && s.done > 0)
      outcome.inFlight.push_back(
          {s.chain->bds[s.bd].op,
           describe(ch) + " stopped partway through a BD"});
    if (s.readAheadWords)
      outcome.inFlight.push_back(
          {nullptr, llvm::formatv("{0} still holds {1} words it read ahead",
                                  describe(ch), s.readAheadWords)
                        .str()});
    if (s.buffered)
      outcome.inFlight.push_back(
          {nullptr, llvm::formatv("{0} has {1} words in front of it nobody "
                                  "took",
                                  describe(ch), s.buffered)
                        .str()});
    if (s.tokens)
      outcome.leftoverTokens[ch] = s.tokens;
  }
  return outcome;
}
