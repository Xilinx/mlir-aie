//===- AIECheckCoreStartGating.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Proves which MM2S channels and cores of a configuration cannot act before the
// configuration is delivered, so control packet reconfiguration may start them
// before the other columns' packets land.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FormatVariadic.h"

#include <map>
#include <optional>
#include <set>
#include <string>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECHECKCORESTARTGATING
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// std::nullopt when proved, otherwise the reason it is not.
using Verdict = std::optional<std::string>;

// Whether a device's agents can act before its configuration is delivered. The
// host cannot, since its writes follow the configuration, and neither can a
// core, which is enabled after it unless proved quiet. An MM2S channel whose
// first buffer descriptor does not wait can, and so can whatever its data sets
// off.
struct Checker {
  explicit Checker(DeviceOp device) : device(device) {}

  DeviceOp device;
  // Every release of a lock, keyed on the aie.lock.
  DenseMap<Value, SmallVector<UseLockOp>> releases;
  // (tile, MM2S?, channel) -> the channel's aie.dma_start.
  std::map<std::tuple<Operation *, bool, int>, DMAStartOp> starts;
  // BD block -> (tile, MM2S?, channel) of the chain it is on.
  DenseMap<Block *, std::tuple<Operation *, bool, int>> blockChannel;
  // A source of a stream into an S2MM channel.
  struct Source {
    Operation *tile;
    WireBundle bundle;
    int channel;
  };
  // (tile, S2MM channel) -> the sources streams come from; a null tile is
  // one the device does not declare.
  std::map<std::pair<Operation *, int>, SmallVector<Source>> sources;

  // Tiles whose core sends on a stream or a cascade.
  DenseSet<Operation *> coreSends;

  DenseMap<Value, Verdict> lockMemo;
  DenseSet<Value> lockBusy;

  static LockOp lockOf(Value lock) {
    return dyn_cast_or_null<LockOp>(lock.getDefiningOp());
  }

  void collect() {
    device.walk([&](UseLockOp u) {
      if (u.release())
        releases[u.getLock()].push_back(u);
    });
    device.walk([&](DMAStartOp s) {
      auto index = dyn_cast<IntegerAttr>(s.getChannel());
      if (!index)
        return;
      Operation *tile = s->getParentOp()->getOperand(0).getDefiningOp();
      std::tuple<Operation *, bool, int> key{tile, s.isSend(),
                                             static_cast<int>(index.getInt())};
      starts[key] = s;
      SmallVector<Block *> work{s.getDest()};
      DenseSet<Block *> seen;
      while (!work.empty()) {
        Block *b = work.pop_back_val();
        if (!seen.insert(b).second)
          continue;
        blockChannel[b] = key;
        if (auto next = dyn_cast<NextBDOp>(b->getTerminator()))
          work.push_back(next.getDest());
      }
    });
    // The streams into S2MM channels, as the flows ask for them before routing
    // and as the switchboxes route them after.
    DenseMap<TileID, Operation *> tiles;
    device.walk([&](TileOp t) { tiles[{t.colIndex(), t.rowIndex()}] = t; });
    auto addStream = [&](const RoutedStream &s) {
      if (s.src.port.bundle == WireBundle::Core)
        if (Operation *src = tiles.lookup(s.src.tile))
          coreSends.insert(src);
      Operation *dst = tiles.lookup(s.dst.tile);
      if (s.dst.port.bundle != WireBundle::DMA || !dst)
        return;
      sources[{dst, s.dst.port.channel}].push_back(
          {tiles.lookup(s.src.tile), s.src.port.bundle, s.src.port.channel});
    };
    for (const RoutedStream &s : requestedStreams(device))
      addStream(s);
    for (const RoutedStream &s : traceRoutedStreams(device))
      addStream(s);
    device.walk([&](CascadeFlowOp c) {
      coreSends.insert(c.getSourceTile().getDefiningOp());
    });
    device.walk([&](ConfigureCascadeOp c) {
      coreSends.insert(c.getTile().getDefiningOp());
    });
  }

  // Whether an acquire cannot succeed on the lock's initial value; nullopt if
  // the value or the initial value is unknown. The configuration sets a lock
  // to its initializer when it has one and an id, and zeroes every lock of a
  // tile with a core (AIERTControl::initLocks).
  static std::optional<bool> gates(UseLockOp u) {
    auto c = u.getValue().getDefiningOp<arith::ConstantOp>();
    auto value = c ? dyn_cast<IntegerAttr>(c.getValue()) : nullptr;
    LockOp l = lockOf(u.getLock());
    if (!value || !l)
      return std::nullopt;
    TileOp t = l.getTileOp();
    std::optional<int> init;
    if (l.getInit() && l.getLockID())
      init = *l.getInit();
    else if (!t.isShimTile() && t.getCoreOp())
      init = 0;
    if (!init)
      return std::nullopt;
    return *init < value.getInt();
  }

  // A channel cannot start moving data when its first buffer descriptor waits
  // on a lock that nothing acting during the configuration releases.
  Verdict channelGated(Operation *tile, int channel) {
    auto it = starts.find({tile, true, channel});
    if (it == starts.end())
      return std::string("an MM2S channel with no aie.dma_start");
    Block *first = it->second.getDest();
    for (UseLockOp u : first->getOps<UseLockOp>()) {
      if (!u.acquire() && !u.acquireGE())
        continue;
      if (!u.getAcqEn() ||
          (u.getBlocking() && *u.getBlocking() == LockBlocking::NonBlocking))
        return std::string("its first buffer descriptor's acquire does not "
                           "wait");
      LockOp l = lockOf(u.getLock());
      if (!l)
        return std::string("a lock not resolved to aie.lock");
      if (l.getTileOp().getOperation() != tile)
        return std::string("its first buffer descriptor acquires another "
                           "tile's lock");
      std::optional<bool> g = gates(u);
      if (!g)
        return std::string("a lock value or initial value that is not a "
                           "constant");
      if (!*g)
        return std::string("the lock's initial value satisfies its first "
                           "buffer descriptor's acquire");
      return lockQuiet(u.getLock());
    }
    return std::string("its first buffer descriptor does not acquire a lock");
  }

  Verdict lockQuiet(Value lock) {
    if (auto it = lockMemo.find(lock); it != lockMemo.end())
      return it->second;
    if (!lockBusy.insert(lock).second)
      return std::string("a cycle through a lock");
    Verdict v = lockQuietImpl(lock);
    lockBusy.erase(lock);
    return lockMemo[lock] = v;
  }

  // Every release of the lock waits for something that starts only after the
  // configuration. A core releases nothing early, since it is enabled after the
  // configuration or proved by coreProved to wait until then.
  Verdict lockQuietImpl(Value lock) {
    for (UseLockOp r : releases.lookup(lock)) {
      if (r->getParentOfType<CoreOp>())
        continue;
      auto it = blockChannel.find(r->getBlock());
      if (it == blockChannel.end())
        return std::string("a release outside a core or a buffer descriptor "
                           "chain");
      auto [tile, mm2s, channel] = it->second;
      Verdict v = mm2s ? channelGated(tile, channel)
                       : s2mmQuiet(r->getBlock(), tile, channel);
      if (v)
        return v;
    }
    return std::nullopt;
  }

  // An S2MM buffer descriptor completes only when data arrives on its
  // channel. Whether a zero-length one completes with none is not modelled.
  Verdict s2mmQuiet(Block *b, Operation *tile, int channel) {
    auto bds = b->getOps<DMABDOp>();
    if (bds.empty())
      return std::string("an S2MM buffer descriptor block with no aie.dma_bd");
    for (DMABDOp bd : bds) {
      std::optional<int64_t> len = bd.getConstantLen();
      if (!len && !bd.getLen())
        if (auto t = dyn_cast<MemRefType>(bd.getBuffer().getType());
            t && t.hasStaticShape())
          len = t.getNumElements();
      if (!len || *len <= 0)
        return std::string("an S2MM buffer descriptor whose length is not a "
                           "known nonzero constant");
    }
    auto it = sources.find({tile, channel});
    if (it == sources.end() || it->second.empty())
      return std::string("an S2MM channel no flow feeds");
    for (const Source &src : it->second) {
      if (src.bundle == WireBundle::Core)
        continue;
      if (!src.tile)
        return std::string("an S2MM channel fed from a tile the device does "
                           "not declare");
      if (src.bundle != WireBundle::DMA)
        return llvm::formatv("an S2MM channel fed from a {0} port",
                             stringifyWireBundle(src.bundle))
            .str();
      // A shim channel no aie.dma_start of the device starts is started only
      // by the runtime sequence, after the configuration.
      if (cast<TileOp>(src.tile).isShimTile() &&
          !starts.count({src.tile, true, src.channel}))
        continue;
      if (Verdict v = channelGated(src.tile, src.channel))
        return v;
    }
    return std::nullopt;
  }

  // Whether the DMA buffer descriptor in `b` stays idle until the
  // configuration is delivered.
  Verdict bdQuiet(Block *b) {
    auto it = blockChannel.find(b);
    if (it == blockChannel.end())
      return std::string("a buffer descriptor outside a chain");
    auto [tile, mm2s, channel] = it->second;
    return mm2s ? channelGated(tile, channel) : s2mmQuiet(b, tile, channel);
  }

  // A core may be enabled before the rest of the configuration is delivered if,
  // until it passes an acquire that cannot succeed early, it touches only its
  // own tile's buffers, which nothing else touches early either, and sends on
  // no stream.
  Verdict coreProved(CoreOp core) {
    if (core.getElfFile())
      return std::string("a core given as elf_file");
    UseLockOp gate;
    DenseMap<Value, int64_t> known;
    DenseSet<Block *> visited;
    if (Verdict v = walk(core.getBody().front(), core, gate, known, visited))
      return v;
    return lockQuiet(gate.getLock());
  }

  // The integer value `v` has on the path walked so far, if known.
  static std::optional<int64_t> valueOf(Value v,
                                        const DenseMap<Value, int64_t> &known) {
    if (auto it = known.find(v); it != known.end())
      return it->second;
    if (auto c = v.getDefiningOp<arith::ConstantOp>())
      if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
        return i.getInt();
    return std::nullopt;
  }

  // Whether a branch condition holds on the path walked so far, for a
  // comparison of known values (a lowered loop's first test).
  static std::optional<bool>
  conditionOf(Value cond, const DenseMap<Value, int64_t> &known) {
    if (std::optional<int64_t> c = valueOf(cond, known))
      return *c != 0;
    auto cmp = cond.getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return std::nullopt;
    std::optional<int64_t> l = valueOf(cmp.getLhs(), known);
    std::optional<int64_t> r = valueOf(cmp.getRhs(), known);
    if (!l || !r)
      return std::nullopt;
    Type type = cmp.getLhs().getType();
    unsigned width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
    return arith::applyCmpPredicate(cmp.getPredicate(), APInt(width, *l, true),
                                    APInt(width, *r, true));
  }

  // Walk the core's program from `b` in program order until the gating
  // acquire (sets `gate`), following branches whose direction is known.
  Verdict walk(Block &b, CoreOp core, UseLockOp &gate,
               DenseMap<Value, int64_t> &known, DenseSet<Block *> &visited) {
    if (!visited.insert(&b).second)
      return std::string("no gating acquire");
    Operation *tile = core.getTileOp().getOperation();
    for (Operation &op : b) {
      if (auto u = dyn_cast<UseLockOp>(op)) {
        if (u.release())
          return std::string("a release before the gating acquire");
        LockOp l = lockOf(u.getLock());
        if (!l)
          return std::string("a lock not resolved to aie.lock");
        if (l.getTileOp().getOperation() != tile)
          return std::string("an acquire of another tile's lock");
        if (u.getBlocking() && *u.getBlocking() == LockBlocking::NonBlocking)
          return std::string("a non-blocking acquire");
        std::optional<bool> g = gates(u);
        if (!g)
          return std::string(
              "a lock value or initial value that is not a constant");
        if (*g) {
          gate = u;
          return std::nullopt;
        }
        continue; // an acquire the lock's initial value satisfies
      }
      if (auto f = dyn_cast<scf::ForOp>(op)) {
        std::optional<int64_t> lb = getConstantIntValue(f.getLowerBound());
        std::optional<int64_t> ub = getConstantIntValue(f.getUpperBound());
        bool runs =
            lb && ub &&
            (f.getUnsignedCmp() ? (uint64_t)*lb < (uint64_t)*ub : *lb < *ub);
        if (!runs)
          return std::string(
              "a loop that may not run before the gating acquire");
        return walk(*f.getBody(), core, gate, known, visited);
      }
      if (auto br = dyn_cast<cf::BranchOp>(op))
        return follow(br.getDest(), br.getDestOperands(), core, gate, known,
                      visited);
      if (auto br = dyn_cast<cf::CondBranchOp>(op)) {
        std::optional<bool> taken = conditionOf(br.getCondition(), known);
        if (!taken)
          return std::string(
              "a branch that may not be taken before the gating acquire");
        return *taken ? follow(br.getTrueDest(), br.getTrueDestOperands(), core,
                               gate, known, visited)
                      : follow(br.getFalseDest(), br.getFalseDestOperands(),
                               core, gate, known, visited);
      }
      if (op.hasTrait<OpTrait::IsTerminator>())
        break;
      if (Verdict v = preGate(op, core))
        return v;
    }
    return std::string("no gating acquire");
  }

  Verdict follow(Block *dest, OperandRange operands, CoreOp core,
                 UseLockOp &gate, DenseMap<Value, int64_t> &known,
                 DenseSet<Block *> &visited) {
    for (auto [arg, v] : llvm::zip(dest->getArguments(), operands)) {
      if (std::optional<int64_t> c = valueOf(v, known))
        known[arg] = *c;
      else
        known.erase(arg);
    }
    return walk(*dest, core, gate, known, visited);
  }

  // One op before the gating acquire. Effect-free ops pass; an op that touches
  // memory passes when every memref it touches is guarded; a call to an
  // external function passes when every memref it is given is guarded and the
  // core sends on no stream or cascade, since a kernel reaches the array only
  // through its arguments, the core's routed ports and lock intrinsics.
  Verdict preGate(Operation &op, CoreOp core) {
    if (isa<UseLockOp>(op))
      return std::string("a lock op inside a region before the gating acquire");
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || !callee.isExternal())
        return std::string(
            "a call to a function with a body before the gating acquire");
      if (coreSends.contains(core.getTileOp().getOperation()))
        return std::string("a call before the gating acquire on a core that "
                           "sends on a stream or cascade");
      for (Value v : call.getOperands())
        if (isa<BaseMemRefType>(v.getType()))
          if (Verdict r = guardedMemref(v, core))
            return r;
      return std::nullopt;
    }
    if (op.getNumRegions()) {
      if (!op.hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        return llvm::formatv("a region op before the gating acquire: {0}",
                             op.getName().getStringRef())
            .str();
      for (Region &r : op.getRegions())
        for (Block &blk : r)
          for (Operation &in : blk)
            if (!in.hasTrait<OpTrait::IsTerminator>() || in.getNumRegions())
              if (Verdict v = preGate(in, core))
                return v;
      return std::nullopt;
    }
    if (isMemoryEffectFree(&op)) {
      // An op declared Pure can still read memory through a memref operand
      // (aievec.upd); a view only computes an address, and any access through
      // it is checked on its own.
      if (isa<ViewLikeOpInterface>(op))
        return std::nullopt;
      for (Value v : op.getOperands())
        if (isa<BaseMemRefType>(v.getType()))
          if (Verdict r = guardedMemref(v, core))
            return r;
      return std::nullopt;
    }
    auto effecting = dyn_cast<MemoryEffectOpInterface>(op);
    SmallVector<MemoryEffects::EffectInstance> effects;
    if (effecting)
      effecting.getEffects(effects);
    if (!effecting || effects.empty())
      return llvm::formatv("a side effect before the gating acquire: {0}",
                           op.getName().getStringRef())
          .str();
    for (const MemoryEffects::EffectInstance &e : effects) {
      Value v = e.getValue();
      if (!v || !isa<BaseMemRefType>(v.getType()))
        return llvm::formatv("a side effect before the gating acquire: {0}",
                             op.getName().getStringRef())
            .str();
      if (Verdict r = guardedMemref(v, core))
        return r;
    }
    return std::nullopt;
  }

  // Whether a memref value is a view or a select of guarded buffers.
  Verdict guardedMemref(Value v, CoreOp core) {
    if (auto b = v.getDefiningOp<BufferOp>())
      return guardedBuffer(b, core);
    Operation *def = v.getDefiningOp();
    if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(def))
      return guardedMemref(view.getViewSource(), core);
    if (auto sel = dyn_cast_or_null<arith::SelectOp>(def)) {
      if (Verdict r = guardedMemref(sel.getTrueValue(), core))
        return r;
      return guardedMemref(sel.getFalseValue(), core);
    }
    if (def && isa<scf::IndexSwitchOp, scf::IfOp>(def)) {
      unsigned i = cast<OpResult>(v).getResultNumber();
      for (Region &r : def->getRegions())
        for (Block &blk : r)
          if (Verdict rv =
                  guardedMemref(blk.getTerminator()->getOperand(i), core))
            return rv;
      return std::nullopt;
    }
    return std::string("a memref not resolved to an aie.buffer before the "
                       "gating acquire");
  }

  // A buffer is guarded when it is on the core's tile, whose writes land
  // before the enable, and otherwise only buffer descriptors that stay idle
  // until the configuration is delivered touch it.
  Verdict guardedBuffer(BufferOp b, CoreOp core) {
    if (b.getTileOp() != core.getTileOp())
      return std::string("an access to another tile's buffer before the "
                         "gating acquire");
    for (Operation *user : b->getUsers()) {
      if (core->isAncestor(user))
        continue;
      if (!isa<DMABDOp>(user))
        return llvm::formatv("a buffer accessed before the gating acquire is "
                             "also used by {0}",
                             user->getName().getStringRef())
            .str();
      if (Verdict v = bdQuiet(user->getBlock()))
        return v;
    }
    return std::nullopt;
  }
};

struct AIECheckCoreStartGatingPass
    : xilinx::AIEX::impl::AIECheckCoreStartGatingBase<
          AIECheckCoreStartGatingPass> {
  void runOnOperation() override {
    AIEX::AIEXDialect::CtrlPktGatedAttrHelper gated(&getContext());
    getOperation().walk([&](DeviceOp device) {
      Checker checker(device);
      checker.collect();
      device.walk([&](DMAStartOp start) {
        auto index = dyn_cast<IntegerAttr>(start.getChannel());
        if (!start.isSend() || !index)
          return;
        auto t =
            cast<TileOp>(start->getParentOp()->getOperand(0).getDefiningOp());
        int channel = index.getInt();
        if (Verdict v = checker.channelGated(t, channel)) {
          if (gated.isAttrPresent(start))
            gated.removeAttr(start);
          start.emitRemark()
              << "MM2S channel " << channel << " of tile (" << t.getCol()
              << ", " << t.getRow() << ") not proved gated: " << *v;
        } else {
          gated.setAttr(start, UnitAttr::get(&getContext()));
        }
      });
      device.walk([&](CoreOp core) {
        TileOp t = core.getTileOp();
        if (Verdict v = checker.coreProved(core)) {
          if (gated.isAttrPresent(core))
            gated.removeAttr(core);
          core.emitRemark() << "core (" << t.getCol() << ", " << t.getRow()
                            << ") not proved gated: " << *v;
        } else {
          gated.setAttr(core, UnitAttr::get(&getContext()));
        }
      });
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIECheckCoreStartGatingPass() {
  return std::make_unique<AIECheckCoreStartGatingPass>();
}
