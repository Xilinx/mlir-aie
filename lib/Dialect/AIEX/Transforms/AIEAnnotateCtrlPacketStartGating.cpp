//===- AIEAnnotateCtrlPacketStartGating.cpp ---------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
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
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"

#include <map>
#include <optional>
#include <string>
#include <utility>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEANNOTATECTRLPACKETSTARTGATING
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// std::nullopt when proved, otherwise the reason it is not.
using Verdict = std::optional<std::string>;

// Whether a device's agents can act before its configuration is delivered.
struct GatingProver {
  explicit GatingProver(DeviceOp device) : device(device) {}

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

  // The tiles of the cores whose enables may let what a proof gates act: those
  // that release a lock it waits on, or stream into an S2MM channel whose
  // descriptors release one. The proof in progress adds to `starters`, if set.
  DenseMap<Value, SetVector<Operation *>> lockStarters;
  SetVector<Operation *> *starters = nullptr;

  static LockOp lockOf(Value lock) {
    return dyn_cast_or_null<LockOp>(lock.getDefiningOp());
  }

  // Fills the maps above from the device.
  void collect() {
    device.walk([&](UseLockOp u) {
      if (u.release())
        releases[u.getLock()].push_back(u);
    });
    DenseMap<TileID, Operation *> tiles;
    device.walk([&](TileOp t) { tiles[{t.colIndex(), t.rowIndex()}] = t; });
    for (const DmaChannelProgram &p : collectDmaPrograms(device)) {
      auto start = dyn_cast<DMAStartOp>(p.op);
      if (!start || start.getEndpoint())
        continue;
      std::tuple<Operation *, bool, int> key{tiles.lookup(p.dma.tile),
                                             p.dma.dir == DMAChannelDir::MM2S,
                                             p.dma.channel};
      starts[key] = start;
      for (Block *b : p.bds)
        blockChannel[b] = key;
    }
    // The streams into S2MM channels, as the flows ask for them before routing
    // and as the switchboxes route them after.
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

  // The MM2S channel rule of the pass description.
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

  // lockQuietImpl, memoized; a cycle through locks is not proved.
  Verdict lockQuiet(Value lock) {
    if (auto it = lockMemo.find(lock); it != lockMemo.end()) {
      if (starters)
        starters->insert(lockStarters[lock].begin(), lockStarters[lock].end());
      return it->second;
    }
    if (!lockBusy.insert(lock).second)
      return std::string("a cycle through a lock");
    SetVector<Operation *> found;
    SetVector<Operation *> *outer = std::exchange(starters, &found);
    Verdict v = lockQuietImpl(lock);
    starters = outer;
    if (starters)
      starters->insert(found.begin(), found.end());
    lockStarters[lock] = std::move(found);
    lockBusy.erase(lock);
    return lockMemo[lock] = v;
  }

  // Every release of the lock waits for something that starts only after the
  // configuration. A core releases nothing early, since it is enabled after the
  // configuration or proved by coreGated to wait until then.
  Verdict lockQuietImpl(Value lock) {
    for (UseLockOp r : releases.lookup(lock)) {
      if (auto core = r->getParentOfType<CoreOp>()) {
        if (starters)
          starters->insert(core.getTileOp());
        continue;
      }
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
      if (src.bundle == WireBundle::Core) {
        if (starters && src.tile)
          starters->insert(src.tile);
        continue;
      }
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

  // Whether the block's DMA buffer descriptor stays idle until the
  // configuration is delivered.
  Verdict bdQuiet(Block *b) {
    auto it = blockChannel.find(b);
    if (it == blockChannel.end())
      return std::string("a buffer descriptor outside a chain");
    auto [tile, mm2s, channel] = it->second;
    return mm2s ? channelGated(tile, channel) : s2mmQuiet(b, tile, channel);
  }

  // The core rule of the pass description.
  Verdict coreGated(CoreOp core) {
    if (core.getElfFile())
      return std::string("a core given as elf_file");
    UseLockOp gate;
    DenseMap<Value, int64_t> known;
    DenseSet<Block *> visited;
    // Only what releases the gate starts the core.
    SetVector<Operation *> *outer = std::exchange(starters, nullptr);
    Verdict v = walkToGate(core.getBody().front(), core, gate, known, visited);
    starters = outer;
    if (v)
      return v;
    return lockQuiet(gate.getLock());
  }

  // The integer value on the path walked so far, if known.
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

  // Walk the core's program from the block in program order until the gating
  // acquire, which it records, following branches whose direction is known.
  Verdict walkToGate(Block &b, CoreOp core, UseLockOp &gate,
                     DenseMap<Value, int64_t> &known,
                     DenseSet<Block *> &visited) {
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
        return walkToGate(*f.getBody(), core, gate, known, visited);
      }
      if (auto w = dyn_cast<scf::WhileOp>(op)) {
        if (!isForeverLoop(w))
          return std::string(
              "a loop that may not run before the gating acquire");
        return walkToGate(w.getAfter().front(), core, gate, known, visited);
      }
      if (auto br = dyn_cast<cf::BranchOp>(op))
        return followBranch(br.getDest(), br.getDestOperands(), core, gate,
                            known, visited);
      if (auto br = dyn_cast<cf::CondBranchOp>(op)) {
        std::optional<bool> taken = conditionOf(br.getCondition(), known);
        if (!taken)
          return std::string(
              "a branch that may not be taken before the gating acquire");
        return *taken
                   ? followBranch(br.getTrueDest(), br.getTrueDestOperands(),
                                  core, gate, known, visited)
                   : followBranch(br.getFalseDest(), br.getFalseDestOperands(),
                                  core, gate, known, visited);
      }
      if (op.hasTrait<OpTrait::IsTerminator>())
        break;
      if (Verdict v = opBeforeGate(op, core))
        return v;
    }
    return std::string("no gating acquire");
  }

  // Walks into a branch's destination, its arguments taking the known values
  // of the branch's operands.
  Verdict followBranch(Block *dest, OperandRange operands, CoreOp core,
                       UseLockOp &gate, DenseMap<Value, int64_t> &known,
                       DenseSet<Block *> &visited) {
    for (auto [arg, v] : llvm::zip(dest->getArguments(), operands)) {
      if (std::optional<int64_t> c = valueOf(v, known))
        known[arg] = *c;
      else
        known.erase(arg);
    }
    return walkToGate(*dest, core, gate, known, visited);
  }

  // One op before the gating acquire. Effect-free ops pass; an op that touches
  // memory passes when every memref it touches is guarded; a call to an
  // external function passes when every memref it is given is guarded and the
  // core sends on no stream or cascade, since a kernel reaches the array only
  // through its arguments, the core's routed ports and lock intrinsics.
  Verdict opBeforeGate(Operation &op, CoreOp core) {
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
              if (Verdict v = opBeforeGate(in, core))
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

struct AIEAnnotateCtrlPacketStartGatingPass
    : xilinx::AIEX::impl::AIEAnnotateCtrlPacketStartGatingBase<
          AIEAnnotateCtrlPacketStartGatingPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    AIEX::AIEXDialect::StartedByAttrHelper startedBy(ctx);
    getOperation().walk([&](DeviceOp device) {
      GatingProver prover(device);
      prover.collect();
      // Proves `op` gated by `prove` and marks it, or returns why not.
      auto mark = [&](Operation *op, auto prove) -> Verdict {
        SetVector<Operation *> starters;
        prover.starters = &starters;
        Verdict v = prove();
        prover.starters = nullptr;
        if (startedBy.isAttrPresent(op))
          startedBy.removeAttr(op);
        if (v)
          return v;
        SmallVector<Attribute> tiles;
        for (Operation *t : starters)
          tiles.push_back(TileAttr::get(ctx, cast<TileOp>(t).getCol(),
                                        cast<TileOp>(t).getRow()));
        startedBy.setAttr(op, ArrayAttr::get(ctx, tiles));
        return std::nullopt;
      };
      device.walk([&](DMAStartOp start) {
        if (!start.isSend() || start.getEndpoint())
          return;
        auto t = cast<TileElement>(start->getParentOp())
                     .getTile()
                     .getDefiningOp<TileOp>();
        int channel = start.getChannelIndex();
        if (Verdict v =
                mark(start, [&] { return prover.channelGated(t, channel); }))
          start.emitRemark()
              << "MM2S channel " << channel << " of tile (" << t.getCol()
              << ", " << t.getRow() << ") not proved gated: " << *v;
      });
      device.walk([&](CoreOp core) {
        TileOp t = core.getTileOp();
        if (Verdict v = mark(core, [&] { return prover.coreGated(core); }))
          core.emitRemark() << "core (" << t.getCol() << ", " << t.getRow()
                            << ") not proved gated: " << *v;
      });
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEAnnotateCtrlPacketStartGatingPass() {
  return std::make_unique<AIEAnnotateCtrlPacketStartGatingPass>();
}
