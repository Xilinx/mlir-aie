//===- BFP16MatMulOperands.cpp - Operands of bfp16 matmuls --------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns that let AIE2P bfp16 matmuls load their operands as bfp16ebs8
// blocks (72 bytes: per row, an exponent byte then 8 mantissa bytes) instead
// of converting bf16 tiles before every multiply.
//
//===----------------------------------------------------------------------===//

#include "BFP16MatMulOperands.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <numeric>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

namespace {

/// The type of a FIFO stream's address. Loops carry it as a pointer, not as
/// an integer converted at every access.
Type fifoAddrType(MLIRContext *ctx) { return LLVM::LLVMPointerType::get(ctx); }

constexpr int64_t kTile = 64;    // Values in an 8x8 tile.
constexpr int64_t kExpBytes = 8; // Exponent bytes in a block.
// Upper bound used for divisors, also standing for "divisible by anything".
constexpr int64_t kMaxDiv = 1 << 30;

/// Returns the aie.buffer that `v` is, or that `v` reshapes as a whole, or
/// null.
static AIE::BufferOp rootBuffer(Value v) {
  while (Operation *d = v.getDefiningOp()) {
    if (auto b = dyn_cast<AIE::BufferOp>(d))
      return b;
    if (!isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::CastOp>(d))
      return {};
    v = d->getOperand(0);
  }
  return {};
}

static int64_t mulCapped(int64_t a, int64_t b) {
  return (a >= kMaxDiv / std::max<int64_t>(b, 1)) ? kMaxDiv : a * b;
}

static int64_t knownDivisor(Value v, int depth = 0);

/// Returns a known divisor of the affine expression `e`, whose dims and
/// symbols are `operands`.
static int64_t exprDivisor(AffineExpr e, ValueRange operands, unsigned nDims,
                           int depth) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return c.getValue() ? std::abs(c.getValue()) : kMaxDiv;
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return knownDivisor(operands[d.getPosition()], depth + 1);
  if (auto s = dyn_cast<AffineSymbolExpr>(e))
    return knownDivisor(operands[nDims + s.getPosition()], depth + 1);
  auto bin = cast<AffineBinaryOpExpr>(e);
  int64_t l = exprDivisor(bin.getLHS(), operands, nDims, depth);
  int64_t r = exprDivisor(bin.getRHS(), operands, nDims, depth);
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return std::gcd(l, r);
  case AffineExprKind::Mul:
    return mulCapped(l, r);
  case AffineExprKind::Mod:
    // x mod c = x - c * (x floordiv c)
    if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
      return std::gcd(l, std::abs(c.getValue()));
    return 1;
  default:
    return 1;
  }
}

/// Returns a divisor of the index `v` that follows from how it is computed,
/// or 1.
static int64_t knownDivisor(Value v, int depth) {
  if (depth > 16)
    return 1;
  if (std::optional<int64_t> c = getConstantIntValue(v))
    return *c ? std::abs(*c) : kMaxDiv;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    if (!loop)
      return 1;
    if (arg == loop.getInductionVar())
      return std::gcd(knownDivisor(loop.getLowerBound(), depth + 1),
                      knownDivisor(loop.getStep(), depth + 1));
    // An iter_arg incremented by the same value every iteration, such as a
    // pointer the loop advances.
    OpOperand *init = loop.getTiedLoopInit(arg);
    Value next = loop.getTiedLoopYieldedValue(arg)->get();
    auto add = next.getDefiningOp<arith::AddIOp>();
    if (!init || !add || (add.getLhs() != arg && add.getRhs() != arg))
      return 1;
    Value step = add.getLhs() == arg ? add.getRhs() : add.getLhs();
    return std::gcd(knownDivisor(init->get(), depth + 1),
                    knownDivisor(step, depth + 1));
  }
  Operation *d = v.getDefiningOp();
  if (isa<arith::AddIOp, arith::SubIOp>(d))
    return std::gcd(knownDivisor(d->getOperand(0), depth + 1),
                    knownDivisor(d->getOperand(1), depth + 1));
  if (isa<arith::MulIOp>(d))
    return mulCapped(knownDivisor(d->getOperand(0), depth + 1),
                     knownDivisor(d->getOperand(1), depth + 1));
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(d))
    return knownDivisor(d->getOperand(0), depth + 1);
  if (auto ap = dyn_cast<affine::AffineApplyOp>(d))
    return exprDivisor(ap.getAffineMap().getResult(0), ap.getMapOperands(),
                       ap.getAffineMap().getNumDims(), depth);
  return 1;
}

/// Returns the trip count of `loop` if it is static.
static std::optional<int64_t> staticTrips(scf::ForOp loop) {
  std::optional<APInt> n = loop.getStaticTripCount();
  return n ? std::optional<int64_t>(n->getSExtValue()) : std::nullopt;
}

/// Returns the trip count of `loop` if it is static, otherwise 2: a loop
/// whose trip count is only known at run time is assumed to repeat.
static int64_t tripsOrTwo(scf::ForOp loop) {
  return staticTrips(loop).value_or(2);
}

/// A read of a whole tile from an aie.buffer.
struct TileRead {
  vector::TransferReadOp read;
  AIE::BufferOp buf;
};

/// Returns the read that `conv` converts if it reads a contiguous,
/// tile-aligned 8x8 tile of an identity-layout aie.buffer.
static std::optional<TileRead> getTileRead(ToBFP16Op conv) {
  VectorType srcTy = conv.getSource().getType();
  if (srcTy.getNumElements() != kTile)
    return std::nullopt;
  Value src = conv.getSource();
  while (auto sc = src.getDefiningOp<vector::ShapeCastOp>())
    src = sc.getSource();
  auto read = src.getDefiningOp<vector::TransferReadOp>();
  if (!read || read.getMask() || !read.getPermutationMap().isMinorIdentity())
    return std::nullopt;
  if (llvm::any_of(read.getInBoundsValues(), [](bool b) { return !b; }))
    return std::nullopt;
  auto memTy = dyn_cast<MemRefType>(read.getBase().getType());
  VectorType vecTy = read.getVectorType();
  if (!memTy || !memTy.hasStaticShape() || !memTy.getLayout().isIdentity() ||
      vecTy.getNumElements() != kTile)
    return std::nullopt;
  // Contiguous: all vector dims but the outermost cover the corresponding
  // memref dims entirely.
  int64_t vr = vecTy.getRank(), mr = memTy.getRank();
  if (vr > mr)
    return std::nullopt;
  for (int64_t i = 1; i < vr; ++i)
    if (vecTy.getDimSize(i) != memTy.getDimSize(mr - vr + i))
      return std::nullopt;
  AIE::BufferOp buf = rootBuffer(read.getBase());
  if (!buf)
    return std::nullopt;
  auto bufTy = cast<MemRefType>(buf.getType());
  if (!bufTy.getLayout().isIdentity() || !bufTy.getElementType().isBF16() ||
      bufTy.getNumElements() % kTile)
    return std::nullopt;
  // Tile-aligned: the element offset is a multiple of the tile size.
  ArrayRef<int64_t> shape = memTy.getShape();
  int64_t stride = 1, div = kMaxDiv;
  for (int64_t i = mr - 1; i >= 0; --i) {
    div = std::gcd(div, mulCapped(knownDivisor(read.getIndices()[i]), stride));
    stride *= shape[i];
  }
  if (div % kTile)
    return std::nullopt;
  return TileRead{read, buf};
}

/// Builds the element offset of `read` in its identity-layout memref.
static Value linearOffset(OpBuilder &b, Location loc,
                          vector::TransferReadOp read) {
  auto memTy = cast<MemRefType>(read.getBase().getType());
  Value off = arith::ConstantIndexOp::create(b, loc, 0);
  int64_t stride = 1;
  for (int64_t i = memTy.getRank() - 1; i >= 0; --i) {
    Value s = arith::ConstantIndexOp::create(b, loc, stride);
    off = arith::AddIOp::create(
        b, loc, off,
        arith::MulIOp::create(b, loc, read.getIndices()[i], s).getResult());
    stride *= memTy.getDimSize(i);
  }
  return off;
}

/// The locks that guard a buffer: for each DMA buffer descriptor that moves
/// it, the lock acquired last before it and the lock released first after
/// it, along the descriptor chain (blocks joined by aie.next_bd). If another
/// core uses the buffer, any lock may guard it.
struct BufferLocks {
  DenseSet<Value> locks;
  bool any = false;

  bool contains(Value lock) const { return any || locks.contains(lock); }
};

static BufferLocks bufferLocks(AIE::BufferOp buf, Operation *core) {
  BufferLocks bl;
  SmallVector<Value> work = {buf.getBuffer()};
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (isa<ViewLikeOpInterface>(user))
        work.push_back(user->getResult(0));
      else if (auto c = user->getParentOfType<AIE::CoreOp>(); c && c != core)
        bl.any = true;
    }
  }
  auto dev = buf->getParentOfType<AIE::DeviceOp>();
  if (!dev)
    return bl;
  // The chain's next and previous descriptor blocks.
  auto next = [](Block *b) -> Block * {
    auto n = dyn_cast<AIE::NextBDOp>(b->getTerminator());
    return n ? n.getDest() : nullptr;
  };
  auto prev = [&](Block *b) -> Block * {
    for (Block *p : b->getPredecessors())
      if (next(p) == b)
        return p;
    return nullptr;
  };
  // The first lock op that `want` accepts from `bd` on, backward or forward
  // along the chain.
  auto scan = [&](AIE::DMABDOp bd, bool forward,
                  function_ref<bool(AIE::UseLockOp)> want) -> Value {
    Block *blk = bd->getBlock();
    Operation *op = forward ? bd->getNextNode() : bd->getPrevNode();
    for (unsigned n = 0; blk && n < 64; ++n) {
      for (; op; op = forward ? op->getNextNode() : op->getPrevNode())
        if (auto ul = dyn_cast<AIE::UseLockOp>(op); ul && want(ul))
          return ul.getLock();
      blk = forward ? next(blk) : prev(blk);
      if (blk)
        op = forward ? &blk->front() : &blk->back();
    }
    return Value();
  };
  dev.walk([&](AIE::DMABDOp bd) {
    if (rootBuffer(bd.getBuffer()) != buf)
      return;
    for (Value lock : {scan(bd, /*forward=*/false,
                            [](AIE::UseLockOp ul) {
                              return ul.acquire() || ul.acquireGE();
                            }),
                       scan(bd, /*forward=*/true,
                            [](AIE::UseLockOp ul) { return ul.release(); })})
      if (lock)
        bl.locks.insert(lock);
  });
  return bl;
}

/// MLIR's local alias analysis, which also knows that aie.buffer ops (which
/// declare no allocation effect) are distinct memory: from each other and
/// from allocations.
class BufferAliasAnalysis : public LocalAliasAnalysis {
protected:
  AliasResult aliasImpl(Value lhs, Value rhs) override {
    AliasResult r = LocalAliasAnalysis::aliasImpl(lhs, rhs);
    if (!r.isMay())
      return r;
    auto root = [](Value v) {
      while (auto view = v.getDefiningOp<ViewLikeOpInterface>())
        v = view.getViewSource();
      return v;
    };
    Operation *a = root(lhs).getDefiningOp(), *b = root(rhs).getDefiningOp();
    auto isBuffer = [](Operation *op) {
      return isa_and_nonnull<AIE::BufferOp>(op);
    };
    auto isMemory = [](Operation *op) {
      return isa_and_nonnull<AIE::BufferOp, memref::AllocOp, memref::AllocaOp,
                             memref::GetGlobalOp>(op);
    };
    if (a != b && (isBuffer(a) || isBuffer(b)) && isMemory(a) && isMemory(b))
      return AliasResult::NoAlias;
    return r;
  }
};

/// Returns true if `op` itself may write or free memory that `mem` views
/// (the ops inside it are checked on their own).
static bool mayWrite(Operation *op, Value mem) {
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return false;
  return BufferAliasAnalysis().getModRef(op, mem).isMod();
}

/// Returns true if `op` may modify `buf`: it writes memory that may alias it,
/// has unknown effects, or acquires a lock of a DMA that fills `buf`.
static bool mayChange(Operation *op, Value mem, const BufferLocks &locks) {
  if (auto ul = dyn_cast<AIE::UseLockOp>(op))
    return (ul.acquire() || ul.acquireGE()) && locks.contains(ul.getLock());
  return mayWrite(op, mem);
}

static bool loopMayChange(scf::ForOp loop, Value mem,
                          const BufferLocks &locks) {
  return loop.getBody()
      ->walk([&](Operation *op) {
        return mayChange(op, mem, locks) ? WalkResult::interrupt()
                                         : WalkResult::advance();
      })
      .wasInterrupted();
}

/// Returns how often one execution of `loop` executes `op`.
static int64_t runsPerExecution(Operation *op, scf::ForOp loop) {
  int64_t n = 1;
  for (auto l = op->getParentOfType<scf::ForOp>(); l;
       l = l->getParentOfType<scf::ForOp>()) {
    n = mulCapped(n, tripsOrTwo(l));
    if (l == loop)
      break;
  }
  return n;
}

/// Returns true if the tile's buffers, its core's stack and `extra` more
/// bytes fit in the tile's local memory.
static bool fitsInCoreMemory(AIE::BufferOp buf, int64_t extra) {
  auto dev = buf->getParentOfType<AIE::DeviceOp>();
  auto tile = buf.getTile().getDefiningOp<AIE::TileOp>();
  if (!dev || !tile)
    return false;
  int64_t bytes = extra;
  dev.walk([&](AIE::BufferOp b) {
    if (b.getTile() == buf.getTile())
      bytes += b.getAllocationSize();
  });
  dev.walk([&](AIE::CoreOp core) {
    if (core.getTile() == buf.getTile())
      bytes += core.getEffectiveStackSize();
  });
  return bytes <= int64_t(dev.getTargetModel().getLocalMemorySize());
}

/// Returns the index `v` as an affine expression whose symbols are the values
/// in `syms`. Values that are not affine in others become new symbols.
static AffineExpr toExpr(Value v, SmallVectorImpl<Value> &syms, int depth = 0) {
  MLIRContext *ctx = v.getContext();
  if (std::optional<int64_t> c = getConstantIntValue(v))
    return getAffineConstantExpr(*c, ctx);
  if (depth < 12) {
    if (auto ap = v.getDefiningOp<affine::AffineApplyOp>()) {
      AffineMap m = ap.getAffineMap();
      ValueRange ops = ap.getMapOperands();
      SmallVector<AffineExpr> dims, symbols;
      for (unsigned i = 0; i < m.getNumDims(); ++i)
        dims.push_back(toExpr(ops[i], syms, depth + 1));
      for (unsigned i = 0; i < m.getNumSymbols(); ++i)
        symbols.push_back(toExpr(ops[m.getNumDims() + i], syms, depth + 1));
      return m.getResult(0).replaceDimsAndSymbols(dims, symbols);
    }
    if (auto add = v.getDefiningOp<arith::AddIOp>())
      return toExpr(add.getLhs(), syms, depth + 1) +
             toExpr(add.getRhs(), syms, depth + 1);
    if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
      if (std::optional<int64_t> c = getConstantIntValue(mul.getRhs()))
        return toExpr(mul.getLhs(), syms, depth + 1) * *c;
      if (std::optional<int64_t> c = getConstantIntValue(mul.getLhs()))
        return toExpr(mul.getRhs(), syms, depth + 1) * *c;
    }
  }
  auto it = llvm::find(syms, v);
  if (it == syms.end()) {
    syms.push_back(v);
    it = std::prev(syms.end());
  }
  return getAffineSymbolExpr(it - syms.begin(), ctx);
}

/// Splits `v` into a value plus a constant.
static std::pair<Value, int64_t> splitConstant(Value v) {
  int64_t c = 0;
  while (auto add = v.getDefiningOp<arith::AddIOp>()) {
    if (std::optional<int64_t> x = getConstantIntValue(add.getRhs())) {
      c += *x;
      v = add.getLhs();
    } else if (std::optional<int64_t> x = getConstantIntValue(add.getLhs())) {
      c += *x;
      v = add.getRhs();
    } else {
      break;
    }
  }
  return {v, c};
}

/// Evaluates `exprs` for every combination of values of `syms`, which must all
/// be induction variables of loops with static bounds. Returns nullopt if
/// that is not the case or if there are too many combinations.
static std::optional<SmallVector<SmallVector<int64_t>>>
valuesOver(ArrayRef<AffineExpr> exprs, ArrayRef<Value> syms) {
  SmallVector<SmallVector<int64_t>> ranges;
  int64_t total = 1;
  for (Value v : syms) {
    auto arg = dyn_cast<BlockArgument>(v);
    auto loop =
        arg ? dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()) : nullptr;
    if (!loop || loop.getInductionVar() != arg)
      return std::nullopt;
    std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
    std::optional<int64_t> st = getConstantIntValue(loop.getStep());
    std::optional<int64_t> n = staticTrips(loop);
    if (!lb || !st || !n || *n <= 0 || *n > 4096 / total)
      return std::nullopt;
    total *= *n;
    SmallVector<int64_t> r;
    for (int64_t i = 0; i < *n; ++i)
      r.push_back(*lb + i * *st);
    ranges.push_back(std::move(r));
  }
  SmallVector<SmallVector<int64_t>> out;
  SmallVector<size_t> at(syms.size(), 0);
  for (int64_t n = 0; n < total; ++n) {
    SmallVector<AffineExpr> vals;
    for (size_t i = 0; i < syms.size(); ++i)
      vals.push_back(
          getAffineConstantExpr(ranges[i][at[i]], syms[i].getContext()));
    SmallVector<int64_t> row;
    for (AffineExpr e : exprs) {
      auto c = dyn_cast<AffineConstantExpr>(
          simplifyAffineExpr(e.replaceSymbols(vals), 0, 0));
      if (!c)
        return std::nullopt;
      row.push_back(c.getValue());
    }
    out.push_back(std::move(row));
    for (size_t i = 0; i < syms.size(); ++i) {
      if (++at[i] < ranges[i].size())
        break;
      at[i] = 0;
    }
  }
  return out;
}

/// The order of the T blocks in a bfp16 copy. A loop of K iterations that
/// reads r tiles per iteration, d tiles apart, and advances s tiles per
/// iteration, reads tile t = s*k + d*i + rest as block rest*r*K + r*k + i, so
/// it reads the blocks in order. With s = 0, tile t is block t.
struct Layout {
  int64_t s = 0, K = 1, d = 1, r = 1, T = 0;

  bool operator==(const Layout &o) const {
    return s == o.s && K == o.K && d == o.d && r == o.r && T == o.T;
  }
  bool identity() const { return s == 0; }

  /// Returns the (stride, extent) of the lower and the higher of the two
  /// digits of t that the loop walks.
  std::pair<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>>
  digits() const {
    std::pair<int64_t, int64_t> k = {s, K}, i = {d, r};
    return s < d ? std::pair(k, i) : std::pair(i, k);
  }

  static std::optional<Layout> make(int64_t s, int64_t K, int64_t d, int64_t r,
                                    int64_t T) {
    Layout l{s, K, d, r, T};
    if (s <= 0 || K <= 0 || d <= 0 || r <= 0 || T <= 0)
      return std::nullopt;
    auto [lo, hi] = l.digits();
    if (hi.first % (lo.first * lo.second) || T % (hi.first * hi.second))
      return std::nullopt;
    return l;
  }

  /// Builds the block index of tile `t`.
  Value block(OpBuilder &b, Location loc, Value t) const {
    if (identity())
      return t;
    auto c = [&](int64_t v) -> Value {
      return arith::ConstantIndexOp::create(b, loc, v);
    };
    auto div = [&](Value x, int64_t v) -> Value {
      return v == 1 ? x : arith::DivUIOp::create(b, loc, x, c(v)).getResult();
    };
    auto rem = [&](Value x, int64_t v) -> Value {
      return arith::RemUIOp::create(b, loc, x, c(v));
    };
    auto mul = [&](Value x, int64_t v) -> Value {
      return v == 1 ? x : arith::MulIOp::create(b, loc, x, c(v)).getResult();
    };
    auto add = [&](Value x, Value y) -> Value {
      return arith::AddIOp::create(b, loc, x, y);
    };
    auto [lo, hi] = digits();
    int64_t m = hi.first / (lo.first * lo.second);
    Value k = rem(div(t, s), K), i = rem(div(t, d), r);
    Value rest =
        add(rem(t, lo.first), mul(add(rem(div(t, lo.first * lo.second), m),
                                      mul(div(t, hi.first * hi.second), m)),
                                  lo.first));
    return add(mul(rest, r * K), add(mul(k, r), i));
  }
};

/// A tile read at an iter_arg index that starts at `init` and advances by
/// `step` tiles every iteration of `loop`.
struct SteppedRead {
  scf::ForOp loop;
  Value init;
  int64_t step;
  int64_t trips;
};

static std::optional<SteppedRead> steppedRead(vector::TransferReadOp rd) {
  auto loop = dyn_cast<scf::ForOp>(rd->getParentOp());
  if (!loop || rd.getIndices().size() != 1)
    return std::nullopt;
  auto arg = dyn_cast<BlockArgument>(rd.getIndices()[0]);
  if (!arg || arg.getOwner() != loop.getBody() || arg.getArgNumber() == 0)
    return std::nullopt;
  auto [base, c] = splitConstant(loop.getTiedLoopYieldedValue(arg)->get());
  std::optional<int64_t> trips = staticTrips(loop);
  if (base != arg || c <= 0 || c % kTile || !trips || *trips <= 0)
    return std::nullopt;
  return SteppedRead{loop, loop.getTiedLoopInit(arg)->get(), c / kTile, *trips};
}

/// Reads of one buffer in one loop that can share a load stream. Each read is
/// paired with its position among the r blocks of an iteration.
struct Stream {
  scf::ForOp loop;
  SmallVector<std::pair<vector::TransferReadOp, int64_t>> reads;
  Value init; // Element index of the first tile.
};

/// Picks the block order for a copy of T tiles so that a loop advancing
/// through `reads` reads consecutive blocks, and returns the streams this
/// enables.
static std::pair<Layout, SmallVector<Stream>>
chooseLayout(ArrayRef<vector::TransferReadOp> reads, int64_t T) {
  llvm::MapVector<Operation *, SmallVector<vector::TransferReadOp>> byLoop;
  for (vector::TransferReadOp rd : reads)
    if (std::optional<SteppedRead> sr = steppedRead(rd))
      byLoop[sr->loop].push_back(rd);
  std::optional<Layout> chosen;
  SmallVector<Stream> streams;
  for (auto &[loopOp, rds] : byLoop) {
    SteppedRead first = *steppedRead(rds.front());
    SmallVector<Value> syms;
    SmallVector<AffineExpr> exprs;
    bool ok = true;
    for (vector::TransferReadOp rd : rds) {
      SteppedRead sr = *steppedRead(rd);
      ok &= sr.step == first.step;
      exprs.push_back(toExpr(sr.init, syms));
    }
    if (!ok)
      continue;
    std::optional<SmallVector<SmallVector<int64_t>>> vals =
        valuesOver(exprs, syms);
    if (!vals)
      continue;
    // Each read must be at a fixed tile offset from the first one.
    SmallVector<int64_t> off;
    for (size_t j = 0; j < rds.size(); ++j) {
      int64_t o = (*vals)[0][j] - (*vals)[0][0];
      for (auto &row : *vals)
        ok &= row[j] - row[0] == o && row[j] % kTile == 0;
      off.push_back(o / kTile);
    }
    int64_t lo = *llvm::min_element(off);
    SmallVector<int64_t> distinct;
    for (int64_t &o : off) {
      o -= lo;
      if (!llvm::is_contained(distinct, o))
        distinct.push_back(o);
    }
    llvm::sort(distinct);
    int64_t r = distinct.size(), d = r > 1 ? distinct[1] : 1;
    for (int64_t j = 0; j < r; ++j)
      ok &= distinct[j] == j * d;
    if (!ok || r > 4)
      continue;
    std::optional<Layout> layout =
        Layout::make(first.step, first.trips, d, r, T);
    if (!layout || (chosen && !(*chosen == *layout)))
      continue;
    // Every entry into the loop must start at the first block of a group.
    int64_t lowest = 0;
    for (size_t j = 0; j < rds.size(); ++j)
      if (off[j] == 0)
        lowest = j;
    for (auto &row : *vals) {
      int64_t t = row[lowest] / kTile;
      ok &= t >= 0 && t < T && (t / layout->s) % layout->K == 0 &&
            (t / layout->d) % layout->r == 0;
    }
    if (!ok)
      continue;
    chosen = layout;
    Stream st{cast<scf::ForOp>(loopOp), {}, steppedRead(rds[lowest])->init};
    for (size_t j = 0; j < rds.size(); ++j)
      st.reads.push_back({rds[j], off[j] / d});
    streams.push_back(std::move(st));
  }
  if (!chosen)
    return {Layout{0, 1, 1, 1, T}, {}};
  return {*chosen, std::move(streams)};
}

/// Returns the number of load streams already carried by `loop`.
static int64_t loadStreams(scf::ForOp loop) {
  int64_t n = 0;
  for (Operation &op : *loop.getBody())
    if (auto pop = dyn_cast<FifoLdPopBFP16Op>(op))
      if (auto arg = dyn_cast<BlockArgument>(pop.getAddr()))
        n += arg.getOwner() == loop.getBody();
  return n;
}

struct HoistBFP16ConversionPattern : public OpRewritePattern<ToBFP16Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ToBFP16Op conv,
                                PatternRewriter &rewriter) const override {
    std::optional<TileRead> tr = getTileRead(conv);
    if (!tr)
      return rewriter.notifyMatchFailure(conv, "not a tile read of a buffer");
    AIE::BufferOp buf = tr->buf;
    auto core = conv->getParentOfType<AIE::CoreOp>();
    if (!core)
      return rewriter.notifyMatchFailure(conv, "not in a core");

    // Find the outermost loop around the conversion that does not modify
    // the buffer.
    BufferLocks locks = bufferLocks(buf, core);
    scf::ForOp loop;
    for (auto l = conv->getParentOfType<scf::ForOp>();
         l && core->isProperAncestor(l); l = l->getParentOfType<scf::ForOp>()) {
      if (loopMayChange(l, buf.getBuffer(), locks))
        break;
      loop = l;
    }
    if (!loop)
      return rewriter.notifyMatchFailure(conv, "no loop keeps the buffer");

    // Collect the conversions of the buffer's tiles in that loop. Only
    // operands of matmuls whose other operand is already loaded as bfp16
    // blocks are considered, so that the matmul needs no conversion at all.
    auto partnered = [](ToBFP16Op c) {
      return !c->use_empty() && llvm::all_of(c->getUses(), [&](OpOperand &u) {
        auto mm = dyn_cast<MatMulBFP16Op_AIE2P>(u.getOwner());
        if (!mm)
          return false;
        Value other = u.getOperandNumber() < 2 ? mm.getRhsMantissas()
                                               : mm.getLhsMantissas();
        return isa_and_nonnull<LoadBFP16Op, FifoLdPopBFP16Op>(
            other.getDefiningOp());
      });
    };
    if (!partnered(conv))
      return rewriter.notifyMatchFailure(conv, "the other operand converted");
    SmallVector<std::pair<ToBFP16Op, vector::TransferReadOp>> group;
    int64_t runs = 0;
    loop.getBody()->walk([&](ToBFP16Op c) {
      if (c.getTranspose() != conv.getTranspose() || !partnered(c))
        return;
      std::optional<TileRead> t = getTileRead(c);
      if (!t || t->buf != buf || !loop->isAncestor(t->read))
        return;
      group.push_back({c, t->read});
      runs = std::min(kMaxDiv, runs + runsPerExecution(c, loop));
    });
    // The copy is built each time the loop starts, at about the cost of one
    // conversion per tile, so require at least two conversions per tile in
    // one execution of the loop.
    auto bufTy = cast<MemRefType>(buf.getType());
    int64_t tiles = bufTy.getNumElements() / kTile;
    if (runs < 2 * tiles)
      return rewriter.notifyMatchFailure(conv, "too few conversions per tile");
    if (!fitsInCoreMemory(buf, tiles * (kTile + kExpBytes)))
      return rewriter.notifyMatchFailure(conv, "no room for the bfp16 copy");

    // The copy holds one bfp16 block per tile, in the order the loops read
    // them.
    Location loc = conv.getLoc();
    Attribute space = bufTy.getMemorySpace();
    Type i8 = rewriter.getI8Type(), i32 = rewriter.getI32Type();
    Type addrTy = fifoAddrType(rewriter.getContext());
    auto stateTy = VectorType::get({32}, i32);
    auto mantTy = VectorType::get({kTile}, i8);
    auto expTy = VectorType::get({kExpBytes}, i8);
    AIE::BufferOp copy;
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(buf);
      std::string base =
          (buf.getSymName() ? buf.getSymName()->str() : std::string("buf")) +
          "_bfp16";
      DenseSet<StringRef> taken;
      if (auto dev = buf->getParentOfType<AIE::DeviceOp>())
        dev.walk([&](Operation *op) {
          if (auto sym = op->getAttrOfType<StringAttr>(
                  SymbolTable::getSymbolAttrName()))
            taken.insert(sym.getValue());
        });
      std::string name = base;
      for (int i = 0; taken.contains(name); ++i)
        name = base + "_" + std::to_string(i);
      copy = AIE::BufferOp::create(
          rewriter, buf.getLoc(),
          MemRefType::get({tiles * (kTile + kExpBytes)}, i8, nullptr, space),
          buf.getTile(), rewriter.getStringAttr(name), nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr);
    }
    SmallVector<vector::TransferReadOp> reads;
    for (auto &[c, rd] : group)
      if (!llvm::is_contained(reads, rd))
        reads.push_back(rd);
    auto [layout, streams] = chooseLayout(reads, tiles);
    auto cIdx = [&](OpBuilder &b, Location l, int64_t v) -> Value {
      return arith::ConstantIndexOp::create(b, l, v);
    };
    auto blockOffset = [&](OpBuilder &b, Location l, Value t) -> Value {
      return arith::MulIOp::create(b, l, layout.block(b, l, t),
                                   cIdx(b, l, kTile + kExpBytes));
    };

    // Before the loop, convert every tile and write the blocks in order
    // through a store stream. The loop nest follows the layout's digits so
    // that tile indices are affine in the induction variables.
    rewriter.setInsertionPoint(loop);
    SmallVector<ReassociationIndices> all(1);
    for (int64_t i = 0; i < bufTy.getRank(); ++i)
      all[0].push_back(i);
    Value flat = bufTy.getRank() == 1 ? Value(buf.getBuffer())
                                      : memref::CollapseShapeOp::create(
                                            rewriter, loc, buf.getBuffer(), all)
                                            .getResult();
    SmallVector<std::pair<int64_t, int64_t>> nest; // (extent, tile stride)
    if (layout.identity()) {
      nest.push_back({tiles, 1});
    } else {
      auto [lo, hi] = layout.digits();
      int64_t m = hi.first / (lo.first * lo.second);
      nest.push_back({tiles / (hi.first * hi.second), hi.first * hi.second});
      nest.push_back({m, lo.first * lo.second});
      nest.push_back({lo.first, 1});
      nest.push_back({layout.K, layout.s});
      nest.push_back({layout.r, layout.d});
    }
    // Handle U tiles per innermost iteration and read them all before the
    // first store; later reads are not moved above the stores.
    while (nest.size() > 1 && nest.back().first == 1)
      nest.pop_back();
    int64_t inner = nest.back().first;
    int64_t U = inner % 4 == 0 ? 4 : inner % 2 == 0 ? 2 : 1;
    Value bfZero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(rewriter.getBF16Type()));
    auto tileTy = VectorType::get({kTile}, rewriter.getBF16Type());
    VectorType convTy = conv.getSource().getType();
    UnitAttr transpose = conv.getTransposeAttr();
    std::function<SmallVector<Value>(OpBuilder &, Location, unsigned, Value,
                                     ValueRange)>
        build = [&](OpBuilder &b, Location l, unsigned level, Value t,
                    ValueRange st) -> SmallVector<Value> {
      if (level == nest.size()) {
        int64_t stride = nest.back().second;
        SmallVector<Value> tileVals;
        for (int64_t u = 0; u < U; ++u) {
          Value tu = arith::AddIOp::create(b, l, t, cIdx(b, l, u * stride));
          Value at = arith::MulIOp::create(b, l, tu, cIdx(b, l, kTile));
          tileVals.push_back(
              vector::TransferReadOp::create(b, l, tileTy, flat, ValueRange{at},
                                             bfZero, ArrayRef<bool>{true}));
        }
        SmallVector<Value> cur(st.begin(), st.end());
        for (Value v : tileVals) {
          auto c = ToBFP16Op::create(
              b, l, mantTy, expTy, vector::ShapeCastOp::create(b, l, convTy, v),
              transpose);
          auto push = FifoStPushBFP16Op::create(
              b, l, addrTy, stateTy, i32, cur[0], cur[1], cur[2],
              c.getMantissas(), c.getExponents());
          cur = SmallVector<Value>(push->getResults());
        }
        return cur;
      }
      auto [extent, stride] = nest[level];
      bool last = level + 1 == nest.size();
      int64_t trips = last ? extent / U : extent;
      int64_t step = last ? stride * U : stride;
      if (trips == 1)
        return build(b, l, level + 1, t, st);
      auto nloop = scf::ForOp::create(
          b, l, cIdx(b, l, 0), cIdx(b, l, trips), cIdx(b, l, 1), st,
          [&](OpBuilder &ib, Location il, Value iv, ValueRange ist) {
            Value ti = arith::AddIOp::create(
                ib, il, t,
                arith::MulIOp::create(ib, il, iv, cIdx(ib, il, step)));
            scf::YieldOp::create(ib, il, build(ib, il, level + 1, ti, ist));
          });
      return SmallVector<Value>(nloop.getResults());
    };
    auto start = FifoStartOp::create(rewriter, loc, addrTy, stateTy, i32,
                                     copy.getBuffer(), cIdx(rewriter, loc, 0));
    (void)build(rewriter, loc, 0, cIdx(rewriter, loc, 0),
                ValueRange(start->getResults()));

    // Replaces the conversions of `rd` with the block (m, e).
    auto setBFP16 = [&](vector::TransferReadOp rd, Value m, Value e) {
      for (auto &[c, r] : group)
        if (r == rd) {
          c.getMantissas().replaceAllUsesWith(m);
          c.getExponents().replaceAllUsesWith(e);
        }
    };
    // Read through load streams where possible, at most two per loop (one per
    // load FIFO). The pops of an iteration are emitted together; a fill
    // provides enough data for the next 8 pops.
    SmallPtrSet<Operation *, 16> streamed;
    for (Stream &st : streams) {
      if (loadStreams(st.loop) >= 2)
        continue;
      rewriter.setInsertionPoint(st.loop);
      Location l = st.loop.getLoc();
      Value t0 = arith::DivUIOp::create(rewriter, l, st.init,
                                        cIdx(rewriter, l, kTile));
      auto begin =
          FifoStartOp::create(rewriter, l, addrTy, stateTy, i32,
                              copy.getBuffer(), blockOffset(rewriter, l, t0));
      int64_t r = layout.r;
      (void)st.loop.replaceWithAdditionalYields(
          rewriter, begin->getResults(), /*replaceInitOperandUsesInLoop=*/false,
          [&](OpBuilder &b, Location, ArrayRef<BlockArgument> args) {
            Operation *first = st.reads.front().first;
            for (auto &[rd, _] : st.reads)
              if (rd->isBeforeInBlock(first))
                first = rd;
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPoint(first);
            Value addr = args[0], state = args[1], count = args[2];
            for (int64_t i = 0; i < r; ++i) {
              auto pop = FifoLdPopBFP16Op::create(
                  b, first->getLoc(), addrTy, stateTy, i32, mantTy, expTy, addr,
                  state, count,
                  r % 8 == 0 && i % 8 ? b.getUnitAttr() : UnitAttr());
              for (auto &[rd, place] : st.reads)
                if (place == i)
                  setBFP16(rd, pop.getMantissas(), pop.getExponents());
              addr = pop.getNext();
              state = pop.getStateOut();
              count = pop.getCountOut();
            }
            return SmallVector<Value>{addr, state, count};
          });
      for (auto &[rd, _] : st.reads)
        streamed.insert(rd);
    }
    // Load the remaining blocks individually.
    for (vector::TransferReadOp rd : reads) {
      if (streamed.contains(rd))
        continue;
      rewriter.setInsertionPoint(rd);
      Location l = rd.getLoc();
      Value t = arith::DivUIOp::create(
          rewriter, l, linearOffset(rewriter, l, rd), cIdx(rewriter, l, kTile));
      auto ld =
          LoadBFP16Op::create(rewriter, l, mantTy, expTy, copy.getBuffer(),
                              blockOffset(rewriter, l, t));
      setBFP16(rd, ld.getMantissas(), ld.getExponents());
    }
    for (auto &[c, rd] : group)
      rewriter.eraseOp(c);
    return success();
  }
};

/// Returns the loop whose induction variable is `v` if it runs from 0 to `n`
/// with step 1.
static scf::ForOp loopOver(Value v, int64_t n) {
  auto arg = dyn_cast<BlockArgument>(v);
  auto loop =
      arg ? dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()) : nullptr;
  if (!loop || loop.getInductionVar() != arg ||
      getConstantIntValue(loop.getLowerBound()) != 0 ||
      getConstantIntValue(loop.getStep()) != 1 ||
      getConstantIntValue(loop.getUpperBound()) != n)
    return nullptr;
  return loop;
}

/// A loop nest that decodes the bfp16ebs8 blocks in `bytes` ([..., 8, 9]) into
/// the bf16 tiles of `tiles` ([..., 8, 8]):
///   tiles[i..., r, c] = bf16(sitofp(bytes[p(i...), r, c + 1]) *
///                            bitcast<f32>((zext(bytes[p(i...), r, 0]) - 6) <<
///                            23))
/// for all elements of `tiles`. `bytes` may have extra unit dims, indexed
/// with 0. Dim j of the block grid is dim perm[j] of the tile grid.
struct Decode {
  AIE::BufferOp bytes, tiles;
  scf::ForOp nest;
  SmallVector<unsigned> perm;
  SmallVector<int64_t> blocks;
};

static std::optional<Decode> matchDecode(memref::StoreOp st) {
  auto trunc = st.getValueToStore().getDefiningOp<arith::TruncFOp>();
  auto mul = trunc ? trunc.getIn().getDefiningOp<arith::MulFOp>() : nullptr;
  if (!mul || !trunc.getType().isBF16())
    return std::nullopt;
  arith::SIToFPOp man;
  arith::BitcastOp scale;
  for (auto [x, y] : {std::pair(mul.getLhs(), mul.getRhs()),
                      std::pair(mul.getRhs(), mul.getLhs())})
    if (x.getDefiningOp<arith::SIToFPOp>() &&
        y.getDefiningOp<arith::BitcastOp>()) {
      man = x.getDefiningOp<arith::SIToFPOp>();
      scale = y.getDefiningOp<arith::BitcastOp>();
    }
  if (!man)
    return std::nullopt;
  auto shl = scale.getIn().getDefiningOp<arith::ShLIOp>();
  auto sub = shl ? shl.getLhs().getDefiningOp<arith::SubIOp>() : nullptr;
  auto ext = sub ? sub.getLhs().getDefiningOp<arith::ExtUIOp>() : nullptr;
  auto expLd = ext ? ext.getIn().getDefiningOp<memref::LoadOp>() : nullptr;
  auto manLd = man.getIn().getDefiningOp<memref::LoadOp>();
  if (!expLd || !manLd || getConstantIntValue(shl.getRhs()) != 23 ||
      getConstantIntValue(sub.getRhs()) != 6 || !scale.getType().isF32() ||
      !man.getType().isF32() || expLd.getMemref() != manLd.getMemref())
    return std::nullopt;
  auto bBuf = expLd.getMemref().getDefiningOp<AIE::BufferOp>();
  auto xBuf = st.getMemref().getDefiningOp<AIE::BufferOp>();
  if (!bBuf || !xBuf)
    return std::nullopt;
  auto bt = cast<MemRefType>(bBuf.getType());
  auto xt = cast<MemRefType>(xBuf.getType());
  int64_t n = xt.getRank(), nb = bt.getRank();
  if (!bt.hasStaticShape() || !xt.hasStaticShape() ||
      !bt.getLayout().isIdentity() || !xt.getLayout().isIdentity() || nb < n ||
      n < 3 || !bt.getElementType().isInteger(8) ||
      !xt.getElementType().isBF16() || bt.getDimSize(nb - 2) != 8 ||
      bt.getDimSize(nb - 1) != 9 || xt.getDimSize(n - 2) != 8 ||
      xt.getDimSize(n - 1) != 8)
    return std::nullopt;
  // Each store index must be the induction variable of a loop over the whole
  // corresponding dim.
  SmallVector<scf::ForOp> loops;
  for (int64_t i = 0; i < n; ++i) {
    scf::ForOp l = loopOver(st.getIndices()[i], xt.getDimSize(i));
    if (!l || llvm::is_contained(loops, l))
      return std::nullopt;
    loops.push_back(l);
  }
  scf::ForOp nest = loops.front();
  for (scf::ForOp l : loops)
    if (l->isProperAncestor(nest))
      nest = l;
  // The nest is removed, so nothing may use what it returns.
  if (!nest->use_empty())
    return std::nullopt;
  // Both loads read the same block and row: the exponent at column 0 and the
  // mantissa at column c + 1.
  Decode d{bBuf, xBuf, nest, {}, {}};
  ValueRange tileIdx = st.getIndices().take_front(n - 2);
  for (int64_t j = 0; j < nb - 2; ++j) {
    Value v = manLd.getIndices()[j];
    if (expLd.getIndices()[j] != v)
      return std::nullopt;
    if (bt.getDimSize(j) == 1 && getConstantIntValue(v) == 0)
      continue;
    auto it = llvm::find(tileIdx, v);
    if (it == tileIdx.end() ||
        llvm::is_contained(d.perm, unsigned(it - tileIdx.begin())))
      return std::nullopt;
    d.perm.push_back(it - tileIdx.begin());
    d.blocks.push_back(bt.getDimSize(j));
  }
  if (d.perm.size() != size_t(n - 2) ||
      manLd.getIndices()[nb - 2] != st.getIndices()[n - 2] ||
      expLd.getIndices()[nb - 2] != st.getIndices()[n - 2] ||
      getConstantIntValue(expLd.getIndices()[nb - 1]) != 0)
    return std::nullopt;
  SmallVector<Value> syms;
  AffineExpr off = toExpr(manLd.getIndices()[nb - 1], syms) -
                   toExpr(st.getIndices()[n - 1], syms);
  auto c =
      dyn_cast<AffineConstantExpr>(simplifyAffineExpr(off, 0, syms.size()));
  if (!c || c.getValue() != 1)
    return std::nullopt;
  // Nothing else in the nest may access memory.
  WalkResult clean = nest->walk([&](Operation *op) {
    if (op == st || op == manLd || op == expLd || isPure(op) ||
        isa<scf::ForOp, scf::YieldOp>(op))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  if (clean.wasInterrupted())
    return std::nullopt;
  return d;
}

/// Builds the index of the block in `d.bytes` that holds tile `t`.
static Value blockOf(OpBuilder &b, Location loc, Decode &d, Value t) {
  auto xt = cast<MemRefType>(d.tiles.getType());
  auto c = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  int64_t n = xt.getRank() - 2;
  SmallVector<Value> dig(n);
  for (int64_t i = n - 1; i >= 0; --i) {
    dig[i] =
        i == 0 ? t : arith::RemUIOp::create(b, loc, t, c(xt.getDimSize(i)));
    if (i)
      t = arith::DivUIOp::create(b, loc, t, c(xt.getDimSize(i)));
  }
  Value blk;
  for (size_t j = 0; j < d.blocks.size(); ++j)
    blk = blk ? arith::AddIOp::create(
                    b, loc, arith::MulIOp::create(b, loc, blk, c(d.blocks[j])),
                    dig[d.perm[j]])
                    .getResult()
              : dig[d.perm[j]];
  return blk;
}

/// Makes matmuls load the bfp16 blocks that a core would otherwise decode into
/// bf16 tiles, when the tiles are only read whole by untransposed
/// conversions. The decode loop and the tile buffer are removed, and the
/// release of the block buffer's lock moves after the last read.
struct ReadDecodedBFP16Pattern : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp st,
                                PatternRewriter &rewriter) const override {
    std::optional<Decode> d = matchDecode(st);
    if (!d)
      return rewriter.notifyMatchFailure(st, "not a bfp16 decode");
    auto core = st->getParentOfType<AIE::CoreOp>();
    BufferLocks bytesLocks = bufferLocks(d->bytes, core);
    BufferLocks tileLocks = bufferLocks(d->tiles, core);
    if (bytesLocks.any || bytesLocks.locks.empty() || tileLocks.any ||
        !tileLocks.locks.empty() || d->tiles.getInitialValue())
      return rewriter.notifyMatchFailure(st, "not DMA bytes into core tiles");

    // Every other use of the tiles must be a whole-tile read, after the
    // decode, that only untransposed conversions use.
    SmallVector<Operation *> views;
    SmallVector<std::pair<ToBFP16Op, vector::TransferReadOp>> convs;
    SmallVector<Value> work = {d->tiles.getBuffer()};
    Block *blk = d->nest->getBlock();
    Operation *last = d->nest;
    while (!work.empty()) {
      Value v = work.pop_back_val();
      for (Operation *user : v.getUsers()) {
        if (user == st.getOperation())
          continue;
        if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::CastOp>(
                user)) {
          views.push_back(user);
          work.push_back(user->getResult(0));
          continue;
        }
        auto rd = dyn_cast<vector::TransferReadOp>(user);
        Operation *anc = rd ? blk->findAncestorOpInBlock(*rd) : nullptr;
        if (!anc || !d->nest->isBeforeInBlock(anc))
          return rewriter.notifyMatchFailure(st, "another use of the tiles");
        for (Operation *u : rd->getUsers()) {
          while (isa<vector::ShapeCastOp>(u) && u->hasOneUse())
            u = *u->getUsers().begin();
          auto conv = dyn_cast<ToBFP16Op>(u);
          std::optional<TileRead> tr = conv ? getTileRead(conv) : std::nullopt;
          if (!conv || conv.getTranspose() || !tr || tr->read != rd)
            return rewriter.notifyMatchFailure(st, "a tile read not converted");
          convs.push_back({conv, rd});
        }
        if (last->isBeforeInBlock(anc))
          last = anc;
      }
    }
    if (convs.empty())
      return rewriter.notifyMatchFailure(st, "no conversions");

    // Find the release of the block buffer after the decode. It has to move
    // after the last read, so no other use of its locks may come between.
    AIE::UseLockOp release;
    for (Operation *op = d->nest->getNextNode(); op; op = op->getNextNode()) {
      bool usesLock = op->walk([&](AIE::UseLockOp ul) {
                          return bytesLocks.contains(ul.getLock())
                                     ? WalkResult::interrupt()
                                     : WalkResult::advance();
                        }).wasInterrupted();
      auto ul = dyn_cast<AIE::UseLockOp>(op);
      if (ul && ul.release() && bytesLocks.contains(ul.getLock()) && !release)
        release = ul;
      else if (usesLock)
        return rewriter.notifyMatchFailure(st, "the bytes' lock in between");
      bool writes = op->walk([&](Operation *o) {
                        return mayChange(o, d->bytes.getBuffer(), {})
                                   ? WalkResult::interrupt()
                                   : WalkResult::advance();
                      }).wasInterrupted();
      if (writes)
        return rewriter.notifyMatchFailure(st, "the bytes may change");
      if (op == last)
        break;
    }

    // Load the blocks from a flat view of the block buffer.
    rewriter.setInsertionPoint(d->nest);
    Location loc = st.getLoc();
    auto bt = cast<MemRefType>(d->bytes.getType());
    SmallVector<ReassociationIndices> all(1);
    for (int64_t i = 0; i < bt.getRank(); ++i)
      all[0].push_back(i);
    Value flat = memref::CollapseShapeOp::create(rewriter, loc,
                                                 d->bytes.getBuffer(), all);
    // One load per tile read, where the read is, which all its conversions
    // use.
    DenseMap<Operation *, LoadBFP16Op> loads;
    for (auto [conv, rd] : convs) {
      LoadBFP16Op &ld = loads[rd];
      if (!ld) {
        rewriter.setInsertionPointAfter(rd);
        Location l = rd.getLoc();
        Value t = arith::DivUIOp::create(
            rewriter, l, linearOffset(rewriter, l, rd),
            arith::ConstantIndexOp::create(rewriter, l, kTile));
        Value off = arith::MulIOp::create(
            rewriter, l, blockOf(rewriter, l, *d, t),
            arith::ConstantIndexOp::create(rewriter, l, kTile + kExpBytes));
        ld = LoadBFP16Op::create(rewriter, l, conv.getMantissas().getType(),
                                 conv.getExponents().getType(), flat, off);
      }
      rewriter.replaceOp(conv, ld.getResults());
    }
    if (release && release->isBeforeInBlock(last))
      rewriter.moveOpAfter(release, last);
    rewriter.eraseOp(d->nest);
    // Erase the reads and their shape casts, the views and the tile buffer
    // once they are unused.
    for (auto &[rd, ld] : loads) {
      for (Operation *u : llvm::make_early_inc_range(rd->getUsers()))
        if (u->use_empty())
          rewriter.eraseOp(u);
      if (rd->use_empty())
        rewriter.eraseOp(rd);
    }
    for (Operation *v : llvm::reverse(views))
      if (v->use_empty())
        rewriter.eraseOp(v);
    if (d->tiles->use_empty())
      rewriter.eraseOp(d->tiles);
    return success();
  }
};

/// Evaluates index computations in `scope`, given the values in `env` for
/// induction variables and iter_args. Supports constants, integer arithmetic
/// and affine.apply; anything else is unknown.
struct TripEvaluator {
  Operation *scope;
  DenseMap<Value, int64_t> env;

  std::optional<int64_t> eval(Value v, int depth = 0) {
    if (auto it = env.find(v); it != env.end())
      return it->second;
    if (std::optional<int64_t> c = getConstantIntValue(v))
      return c;
    Operation *d = v.getDefiningOp();
    if (!d || depth > 32 || !scope->isAncestor(d))
      return std::nullopt;
    auto bin = [&](auto f) -> std::optional<int64_t> {
      std::optional<int64_t> a = eval(d->getOperand(0), depth + 1);
      std::optional<int64_t> b = eval(d->getOperand(1), depth + 1);
      if (!a || !b)
        return std::nullopt;
      return f(*a, *b);
    };
    if (isa<arith::AddIOp>(d))
      return bin(
          [](int64_t a, int64_t b) -> std::optional<int64_t> { return a + b; });
    if (isa<arith::SubIOp>(d))
      return bin(
          [](int64_t a, int64_t b) -> std::optional<int64_t> { return a - b; });
    if (isa<arith::MulIOp>(d))
      return bin(
          [](int64_t a, int64_t b) -> std::optional<int64_t> { return a * b; });
    if (isa<arith::DivUIOp, arith::DivSIOp>(d))
      return bin([](int64_t a, int64_t b) -> std::optional<int64_t> {
        return b > 0 && a >= 0 ? std::optional<int64_t>(a / b) : std::nullopt;
      });
    if (isa<arith::RemUIOp, arith::RemSIOp>(d))
      return bin([](int64_t a, int64_t b) -> std::optional<int64_t> {
        return b > 0 && a >= 0 ? std::optional<int64_t>(a % b) : std::nullopt;
      });
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(d))
      return eval(d->getOperand(0), depth + 1);
    if (auto ap = dyn_cast<affine::AffineApplyOp>(d)) {
      SmallVector<int64_t> vals;
      for (Value o : ap.getMapOperands()) {
        std::optional<int64_t> x = eval(o, depth + 1);
        if (!x)
          return std::nullopt;
        vals.push_back(*x);
      }
      AffineMap m = ap.getAffineMap();
      SmallVector<AffineExpr> dims, syms;
      MLIRContext *ctx = ap.getContext();
      for (unsigned i = 0; i < m.getNumDims(); ++i)
        dims.push_back(getAffineConstantExpr(vals[i], ctx));
      for (unsigned i = 0; i < m.getNumSymbols(); ++i)
        syms.push_back(getAffineConstantExpr(vals[m.getNumDims() + i], ctx));
      auto c = dyn_cast<AffineConstantExpr>(
          m.getResult(0).replaceDimsAndSymbols(dims, syms));
      return c ? std::optional<int64_t>(c.getValue()) : std::nullopt;
    }
    return std::nullopt;
  }
};

/// Returns the values from outside `loop` that the loop body's `vals` depend
/// on, directly or through the values the loop carries.
static SmallVector<Value> valuesFromOutside(scf::ForOp loop, ValueRange vals) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<Value> outside, work(vals.begin(), vals.end());
  DenseSet<Value> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    auto arg = dyn_cast<BlockArgument>(v);
    if (arg && arg.getOwner() == loop.getBody()) {
      if (OpOperand *init = loop.getTiedLoopInit(arg)) {
        outside.push_back(init->get());
        work.push_back(
            yield.getOperand(arg.getArgNumber() - loop.getNumInductionVars()));
      }
    } else if (Operation *d = v.getDefiningOp(); d && loop->isAncestor(d)) {
      work.append(d->getOperands().begin(), d->getOperands().end());
    } else {
      outside.push_back(v);
    }
  }
  return outside;
}

/// Returns the induction variables of the loops around `loop` that `vals`
/// depend on, or nullopt if they depend on other block arguments.
static std::optional<SmallVector<Value>> outerIVs(scf::ForOp loop,
                                                  ArrayRef<Value> vals) {
  SmallVector<Value> ivs, work(vals.begin(), vals.end());
  DenseSet<Value> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto l = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      if (!l || l.getInductionVar() != arg || !l->isProperAncestor(loop))
        return std::nullopt;
      ivs.push_back(v);
    } else if (Operation *d = v.getDefiningOp()) {
      work.append(d->getOperands().begin(), d->getOperands().end());
    }
  }
  return ivs;
}

/// Turns the r loads of one buffer in a loop body into one load stream when
/// iteration k loads blocks n + r*k ... n + r*k + r - 1 in program order. n
/// may depend on the induction variables of loops with static bounds around
/// the loop; the order is checked for each of their values.
/// The stream starts empty before the loop; its first pop fills it.
struct BFP16LoadStreamPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
    std::optional<int64_t> st = getConstantIntValue(loop.getStep());
    std::optional<int64_t> n = staticTrips(loop);
    if (!lb || !st || !n || *n <= 0)
      return rewriter.notifyMatchFailure(loop, "trips not known");
    int64_t trips = *n;
    // Group the loads in the body by buffer. Loads or streams in nested
    // regions are not handled.
    llvm::MapVector<Value, SmallVector<LoadBFP16Op>> byBase;
    for (Operation &op : *loop.getBody())
      if (auto ld = dyn_cast<LoadBFP16Op>(op))
        byBase[ld.getBase()].push_back(ld);
    bool nested = false;
    loop.getBody()->walk([&](Operation *op) {
      if (op->getParentOp() != loop &&
          isa<LoadBFP16Op, FifoStartOp, FifoLdPopBFP16Op>(op))
        nested = true;
    });
    if (byBase.empty() || nested)
      return rewriter.notifyMatchFailure(loop, "no loads to stream");

    // Evaluate each load's offset in every iteration to check the order, for
    // each value of the outer induction variables the offsets depend on.
    Operation *scope = loop->getParentOfType<AIE::CoreOp>();
    if (!scope)
      scope = loop->getParentOp();
    SmallVector<Value> offsets;
    for (auto &[base, loads] : byBase)
      for (LoadBFP16Op ld : loads)
        offsets.push_back(ld.getOffset());
    std::optional<SmallVector<Value>> ivs =
        outerIVs(loop, valuesFromOutside(loop, offsets));
    std::optional<SmallVector<SmallVector<int64_t>>> combos;
    if (ivs) {
      SmallVector<AffineExpr> syms;
      for (unsigned i = 0; i < ivs->size(); ++i)
        syms.push_back(getAffineSymbolExpr(i, loop.getContext()));
      combos = valuesOver(syms, *ivs);
    }
    // Each offset is evaluated once per trip and outer combination.
    constexpr int64_t kMaxEvals = 1 << 16;
    if (combos && (int64_t(offsets.size()) * trips > kMaxEvals ||
                   int64_t(combos->size()) >
                       kMaxEvals / (int64_t(offsets.size()) * trips)))
      combos.reset();
    SmallVector<std::pair<Value, SmallVector<LoadBFP16Op>>> streams;
    for (auto &[base, loads] : byBase) {
      // The pops read ahead of the loads they replace, so the loop must not
      // change the buffer.
      AIE::BufferOp root = rootBuffer(base);
      BufferLocks locks;
      locks.any = true;
      if (root)
        locks = bufferLocks(root, loop->getParentOfType<AIE::CoreOp>());
      bool changes =
          loop.getBody()
              ->walk([&](Operation *op) {
                auto ul = dyn_cast<AIE::UseLockOp>(op);
                bool c = ul ? (ul.acquire() || ul.acquireGE()) &&
                                  locks.contains(ul.getLock())
                            : mayWrite(op, base);
                return c ? WalkResult::interrupt() : WalkResult::advance();
              })
              .wasInterrupted();
      if (!loop.isDefinedOutsideOfLoop(base) || !combos || changes)
        continue;
      int64_t r = loads.size();
      bool ok = true;
      for (ArrayRef<int64_t> combo : *combos) {
        DenseMap<Value, int64_t> outer;
        for (auto [iv, x] : llvm::zip(*ivs, combo))
          outer[iv] = x;
        TripEvaluator ev{scope, outer};
        // Track the iter_arg values from one iteration to the next.
        SmallVector<std::optional<int64_t>> carried;
        for (Value init : loop.getInitArgs())
          carried.push_back(ev.eval(init));
        std::optional<int64_t> start;
        for (int64_t k = 0; k < trips && ok; ++k) {
          ev.env = outer;
          ev.env[loop.getInductionVar()] = *lb + k * *st;
          for (auto [arg, val] : llvm::zip(loop.getRegionIterArgs(), carried))
            if (val)
              ev.env[arg] = *val;
          for (int64_t j = 0; j < r && ok; ++j) {
            std::optional<int64_t> off = ev.eval(loads[j].getOffset());
            if (!off || *off % (kTile + kExpBytes)) {
              ok = false;
              break;
            }
            if (!start)
              start = *off;
            ok = *off == *start + (kTile + kExpBytes) * (r * k + j);
          }
          auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
          for (auto [i, y] : llvm::enumerate(yield.getOperands()))
            carried[i] = ev.eval(y);
        }
        if (!ok || !start) {
          ok = false;
          break;
        }
      }
      if (ok)
        streams.push_back({base, loads});
    }
    // A core has two load FIFOs.
    if (streams.empty() || int64_t(streams.size()) + loadStreams(loop) > 2)
      return rewriter.notifyMatchFailure(loop, "no streams, or too many");

    Location loc = loop.getLoc();
    rewriter.setInsertionPoint(loop);
    SmallVector<Value> inits;
    for (auto &[base, loads] : streams) {
      // The first offset, computed before the loop as in its first iteration.
      IRMapping first;
      first.map(loop.getInductionVar(), loop.getLowerBound());
      first.map(loop.getRegionIterArgs(), loop.getInitArgs());
      std::function<Value(Value)> before = [&](Value v) -> Value {
        if (Value m = first.lookupOrNull(v))
          return m;
        Operation *d = v.getDefiningOp();
        if (!d || !loop->isAncestor(d))
          return v;
        for (Value o : d->getOperands())
          before(o);
        return rewriter.clone(*d, first)->getResult(
            cast<OpResult>(v).getResultNumber());
      };
      auto begin = FifoStartOp::create(
          rewriter, loc, fifoAddrType(rewriter.getContext()),
          VectorType::get({32}, rewriter.getI32Type()), rewriter.getI32Type(),
          base, before(loads.front().getOffset()));
      inits.append(begin->getResults().begin(), begin->getResults().end());
    }
    (void)loop.replaceWithAdditionalYields(
        rewriter, inits, /*replaceInitOperandUsesInLoop=*/false,
        [&](OpBuilder &b, Location l, ArrayRef<BlockArgument> args) {
          SmallVector<Value> yields;
          for (auto [s, entry] : llvm::enumerate(streams)) {
            Value addr = args[3 * s], state = args[3 * s + 1],
                  count = args[3 * s + 2];
            // Emit the pops of an iteration together. A fill provides enough
            // data for the next 8 pops, so with a multiple of 8 blocks per
            // iteration only every eighth pop fills; otherwise every pop
            // fills.
            int64_t r = entry.second.size();
            Operation *first = entry.second.front();
            for (LoadBFP16Op ld : entry.second)
              if (ld->isBeforeInBlock(first))
                first = ld;
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPoint(first);
            for (auto [j, ld] : llvm::enumerate(entry.second)) {
              auto pop = FifoLdPopBFP16Op::create(
                  b, ld.getLoc(), fifoAddrType(b.getContext()), state.getType(),
                  b.getI32Type(), ld.getMantissas().getType(),
                  ld.getExponents().getType(), addr, state, count,
                  r % 8 == 0 && j % 8 ? b.getUnitAttr() : UnitAttr());
              addr = pop.getNext();
              state = pop.getStateOut();
              count = pop.getCountOut();
              ld.getMantissas().replaceAllUsesWith(pop.getMantissas());
              ld.getExponents().replaceAllUsesWith(pop.getExponents());
            }
            yields.append({addr, state, count});
          }
          return yields;
        });
    for (auto &[base, loads] : streams)
      for (LoadBFP16Op ld : loads)
        rewriter.eraseOp(ld);
    return success();
  }
};

} // namespace

void xilinx::aievec::populateReadDecodedBFP16Patterns(
    RewritePatternSet &patterns) {
  patterns.add<ReadDecodedBFP16Pattern>(patterns.getContext());
}

void xilinx::aievec::populateHoistBFP16ConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<HoistBFP16ConversionPattern>(patterns.getContext());
}

void xilinx::aievec::populateBFP16LoadStreamPatterns(
    RewritePatternSet &patterns) {
  patterns.add<BFP16LoadStreamPattern>(patterns.getContext());
}
