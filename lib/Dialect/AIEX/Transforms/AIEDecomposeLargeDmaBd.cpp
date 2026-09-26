//===- AIEDecomposeLargeDmaBd.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decomposes oversized non-contiguous aiex.npu.dma_memcpy_nd ops and task-path
// aie.dma_bd ops into one or more hardware-legal ND transfers before
// aie-dma-to-npu / aie-dma-tasks-to-npu lowering.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/DmaDecomposition.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <numeric>
#include <string>
#include <tuple>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEDECOMPOSELARGEDMABD
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

static bool allConstant(NpuDmaMemcpyNdOp op) {
  return llvm::all_of(op.getMixedSizes(),
                      [](OpFoldResult s) {
                        return getConstantIntValue(s).has_value();
                      }) &&
         llvm::all_of(op.getMixedStrides(),
                      [](OpFoldResult s) {
                        return getConstantIntValue(s).has_value();
                      }) &&
         llvm::all_of(op.getMixedOffsets(), [](OpFoldResult s) {
           return getConstantIntValue(s).has_value();
         });
}

static bool allConstant(AIE::DMABDOp op) {
  if (op.getPadDimensions().has_value())
    return false;
  if (op.getOffset() && !op.getConstantOffset())
    return false;
  if (op.getLen() && !op.getConstantLen())
    return false;
  if (op.getMixedSizes().empty())
    return false;
  return llvm::all_of(op.getMixedSizes(),
                      [](OpFoldResult s) {
                        return getConstantIntValue(s).has_value();
                      }) &&
         llvm::all_of(op.getMixedStrides(), [](OpFoldResult s) {
           return getConstantIntValue(s).has_value();
         });
}

static NdDmaPattern patternFromOp(NpuDmaMemcpyNdOp op) {
  NdDmaPattern pattern;
  pattern.offsets = llvm::map_to_vector(
      llvm::reverse(op.getMixedOffsets()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  pattern.sizes = llvm::map_to_vector(
      llvm::reverse(op.getMixedSizes()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  pattern.strides = llvm::map_to_vector(
      llvm::reverse(op.getMixedStrides()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  return pattern;
}

// Only called after allConstant(op) confirmed every size/stride resolves to
// a constant, so the has_value() checks below are redundant in practice --
// asserted rather than re-verified to keep that invariant visible here too.
static NdDmaPattern patternFromDmaBd(AIE::DMABDOp op) {
  SmallVector<int64_t, kNdDmaDims> outerSizes;
  SmallVector<int64_t, kNdDmaDims> outerStrides;
  for (OpFoldResult s : op.getMixedSizes()) {
    auto c = getConstantIntValue(s);
    assert(c && "size must be constant (already checked by allConstant)");
    outerSizes.push_back(*c);
  }
  for (OpFoldResult s : op.getMixedStrides()) {
    auto c = getConstantIntValue(s);
    assert(c && "stride must be constant (already checked by allConstant)");
    outerStrides.push_back(*c);
  }
  while (outerSizes.size() < kNdDmaDims) {
    outerSizes.insert(outerSizes.begin(), 1);
    outerStrides.insert(outerStrides.begin(), 0);
  }

  NdDmaPattern pattern;
  pattern.offsets.assign(outerSizes.size(), 0);
  pattern.sizes = llvm::map_to_vector(llvm::reverse(outerSizes),
                                      [](int64_t v) { return v; });
  pattern.strides = llvm::map_to_vector(llvm::reverse(outerStrides),
                                        [](int64_t v) { return v; });
  return pattern;
}

static SmallVector<int64_t, 4> toOuter(ArrayRef<int64_t> inner) {
  return llvm::map_to_vector(llvm::reverse(inner), [](int64_t v) { return v; });
}

static int64_t flatOffsetFromPattern(int64_t baseFlatOffset,
                                     const NdDmaPattern &pattern) {
  int64_t flat = baseFlatOffset + pattern.baseOffset;
  for (unsigned k = 0; k < 4; ++k)
    flat += pattern.offsets[k] * pattern.strides[k];
  return flat;
}

static int64_t lenFromInnermost3(ArrayRef<int64_t> sizesInnermostFirst) {
  int64_t len = 1;
  for (unsigned i = 0; i < 3; ++i)
    len *= sizesInnermostFirst[i];
  return len;
}

static AIE::BDDimLayoutArrayAttr outerDimsAttr(MLIRContext *ctx,
                                               ArrayRef<int64_t> outerSizes,
                                               ArrayRef<int64_t> outerStrides) {
  SmallVector<AIE::BDDimLayoutAttr> dims;
  dims.reserve(outerSizes.size());
  for (auto [s, t] : llvm::zip(outerSizes, outerStrides))
    dims.push_back(AIE::BDDimLayoutAttr::get(ctx, static_cast<uint32_t>(s),
                                             static_cast<uint32_t>(t)));
  return AIE::BDDimLayoutArrayAttr::get(ctx, dims);
}

static void updateTaskBdInPlace(AIE::DMABDOp bd, int32_t offset, int32_t len,
                                ArrayRef<int64_t> outerSizes,
                                ArrayRef<int64_t> outerStrides) {
  bd.getOffsetMutable().clear();
  bd.setStaticOffset(offset);
  bd.getLenMutable().clear();
  bd.setStaticLen(len);
  bd.getSizesMutable().clear();
  bd.getStridesMutable().clear();
  bd.setStaticSizes(DenseI64ArrayAttr::get(bd.getContext(), outerSizes));
  bd.setStaticStrides(DenseI64ArrayAttr::get(bd.getContext(), outerStrides));
}

static AIE::DMABDOp createTaskBd(PatternRewriter &rewriter, Location loc,
                                 AIE::DMABDOp tmpl, int32_t offset, int32_t len,
                                 ArrayRef<int64_t> outerSizes,
                                 ArrayRef<int64_t> outerStrides) {
  auto dims = outerDimsAttr(rewriter.getContext(), outerSizes, outerStrides);
  auto bd =
      AIE::DMABDOp::create(rewriter, loc, tmpl.getBuffer(), offset, len, dims);
  if (tmpl.getPacketAttr())
    bd.setPacketAttr(tmpl.getPacketAttr());
  if (tmpl.getBurstLengthAttr())
    bd.setBurstLengthAttr(tmpl.getBurstLengthAttr());
  if (tmpl.getAxcacheAttr())
    bd.setAxcacheAttr(tmpl.getAxcacheAttr());
  if (tmpl.getOffsetParameterAttr())
    bd.setOffsetParameterAttr(tmpl.getOffsetParameterAttr());
  // out_of_order_id is not copied because slicing an OoO BD is rejected above.
  return bd;
}

// Each BD execution advances the fourth (iteration) dimension once. If
// decomposition grows it, scale the task's execution count to match.
static int32_t getTaskRepeatCount(Operation *taskOp) {
  if (auto cfg = dyn_cast<DMAConfigureTaskOp>(taskOp))
    return cfg.getRepeatCount();
  return cast<DMAConfigureTaskForOp>(taskOp).getRepeatCount();
}

static Value getTaskRepeatCountVal(Operation *taskOp) {
  if (auto cfg = dyn_cast<DMAConfigureTaskOp>(taskOp))
    return cfg.getRepeatCountVal();
  return cast<DMAConfigureTaskForOp>(taskOp).getRepeatCountVal();
}

static void setTaskRepeatCount(Operation *taskOp, int32_t value) {
  if (auto cfg = dyn_cast<DMAConfigureTaskOp>(taskOp)) {
    cfg.setRepeatCount(value);
    return;
  }
  cast<DMAConfigureTaskForOp>(taskOp).setRepeatCount(value);
}

// How many executions one pass over a task BD's pattern takes: one per index
// of its iteration dimensions, all of them past d2.
static int64_t iterationCount(const NdDmaPattern &pattern) {
  int64_t count = 1;
  for (int64_t size : llvm::drop_begin(pattern.sizes, 3))
    count *= size;
  return count;
}

// The factor by which decomposition scaled the executions of a pass, num/den
// in lowest terms: a pass over the original takes `iterations` executions,
// and over `after` one per index of its iteration dimension. Below one, the
// executions merged, `after`'s inner dimensions taking in one the original
// iterated over.
struct IterationScale {
  int64_t num = 1;
  int64_t den = 1;

  bool isOne() const { return num == den; }
  // Whether `runs` executions of the original are whole executions of the
  // rewrite.
  bool divides(int64_t runs) const { return runs % den == 0; }
  int64_t apply(int64_t runs) const { return runs / den * num; }
  std::string str() const {
    return den == 1 ? std::to_string(num)
                    : std::to_string(num) + "/" + std::to_string(den);
  }
};

static IterationScale iterationScale(int64_t iterations,
                                     const NdDmaPattern &after) {
  int64_t common = std::gcd(after.sizes[3], iterations);
  return {after.sizes[3] / common, iterations / common};
}

static unsigned countTaskBds(Operation *taskOp) {
  unsigned n = 0;
  taskOp->walk([&](AIE::DMABDOp) { ++n; });
  return n;
}

static Region *getTaskBody(Operation *taskOp) {
  if (auto cfg = dyn_cast<DMAConfigureTaskOp>(taskOp))
    return &cfg.getBody();
  if (auto cfgFor = dyn_cast<DMAConfigureTaskForOp>(taskOp))
    return &cfgFor.getBody();
  return nullptr;
}

static bool isUnderRuntimeControlFlow(AIE::DMABDOp op) {
  auto seq = op->getParentOfType<AIE::RuntimeSequenceOp>();
  if (!seq)
    return false;
  for (Operation *parent = op->getParentOp(); parent && parent != seq;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(parent))
      return true;
  }
  return false;
}

static std::optional<std::pair<AIE::TileOp, Operation *>>
resolveTaskAndTile(AIE::DMABDOp op) {
  if (auto cfg = op->getParentOfType<DMAConfigureTaskOp>()) {
    // Decomposition is a shape rewrite, so an unplaced tile is not an error
    // here: decline and let the pattern run again after placement.
    AIE::TileOp tile = cfg.tryGetTileOp();
    if (!tile)
      return std::nullopt;
    return std::make_pair(tile, cfg.getOperation());
  }
  if (auto cfgFor = op->getParentOfType<DMAConfigureTaskForOp>()) {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return std::nullopt;
    auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(
        dev, cfgFor.getAlloc().getRootReference());
    if (!allocOp)
      return std::nullopt;
    AIE::TileOp tile = allocOp.getTileOp();
    if (!tile)
      return std::nullopt;
    return std::make_pair(tile, cfgFor.getOperation());
  }
  return std::nullopt;
}

static NpuDmaMemcpyNdOp createDecomposedOp(PatternRewriter &rewriter,
                                           NpuDmaMemcpyNdOp op,
                                           const NdDmaPattern &pattern,
                                           int64_t id, bool issueToken) {
  assert(pattern.baseOffset == 0 &&
         "a memcpy pattern has no peeled dimensions");
  auto outerOffsets = toOuter(pattern.offsets);
  auto outerSizes = toOuter(pattern.sizes);
  auto outerStrides = toOuter(pattern.strides);

  return NpuDmaMemcpyNdOp::create(
      rewriter, op.getLoc(), op.getMemref(),
      /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{},
      /*strides=*/ValueRange{},
      DenseI64ArrayAttr::get(op.getContext(), outerOffsets),
      DenseI64ArrayAttr::get(op.getContext(), outerSizes),
      DenseI64ArrayAttr::get(op.getContext(), outerStrides), op.getPacketAttr(),
      op.getMetadata(), rewriter.getI64IntegerAttr(id),
      rewriter.getBoolAttr(issueToken), op.getD0ZeroBeforeAttr(),
      op.getD1ZeroBeforeAttr(), op.getD2ZeroBeforeAttr(),
      op.getD0ZeroAfterAttr(), op.getD1ZeroAfterAttr(), op.getD2ZeroAfterAttr(),
      op.getBurstLengthAttr(), op.getAxcacheAttr(), op.getOffsetParameterAttr(),
      op.getOffsetStateTableIdxAttr());
}

static int64_t allocateNextId(NpuDmaMemcpyNdOp op, int64_t startId,
                              llvm::DenseSet<int64_t> &usedIds) {
  int64_t id = startId;
  while (usedIds.contains(id))
    ++id;
  usedIds.insert(id);
  return id;
}

static bool getTaskIssueToken(Operation *taskOp) {
  if (auto cfg = dyn_cast<DMAConfigureTaskOp>(taskOp))
    return cfg.getIssueToken();
  return cast<DMAConfigureTaskForOp>(taskOp).getIssueToken();
}

static void setTaskIssueToken(Operation *taskOp, bool token) {
  if (auto cfg = dyn_cast<DMAConfigureTaskOp>(taskOp)) {
    cfg.setIssueToken(token);
    return;
  }
  cast<DMAConfigureTaskForOp>(taskOp).setIssueToken(token);
}

// Marks the starts splitIntoTasks emits until orderSlices has placed them:
// [group, index within the group, group size].
static constexpr llvm::StringLiteral kSliceAttr = "aiex.decompose_slice";
// Marks the configures of the slices after the first, which orderSlices moves
// up to their first start.
static constexpr llvm::StringLiteral kSliceTaskAttr =
    "aiex.decompose_slice_task";

// Why the slices of `bd` cannot go out as one task each, or nullopt if they
// can. Every use of the task has to be one splitIntoTasks rewrites, and every
// start has to run whole passes over the pattern, a pass being one execution
// per index of the `iterations`-long iteration dimension: the slices of a pass
// are issued in turn, so a partial pass has no slice boundary to stop at.
static std::optional<std::string>
whyNotSeparateTasks(AIE::DMABDOp bd, Operation *taskOp, int64_t iterations) {
  std::string why;
  llvm::raw_string_ostream os(why);
  if (bd.getBdIdVal()) {
    os << "the descriptor's bd_id is a runtime value";
    return why;
  }
  if (getTaskRepeatCountVal(taskOp)) {
    os << "the task's repeat count is a runtime value";
    return why;
  }
  for (Operation *user : taskOp->getResult(0).getUsers()) {
    if (!isa<DMAStartTaskOp, DMAAwaitTaskOp, DMAFreeTaskOp>(user)) {
      os << "the task is used by '" << user->getName() << "'";
      return why;
    }
    if (user->getBlock() != taskOp->getBlock()) {
      os << "the task is used outside the block that configures it";
      return why;
    }
    auto start = dyn_cast<DMAStartTaskOp>(user);
    if (!start)
      continue;
    std::optional<uint32_t> rc = start.getRepeatCount();
    int64_t runs = static_cast<int64_t>(rc ? *rc : getTaskRepeatCount(taskOp));
    ++runs;
    if (runs % iterations != 0) {
      os << "a start runs it " << runs
         << " times, not a whole number of passes over its " << iterations
         << "-long iteration dimension";
      return why;
    }
  }
  return std::nullopt;
}

// Issue the slices of `bd` as one single-descriptor task each. The task
// becomes the first slice and the others are configured right after it, until
// orderSlices moves each up to its first start. Only
// the last slice issues the task's token, so an await of the task becomes an
// await of the last slice, and a free frees every slice. Each start becomes a
// group of consecutive starts, every slice once per pass, which orderSlices
// then interleaves with the starts around it.
static void splitIntoTasks(PatternRewriter &rewriter, AIE::DMABDOp bd,
                           Operation *taskOp, ArrayRef<NdDmaPattern> slices,
                           int64_t baseFlatOffset, int64_t iterations,
                           int64_t &nextGroup) {
  bool token = getTaskIssueToken(taskOp);
  int64_t taskRuns = static_cast<int64_t>(getTaskRepeatCount(taskOp)) + 1;
  SmallVector<Operation *> users(taskOp->getResult(0).getUsers());

  SmallVector<Operation *> tasks{taskOp};
  rewriter.setInsertionPointAfter(taskOp);
  for (size_t i = 1; i < slices.size(); ++i)
    tasks.push_back(rewriter.clone(*taskOp));
  for (Operation *task : llvm::drop_begin(tasks))
    task->setAttr(kSliceTaskAttr, rewriter.getUnitAttr());

  for (auto it : llvm::enumerate(slices)) {
    size_t i = it.index();
    const NdDmaPattern &sub = it.value();
    Operation *task = tasks[i];
    AIE::DMABDOp sliceBd = bd;
    if (i > 0)
      task->walk([&](AIE::DMABDOp b) { sliceBd = b; });
    int64_t flatOffset = flatOffsetFromPattern(baseFlatOffset, sub);
    int32_t len = static_cast<int32_t>(lenFromInnermost3(sub.sizes));
    rewriter.modifyOpInPlace(sliceBd, [&]() {
      updateTaskBdInPlace(sliceBd, static_cast<int32_t>(flatOffset), len,
                          toOuter(sub.sizes), toOuter(sub.strides));
      // A pinned id stays with the first slice, as in a chain.
      if (i > 0)
        sliceBd.removeBdIdAttr();
    });
    rewriter.modifyOpInPlace(task, [&]() {
      // One pass over the slice is one execution per index of its own
      // iteration dimension.
      setTaskRepeatCount(task, static_cast<int32_t>(sub.sizes[3] - 1));
      setTaskIssueToken(task, token && i + 1 == slices.size());
    });
  }

  for (Operation *user : users) {
    if (auto await = dyn_cast<DMAAwaitTaskOp>(user)) {
      rewriter.modifyOpInPlace(await, [&]() {
        await.getTaskMutable().assign(tasks.back()->getResult(0));
      });
      continue;
    }
    if (auto free = dyn_cast<DMAFreeTaskOp>(user)) {
      rewriter.setInsertionPointAfter(free);
      for (Operation *task : llvm::drop_begin(tasks))
        DMAFreeTaskOp::create(rewriter, free.getLoc(), task->getResult(0));
      continue;
    }
    auto start = cast<DMAStartTaskOp>(user);
    std::optional<uint32_t> rc = start.getRepeatCount();
    int64_t runs = rc ? static_cast<int64_t>(*rc) + 1 : taskRuns;
    int64_t count = runs / iterations * static_cast<int64_t>(slices.size());
    int64_t group = nextGroup++;
    // Only the last start of the last slice may issue the token: an earlier
    // pass's would complete an await before the transfer has.
    bool startToken = token && !start.getNoToken();
    auto mark = [&](DMAStartTaskOp s, int64_t index) {
      s->setAttr(kSliceAttr,
                 rewriter.getDenseI64ArrayAttr({group, index, count}));
    };
    rewriter.modifyOpInPlace(start, [&]() {
      start.removeRepeatCountAttr();
      mark(start, 0);
    });
    rewriter.setInsertionPointAfter(start);
    for (int64_t k = 1; k < count; ++k) {
      size_t i = static_cast<size_t>(k) % slices.size();
      bool withholds =
          token && i + 1 == slices.size() && (!startToken || k + 1 != count);
      auto s = DMAStartTaskOp::create(
          rewriter, start.getLoc(), tasks[i]->getResult(0),
          /*repeat_count=*/nullptr,
          /*no_token=*/withholds ? rewriter.getUnitAttr() : nullptr);
      mark(s, k);
    }
  }
}

using ChannelKey = std::tuple<const void *, unsigned, int64_t>;

// The channel a start pushes onto, or nullopt when its task is not a configure
// this pass can read.
static std::optional<ChannelKey> channelOf(DMAStartTaskOp start) {
  Operation *def = start.getTask().getDefiningOp();
  if (auto cfg = dyn_cast_or_null<DMAConfigureTaskOp>(def))
    return ChannelKey{cfg.getTile().getAsOpaquePointer(),
                      static_cast<unsigned>(cfg.getDirection()),
                      cfg.getChannel()};
  auto cfgFor = dyn_cast_or_null<DMAConfigureTaskForOp>(def);
  if (!cfgFor)
    return std::nullopt;
  auto dev = cfgFor->getParentOfType<AIE::DeviceOp>();
  auto alloc = dev ? AIE::ShimDMAAllocationOp::getForSymbol(
                         dev, cfgFor.getAlloc().getRootReference())
                   : AIE::ShimDMAAllocationOp();
  if (!alloc)
    return std::nullopt;
  return ChannelKey{alloc.getTile().getAsOpaquePointer(),
                    static_cast<unsigned>(alloc.getChannelDir()),
                    alloc.getChannelIndex()};
}

// Interleave the slices splitIntoTasks emitted with the starts around them.
// Later slices wait for earlier ones to finish (see
// aie-assign-runtime-sequence-bd-ids), so a counterpart transfer left behind
// all of them (a core's input for an output, say) would deadlock.
//
// A round runs from one start up to the next start on a channel it already
// used. Each start's first slice keeps its place; later ones follow the
// round's last start, ordered by index / size, ties in program order. A round
// ends at anything a push may not move past: an await, sync, poll, untracked
// push, or free of an unplaced slice. Configures, write32, blockwrite, RTP
// writes and lock sets only delay a push, so slices move past them.
static void orderSlices(Block &block) {
  struct Pending {
    DMAStartTaskOp start;
    int64_t index, count, head;
  };
  SmallVector<Pending> pending;
  llvm::DenseSet<Value> pendingTasks;
  llvm::DenseSet<ChannelKey> round;
  llvm::DenseMap<int64_t, int64_t> headOfGroup;
  Operation *anchor = nullptr;
  int64_t heads = 0;

  auto flush = [&]() {
    llvm::stable_sort(pending, [](const Pending &a, const Pending &b) {
      int64_t lhs = a.index * b.count;
      int64_t rhs = b.index * a.count;
      return lhs != rhs ? lhs < rhs : a.head < b.head;
    });
    Operation *after = anchor;
    for (Pending &p : pending) {
      p.start->moveAfter(after);
      after = p.start;
      // Configure a slice only once it is due, so that the descriptors it
      // takes are not held while earlier slices wait: the BD-ID pass can take
      // descriptors back only from started tasks.
      Operation *task = p.start.getTask().getDefiningOp();
      if (task && task->hasAttr(kSliceTaskAttr)) {
        task->removeAttr(kSliceTaskAttr);
        task->moveBefore(p.start);
      }
    }
    pending.clear();
    pendingTasks.clear();
    round.clear();
    headOfGroup.clear();
    anchor = nullptr;
  };

  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (auto start = dyn_cast<DMAStartTaskOp>(op)) {
      std::optional<ChannelKey> channel = channelOf(start);
      auto slice = start->getAttrOfType<DenseI64ArrayAttr>(kSliceAttr);
      if (channel && slice && slice[1] > 0) {
        auto head = headOfGroup.find(slice[0]);
        if (head != headOfGroup.end()) {
          pending.push_back({start, slice[1], slice[2], head->second});
          pendingTasks.insert(start.getTask());
          continue;
        }
      }
      if (!channel || round.contains(*channel))
        flush();
      if (!channel)
        continue;
      round.insert(*channel);
      if (slice)
        headOfGroup[slice[0]] = heads;
      ++heads;
      anchor = &op;
      continue;
    }
    if (auto free = dyn_cast<DMAFreeTaskOp>(op)) {
      if (pendingTasks.contains(free.getTask()))
        flush();
      continue;
    }
    if (isa<DMAConfigureTaskOp, DMAConfigureTaskForOp, NpuWrite32Op,
            NpuBlockWriteOp, NpuWriteRTPOp, SetLockOp>(op) ||
        isMemoryEffectFree(&op))
      continue;
    flush();
  }
  flush();
}

struct DecomposeLargeDmaBdPattern : OpRewritePattern<NpuDmaMemcpyNdOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NpuDmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {
    if (!allConstant(op))
      return failure();

    NdDmaPattern pattern = patternFromOp(op);
    if (isContiguousTransfer(pattern.sizes, pattern.strides))
      return failure();

    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return failure();

    auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(
        dev, op.getMetadata().getRootReference());
    if (!allocOp)
      return failure();

    AIE::TileOp tile = allocOp.getTileOp();
    if (!tile)
      return failure();

    int col = tile.getCol();
    int row = tile.getRow();
    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(op);
    auto bufferType = cast<BaseMemRefType>(op.getMemref().getType());

    if (patternPassesVerification(op, bufferType, targetModel, col, row,
                                  pattern))
      return failure();

    auto decomposed =
        decomposeNdDmaPattern(op, bufferType, pattern, targetModel, col, row);
    // failed() already guards both dereferences below via short-circuit /
    // prior control flow; the checker just doesn't associate FailureOr's
    // failed()/succeeded() idiom with the std::optional base it derives from.
    if (failed(decomposed) ||
        decomposed->empty()) // NOLINT(bugprone-unchecked-optional-access)
      return failure();
    // Bind a plain reference now that decomposed is known non-failed and
    // non-empty, so nothing past this point looks like an optional access.
    SmallVector<NdDmaPattern> &bds =
        *decomposed; // NOLINT(bugprone-unchecked-optional-access)
    if (bds.size() > targetModel.getNumBDs(col, row))
      return failure();

    if (bds.size() == 1) {
      rewriter.replaceOpWithNewOp<NpuDmaMemcpyNdOp>(
          op, op.getMemref(), ValueRange{}, ValueRange{}, ValueRange{},
          DenseI64ArrayAttr::get(op.getContext(), toOuter(bds.front().offsets)),
          DenseI64ArrayAttr::get(op.getContext(), toOuter(bds.front().sizes)),
          DenseI64ArrayAttr::get(op.getContext(), toOuter(bds.front().strides)),
          op.getPacketAttr(), op.getMetadata(), op.getIdAttr(),
          op.getIssueTokenAttr(), op.getD0ZeroBeforeAttr(),
          op.getD1ZeroBeforeAttr(), op.getD2ZeroBeforeAttr(),
          op.getD0ZeroAfterAttr(), op.getD1ZeroAfterAttr(),
          op.getD2ZeroAfterAttr(), op.getBurstLengthAttr(), op.getAxcacheAttr(),
          op.getOffsetParameterAttr(), op.getOffsetStateTableIdxAttr());
      return success();
    }

    llvm::DenseSet<int64_t> usedIds;
    if (auto seq = op->getParentOfType<AIE::RuntimeSequenceOp>()) {
      seq.walk([&](NpuDmaMemcpyNdOp other) {
        if (other == op)
          return;
        if (other.getMetadata() == op.getMetadata())
          usedIds.insert(other.getId());
      });
    }

    int64_t nextId = op.getId();
    rewriter.setInsertionPoint(op);
    for (auto [idx, subPattern] : llvm::enumerate(bds)) {
      bool last = idx + 1 == bds.size();
      int64_t id = allocateNextId(op, nextId, usedIds);
      nextId = id + 1;
      createDecomposedOp(rewriter, op, subPattern, id,
                         last && op.getIssueToken());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct DecomposeLargeDmaBdTaskPattern : OpRewritePattern<AIE::DMABDOp> {
  DecomposeLargeDmaBdTaskPattern(MLIRContext *ctx, int64_t &nextGroup)
      : OpRewritePattern(ctx), nextGroup(&nextGroup) {}

  // Numbers the start groups splitIntoTasks emits, across the pass.
  int64_t *nextGroup;

  LogicalResult matchAndRewrite(AIE::DMABDOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<AIE::MemOp>() ||
        op->getParentOfType<AIE::ShimDMAOp>() ||
        op->getParentOfType<AIE::MemTileDMAOp>() ||
        op->getParentOfType<AIE::DMAOp>())
      return failure();

    auto taskAndTile = resolveTaskAndTile(op);
    if (!taskAndTile)
      return failure();

    AIE::TileOp tile = taskAndTile->first;
    Operation *taskOp = taskAndTile->second;

    // A descriptor with more dimensions than a BD has cannot be lowered as it
    // is, so where this pass cannot reduce it, it says why.
    bool tooManyDims = op.getMixedSizes().size() > kNdDmaDims;
    auto cannotReduce = [&]() {
      return op.emitOpError()
             << "has " << op.getMixedSizes().size()
             << " dimensions, and a buffer descriptor holds " << kNdDmaDims
             << "; the extra ones can only be split off ";
    };
    if (!allConstant(op)) {
      if (tooManyDims)
        return cannotReduce()
               << "a descriptor whose offset, length, sizes and strides are "
                  "all constant and that has no padding";
      return failure();
    }
    if (countTaskBds(taskOp) != 1) {
      if (tooManyDims)
        return cannotReduce() << "a task's only descriptor";
      return failure();
    }
    // A descriptor takes its locks once per execution, which decomposition
    // regroups, and a slice of its own would take none.
    bool takesLocks = false;
    taskOp->walk([&](AIE::UseLockOp) { takesLocks = true; });
    if (takesLocks) {
      if (tooManyDims)
        return cannotReduce() << "a descriptor that takes no locks";
      return failure();
    }

    int col = tile.getCol();
    int row = tile.getRow();
    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(op);
    auto bufferType = cast<BaseMemRefType>(op.getBuffer().getType());
    // See contiguousAndFits for why only d3 can make one too long.
    int64_t maxIterations = 1LL << targetModel.getDmaBdIterBits(col, row);
    auto lowerable = [&](const NdDmaPattern &p) {
      if (p.sizes.size() != kNdDmaDims)
        return false;
      if (isContiguousTransfer(p.sizes, p.strides))
        return p.sizes[3] <= maxIterations;
      return patternPassesVerification(op, bufferType, targetModel, col, row,
                                       p);
    };

    NdDmaPattern pattern = patternFromDmaBd(op);
    if (lowerable(pattern))
      return failure();

    // Outer iteration dimensions that re-read the same data only repeat what
    // is inside them, as the task's repeat count does. Where dropping them is
    // all the pattern needs, they go, and a pass shrinks to what they repeat.
    NdDmaPattern unrepeated = pattern;
    for (unsigned d = unrepeated.sizes.size();
         d-- > 3 && (unrepeated.sizes[d] == 1 || unrepeated.strides[d] == 0);)
      unrepeated.sizes[d] = 1;
    bool dropRepeats = lowerable(unrepeated);
    if (dropRepeats)
      pattern = unrepeated;
    // One pass over the pattern, in executions of the descriptor.
    int64_t iterations = iterationCount(pattern);

    auto decomposed = dropRepeats
                          ? FailureOr<SmallVector<NdDmaPattern>>(
                                SmallVector<NdDmaPattern>{pattern})
                          : decomposeNdDmaPattern(op, bufferType, pattern,
                                                  targetModel, col, row);
    // failed() already guards both dereferences below via short-circuit /
    // prior control flow; the checker just doesn't associate FailureOr's
    // failed()/succeeded() idiom with the std::optional base it derives from.
    if (failed(decomposed) ||
        decomposed->empty()) { // NOLINT(bugprone-unchecked-optional-access)
      if (tooManyDims)
        return op.emitOpError()
               << "has " << op.getMixedSizes().size()
               << " dimensions, and no split into descriptors of " << kNdDmaDims
               << " fits this tile";
      return failure();
    }
    // Bind a plain reference now that decomposed is known non-failed and
    // non-empty, so nothing past this point looks like an optional access.
    SmallVector<NdDmaPattern> &bds =
        *decomposed; // NOLINT(bugprone-unchecked-optional-access)

    if (bds.size() > 1 && isUnderRuntimeControlFlow(op)) {
      if (tooManyDims)
        return cannotReduce() << "outside runtime control flow, since it "
                                 "splits into "
                              << bds.size() << " descriptors";
      op.emitRemark()
          << "deferring multi-BD decomposition under runtime control flow "
             "(dynamic BD pool supports single-BD tasks only)";
      return failure();
    }

    int64_t baseFlatOffset = op.getConstantOffset().value_or(0);

    if (bds.size() == 1) {
      const NdDmaPattern &sub = bds.front();
      int64_t flatOffset = flatOffsetFromPattern(baseFlatOffset, sub);
      auto outerSizes = toOuter(sub.sizes);
      auto outerStrides = toOuter(sub.strides);
      // len covers one BD invocation, so it tracks the innermost three
      // dimensions only. Rewriting in place can move extent into the fourth
      // (repeat) dimension -- e.g. an innermost run too long for the wrap
      // field factors out an extra dimension and pushes the outermost one into
      // the repeat slot -- so len has to be recomputed alongside the shape.
      int32_t len = static_cast<int32_t>(lenFromInnermost3(sub.sizes));

      IterationScale scale = iterationScale(iterations, sub);
      int64_t runs = 0;
      if (!scale.isOne()) {
        if (getTaskRepeatCountVal(taskOp))
          return op.emitOpError()
                 << "cannot decompose a buffer descriptor whose repeat count "
                    "is a runtime value: decomposition needs to scale it by "
                 << scale.str();
        // A scaled count past what one queue push carries is fine: the BD-ID
        // pass issues it as several starts. Only the attribute width limits
        // it. Widen before multiplying: the accessor returns int32_t, so the
        // addition alone would overflow in int and wrap past this check.
        // Merged executions need every start to run whole ones.
        auto check = [&](Operation *at, int64_t repeat, StringRef runner,
                         StringRef count) -> LogicalResult {
          int64_t before = repeat + 1;
          if (!scale.divides(before))
            return at->emitOpError()
                   << "cannot decompose: it merges every " << scale.den
                   << " executions of the buffer descriptor into " << scale.num
                   << ", and " << runner << " runs it " << before << " times";
          int64_t after = scale.apply(before);
          if (after - 1 > std::numeric_limits<int32_t>::max())
            return at->emitOpError()
                   << "decomposition scales " << count << " by " << scale.str()
                   << " to " << (after - 1) << ", beyond a 32-bit repeat_count";
          return success();
        };
        runs = getTaskRepeatCount(taskOp) + int64_t{1};
        if (failed(check(op, runs - 1, "the task", "the repeat count")))
          return failure();
        // A start that overrides the task's count repeats the same BD, so it
        // scales by the same factor.
        for (Operation *user : taskOp->getResult(0).getUsers())
          if (auto start = dyn_cast<DMAStartTaskOp>(user))
            if (IntegerAttr rc = start.getRepeatCountAttr())
              if (failed(check(start, rc.getInt(), "this start",
                               "this start's repeat count")))
                return failure();
        runs = scale.apply(runs);
      }

      rewriter.modifyOpInPlace(op, [&]() {
        updateTaskBdInPlace(op, static_cast<int32_t>(flatOffset), len,
                            outerSizes, outerStrides);
      });
      if (!scale.isOne()) {
        rewriter.modifyOpInPlace(taskOp, [&]() {
          setTaskRepeatCount(taskOp, static_cast<int32_t>(runs - 1));
        });
        for (Operation *user : taskOp->getResult(0).getUsers())
          if (auto start = dyn_cast<DMAStartTaskOp>(user))
            if (IntegerAttr rc = start.getRepeatCountAttr())
              rewriter.modifyOpInPlace(start, [&]() {
                start.setRepeatCountAttr(rewriter.getI32IntegerAttr(
                    scale.apply(rc.getInt() + 1) - 1));
              });
      }
      return success();
    }

    // A split would make every slice reuse the one out_of_order_id slot.
    // TODO: BD iteration plus padding may enable this feature.
    if (op.getOutOfOrderId().has_value())
      return op.emitOpError() << "splitting an out-of-order buffer descriptor "
                                 "into multiple descriptors is not implemented";

    // Every member of a chain runs once per task execution, so the one
    // queue-push repeat count is shared by all of them. A slice that also wants
    // its own iteration factor needs a private repeat, which only a task of
    // its own has. A chain also holds all of its descriptors until it
    // finishes, so past the queue depth the slices go out as separate tasks,
    // whose descriptors aie-assign-runtime-sequence-bd-ids can recycle one by
    // one.
    bool sharedRepeat = llvm::all_of(bds, [&](const NdDmaPattern &sub) {
      return iterationScale(iterations, sub).isOne();
    });
    bool chainFits =
        sharedRepeat && bds.size() <= targetModel.getNumBDs(col, row);
    uint32_t depth = targetModel.getDmaTaskQueueDepth();
    if (!chainFits || (depth > 0 && bds.size() > depth)) {
      std::optional<std::string> why =
          whyNotSeparateTasks(op, taskOp, iterations);
      if (!why) {
        splitIntoTasks(rewriter, op, taskOp, bds, baseFlatOffset, iterations,
                       *nextGroup);
        return success();
      }
      if (!sharedRepeat)
        return op.emitOpError()
               << "cannot split this buffer descriptor: its slices need "
                  "per-descriptor repeat counts, which a chain shares, and "
                  "they cannot be separate tasks because "
               << *why;
      if (!chainFits)
        return op.emitOpError()
               << "cannot split this buffer descriptor: its " << bds.size()
               << " slices outnumber the tile's "
               << targetModel.getNumBDs(col, row)
               << " buffer descriptors, and they cannot be separate tasks "
                  "because "
               << *why;
    }

    Region *body = getTaskBody(taskOp);
    if (!body || body->empty())
      return failure();

    SmallVector<Block *> blocks;
    blocks.push_back(op->getBlock());
    for (unsigned i = 1; i < bds.size(); ++i)
      blocks.push_back(rewriter.createBlock(body));

    for (auto [idx, subPattern] : llvm::enumerate(bds)) {
      Block *block = blocks[idx];
      int64_t flatOffset = flatOffsetFromPattern(baseFlatOffset, subPattern);
      auto outerSizes = toOuter(subPattern.sizes);
      auto outerStrides = toOuter(subPattern.strides);
      int32_t len = static_cast<int32_t>(lenFromInnermost3(subPattern.sizes));

      if (idx == 0) {
        rewriter.modifyOpInPlace(op, [&]() {
          updateTaskBdInPlace(op, static_cast<int32_t>(flatOffset), len,
                              outerSizes, outerStrides);
        });
        Operation *oldTerm = block->getTerminator();
        rewriter.setInsertionPoint(oldTerm);
        if (idx + 1 < bds.size())
          AIE::NextBDOp::create(rewriter, op.getLoc(), blocks[idx + 1]);
        else
          AIE::EndOp::create(rewriter, op.getLoc());
        rewriter.eraseOp(oldTerm);
      } else {
        rewriter.setInsertionPointToStart(block);
        createTaskBd(rewriter, op.getLoc(), op,
                     static_cast<int32_t>(flatOffset), len, outerSizes,
                     outerStrides);
        rewriter.setInsertionPointToEnd(block);
        if (idx + 1 < bds.size())
          AIE::NextBDOp::create(rewriter, op.getLoc(), blocks[idx + 1]);
        else
          AIE::EndOp::create(rewriter, op.getLoc());
      }
    }

    return success();
  }
};

struct AIEDecomposeLargeDmaBdPass
    : xilinx::AIEX::impl::AIEDecomposeLargeDmaBdBase<
          AIEDecomposeLargeDmaBdPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    int64_t nextGroup = 0;
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeLargeDmaBdPattern>(&getContext());
    patterns.add<DecomposeLargeDmaBdTaskPattern>(&getContext(), nextGroup);
    if (failed(applyPatternsGreedily(device, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    llvm::SetVector<Block *> blocks;
    device.walk([&](DMAStartTaskOp start) {
      if (start->hasAttr(kSliceAttr))
        blocks.insert(start->getBlock());
    });
    for (Block *block : blocks)
      orderSlices(*block);
    device.walk([&](Operation *op) {
      op->removeAttr(kSliceAttr);
      op->removeAttr(kSliceTaskAttr);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEDecomposeLargeDmaBdPass() {
  return std::make_unique<AIEDecomposeLargeDmaBdPass>();
}
