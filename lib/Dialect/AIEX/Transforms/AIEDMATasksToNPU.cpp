//===- AIEDMATasksToNPU.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEDMATASKSTONPU
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct DMAStartTaskOpPattern : OpConversionPattern<DMAStartTaskOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DMAStartTaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DMAConfigureTaskOp task_op = op.getTaskOp();
    if (!task_op) {
      // Cannot rewrite this; probably points to a DMAStartTaskForOp,
      // which we will lower once it has been rewritten into a DMAStartTaskOp.
      return failure();
    }
    AIE::TileOp tile = task_op.getTileOp();
    Location loc = op.getLoc();

    // The bd_id for the queue push: the runtime pool value (dynamic free-list)
    // if the configure carries one, else the statically-assigned first BD id.
    Value bdIdVal;
    if (Value runtimeBdId = task_op.getBdIdVal()) {
      bdIdVal = runtimeBdId;
    } else {
      std::optional<uint32_t> first_bd_id = task_op.getFirstBdId();
      if (!first_bd_id) {
        auto err = op.emitOpError(
            "First buffer descriptor in chain has not been assigned an ID");
        err.attachNote()
            << "Run the `aie-assign-runtime-buffer-descriptor-ids` "
               "pass first or manually assign an ID.";
        return failure();
      }
      bdIdVal = createConstantI32(rewriter, loc, *first_bd_id);
    }
    // push_queue takes bd_id + repeat_count as SSA operands. repeat_count is a
    // runtime operand when present (dynamic tile count), else the compile-time
    // attribute materialized as a constant.
    Value repeatCount = getAsValue(rewriter, loc, task_op.getRepeatCountValue(),
                                   rewriter.getI32Type());
    rewriter.replaceOpWithNewOp<NpuPushQueueOp>(
        op, tile.getCol(), tile.getRow(), task_op.getDirection(),
        task_op.getChannel(), task_op.getIssueToken(), repeatCount, bdIdVal);
    return success();
  }
};

// Resolve a task value to a configure that gives the sync its physical channel.
// The value is usually a configure result, but under the dynamic BD pool path a
// task threaded through runtime control flow surfaces as an scf result: a loop
// result (the task carried across iterations) or an scf.if result (the phi of
// the task in flight, whichever branch ran). Walk such a result back to a
// configure via the yields (and, for a loop, the init). Every reachable
// configure targets the same physical channel -- the pool pass verified both
// branches of an scf.if agree -- so the first one found gives the right
// channel.
static DMAConfigureTaskOp resolveConfigureThroughCF(Value task) {
  llvm::SmallPtrSet<Value, 8> seen;
  SmallVector<Value> worklist{task};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!v || !seen.insert(v).second)
      continue;
    if (auto cfg = v.getDefiningOp<DMAConfigureTaskOp>())
      return cfg;
    if (auto res = dyn_cast<OpResult>(v)) {
      Operation *def = res.getOwner();
      unsigned k = res.getResultNumber();
      if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
        worklist.push_back(ifOp.thenBlock()->getTerminator()->getOperand(k));
        worklist.push_back(ifOp.elseBlock()->getTerminator()->getOperand(k));
      } else if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        worklist.push_back(forOp.getInitArgs()[k]);
        worklist.push_back(forOp.getBody()->getTerminator()->getOperand(k));
      }
    }
  }
  return nullptr;
}

struct DMAAwaitTaskOpPattern : OpConversionPattern<DMAAwaitTaskOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DMAAwaitTaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DMAConfigureTaskOp task_op = op.getTaskOp();
    if (!task_op) {
      // The awaited task threads through runtime control flow (an scf result),
      // so it has no directly-defining configure; walk the SSA carry to one.
      task_op = resolveConfigureThroughCF(op.getTask());
    }
    if (!task_op) {
      return failure();
    }
    if (!task_op.getIssueToken()) {
      auto err = op.emitOpError(
          "Cannot wait on a BD that is not configured to issue a token.");
      err.attachNote(task_op.getLoc())
          << "Consider adding attribute `issue_token=true` here.";
      return err;
    }
    AIE::TileOp tile = task_op.getTileOp();
    Location loc = op.getLoc();
    rewriter.replaceOpWithNewOp<NpuSyncOp>(
        op, createConstantI32(rewriter, loc, tile.getCol()),
        createConstantI32(rewriter, loc, tile.getRow()),
        createConstantI32(rewriter, loc, (uint32_t)task_op.getDirection()),
        createConstantI32(rewriter, loc, task_op.getChannel()),
        createConstantI32(rewriter, loc, 1),
        createConstantI32(rewriter, loc, 1));
    return success();
  }
};

struct AIEDMATasksToNPUPass
    : xilinx::AIEX::impl::AIEDMATasksToNPUBase<AIEDMATasksToNPUPass> {

  bool shouldSkipBlock(Block &block) {
    // Allow blocks in the input IR that contain nothing but a next_bd operation
    // as the entry block. We will skip these blocks and not lower them to
    // anything.
    auto it = block.without_terminator();
    return block.isEntryBlock() && it.begin() == it.end();
  }

  LogicalResult verifyBdInBlock(Block &block, bool hasRuntimeBdId = false) {
    auto bd_ops = block.getOps<AIE::DMABDOp>();
    // Exactly one BD op per block
    int n_bd_ops = std::distance(bd_ops.begin(), bd_ops.end());
    if (n_bd_ops < 1) {
      auto error = block.getTerminator()->emitError(
          "Block ending in this terminator does not contain a required "
          "aie.dma_bd operation.");
      error.attachNote(block.getParentOp()->getLoc())
          << "Error encountered while lowering this BD configuration.";
      return failure();
    } else if (n_bd_ops > 1) {
      auto error = block.getTerminator()->emitOpError(
          "This block contains multiple aie.dma_bd operations. Exactly one is "
          "required.");
      auto it = bd_ops.begin();
      ++it;
      for (; it != bd_ops.end(); ++it) {
        error.attachNote((*it)->getLoc()) << "Extra aie.dma_bd operation here.";
      }
      return failure();
    }
    AIE::DMABDOp bd_op = *bd_ops.begin();
    // A runtime bd_id (dynamic free-list pool, on the configure's bd_id_val)
    // takes the place of the static attribute; only require the attribute when
    // there is no runtime id.
    if (!hasRuntimeBdId && !bd_op.getBdId().has_value()) {
      auto error = bd_op.emitOpError(
          "Cannot lower buffer descriptor without assigned ID.");
      error.attachNote()
          << "Run the `--aie-assign-runtime-sequence-bd-ids` pass first or "
             "manually assign an ID to this buffer descriptor.";
      error.attachNote(block.getParentOp()->getLoc())
          << "Error encountered while lowering this BD configuration.";
      return failure();
    }
    return success();
  }

  LogicalResult verifyOptionalLocksInBlock(Block &block) {
    auto lock_ops = block.getOps<AIE::UseLockOp>();
    int n_lock_ops = std::distance(lock_ops.begin(), lock_ops.end());
    // Allow exactly 0 or 2 lock ops (acquire and release)
    if (n_lock_ops != 0 && n_lock_ops != 2) {
      AIE::UseLockOp lock_op = *lock_ops.begin();
      lock_op.emitOpError(
          "BD blocks must have either 0 or 2 lock operations (acquire and "
          "release). Found ")
          << n_lock_ops << " lock operations.";
      return failure();
    }
    return success();
  }

  LogicalResult verifyNoUnsupportedOpsInBlock(Block &block) {
    WalkResult unsupported_ops = block.walk([&](Operation *inner_op) {
      return llvm::TypeSwitch<Operation *, WalkResult>(inner_op)
          .Case<AIE::DMABDOp>(
              [&](AIE::DMABDOp bd_op) { return WalkResult::advance(); })
          .Case<AIE::UseLockOp>(
              [&](AIE::UseLockOp lock_op) { return WalkResult::advance(); })
          .Case<arith::ConstantOp>(
              [&](arith::ConstantOp const_op) { return WalkResult::advance(); })
          .Case<AIE::NextBDOp>(
              [&](AIE::NextBDOp lock_op) { return WalkResult::advance(); })
          .Case<AIE::EndOp>(
              [&](AIE::EndOp lock_op) { return WalkResult::advance(); })
          .Default([&](Operation *inner_op) {
            auto error = block.getParentOp()->emitOpError(
                "Unsupported operation within BD block.");
            error.attachNote(inner_op->getLoc())
                << "No lowering to NPU instructions available for this "
                   "operation.";
            return WalkResult::interrupt();
          });
    });
    if (unsupported_ops.wasInterrupted()) {
      return failure();
    }
    return success();
  }

  AIE::DMABDOp getBdForBlock(Block &block) {
    auto bd_ops = block.getOps<AIE::DMABDOp>();
    AIE::DMABDOp bd_op = *bd_ops.begin(); // Dereference first (and only, after
                                          // previous checks) bd op iterator
    return bd_op;
  }

  // Returns pair of (acquire_lock_op, release_lock_op) if present
  std::optional<std::pair<AIE::UseLockOp, AIE::UseLockOp>>
  getOptionalLockOpsForBlock(Block &block) {
    auto lock_ops = block.getOps<AIE::UseLockOp>();
    int n_lock_ops = std::distance(lock_ops.begin(), lock_ops.end());
    if (n_lock_ops != 2) {
      return std::nullopt;
    }

    AIE::UseLockOp acquire_op = nullptr;
    AIE::UseLockOp release_op = nullptr;

    for (auto lock_op : lock_ops) {
      if (lock_op.acquire() || lock_op.acquireGE()) {
        acquire_op = lock_op;
      } else if (lock_op.release()) {
        release_op = lock_op;
      }
    }

    if (acquire_op && release_op) {
      return std::make_pair(acquire_op, release_op);
    }
    return std::nullopt;
  }

  LogicalResult setAddressForSingleBD(OpBuilder &builder, AIE::DMABDOp &bd_op,
                                      AIE::TileOp &tile,
                                      OpFoldResult bdId = {}) {
    const AIE::AIETargetModel &target_model = AIE::getTargetModel(bd_op);
    auto buf = bd_op.getBuffer();
    auto col = tile.getCol();
    auto row = tile.getRow();
    // The static register address uses the pinned bd_id attribute; on the
    // runtime pool path the attribute is absent and runtimeRegisterAddr (below)
    // supplies the address instead, so fall back to bd 0 for the constant.
    uint32_t bd_id = bd_op.getBdId().value_or(0);
    uint64_t register_addr = target_model.getDmaBdAddress(col, row, bd_id) +
                             target_model.getDmaBdAddressOffset(col, row);
    // On the runtime-bd_id path the patched register (BD buffer-address word)
    // is itself runtime: getBdRegisterBase(bd_id) + getDmaBdAddressOffset.
    // Emitted as an SSA operand on the address patch; null keeps the static
    // constant.
    Value runtimeRegisterAddr;
    if (bdId && !getConstantIntValue(bdId)) {
      Value base = getBdRegisterBase(builder, bd_op.getLoc(), target_model, col,
                                     row, bdId);
      runtimeRegisterAddr = arith::AddIOp::create(
          builder, bd_op.getLoc(), base,
          createConstantI32(builder, bd_op.getLoc(),
                            target_model.getDmaBdAddressOffset(col, row)));
    }

    // A buffer descriptor can refer to a statically allocated aie.buffer, or to
    // a DDR buffer which will be passed as a runtime argument (block
    // argument). Try to find the root block argument, either directly or
    // through supported subview/view/cast chains.
    mlir::BlockArgument buf_arg = nullptr;
    int64_t offset = 0;

    if (auto directArg = llvm::dyn_cast<mlir::BlockArgument>(buf)) {
      buf_arg = directArg;
      offset = 0;
    } else if (auto traceResult = traceSubviewToBlockArgument(buf)) {
      buf_arg = traceResult->rootArg;
      offset = traceResult->offsetInBytes;
    }

    if (buf_arg) {
      if (!target_model.isShimNOCTile(tile.getCol(), tile.getRow())) {
        return bd_op->emitOpError("DDR memory (runtime input arguments) can "
                                  "only be referred to on shim tiles.");
      }

      unsigned arg_idx = buf_arg.getArgNumber();
      // arg_plus = buffer byte offset. The dma_bd offset is a single element
      // offset (stride 1); a constant folds to a constant (byte-identical to
      // before), a runtime offset operand is built with arith. `offset` here is
      // the constant subview base in bytes.
      OpFoldResult offsetOfr =
          bd_op.getOffset() ? OpFoldResult(bd_op.getOffset())
                            : OpFoldResult(builder.getI32IntegerAttr(
                                  bd_op.getConstantOffset().value_or(0)));
      OpFoldResult oneStride = builder.getI32IntegerAttr(1);
      Value argPlus =
          buildArgPlusValue(builder, bd_op.getLoc(), {offsetOfr}, {oneStride},
                            bd_op.getBufferElementTypeWidthInBytes(), offset);
      NpuAddressPatchOp::create(builder, bd_op.getLoc(),
                                /*addr*/ register_addr,
                                /*addr_val*/ runtimeRegisterAddr,
                                /*arg_idx*/ arg_idx, argPlus);
    } else if (AIE::BufferOp buffer =
                   llvm::dyn_cast<AIE::BufferOp>(buf.getDefiningOp())) {
      uint64_t buf_addr;
      if (!buffer.getAddress().has_value()) {
        return bd_op->emitOpError(
            "Cannot lower buffer without associated address. Run pass "
            "--aie-assign-buffer-addresses first or manually assign an "
            "address.");
      }
      buf_addr = *buffer.getAddress();
      buf_addr += bd_op.getOffsetInBytes();
      if (target_model.isCoreTile(col, row)) {
        NpuMaskWrite32Op::create(
            builder, bd_op.getLoc(),
            createConstantI32(builder, bd_op.getLoc(),
                              static_cast<uint32_t>(register_addr)),
            createConstantI32(builder, bd_op.getLoc(),
                              static_cast<uint32_t>((buf_addr / 4) << 14)),
            createConstantI32(builder, bd_op.getLoc(), 0x0fffc000), nullptr,
            nullptr, nullptr);
      } else if (target_model.isMemTile(col, row)) {
        // On AIE2p (NPU2), memtile DMAs use an offset-based address
        // space where the base depends on the relative position of the
        // buffer's tile (west=0, internal=getMemTileSize, east=2x).
        // On AIE2 (NPU1), memtile DMAs address local memory directly
        // starting at 0. Only add the offset for AIE2p.
        if (target_model.getTargetArch() == AIE::AIEArch::AIE2p) {
          auto addrOffset = target_model.getMemLocalBaseAddress(
              col, row, buffer.getTileOp().getCol(),
              buffer.getTileOp().getRow());
          if (addrOffset)
            buf_addr += addrOffset.value();
        }
        NpuMaskWrite32Op::create(
            builder, bd_op.getLoc(),
            createConstantI32(builder, bd_op.getLoc(),
                              static_cast<uint32_t>(register_addr)),
            createConstantI32(builder, bd_op.getLoc(),
                              static_cast<uint32_t>(buf_addr / 4)),
            createConstantI32(builder, bd_op.getLoc(), 0x0007FFFF), nullptr,
            nullptr, nullptr);
      } else {
        NpuWrite32Op::create(
            builder, bd_op.getLoc(),
            createConstantI32(builder, bd_op.getLoc(),
                              static_cast<uint32_t>(register_addr)),
            createConstantI32(builder, bd_op.getLoc(),
                              static_cast<uint32_t>(buf_addr)),
            nullptr, nullptr, nullptr);
      }
    } else {
      return bd_op->emitOpError(
          "Buffer argument must be a constant aie.buffer, a runtime sequence "
          "input argument, or a supported chain of memref.subview, "
          "memref.view, memref.cast, or memref.reinterpret_cast operations "
          "rooted at a block argument. Subviews must be static and contiguous; "
          "views must have a constant byte shift and no dynamic result sizes.");
    }

    // If this BD has an offset_state_table_idx, emit update_from_scratchpad to
    // add the runtime offset to the BD address register. This is applied after
    // the base address is set (by either NpuAddressPatchOp for DDR buffers or
    // NpuMaskWrite32Op/NpuWrite32Op for on-chip buffers), since the hardware
    // update_from_scratchpad instruction is additive -- it reads the existing
    // register value and adds a computed delta to it.
    if (bd_op.getOffsetStateTableIdxAttr()) {
      auto bufType = llvm::cast<BaseMemRefType>(bd_op.getBuffer().getType());
      if (failed(emitUpdateBdAddressFromOffsetParameter(builder, bd_op, bufType,
                                                        register_addr)))
        return failure();
    }

    return success();
  }

  // Compile-time BD template fields shared by the static and dynamic lowering
  // paths (only sizes/strides/len can be runtime on a dma_bd; everything here
  // is always constant). The dynamic path bakes these into its zero-template
  // NpuWriteBdOp exactly as the static path does, so a runtime BD keeps its
  // locks, packet header and next_bd chaining.
  struct BdTemplateFields {
    uint32_t use_next_bd = 0, next_bd_id = 0;
    int32_t enable_packet = 0, packet_id = 0, packet_type = 0;
    int32_t lock_rel_val = 0, lock_rel_id = 0;
    int32_t lock_acq_enable = 0, lock_acq_val = 0, lock_acq_id = 0;
  };

  // Gather the constant lock / packet / next_bd fields for a BD block. Returns
  // failure if a lock carries a non-constant value.
  FailureOr<BdTemplateFields>
  gatherBdTemplateFields(Block &block, AIE::DMABDOp bd_op, AIE::TileOp &tile,
                         const AIE::AIETargetModel &target_model,
                         std::optional<xilinx::AIE::PacketInfoAttr> packet) {
    BdTemplateFields f;
    if (bd_op.getNextBdId().has_value()) {
      f.next_bd_id = bd_op.getNextBdId().value();
      f.use_next_bd = 1;
    }

    auto info = bd_op.getPacket().value_or(packet.value_or(nullptr));
    if (info) {
      f.enable_packet = 1;
      f.packet_type = info.getPktType();
      f.packet_id = info.getPktId();
    }

    auto lock_ops = getOptionalLockOpsForBlock(block);
    if (lock_ops) {
      auto [acquire_op, release_op] = *lock_ops;
      AIE::LockOp acq_lock = acquire_op.getLockOp();
      AIE::LockOp rel_lock = release_op.getLockOp();

      if (acq_lock.getLockID().has_value()) {
        f.lock_acq_id = acq_lock.getLockID().value();
        auto value = acquire_op.getConstantValue();
        if (failed(value))
          return failure();
        f.lock_acq_val = *value;
        // For AcquireGreaterEqual, negate the value to signal the hardware to
        // use >= comparison instead of == comparison.
        if (acquire_op.acquireGE())
          f.lock_acq_val = -f.lock_acq_val;
        f.lock_acq_enable = 1;
      }

      if (rel_lock.getLockID().has_value()) {
        f.lock_rel_id = rel_lock.getLockID().value();
        auto value = release_op.getConstantValue();
        if (failed(value))
          return failure();
        f.lock_rel_val = *value;
      }

      // For memtile, add lock offset using getLockLocalBaseIndex. This matches
      // AIERT.cpp implementation.
      if (target_model.isMemTile(tile.getCol(), tile.getRow())) {
        auto lockOffset = target_model.getLockLocalBaseIndex(
            tile.getCol(), tile.getRow(), acq_lock.colIndex(),
            acq_lock.rowIndex());
        if (lockOffset && acq_lock.getLockID().has_value())
          f.lock_acq_id += lockOffset.value();
        if (lockOffset && rel_lock.getLockID().has_value())
          f.lock_rel_id += lockOffset.value();
      }
    }
    return f;
  }

  // Dynamic (runtime SSA size/stride/len) shim-NOC BD lowering, the dma_task
  // sibling of DmaToNpuPattern::lowerDynamic. Emits a zero-template
  // NpuWriteBdOp (constant locks/packet/next_bd baked in, size/stride words
  // zeroed) folded to one blockwrite, then per-word write32 overrides via the
  // shared encoder. Scope (shim NOC, no padding, realizability) is enforced by
  // the caller.
  LogicalResult
  rewriteSingleBDDynamic(OpBuilder &builder, Block &block, AIE::DMABDOp bd_op,
                         AIE::TileOp &tile,
                         std::optional<xilinx::AIE::PacketInfoAttr> packet,
                         Value runtimeBdId = nullptr) {
    const auto &target_model = AIE::getTargetModel(bd_op);
    Location loc = bd_op.getLoc();
    auto i32ty = builder.getIntegerType(32);
    int col = tile.getCol();
    int row = tile.getRow();

    auto fieldsOr =
        gatherBdTemplateFields(block, bd_op, tile, target_model, packet);
    if (failed(fieldsOr))
      return failure();
    BdTemplateFields f = *fieldsOr;

    // The bd_id as an OpFoldResult: the runtime pool value if present, else the
    // pinned constant attribute.
    OpFoldResult bdIdOfr =
        runtimeBdId
            ? OpFoldResult(runtimeBdId)
            : OpFoldResult(builder.getI32IntegerAttr(bd_op.getBdId().value()));

    if (runtimeBdId) {
      // A runtime bd_id makes the register block's address runtime, so the
      // constant-address blockwrite path can't be used. Emit the template words
      // as write32s instead -- register replay treats N write32s and an N-word
      // blockwrite identically, matching the static BD.
      if (failed(emitShimTemplateWordOverrides(builder, loc, target_model, col,
                                               row, bdIdOfr, f,
                                               bd_op.getBurstLength())))
        return failure();
    } else {
      // Zero-template BD: constant fields baked in, size/stride words zeroed
      // for the write32 overrides to fill. valid_bd = 1.
      NpuWriteBdOp::create(
          builder, loc, col, bd_op.getBdId().value(), /*buffer_length=*/0,
          /*buffer_offset=*/0, f.enable_packet, /*out_of_order_id=*/0,
          f.packet_id, f.packet_type, /*d0_size=*/0, /*d0_stride=*/0,
          /*d1_size=*/0,
          /*d1_stride=*/0, /*d2_size=*/0, /*d2_stride=*/0,
          /*iteration_current=*/0,
          /*iteration_size=*/0, /*iteration_stride=*/0, f.next_bd_id, row,
          f.use_next_bd, /*valid_bd=*/1, f.lock_rel_val, f.lock_rel_id,
          f.lock_acq_enable, f.lock_acq_val, f.lock_acq_id,
          /*d0_zero_before=*/0,
          /*d1_zero_before=*/0, /*d2_zero_before=*/0, /*d0_zero_after=*/0,
          /*d1_zero_after=*/0, /*d2_zero_after=*/0, bd_op.getBurstLength());
    }

    // Normalize sizes/strides to a 4-element outermost-first mixed list (the
    // shared emitter's contract, matching memcpy_nd's always-4D operands),
    // padding absent leading dims with size 1 / stride 0. dma_bd's mixed lists
    // are outermost-first and variable-length (0..4 dims).
    SmallVector<OpFoldResult> sizes(bd_op.getMixedSizes());
    SmallVector<OpFoldResult> strides(bd_op.getMixedStrides());
    if (sizes.size() > 4 || strides.size() > 4)
      return bd_op->emitOpError("At most four data layout transformation "
                                "dimensions may be provided.");
    OpFoldResult one = builder.getI64IntegerAttr(1);
    OpFoldResult zeroOfr = builder.getI64IntegerAttr(0);
    SmallVector<OpFoldResult, 4> sizes4(4, one), strides4(4, zeroOfr);
    for (size_t i = 0; i < sizes.size(); i++) {
      sizes4[4 - sizes.size() + i] = sizes[i];
      strides4[4 - strides.size() + i] = strides[i];
    }

    // buffer_length override = len (elements) * elemWidth / addressGranularity,
    // as an SSA value. dma_task carries the transfer length explicitly (unlike
    // memcpy_nd's size-product), and it may be runtime.
    uint64_t elemWidth =
        static_cast<uint64_t>(bd_op.getBufferElementTypeWidthInBytes()) * 8;
    uint32_t gran = target_model.getAddressGenGranularity();
    // len as OpFoldResult: the runtime operand if present, else the static_len
    // attr (a constant BD with runtime sizes/strides still reaches here).
    OpFoldResult lenOfr;
    if (Value lenOperand = bd_op.getLen())
      lenOfr = lenOperand;
    else
      lenOfr = builder.getI32IntegerAttr(bd_op.getConstantLen().value());
    Value lenVal = getAsValue(builder, loc, lenOfr, i32ty);
    Value bufLen = arith::DivUIOp::create(
        builder, loc,
        arith::MulIOp::create(builder, loc, lenVal,
                              createConstantI32(builder, loc, elemWidth)),
        createConstantI32(builder, loc, gran));

    // The BD-level repeat_count (encoder output) is unused here: the dma_task
    // queue push is emitted separately by DMAStartTaskOpPattern from the task
    // op's repeat_count, not the BD's outer dim.
    Value bdRepeatCount;
    if (failed(emitDynamicShimBdWordOverrides(
            builder, loc, target_model, col, row, bdIdOfr, sizes4, strides4,
            elemWidth, bd_op.getBurstLength(), bufLen, bdRepeatCount)))
      return failure();
    return setAddressForSingleBD(builder, bd_op, tile, bdIdOfr);
  }

  // Emit the shim BD template words (those the size/stride encoder doesn't own)
  // as runtime-addressed write32s, for the runtime-bd_id path where a constant-
  // address zero-template blockwrite can't be formed. Layout mirrors
  // WriteBdToBlockWritePattern. Words 4/5 carry constant burst_length/AXCache
  // bits the encoder's later ND write32 overwrites by last-write; 1/2/7 unused.
  LogicalResult emitShimTemplateWordOverrides(
      OpBuilder &builder, Location loc, const AIE::AIETargetModel &target_model,
      int col, int row, OpFoldResult bdId, const BdTemplateFields &f,
      uint32_t burstLength) {
    Value bdBase =
        getBdRegisterBase(builder, loc, target_model, col, row, bdId);
    auto writeWord = [&](uint32_t wordIdx, uint32_t val) {
      Value addr = arith::AddIOp::create(
          builder, loc, bdBase, createConstantI32(builder, loc, wordIdx * 4));
      NpuWrite32Op::create(builder, loc, addr,
                           createConstantI32(builder, loc, val), nullptr,
                           nullptr, nullptr);
    };
    // word[1] buffer_offset: 0 (the address patch supplies the buffer pointer).
    writeWord(1, 0);
    // word[2] enable_packet [30], out_of_order_id [29:24], packet_id [23:19],
    // packet_type [18:16].
    uint32_t w2 = ((f.enable_packet & 0x1) << 30) |
                  ((f.packet_id & 0x1f) << 19) | ((f.packet_type & 0x7) << 16);
    writeWord(2, w2);
    // word[4] burst_length [31:30] (constant); d1_size/stride overlaid by the
    // encoder in ND mode.
    writeWord(4,
              (AIE::getShimBurstLengthEncoding(target_model, burstLength) & 0x3)
                  << 30);
    // word[5] AXCache [27:24] = 2 (constant, enables NoC upsizing); d2_stride
    // overlaid by the encoder in ND mode.
    writeWord(5, (2u & 0xf) << 24);
    // word[7] next_bd [30:27], use_next_bd [26], valid_bd [25], lock fields.
    uint32_t w7 = ((f.next_bd_id & 0xf) << 27) | ((f.use_next_bd & 0x1) << 26) |
                  (1u << 25) | ((f.lock_rel_val & 0x7f) << 18) |
                  ((f.lock_rel_id & 0xf) << 13) |
                  ((f.lock_acq_enable & 0x1) << 12) |
                  ((f.lock_acq_val & 0x7f) << 5) | (f.lock_acq_id & 0xf);
    writeWord(7, w7);
    return success();
  }

  LogicalResult
  rewriteSingleBD(OpBuilder &builder, Block &block, AIE::TileOp &tile,
                  AIE::DMAChannelDir channelDir,
                  std::optional<xilinx::AIE::PacketInfoAttr> packet,
                  Value runtimeBdId = nullptr) {
    AIE::DMABDOp bd_op = getBdForBlock(block);
    const auto &target_model = AIE::getTargetModel(bd_op);
    auto buffer_type = llvm::cast<BaseMemRefType>(bd_op.getBuffer().getType());
    uint32_t addr_granularity = target_model.getAddressGenGranularity();

    // Runtime (SSA) sizes/strides/len/offset take the dynamic BD-word encoder
    // path; a runtime bd_id (dynamic free-list pool) also forces it, since the
    // BD register addresses are then runtime and cannot fold into a blockwrite.
    // A fully-constant descriptor with a pinned bd_id takes the static path
    // below unchanged. Only the shim-NOC layout is encodable this way (see
    // rewriteSingleBDDynamic), so anything the dynamic path can't represent
    // stays a clean diagnostic.
    bool runtimeLen = bd_op.getLen() && !bd_op.getConstantLen();
    bool runtimeOffset = bd_op.getOffset() && !bd_op.getConstantOffset();
    bool runtimeDims =
        llvm::any_of(bd_op.getMixedSizes(),
                     [](OpFoldResult s) { return !getConstantIntValue(s); }) ||
        llvm::any_of(bd_op.getMixedStrides(),
                     [](OpFoldResult s) { return !getConstantIntValue(s); });
    if (runtimeLen || runtimeDims || runtimeOffset || runtimeBdId) {
      if (!target_model.isShimNOCTile(tile.getCol(), tile.getRow()))
        return bd_op->emitOpError(
            "runtime-valued BD size/stride/len/bd_id is only supported on shim "
            "NOC tiles; use compile-time constants on other tiles.");
      if (bd_op.getPadDimensions().has_value())
        return bd_op->emitOpError(
            "zero padding is not supported with runtime sizes/strides/len.");
      // Realizability of the constant size/stride operands (runtime ones are
      // guarded at lowering by the shared encoder). Mixed lists are
      // outermost-first; the helper wants innermost-first.
      uint64_t elemWidth =
          static_cast<uint64_t>(bd_op.getBufferElementTypeWidthInBytes()) * 8;
      SmallVector<OpFoldResult, 4> sizesRev(
          llvm::reverse(bd_op.getMixedSizes()));
      SmallVector<OpFoldResult, 4> stridesRev(
          llvm::reverse(bd_op.getMixedStrides()));
      if (failed(verifyConstBdRealizability(
              bd_op, sizesRev, stridesRev, elemWidth,
              target_model.getAddressGenGranularity())))
        return failure();
      return rewriteSingleBDDynamic(builder, block, bd_op, tile, packet,
                                    runtimeBdId);
    }

    // Static path: bd_id is a pinned attribute (the dynamic/runtime-bd_id path
    // returned above) and the offset is constant (runtime offset routed above).
    uint32_t bd_id = bd_op.getBdId().value();
    int64_t offset = bd_op.getOffsetInBytes();
    uint64_t len = bd_op.getLenInBytes();
    uint64_t len_addr_granularity = len * 8 / addr_granularity;

    if (offset * 8 % addr_granularity != 0) {
      return bd_op->emitOpError("Offset must be aligned to ")
             << (addr_granularity / 8) << " byte boundary.";
    }

    if (len < addr_granularity / 8) {
      return bd_op->emitOpError("Transfer size of ")
             << len << " bytes falls below minimum hardware transfer unit of "
             << (addr_granularity / 8) << " bytes.";
    }
    // The owning storage must outlive the ArrayRef view below.
    std::optional<llvm::SmallVector<AIE::BDDimLayoutAttr>> dimsStorage =
        bd_op.getConstantDimensions();
    if (!dimsStorage)
      return bd_op->emitOpError("internal error folding BD dimensions");
    std::optional<llvm::ArrayRef<AIE::BDDimLayoutAttr>> dims;
    if (!dimsStorage->empty())
      dims = llvm::ArrayRef<AIE::BDDimLayoutAttr>(*dimsStorage);
    llvm::SmallVector<int64_t, 4> sizes = llvm::SmallVector<int64_t, 4>(4, 0);
    llvm::SmallVector<int64_t, 4> strides = llvm::SmallVector<int64_t, 4>(4, 0);

    // Padding
    std::optional<llvm::ArrayRef<AIE::BDPadLayoutAttr>> padDims =
        bd_op.getPadDimensions();
    llvm::SmallVector<int64_t, 4> padBefore =
        llvm::SmallVector<int64_t, 4>(4, 0);
    llvm::SmallVector<int64_t, 4> padAfter =
        llvm::SmallVector<int64_t, 4>(4, 0);
    std::fill(padBefore.begin(), padBefore.end(), 0);
    std::fill(padAfter.begin(), padAfter.end(), 0);

    auto out_of_order_id = 0;
    auto d0size = 0;
    auto d0stride = 0;
    auto d1size = 0;
    auto d1stride = 0;
    auto d2size = 0;
    auto d2stride = 0;
    auto iteration_size = 0;
    auto iteration_stride = 0;

    if (dims && dims->size() > 0) {
      llvm::SmallVector<int64_t, 4> input_sizes =
          llvm::SmallVector<int64_t, 4>(4, 1);
      llvm::SmallVector<int64_t, 4> input_strides =
          llvm::SmallVector<int64_t, 4>(4, 0);
      if (dims->size() > 4) {
        return bd_op->emitOpError("At most four data layout transformation "
                                  "dimensions may be provided.");
      }

      for (size_t i = 0; i < dims->size(); i++) {
        // Pass down dimensions in reverse order; in the MLIR, this allows
        // us to specify step sizes/wraps in the same order as we would
        // access a multi-dim C array, with the highest dimension first.
        int j = dims->size() - i - 1;
        input_sizes[i] = (*dims)[j].getSize();
        input_strides[i] = (*dims)[j].getStride();
      }

      // d3 (repeat) is excluded; a repeated linear transfer is still linear.
      // A contiguous row-major ND access on a shim NOC tile is also lowered
      // using the wide buffer_length register, exempt from the 10-bit ND
      // wrap-size limit.  Canonicalization zeroes size-1 strides before this
      // pass runs, so isContiguousTransfer is sufficient.
      bool treatAsLinear =
          isLinearTransfer(input_sizes, input_strides) ||
          (target_model.isShimNOCTile(tile.getCol(), tile.getRow()) &&
           isContiguousTransfer(input_sizes, input_strides));

      if (dims->size() > 2) {
        d2size = (target_model.isMemTile(tile.getCol(), tile.getRow()))
                     ? (*dims)[2].getSize()
                     : 0;
      }
      if (padDims.has_value()) {
        if (!target_model.isMemTile(tile.getCol(), tile.getRow()))
          return bd_op->emitOpError()
                 << "Padding is only supported by memtile dma bds.";
        if (padDims->size() > dims->size())
          return bd_op->emitOpError()
                 << "Mismatch number of dimensions between padding(s)"
                 << " and wrap(s) and stride(s).";
        if (channelDir == AIE::DMAChannelDir::MM2S) {
          for (size_t i = 0; i < padDims->size(); i++) {
            int j = padDims->size() - i - 1;
            padBefore[i] = (*padDims)[j].getConstPadBefore();
            padAfter[i] = (*padDims)[j].getConstPadAfter();
          }
          for (size_t i = padDims->size(); i < dims->size(); i++) {
            padBefore[i] = 0;
            padAfter[i] = 0;
          }
        } else
          return bd_op->emitOpError()
                 << "supports padding only for MM2S direction on MemTiles.";
      }
      getHardwareStridesWraps(target_model, bd_op, buffer_type, input_sizes,
                              input_strides, sizes, strides);

      if (failed(verifyStridesWraps(bd_op, buffer_type, tile.getCol(),
                                    tile.getRow(), input_sizes, input_strides,
                                    sizes, strides, treatAsLinear))) {
        return failure();
      }

      iteration_size = sizes[3];
      iteration_stride = strides[3];

      if (!treatAsLinear) {
        // d0_size, d0_stride
        d0size = sizes[0];
        d0stride = strides[0];

        // d1_size, d1_stride
        d1size = sizes[1];
        d1stride = strides[1];

        // d2_stride
        d2stride = strides[2];
        // d2_size set elsewhere
      }
      if (input_sizes[3] > 1 && input_strides[3] == 0) {
        // We allow users to encode the repeat_count as a dimension 3 stride
        // of 0. This must lower to a iteration wrap of 0, so no stride is
        // ever added. We then repeat the BD using the repeat_count in
        // NpuPushQueueOp.
        iteration_size = 0;
        iteration_stride = 0;
      }

      // Ensure the total transfer length and the length expressed in the lowest
      // three dimensions of strides/wraps agree. (Fourth dimension is
      // iteration/repeat count and repeats the whole BD, so should not be
      // incorporated in length of a single BD invocation.)
      uint64_t len_dims_addr_granularity = 1;
      for (size_t i = 0; i < 3; i++) {
        len_dims_addr_granularity *= sizes[i];
      }
      if (len_dims_addr_granularity != len_addr_granularity) {
        auto err =
            bd_op->emitOpError(
                "Buffer descriptor length does not match length of transfer "
                "expressed by lowest three dimensions of data layout "
                "transformation strides/wraps. ")
            << "BD length is " << (len_addr_granularity * addr_granularity / 8)
            << " bytes. "
            << "Lowest three dimensions of data layout transformation would "
               "result in transfer of "
            << (len_dims_addr_granularity * addr_granularity / 8) << " bytes. ";
        err.attachNote() << "Do not include the highest dimension size in "
                            "transfer length, as this is the BD repeat count.";
        return failure();
      }
    } else {
      if (padDims && target_model.isMemTile(tile.getCol(), tile.getRow()) &&
          channelDir == AIE::DMAChannelDir::MM2S) {
        return bd_op->emitOpError()
               << "Padding requires n-d data layouts expressed as "
               << "wrap(s) and stride(s).";
      } else if (padDims) {
        return bd_op->emitOpError() << "Padding is supported only on MemTiles.";
      }
    }
    auto fieldsOr =
        gatherBdTemplateFields(block, bd_op, tile, target_model, packet);
    if (failed(fieldsOr))
      return failure();
    BdTemplateFields f = *fieldsOr;

    NpuWriteBdOp::create(
        builder, bd_op.getLoc(), tile.getCol(), bd_id, len_addr_granularity,
        offset,
        /*enable_packet=*/f.enable_packet,
        /*out_of_order_id=*/out_of_order_id,
        /*packet_id=*/f.packet_id,
        /*packet_type=*/f.packet_type,
        /*d0_size=*/d0size, /*d0_stride=*/d0stride,
        /*d1_size=*/d1size, /*d1_stride=*/d1stride,
        /*d2_size=*/d2size, /*d2_stride=*/d2stride,
        /*iteration_current=*/0, /*iteration_size=*/iteration_size,
        /*iteration_stride=*/iteration_stride,
        /*next_bd=*/f.next_bd_id,
        /*row=*/tile.getRow(),
        /*use_next_bd=*/f.use_next_bd,
        /*valid_bd=*/1,
        /*lock_rel_val=*/f.lock_rel_val, /*lock_rel_id=*/f.lock_rel_id,
        /*lock_acq_enable=*/f.lock_acq_enable,
        /*lock_acq_val=*/f.lock_acq_val, /*lock_acq_id=*/f.lock_acq_id,
        /*d0_zero_before=*/padBefore[0],
        /*d1_zero_before=*/padBefore[1], /*d2_zero_before=*/padBefore[2],
        /*d0_zero_after=*/padAfter[0], /*d1_zero_after=*/padAfter[1],
        /*d2_zero_after=*/padAfter[2],
        /*burst_length=*/bd_op.getBurstLength());
    return setAddressForSingleBD(builder, bd_op, tile);
  }

  LogicalResult hoistNextBdOpsIntoAttrs(DMAConfigureTaskOp op) {
    Region &body = op.getBody();
    for (auto it = body.begin(); it != body.end(); ++it) {
      Block &block = *it;
      if (shouldSkipBlock(block)) {
        continue;
      }
      AIE::DMABDOp bd_op = getBdForBlock(block);
      if (AIE::NextBDOp next_bd_op =
              llvm::dyn_cast<AIE::NextBDOp>(block.getTerminator())) {
        if (bd_op.getNextBdId().has_value()) {
          auto error =
              bd_op.emitOpError("Cannot specify both next_bd_id attribute and "
                                "aie.next_bd operation.");
          error.attachNote(next_bd_op.getLoc())
              << "Potentially conflicting next buffer descriptor ID specified "
                 "here.";
          return failure();
        }
        Block &next_bd_block = *next_bd_op.getDest();
        AIE::DMABDOp next_dma_bd_op = getBdForBlock(next_bd_block);
        assert(next_dma_bd_op.getBdId()
                   .has_value()); // Next BD should have assigned ID, and this
                                  // should have been checked by earlier
                                  // verifyBdInBlock() call
        bd_op.setNextBdId(next_dma_bd_op.getBdId().value());
        OpBuilder builder(next_bd_op);
        AIE::EndOp::create(builder, next_bd_op.getLoc());
        next_bd_op.erase();
      }
    }
    return success();
  }

  LogicalResult rewriteSingleDMAConfigureTaskOp(DMAConfigureTaskOp op) {
    OpBuilder builder(op);
    AIE::TileOp tile = op.getTileOp();

    if (!op.use_empty()) {
      auto err = op.emitOpError("Cannot lower while op still has uses.");
      mlir::Operation::use_range uses = op.getOperation()->getUses();
      for (auto it = uses.begin(); it != uses.end(); ++it) {
        err.attachNote(it->getOwner()->getLoc()) << "Used here.";
      }
      return failure();
    }

    Region &body = op.getBody();
    bool hasRuntimeBdId = op.getBdIdVal() != nullptr;

    // Verify each BD block first; subsequent functions rely on them being
    // well-formed
    for (auto it = body.begin(); it != body.end(); ++it) {
      if (shouldSkipBlock(*it)) {
        continue;
      }
      if (failed(verifyNoUnsupportedOpsInBlock(*it))) {
        return failure();
      }
      if (failed(verifyBdInBlock(*it, hasRuntimeBdId))) {
        return failure();
      }
      if (failed(verifyOptionalLocksInBlock(*it))) {
        return failure();
      }
    }

    // Hoist next_bd operations into next_bd_id attribute of the dma_bd
    if (failed(hoistNextBdOpsIntoAttrs(op))) {
      return failure();
    }

    auto channelDir = op.getDirection();
    auto packet = op.getPacket();
    // A runtime bd_id (dynamic free-list pool) supplied by
    // aie-lower-dynamic-bd-pool; null on the static/pinned path.
    Value runtimeBdId = op.getBdIdVal();

    // Lower all BDs
    for (auto it = body.begin(); it != body.end(); ++it) {
      Block &block = *it;
      if (shouldSkipBlock(block)) {
        continue;
      }
      if (failed(rewriteSingleBD(builder, block, tile, channelDir, packet,
                                 runtimeBdId))) {
        return failure();
      }
    }

    op.erase();

    return success();
  }

  LogicalResult rewriteDMAConfigureTaskOp(AIE::DeviceOp device) {
    WalkResult result = device.walk([&](DMAConfigureTaskOp op) {
      if (failed(rewriteSingleDMAConfigureTaskOp(op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return failure();
    }
    return success();
  }

  // Drop the dead task-index carries left by the dynamic BD pool path once
  // awaits have lowered to npu.sync. scf.for/scf.if carry the task value as a
  // result purely to hold the await's data dependence on the configure; with the
  // await gone that carry is dead but still counts as a use of the branch-local
  // configure, blocking its use_empty lowering. scf's own canonicalizations
  // remove dead iter-args/results and prune the yields feeding them. Run them
  // device-wide but with folding and constant-CSE DISABLED, so the static path's
  // byte-golden constant emission is untouched -- only the dead scf carries go.
  LogicalResult dropDeadTaskCarries(AIE::DeviceOp device) {
    RewritePatternSet patterns(&getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
    GreedyRewriteConfig config;
    config.enableFolding(false).enableConstantCSE(false);
    return applyPatternsGreedily(device, std::move(patterns), config);
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    // Convert DMAStartBD and DMAAwaitBD ops
    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<DMAStartTaskOp>();
    target.addIllegalOp<DMAAwaitTaskOp>();
    RewritePatternSet patterns(&getContext());
    patterns.insert<DMAStartTaskOpPattern>(&getContext());
    patterns.insert<DMAAwaitTaskOpPattern>(&getContext());
    if (failed(applyPartialConversion(device, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Drop the now-dead task-index carries the awaits held, so the branch-local
    // configures they used can reach use_empty and lower below.
    if (failed(dropDeadTaskCarries(device)))
      signalPassFailure();

    // Lower the configuration for the BDs
    if (failed(rewriteDMAConfigureTaskOp(device))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEDMATasksToNPUPass() {
  return std::make_unique<AIEDMATasksToNPUPass>();
}
