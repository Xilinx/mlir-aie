//===- AIEDMATasksToNPU.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
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
    std::optional<uint32_t> first_bd_id = task_op.getFirstBdId();
    if (!first_bd_id) {
      auto err = op.emitOpError(
          "First buffer descriptor in chain has not been assigned an ID");
      err.attachNote() << "Run the `aie-assign-runtime-buffer-descriptor-ids` "
                          "pass first or manually assign an ID.";
      return failure();
    }

    // Check if the first BD has dynamic sizes (SSA operands).
    // If so, repeat_count must be computed dynamically from dyn_sizes[0]
    // (the outermost dimension) and emitted as a direct NpuWrite32Op to
    // the queue register, bypassing NpuPushQueueOp.
    AIE::DMABDOp firstBd;
    for (Block &block : task_op.getBody()) {
      auto bds = block.getOps<AIE::DMABDOp>();
      if (!bds.empty()) {
        firstBd = *bds.begin();
        break;
      }
    }

    if (firstBd && !firstBd.getDynSizes().empty()) {
      // Dynamic repeat_count path — mirrors AIEDmaToNpu.cpp lines 900-934.
      auto loc = op.getLoc();
      auto i32ty = rewriter.getIntegerType(32);
      auto cst = [&](int64_t v) -> Value {
        return arith::ConstantOp::create(rewriter, loc,
                                         IntegerAttr::get(i32ty, v));
      };

      const auto &targetModel = AIE::getTargetModel(op);
      uint32_t tileCol = tile.getCol();
      uint32_t tileRow = tile.getRow();
      auto channelDir = task_op.getDirection();
      auto channelIdx = task_op.getChannel();

      // Compute dynamic repeat_count from dyn_sizes[0] (outermost dim).
      // Hardware repeat_count = sizes[0] - 1, clamped >= 0.
      Value outerSize = firstBd.getDynSizes()[0]; // i64
      Value outerSize32 =
          arith::TruncIOp::create(rewriter, loc, i32ty, outerSize);
      Value one = cst(1);
      Value repeatCount =
          arith::SubIOp::create(rewriter, loc, outerSize32, one);
      Value zero = cst(0);
      Value isPositive = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sgt, repeatCount, zero);
      repeatCount =
          arith::SelectOp::create(rewriter, loc, isPositive, repeatCount, zero);

      // Build queue command:
      //   cmd = (bd_id & 0xF) | ((repeat_count & 0xFF) << 16)
      //       | (issue_token ? 1<<31 : 0)
      Value bdIdVal = cst(*first_bd_id & 0xF);
      Value rcShifted = buildBdWord(rewriter, loc, {{repeatCount, 0xFFu, 16u}});
      Value cmd = arith::OrIOp::create(rewriter, loc, bdIdVal, rcShifted);
      if (task_op.getIssueToken()) {
        Value tokenBit = cst(static_cast<int32_t>(0x80000000u));
        cmd = arith::OrIOp::create(rewriter, loc, cmd, tokenBit);
      }

      // Emit controller_id maskwrite for issue_token (same as
      // PushQueuetoWrite32Pattern).
      uint32_t ctrlOffset = targetModel.getDmaControlAddress(
          tileCol, tileRow, channelIdx, channelDir);
      if (task_op.getIssueToken()) {
        auto device = op->getParentOfType<AIE::DeviceOp>();
        for (auto t : device.getOps<AIE::TileOp>()) {
          if (static_cast<uint32_t>(t.getCol()) == tileCol &&
              static_cast<uint32_t>(t.getRow()) == tileRow &&
              t->hasAttr("controller_id")) {
            auto controllerIdAttr =
                t->getAttrOfType<AIE::PacketInfoAttr>("controller_id");
            uint32_t data = controllerIdAttr.getPktId() << 8;
            uint32_t mask = 0x00001F00;
            NpuMaskWrite32Op::create(rewriter, loc, ctrlOffset, data, mask,
                                     nullptr, nullptr, nullptr);
            break;
          }
        }
      }

      // Emit NpuWrite32Op to queue register.
      uint32_t queueOffset = ctrlOffset + 0x4;
      NpuWrite32Op::create(rewriter, loc,
                           /*address=*/static_cast<uint32_t>(0),
                           /*value=*/static_cast<uint32_t>(0),
                           /*buffer=*/FlatSymbolRefAttr{},
                           /*column=*/IntegerAttr{},
                           /*row=*/IntegerAttr{},
                           /*dyn_address=*/cst(queueOffset),
                           /*dyn_value=*/cmd,
                           /*bd_group=*/IntegerAttr{});
      rewriter.eraseOp(op);
      return success();
    }

    // Static path: emit NpuPushQueueOp (converted to blockwrite by
    // subsequent AIEDmaToNpu pass).
    rewriter.replaceOpWithNewOp<NpuPushQueueOp>(
        op, tile.getCol(), tile.getRow(), task_op.getDirection(),
        task_op.getChannel(), task_op.getIssueToken(), task_op.getRepeatCount(),
        *first_bd_id);
    return success();
  }
};

struct DMAAwaitTaskOpPattern : OpConversionPattern<DMAAwaitTaskOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DMAAwaitTaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DMAConfigureTaskOp task_op = op.getTaskOp();
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
    Type i32 = rewriter.getI32Type();
    auto cst = [&](int32_t v) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, i32, v);
    };
    // Emit constants in operand order (column, row, direction, channel,
    // column_num, row_num) so the generated IR reads top-to-bottom.
    Value column = cst(tile.getCol());
    Value row = cst(tile.getRow());
    Value direction = cst((int32_t)task_op.getDirection());
    Value channel = cst(task_op.getChannel());
    Value columnNum = cst(1);
    Value rowNum = cst(1);
    rewriter.replaceOpWithNewOp<NpuSyncOp>(op, column, row, direction, channel,
                                           columnNum, rowNum);
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

  LogicalResult verifyBdInBlock(Block &block) {
    auto bd_ops = block.getOps<AIE::DMABDOp>();
    // Exactly one BD op per block
    int n_bd_ops = 0;
    for ([[maybe_unused]] auto op : bd_ops) {
      if (++n_bd_ops > 1)
        break;
    }
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
    if (!bd_op.getBdId().has_value()) {
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
    int n_lock_ops = 0;
    for ([[maybe_unused]] auto op : lock_ops) {
      if (++n_lock_ops > 2)
        break;
    }
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
          .Case<AIE::NextBDOp>(
              [&](AIE::NextBDOp lock_op) { return WalkResult::advance(); })
          .Case<AIE::EndOp>(
              [&](AIE::EndOp lock_op) { return WalkResult::advance(); })
          .Default([&](Operation *inner_op) {
            // Allow arith dialect ops inside BD blocks; they compute
            // dynamic operand values (dyn_offset, dyn_len, dyn_sizes,
            // dyn_strides) consumed by aie.dma_bd.
            if (inner_op->getDialect()->getNamespace() == "arith")
              return WalkResult::advance();
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
    int n_lock_ops = 0;
    for ([[maybe_unused]] auto op : lock_ops) {
      if (++n_lock_ops > 2)
        break;
    }
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
                                      AIE::TileOp &tile) {
    uint32_t bd_id = bd_op.getBdId().value();
    const AIE::AIETargetModel &target_model = AIE::getTargetModel(bd_op);
    auto buf = bd_op.getBuffer();
    auto col = tile.getCol();
    auto row = tile.getRow();
    uint64_t register_addr = target_model.getDmaBdAddress(col, row, bd_id) +
                             target_model.getDmaBdAddressOffset(col, row);

    // A buffer descriptor can refer to a statically allocated aie.buffer, or to
    // a DDR buffer which will be passed as a runtime argument (block
    // argument). Try to find the root block argument, either directly or
    // through subviews/casts.
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
      auto i32ty = builder.getIntegerType(32);
      // arg_plus is an SSA operand. For the dynamic-offset path it is the
      // runtime offset (in bytes) added to the static subview base; for the
      // static path it is simply an arith.constant byte offset.
      Value argPlus;
      if (Value dynOff = bd_op.getDynOffset()) {
        // dynOff is i64 in element-width units; trunc to i32 then convert to
        // bytes via mul by element-width-in-bytes.
        Value dynOff32 =
            arith::TruncIOp::create(builder, bd_op.getLoc(), i32ty, dynOff);
        int32_t elemBytes = bd_op.getBufferElementTypeWidthInBytes();
        Value dynBytes = dynOff32;
        if (elemBytes != 1) {
          Value mulFactor = arith::ConstantIntOp::create(builder, bd_op.getLoc(),
                                                         i32ty, elemBytes);
          dynBytes = arith::MulIOp::create(builder, bd_op.getLoc(), dynOff32,
                                           mulFactor);
        }
        // Add the static subview base offset on top of the runtime offset.
        if (offset != 0) {
          Value base = arith::ConstantIntOp::create(
              builder, bd_op.getLoc(), i32ty, static_cast<int32_t>(offset));
          dynBytes =
              arith::AddIOp::create(builder, bd_op.getLoc(), dynBytes, base);
        }
        argPlus = dynBytes;
      } else {
        offset += bd_op.getOffsetInBytes();
        argPlus = arith::ConstantIntOp::create(builder, bd_op.getLoc(), i32ty,
                                               static_cast<int32_t>(offset));
      }
      NpuAddressPatchOp::create(builder, bd_op.getLoc(),
                                /*addr*/ register_addr,
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
        NpuMaskWrite32Op::create(builder, bd_op.getLoc(), register_addr,
                                 (buf_addr / 4) << 14, 0x0fffc000, nullptr,
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
        NpuMaskWrite32Op::create(builder, bd_op.getLoc(), register_addr,
                                 buf_addr / 4, 0x0007FFFF, nullptr, nullptr,
                                 nullptr);
      } else {
        NpuWrite32Op::create(builder, bd_op.getLoc(), register_addr, buf_addr,
                             nullptr, nullptr, nullptr);
      }
    } else {
      return bd_op->emitOpError(
          "Buffer argument must be a constant aie.buffer, a runtime sequence "
          "input argument, or a (chain of) subview(s) or cast(s) of a block "
          "argument with constant offsets and strides equal to one.");
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

  // Dynamic-operand path: emit NpuWriteBdOp with static placeholder values
  // for dynamic fields (the subsequent AIEDmaToNpu pass converts this to a
  // blockwrite), then selective NpuWrite32Op overrides for BD words that
  // contain dynamic content. This produces the same blockwrite + write32
  // TXN format as the npu_dma_memcpy_nd path.
  LogicalResult
  rewriteSingleBDDynamic(OpBuilder &builder, Block &block, AIE::TileOp &tile,
                         std::optional<xilinx::AIE::PacketInfoAttr> outerPacket,
                         AIE::DMABDOp bd_op) {
    const auto &target_model = AIE::getTargetModel(bd_op);
    if (!target_model.isShimNOCTile(tile.getCol(), tile.getRow())) {
      return bd_op->emitOpError(
          "dynamic operands on aie.dma_bd are only supported on shim NOC "
          "tiles.");
    }
    if (bd_op.getPadDimensions().has_value()) {
      return bd_op->emitOpError(
          "pad_dimensions is incompatible with dynamic dma_bd operands.");
    }

    auto loc = bd_op.getLoc();
    auto i32ty = builder.getIntegerType(32);
    auto zero = IntegerAttr::get(i32ty, 0);
    auto cst = [&](int64_t v) -> Value {
      return arith::ConstantOp::create(builder, loc,
                                       IntegerAttr::get(i32ty, v));
    };

    auto buffer_type = llvm::cast<BaseMemRefType>(bd_op.getBuffer().getType());
    uint64_t elemWidth = buffer_type.getElementType().getIntOrFloatBitWidth();
    uint32_t addrGran = target_model.getAddressGenGranularity();

    // Build mixedSizesRev / mixedStridesRev (4 elements, innermost-first).
    SmallVector<OpFoldResult, 4> mixedSizesRev(4, builder.getI64IntegerAttr(1));
    SmallVector<OpFoldResult, 4> mixedStridesRev(4,
                                                 builder.getI64IntegerAttr(0));
    ValueRange dynSizes = bd_op.getDynSizes();
    ValueRange dynStrides = bd_op.getDynStrides();
    if (!dynSizes.empty()) {
      unsigned n = dynSizes.size();
      for (unsigned i = 0; i < n; ++i) {
        mixedSizesRev[i] = dynSizes[n - 1 - i];
        mixedStridesRev[i] = dynStrides[n - 1 - i];
      }
    } else if (auto dims = bd_op.getDimensions(); dims && !dims->empty()) {
      unsigned n = dims->size();
      for (unsigned i = 0; i < n; ++i) {
        mixedSizesRev[i] =
            builder.getI64IntegerAttr((*dims)[n - 1 - i].getSize());
        mixedStridesRev[i] =
            builder.getI64IntegerAttr((*dims)[n - 1 - i].getStride());
      }
    }

    // Compute dynamic hardware BD encoding via shared utility.
    HwBdEncoding hw =
        emitDynamicHwBdEncoding(builder, loc, target_model, buffer_type,
                                mixedSizesRev, mixedStridesRev);

    // bufLen: prefer dyn_len > hw.bufLen (from dyn_sizes) > static len.
    Value bufLen;
    if (Value dynLen = bd_op.getDynLen()) {
      Value dynLen32 = arith::TruncIOp::create(builder, loc, i32ty, dynLen);
      bufLen = dynLen32;
      if (elemWidth != addrGran) {
        Value scaled =
            arith::MulIOp::create(builder, loc, dynLen32, cst(elemWidth));
        bufLen = arith::DivUIOp::create(builder, loc, scaled, cst(addrGran));
      }
    } else if (!dynSizes.empty()) {
      bufLen = hw.bufLen;
    } else if (bd_op.getLen().has_value()) {
      bufLen = cst(static_cast<int64_t>(bd_op.getLenInBytes() * 8 / addrGran));
    } else {
      uint64_t elemBits = buffer_type.getElementTypeBitWidth();
      int64_t totalUnits =
          static_cast<int64_t>(buffer_type.getNumElements() * elemBits) /
          addrGran;
      bufLen = cst(totalUnits);
    }

    // --- Determine which fields are dynamic vs static ---
    auto isConst = [](OpFoldResult ofr) {
      return getConstantIntValue(ofr).has_value();
    };
    auto getConstOr0 = [](OpFoldResult ofr) -> int64_t {
      if (auto v = getConstantIntValue(ofr))
        return *v;
      return 0;
    };

    bool d0SizeDyn = !isConst(mixedSizesRev[0]);
    bool d1SizeDyn = !isConst(mixedSizesRev[1]);
    bool d2SizeDyn = !isConst(mixedSizesRev[2]);
    bool d3SizeDyn = !isConst(mixedSizesRev[3]);
    bool d0StrideDyn = !isConst(mixedStridesRev[0]);
    bool d1StrideDyn = !isConst(mixedStridesRev[1]);
    bool d2StrideDyn = !isConst(mixedStridesRev[2]);
    bool d3StrideDyn = !isConst(mixedStridesRev[3]);

    // --- Compute static placeholder values for NpuWriteBdOp ---
    // For constant fields use actual hw value; for dynamic fields use 0.
    auto computeStaticHwD0Size = [&]() -> int64_t {
      if (d0SizeDyn)
        return 0;
      return getConstOr0(mixedSizesRev[0]) * static_cast<int64_t>(elemWidth) /
             static_cast<int64_t>(addrGran);
    };
    auto computeStaticHwD0Stride = [&]() -> int64_t {
      if (d0StrideDyn)
        return 0;
      if (elemWidth < addrGran || elemWidth > addrGran)
        return 0;
      return getConstOr0(mixedStridesRev[0]) - 1;
    };
    auto computeStaticHwD1Size = [&]() -> int64_t {
      return d1SizeDyn ? 0 : getConstOr0(mixedSizesRev[1]);
    };
    auto computeStaticHwD1Stride = [&]() -> int64_t {
      if (d1StrideDyn || d1SizeDyn)
        return 0;
      int64_t s = getConstOr0(mixedStridesRev[1]);
      int64_t sz = getConstOr0(mixedSizesRev[1]);
      if (sz <= 1)
        return 0;
      return s * static_cast<int64_t>(elemWidth) /
                 static_cast<int64_t>(addrGran) -
             1;
    };
    auto computeStaticHwD2Stride = [&]() -> int64_t {
      if (d2StrideDyn || d2SizeDyn)
        return 0;
      int64_t s = getConstOr0(mixedStridesRev[2]);
      int64_t sz = getConstOr0(mixedSizesRev[2]);
      if (sz <= 1)
        return 0;
      return s * static_cast<int64_t>(elemWidth) /
                 static_cast<int64_t>(addrGran) -
             1;
    };
    auto computeStaticIterSize = [&]() -> int64_t {
      if (d3SizeDyn)
        return 0;
      int64_t s3 = getConstOr0(mixedSizesRev[3]);
      if (s3 <= 1)
        return 0;
      if (!d3StrideDyn && getConstOr0(mixedStridesRev[3]) <= 0)
        return 0;
      return s3 - 1;
    };
    auto computeStaticIterStride = [&]() -> int64_t {
      if (d3StrideDyn || d3SizeDyn)
        return 0;
      int64_t s3 = getConstOr0(mixedSizesRev[3]);
      int64_t st3 = getConstOr0(mixedStridesRev[3]);
      if (s3 <= 1 || st3 <= 0)
        return 0;
      return st3 * static_cast<int64_t>(elemWidth) /
                 static_cast<int64_t>(addrGran) -
             1;
    };

    int64_t staticBufLen = 0;
    if (!d0SizeDyn && !d1SizeDyn && !d2SizeDyn) {
      staticBufLen = computeStaticHwD0Size() * getConstOr0(mixedSizesRev[1]) *
                     getConstOr0(mixedSizesRev[2]);
    }

    // --- Packet info ---
    int32_t enable_packet = 0, packet_id = 0, packet_type = 0,
            out_of_order_id = 0;
    auto info = bd_op.getPacket().value_or(outerPacket.value_or(nullptr));
    if (info) {
      enable_packet = 1;
      packet_type = info.getPktType();
      packet_id = info.getPktId();
    }

    // --- Lock info ---
    int32_t lock_rel_val = 0, lock_rel_id = 0;
    int32_t lock_acq_enable = 0, lock_acq_val = 0, lock_acq_id = 0;
    auto lock_ops = getOptionalLockOpsForBlock(block);
    if (lock_ops) {
      auto [acquire_op, release_op] = *lock_ops;
      AIE::LockOp acq_lock = acquire_op.getLockOp();
      AIE::LockOp rel_lock = release_op.getLockOp();
      if (acq_lock.getLockID().has_value()) {
        lock_acq_id = acq_lock.getLockID().value();
        lock_acq_val = acquire_op.getLockValue();
        if (acquire_op.acquireGE())
          lock_acq_val = -lock_acq_val;
        lock_acq_enable = 1;
      }
      if (rel_lock.getLockID().has_value()) {
        lock_rel_id = rel_lock.getLockID().value();
        lock_rel_val = release_op.getLockValue();
      }
    }

    // --- Next BD ---
    uint32_t use_next_bd = 0, next_bd_id = 0;
    if (bd_op.getNextBdId().has_value()) {
      next_bd_id = bd_op.getNextBdId().value();
      use_next_bd = 1;
    }

    uint32_t bd_id = bd_op.getBdId().value();

    // --- Emit NpuWriteBdOp with static/placeholder values ---
    // The subsequent AIEDmaToNpu pass converts this to a blockwrite.
    NpuWriteBdOp::create(
        builder, loc,
        /*column=*/IntegerAttr::get(i32ty, tile.getCol()),
        /*bd_id=*/IntegerAttr::get(i32ty, bd_id),
        /*buffer_length=*/IntegerAttr::get(i32ty, staticBufLen),
        /*buffer_offset=*/zero,
        /*enable_packet=*/IntegerAttr::get(i32ty, enable_packet),
        /*out_of_order_id=*/IntegerAttr::get(i32ty, out_of_order_id),
        /*packet_id=*/IntegerAttr::get(i32ty, packet_id),
        /*packet_type=*/IntegerAttr::get(i32ty, packet_type),
        /*d0_size=*/IntegerAttr::get(i32ty, computeStaticHwD0Size()),
        /*d0_stride=*/IntegerAttr::get(i32ty, computeStaticHwD0Stride()),
        /*d1_size=*/IntegerAttr::get(i32ty, computeStaticHwD1Size()),
        /*d1_stride=*/IntegerAttr::get(i32ty, computeStaticHwD1Stride()),
        /*d2_size=*/zero,
        /*d2_stride=*/IntegerAttr::get(i32ty, computeStaticHwD2Stride()),
        /*iteration_current=*/zero,
        /*iteration_size=*/IntegerAttr::get(i32ty, computeStaticIterSize()),
        /*iteration_stride=*/
        IntegerAttr::get(i32ty, computeStaticIterStride()),
        /*next_bd=*/IntegerAttr::get(i32ty, next_bd_id),
        /*row=*/IntegerAttr::get(i32ty, tile.getRow()),
        /*use_next_bd=*/IntegerAttr::get(i32ty, use_next_bd),
        /*valid_bd=*/IntegerAttr::get(i32ty, 1),
        /*lock_rel_val=*/IntegerAttr::get(i32ty, lock_rel_val),
        /*lock_rel_id=*/IntegerAttr::get(i32ty, lock_rel_id),
        /*lock_acq_enable=*/IntegerAttr::get(i32ty, lock_acq_enable),
        /*lock_acq_val=*/IntegerAttr::get(i32ty, lock_acq_val),
        /*lock_acq_id=*/IntegerAttr::get(i32ty, lock_acq_id),
        /*d0_zero_before=*/zero, /*d1_zero_before=*/zero,
        /*d2_zero_before=*/zero, /*d0_zero_after=*/zero,
        /*d1_zero_after=*/zero, /*d2_zero_after=*/zero,
        /*burst_length=*/IntegerAttr::get(i32ty, bd_op.getBurstLength()));

    // --- Emit NpuWrite32Op overrides only for dynamic BD words ---
    uint64_t bdAddr =
        target_model.getDmaBdAddress(tile.getCol(), tile.getRow(), bd_id);

    uint32_t bdAddrU32 = static_cast<uint32_t>(bdAddr);
    auto emitDynBdWord = [&](uint32_t wordIdx, Value wordValue) {
      uint32_t wordAddr = bdAddrU32 + wordIdx * 4;
      NpuWrite32Op::create(builder, loc,
                           /*address=*/static_cast<uint32_t>(0),
                           /*value=*/static_cast<uint32_t>(0),
                           /*buffer=*/FlatSymbolRefAttr{},
                           /*column=*/IntegerAttr{},
                           /*row=*/IntegerAttr{},
                           /*dyn_address=*/cst(wordAddr),
                           /*dyn_value=*/wordValue,
                           /*bd_group=*/bdAddrU32);
    };

    // word[0]: buffer_length — dynamic if any of d0/d1/d2 sizes are dynamic
    if (d0SizeDyn || d1SizeDyn || d2SizeDyn) {
      emitDynBdWord(0, bufLen);
    }

    // word[3]: d0_size, d0_stride
    if (d0SizeDyn || d0StrideDyn) {
      emitDynBdWord(3, buildBdWord(builder, loc,
                                   {{hw.d0Size, 0x3FFu, 20u},
                                    {hw.d0Stride, 0xFFFFFu, 0u}}));
    }

    // word[4]: burst_length (static), d1_size, d1_stride
    if (d1SizeDyn || d1StrideDyn) {
      uint32_t burstEnc =
          AIE::getShimBurstLengthEncoding(target_model, bd_op.getBurstLength());
      Value burstVal = cst(static_cast<int64_t>((burstEnc & 0x3u) << 30));
      Value sizeStride =
          buildBdWord(builder, loc,
                      {{hw.d1Size, 0x3FFu, 20u}, {hw.d1Stride, 0xFFFFFu, 0u}});
      emitDynBdWord(4,
                    arith::OrIOp::create(builder, loc, burstVal, sizeStride));
    }

    // word[5]: AXCache (static), d2_stride
    if (d2StrideDyn || d2SizeDyn) {
      Value axcache = cst((2u & 0xfu) << 24);
      Value strMasked =
          buildBdWord(builder, loc, {{hw.d2Stride, 0xFFFFFu, 0u}});
      emitDynBdWord(5, arith::OrIOp::create(builder, loc, axcache, strMasked));
    }

    // word[6]: iteration_size, iteration_stride
    if (d3SizeDyn || d3StrideDyn) {
      emitDynBdWord(6, buildBdWord(builder, loc,
                                   {{hw.iterSize, 0x3Fu, 20u},
                                    {hw.iterStride, 0xFFFFFu, 0u}}));
    }

    return setAddressForSingleBD(builder, bd_op, tile);
  }

  LogicalResult
  rewriteSingleBD(OpBuilder &builder, Block &block, AIE::TileOp &tile,
                  AIE::DMAChannelDir channelDir,
                  std::optional<xilinx::AIE::PacketInfoAttr> packet) {
    AIE::DMABDOp bd_op = getBdForBlock(block);

    // Dispatch to the dynamic-operand path if any SSA operand override is
    // present. Otherwise fall through to the existing static blockwrite path.
    if (bd_op.getDynOffset() || bd_op.getDynLen() ||
        !bd_op.getDynSizes().empty()) {
      return rewriteSingleBDDynamic(builder, block, tile, packet, bd_op);
    }

    const auto &target_model = AIE::getTargetModel(bd_op);
    auto buffer_type = llvm::cast<BaseMemRefType>(bd_op.getBuffer().getType());
    uint32_t addr_granularity = target_model.getAddressGenGranularity();

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
    // Process strides/wraps
    std::optional<llvm::ArrayRef<AIE::BDDimLayoutAttr>> dims =
        bd_op.getDimensions();
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

    auto enable_packet = 0;
    auto out_of_order_id = 0;
    auto packet_id = 0;
    auto packet_type = 0;
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
    // find next BD ID, if any
    uint32_t use_next_bd = 0;
    uint32_t next_bd_id = 0;
    if (bd_op.getNextBdId().has_value()) {
      next_bd_id = bd_op.getNextBdId().value();
      use_next_bd = 1;
    }

    // enable_packet
    // auto info = bd_op.getPacket() ? bd_op.getPacket() : packet;
    auto info = bd_op.getPacket().value_or(packet.value_or(nullptr));
    if (info) {
      enable_packet = 1;
      packet_type = info.getPktType();
      packet_id = info.getPktId();
    }

    // Extract lock information if present
    int32_t lock_rel_val = 0;
    int32_t lock_rel_id = 0;
    int32_t lock_acq_enable = 0;
    int32_t lock_acq_val = 0;
    int32_t lock_acq_id = 0;

    auto lock_ops = getOptionalLockOpsForBlock(block);
    if (lock_ops) {
      auto [acquire_op, release_op] = *lock_ops;

      // Get lock IDs from the lock operations
      AIE::LockOp acq_lock = acquire_op.getLockOp();
      AIE::LockOp rel_lock = release_op.getLockOp();

      if (acq_lock.getLockID().has_value()) {
        lock_acq_id = acq_lock.getLockID().value();
        lock_acq_val = acquire_op.getLockValue();
        // For AcquireGreaterEqual, negate the value to signal the hardware
        // to use >= comparison instead of == comparison.
        if (acquire_op.acquireGE())
          lock_acq_val = -lock_acq_val;
        lock_acq_enable = 1;
      }

      if (rel_lock.getLockID().has_value()) {
        lock_rel_id = rel_lock.getLockID().value();
        lock_rel_val = release_op.getLockValue();
      }

      // For memtile, add lock offset using getLockLocalBaseIndex.
      // This matches AIERT.cpp implementation.
      if (target_model.isMemTile(tile.getCol(), tile.getRow())) {
        auto lockOffset = target_model.getLockLocalBaseIndex(
            tile.getCol(), tile.getRow(), acq_lock.colIndex(),
            acq_lock.rowIndex());
        if (lockOffset && acq_lock.getLockID().has_value())
          lock_acq_id += lockOffset.value();
        if (lockOffset && rel_lock.getLockID().has_value())
          lock_rel_id += lockOffset.value();
      }
    }

    NpuWriteBdOp::create(
        builder, bd_op.getLoc(), tile.getCol(), bd_id, len_addr_granularity,
        offset,
        /*enable_packet=*/enable_packet,
        /*out_of_order_id=*/out_of_order_id,
        /*packet_id=*/packet_id,
        /*packet_type=*/packet_type,
        /*d0_size=*/d0size, /*d0_stride=*/d0stride,
        /*d1_size=*/d1size, /*d1_stride=*/d1stride,
        /*d2_size=*/d2size, /*d2_stride=*/d2stride,
        /*iteration_current=*/0, /*iteration_size=*/iteration_size,
        /*iteration_stride=*/iteration_stride,
        /*next_bd=*/next_bd_id,
        /*row=*/tile.getRow(),
        /*use_next_bd=*/use_next_bd,
        /*valid_bd=*/1,
        /*lock_rel_val=*/lock_rel_val, /*lock_rel_id=*/lock_rel_id,
        /*lock_acq_enable=*/lock_acq_enable,
        /*lock_acq_val=*/lock_acq_val, /*lock_acq_id=*/lock_acq_id,
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

  // Hoist non-BD/lock/end/next_bd ops (e.g. arith ops computing dynamic
  // operand values) from inside BD blocks to just before the task op.
  // This ensures the SSA values they define are available after the task
  // op's region is erased during lowering.
  void hoistHelperOpsFromBdBlocks(DMAConfigureTaskOp op) {
    for (Block &block : op.getBody()) {
      if (shouldSkipBlock(block))
        continue;
      SmallVector<Operation *> toHoist;
      for (Operation &inner : block) {
        if (!isa<AIE::DMABDOp, AIE::UseLockOp, AIE::NextBDOp, AIE::EndOp>(
                &inner)) {
          toHoist.push_back(&inner);
        }
      }
      for (Operation *hoistOp : toHoist) {
        hoistOp->moveBefore(op);
      }
    }
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

    // Verify each BD block first; subsequent functions rely on them being
    // well-formed
    for (auto it = body.begin(); it != body.end(); ++it) {
      if (shouldSkipBlock(*it)) {
        continue;
      }
      if (failed(verifyNoUnsupportedOpsInBlock(*it))) {
        return failure();
      }
      if (failed(verifyBdInBlock(*it))) {
        return failure();
      }
      if (failed(verifyOptionalLocksInBlock(*it))) {
        return failure();
      }
    }

    // Move helper ops (arith, etc.) that compute dynamic BD operands out of
    // the BD block so they survive the task op's erasure.
    hoistHelperOpsFromBdBlocks(op);

    // Hoist next_bd operations into next_bd_id attribute of the dma_bd
    if (failed(hoistNextBdOpsIntoAttrs(op))) {
      return failure();
    }

    auto channelDir = op.getDirection();
    auto packet = op.getPacket();

    // Lower all BDs
    for (auto it = body.begin(); it != body.end(); ++it) {
      Block &block = *it;
      if (shouldSkipBlock(block)) {
        continue;
      }
      if (failed(rewriteSingleBD(builder, block, tile, channelDir, packet))) {
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
      return;
    }

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
