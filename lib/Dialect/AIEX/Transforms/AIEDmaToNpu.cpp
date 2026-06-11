//===- AIEDmaToNpu.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>
#include <cstdint>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEDMATONPU
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct Write32SymToAddr : OpConversionPattern<NpuWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  Write32SymToAddr(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getBuffer())
      return failure();

    std::optional<uint32_t> address = op.getAbsoluteAddress();
    if (!address.has_value()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<NpuWrite32Op>(op, *address, op.getValue(),
                                              nullptr, nullptr, nullptr);
    return success();
  }
};

struct BlockWriteSymToAddr : OpConversionPattern<NpuBlockWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  BlockWriteSymToAddr(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuBlockWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getBuffer())
      return failure();

    std::optional<uint32_t> address = op.getAbsoluteAddress();
    if (!address.has_value()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<NpuBlockWriteOp>(op, *address, op.getData(),
                                                 nullptr, nullptr, nullptr);
    return success();
  }
};

struct MaskWrite32SymToAddr : OpConversionPattern<NpuMaskWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  MaskWrite32SymToAddr(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuMaskWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getBuffer())
      return failure();

    std::optional<uint32_t> absoluteAddress = op.getAbsoluteAddress();
    if (!absoluteAddress.has_value()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<NpuMaskWrite32Op>(op, *absoluteAddress,
                                                  op.getValue(), op.getMask(),
                                                  nullptr, nullptr, nullptr);
    return success();
  }
};

struct RtpToWrite32Pattern : OpConversionPattern<NpuWriteRTPOp> {
  using OpConversionPattern::OpConversionPattern;

  RtpToWrite32Pattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuWriteRTPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto device = op->getParentOfType<AIE::DeviceOp>();

    auto buffer = device.lookupSymbol<AIE::BufferOp>(op.getBuffer());
    if (!buffer) {
      op->emitError("buffer '" + op.getBuffer() + "' not found in device");
      return failure();
    }

    if (!buffer.getAddress()) {
      op->emitError("buffer must have address assigned");
      return failure();
    }
    AIE::TileOp tile = buffer.getTileOp();

    uint32_t idx = op.getIndex() * sizeof(uint32_t);
    uint32_t address = buffer.getAddress().value() + idx;

    if (op.hasDynamicValue()) {
      // Dynamic RTP write: compute absolute address, pass value as SSA
      const AIE::AIETargetModel &tm = device.getTargetModel();
      uint32_t colShift = tm.getColumnShift();
      uint32_t rowShift = tm.getRowShift();
      uint32_t absAddr = (static_cast<uint32_t>(tile.getCol()) << colShift) |
                         (static_cast<uint32_t>(tile.getRow()) << rowShift) |
                         (address & 0xfffff);

      auto i32Type = rewriter.getI32Type();
      auto addrConst = arith::ConstantOp::create(
          rewriter, op->getLoc(), rewriter.getIntegerAttr(i32Type, absAddr));

      NpuWrite32Op::create(rewriter, op->getLoc(),
                           /*address=*/0u, /*value=*/0u,
                           /*buffer=*/nullptr, /*column=*/nullptr,
                           /*row=*/nullptr,
                           /*dyn_address=*/addrConst.getResult(),
                           /*dyn_value=*/adaptor.getDynValue(),
                           /*bd_group=*/IntegerAttr{});
    } else {
      // Static path
      NpuWrite32Op::create(rewriter, op->getLoc(), address,
                           static_cast<uint32_t>(*op.getValue()), nullptr,
                           rewriter.getI32IntegerAttr(tile.getCol()),
                           rewriter.getI32IntegerAttr(tile.getRow()));
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct PushQueuetoWrite32Pattern : OpConversionPattern<NpuPushQueueOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  PushQueuetoWrite32Pattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuPushQueueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const auto &tm = AIE::getTargetModel(op);
    uint32_t ctrl_offset = tm.getDmaControlAddress(
        op.getColumn(), op.getRow(), op.getChannel(), op.getDirection());

    // control packet for issuing token
    if (op.getIssueToken()) {
      // set the task-complete-token controller ID field in the dma control
      // register
      AIE::TileOp shimTile = AIE::TileOp::getOrCreate(
          rewriter, op->getParentOfType<AIE::DeviceOp>(), op.getColumn(),
          op.getRow());
      if (shimTile->hasAttr("controller_id")) {
        AIE::PacketInfoAttr controller_id_attr =
            shimTile->getAttrOfType<AIE::PacketInfoAttr>("controller_id");
        uint32_t data = controller_id_attr.getPktId() << 8;
        uint32_t mask = 0x00001F00;
        NpuMaskWrite32Op::create(rewriter, op->getLoc(), ctrl_offset, data,
                                 mask, nullptr, nullptr, nullptr);
      }
    }

    // the offset of the task queue register in the tile
    uint32_t queue_offset = ctrl_offset + 0x4;

    // the value to write
    uint32_t bd_id = op.getBdId();
    uint32_t repeat_cnt = op.getRepeatCount();
    uint32_t cmd = 0;
    cmd |= bd_id & 0xF;
    cmd |= (repeat_cnt & 0xFF) << 16;
    if (op.getIssueToken())
      cmd |= 0x80000000;

    NpuWrite32Op::create(rewriter, op->getLoc(), queue_offset, cmd, nullptr,
                         nullptr, nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

struct DmaToNpuPattern : OpConversionPattern<NpuDmaMemcpyNdOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  DmaToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuDmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &targetModel = AIE::getTargetModel(op);
    BaseMemRefType bufferType = op.getMemref().getType();
    auto *ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    auto zero = IntegerAttr::get(i32ty, 0);
    auto memref = adaptor.getMemref();

    auto dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return failure();

    auto infoOp = AIE::ShimDMAAllocationOp::getForSymbol(
        dev, op.getMetadata().getRootReference());
    if (!infoOp) {
      return op->emitOpError("couldn't find shim_dma_allocation op.");
    }

    AIE::TileOp shimTile = infoOp.getTileOp();
    if (!shimTile) {
      return op->emitOpError(
          "shim_dma_allocation op must reference a valid TileOp.");
    }

    auto channelDir = infoOp.getChannelDir();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    int tileCol = shimTile.getCol();
    int tileRow = shimTile.getRow();

    // Check whether all sizes and strides are compile-time constants.
    bool allSizesConstant =
        llvm::all_of(op.getMixedSizes(), [](OpFoldResult s) {
          return getConstantIntValue(s).has_value();
        });
    bool allStridesConstant =
        llvm::all_of(op.getMixedStrides(), [](OpFoldResult s) {
          return getConstantIntValue(s).has_value();
        });

    if (allSizesConstant && allStridesConstant) {
      // =====================================================================
      // STATIC CODE PATH -- all sizes/strides are constants
      // =====================================================================

      // initialize fields to zero
      auto column = zero;
      auto bd_id = zero;
      auto buffer_length = zero;
      auto buffer_offset = zero;
      auto enable_packet = zero;
      auto out_of_order_id = zero;
      auto packet_id = zero;
      auto packet_type = zero;
      auto d0_size = zero;
      auto d0_stride = zero;
      auto d1_size = zero;
      auto d1_stride = zero;
      auto d2_size = zero;
      auto d2_stride = zero;
      auto iteration_current = zero;
      auto iteration_size = zero;
      auto iteration_stride = zero;
      auto next_bd = zero;
      auto row = zero;
      auto use_next_bd = zero;
      auto valid_bd = zero;
      auto lock_rel_val = zero;
      auto lock_rel_id = zero;
      auto lock_acq_enable = zero;
      auto lock_acq_val = zero;
      auto lock_acq_id = zero;
      auto d0_zero_before = zero;
      auto d1_zero_before = zero;
      auto d2_zero_before = zero;
      auto d0_zero_after = zero;
      auto d1_zero_after = zero;
      auto d2_zero_after = zero;
      auto burst_length = zero;

      auto issue_token = BoolAttr::get(ctx, false);
      auto repeat_count = zero;
      llvm::SmallVector<int64_t, 4> inputSizes = llvm::map_to_vector(
          llvm::reverse(op.getMixedSizes()),
          [](OpFoldResult s) { return getConstantIntValue(s).value(); });
      llvm::SmallVector<int64_t, 4> inputStrides = llvm::map_to_vector(
          llvm::reverse(op.getMixedStrides()),
          [](OpFoldResult s) { return getConstantIntValue(s).value(); });
      llvm::SmallVector<int64_t, 4> sizes(4);
      llvm::SmallVector<int64_t, 4> strides(4);
      getHardwareStridesWraps(targetModel, op, bufferType, inputSizes,
                              inputStrides, sizes, strides);
      int64_t offset = op.getOffsetInBytes();

      // column
      column = IntegerAttr::get(i32ty, tileCol);

      // row
      row = IntegerAttr::get(i32ty, tileRow);

      // A contiguous row-major ND access on a shim NOC tile is lowered to
      // linear mode (d0_size=d1_size=0) just like an already-canonical linear
      // transfer. This allows naturally-expressed multidimensional transfers
      // (e.g., a 2D image as [height, width]) without hitting the 10-bit ND
      // wrap-size limit.
      bool isLinear = op.isLinearTransferWithoutTransformation() ||
                      (targetModel.isShimNOCTile(tileCol, tileRow) &&
                       isContiguousTransfer(inputSizes, inputStrides));
      if (failed(verifyStridesWraps(op, bufferType, tileCol, tileRow,
                                    inputSizes, inputStrides, sizes, strides,
                                    isLinear))) {
        return failure();
      }

      // arg_idx and offset for block arguments
      AIE::RuntimeSequenceOp seq_op =
          op->getParentOfType<AIE::RuntimeSequenceOp>();
      if (!seq_op) {
        op->emitOpError(
            "NpuDmaMemcpyNdOps must have RuntimeSequenceOp parent at "
            "time of lowering.");
        return failure();
      }

      mlir::Value rootMemref = memref;
      int64_t subviewOffset = 0;

      // Trace through memref.subview and memref.reinterpret_cast chain, if
      // any, to find root block argument
      auto traceResult = traceSubviewToBlockArgument(memref);
      if (!traceResult) {
        return op->emitOpError(
            "memref must be a block argument or subview/cast/reinterpret_cast "
            "of a block argument with static offsets, sizes, and strides");
      }
      rootMemref = traceResult->rootArg;
      subviewOffset = traceResult->offsetInBytes;

      // Find the argument index of the root memref
      Block &entryBB = seq_op.getBody().front();
      int arg_idx = -1;
      for (int i = 0, e = entryBB.getNumArguments(); i < e; i++) {
        if (entryBB.getArgument(i) == rootMemref) {
          arg_idx = i;
          break;
        }
      }
      if (arg_idx < 0)
        return failure();

      offset += subviewOffset;

      // bd_id
      bd_id = IntegerAttr::get(i32ty, op.getId());

      // buffer_length
      uint64_t buffer_length_val = inputSizes[0] * op.getElementTypeBitwidth() /
                                   targetModel.getAddressGenGranularity();
      if (inputSizes.size() > 1) {
        for (size_t i = 1; i < std::min(inputSizes.size(), (size_t)3); i++) {
          buffer_length_val *= inputSizes[i];
        }
      }
      buffer_length = IntegerAttr::get(i32ty, buffer_length_val);

      // buffer_offset - zero because the complete address is set by the
      // patch op
      buffer_offset = IntegerAttr::get(i32ty, 0);

      // enable_packet
      if (auto packetInfo = op.getPacket()) {
        enable_packet = IntegerAttr::get(i32ty, 1);
        packet_type = IntegerAttr::get(i32ty, packetInfo->getPktType());
        packet_id = IntegerAttr::get(i32ty, packetInfo->getPktId());
      }

      // out_of_order_id

      if (!isLinear) {
        // d0_size, d0_stride
        d0_size = IntegerAttr::get(i32ty, sizes[0]);
        d0_stride = IntegerAttr::get(i32ty, strides[0]);

        // d1_size, d1_stride
        d1_size = IntegerAttr::get(i32ty, sizes[1]);
        d1_stride = IntegerAttr::get(i32ty, strides[1]);

        // d2_stride
        d2_stride = IntegerAttr::get(i32ty, strides[2]);

        // d2_size
        if (targetModel.isMemTile(tileCol, 0)) // Need to be any row
          d2_size = IntegerAttr::get(i32ty, sizes[2]);
        else
          d2_size = IntegerAttr::get(i32ty, 0);
      }
      // iteration_current, iteration_size, iteration_stride, repeat_count
      if (inputSizes[3] > 1) {
        if (inputStrides[3] > 0) {
          iteration_size = IntegerAttr::get(i32ty, sizes[3]);
          iteration_stride = IntegerAttr::get(i32ty, strides[3]);
        } else {
          // We allow users to encode the repeat_count as a dimension 3 stride
          // of 0. This must lower to a iteration wrap of 0, so no stride is
          // ever added. We then repeat the BD using the repeat_count in
          // NpuPushQueueOp.
          iteration_size = zero;
          iteration_stride = zero;
        }
      }
      repeat_count = IntegerAttr::get(i32ty, sizes[3]);

      // next_bd

      // use_next_bd

      // valid_bd
      valid_bd = IntegerAttr::get(i32ty, 1);

      // lock_rel_val

      // lock_rel_id

      // lock_acq_enable

      // lock_acq_val

      // lock_acq_id

      // d0_zero_before
      d0_zero_before = IntegerAttr::get(i32ty, op.getD0ZeroBefore());

      // d1_zero_before
      d1_zero_before = IntegerAttr::get(i32ty, op.getD1ZeroBefore());

      // d2_zero_before
      d2_zero_before = IntegerAttr::get(i32ty, op.getD2ZeroBefore());

      // d0_zero_after
      d0_zero_after = IntegerAttr::get(i32ty, op.getD0ZeroAfter());

      // d1_zero_after
      d1_zero_after = IntegerAttr::get(i32ty, op.getD1ZeroAfter());

      // d2_zero_after
      d2_zero_after = IntegerAttr::get(i32ty, op.getD2ZeroAfter());

      // burst_size
      burst_length = IntegerAttr::get(i32ty, op.getBurstLength());

      // Set the issue_token
      issue_token = BoolAttr::get(ctx, op.getIssueToken());
      // Earlier, all S2MM channels were implicitly assumed to issue a token.
      // This logic is kept for now for backward compatibility.
      if (!isMM2S)
        issue_token = BoolAttr::get(ctx, true);

      if (targetModel.isMemTile(tileCol, tileRow) && (!isMM2S) &&
          (op.getD0ZeroBefore() != 0 || op.getD0ZeroAfter() != 0 ||
           op.getD1ZeroBefore() != 0 || op.getD1ZeroAfter() != 0 ||
           op.getD2ZeroBefore() != 0 || op.getD2ZeroAfter() != 0)) {
        op->emitOpError("MemTile supports zero padding only on MM2S direction");
        return failure();
      }

      // write the buffer descriptor to the array
      NpuWriteBdOp::create(
          rewriter, op->getLoc(), column, bd_id, buffer_length, buffer_offset,
          enable_packet, out_of_order_id, packet_id, packet_type, d0_size,
          d0_stride, d1_size, d1_stride, d2_size, d2_stride, iteration_current,
          iteration_size, iteration_stride, next_bd, row, use_next_bd, valid_bd,
          lock_rel_val, lock_rel_id, lock_acq_enable, lock_acq_val, lock_acq_id,
          d0_zero_before, d1_zero_before, d2_zero_before, d0_zero_after,
          d1_zero_after, d2_zero_after, burst_length);

      // compute the location of the address to patch in the bd and emit patch
      // instruction to perform the patch.
      uint64_t addr =
          targetModel.getDmaBdAddress(tileCol, tileRow, op.getId()) +
          targetModel.getDmaBdAddressOffset(tileCol, tileRow);
      NpuAddressPatchOp::create(rewriter, op->getLoc(), addr, arg_idx, offset,
                                /*dyn_arg_plus=*/Value{});

      // push the patched bd onto the dma task queue
      NpuPushQueueOp::create(
          rewriter, op->getLoc(), column, row, infoOp.getChannelDirAttr(),
          infoOp.getChannelIndexAttr(), issue_token, repeat_count, bd_id);

      rewriter.eraseOp(op);
      return success();
    }

    // =====================================================================
    // DYNAMIC CODE PATH -- some sizes/strides are SSA values
    // =====================================================================
    // We cannot use NpuWriteBdOp (which expects all-constant fields).
    // Instead, we compute the BD words using arith ops and emit:
    //   1. npu_blockwrite with a static BD template (all-zero or partial)
    //   2. npu_write32_dynamic for each BD word that depends on SSA values
    //   3. npu_address_patch for the buffer pointer (same as static path)
    //   4. npu_write32_dynamic for queue push if repeat_count is dynamic

    // Currently only ShimNOC tiles are supported for the dynamic path
    if (!targetModel.isShimNOCTile(tileCol, tileRow)) {
      return op->emitOpError(
          "dynamic sizes/strides are only supported for shim tile DMAs");
    }

    Location loc = op->getLoc();

    // --- Common setup: arg_idx, offset, seq_op ---
    AIE::RuntimeSequenceOp seq_op =
        op->getParentOfType<AIE::RuntimeSequenceOp>();
    if (!seq_op) {
      op->emitOpError("NpuDmaMemcpyNdOps must have RuntimeSequenceOp parent at "
                      "time of lowering.");
      return failure();
    }

    mlir::Value rootMemref = memref;
    int64_t subviewOffset = 0;
    auto traceResult = traceSubviewToBlockArgument(memref);
    if (!traceResult) {
      return op->emitOpError(
          "memref must be a block argument or subview/cast/reinterpret_cast "
          "of a block argument with static offsets, sizes, and strides");
    }
    rootMemref = traceResult->rootArg;
    subviewOffset = traceResult->offsetInBytes;

    Block &entryBB = seq_op.getBody().front();
    int arg_idx = -1;
    for (int i = 0, e = entryBB.getNumArguments(); i < e; i++) {
      if (entryBB.getArgument(i) == rootMemref) {
        arg_idx = i;
        break;
      }
    }
    if (arg_idx < 0)
      return failure();

    // Compute the byte offset. In the dynamic path, offsets may be SSA
    // values, so we compute an SSA Value for the offset.
    bool allOffsetsConstant =
        llvm::all_of(op.getMixedOffsets(), [](OpFoldResult s) {
          return getConstantIntValue(s).has_value();
        });
    int64_t staticOffset = subviewOffset;
    Value dynOffset;
    if (allOffsetsConstant && allStridesConstant) {
      staticOffset += op.getOffsetInBytes();
    } else {
      // Compute offset dynamically: sum(offset[i] * stride[i]) * elemBytes
      size_t elBitWidth = cast<MemRefType>(memref.getType())
                              .getElementType()
                              .getIntOrFloatBitWidth();
      size_t elemBytes = elBitWidth / 8;
      auto offsets = op.getMixedOffsets();
      auto strides = op.getMixedStrides();
      auto i64ty = IntegerType::get(ctx, 64);
      Value sum = arith::ConstantOp::create(
          rewriter, loc, IntegerAttr::get(i64ty, subviewOffset));
      for (size_t i = 0; i < offsets.size(); i++) {
        Value off = getAsValue(rewriter, loc, offsets[i], i64ty);
        Value str = getAsValue(rewriter, loc, strides[i], i64ty);
        Value prod = arith::MulIOp::create(rewriter, loc, off, str);
        Value bytes = arith::MulIOp::create(
            rewriter, loc, prod,
            arith::ConstantOp::create(rewriter, loc,
                                      IntegerAttr::get(i64ty, elemBytes)));
        sum = arith::AddIOp::create(rewriter, loc, sum, bytes);
      }
      // Truncate to i32 for the address patch offset
      auto i32ty_ = IntegerType::get(ctx, 32);
      dynOffset = arith::TruncIOp::create(rewriter, loc, i32ty_, sum);
    }

    // --- Retrieve mixed sizes/strides as OpFoldResults (reversed to match
    // the convention: [d0, d1, d2, d3/iter]) ---
    // IMPORTANT: Use adaptor operands (not op operands) because inside
    // applyPartialConversion the original SSA values may be remapped.
    auto buildMixed = [&](ValueRange dynVals, ArrayRef<int64_t> staticVals) {
      SmallVector<OpFoldResult, 4> result;
      unsigned dynIdx = 0;
      for (int64_t sv : staticVals) {
        if (ShapedType::isDynamic(sv))
          result.push_back(dynVals[dynIdx++]);
        else
          result.push_back(rewriter.getI64IntegerAttr(sv));
      }
      return result;
    };
    SmallVector<OpFoldResult, 4> mixedSizesRev(
        llvm::reverse(buildMixed(adaptor.getSizes(), op.getStaticSizes())));
    SmallVector<OpFoldResult, 4> mixedStridesRev(
        llvm::reverse(buildMixed(adaptor.getStrides(), op.getStaticStrides())));

    // --- Compute hardware sizes and strides as SSA Values via shared util ---
    // NOTE: The hardware d0_size field is 10 bits wide (max 1023). The static
    // lowering path applies a linear-mode optimization for contiguous transfers
    // that avoids this limit, but the dynamic path does not. Dynamic d0_size
    // values exceeding 1023 will be silently truncated by the hardware.
    HwBdEncoding hw = emitDynamicHwBdEncoding(
        rewriter, loc, targetModel, bufferType, mixedSizesRev, mixedStridesRev);
    Value hwD0Size = hw.d0Size;
    Value hwD0Stride = hw.d0Stride;
    Value hwD1Size = hw.d1Size;
    Value hwD1Stride = hw.d1Stride;
    Value hwD2Stride = hw.d2Stride;
    // hw.d2Size is not extracted: ShimNOC BDs always set d2_size=0 in the
    // template and use bufLen (word[0]) instead.
    Value hwIterSize = hw.iterSize;
    Value hwIterStride = hw.iterStride;
    Value bufLen = hw.bufLen;
    Value repeatCount = hw.repeatCount;

    // Element width / address-gen granularity are used by the static
    // placeholder lambdas below to mirror the per-target scaling that
    // emitDynamicHwBdEncoding performs SSA-side. Keep them in sync with
    // BdLowering.cpp.
    uint64_t elemWidth = bufferType.getElementType().getIntOrFloatBitWidth();
    uint32_t addrGran = targetModel.getAddressGenGranularity();

    // Helper to create i32 constants (still used for static word values)
    auto cst = [&](int64_t val) -> Value {
      return arith::ConstantOp::create(rewriter, loc,
                                       IntegerAttr::get(i32ty, val));
    };

    // --- Compute BD base address ---
    uint32_t bdId = op.getId();
    uint64_t bdAddr = targetModel.getDmaBdAddress(tileCol, tileRow, bdId);

    // --- Emit NpuWriteBdOp with placeholder values for dynamic fields,
    //     then selective NpuWrite32Op overrides for dynamic BD words ---
    // This produces the same blockwrite TXN format as the static path,
    // with dynamic words patched via write32 overrides that the EmitC
    // fusion can fold back into the blockwrite.

    uint32_t burstEnc =
        getShimBurstLengthEncoding(targetModel, op.getBurstLength());

    // Helper: check if an OpFoldResult is a compile-time constant
    auto isConst = [](OpFoldResult ofr) {
      return getConstantIntValue(ofr).has_value();
    };
    // Returns the compile-time constant value of an OpFoldResult, or 0 as a
    // placeholder for dynamic (SSA) values. The placeholder 0s are written
    // into the NpuWriteBdOp template; dynamic fields are then overridden by
    // selective NpuWrite32Op patches emitted below.
    auto getConstOr0 = [](OpFoldResult ofr) -> int64_t {
      if (auto v = getConstantIntValue(ofr))
        return *v;
      return 0;
    };

    // Determine which input sizes/strides are dynamic
    bool d0SizeDyn = !isConst(mixedSizesRev[0]);
    bool d1SizeDyn = !isConst(mixedSizesRev[1]);
    bool d2SizeDyn = !isConst(mixedSizesRev[2]);
    bool d3SizeDyn = !isConst(mixedSizesRev[3]);
    bool d0StrideDyn = !isConst(mixedStridesRev[0]);
    bool d1StrideDyn = !isConst(mixedStridesRev[1]);
    bool d2StrideDyn = !isConst(mixedStridesRev[2]);
    bool d3StrideDyn = !isConst(mixedStridesRev[3]);

    // Compute static placeholder values for NpuWriteBdOp attributes.
    // For constant fields, use the actual hardware value; for dynamic
    // fields, use 0 as placeholder (will be overridden by write32).

    // Compute static hardware values for constant fields (replicating
    // getHardwareStridesWraps logic for constants only).
    auto computeStaticHwD0Size = [&]() -> int64_t {
      if (d0SizeDyn)
        return 0;
      int64_t s = getConstOr0(mixedSizesRev[0]);
      return s * static_cast<int64_t>(elemWidth) /
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
      int64_t scaled =
          s * static_cast<int64_t>(elemWidth) / static_cast<int64_t>(addrGran);
      return scaled - 1;
    };
    auto computeStaticHwD2Stride = [&]() -> int64_t {
      if (d2StrideDyn || d2SizeDyn)
        return 0;
      int64_t s = getConstOr0(mixedStridesRev[2]);
      int64_t sz = getConstOr0(mixedSizesRev[2]);
      if (sz <= 1)
        return 0;
      int64_t scaled =
          s * static_cast<int64_t>(elemWidth) / static_cast<int64_t>(addrGran);
      return scaled - 1;
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
      int64_t scaled = st3 * static_cast<int64_t>(elemWidth) /
                       static_cast<int64_t>(addrGran);
      return scaled - 1;
    };

    // Compute static buffer_length for constant fields
    int64_t staticBufLen = 0;
    if (!d0SizeDyn && !d1SizeDyn && !d2SizeDyn) {
      staticBufLen = computeStaticHwD0Size() * getConstOr0(mixedSizesRev[1]) *
                     getConstOr0(mixedSizesRev[2]);
    }

    // Build NpuWriteBdOp attrs
    auto column = IntegerAttr::get(i32ty, tileCol);
    auto bd_id = IntegerAttr::get(i32ty, bdId);
    auto buffer_length = IntegerAttr::get(i32ty, staticBufLen);
    auto buffer_offset = IntegerAttr::get(i32ty, 0);
    auto enable_packet = zero;
    auto out_of_order_id = zero;
    auto packet_id = zero;
    auto packet_type = zero;
    if (auto packetInfo = op.getPacket()) {
      enable_packet = IntegerAttr::get(i32ty, 1);
      packet_type = IntegerAttr::get(i32ty, packetInfo->getPktType());
      packet_id = IntegerAttr::get(i32ty, packetInfo->getPktId());
    }
    auto d0_size = IntegerAttr::get(i32ty, computeStaticHwD0Size());
    auto d0_stride = IntegerAttr::get(i32ty, computeStaticHwD0Stride());
    auto d1_size = IntegerAttr::get(i32ty, computeStaticHwD1Size());
    auto d1_stride = IntegerAttr::get(i32ty, computeStaticHwD1Stride());
    auto d2_size = zero;
    auto d2_stride = IntegerAttr::get(i32ty, computeStaticHwD2Stride());
    auto iteration_current = zero;
    auto iteration_size = IntegerAttr::get(i32ty, computeStaticIterSize());
    auto iteration_stride = IntegerAttr::get(i32ty, computeStaticIterStride());
    auto next_bd = zero;
    auto row = IntegerAttr::get(i32ty, tileRow);
    auto use_next_bd = zero;
    auto valid_bd = IntegerAttr::get(i32ty, 1);
    auto lock_rel_val = zero;
    auto lock_rel_id = zero;
    auto lock_acq_enable = zero;
    auto lock_acq_val = zero;
    auto lock_acq_id = zero;
    auto d0_zero_before = IntegerAttr::get(i32ty, op.getD0ZeroBefore());
    auto d1_zero_before = IntegerAttr::get(i32ty, op.getD1ZeroBefore());
    auto d2_zero_before = IntegerAttr::get(i32ty, op.getD2ZeroBefore());
    auto d0_zero_after = IntegerAttr::get(i32ty, op.getD0ZeroAfter());
    auto d1_zero_after = IntegerAttr::get(i32ty, op.getD1ZeroAfter());
    auto d2_zero_after = IntegerAttr::get(i32ty, op.getD2ZeroAfter());
    auto burst_length = IntegerAttr::get(i32ty, op.getBurstLength());

    NpuWriteBdOp::create(
        rewriter, loc, column, bd_id, buffer_length, buffer_offset,
        enable_packet, out_of_order_id, packet_id, packet_type, d0_size,
        d0_stride, d1_size, d1_stride, d2_size, d2_stride, iteration_current,
        iteration_size, iteration_stride, next_bd, row, use_next_bd, valid_bd,
        lock_rel_val, lock_rel_id, lock_acq_enable, lock_acq_val, lock_acq_id,
        d0_zero_before, d1_zero_before, d2_zero_before, d0_zero_after,
        d1_zero_after, d2_zero_after, burst_length);

    // --- Emit NpuWrite32Op overrides for dynamic BD words ---
    // Only emit write32 for words that contain dynamic content.
    uint32_t bdAddrU32 = static_cast<uint32_t>(bdAddr);
    auto emitDynBdWord = [&](uint32_t wordIdx, Value wordValue) {
      uint32_t wordAddr = bdAddrU32 + wordIdx * 4;
      Value addrSSA = cst(wordAddr);
      NpuWrite32Op::create(rewriter, loc,
                           /*address=*/static_cast<uint32_t>(0),
                           /*value=*/static_cast<uint32_t>(0),
                           /*buffer=*/FlatSymbolRefAttr{},
                           /*column=*/IntegerAttr{},
                           /*row=*/IntegerAttr{},
                           /*dyn_address=*/addrSSA,
                           /*dyn_value=*/wordValue,
                           /*bd_group=*/bdAddrU32);
    };

    // word[0]: buffer_length — dynamic if any of d0/d1/d2 sizes are dynamic
    bool word0Dyn = d0SizeDyn || d1SizeDyn || d2SizeDyn;
    if (word0Dyn) {
      emitDynBdWord(0, bufLen);
    }

    // word[3]: d0_size, d0_stride — dynamic if either is dynamic
    bool word3Dyn = d0SizeDyn || d0StrideDyn;
    if (word3Dyn) {
      emitDynBdWord(
          3, buildBdWord(rewriter, loc,
                         {{hwD0Size, 0x3FF, 20}, {hwD0Stride, 0xFFFFF, 0}}));
    }

    // word[4]: burst_length (static), d1_size, d1_stride
    bool word4Dyn = d1SizeDyn || d1StrideDyn;
    if (word4Dyn) {
      Value burstVal = cst((burstEnc & 0x3) << 30);
      Value sizeStride = buildBdWord(
          rewriter, loc, {{hwD1Size, 0x3FF, 20}, {hwD1Stride, 0xFFFFF, 0}});
      emitDynBdWord(4,
                    arith::OrIOp::create(rewriter, loc, burstVal, sizeStride));
    }

    // word[5]: AXCache (static), d2_stride
    bool word5Dyn = d2StrideDyn || d2SizeDyn;
    if (word5Dyn) {
      Value axcache = cst((2u & 0xf) << 24);
      Value strMasked = buildBdWord(rewriter, loc, {{hwD2Stride, 0xFFFFF, 0}});
      emitDynBdWord(5, arith::OrIOp::create(rewriter, loc, axcache, strMasked));
    }

    // word[6]: iteration_size, iteration_stride
    bool word6Dyn = d3SizeDyn || d3StrideDyn;
    if (word6Dyn) {
      emitDynBdWord(
          6, buildBdWord(rewriter, loc,
                         {{hwIterSize, 0x3F, 20}, {hwIterStride, 0xFFFFF, 0}}));
    }

    // word[1] (base_addr) and word[2] (packet ctrl) are always static
    // word[7] (valid_bd) is always static
    // These are fully handled by the NpuWriteBdOp above.

    // --- Address patch ---
    uint64_t patchAddr =
        bdAddr + targetModel.getDmaBdAddressOffset(tileCol, tileRow);
    if (dynOffset) {
      // Dynamic offset: pass 0 as static arg_plus and provide SSA value
      NpuAddressPatchOp::create(rewriter, loc, static_cast<uint32_t>(patchAddr),
                                static_cast<uint32_t>(arg_idx),
                                /*arg_plus=*/static_cast<uint32_t>(0),
                                /*dyn_arg_plus=*/dynOffset);
    } else {
      NpuAddressPatchOp::create(rewriter, loc, static_cast<uint32_t>(patchAddr),
                                static_cast<uint32_t>(arg_idx),
                                static_cast<uint32_t>(staticOffset),
                                /*dyn_arg_plus=*/Value{});
    }

    // If this DMA op has an offset_state_table_idx, emit an
    // update_from_scratchpad to add the runtime offset to the BD address
    // register.
    if (op.getOffsetStateTableIdxAttr()) {
      auto bufType = cast<BaseMemRefType>(op.getMemref().getType());
      if (failed(emitUpdateBdAddressFromOffsetParameter(rewriter, op, bufType,
                                                        patchAddr)))
        return failure();
    }

    // --- Queue push ---
    // Determine issue_token
    bool issueTokenVal = op.getIssueToken();
    if (!isMM2S)
      issueTokenVal = true;

    // Check if repeat_count (sizes[3]) is dynamic
    bool repeatCountDynamic =
        !getConstantIntValue(mixedSizesRev[3]).has_value();

    // Compute the queue push address
    uint32_t ctrlOffset = targetModel.getDmaControlAddress(
        tileCol, tileRow, infoOp.getChannelIndex(), channelDir);

    uint32_t queueOffset = ctrlOffset + 0x4;

    if (repeatCountDynamic) {
      // Dynamic-repeat-count path lowers directly to NpuWrite32Op below,
      // bypassing PushQueuetoWrite32Pattern. Emit the controller_id
      // task-complete-token maskwrite inline here. (In the static `else`
      // branch we create an NpuPushQueueOp; PushQueuetoWrite32Pattern
      // emits the controller_id maskwrite for that case, so emitting it
      // here too would duplicate the op.)
      if (issueTokenVal) {
        if (shimTile->hasAttr("controller_id")) {
          AIE::PacketInfoAttr controllerIdAttr =
              shimTile->getAttrOfType<AIE::PacketInfoAttr>("controller_id");
          uint32_t data = controllerIdAttr.getPktId() << 8;
          uint32_t mask = 0x00001F00;
          NpuMaskWrite32Op::create(rewriter, loc, ctrlOffset, data, mask,
                                   nullptr, nullptr, nullptr);
        }
      }

      // Build queue push command as SSA:
      // cmd = (bd_id & 0xF) | ((repeat_count & 0xFF) << 16) |
      //       (issue_token ? 1<<31 : 0)
      Value bdIdVal = cst(bdId & 0xF);
      Value rcShifted = buildBdWord(rewriter, loc, {{repeatCount, 0xFF, 16}});
      Value cmd = arith::OrIOp::create(rewriter, loc, bdIdVal, rcShifted);
      if (issueTokenVal) {
        Value tokenBit = cst(static_cast<int32_t>(0x80000000u));
        cmd = arith::OrIOp::create(rewriter, loc, cmd, tokenBit);
      }
      Value queueAddrSSA = cst(queueOffset);
      NpuWrite32Op::create(rewriter, loc, rewriter.getUI32IntegerAttr(0),
                           rewriter.getUI32IntegerAttr(0),
                           /*buffer=*/FlatSymbolRefAttr(),
                           /*column=*/IntegerAttr(),
                           /*row=*/IntegerAttr(),
                           /*dyn_address=*/queueAddrSSA, /*dyn_value=*/cmd,
                           /*bd_group=*/IntegerAttr{});
    } else {
      // Static queue push
      auto columnAttr = IntegerAttr::get(i32ty, tileCol);
      auto rowAttr = IntegerAttr::get(i32ty, tileRow);
      auto bdIdAttr = IntegerAttr::get(i32ty, bdId);
      auto issueTokenAttr = BoolAttr::get(ctx, issueTokenVal);
      // repeat_count is constant here. Apply the same -1 conversion
      // as getHardwareStridesWraps (hardware encodes additional repeats).
      int64_t rcVal = getConstantIntValue(mixedSizesRev[3]).value() - 1;
      if (rcVal < 0)
        rcVal = 0;
      auto repeatCountAttr = IntegerAttr::get(i32ty, rcVal);
      NpuPushQueueOp::create(rewriter, loc, columnAttr, rowAttr,
                             infoOp.getChannelDirAttr(),
                             infoOp.getChannelIndexAttr(), issueTokenAttr,
                             repeatCountAttr, bdIdAttr);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert NpuDmaWaitOp into NpuSyncOp by retrieving the necessary
/// information from the ShimDMAAllocationOp referenced through the
/// symbol argument of this op.
struct DmaWaitToSyncPattern : OpConversionPattern<NpuDmaWaitOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  DmaWaitToSyncPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuDmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return op->emitError("couldn't find parent of type DeviceOp");

    AIE::ShimDMAAllocationOp shimDmaAllocOp =
        AIE::ShimDMAAllocationOp::getForSymbol(dev, op.getSymbol());
    if (!shimDmaAllocOp) {
      return op->emitError("couldn't find shim_dma_allocation op");
    }

    AIE::TileOp shimTile = shimDmaAllocOp.getTileOp();
    if (!shimTile) {
      return op->emitError(
          "shim_dma_allocation op must reference a valid TileOp");
    }

    // Create with `column_num == 1` and `row_num == 1` to check for a single
    // column and row.
    (void)rewriter.replaceOpWithNewOp<NpuSyncOp>(
        op, shimTile.getCol(), shimTile.getRow(),
        static_cast<uint32_t>(shimDmaAllocOp.getChannelDir()),
        shimDmaAllocOp.getChannelIndex(), 1, 1);

    return success();
  }
};

struct WriteBdToBlockWritePattern : OpConversionPattern<NpuWriteBdOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  WriteBdToBlockWritePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuWriteBdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    const AIE::AIETargetModel &tm = dev.getTargetModel();
    int col = op.getColumn();
    int row = op.getRow();

    int num_words = 0;
    if (isa<AIE::AIE2TargetModel>(tm)) {
      // Tile DMAs have 6 words, MemTile and Shim have 8 words
      if (tm.isCoreTile(col, row))
        num_words = 6;
      else
        num_words = 8;
    } else {
      llvm_unreachable(
          "Unsupported AIETargetModel in WriteBdToBlockWritePattern");
    }

    std::vector<uint32_t> words(num_words, 0);

    uint32_t bd_id = op.getBdId();
    uint64_t bd_addr = tm.getDmaBdAddress(col, row, bd_id);
    if (tm.isShimNOCTile(col, row)) {
      // DMA_BDX_0
      words[0] = op.getBufferLength();

      // DMA_BDX_1
      words[1] = op.getBufferOffset();

      // DMA_BDX_2
      // En Packet , OoO BD ID , Packet ID , Packet Type
      words[2] |= (op.getEnablePacket() & 0x1) << 30;
      words[2] |= (op.getOutOfOrderId() & 0x3f) << 24;
      words[2] |= (op.getPacketId() & 0x1f) << 19;
      words[2] |= (op.getPacketType() & 0x7) << 16;

      // DMA_BDX_3
      // TODO: Secure Access
      words[3] |= (op.getD0Size() & 0x3ff) << 20;
      words[3] |= op.getD0Stride() & 0xfffff;

      // DMA_BDX_4
      words[4] = (getShimBurstLengthEncoding(tm, op.getBurstLength()) & 0x3)
                 << 30;
      words[4] |= (op.getD1Size() & 0x3ff) << 20;
      words[4] |= op.getD1Stride() & 0xfffff;

      // DMA_BDX_5
      // TODO: SIMID, AXQoS
      words[5] |= (2 & 0xf) << 24; // AXCache = 2 to enable upsizing in NoC
      words[5] |= op.getD2Stride() & 0xfffff;

      // DMA_BDX_6
      words[6] |= (op.getIterationCurrent() & 0x3f) << 26;
      words[6] |= (op.getIterationSize() & 0x3f) << 20;
      words[6] |= op.getIterationStride() & 0xfffff;

      // DMA_BDX_7
      // TODO: TLAST Suppress
      words[7] |= (op.getNextBd() & 0xf) << 27;
      words[7] |= (op.getUseNextBd() & 0x1) << 26;
      words[7] |= (op.getValidBd() & 0x1) << 25;
      words[7] |= (op.getLockRelVal() & 0x7f) << 18;
      words[7] |= (op.getLockRelId() & 0xf) << 13;
      words[7] |= (op.getLockAcqEnable() & 0x1) << 12;
      words[7] |= (op.getLockAcqVal() & 0x7f) << 5;
      words[7] |= op.getLockAcqId() & 0xf;
      if (op.getD0ZeroBefore() || op.getD1ZeroBefore() ||
          op.getD2ZeroBefore() || op.getD0ZeroAfter() || op.getD1ZeroAfter() ||
          op.getD2ZeroAfter()) {
        op->emitError("Zero padding is only available on MemTile");
      }
    } else if (tm.isMemTile(op.getColumn(), op.getRow())) {

      // DMA_BDX_0
      words[0] |= (op.getEnablePacket() & 0x1) << 31;
      words[0] |= (op.getPacketType() & 0x7) << 28;
      words[0] |= (op.getPacketId() & 0x1f) << 23;
      words[0] |= (op.getOutOfOrderId() & 0x3f) << 17;
      words[0] |= op.getBufferLength() & 0x1ffff;

      // DMA_BDX_1
      words[1] |= (op.getD0ZeroBefore() & 0x3F) << 26;
      words[1] |= (op.getNextBd() & 0x3f) << 20;
      words[1] |= (op.getUseNextBd() & 0x1) << 19;
      words[1] |= op.getBufferOffset() & 0x7ffff;

      // DMA_BDX_2
      words[2] |= (op.getD0Size() & 0x3ff) << 17;
      words[2] |= op.getD0Stride() & 0x1ffff;

      // DMA_BDX_3
      // TODO: Secure Access
      words[3] |= (op.getD1ZeroBefore() & 0x1F) << 27;
      words[3] |= (op.getD1Size() & 0x3ff) << 17;
      words[3] |= op.getD1Stride() & 0x1ffff;

      // DMA_BDX_4
      // TODO: D2Size
      words[4] |= (op.getD2ZeroBefore() & 0xF) << 27;
      words[4] |= op.getD2Stride() & 0x1ffff;

      // DMA_BDX_5
      // ToDO: D3Stride
      words[5] |= (op.getD2ZeroAfter() & 0xF) << 28;
      words[5] |= (op.getD1ZeroAfter() & 0x1F) << 23;
      words[5] |= (op.getD0ZeroAfter() & 0x3F) << 17;

      // DMA_BDX_6
      words[6] |= (op.getIterationCurrent() & 0x3f) << 23;
      words[6] |= (op.getIterationSize() & 0x3f) << 17;
      words[6] |= op.getIterationStride() & 0x1ffff;

      // DMA_BDX_7
      words[7] |= (op.getValidBd() & 0x1) << 31;
      words[7] |= (op.getLockRelVal() & 0x7f) << 24;
      words[7] |= (op.getLockRelId() & 0xff) << 16;
      words[7] |= (op.getLockAcqEnable() & 0x1) << 15;
      words[7] |= (op.getLockAcqVal() & 0x7f) << 8;
      words[7] |= op.getLockAcqId() & 0xff;
    } else {
      // AIE2 Tile DMA - 6 words
      // DMA_BDX_0
      // Base_Address [27:14], Buffer_Length [13:0]
      words[0] = ((op.getBufferOffset() / 4) & 0x3fff) << 14;
      words[0] |= op.getBufferLength() & 0x3fff;

      // DMA_BDX_1
      // Enable_Compression [31], Enable_Packet [30], Out_Of_Order_BD_ID
      // [29:24], Packet_ID [23:19], Packet_Type [18:16]
      words[1] = 0; // Enable_Compression
      words[1] |= (op.getEnablePacket() & 0x1) << 30;
      words[1] |= (op.getOutOfOrderId() & 0x3f) << 24;
      words[1] |= (op.getPacketId() & 0x1f) << 19;
      words[1] |= (op.getPacketType() & 0x7) << 16;

      // DMA_BDX_2
      // D1_Stepsize [25:13], D0_Stepsize [12:0]
      words[2] = (op.getD1Stride() & 0x1fff) << 13;
      words[2] |= op.getD0Stride() & 0x1fff;

      // DMA_BDX_3
      // D1_Wrap [28:21], D0_Wrap [20:13], D2_Stepsize [12:0]
      words[3] = (op.getD1Size() & 0xff) << 21;
      words[3] |= (op.getD0Size() & 0xff) << 13;
      words[3] |= op.getD2Stride() & 0x1fff;

      // DMA_BDX_4
      // Iteration_Current [24:19], Iteration_Wrap [18:13], Iteration_Stepsize
      // [12:0]
      words[4] = (op.getIterationCurrent() & 0x3f) << 19;
      words[4] |= (op.getIterationSize() & 0x3f) << 13;
      words[4] |= op.getIterationStride() & 0x1fff;

      // DMA_BDX_5
      // TLAST_Suppress [31], Next_BD [30:27], Use_Next_BD [26], Valid_BD [25],
      // Lock_Rel_Value [24:18], Lock_Rel_ID [16:13], Lock_Acq_Enable [12],
      // Lock_Acq_Value [11:5], Lock_Acq_ID [3:0]
      words[5] = 0; // TLAST_Suppress
      words[5] |= (op.getNextBd() & 0xf) << 27;
      words[5] |= (op.getUseNextBd() & 0x1) << 26;
      words[5] |= (op.getValidBd() & 0x1) << 25;
      words[5] |= (op.getLockRelVal() & 0x7f) << 18;
      words[5] |= (op.getLockRelId() & 0xf) << 13;
      words[5] |= (op.getLockAcqEnable() & 0x1) << 12;
      words[5] |= (op.getLockAcqVal() & 0x7f) << 5;
      words[5] |= op.getLockAcqId() & 0xf;
    }

    memref::GlobalOp global = nullptr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op->getParentOfType<AIE::RuntimeSequenceOp>());
      global = getOrCreateDataMemref(rewriter, dev, op.getLoc(), words);
    }
    auto memref = memref::GetGlobalOp::create(
        rewriter, op.getLoc(), global.getType(), global.getName());

    (void)rewriter.replaceOpWithNewOp<NpuBlockWriteOp>(
        op, rewriter.getUI32IntegerAttr(bd_addr), memref.getResult(), nullptr,
        nullptr, nullptr);
    return success();
  }
};

struct AIEDmaToNpuPass : xilinx::AIEX::impl::AIEDmaToNpuBase<AIEDmaToNpuPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalOp<AIE::BufferOp>();
    target.addLegalOp<AIE::ShimDMAAllocationOp>();
    target.addLegalOp<AIE::TileOp>();

    target.addIllegalOp<NpuDmaMemcpyNdOp>();
    target.addIllegalOp<NpuDmaWaitOp>();
    target.addIllegalOp<NpuPushQueueOp>();
    target.addIllegalOp<NpuWriteRTPOp>();
    target.addIllegalOp<NpuWriteBdOp>();
    target.addDynamicallyLegalOp<NpuWrite32Op>(
        [&](NpuWrite32Op op) { return !op.getBuffer(); });
    target.addDynamicallyLegalOp<NpuBlockWriteOp>(
        [&](NpuBlockWriteOp op) { return !op.getBuffer(); });
    target.addDynamicallyLegalOp<NpuMaskWrite32Op>(
        [&](NpuMaskWrite32Op op) { return !op.getBuffer(); });

    RewritePatternSet patterns(&getContext());
    patterns.insert<BlockWriteSymToAddr>(&getContext());
    patterns.insert<DmaToNpuPattern>(&getContext());
    patterns.insert<DmaWaitToSyncPattern>(&getContext());
    patterns.insert<MaskWrite32SymToAddr>(&getContext());
    patterns.insert<PushQueuetoWrite32Pattern>(&getContext());
    patterns.insert<RtpToWrite32Pattern>(&getContext());
    patterns.insert<Write32SymToAddr>(&getContext());
    patterns.insert<WriteBdToBlockWritePattern>(&getContext());

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPartialConversion(device, target, frozenPatterns))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEDmaToNpuPass() {
  return std::make_unique<AIEDmaToNpuPass>();
}
