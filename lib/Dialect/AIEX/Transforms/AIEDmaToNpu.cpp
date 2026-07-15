//===- AIEDmaToNpu.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    Value addressVal = createConstantI32(rewriter, op->getLoc(), *address);
    rewriter.replaceOpWithNewOp<NpuWrite32Op>(
        op, addressVal, adaptor.getValue(), nullptr, nullptr, nullptr);
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

    Value addressVal =
        createConstantI32(rewriter, op->getLoc(), *absoluteAddress);
    rewriter.replaceOpWithNewOp<NpuMaskWrite32Op>(
        op, addressVal, adaptor.getValue(), adaptor.getMask(), nullptr, nullptr,
        nullptr);
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

    NpuWrite32Op::create(rewriter, op->getLoc(),
                         createConstantI32(rewriter, op->getLoc(), address),
                         adaptor.getValue(), nullptr,
                         rewriter.getI32IntegerAttr(tile.getCol()),
                         rewriter.getI32IntegerAttr(tile.getRow()));

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
        NpuMaskWrite32Op::create(
            rewriter, op->getLoc(),
            createConstantI32(rewriter, op->getLoc(), ctrl_offset),
            createConstantI32(rewriter, op->getLoc(), data),
            createConstantI32(rewriter, op->getLoc(), mask), nullptr, nullptr,
            nullptr);
      }
    }

    // the offset of the task queue register in the tile
    uint32_t queue_offset = ctrl_offset + 0x4;

    // The command word packs bd_id [3:0], repeat_count [23:16], and the
    // issue-token bit [31]. bd_id is always compile-time (the BD-ID allocation
    // pass); repeat_count may be a runtime SSA value (a runtime outer/repeat
    // dimension). When both are constant we fold the whole word to a constant;
    // when repeat_count is runtime we build the word with arith (bd_id and the
    // issue bit fold in as a constant base), so a runtime repeat still lowers
    // to a valid write32 instead of being rejected.
    std::optional<uint32_t> bd_id = getConstantIntOperand(op.getBdId());
    if (!bd_id)
      return op.emitOpError("cannot lower push_queue with non-constant bd_id");
    uint32_t cmdBase = (*bd_id & 0xF) | (op.getIssueToken() ? 0x80000000 : 0);

    Value cmdVal;
    if (std::optional<uint32_t> repeat_cnt =
            getConstantIntOperand(op.getRepeatCount())) {
      cmdVal = createConstantI32(rewriter, op->getLoc(),
                                 cmdBase | ((*repeat_cnt & 0xFF) << 16));
    } else {
      // cmdBase | ((repeat & 0xFF) << 16), as arith over the SSA repeat_count.
      Location loc = op->getLoc();
      Value repeat = op.getRepeatCount();
      Value masked = arith::AndIOp::create(
          rewriter, loc, repeat, createConstantI32(rewriter, loc, 0xFF));
      Value shifted = arith::ShLIOp::create(
          rewriter, loc, masked, createConstantI32(rewriter, loc, 16));
      cmdVal = arith::OrIOp::create(
          rewriter, loc, createConstantI32(rewriter, loc, cmdBase), shifted);
    }

    NpuWrite32Op::create(
        rewriter, op->getLoc(),
        createConstantI32(rewriter, op->getLoc(), queue_offset), cmdVal,
        nullptr, nullptr, nullptr);
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
    // Runtime (SSA) sizes/strides take the dynamic path; a fully-constant
    // descriptor takes the original static path below unchanged (so its output
    // stays byte-identical). The op verifier has already enforced the supported
    // scope for the dynamic case (shim NOC, innermost stride 1, all-or-nothing,
    // no padding, in-range constants).
    bool allSizesConstant =
        llvm::all_of(op.getMixedSizes(),
                     [](OpFoldResult s) { return getConstantIntValue(s); });
    bool allStridesConstant =
        llvm::all_of(op.getMixedStrides(),
                     [](OpFoldResult s) { return getConstantIntValue(s); });
    if (!allSizesConstant || !allStridesConstant)
      return lowerDynamic(op, adaptor, rewriter);

    const auto &targetModel = AIE::getTargetModel(op);
    BaseMemRefType bufferType = op.getMemref().getType();
    auto *ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    auto zero = IntegerAttr::get(i32ty, 0);

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

    // column
    column = IntegerAttr::get(i32ty, tileCol);

    // row
    row = IntegerAttr::get(i32ty, tileRow);

    // A contiguous row-major ND access on a shim NOC tile is lowered to linear
    // mode (d0_size=d1_size=0) just like an already-canonical linear transfer.
    // This allows naturally-expressed multidimensional transfers (e.g., a 2D
    // image as [height, width]) without hitting the 10-bit ND wrap-size limit.
    bool isLinear = op.isLinearTransferWithoutTransformation() ||
                    (targetModel.isShimNOCTile(tileCol, tileRow) &&
                     isContiguousTransfer(inputSizes, inputStrides));
    if (failed(verifyStridesWraps(op, bufferType, tileCol, tileRow, inputSizes,
                                  inputStrides, sizes, strides, isLinear))) {
      return failure();
    }

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

    // buffer_offset - zero because the complete address is set by the patch op
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

    // Resolve the buffer's runtime-sequence arg and emit the address patch
    // (plus any offset-state update).
    int arg_idx = -1;
    if (failed(emitBufferAddressPatch(op, adaptor, rewriter, tileCol, tileRow,
                                      arg_idx)))
      return failure();

    // push the patched bd onto the dma task queue. bd_id and repeat_count are
    // SSA operands; materialize them as constants here (the static path).
    NpuPushQueueOp::create(
        rewriter, op->getLoc(), column, row, infoOp.getChannelDirAttr(),
        infoOp.getChannelIndexAttr(), issue_token,
        createConstantI32(rewriter, op->getLoc(),
                          static_cast<uint32_t>(repeat_count.getInt())),
        createConstantI32(rewriter, op->getLoc(),
                          static_cast<uint32_t>(bd_id.getInt())));

    rewriter.eraseOp(op);
    return success();
  }

  // Resolve the runtime-sequence argument index the descriptor's buffer traces
  // back to, and emit the address-patch (plus an offset-state update, if the op
  // carries one) that binds the runtime buffer pointer into the BD. Shared by
  // the static and dynamic lowering paths, which are otherwise identical here.
  // On success `argIdx` receives the resolved index.
  LogicalResult emitBufferAddressPatch(NpuDmaMemcpyNdOp op, OpAdaptor adaptor,
                                       ConversionPatternRewriter &rewriter,
                                       int tileCol, int tileRow,
                                       int &argIdx) const {
    AIE::RuntimeSequenceOp seqOp =
        op->getParentOfType<AIE::RuntimeSequenceOp>();
    if (!seqOp)
      return op->emitOpError("NpuDmaMemcpyNdOps must have RuntimeSequenceOp "
                             "parent at time of lowering.");
    auto traceResult = traceSubviewToBlockArgument(adaptor.getMemref());
    if (!traceResult)
      return op->emitOpError(
          "memref must be a block argument or subview/cast/reinterpret_cast of "
          "a block argument with static offsets, sizes, and strides");
    argIdx = -1;
    Block &entryBB = seqOp.getBody().front();
    for (int i = 0, e = entryBB.getNumArguments(); i < e; i++)
      if (entryBB.getArgument(i) == traceResult->rootArg) {
        argIdx = i;
        break;
      }
    if (argIdx < 0)
      return failure();
    int64_t offset = op.getOffsetInBytes() + traceResult->offsetInBytes;

    const auto &targetModel = AIE::getTargetModel(op);
    uint64_t patchAddr =
        targetModel.getDmaBdAddress(tileCol, tileRow, op.getId()) +
        targetModel.getDmaBdAddressOffset(tileCol, tileRow);
    NpuAddressPatchOp::create(rewriter, op->getLoc(), patchAddr, argIdx,
                              createConstantI32(rewriter, op->getLoc(),
                                                static_cast<uint32_t>(offset)));

    // If this DMA op has an offset_state_table_idx, emit an
    // update_from_scratchpad to add the runtime offset to the BD address
    // register (additive; applied after the base patch above).
    if (op.getOffsetStateTableIdxAttr()) {
      auto bufType = cast<BaseMemRefType>(op.getMemref().getType());
      if (failed(emitUpdateBdAddressFromOffsetParameter(rewriter, op, bufType,
                                                        patchAddr)))
        return failure();
    }
    return success();
  }

  // Lower a shim-NOC dma_memcpy_nd carrying runtime (SSA) sizes/strides.
  //
  // Strategy (see the milestone #3222 "dynamic BD-word encoder" step): emit the
  // SAME NpuWriteBdOp as the static path but with zero placeholders in every
  // size/stride-bearing field, so WriteBdToBlockWritePattern still folds it
  // into a single static-template blockwrite. Then override each size/stride BD
  // word with an npu.write32 whose value is computed from the runtime operands
  // via the shared encoder (SsaStridePolicy) -- identical arithmetic to the
  // static path, so a runtime value equal to a constant reproduces the same
  // word. The write32s follow the blockwrite in program order, at fixed
  // absolute register addresses, so ordering is guaranteed without any
  // grouping.
  LogicalResult lowerDynamic(NpuDmaMemcpyNdOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    const auto &targetModel = AIE::getTargetModel(op);
    auto *ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    auto zero = IntegerAttr::get(i32ty, 0);
    Location loc = op->getLoc();

    auto dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return failure();
    auto infoOp = AIE::ShimDMAAllocationOp::getForSymbol(
        dev, op.getMetadata().getRootReference());
    if (!infoOp)
      return op->emitOpError("couldn't find shim_dma_allocation op.");
    AIE::TileOp shimTile = infoOp.getTileOp();
    if (!shimTile)
      return op->emitOpError(
          "shim_dma_allocation op must reference a valid TileOp.");
    auto channelDir = infoOp.getChannelDir();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    int tileCol = shimTile.getCol();
    int tileRow = shimTile.getRow();

    // Packet / token setup (identical to the static path).
    auto column = IntegerAttr::get(i32ty, tileCol);
    auto row = IntegerAttr::get(i32ty, tileRow);
    auto enable_packet = zero, packet_id = zero, packet_type = zero;
    if (auto packetInfo = op.getPacket()) {
      enable_packet = IntegerAttr::get(i32ty, 1);
      packet_type = IntegerAttr::get(i32ty, packetInfo->getPktType());
      packet_id = IntegerAttr::get(i32ty, packetInfo->getPktId());
    }
    auto issue_token = BoolAttr::get(ctx, op.getIssueToken());
    if (!isMM2S)
      issue_token = BoolAttr::get(ctx, true);

    // Emit the BD template with zeros in every size/stride word; the write32
    // overrides below supply the runtime values. valid_bd = 1.
    NpuWriteBdOp::create(
        rewriter, loc, column, /*bd_id=*/IntegerAttr::get(i32ty, op.getId()),
        /*buffer_length=*/zero, /*buffer_offset=*/zero, enable_packet,
        /*out_of_order_id=*/zero, packet_id, packet_type, /*d0_size=*/zero,
        /*d0_stride=*/zero, /*d1_size=*/zero, /*d1_stride=*/zero,
        /*d2_size=*/zero, /*d2_stride=*/zero, /*iteration_current=*/zero,
        /*iteration_size=*/zero, /*iteration_stride=*/zero, /*next_bd=*/zero,
        row, /*use_next_bd=*/zero, /*valid_bd=*/IntegerAttr::get(i32ty, 1),
        /*lock_rel_val=*/zero, /*lock_rel_id=*/zero, /*lock_acq_enable=*/zero,
        /*lock_acq_val=*/zero, /*lock_acq_id=*/zero, /*d0_zero_before=*/zero,
        /*d1_zero_before=*/zero, /*d2_zero_before=*/zero,
        /*d0_zero_after=*/zero,
        /*d1_zero_after=*/zero, /*d2_zero_after=*/zero,
        /*burst_length=*/IntegerAttr::get(i32ty, op.getBurstLength()));

    // Emit the runtime size/stride BD-word overrides (shared with dma_task).
    // buffer_length is the size-product here, so pass no override (null); the
    // encoder returns the hw repeat_count for the queue push.
    Value repeatCount;
    if (failed(emitDynamicShimBdWordOverrides(
            rewriter, loc, targetModel, tileCol, tileRow, op.getId(),
            op.getMixedSizes(), op.getMixedStrides(),
            op.getElementTypeBitwidth(), op.getBurstLength(),
            /*bufLenOverride=*/Value(), repeatCount)))
      return failure();

    // Address patch for the buffer pointer (offset is compile-time here).
    int arg_idx = -1;
    if (failed(emitBufferAddressPatch(op, adaptor, rewriter, tileCol, tileRow,
                                      arg_idx)))
      return failure();

    NpuPushQueueOp::create(
        rewriter, loc, column, row, infoOp.getChannelDirAttr(),
        infoOp.getChannelIndexAttr(), issue_token, repeatCount,
        createConstantI32(rewriter, loc, op.getId()));

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
    Location loc = op->getLoc();
    (void)rewriter.replaceOpWithNewOp<NpuSyncOp>(
        op, createConstantI32(rewriter, loc, shimTile.getCol()),
        createConstantI32(rewriter, loc, shimTile.getRow()),
        createConstantI32(
            rewriter, loc,
            static_cast<uint32_t>(shimDmaAllocOp.getChannelDir())),
        createConstantI32(rewriter, loc, shimDmaAllocOp.getChannelIndex()),
        createConstantI32(rewriter, loc, 1),
        createConstantI32(rewriter, loc, 1));

    return success();
  }
};

struct WriteBdToBlockWritePattern : OpConversionPattern<NpuWriteBdOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  WriteBdToBlockWritePattern(
      MLIRContext *context,
      llvm::DenseMap<Attribute, memref::GlobalOp> *dedupCache, unsigned *nextId,
      PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), dedupCache(dedupCache),
        nextId(nextId) {}

  // Per-pass-run memo + next free name index for blockwrite data globals (see
  // getOrCreateDataMemref). Owned by the pass; live only during the conversion.
  llvm::DenseMap<Attribute, memref::GlobalOp> *dedupCache;
  unsigned *nextId;

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
      global = getOrCreateDataMemref(rewriter, dev, op.getLoc(), words,
                                     dedupCache, nextId);
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

  void runOnOperation() override {

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
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

    // Seed, from the device's existing globals, a dedup cache (initial-value ->
    // global) and the next free "blockwrite_data_<n>" name index (one past the
    // current max). WriteBdToBlockWritePattern keeps both in sync as it creates
    // globals, giving O(1) dedup and name-uniquing per BD; seeding the index
    // past the max keeps names unique by construction. Both are locals holding
    // raw op handles and must not outlive this conversion.
    llvm::DenseMap<Attribute, memref::GlobalOp> dataMemrefCache;
    unsigned nextBlockwriteId = 0;
    for (auto g : device.getOps<memref::GlobalOp>()) {
      if (auto initVal = g.getInitialValue())
        dataMemrefCache.try_emplace(*initVal, g);
      StringRef suffix = g.getSymName();
      if (suffix.consume_front("blockwrite_data_")) {
        unsigned idx;
        if (!suffix.getAsInteger(10, idx) && idx >= nextBlockwriteId)
          nextBlockwriteId = idx + 1;
      }
    }

    RewritePatternSet patterns(&getContext());
    patterns.insert<BlockWriteSymToAddr>(&getContext());
    patterns.insert<DmaToNpuPattern>(&getContext());
    patterns.insert<DmaWaitToSyncPattern>(&getContext());
    patterns.insert<MaskWrite32SymToAddr>(&getContext());
    patterns.insert<PushQueuetoWrite32Pattern>(&getContext());
    patterns.insert<RtpToWrite32Pattern>(&getContext());
    patterns.insert<Write32SymToAddr>(&getContext());
    patterns.insert<WriteBdToBlockWritePattern>(&getContext(), &dataMemrefCache,
                                                &nextBlockwriteId);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEDmaToNpuPass() {
  return std::make_unique<AIEDmaToNpuPass>();
}
