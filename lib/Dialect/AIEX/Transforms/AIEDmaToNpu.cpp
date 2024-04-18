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
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct RtpToNpuPattern : OpConversionPattern<NpuWriteRTPOp> {
  using OpConversionPattern::OpConversionPattern;

  RtpToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuWriteRTPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    auto ui32ty =
        IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Unsigned);
    auto device = op->getParentOfType<AIE::DeviceOp>();

    uint32_t rtp_buffer_addr = UINT_MAX;
    int c = op.getCol();
    int r = op.getRow();
    uint32_t v = op.getValue();
    uint32_t idx = op.getIndex();

    if (auto buffer = device.lookupSymbol<AIE::BufferOp>(op.getBufferSymName()))
      if (AIE::TileOp tile = buffer.getTileOp();
          tile.colIndex() == c && tile.rowIndex() == r) {
        assert(buffer.getAddress().has_value() &&
               "buffer must have address assigned");
        rtp_buffer_addr = static_cast<uint32_t>(buffer.getAddress().value());
      }

    if (rtp_buffer_addr == UINT_MAX)
      return op.emitOpError("RTP buffer address cannot be found. Has an RTP "
                            "buffer been allocated?\n");

    rtp_buffer_addr += idx * sizeof(uint32_t);

    IntegerAttr column = IntegerAttr::get(i32ty, c);
    IntegerAttr row = IntegerAttr::get(i32ty, r);
    IntegerAttr address = IntegerAttr::get(ui32ty, rtp_buffer_addr);
    IntegerAttr value = IntegerAttr::get(i32ty, v);
    rewriter.create<NpuWrite32Op>(op->getLoc(), column.getInt(), row.getInt(),
                                  address.getUInt(), value.getInt());

    rewriter.eraseOp(op);
    return success();
  }
};

std::optional<AIE::ShimDMAAllocationOp>
getAllocOpForSymbol(AIE::DeviceOp dev, StringRef sym_name) {
  auto sym = dev.lookupSymbol(sym_name);
  if (!sym)
    return std::nullopt;

  auto uses = SymbolTable::getSymbolUses(sym, dev);
  for (auto use : *uses)
    if (auto infoOp = dyn_cast<AIE::ShimDMAAllocationOp>(use.getUser()))
      return infoOp;

  return std::nullopt;
}

struct PushToNpuPattern : OpConversionPattern<NpuShimTilePushQueueOp> {
  using OpConversionPattern::OpConversionPattern;

  PushToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuShimTilePushQueueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    auto zero = IntegerAttr::get(i32ty, 0);
    auto ui32ty =
        IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Unsigned);
    bool send_tct = op.getIssueToken();
    uint32_t channel_num = 0;

    // initialize fields to zero
    auto dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return failure();

    auto infoOp = getAllocOpForSymbol(dev, op.getMetadata());
    if (!infoOp) {
      op.emitOpError("couldn't find shim_dma_allocation op");
      return failure();
    }

    auto channelDir = infoOp->getChannelDir();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    channel_num += infoOp->getChannelIndex();

    IntegerAttr column = IntegerAttr::get(i32ty, infoOp->getCol());

    uint32_t queue_offset;
    if (isMM2S)
      queue_offset = 0x1D214;
    else
      queue_offset = 0x1D204;
    if (channel_num == 1)
      queue_offset += 0x8;
    IntegerAttr address = IntegerAttr::get(ui32ty, queue_offset);

    // value
    uint32_t bd_id = op.getBdId();
    uint32_t repeat_cnt = op.getRepeatCount();
    uint32_t cmd = 0;
    cmd |= bd_id & 0xF;
    cmd |= (repeat_cnt & 0xFF) << 16;
    if (send_tct)
      cmd |= 0x80000000;
    IntegerAttr value = IntegerAttr::get(ui32ty, cmd);

    rewriter.create<NpuWrite32Op>(op->getLoc(), column.getInt(), zero.getInt(),
                                  address.getUInt(), value.getUInt());

    rewriter.eraseOp(op);
    return success();
  }
};

struct DmaToNpuPattern : OpConversionPattern<NpuDmaMemcpyNdOp> {
  using OpConversionPattern::OpConversionPattern;

  DmaToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuDmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    auto zero = IntegerAttr::get(i32ty, 0);
    auto memref = adaptor.getMemref();

    auto dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return failure();

    auto infoOp = getAllocOpForSymbol(dev, op.getMetadata());
    if (!infoOp) {
      op.emitOpError("couldn't find shim_dma_allocation op");
      return failure();
    }

    auto channelDir = infoOp->getChannelDir();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    int col = infoOp->getCol();

    // initialize fields to zero
    auto column = zero;
    auto column_num = zero;
    auto ddr_id = zero;
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
    auto d2_stride = zero;
    auto iteration_current = zero;
    auto iteration_size = zero;
    auto iteration_stride = zero;
    auto next_bd = zero;
    auto use_next_bd = zero;
    auto valid_bd = zero;
    auto lock_rel_val = zero;
    auto lock_rel_id = zero;
    auto lock_acq_enable = zero;
    auto lock_acq_val = zero;
    auto lock_acq_id = zero;

    auto issue_token = BoolAttr::get(ctx, false);
    auto repeat_count = zero;

    llvm::SmallVector<int64_t, 3> strides = llvm::map_to_vector(
        llvm::reverse(op.getMixedStrides()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });
    llvm::SmallVector<int64_t, 4> sizes = llvm::map_to_vector(
        llvm::reverse(op.getMixedSizes()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });
    llvm::SmallVector<int64_t, 4> offsets = llvm::map_to_vector(
        llvm::reverse(op.getMixedOffsets()),
        [](OpFoldResult s) { return getConstantIntValue(s).value(); });

    // column
    column = IntegerAttr::get(i32ty, col);

    // column_num
    column_num = IntegerAttr::get(i32ty, 1);

    // ddr_id
    Block &entryBB = op->getParentOfType<func::FuncOp>().getBody().front();
    int arg_idx = -1;
    for (int i = 0, e = entryBB.getNumArguments(); i < e; i++) {
      if (entryBB.getArgument(i) == memref) {
        arg_idx = i;
        break;
      }
    }
    if (arg_idx < 0)
      return failure();
    ddr_id = IntegerAttr::get(i32ty, arg_idx);

    // bd_id
    bd_id = IntegerAttr::get(i32ty, op.getId());

    // buffer_length
    int32_t repeat_length = 0;
    for (int32_t index_3d = 0; index_3d < sizes[2]; index_3d++)
      for (int32_t index_2d = 0; index_2d < sizes[1]; index_2d++)
        repeat_length += sizes[0];
    buffer_length = IntegerAttr::get(i32ty, repeat_length);

    // buffer_offset
    size_t stride = 1;
    size_t offset = 0;
    MemRefType my_memref = op.getMemref().getType();
    auto shape = my_memref.getShape();
    size_t R = shape.size();
    size_t el_bit_width = my_memref.getElementTypeBitWidth();
    assert(el_bit_width % 8 == 0 &&
           "Expected Memref element bitwidth to be multiple of 8.");
    size_t S = el_bit_width / 8;
    for (size_t i = 0; i < R; i++) {
      offset += offsets[i] * stride * S;
      stride *= shape[R - i - 1];
    }
    buffer_offset = IntegerAttr::get(i32ty, offset);

    // enable_packet

    // out_of_order_id

    // packet_id

    // packet_type

    // d0_size
    if (strides[0])
      d0_size = IntegerAttr::get(i32ty, sizes[0]);

    // d0_stride
    d0_stride = IntegerAttr::get(i32ty, 0);

    // d1_size
    if (strides[1])
      d1_size = IntegerAttr::get(i32ty, sizes[1]);

    // d1_stride
    if (strides[0])
      d1_stride = IntegerAttr::get(i32ty, strides[0] - 1);

    // d2_stride
    if (strides[1])
      d2_stride = IntegerAttr::get(i32ty, strides[1] - 1);

    // iteration_current

    // iteration_size
    if (strides[2])
      iteration_size = IntegerAttr::get(i32ty, sizes[3] - 1);

    // iteration_stride
    if (strides[2])
      iteration_stride = IntegerAttr::get(i32ty, strides[2] - 1);

    // next_bd

    // use_next_bd

    // valid_bd
    valid_bd = IntegerAttr::get(i32ty, 1);

    // lock_rel_val

    // lock_rel_id

    // lock_acq_enable

    // lock_acq_val

    // lock_acq_id

    // repeat_count
    repeat_count = IntegerAttr::get(i32ty, sizes[3] - 1);

    // Set the issue_token
    issue_token = BoolAttr::get(ctx, op.getIssueToken());
    // Earlier, all S2MM channels were implicitly assumed to issue a token.
    // This logic is kept for now for backward compatibility.
    if (!isMM2S)
      issue_token = BoolAttr::get(ctx, true);

    (void)rewriter.create<NpuWriteBdExShimTileOp>(
        op->getLoc(), column, column_num, ddr_id, bd_id, buffer_length,
        buffer_offset, enable_packet, out_of_order_id, packet_id, packet_type,
        d0_size, d0_stride, d1_size, d1_stride, d2_stride, iteration_current,
        iteration_size, iteration_stride, next_bd, use_next_bd, valid_bd,
        lock_rel_val, lock_rel_id, lock_acq_enable, lock_acq_val, lock_acq_id);

    rewriter.create<NpuShimTilePushQueueOp>(op->getLoc(), op.getMetadataAttr(),
                                            issue_token, repeat_count, bd_id);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert NpuDmaWaitOp into NpuSyncOp by retrieving the necessary
/// information from the ShimDMAAllocationOp referenced through the
/// symbol argument of this op.
struct DmaWaitToNpuPattern : OpConversionPattern<NpuDmaWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  DmaWaitToNpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuDmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return op.emitOpError("couldn't find parent of type DeviceOp");

    std::optional<AIE::ShimDMAAllocationOp> shimDmaAllocOp =
        getAllocOpForSymbol(dev, op.getSymbol());
    if (!shimDmaAllocOp) {
      op.emitOpError("couldn't find shim_dma_allocation op");
      return failure();
    }
    AIE::DMAChannelDir channelDir = shimDmaAllocOp->getChannelDir();
    int channel = shimDmaAllocOp->getChannelIndex();
    int direction = (int)(channelDir == AIE::DMAChannelDir::MM2S);
    int column = shimDmaAllocOp->getCol();

    // Create with `column_num == 1` and `row_num == 1` to check for a single
    // column and row. Row is always 0 for shim tiles.
    (void)rewriter.replaceOpWithNewOp<NpuSyncOp>(op, column, 0, direction,
                                                 channel, 1, 1);
    return success();
  }
};

struct AIEDmaToNpuPass : AIEDmaToNpuBase<AIEDmaToNpuPass> {
  void runOnOperation() override {

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addLegalOp<AIE::BufferOp>();
    target.addLegalOp<AIE::ShimDMAAllocationOp>();
    target.addIllegalOp<NpuWriteRTPOp>();
    target.addIllegalOp<NpuDmaMemcpyNdOp>();
    target.addIllegalOp<NpuDmaWaitOp>();
    target.addIllegalOp<NpuShimTilePushQueueOp>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<DmaToNpuPattern>(&getContext());
    patterns.insert<DmaWaitToNpuPattern>(&getContext());
    patterns.insert<PushToNpuPattern>(&getContext());
    patterns.insert<RtpToNpuPattern>(&getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEDmaToNpuPass() {
  return std::make_unique<AIEDmaToNpuPass>();
}
