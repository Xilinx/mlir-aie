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

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// Helper class to get a ShimDMAAllocationOp for a given <device, symbol name>
// pair. An object of this class is invalidated if, for any symbol_name, a
// ShimDMAAllocationOp that uses it changes, as the cache is not updated in this
// case.
struct ShimDMAllocationGetter {

public:
  // Return the first ShimDMAAllocationOp nested inside the DeviceOp 'dev' that
  // uses the symbol 'sym_name'
  std::optional<AIE::ShimDMAAllocationOp> get(AIE::DeviceOp dev,
                                              StringRef sym_name) {

    auto key = std::make_pair(dev, sym_name);
    auto it = allocGetter.find(key);
    if (it != allocGetter.end())
      return it->second;

    auto allocOp = cachelessGet(dev, sym_name);
    allocGetter[key] = allocOp;
    return allocOp;
  }

private:
  llvm::DenseMap<std::pair<AIE::DeviceOp, StringRef>,
                 std::optional<AIE::ShimDMAAllocationOp>>
      allocGetter;

  // Finding the ShimDMAAllocationOp for a given <DeviceOp, symbol_name> pair
  // can be slow when the symbol is used in many places. This version of the
  // function is only called when the cache does not have a ShimDMAAllocationOp
  // stored from a previous lookup.
  std::optional<AIE::ShimDMAAllocationOp> cachelessGet(AIE::DeviceOp dev,
                                                       StringRef sym_name) {
    auto *sym = dev.lookupSymbol(sym_name);
    if (!sym)
      return std::nullopt;

    auto uses = SymbolTable::getSymbolUses(sym, dev);
    for (auto use : *uses)
      if (auto infoOp = dyn_cast<AIE::ShimDMAAllocationOp>(use.getUser()))
        return infoOp;

    return std::nullopt;
  }
};
} // namespace

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

    rewriter.create<NpuWrite32Op>(op->getLoc(), address, op.getValue(),
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

    // the offset of the task queue register in the tile
    uint32_t queue_offset;
    if (op.getDirection() == AIE::DMAChannelDir::MM2S)
      queue_offset = 0x1D214;
    else
      queue_offset = 0x1D204;
    if (op.getChannel() == 1)
      queue_offset += 0x8;

    // the value to write
    uint32_t bd_id = op.getBdId();
    uint32_t repeat_cnt = op.getRepeatCount();
    uint32_t cmd = 0;
    cmd |= bd_id & 0xF;
    cmd |= (repeat_cnt & 0xFF) << 16;
    if (op.getIssueToken())
      cmd |= 0x80000000;

    auto i32ty = IntegerType::get(op->getContext(), 32);
    auto column = IntegerAttr::get(i32ty, op.getColumn());
    auto row = IntegerAttr::get(i32ty, 0);
    rewriter.create<NpuWrite32Op>(op->getLoc(), queue_offset, cmd, column, row);
    rewriter.eraseOp(op);
    return success();
  }
};

struct DmaToNpuPattern : OpConversionPattern<NpuDmaMemcpyNdOp> {
  using OpConversionPattern::OpConversionPattern;

private:
  ShimDMAllocationGetter &allocGetter;

public:
  DmaToNpuPattern(MLIRContext *context, ShimDMAllocationGetter &getter,
                  PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), allocGetter(getter) {}

  LogicalResult
  matchAndRewrite(NpuDmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op->getContext();
    auto i32ty = IntegerType::get(ctx, 32);
    auto zero = IntegerAttr::get(i32ty, 0);
    auto memref = adaptor.getMemref();

    auto dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return failure();

    auto infoOp = allocGetter.get(dev, op.getMetadata());
    if (!infoOp) {
      return op->emitOpError("couldn't find shim_dma_allocation op.");
    }

    auto channelDir = infoOp->getChannelDir();
    bool isMM2S = channelDir == AIE::DMAChannelDir::MM2S;
    int col = infoOp->getCol();

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

    auto issue_token = BoolAttr::get(ctx, false);
    auto repeat_count = zero;

    llvm::SmallVector<int64_t, 4> strides = op.getStridesInAddressGranularity();
    llvm::SmallVector<int64_t, 4> sizes = op.getSizesInAddressGranularity();
    int64_t offset = op.getOffsetInBytes();

    // column
    column = IntegerAttr::get(i32ty, col);

    // arg_idx
    AIEX::RuntimeSequenceOp seq_op =
        op->getParentOfType<AIEX::RuntimeSequenceOp>();
    assert(seq_op && "NpuDmaMemcpyNdOp must be inside a RuntimeSequenceOp; "
                     "verify() should have ensured this.");
    Block &entryBB = seq_op.getBody().front();
    int arg_idx = -1;
    for (int i = 0, e = entryBB.getNumArguments(); i < e; i++) {
      if (entryBB.getArgument(i) == memref) {
        arg_idx = i;
        break;
      }
    }
    if (arg_idx < 0)
      return failure();

    // bd_id
    bd_id = IntegerAttr::get(i32ty, op.getId());

    // buffer_length
    int32_t repeat_length = 0;
    for (int32_t index_3d = 0; index_3d < sizes[2]; index_3d++)
      for (int32_t index_2d = 0; index_2d < sizes[1]; index_2d++)
        repeat_length += sizes[0];
    buffer_length = IntegerAttr::get(i32ty, repeat_length);

    // buffer_offset
    buffer_offset = IntegerAttr::get(i32ty, offset);

    // enable_packet

    // out_of_order_id

    // packet_id

    // packet_type

    // d0_size
    if (strides[1])
      d0_size = IntegerAttr::get(i32ty, sizes[0]);

    // d0_stride
    if (strides[0])
      d0_stride = IntegerAttr::get(i32ty, strides[0] - 1);

    // d1_size
    if (strides[2])
      d1_size = IntegerAttr::get(i32ty, sizes[1]);

    // d1_stride
    if (strides[1])
      d1_stride = IntegerAttr::get(i32ty, strides[1] - 1);

    // d2_stride
    if (strides[2])
      d2_stride = IntegerAttr::get(i32ty, strides[2] - 1);

    // iteration_current

    // iteration_size
    // strides[3] doesn't need to lower to hardware if sizes[3] is one
    if (strides[3] && sizes[3] != 1)
      iteration_size = IntegerAttr::get(i32ty, sizes[3] - 1);

    // iteration_stride
    if (strides[3])
      iteration_stride = IntegerAttr::get(i32ty, strides[3] - 1);

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

    rewriter.create<NpuWriteBdOp>(
        op->getLoc(), column, bd_id, buffer_length, buffer_offset,
        enable_packet, out_of_order_id, packet_id, packet_type, d0_size,
        d0_stride, d1_size, d1_stride, d2_stride, iteration_current,
        iteration_size, iteration_stride, next_bd, row, use_next_bd, valid_bd,
        lock_rel_val, lock_rel_id, lock_acq_enable, lock_acq_val, lock_acq_id);

    const AIE::AIETargetModel &tm =
        op->getParentOfType<AIE::DeviceOp>().getTargetModel();

    uint32_t addr =
        (col << tm.getColumnShift()) | (0x1D004 + op.getId() * 0x20);
    rewriter.create<NpuAddressPatchOp>(op->getLoc(), addr, arg_idx, offset);

    rewriter.create<NpuPushQueueOp>(
        op->getLoc(), column, row, infoOp->getChannelDirAttr(),
        infoOp->getChannelIndexAttr(), issue_token, repeat_count, bd_id);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert NpuDmaWaitOp into NpuSyncOp by retrieving the necessary
/// information from the ShimDMAAllocationOp referenced through the
/// symbol argument of this op.
struct DmaWaitToSyncPattern : OpConversionPattern<NpuDmaWaitOp> {

private:
  ShimDMAllocationGetter &allocGetter;

public:
  using OpConversionPattern::OpConversionPattern;

  DmaWaitToSyncPattern(MLIRContext *context, ShimDMAllocationGetter &getter,
                       PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), allocGetter(getter) {}

  LogicalResult
  matchAndRewrite(NpuDmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return op->emitError("couldn't find parent of type DeviceOp");

    std::optional<AIE::ShimDMAAllocationOp> shimDmaAllocOp =
        allocGetter.get(dev, op.getSymbol());
    if (!shimDmaAllocOp) {
      return op->emitError("couldn't find shim_dma_allocation op");
    }

    // Create with `column_num == 1` and `row_num == 1` to check for a single
    // column and row. Row is always 0 for shim tiles.
    (void)rewriter.replaceOpWithNewOp<NpuSyncOp>(
        op, shimDmaAllocOp->getCol(), /* row */ 0,
        static_cast<uint32_t>(shimDmaAllocOp->getChannelDir()),
        shimDmaAllocOp->getChannelIndex(), 1, 1);

    return success();
  }
};

struct WriteBdToBlockWritePattern : OpConversionPattern<NpuWriteBdOp> {
  using OpConversionPattern::OpConversionPattern;

  WriteBdToBlockWritePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuWriteBdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    const AIE::AIETargetModel &tm = dev.getTargetModel();

    auto bd_id = op.getBdId();
    uint32_t bd_addr = (op.getColumn() << tm.getColumnShift()) |
                       (op.getRow() << tm.getRowShift()) |
                       (0x1D000 + bd_id * 0x20);

    std::vector<uint32_t> words(8, 0);

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
    words[4] = 0x80000000; // burst length;
    words[4] |= (op.getD1Size() & 0x3ff) << 20;
    words[4] |= op.getD1Stride() & 0xfffff;

    // DMA_BDX_5
    // TODO: SIMID, AxCache, AXQoS
    words[5] = op.getD2Stride() & 0xfffff;

    // DMA_BDX_6
    words[6] |= (op.getIterationCurrent() & 0x3f) << 26;
    words[6] |= (op.getIterationSize() & 0x3f) << 20;
    words[6] |= op.getIterationStride() & 0xfffff;

    // DMA_BDX_7
    // TODO: TLAST Suppress
    words[7] |= (op.getNextBd() & 0xf) << 27;
    words[7] |= (op.getUseNextBd() & 0x1) << 26;
    words[7] |= (op.getValidBd() & 0x1) << 25;
    words[7] |= (op.getLockRelVal() & 0xef) << 18;
    words[7] |= (op.getLockRelId() & 0xf) << 13;
    words[7] |= (op.getLockAcqEnable() & 0x1) << 12;
    words[7] |= (op.getLockAcqVal() & 0xef) << 5;
    words[7] |= op.getLockAcqId() & 0xf;

    MemRefType memrefType = MemRefType::get({8}, rewriter.getI32Type());
    TensorType tensorType = RankedTensorType::get({8}, rewriter.getI32Type());
    memref::GlobalOp global = nullptr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      std::string name = "blockwrite_data_";
      rewriter.setInsertionPoint(
          op->getParentOfType<AIEX::RuntimeSequenceOp>());
      int id = 0;
      while (dev.lookupSymbol(name + std::to_string(id)))
        id++;
      name += std::to_string(id);
      global = rewriter.create<memref::GlobalOp>(
          op->getLoc(), name, rewriter.getStringAttr("private"), memrefType,
          DenseElementsAttr::get<uint32_t>(tensorType, words), true, nullptr);
    }
    auto memref = rewriter.create<memref::GetGlobalOp>(op->getLoc(), memrefType,
                                                       global.getName());
    (void)rewriter.replaceOpWithNewOp<NpuBlockWriteOp>(
        op, memref.getResult(), rewriter.getUI32IntegerAttr(bd_addr));
    return success();
  }
};

struct AIEDmaToNpuPass : AIEDmaToNpuBase<AIEDmaToNpuPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {

    ShimDMAllocationGetter cachingGetter;

    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalOp<AIE::BufferOp>();
    target.addLegalOp<AIE::ShimDMAAllocationOp>();

    target.addIllegalOp<NpuDmaMemcpyNdOp>();
    target.addIllegalOp<NpuDmaWaitOp>();
    target.addIllegalOp<NpuPushQueueOp>();
    target.addIllegalOp<NpuWriteRTPOp>();
    target.addIllegalOp<NpuWriteBdOp>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<DmaToNpuPattern>(&getContext(), cachingGetter);
    patterns.insert<DmaWaitToSyncPattern>(&getContext(), cachingGetter);
    patterns.insert<PushQueuetoWrite32Pattern>(&getContext());
    patterns.insert<RtpToWrite32Pattern>(&getContext());
    patterns.insert<WriteBdToBlockWritePattern>(&getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEDmaToNpuPass() {
  return std::make_unique<AIEDmaToNpuPass>();
}
