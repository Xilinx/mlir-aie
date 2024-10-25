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

struct Write32SymToAddr : OpConversionPattern<NpuWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  Write32SymToAddr(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getBuffer())
      return failure();

    auto device = op->getParentOfType<AIE::DeviceOp>();
    auto buffer = device.lookupSymbol<AIE::BufferOp>(*op.getBuffer());
    if (!buffer)
      return op->emitError("buffer '" + *op.getBuffer() +
                           "' not found in device");

    if (!buffer.getAddress())
      return op->emitError("buffer must have address assigned");

    const AIE::AIETargetModel &tm = device.getTargetModel();
    uint32_t address = static_cast<uint32_t>(*buffer.getAddress()) +
                       op.getAddress() * sizeof(uint32_t);
    auto col = buffer.getTileOp().getCol();
    auto row = buffer.getTileOp().getRow();
    address |= ((col & 0xff) << tm.getColumnShift()) |
               ((row & 0xff) << tm.getRowShift()) | (address & 0xFFFFF);

    rewriter.replaceOpWithNewOp<NpuWrite32Op>(op, address, op.getValue(),
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

    auto device = op->getParentOfType<AIE::DeviceOp>();

    auto buffer = device.lookupSymbol<AIE::BufferOp>(*op.getBuffer());
    if (!buffer)
      return op->emitError("buffer '" + *op.getBuffer() +
                           "' not found in device");

    if (!buffer.getAddress())
      return op->emitError("buffer must have address assigned");

    const AIE::AIETargetModel &tm = device.getTargetModel();
    uint32_t address = static_cast<uint32_t>(*buffer.getAddress()) +
                       op.getAddress() * sizeof(uint32_t);
    auto col = buffer.getTileOp().getCol();
    auto row = buffer.getTileOp().getRow();
    address |= ((col & 0xff) << tm.getColumnShift()) |
               ((row & 0xff) << tm.getRowShift()) | (address & 0xFFFFF);

    rewriter.replaceOpWithNewOp<NpuBlockWriteOp>(op, address, op.getData(),
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

    auto device = op->getParentOfType<AIE::DeviceOp>();

    auto buffer = device.lookupSymbol<AIE::BufferOp>(*op.getBuffer());
    if (!buffer)
      return op->emitError("buffer '" + *op.getBuffer() +
                           "' not found in device");

    if (!buffer.getAddress())
      return op->emitError("buffer must have address assigned");

    const AIE::AIETargetModel &tm = device.getTargetModel();
    uint32_t address = static_cast<uint32_t>(*buffer.getAddress()) +
                       op.getAddress() * sizeof(uint32_t);
    auto col = buffer.getTileOp().getCol();
    auto row = buffer.getTileOp().getRow();
    address |= ((col & 0xff) << tm.getColumnShift()) |
               ((row & 0xff) << tm.getRowShift()) | (address & 0xFFFFF);

    rewriter.replaceOpWithNewOp<NpuMaskWrite32Op>(
        op, address, op.getValue(), op.getMask(), nullptr, nullptr, nullptr);
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

    rewriter.create<NpuWrite32Op>(op->getLoc(), address, op.getValue(), nullptr,
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

    auto column = rewriter.getI32IntegerAttr(op.getColumn());
    auto row = rewriter.getI32IntegerAttr(0);

    // control packet for issuing token
    if (op.getIssueToken()) {
      uint32_t ctrl_offset =
          op.getDirection() == AIE::DMAChannelDir::MM2S ? 0x1D210 : 0x1D200;
      if (op.getChannel() == 1)
        ctrl_offset += 0x8;
      uint32_t controller_id = 0;
      auto builder = OpBuilder::atBlockBegin(
          op->getParentOfType<AIE::DeviceOp>().getBody());
      auto tOp = AIE::TileOp::getOrCreate(
          builder, op->getParentOfType<AIE::DeviceOp>(), op.getColumn(), 0);
      if (tOp && tOp->hasAttr("controller_id")) {
        auto controllerIdAttr =
            tOp->getAttrOfType<AIE::PacketInfoAttr>("controller_id");
        controller_id = controllerIdAttr.getPktId();
      }
      controller_id = controller_id << 8;
      if (controller_id)
        rewriter.create<NpuMaskWrite32Op>(op->getLoc(), ctrl_offset,
                                          controller_id, 0x0F00, nullptr,
                                          column, row);
    }

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

    rewriter.create<NpuWrite32Op>(op->getLoc(), queue_offset, cmd, nullptr,
                                  column, row);
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
    const auto &targetModel = AIE::getTargetModel(op);
    MemRefType bufferType = op.getMemref().getType();
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
    auto d0_zero_before = zero;
    auto d1_zero_before = zero;
    auto d2_zero_before = zero;
    auto d0_zero_after = zero;
    auto d1_zero_after = zero;
    auto d2_zero_after = zero;

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
    getHardwareStridesWraps(targetModel, bufferType, inputSizes, inputStrides,
                            sizes, strides);
    int64_t offset = op.getOffsetInBytes();

    // column
    column = IntegerAttr::get(i32ty, col);

    // row
    row = IntegerAttr::get(i32ty, 0);

    // arg_idx
    AIEX::RuntimeSequenceOp seq_op =
        op->getParentOfType<AIEX::RuntimeSequenceOp>();
    if (!seq_op) {
      op->emitOpError("NpuDmaMemcpyNdOps must have RuntimeSequenceOp parent at "
                      "time of lowering.");
      return failure();
    }
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
    uint64_t buffer_length_val = inputSizes[0] *
                                 bufferType.getElementTypeBitWidth() /
                                 targetModel.getAddressGenGranularity();
    if (inputSizes.size() > 1) {
      for (size_t i = 1; i < std::min(inputSizes.size(), (size_t)3); i++) {
        buffer_length_val *= inputSizes[i];
      }
    }
    buffer_length = IntegerAttr::get(i32ty, buffer_length_val);

    // buffer_offset
    buffer_offset = IntegerAttr::get(i32ty, offset);

    // enable_packet
    if (auto packetInfo = op.getPacket()) {
      enable_packet = IntegerAttr::get(i32ty, 1);
      packet_type = IntegerAttr::get(i32ty, packetInfo->getPktType());
      packet_id = IntegerAttr::get(i32ty, packetInfo->getPktId());
    }

    // out_of_order_id

    if (!op.isLinearTransferWithoutTransformation()) {
      // d0_size, d0_stride
      d0_size = IntegerAttr::get(i32ty, sizes[0]);
      d0_stride = IntegerAttr::get(i32ty, strides[0]);

      // d1_size, d1_stride
      d1_size = IntegerAttr::get(i32ty, sizes[1]);
      d1_stride = IntegerAttr::get(i32ty, strides[1]);

      // d2_stride
      d2_stride = IntegerAttr::get(i32ty, strides[2]);
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

    // Set the issue_token
    issue_token = BoolAttr::get(ctx, op.getIssueToken());
    // Earlier, all S2MM channels were implicitly assumed to issue a token.
    // This logic is kept for now for backward compatibility.
    if (!isMM2S)
      issue_token = BoolAttr::get(ctx, true);

    // TODO: Need to add a check to only allow zero padding on MM2S channel of
    // MemTile As of now, run time MemTile DMA configuration is supported only
    // from BD level, not at NpuDmaMemcpyNdOp.

    rewriter.create<NpuWriteBdOp>(
        op->getLoc(), column, bd_id, buffer_length, buffer_offset,
        enable_packet, out_of_order_id, packet_id, packet_type, d0_size,
        d0_stride, d1_size, d1_stride, d2_stride, iteration_current,
        iteration_size, iteration_stride, next_bd, row, use_next_bd, valid_bd,
        lock_rel_val, lock_rel_id, lock_acq_enable, lock_acq_val, lock_acq_id,
        d0_zero_before, d1_zero_before, d2_zero_before, d0_zero_after,
        d1_zero_after, d2_zero_after);

    uint64_t addr = getBufferDescriptorAddressRegisterAddress(
        targetModel, op.getId(), col, 0);

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

private:
  int &cachedId;

public:
  WriteBdToBlockWritePattern(MLIRContext *context, int &cachedId,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), cachedId(cachedId) {}

  LogicalResult
  matchAndRewrite(NpuWriteBdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    const AIE::AIETargetModel &tm = dev.getTargetModel();

    std::vector<uint32_t> words(8, 0);
    auto bd_id = op.getBdId();
    uint32_t bd_addr;
    if (tm.isShimNOCTile(op.getColumn(), op.getRow())) {
      bd_addr = (op.getColumn() << tm.getColumnShift()) |
                (op.getRow() << tm.getRowShift()) | (0x1D000 + bd_id * 0x20);

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

      if (op.getD0ZeroBefore() || op.getD1ZeroBefore() ||
          op.getD2ZeroBefore() || op.getD0ZeroAfter() || op.getD1ZeroAfter() ||
          op.getD2ZeroAfter()) {
        op->emitError("Zero padding is only available on MemTile");
      }
    } else if (tm.isMemTile(op.getColumn(), op.getRow())) {
      bd_addr = (op.getColumn() << tm.getColumnShift()) |
                (op.getRow() << tm.getRowShift()) | (0xA0000 + bd_id * 0x20);
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
      // TODO: DMA BD configuration for Compute Tiles
      op->emitError("Run-time DMA configuration is supported only for "
                    "ShimTiles and MemTiles currently.");
      return failure();
    }
    MemRefType memrefType = MemRefType::get({8}, rewriter.getI32Type());
    TensorType tensorType = RankedTensorType::get({8}, rewriter.getI32Type());
    memref::GlobalOp global = nullptr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      std::string name = "blockwrite_data_";
      rewriter.setInsertionPoint(
          op->getParentOfType<AIEX::RuntimeSequenceOp>());
      int id = cachedId;
      while (dev.lookupSymbol(name + std::to_string(id)))
        id++;
      name += std::to_string(id);
      cachedId = id;
      global = rewriter.create<memref::GlobalOp>(
          op->getLoc(), name, rewriter.getStringAttr("private"), memrefType,
          DenseElementsAttr::get<uint32_t>(tensorType, words), true, nullptr);
    }
    auto memref = rewriter.create<memref::GetGlobalOp>(op->getLoc(), memrefType,
                                                       global.getName());
    (void)rewriter.replaceOpWithNewOp<NpuBlockWriteOp>(
        op, rewriter.getUI32IntegerAttr(bd_addr), memref.getResult(), nullptr,
        nullptr, nullptr);
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
    target.addDynamicallyLegalOp<NpuWrite32Op>(
        [&](NpuWrite32Op op) { return !op.getBuffer(); });
    target.addDynamicallyLegalOp<NpuBlockWriteOp>(
        [&](NpuBlockWriteOp op) { return !op.getBuffer(); });
    target.addDynamicallyLegalOp<NpuMaskWrite32Op>(
        [&](NpuMaskWrite32Op op) { return !op.getBuffer(); });

    RewritePatternSet patterns(&getContext());
    patterns.insert<BlockWriteSymToAddr>(&getContext());
    patterns.insert<DmaToNpuPattern>(&getContext(), cachingGetter);
    patterns.insert<DmaWaitToSyncPattern>(&getContext(), cachingGetter);
    patterns.insert<MaskWrite32SymToAddr>(&getContext());
    patterns.insert<PushQueuetoWrite32Pattern>(&getContext());
    patterns.insert<RtpToWrite32Pattern>(&getContext());
    patterns.insert<Write32SymToAddr>(&getContext());
    int cachedId = 0;
    patterns.insert<WriteBdToBlockWritePattern>(&getContext(), cachedId);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEDmaToNpuPass() {
  return std::make_unique<AIEDmaToNpuPass>();
}