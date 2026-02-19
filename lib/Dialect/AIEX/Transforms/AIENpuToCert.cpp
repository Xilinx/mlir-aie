//===- AIENpuToCert.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <type_traits>
#include <vector>

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "npu-to-cert"

namespace {

// slightly smaller than the actual page size to account for overheads and
// estimation errors
static constexpr uint32_t cert_page_size = 8000;

struct RuntimeSequenceToCertJob : OpConversionPattern<AIE::RuntimeSequenceOp> {
  using OpConversionPattern::OpConversionPattern;

  RuntimeSequenceToCertJob(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(AIE::RuntimeSequenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto symName = op.getSymName();
    uint32_t newJobId = 1;
    if (symName != "configure") {
      uint32_t maxJobId = 1;
      op->getParentOp()->walk([&](AIEX::CertJobOp certJobOp) {
        maxJobId = std::max(maxJobId, certJobOp.getJobId());
      });
      newJobId = maxJobId + 1;
    }
    auto jobOp = rewriter.replaceOpWithNewOp<AIEX::CertJobOp>(
        op, op->getResultTypes(), newJobId);
    IRMapping remap;
    op.getRegion().cloneInto(&jobOp.getBody(), remap);
    AIEX::CertJobOp::ensureTerminator(jobOp.getBody(), rewriter, op->getLoc());

    return success();
  }
};

struct NpuWrite32ToCertWrite32 : OpConversionPattern<AIEX::NpuWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AIEX::CertWrite32Op>(op, op.getAddress(),
                                                     op.getValue());
    return success();
  }
};

struct NpuMaskWrite32ToCertMaskWrite32
    : OpConversionPattern<AIEX::NpuMaskWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuMaskWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AIEX::CertMaskWrite32Op>(
        op, op.getAddress(), op.getMask(), op.getValue());
    return success();
  }
};

struct NpuBlockWriteToCertUcDma : OpConversionPattern<AIEX::NpuBlockWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuBlockWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    memref::GetGlobalOp dataOperand =
        dyn_cast_or_null<memref::GetGlobalOp>(op.getData().getDefiningOp());
    if (!dataOperand)
      return failure();
    MemRefType dataType = cast<MemRefType>(dataOperand.getResult().getType());
    uint32_t dataSize = dataType.getNumElements();

    int id = 0;
    std::string symbolName = "chain_" + std::to_string(id);
    while (op->getParentOfType<AIE::DeviceOp>().lookupSymbol(symbolName))
      symbolName = "chain_" + std::to_string(++id);

    // Create a new uc_dma_write_des_sync operation
    rewriter.replaceOpWithNewOp<AIEX::CertUcDmaWriteDesSyncOp>(op, symbolName);

    // Create the uc_dma_chain operation
    rewriter.setInsertionPoint(op->getParentOfType<AIEX::CertJobOp>());
    auto symbolAttr = rewriter.getStringAttr(symbolName);
    auto chainOp =
        rewriter.create<AIEX::CertUcDmaChainOp>(op.getLoc(), symbolAttr);

    Block *bb = new Block();
    chainOp.getRegion().push_back(bb);
    rewriter.setInsertionPointToStart(bb);
    rewriter.create<AIEX::CertUcDmaBdOp>(op.getLoc(), SmallVector<Type>{},
                                         dataOperand.getName(), op.getAddress(),
                                         dataSize, false);

    AIEX::CertUcDmaChainOp::ensureTerminator(chainOp.getBody(), rewriter,
                                             op->getLoc());
    return success();
  }
};

struct NpuSyncToCertWaitTCTS : OpConversionPattern<AIEX::NpuSyncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    uint32_t row = op.getRow();
    uint32_t col = op.getColumn();

    // These are the shift amounts from the tct packet format.
    // The firmware expects the row and column packed and shifted down to zero.
    const int row_id_shift = 16;
    const int col_id_shift = 21;
    uint16_t tile_id = col << (col_id_shift - row_id_shift) | row;
    uint32_t channel = op.getChannel();
    uint32_t direction = op.getDirection();

    uint8_t actor_id = 0;

    std::vector<int> chan2actor_shim_s2mm = {0, 2};
    std::vector<int> chan2actor_shim_mm2s = {6, 7, 8, 9};

    std::vector<int> chan2actor_mem_s2mm = {1, 2, 3, 4, 5, 6, 7};
    std::vector<int> chan2actor_mem_mm2s = {16, 17, 18, 19, 20,
                                            22, 23, 24, 25, 26};
    std::vector<int> chan2actor_tile_s2mm = {0, 1};
    std::vector<int> chan2actor_tile_mm2s = {6};
    assert(channel < chan2actor_shim_mm2s.size());
    if (direction == static_cast<std::underlying_type_t<AIE::DMAChannelDir>>(
                         AIE::DMAChannelDir::S2MM))
      actor_id = chan2actor_shim_s2mm[channel];
    else
      actor_id = chan2actor_shim_mm2s[channel];

    uint8_t num_tcts = 1;
    rewriter.replaceOpWithNewOp<AIEX::CertWaitTCTSOp>(op, tile_id, actor_id,
                                                      num_tcts);
    return success();
  }
};

struct NpuAddressPatchToCertApplyOffset57
    : OpConversionPattern<AIEX::NpuAddressPatchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuAddressPatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // find the previous blockwrite operation
    Block::iterator it(op);
    if (it == op->getBlock()->begin())
      return failure();
    do {
      auto blockWriteOp = dyn_cast<AIEX::NpuBlockWriteOp>(*it--);
      if (!blockWriteOp)
        continue;

      const auto &tm = AIE::getTargetModel(op);
      uint32_t addr = op.getAddr();
      int col = (addr >> tm.getColumnShift()) & 0x1f;
      int row = (addr >> tm.getRowShift()) & 0x1f;
      if (!tm.isValidTile({col, row}))
        return failure();

      // if it's not a matching blockwrite, give up.
      if (blockWriteOp.getAddress() + tm.getDmaBdAddressOffset(col, row) !=
          addr)
        break;

      Value data = blockWriteOp.getData();
      auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(data.getDefiningOp());
      if (!getGlobalOp)
        break;

      // replace the address with the new address
      rewriter.setInsertionPoint(blockWriteOp);
      rewriter.replaceOpWithNewOp<AIEX::CertApplyOffset57Op>(
          op, getGlobalOp.getName(), 1, op.getArgIdx());
      return success();
    } while (it != op->getBlock()->begin());

    return failure();
  }
};

struct MergeConsecutiveCertUcDmaWriteDesSyncOps
    : OpRewritePattern<AIEX::CertUcDmaWriteDesSyncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::CertUcDmaWriteDesSyncOp op,
                                PatternRewriter &rewriter) const override {
    // Get the previous operation in the block
    Block::iterator it(op);
    if (it == op->getBlock()->begin())
      return failure();
    AIEX::CertUcDmaWriteDesSyncOp prevWriteDesSync = nullptr;
    do {
      if (--it == op->getBlock()->end())
        return failure();
      Operation *prevOp = &*it;
      if (isa<AIEX::CertWrite32Op, AIEX::CertMaskWrite32Op,
              AIEX::CertApplyOffset57Op, AIEX::CertWaitTCTSOp>(prevOp))
        return failure();
      prevWriteDesSync = dyn_cast<AIEX::CertUcDmaWriteDesSyncOp>(prevOp);
    } while (!prevWriteDesSync);

    // find the uc_dma_chain
    StringRef sym_name = op.getSymbol();
    StringRef prev_sym_name = prevWriteDesSync.getSymbol();
    auto chain = dyn_cast_if_present<AIEX::CertUcDmaChainOp>(
        op->getParentOfType<AIE::DeviceOp>().lookupSymbol(sym_name));
    auto prevChain = dyn_cast_if_present<AIEX::CertUcDmaChainOp>(
        prevWriteDesSync->getParentOfType<AIE::DeviceOp>().lookupSymbol(
            prev_sym_name));
    if (!chain || !prevChain)
      return failure();

    // Compute the size of the current and previous chains. If their combined
    // data size is greater than the cert page size, then we cannot merge them.
    uint32_t prevChainSize = 0;
    for (auto &o : prevChain.getBody().front().getOperations()) {
      auto bdOp = dyn_cast<AIEX::CertUcDmaBdOp>(o);
      if (!bdOp)
        continue;
      prevChainSize += bdOp.getLength() * sizeof(int);
    }
    uint32_t currChainSize = 0;
    for (auto &o : chain.getBody().front().getOperations()) {
      auto bdOp = dyn_cast<AIEX::CertUcDmaBdOp>(o);
      if (!bdOp)
        continue;
      currChainSize += bdOp.getLength() * sizeof(int);
    }
    if ((currChainSize + prevChainSize) >= cert_page_size)
      return failure();

    IRMapping map;
    rewriter.setInsertionPointToStart(&chain.getBody().front());
    for (auto &o : prevChain.getBody().front().getOperations()) {
      auto bdOp = dyn_cast<AIEX::CertUcDmaBdOp>(o);
      if (!bdOp)
        continue;
      rewriter.create<AIEX::CertUcDmaBdOp>(
          bdOp.getLoc(), bdOp.getRemoteAddress(), bdOp.getLocalAddress(),
          bdOp.getLength(), true);
    }
    prevChain.getBody().cloneInto(&chain.getBody(), map);
    rewriter.eraseOp(prevChain);
    rewriter.eraseOp(prevWriteDesSync);
    return success();
  }
};

struct SplitNpuBlockWriteOpPattern : OpRewritePattern<AIEX::NpuBlockWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::NpuBlockWriteOp op,
                                PatternRewriter &rewriter) const override {

    memref::GetGlobalOp dataOperand =
        dyn_cast_or_null<memref::GetGlobalOp>(op.getData().getDefiningOp());
    if (!dataOperand)
      return failure();

    MemRefType dataType = cast<MemRefType>(dataOperand.getResult().getType());
    uint32_t dataSize = dataType.getNumElements();

    uint32_t dataSizeBytes = dataSize * sizeof(int);
    if (dataSizeBytes < cert_page_size)
      return failure();

    auto loc = op.getLoc();

    // Calculate split point (split roughly in half)
    uint32_t splitElements = dataSize / 2;
    uint32_t firstChunkSize = splitElements;
    uint32_t secondChunkSize = dataSize - splitElements;

    // Find the original memref.global operation
    auto deviceOp = op->getParentOfType<AIE::DeviceOp>();
    auto originalGlobal = dyn_cast_if_present<memref::GlobalOp>(
        deviceOp.lookupSymbol(dataOperand.getName()));
    if (!originalGlobal)
      return failure();

    // Get the original data attribute
    auto originalData = originalGlobal.getInitialValue();
    if (!originalData)
      return failure();

    auto denseData = dyn_cast<DenseIntElementsAttr>(*originalData);
    if (!denseData)
      return failure();

    // Split the data into two chunks
    auto dataValues = denseData.getValues<APInt>();
    std::vector<APInt> firstChunkData(dataValues.begin(),
                                      dataValues.begin() + firstChunkSize);
    std::vector<APInt> secondChunkData(dataValues.begin() + firstChunkSize,
                                       dataValues.end());

    // Create new global operations for the split data
    auto elementType = rewriter.getI32Type();
    auto firstChunkType = MemRefType::get({firstChunkSize}, elementType);
    auto secondChunkType = MemRefType::get({secondChunkSize}, elementType);
    TensorType firstTensorType =
        RankedTensorType::get({firstChunkSize}, elementType);
    TensorType secondTensorType =
        RankedTensorType::get({secondChunkSize}, elementType);

    auto firstChunkAttr =
        DenseIntElementsAttr::get(firstTensorType, firstChunkData);
    auto secondChunkAttr =
        DenseIntElementsAttr::get(secondTensorType, secondChunkData);

    // Generate unique names for the new globals
    std::string firstName = dataOperand.getName().str() + "_split_0";
    std::string secondName = dataOperand.getName().str() + "_split_1";

    // Ensure unique names
    int counter = 0;
    while (deviceOp.lookupSymbol(firstName)) {
      firstName =
          dataOperand.getName().str() + "_split_0_" + std::to_string(counter++);
    }
    counter = 0;
    while (deviceOp.lookupSymbol(secondName)) {
      secondName =
          dataOperand.getName().str() + "_split_1_" + std::to_string(counter++);
    }

    // Create the new global operations
    rewriter.setInsertionPoint(originalGlobal);
    rewriter.create<memref::GlobalOp>(
        loc, firstName, rewriter.getStringAttr("private"), firstChunkType,
        firstChunkAttr, true, nullptr);

    rewriter.create<memref::GlobalOp>(
        loc, secondName, rewriter.getStringAttr("private"), secondChunkType,
        secondChunkAttr, true, nullptr);

    // Create get_global operations for the new data
    rewriter.setInsertionPoint(op);

    auto firstGetGlobal =
        rewriter.create<memref::GetGlobalOp>(loc, firstChunkType, firstName);
    auto secondGetGlobal =
        rewriter.create<memref::GetGlobalOp>(loc, secondChunkType, secondName);

    uint32_t baseAddr = op.getAddress();

    rewriter.create<AIEX::NpuBlockWriteOp>(
        loc, baseAddr, firstGetGlobal.getResult(), nullptr, nullptr, nullptr);

    rewriter.create<AIEX::NpuBlockWriteOp>(loc, baseAddr + firstChunkSize * 4,
                                           secondGetGlobal.getResult(), nullptr,
                                           nullptr, nullptr);

    // Replace the original operation
    rewriter.eraseOp(op);

    LLVM_DEBUG(llvm::outs()
               << "Split NpuBlockWriteOp with data size: " << dataSizeBytes
               << " bytes into chunks of " << firstChunkSize << " and "
               << secondChunkSize << " elements\n");

    return success();
  }
};

struct AIENpuToCertPass : AIEX::AIENpuToCertBase<AIENpuToCertPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalOp<AIE::RuntimeSequenceOp>();

    target.addLegalOp<AIEX::CertApplyOffset57Op>();
    target.addLegalOp<AIEX::CertJobOp>();
    target.addLegalOp<AIEX::CertMaskWrite32Op>();
    target.addLegalOp<AIEX::CertUcDmaWriteDesSyncOp>();
    target.addLegalOp<AIEX::CertUcDmaChainOp>();
    target.addLegalOp<AIEX::CertUcDmaBdOp>();
    target.addLegalOp<AIEX::CertWrite32Op>();
    target.addLegalOp<AIEX::CertWaitTCTSOp>();
    target.addLegalDialect<AIE::AIEDialect>();

    RewritePatternSet p0(&getContext());
    p0.insert<RuntimeSequenceToCertJob>(&getContext());
    p0.insert<NpuAddressPatchToCertApplyOffset57>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(p0))))
      signalPassFailure();

    target.addIllegalOp<AIEX::NpuAddressPatchOp>();

    // patch conversion must come before blockwrite conversion
    RewritePatternSet p1(&getContext());
    p1.insert<NpuAddressPatchToCertApplyOffset57>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(p1))))
      signalPassFailure();

    // Split oversized NpuBlockWriteOps before lowering them to cert ops
    {
      RewritePatternSet p(&getContext());
      p.insert<SplitNpuBlockWriteOpPattern>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(p))))
        signalPassFailure();
    }

    target.addIllegalOp<AIEX::NpuBlockWriteOp>();
    target.addIllegalOp<AIEX::NpuMaskWrite32Op>();
    target.addIllegalOp<AIEX::NpuSyncOp>();
    target.addIllegalOp<AIEX::NpuWrite32Op>();

    // Run npu to cert conversion patterns
    {
      RewritePatternSet p(&getContext());
      p.insert<NpuBlockWriteToCertUcDma>(&getContext());
      p.insert<NpuMaskWrite32ToCertMaskWrite32>(&getContext());
      p.insert<NpuWrite32ToCertWrite32>(&getContext());
      p.insert<NpuSyncToCertWaitTCTS>(&getContext());

      if (failed(applyPartialConversion(getOperation(), target, std::move(p))))
        signalPassFailure();
    }

    // Run the merge pattern for CertUcDmaWriteDesSyncOps
    {
      RewritePatternSet p(&getContext());
      p.insert<MergeConsecutiveCertUcDmaWriteDesSyncOps>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(p))))
        signalPassFailure();
    }
  }
};

} // namespace

static uint32_t estimateCost(Operation *op) {
  if (auto jobOp = dyn_cast<AIEX::CertJobOp>(op)) {
    return estimateCost(jobOp);
  } else if (!isa<AIE::EndOp>(op)) {
    op->emitOpError("contains unsupported operation in cert job");
  }
  return 0;
}

static uint32_t estimateCost(AIEX::CertJobOp op, uint32_t split_target,
                             Block::iterator &split_iter) {
  // assume a job is on its own page
  uint32_t text_cost = 32; // page header
  uint32_t data_cost = 0;
  uint32_t split_cost = 0;
  for (auto &o : op.getBody().front().getOperations()) {
    if (!split_cost && (text_cost + data_cost) >= split_target) {
      split_iter = Block::iterator(&o);
      split_cost = text_cost + data_cost;
    }
    if (isa<AIEX::CertLocalBarrierOp>(o)) {
      text_cost += 8; // local barrier
    } else if (isa<AIEX::CertRemoteBarrierOp>(o)) {
      text_cost += 8; // remote barrier
    } else if (isa<AIEX::CertWaitTCTSOp>(o)) {
      text_cost += 8; // wait tct
    } else if (isa<AIEX::CertMaskWrite32Op>(o)) {
      text_cost += 16; // mask write
    } else if (isa<AIEX::CertWrite32Op>(o)) {
      text_cost += 12; // write
    } else if (isa<AIEX::CertApplyOffset57Op>(o)) {
      text_cost += 16; // apply offset
    } else if (auto syncOp = dyn_cast<AIEX::CertUcDmaWriteDesSyncOp>(o)) {
      text_cost += 16; // write des sync
      // find the uc_dma_chain
      StringRef sym_name = syncOp.getSymbol();
      auto chain = dyn_cast_if_present<AIEX::CertUcDmaChainOp>(
          op->getParentOfType<AIE::DeviceOp>().lookupSymbol(sym_name));
      if (!chain)
        continue;
      for (auto bdOp : chain.getBody().front().getOps<AIEX::CertUcDmaBdOp>()) {
        data_cost += 16; // bd op
        StringRef data_sym_name = bdOp.getRemoteAddress();
        auto global = dyn_cast_if_present<memref::GlobalOp>(
            op->getParentOfType<AIE::DeviceOp>().lookupSymbol(data_sym_name));
        if (!global)
          continue;
        auto initVal = global.getInitialValue();
        if (!initVal)
          continue;
        auto data = dyn_cast<DenseIntElementsAttr>(*initVal);
        data_cost += data.getNumElements() * 4; // 4 bytes per element
      }
    }
  }
  return text_cost + data_cost;
}

namespace {
struct SplitCertJobOpPattern : OpRewritePattern<AIEX::CertJobOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::CertJobOp op,
                                PatternRewriter &rewriter) const override {

    constexpr uint32_t split_threshold = cert_page_size;

    Block::iterator split_iter;
    uint32_t cost = estimateCost(op, cert_page_size / 2, split_iter);
    LLVM_DEBUG(llvm::outs() << "Estimate cost for job: " << op.getJobId()
                            << " is " << cost << "\n");

    if (cost < split_threshold)
      return failure();

    auto loc = op.getLoc();
    op->getParentOfType<AIE::DeviceOp>().walk([&](AIEX::CertJobOp certJobOp) {
      if (certJobOp.getJobId() > op.getJobId())
        certJobOp.setJobId(certJobOp.getJobId() + 1);
    });

    // split the job
    auto jobId = op.getJobId();
    auto newJobOp0 = rewriter.create<AIEX::CertJobOp>(loc, jobId);
    auto newJobOp1 = rewriter.create<AIEX::CertJobOp>(loc, jobId + 1);

    newJobOp0.getBody().push_back(new Block());
    rewriter.setInsertionPointToStart(&newJobOp0.getBody().front());
    for (Block::iterator oi = op.getBody().front().getOperations().begin();
         oi != split_iter; ++oi) {
      rewriter.clone(*oi);
    }
    AIEX::CertJobOp::ensureTerminator(newJobOp0.getBody(), rewriter, loc);

    newJobOp1.getBody().push_back(new Block());
    rewriter.setInsertionPointToStart(&newJobOp1.getBody().front());
    for (Block::iterator oi = split_iter;
         oi != op.getBody().front().getOperations().end(); ++oi) {
      rewriter.clone(*oi);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct AIECertPagesPass : AIEX::AIECertPagesBase<AIECertPagesPass> {
  void runOnOperation() override {
    // First apply the blockwrite splitting pattern
    RewritePatternSet p0(&getContext());
    p0.insert<SplitNpuBlockWriteOpPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(p0))))
      signalPassFailure();

    // Then apply the job splitting pattern
    RewritePatternSet p1(&getContext());
    p1.insert<SplitCertJobOpPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(p1))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIENpuToCertPass() {
  return std::make_unique<AIENpuToCertPass>();
}

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIECertPagesPass() {
  return std::make_unique<AIECertPagesPass>();
}
