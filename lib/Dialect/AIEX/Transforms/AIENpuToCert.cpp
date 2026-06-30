//===- AIENpuToCert.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIENPUTOCERT
#define GEN_PASS_DEF_AIECERTPAGES
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "npu-to-cert"

namespace {

// slightly smaller than the actual page size to account for overheads and
// estimation errors
static constexpr uint32_t cert_page_size = 8000;

// Returns the PDI id to use for a cert.load_pdi that references `deviceSymName`
// within `parentDevice`. If a load_pdi for that symbol already exists, its id
// is reused so repeated loads of the same PDI share an id; otherwise a fresh id
// (1-based, one past the current maximum) is assigned.
static uint32_t getOrAssignPdiId(AIE::DeviceOp parentDevice,
                                 StringRef deviceSymName) {
  uint32_t pdiId = 1;
  bool foundExisting = false;
  parentDevice.walk([&](AIEX::CertLoadPdiOp loadPdiOp) {
    if (loadPdiOp.getSymbol() == deviceSymName) {
      pdiId = loadPdiOp.getPdiId();
      foundExisting = true;
    } else if (!foundExisting) {
      pdiId = std::max(pdiId, loadPdiOp.getPdiId() + 1);
    }
  });
  return pdiId;
}

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

    // Create page wrapper at the same location as the runtime sequence
    rewriter.setInsertionPoint(op);
    auto pageOp = rewriter.create<AIEX::CertPageOp>(op.getLoc());
    Block *pageBlock = new Block();
    pageOp.getBody().push_back(pageBlock);
    rewriter.setInsertionPointToStart(pageBlock);

    // Create job inside page
    auto jobOp = rewriter.create<AIEX::CertJobOp>(op.getLoc(), newJobId);

    // Clone runtime sequence body into job
    // Note: This preserves block arguments from the runtime sequence, which
    // will be present in the MLIR IR but are not emitted in the final assembly
    IRMapping remap;
    op.getRegion().cloneInto(&jobOp.getBody(), remap);
    AIEX::CertJobOp::ensureTerminator(jobOp.getBody(), rewriter, op->getLoc());
    AIEX::CertPageOp::ensureTerminator(pageOp.getBody(), rewriter,
                                       op->getLoc());

    // Erase the original runtime sequence
    rewriter.eraseOp(op);

    return success();
  }
};

struct NpuWrite32ToCertWrite32 : OpConversionPattern<AIEX::NpuWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    std::optional<uint32_t> address =
        AIEX::getConstantIntOperand(op.getAddress());
    std::optional<uint32_t> value = AIEX::getConstantIntOperand(op.getValue());
    if (!address || !value)
      return op.emitOpError(
          "cannot lower to cert.write32 with non-constant address or value");

    // Get the absolute address, which properly handles row/col if present
    std::optional<uint32_t> absAddress = op.getAbsoluteAddress();
    if (!absAddress)
      return failure();

    uint32_t absoluteAddr = *absAddress;

    // If row and col are specified, validate that the address upper bits match
    std::optional<uint32_t> col = op.getColumn();
    std::optional<uint32_t> row = op.getRow();
    if (col && row) {
      const auto &tm = AIE::getTargetModel(op);
      uint32_t expectedUpperBits = ((*col & 0xff) << tm.getColumnShift()) |
                                   ((*row & 0xff) << tm.getRowShift());

      // Warn if the original address had non-zero upper bits that don't match
      uint32_t origAddress = *address;
      uint32_t origUpperBits = origAddress & ~0xfffff;
      if (origUpperBits != 0 && origUpperBits != expectedUpperBits) {
        op.emitWarning() << "address upper bits (0x"
                         << llvm::utohexstr(origUpperBits)
                         << ") don't match row=" << *row << " col=" << *col
                         << " computed bits (0x"
                         << llvm::utohexstr(expectedUpperBits) << ")";
      }
    }

    rewriter.replaceOpWithNewOp<AIEX::CertWrite32Op>(op, absoluteAddr, *value);
    return success();
  }
};

struct NpuMaskWrite32ToCertMaskWrite32
    : OpConversionPattern<AIEX::NpuMaskWrite32Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuMaskWrite32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    std::optional<uint32_t> address =
        AIEX::getConstantIntOperand(op.getAddress());
    std::optional<uint32_t> mask = AIEX::getConstantIntOperand(op.getMask());
    std::optional<uint32_t> value = AIEX::getConstantIntOperand(op.getValue());
    if (!address || !mask || !value)
      return op.emitOpError(
          "cannot lower to cert.maskwrite32 with non-constant "
          "address, mask, or value");

    // Get the absolute address, which properly handles row/col if present
    std::optional<uint32_t> absAddress = op.getAbsoluteAddress();
    if (!absAddress)
      return failure();

    uint32_t absoluteAddr = *absAddress;

    // If row and col are specified, validate that the address upper bits match
    std::optional<uint32_t> col = op.getColumn();
    std::optional<uint32_t> row = op.getRow();
    if (col && row) {
      const auto &tm = AIE::getTargetModel(op);
      uint32_t expectedUpperBits = ((*col & 0xff) << tm.getColumnShift()) |
                                   ((*row & 0xff) << tm.getRowShift());

      // Warn if the original address had non-zero upper bits that don't match
      uint32_t origAddress = *address;
      uint32_t origUpperBits = origAddress & ~0xfffff;
      if (origUpperBits != 0 && origUpperBits != expectedUpperBits) {
        op.emitWarning() << "address upper bits (0x"
                         << llvm::utohexstr(origUpperBits)
                         << ") don't match row=" << *row << " col=" << *col
                         << " computed bits (0x"
                         << llvm::utohexstr(expectedUpperBits) << ")";
      }
    }

    rewriter.replaceOpWithNewOp<AIEX::CertMaskWrite32Op>(op, absoluteAddr,
                                                         *mask, *value);
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
    // Find the nearest device to insert the chain
    auto parentDevice = op->getParentOfType<AIE::DeviceOp>();
    if (!parentDevice) {
      // No parent device - this shouldn't happen but handle gracefully
      return failure();
    }

    // Insert after the last existing uc_dma_chain (before any pages/jobs),
    // preserving the order of blockwrite ops.
    Block *deviceBody = parentDevice.getBody();
    Operation *insertAfter = nullptr;
    for (Operation &bodyOp : *deviceBody) {
      if (isa<AIEX::CertUcDmaChainOp>(bodyOp))
        insertAfter = &bodyOp;
      else
        break;
    }
    if (insertAfter)
      rewriter.setInsertionPointAfter(insertAfter);
    else
      rewriter.setInsertionPointToStart(deviceBody);

    auto symbolAttr = rewriter.getStringAttr(symbolName);
    auto chainOp =
        AIEX::CertUcDmaChainOp::create(rewriter, op.getLoc(), symbolAttr);

    Block *bb = new Block();
    chainOp.getRegion().push_back(bb);
    rewriter.setInsertionPointToStart(bb);
    AIEX::CertUcDmaBdOp::create(rewriter, op.getLoc(), dataOperand.getName(),
                                op.getAddress(), dataSize, false);

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
    std::optional<uint32_t> rowOpt = AIEX::getConstantIntOperand(op.getRow());
    std::optional<uint32_t> colOpt =
        AIEX::getConstantIntOperand(op.getColumn());
    std::optional<uint32_t> channelOpt =
        AIEX::getConstantIntOperand(op.getChannel());
    std::optional<uint32_t> directionOpt =
        AIEX::getConstantIntOperand(op.getDirection());
    if (!rowOpt || !colOpt || !channelOpt || !directionOpt)
      return op.emitOpError(
          "cannot lower to cert.wait_tcts with non-constant sync parameters");
    uint32_t row = *rowOpt;
    uint32_t col = *colOpt;

    // These are the shift amounts from the tct packet format in the
    // architecture spec. The firmware expects the row and column packed and
    // shifted down to zero.
    const int row_id_shift = 16;
    const int col_id_shift = 21;
    uint16_t tile_id = col << (col_id_shift - row_id_shift) | row;
    uint32_t channel = *channelOpt;
    uint32_t direction = *directionOpt;

    const std::vector<int> chan2actor_shim_s2mm = {0, 2, 3, 4};
    const std::vector<int> chan2actor_shim_mm2s = {6, 7, 8, 9, 10, 11, 12, 13};
    const std::vector<int> chan2actor_mem_s2mm = {1, 2, 3, 4, 5, 6, 7};
    const std::vector<int> chan2actor_mem_mm2s = {16, 17, 18, 19, 20,
                                                  22, 23, 24, 25, 26};
    const std::vector<int> chan2actor_tile_s2mm = {0, 1};
    const std::vector<int> chan2actor_tile_mm2s = {6};

    const auto &tm = AIE::getTargetModel(op);
    const bool isS2MM =
        direction == static_cast<std::underlying_type_t<AIE::DMAChannelDir>>(
                         AIE::DMAChannelDir::S2MM);

    const std::vector<int> *chan2actor = nullptr;
    if (tm.isCoreTile(col, row))
      chan2actor = isS2MM ? &chan2actor_tile_s2mm : &chan2actor_tile_mm2s;
    else if (tm.isMemTile(col, row))
      chan2actor = isS2MM ? &chan2actor_mem_s2mm : &chan2actor_mem_mm2s;
    else
      chan2actor = isS2MM ? &chan2actor_shim_s2mm : &chan2actor_shim_mm2s;

    size_t chanIdx = static_cast<size_t>(channel);
    if (chanIdx >= chan2actor->size()) {
      op.emitError("invalid DMA channel ")
          << channel << " for " << (isS2MM ? "S2MM" : "MM2S")
          << " direction in NpuSyncToCertWaitTCTS conversion";
      return failure();
    }

    uint8_t actor_id = static_cast<uint8_t>((*chan2actor)[chanIdx]);
    uint8_t num_tcts = 1;
    rewriter.replaceOpWithNewOp<AIEX::CertWaitTCTSOp>(op, tile_id, actor_id,
                                                      num_tcts);
    return success();
  }
};

struct NpuLoadPdiToCertLoadPdi : OpConversionPattern<AIEX::NpuLoadPdiOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AIEX::NpuLoadPdiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the device reference
    auto deviceRef = op.getDeviceRef();
    if (!deviceRef)
      return failure();

    StringRef deviceSymName = *deviceRef;

    // Find parent device to get unique PDI ID
    auto parentDevice = op->getParentOfType<AIE::DeviceOp>();
    if (!parentDevice)
      return failure();

    // Assign a PDI ID, reusing an existing one for this device if present.
    uint32_t pdiId = getOrAssignPdiId(parentDevice, deviceSymName);

    // Replace with cert.load_pdi
    rewriter.replaceOpWithNewOp<AIEX::CertLoadPdiOp>(
        op, rewriter.getUI32IntegerAttr(pdiId),
        FlatSymbolRefAttr::get(rewriter.getContext(), deviceSymName));

    return success();
  }
};

struct RunOpInlining : OpRewritePattern<AIEX::RunOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::RunOp op,
                                PatternRewriter &rewriter) const override {
    // Get the callee runtime sequence
    AIE::RuntimeSequenceOp calleeRuntimeSequence =
        op.getCalleeRuntimeSequenceOp();
    if (!calleeRuntimeSequence)
      return failure();

    // Get the callee body region
    Region &calleeBody = calleeRuntimeSequence.getBody();

    // Create argument mapping from run op arguments to callee parameters
    IRMapping argMap;
    ValueRange values = op.getArgs();
    for (unsigned i = 0, n = calleeBody.getNumArguments(); i < n; i++) {
      BlockArgument arg = calleeBody.getArgument(i);
      Value val = values[i];
      argMap.map(arg, val);
    }

    // Clone operations from callee into current location
    rewriter.setInsertionPoint(op);
    for (Operation &o : calleeBody.front().getOperations()) {
      // Skip the terminator
      if (isa<AIE::EndOp>(o))
        continue;
      rewriter.clone(o, argMap);
    }

    // Erase the run op
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConfigureOpToCertSection : OpRewritePattern<AIEX::ConfigureOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::ConfigureOp op,
                                PatternRewriter &rewriter) const override {
    // Get the referenced device
    AIE::DeviceOp referencedDevice = op.getReferencedDeviceOp();
    if (!referencedDevice)
      return failure();

    // Get the device's symbol name for the section
    StringRef deviceSymName = referencedDevice.getSymName();

    // Find the parent DeviceOp to insert the section into
    auto parentDevice = op->getParentOfType<AIE::DeviceOp>();
    if (!parentDevice)
      return failure();

    // Check if section already exists (avoid creating duplicates)
    if (parentDevice.lookupSymbol(deviceSymName)) {
      // Section already exists, just create load_pdi at call site, reusing the
      // PDI ID already assigned to this section (or assigning a new one).
      uint32_t pdiId = getOrAssignPdiId(parentDevice, deviceSymName);

      rewriter.setInsertionPoint(op);
      rewriter.create<AIEX::CertLoadPdiOp>(
          op.getLoc(), rewriter.getUI32IntegerAttr(pdiId),
          FlatSymbolRefAttr::get(rewriter.getContext(), deviceSymName));

      // Clone configure body operations after the load_pdi
      IRMapping bodyMapper;
      for (Operation &o : op.getRegion().front().getOperations()) {
        if (isa<AIE::EndOp>(o))
          continue;
        rewriter.clone(o, bodyMapper);
      }

      rewriter.eraseOp(op);
      return success();
    }

    // Create cert.section with the device symbol name
    rewriter.setInsertionPoint(parentDevice.getBody()->getTerminator());
    auto sectionOp = rewriter.create<AIEX::CertSectionOp>(
        op.getLoc(), rewriter.getStringAttr(deviceSymName));

    // Create the section body with a page containing a job
    Block *sectionBlock = new Block();
    sectionOp.getBody().push_back(sectionBlock);
    rewriter.setInsertionPointToStart(sectionBlock);

    // Create page within section
    auto pageOp = rewriter.create<AIEX::CertPageOp>(op.getLoc());
    Block *pageBlock = new Block();
    pageOp.getBody().push_back(pageBlock);
    rewriter.setInsertionPointToStart(pageBlock);

    // Create job within page - assign unique job ID across all devices
    // Find the maximum job ID in the parent device
    uint32_t maxJobId = 0;
    parentDevice.walk([&](AIEX::CertJobOp certJobOp) {
      maxJobId = std::max(maxJobId, certJobOp.getJobId());
    });
    uint32_t sectionJobId = maxJobId + 1;

    auto jobOp = rewriter.create<AIEX::CertJobOp>(op.getLoc(), sectionJobId);
    Block *jobBlock = new Block();
    jobOp.getBody().push_back(jobBlock);
    rewriter.setInsertionPointToStart(jobBlock);

    // Find the "configure" sequence content from the referenced device
    // It could be either:
    // 1. Still a RuntimeSequenceOp named "configure" (if that device hasn't
    // been processed yet)
    // 2. Already converted to cert.page/cert.job (if that device was processed
    // first)

    AIE::RuntimeSequenceOp configureSeq = nullptr;
    referencedDevice.walk([&](AIE::RuntimeSequenceOp seq) {
      if (seq.getSymName() == "configure") {
        configureSeq = seq;
      }
    });

    if (configureSeq) {
      // Clone operations from the "configure" runtime_sequence
      IRMapping mapper;
      for (Operation &o : configureSeq.getRegion().front().getOperations()) {
        if (isa<AIE::EndOp>(o))
          continue;
        rewriter.clone(o, mapper);
      }
    } else {
      // "configure" was already converted - find the first cert.page
      // (configuration page)
      AIEX::CertPageOp configPage = nullptr;
      for (auto &op : referencedDevice.getBody()->getOperations()) {
        if (auto page = dyn_cast<AIEX::CertPageOp>(op)) {
          configPage = page;
          break; // Take first page (configure)
        }
      }

      if (configPage) {
        // Clone operations from the config page's job
        IRMapping mapper;
        for (auto &op : configPage.getBody().front().getOperations()) {
          if (auto job = dyn_cast<AIEX::CertJobOp>(op)) {
            for (Operation &jobOp : job.getBody().front().getOperations()) {
              if (isa<AIE::EndOp>(jobOp))
                continue;
              rewriter.clone(jobOp, mapper);
            }
            break; // Only clone first job
          }
        }
      }
    }

    // Ensure terminators
    AIEX::CertJobOp::ensureTerminator(jobOp.getBody(), rewriter, op.getLoc());
    AIEX::CertPageOp::ensureTerminator(pageOp.getBody(), rewriter, op.getLoc());
    AIEX::CertSectionOp::ensureTerminator(sectionOp.getBody(), rewriter,
                                          op.getLoc());

    // At the configure call site, replace with load_pdi, reusing the PDI ID
    // already assigned to this device if present (or assigning a new one).
    uint32_t pdiId = getOrAssignPdiId(parentDevice, deviceSymName);

    rewriter.setInsertionPoint(op);
    rewriter.create<AIEX::CertLoadPdiOp>(
        op.getLoc(), rewriter.getUI32IntegerAttr(pdiId),
        FlatSymbolRefAttr::get(rewriter.getContext(), deviceSymName));

    // Clone configure body operations after the load_pdi
    // Operations that reference external values were not cloned into the
    // section, so we clone them here into the main control flow
    IRMapping bodyMapper;
    for (Operation &o : op.getRegion().front().getOperations()) {
      if (isa<AIE::EndOp>(o))
        continue;
      rewriter.clone(o, bodyMapper);
    }

    // Erase the original configure op
    rewriter.eraseOp(op);
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
    while (it != op->getBlock()->begin()) {
      --it;
      auto blockWriteOp = dyn_cast<AIEX::NpuBlockWriteOp>(*it);
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
    }

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
    AIEX::CertUcDmaWriteDesSyncOp prevWriteDesSync = nullptr;
    while (it != op->getBlock()->begin() && !prevWriteDesSync) {
      --it;
      Operation *prevOp = &*it;
      if (isa<AIEX::CertWrite32Op, AIEX::CertMaskWrite32Op,
              AIEX::CertApplyOffset57Op, AIEX::CertWaitTCTSOp>(prevOp))
        return failure();
      prevWriteDesSync = dyn_cast<AIEX::CertUcDmaWriteDesSyncOp>(prevOp);
    }
    if (!prevWriteDesSync)
      return failure();

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
      AIEX::CertUcDmaBdOp::create(
          rewriter, bdOp.getLoc(), bdOp.getRemoteAddress(),
          bdOp.getLocalAddress(), bdOp.getLength(), true);
    }
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
    memref::GlobalOp::create(rewriter, loc, firstName,
                             rewriter.getStringAttr("private"), firstChunkType,
                             firstChunkAttr, true, nullptr);

    memref::GlobalOp::create(rewriter, loc, secondName,
                             rewriter.getStringAttr("private"), secondChunkType,
                             secondChunkAttr, true, nullptr);

    // Create get_global operations for the new data
    rewriter.setInsertionPoint(op);

    auto firstGetGlobal =
        memref::GetGlobalOp::create(rewriter, loc, firstChunkType, firstName);
    auto secondGetGlobal =
        memref::GetGlobalOp::create(rewriter, loc, secondChunkType, secondName);

    uint32_t baseAddr = op.getAddress();

    AIEX::NpuBlockWriteOp::create(rewriter, loc, baseAddr,
                                  firstGetGlobal.getResult(), nullptr, nullptr,
                                  nullptr);

    AIEX::NpuBlockWriteOp::create(rewriter, loc, baseAddr + firstChunkSize * 4,
                                  secondGetGlobal.getResult(), nullptr, nullptr,
                                  nullptr);

    // Replace the original operation
    rewriter.eraseOp(op);

    LLVM_DEBUG(llvm::outs()
               << "Split NpuBlockWriteOp with data size: " << dataSizeBytes
               << " bytes into chunks of " << firstChunkSize << " and "
               << secondChunkSize << " elements\n");

    return success();
  }
};

struct AIENpuToCertPass
    : xilinx::AIEX::impl::AIENpuToCertBase<AIENpuToCertPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Collect all devices
    llvm::SmallVector<AIE::DeviceOp, 4> devices;
    for (auto &op : moduleOp.getBody()->getOperations()) {
      if (auto deviceOp = dyn_cast<AIE::DeviceOp>(op)) {
        devices.push_back(deviceOp);
      }
    }

    // Process main device first (named "main" or specified by --device-name)
    // so referenced devices' "configure" sequences can be found before they
    // get converted to cert.job
    AIE::DeviceOp mainDevice = nullptr;
    for (auto dev : devices) {
      if (dev.getSymName() == deviceName) {
        mainDevice = dev;
        processDevice(dev);
        break;
      }
    }

    // Then process other devices
    for (auto dev : devices) {
      if (dev != mainDevice) {
        processDevice(dev);
      }
    }

    // Remove absorbed devices (keep only main device)
    // Referenced devices have been absorbed as cert.sections, so remove them
    llvm::SmallVector<AIE::DeviceOp, 4> devicesToRemove;
    for (auto dev : devices) {
      if (dev != mainDevice) {
        devicesToRemove.push_back(dev);
      }
    }

    for (auto dev : devicesToRemove) {
      dev.erase();
    }
  }

  void processDevice(AIE::DeviceOp currentDevice) {
    auto moduleOp = currentDevice->getParentOfType<ModuleOp>();

    // Identify and convert referenced "configure" sequences early, before the
    // RuntimeSequenceOp conversion runs on other devices.
    if (moduleOp) {

      // Collect npu.load_pdi operations (before they are converted below)
      llvm::SmallVector<StringRef, 4> referencedDeviceSyms;
      currentDevice.walk([&](AIEX::NpuLoadPdiOp loadPdiOp) {
        if (loadPdiOp.getDeviceRef())
          referencedDeviceSyms.push_back(*loadPdiOp.getDeviceRef());
      });

      if (!referencedDeviceSyms.empty()) {
        OpBuilder builder(&getContext());

        for (StringRef refSymName : referencedDeviceSyms) {
          // Skip if section already exists
          if (currentDevice.lookupSymbol(refSymName))
            continue;

          // Find referenced device
          AIE::DeviceOp refDevice = nullptr;
          for (auto &op : moduleOp.getBody()->getOperations()) {
            if (auto dev = dyn_cast<AIE::DeviceOp>(op)) {
              if (dev.getSymName() == refSymName) {
                refDevice = dev;
                break;
              }
            }
          }

          if (!refDevice)
            continue;

          // Find "configure" runtime sequence
          AIE::RuntimeSequenceOp configureSeq = nullptr;
          refDevice.walk([&](AIE::RuntimeSequenceOp seq) {
            if (seq.getSymName() == "configure") {
              configureSeq = seq;
            }
          });

          if (!configureSeq)
            continue;

          // Create cert.section in current device
          builder.setInsertionPoint(currentDevice.getBody()->getTerminator());
          auto sectionOp = builder.create<AIEX::CertSectionOp>(
              configureSeq.getLoc(), builder.getStringAttr(refSymName));

          Block *sectionBlock = new Block();
          sectionOp.getBody().push_back(sectionBlock);
          builder.setInsertionPointToStart(sectionBlock);

          // Create page
          auto pageOp = builder.create<AIEX::CertPageOp>(configureSeq.getLoc());
          Block *pageBlock = new Block();
          pageOp.getBody().push_back(pageBlock);
          builder.setInsertionPointToStart(pageBlock);

          // Get unique job ID
          uint32_t maxJobId = 0;
          currentDevice.walk([&](AIEX::CertJobOp certJobOp) {
            maxJobId = std::max(maxJobId, certJobOp.getJobId());
          });

          // Create job
          auto jobOp = builder.create<AIEX::CertJobOp>(configureSeq.getLoc(),
                                                       maxJobId + 1);

          // Clone configure sequence body into job
          IRMapping mapper;
          configureSeq.getRegion().cloneInto(&jobOp.getBody(), mapper);

          // Ensure terminators
          AIEX::CertJobOp::ensureTerminator(jobOp.getBody(), builder,
                                            configureSeq.getLoc());
          AIEX::CertPageOp::ensureTerminator(pageOp.getBody(), builder,
                                             configureSeq.getLoc());
          AIEX::CertSectionOp::ensureTerminator(sectionOp.getBody(), builder,
                                                configureSeq.getLoc());
        }
      }
    }

    // Inline RunOps first
    RewritePatternSet p_run_inline(&getContext());
    p_run_inline.insert<RunOpInlining>(&getContext());
    if (failed(applyPatternsGreedily(currentDevice, std::move(p_run_inline))))
      return signalPassFailure();

    // Then convert ConfigureOps to cert.section + cert.load_pdi
    RewritePatternSet p_configure(&getContext());
    p_configure.insert<ConfigureOpToCertSection>(&getContext());
    if (failed(applyPatternsGreedily(currentDevice, std::move(p_configure))))
      return signalPassFailure();

    ConversionTarget target(getContext());
    target.addIllegalOp<AIE::RuntimeSequenceOp>();

    target.addLegalOp<AIEX::CertApplyOffset57Op>();
    target.addLegalOp<AIEX::CertJobOp>();
    target.addLegalOp<AIEX::CertPageOp>();
    target.addLegalOp<AIEX::CertSectionOp>();
    target.addLegalOp<AIEX::CertLoadPdiOp>();
    target.addLegalOp<AIEX::CertMaskWrite32Op>();
    target.addLegalOp<AIEX::CertUcDmaWriteDesSyncOp>();
    target.addLegalOp<AIEX::CertUcDmaChainOp>();
    target.addLegalOp<AIEX::CertUcDmaBdOp>();
    target.addLegalOp<AIEX::CertWrite32Op>();
    target.addLegalOp<AIEX::CertWaitTCTSOp>();
    target.addLegalOp<AIEX::ConfigureOp>(); // TODO: Convert in separate pass
    target.addLegalDialect<AIE::AIEDialect>();

    RewritePatternSet p0(&getContext());
    p0.insert<RuntimeSequenceToCertJob>(&getContext());

    if (failed(applyPartialConversion(currentDevice, target, std::move(p0))))
      return signalPassFailure();

    target.addIllegalOp<AIEX::NpuAddressPatchOp>();

    // patch conversion must come before blockwrite conversion
    RewritePatternSet p1(&getContext());
    p1.insert<NpuAddressPatchToCertApplyOffset57>(&getContext());

    if (failed(applyPartialConversion(currentDevice, target, std::move(p1))))
      return signalPassFailure();

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
    target.addIllegalOp<AIEX::NpuLoadPdiOp>();

    RewritePatternSet p2(&getContext());
    p2.insert<NpuBlockWriteToCertUcDma>(&getContext());
    p2.insert<NpuMaskWrite32ToCertMaskWrite32>(&getContext());
    p2.insert<NpuWrite32ToCertWrite32>(&getContext());
    p2.insert<NpuSyncToCertWaitTCTS>(&getContext());
    p2.insert<NpuLoadPdiToCertLoadPdi>(&getContext());

    if (failed(applyPartialConversion(currentDevice, target, std::move(p2))))
      return signalPassFailure();

    // Add the merge pattern for CertUcDmaWriteDesSyncOps
    RewritePatternSet p3(&getContext());
    p3.insert<MergeConsecutiveCertUcDmaWriteDesSyncOps>(&getContext());
    if (failed(applyPatternsGreedily(currentDevice, std::move(p3))))
      return signalPassFailure();

    // Convert referenced devices to cert.sections
    // Now that cert.load_pdi ops exist, find referenced devices and convert
    // their cert jobs into sections in this device
    {
      OpBuilder builder(&getContext());

      // Find all cert.load_pdi in current device
      llvm::SetVector<StringRef> referencedDeviceNames;
      currentDevice.walk([&](AIEX::CertLoadPdiOp loadPdiOp) {
        StringRef refSymName = loadPdiOp.getSymbol();
        if (!currentDevice.lookupSymbol(refSymName))
          referencedDeviceNames.insert(refSymName);
      });

      // For each referenced device, create a section with its cert jobs
      for (StringRef refSymName : referencedDeviceNames) {
        // Find the referenced device
        AIE::DeviceOp refDevice = nullptr;
        for (auto &op : moduleOp.getBody()->getOperations()) {
          if (auto dev = dyn_cast<AIE::DeviceOp>(op)) {
            if (dev.getSymName() == refSymName) {
              refDevice = dev;
              break;
            }
          }
        }

        if (!refDevice)
          continue;

        // Find all cert.page ops in the referenced device that are direct
        // children
        llvm::SmallVector<AIEX::CertPageOp, 4> certPages;
        for (auto &op : refDevice.getBody()->getOperations()) {
          if (auto pageOp = dyn_cast<AIEX::CertPageOp>(op)) {
            certPages.push_back(pageOp);
          }
        }

        if (certPages.empty())
          continue;

        // Create cert.section in current device
        builder.setInsertionPoint(currentDevice.getBody()->getTerminator());
        auto sectionOp = builder.create<AIEX::CertSectionOp>(
            refDevice.getLoc(), builder.getStringAttr(refSymName));

        Block *sectionBlock = new Block();
        sectionOp.getBody().push_back(sectionBlock);

        // Clone all cert.page operations into the section
        IRMapping mapper;
        for (auto pageOp : certPages) {
          builder.setInsertionPointToEnd(sectionBlock);
          builder.clone(*pageOp.getOperation(), mapper);
        }

        // Ensure section terminator
        AIEX::CertSectionOp::ensureTerminator(sectionOp.getBody(), builder,
                                              refDevice.getLoc());
      }
    }

    // Clean up unused block arguments from cert.job operations
    currentDevice.walk([&](AIEX::CertJobOp jobOp) {
      Block &jobBlock = jobOp.getBody().front();

      // Remove unused block arguments (in reverse order)
      for (int i = jobBlock.getNumArguments() - 1; i >= 0; --i) {
        BlockArgument arg = jobBlock.getArgument(i);
        if (arg.use_empty()) {
          jobBlock.eraseArgument(i);
        }
      }
    });
  }
};

} // namespace

static void updateCostForOp(Operation &o, AIE::DeviceOp deviceOp,
                            uint32_t &text_cost, uint32_t &data_cost) {
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
        deviceOp.lookupSymbol(sym_name));
    if (!chain)
      return;
    for (auto bdOp : chain.getBody().front().getOps<AIEX::CertUcDmaBdOp>()) {
      data_cost += 16; // bd op
      StringRef data_sym_name = bdOp.getRemoteAddress();
      auto global = dyn_cast_if_present<memref::GlobalOp>(
          deviceOp.lookupSymbol(data_sym_name));
      if (!global)
        continue;
      auto initVal = global.getInitialValue();
      if (!initVal)
        continue;
      auto data = dyn_cast<DenseIntElementsAttr>(*initVal);
      if (!data)
        continue;
      data_cost += data.getNumElements() * 4; // 4 bytes per element
    }
  }
}

static uint32_t estimateCost(AIEX::CertPageOp op, uint32_t split_target,
                             AIEX::CertJobOp &split_job,
                             Block::iterator &split_iter,
                             bool &found_split_point) {
  uint32_t text_cost = 32; // page header
  uint32_t data_cost = 0;
  found_split_point = false;
  AIE::DeviceOp deviceOp = op->getParentOfType<AIE::DeviceOp>();

  for (auto job : op.getBody().front().getOps<AIEX::CertJobOp>()) {
    for (auto &o : job.getBody().front().getOperations()) {
      Block::iterator current(&o);
      if (!found_split_point && !isa<AIE::EndOp>(o) &&
          current != job.getBody().front().begin() &&
          (text_cost + data_cost) >= split_target) {
        split_job = job;
        split_iter = current;
        found_split_point = true;
      }

      updateCostForOp(o, deviceOp, text_cost, data_cost);

      if (!found_split_point && (text_cost + data_cost) >= split_target) {
        Block::iterator next = current;
        ++next;
        if (next != job.getBody().front().end() && !isa<AIE::EndOp>(*next)) {
          split_job = job;
          split_iter = next;
          found_split_point = true;
        }
      }
    }
  }
  return text_cost + data_cost;
}

namespace {

struct WrapStandaloneCertJobOpPattern : OpRewritePattern<AIEX::CertJobOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::CertJobOp jobOp,
                                PatternRewriter &rewriter) const override {
    if (jobOp->getParentOfType<AIEX::CertPageOp>())
      return failure();

    auto loc = jobOp.getLoc();
    rewriter.setInsertionPoint(jobOp);

    auto pageOp = AIEX::CertPageOp::create(rewriter, loc);
    Block *pageBlock = new Block();
    pageOp.getBody().push_back(pageBlock);

    jobOp->moveBefore(pageBlock, pageBlock->begin());
    AIEX::CertPageOp::ensureTerminator(pageOp.getBody(), rewriter, loc);

    return success();
  }
};

// Pattern to isolate load_pdi and preempt operations into their own job and
// page According to CERT spec: "load_pdi and preempt should take one whole job
// which in turn should take one whole page"
struct IsolateFullPageOpsPattern : OpRewritePattern<AIEX::CertJobOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::CertJobOp jobOp,
                                PatternRewriter &rewriter) const override {

    // Find first load_pdi or preempt in this job
    Operation *fullPageOp = nullptr;
    Block::iterator fullPageOpIter;
    for (Block::iterator it = jobOp.getBody().front().begin();
         it != jobOp.getBody().front().end(); ++it) {
      if (isa<AIEX::CertLoadPdiOp, AIEX::CertPreemptOp>(*it)) {
        fullPageOp = &*it;
        fullPageOpIter = it;
        break;
      }
    }

    if (!fullPageOp)
      return failure(); // No full-page op in this job

    // Check if this job ONLY contains the full-page op (and terminator)
    size_t opCount = 0;
    for (Operation &op : jobOp.getBody().front().getOperations()) {
      if (!isa<AIE::EndOp>(op))
        opCount++;
    }

    if (opCount == 1) {
      // Job only contains full-page op - check if it's in its own page
      auto parentPage = jobOp->getParentOfType<AIEX::CertPageOp>();
      if (!parentPage)
        return failure(); // No parent page (unusual but skip)

      // Count jobs in parent page
      size_t jobCount = 0;
      for (Operation &op : parentPage.getBody().front().getOperations()) {
        if (isa<AIEX::CertJobOp>(op))
          jobCount++;
      }

      if (jobCount == 1)
        return failure(); // Already properly isolated

      // Job is isolated but shares page - need to move to own page
      auto loc = jobOp.getLoc();
      rewriter.setInsertionPointAfter(parentPage);

      // Create new page for this job
      auto newPageOp = AIEX::CertPageOp::create(rewriter, loc);
      Block *newPageBlock = new Block();
      newPageOp.getBody().push_back(newPageBlock);

      // Move the job to the new page
      rewriter.setInsertionPointToStart(newPageBlock);
      jobOp->moveBefore(newPageBlock, newPageBlock->begin());

      AIEX::CertPageOp::ensureTerminator(newPageOp.getBody(), rewriter, loc);

      return success();
    }

    // Job contains full-page op mixed with other operations - need to split
    auto loc = jobOp.getLoc();
    auto parentDevice = jobOp->getParentOfType<AIE::DeviceOp>();

    // Assign new job IDs
    uint32_t maxJobId = 0;
    parentDevice.walk([&](AIEX::CertJobOp certJobOp) {
      maxJobId = std::max(maxJobId, certJobOp.getJobId());
    });

    uint32_t beforeJobId = jobOp.getJobId();
    uint32_t fullPageJobId = maxJobId + 1;
    uint32_t afterJobId = maxJobId + 2;

    // Collect operations before full-page op
    SmallVector<Operation *> beforeOps;
    for (Block::iterator it = jobOp.getBody().front().begin();
         it != fullPageOpIter; ++it) {
      if (!isa<AIE::EndOp>(*it))
        beforeOps.push_back(&*it);
    }

    // Collect operations after full-page op
    SmallVector<Operation *> afterOps;
    Block::iterator afterStart = fullPageOpIter;
    ++afterStart; // Skip the full-page op itself
    for (Block::iterator it = afterStart; it != jobOp.getBody().front().end();
         ++it) {
      if (!isa<AIE::EndOp>(*it))
        afterOps.push_back(&*it);
    }

    // Get parent page to insert new pages after it
    auto parentPage = jobOp->getParentOfType<AIEX::CertPageOp>();
    rewriter.setInsertionPoint(parentPage);

    // Create first page with operations before full-page op (if any)
    if (!beforeOps.empty()) {
      auto page1 = AIEX::CertPageOp::create(rewriter, loc);
      Block *page1Block = new Block();
      page1.getBody().push_back(page1Block);
      rewriter.setInsertionPointToStart(page1Block);

      auto job1 = AIEX::CertJobOp::create(rewriter, loc, beforeJobId);
      Block *job1Block = new Block();
      job1.getBody().push_back(job1Block);
      rewriter.setInsertionPointToStart(job1Block);

      for (Operation *op : beforeOps) {
        op->moveBefore(job1Block, job1Block->end());
      }

      AIEX::CertJobOp::ensureTerminator(job1.getBody(), rewriter, loc);
      AIEX::CertPageOp::ensureTerminator(page1.getBody(), rewriter, loc);
    }

    // Create page with full-page op in its own job
    rewriter.setInsertionPointAfter(parentPage);
    auto page2 = AIEX::CertPageOp::create(rewriter, loc);
    Block *page2Block = new Block();
    page2.getBody().push_back(page2Block);
    rewriter.setInsertionPointToStart(page2Block);

    auto job2 = AIEX::CertJobOp::create(rewriter, loc, fullPageJobId);
    Block *job2Block = new Block();
    job2.getBody().push_back(job2Block);
    rewriter.setInsertionPointToStart(job2Block);

    fullPageOp->moveBefore(job2Block, job2Block->end());

    AIEX::CertJobOp::ensureTerminator(job2.getBody(), rewriter, loc);
    AIEX::CertPageOp::ensureTerminator(page2.getBody(), rewriter, loc);

    // Create third page with operations after full-page op (if any)
    if (!afterOps.empty()) {
      rewriter.setInsertionPointAfter(page2);
      auto page3 = AIEX::CertPageOp::create(rewriter, loc);
      Block *page3Block = new Block();
      page3.getBody().push_back(page3Block);
      rewriter.setInsertionPointToStart(page3Block);

      auto job3 = AIEX::CertJobOp::create(rewriter, loc, afterJobId);
      Block *job3Block = new Block();
      job3.getBody().push_back(job3Block);
      rewriter.setInsertionPointToStart(job3Block);

      for (Operation *op : afterOps) {
        op->moveBefore(job3Block, job3Block->end());
      }

      AIEX::CertJobOp::ensureTerminator(job3.getBody(), rewriter, loc);
      AIEX::CertPageOp::ensureTerminator(page3.getBody(), rewriter, loc);
    }

    // Erase the original job and page
    rewriter.eraseOp(jobOp);
    if (parentPage.getBody().front().empty() ||
        llvm::all_of(parentPage.getBody().front(),
                     [](Operation &op) { return isa<AIE::EndOp>(op); })) {
      rewriter.eraseOp(parentPage);
    }

    return success();
  }
};

struct SplitCertPageOpPattern : OpRewritePattern<AIEX::CertPageOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIEX::CertPageOp op,
                                PatternRewriter &rewriter) const override {

    constexpr uint32_t split_threshold = cert_page_size;

    AIEX::CertJobOp split_job;
    Block::iterator split_iter;
    bool found_split_point = false;
    uint32_t cost = estimateCost(op, cert_page_size / 2, split_job, split_iter,
                                 found_split_point);
    LLVM_DEBUG(llvm::outs() << "Estimate cost for page: "
                            << " is " << cost << "\n");

    if (cost < split_threshold || !found_split_point)
      return failure();

    auto loc = op.getLoc();
    op->getParentOfType<AIE::DeviceOp>().walk([&](AIEX::CertJobOp certJobOp) {
      if (certJobOp.getJobId() > split_job.getJobId())
        certJobOp.setJobId(certJobOp.getJobId() + 1);
    });

    auto cloneJobRange = [&](AIEX::CertJobOp sourceJob, uint32_t jobId,
                             Block::iterator begin, Block::iterator end) {
      auto newJobOp = AIEX::CertJobOp::create(rewriter, loc, jobId);
      Block *newJobBlock = new Block();
      newJobOp.getBody().push_back(newJobBlock);

      IRMapping mapper;
      Block &sourceBlock = sourceJob.getBody().front();
      for (BlockArgument arg : sourceBlock.getArguments()) {
        BlockArgument newArg =
            newJobBlock->addArgument(arg.getType(), arg.getLoc());
        mapper.map(arg, newArg);
      }

      rewriter.setInsertionPointToStart(newJobBlock);
      for (Block::iterator oi = begin; oi != end; ++oi) {
        if (!isa<AIE::EndOp>(*oi))
          rewriter.clone(*oi, mapper);
      }
      AIEX::CertJobOp::ensureTerminator(newJobOp.getBody(), rewriter, loc);
    };

    rewriter.setInsertionPoint(op);
    auto newPageOp0 = AIEX::CertPageOp::create(rewriter, loc);
    Block *newPageBlock0 = new Block();
    newPageOp0.getBody().push_back(newPageBlock0);

    auto newPageOp1 = AIEX::CertPageOp::create(rewriter, loc);
    Block *newPageBlock1 = new Block();
    newPageOp1.getBody().push_back(newPageBlock1);

    for (auto job : op.getBody().front().getOps<AIEX::CertJobOp>()) {
      if (job == split_job) {
        rewriter.setInsertionPointToEnd(newPageBlock0);
        cloneJobRange(job, job.getJobId(), job.getBody().front().begin(),
                      split_iter);

        rewriter.setInsertionPointToEnd(newPageBlock1);
        cloneJobRange(job, job.getJobId() + 1, split_iter,
                      job.getBody().front().end());
        continue;
      }

      if (job->isBeforeInBlock(split_job)) {
        rewriter.setInsertionPointToEnd(newPageBlock0);
        cloneJobRange(job, job.getJobId(), job.getBody().front().begin(),
                      job.getBody().front().end());
      } else {
        rewriter.setInsertionPointToEnd(newPageBlock1);
        cloneJobRange(job, job.getJobId(), job.getBody().front().begin(),
                      job.getBody().front().end());
      }
    }

    AIEX::CertPageOp::ensureTerminator(newPageOp0.getBody(), rewriter, loc);
    AIEX::CertPageOp::ensureTerminator(newPageOp1.getBody(), rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

struct AIECertPagesPass
    : xilinx::AIEX::impl::AIECertPagesBase<AIECertPagesPass> {
  void runOnOperation() override {
    // Normalize legacy standalone jobs to the page-based representation.
    RewritePatternSet wrapPatterns(&getContext());
    wrapPatterns.insert<WrapStandaloneCertJobOpPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(wrapPatterns))))
      signalPassFailure();

    // First, isolate load_pdi and preempt operations into their own job/page
    RewritePatternSet isolatePatterns(&getContext());
    isolatePatterns.insert<IsolateFullPageOpsPattern>(&getContext());
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(isolatePatterns))))
      signalPassFailure();

    // Then apply the page splitting pattern
    RewritePatternSet p1(&getContext());
    p1.insert<SplitCertPageOpPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(p1))))
      signalPassFailure();

    // Add the merge pattern for CertUcDmaWriteDesSyncOps
    RewritePatternSet p3(&getContext());
    p3.insert<MergeConsecutiveCertUcDmaWriteDesSyncOps>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(p3))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> AIEX::createAIENpuToCertPass() {
  return std::make_unique<AIENpuToCertPass>();
}

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIECertPagesPass() {
  return std::make_unique<AIECertPagesPass>();
}
