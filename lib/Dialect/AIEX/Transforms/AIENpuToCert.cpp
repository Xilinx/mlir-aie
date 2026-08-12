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

#include <map>
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

// Two distinct thresholds, doing two different jobs.
//
// `cert_page_size` is the conservative trigger for *attempting* a split. It
// sits below the real limit to absorb the imprecision in estimateCost (page
// `.align 16` padding, `.eop`, and the data-vs-page layout it approximates).
//
// `cert_page_limit` is the hard architectural bound: a uC page buffer is 8 KB.
// Anything at or under it is legal hardware.
//
// The distinction matters for diagnostics. A page between the two values trips
// the trigger but is still perfectly loadable, so failing to split it is not an
// error -- only a page whose estimate exceeds `cert_page_limit` is genuinely
// unlowerable and worth rejecting at compile time.
static constexpr uint32_t cert_page_size = 8000;
static constexpr uint32_t cert_page_limit = 8192;

// Per-job control-code overhead: START_JOB (ISA_OPSIZE_START_JOB = 0x08) plus
// END_JOB (ISA_OPSIZE_END_JOB = 0x04), emitted around every job's body by
// emitJob.
static constexpr uint32_t cert_job_start_cost = 8;
static constexpr uint32_t cert_job_end_cost = 4;

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

    // Create the job in place of the runtime sequence, WITHOUT a page wrapper.
    // A later "form implicit pages" step (in cert-legalize-pages) groups
    // contiguous top-level jobs into pages; explicit cert.page ops are the only
    // delimiters. CertJobOp legally parents under DeviceOp, so a bare
    // device-level job is valid between the two passes.
    rewriter.setInsertionPoint(op);
    auto jobOp = AIEX::CertJobOp::create(rewriter, op.getLoc(), newJobId);

    // Tag the implicit configuration job so a later step can force it to be the
    // first page on uC0 and keep it out of user implicit pages.
    if (symName == "configure")
      jobOp->setAttr("cert.configure", rewriter.getUnitAttr());

    // Clone runtime sequence body into job
    // Note: This preserves block arguments from the runtime sequence, which
    // will be present in the MLIR IR but are not emitted in the final assembly
    IRMapping remap;
    op.getRegion().cloneInto(&jobOp.getBody(), remap);
    AIEX::CertJobOp::ensureTerminator(jobOp.getBody(), rewriter, op->getLoc());

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
    // preserving the order of blockwrite ops. Globals are allowed to precede
    // the chains, so skip over them too.
    Block *deviceBody = parentDevice.getBody();
    Operation *insertAfter = nullptr;
    for (Operation &bodyOp : *deviceBody) {
      if (isa<AIEX::CertUcDmaChainOp>(bodyOp) || isa<memref::GlobalOp>(bodyOp))
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
    if (values.size() != calleeBody.getNumArguments()) {
      op.emitError("number of run op arguments (")
          << values.size()
          << ") does not match number of callee runtime sequence "
             "arguments ("
          << calleeBody.getNumArguments() << ")";
      return failure();
    }
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
    auto existingSection = dyn_cast_if_present<AIEX::CertSectionOp>(
        parentDevice.lookupSymbol(deviceSymName));
    if (existingSection) {
      // Section already exists, just create load_pdi at call site, reusing the
      // PDI ID already assigned to this section (or assigning a new one).
      uint32_t pdiId = getOrAssignPdiId(parentDevice, deviceSymName);

      rewriter.setInsertionPoint(op);
      AIEX::CertLoadPdiOp::create(
          rewriter, op.getLoc(), rewriter.getUI32IntegerAttr(pdiId),
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
    auto sectionOp = AIEX::CertSectionOp::create(
        rewriter, op.getLoc(), rewriter.getStringAttr(deviceSymName));

    // Create the section body with a page containing a job
    Block *sectionBlock = new Block();
    sectionOp.getBody().push_back(sectionBlock);
    rewriter.setInsertionPointToStart(sectionBlock);

    // Create page within section
    auto pageOp = AIEX::CertPageOp::create(rewriter, op.getLoc());
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

    auto jobOp = AIEX::CertJobOp::create(rewriter, op.getLoc(), sectionJobId);
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
    AIEX::CertLoadPdiOp::create(
        rewriter, op.getLoc(), rewriter.getUI32IntegerAttr(pdiId),
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
    //
    // A chain costs 16 bytes of descriptor per BD on top of the BD's payload --
    // the same accounting updateCostForOp (the page splitter's authoritative
    // cost model) uses. Charging payload only under-reports a chain by 16 bytes
    // per BD, which for many small BDs is most of its real size.
    auto chainSize = [](AIEX::CertUcDmaChainOp c) {
      uint32_t size = 0;
      for (auto bdOp : c.getBody().front().getOps<AIEX::CertUcDmaBdOp>())
        size += bdOp.getLength() * sizeof(int) + 16; // payload + bd descriptor
      return size;
    };
    if ((chainSize(chain) + chainSize(prevChain)) >= cert_page_size)
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

    if (!mainDevice) {
      moduleOp.emitError("no device found matching --device-name \"")
          << deviceName << "\"";
      signalPassFailure();
      return;
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
          auto sectionOp =
              AIEX::CertSectionOp::create(builder, configureSeq.getLoc(),
                                          builder.getStringAttr(refSymName));

          Block *sectionBlock = new Block();
          sectionOp.getBody().push_back(sectionBlock);
          builder.setInsertionPointToStart(sectionBlock);

          // Create page
          auto pageOp =
              AIEX::CertPageOp::create(builder, configureSeq.getLoc());
          Block *pageBlock = new Block();
          pageOp.getBody().push_back(pageBlock);
          builder.setInsertionPointToStart(pageBlock);

          // Get unique job ID
          uint32_t maxJobId = 0;
          currentDevice.walk([&](AIEX::CertJobOp certJobOp) {
            maxJobId = std::max(maxJobId, certJobOp.getJobId());
          });

          // Create job
          auto jobOp = AIEX::CertJobOp::create(builder, configureSeq.getLoc(),
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
    target.addIllegalOp<AIEX::NpuBlockWriteValuesOp>();
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

    // Run the greedy driver with no patterns, purely for its DCE: p2
    // leaves each converted blockwrite's memref.get_global (and any constant
    // feeding it) dead, since the new op references the global by symbol
    // rather than by SSA value. CertJobOp's verifier rejects those ops
    RewritePatternSet cleanup(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(cleanup))))
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
        auto sectionOp = AIEX::CertSectionOp::create(
            builder, refDevice.getLoc(), builder.getStringAttr(refSymName));

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

// Instruction sizes are the authoritative ISA_OPSIZE_* values from
// third_party/aiebu/specification/aie2ps/isa_stubs.h -- the same table the
// assembler encodes against -- so the running total here matches the bytes
// emitJob actually produces. Keep the two in sync when adding a cert op:
// an op the emitter can emit but this function ignores is counted as free,
// which makes the page look smaller than it is.
static void updateCostForOp(Operation &o, AIE::DeviceOp deviceOp,
                            uint32_t &text_cost, uint32_t &data_cost) {
  // Several distinct op kinds share an encoded instruction size below --
  // an instruction-size table, not a copy-paste clone.
  // NOLINTBEGIN(bugprone-branch-clone)
  if (isa<AIEX::CertLocalBarrierOp>(o)) {
    text_cost += 4; // ISA_OPSIZE_LOCAL_BARRIER
  } else if (isa<AIEX::CertRemoteBarrierOp>(o)) {
    text_cost += 8; // ISA_OPSIZE_REMOTE_BARRIER
  } else if (isa<AIEX::CertWaitTCTSOp>(o)) {
    text_cost += 8; // ISA_OPSIZE_WAIT_TCTS
  } else if (isa<AIEX::CertMaskWrite32Op>(o)) {
    text_cost += 16; // ISA_OPSIZE_MASK_WRITE_32
  } else if (isa<AIEX::CertWrite32Op>(o)) {
    text_cost += 12; // ISA_OPSIZE_WRITE_32
  } else if (isa<AIEX::CertApplyOffset57Op>(o)) {
    text_cost += 8; // ISA_OPSIZE_APPLY_OFFSET_57
  } else if (isa<AIEX::CertNopOp>(o)) {
    text_cost += 4; // ISA_OPSIZE_NOP
  } else if (isa<AIEX::CertPreemptOp>(o)) {
    text_cost += 8; // ISA_OPSIZE_PREEMPT
  } else if (isa<AIEX::CertLoadPdiOp>(o)) {
    text_cost += 12; // ISA_OPSIZE_LOAD_PDI
    // NOLINTEND(bugprone-branch-clone)
  } else if (auto syncOp = dyn_cast<AIEX::CertUcDmaWriteDesSyncOp>(o)) {
    text_cost += 4; // ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC
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
  uint32_t text_cost = 32; // page header: `.align 16` padding plus `.eop`
  uint32_t data_cost = 0;
  found_split_point = false;
  AIE::DeviceOp deviceOp = op->getParentOfType<AIE::DeviceOp>();

  for (auto job : op.getBody().front().getOps<AIEX::CertJobOp>()) {
    // START_JOB precedes the body, so it is already spent at every candidate
    // split point inside this job; END_JOB is charged after the body below.
    text_cost += cert_job_start_cost;
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
    text_cost += cert_job_end_cost;
  }
  return text_cost + data_cost;
}

// local_barrier co-location across a page split (G-localbar).
//
// A page is a single uC scope, so all cert.local_barrier ops that share a
// local_barrier_id are participants of one barrier and must stay together on a
// single page (splitting them across an .eop hangs the firmware). We model a
// candidate split as a "cut" in the page's flat textual op order: ops before
// the cut land on the earlier page, ops at/after the cut land on the later
// page. A barrier group is "separated" iff the cut falls strictly inside the
// span of its participants' flat indices.

// Assign each non-terminator op in `page` a flat textual index (jobs in order,
// ops in order) and record, per local_barrier_id, the [min,max] flat-index span
// of its participants.
static void computeBarrierSpans(AIEX::CertPageOp page,
                                std::map<Operation *, int> &flatIndex,
                                std::map<int, std::pair<int, int>> &spans) {
  int idx = 0;
  for (auto job : page.getBody().front().getOps<AIEX::CertJobOp>()) {
    for (auto &o : job.getBody().front().getOperations()) {
      if (isa<AIE::EndOp>(o))
        continue;
      flatIndex[&o] = idx;
      if (auto bar = dyn_cast<AIEX::CertLocalBarrierOp>(o)) {
        int id = bar.getLocalBarrierId();
        auto it = spans.find(id);
        if (it == spans.end())
          spans[id] = {idx, idx};
        else {
          it->second.first = std::min(it->second.first, idx);
          it->second.second = std::max(it->second.second, idx);
        }
      }
      ++idx;
    }
  }
}

// A cut before flat index `p` (ops with index < p -> earlier page) separates a
// local_barrier group iff some participant is before and some at/after the cut.
static bool cutSeparatesBarrier(const std::map<int, std::pair<int, int>> &spans,
                                int p) {
  return llvm::any_of(spans, [p](const auto &entry) {
    auto [lo, hi] = entry.second;
    return lo < p && hi >= p;
  });
}

// Search for an interior split point (a position inside some job, with at least
// one op before it in that job and a real op at/after it) whose cut keeps every
// local_barrier group intact. Among the legal candidates pick the one whose
// earlier-page cost is closest to `split_target`. Returns false if no legal
// split point exists (the caller then leaves the page intact).
static bool findLegalSplitPoint(AIEX::CertPageOp page, uint32_t split_target,
                                const std::map<Operation *, int> &flatIndex,
                                const std::map<int, std::pair<int, int>> &spans,
                                AIEX::CertJobOp &split_job,
                                Block::iterator &split_iter) {
  AIE::DeviceOp deviceOp = page->getParentOfType<AIE::DeviceOp>();
  uint32_t text_cost = 32; // page header, mirrors estimateCost
  uint32_t data_cost = 0;
  bool found = false;
  uint32_t best_delta = 0;
  for (auto job : page.getBody().front().getOps<AIEX::CertJobOp>()) {
    text_cost += cert_job_start_cost; // mirrors estimateCost
    bool sawOpInJob = false;
    for (auto &o : job.getBody().front().getOperations()) {
      if (isa<AIE::EndOp>(o))
        continue;
      Block::iterator current(&o);
      // Candidate: cut before `o`. Only interior points are valid (at least one
      // op precedes `o` in this job), matching what the splitter can execute.
      if (sawOpInJob) {
        auto fi = flatIndex.find(&o);
        int p = fi == flatIndex.end() ? 0 : fi->second;
        if (!cutSeparatesBarrier(spans, p)) {
          uint32_t cost = text_cost + data_cost;
          uint32_t delta =
              cost > split_target ? cost - split_target : split_target - cost;
          if (!found || delta < best_delta) {
            found = true;
            best_delta = delta;
            split_job = job;
            split_iter = current;
          }
        }
      }
      updateCostForOp(o, deviceOp, text_cost, data_cost);
      sawOpInJob = true;
    }
    text_cost += cert_job_end_cost; // mirrors estimateCost
  }
  return found;
}

namespace {

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
  // `hadError` is raised when the pattern emits an error diagnostic. A rewrite
  // pattern returning failure() is ordinary control flow -- it does NOT fail
  // the pass -- so without this the "cannot split" error below would print and
  // the oversized page would still reach codegen. The pass checks the flag
  // after applyPatternsGreedily and calls signalPassFailure().
  SplitCertPageOpPattern(MLIRContext *context, bool *hadError)
      : OpRewritePattern(context), hadError(hadError) {}

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

    if (cost < split_threshold)
      return failure();

    // Over the split trigger but with nowhere to cut. A split point must fall
    // inside a job with at least one op ahead of it, so a page whose bulk sits
    // in a single one-op job offers none. Leave the page alone -- but if it is
    // also over the hard limit, it will not load, so say so instead of emitting
    // a page the firmware will reject.
    if (!found_split_point) {
      if (cost > cert_page_limit) {
        op.emitError() << "cert.page is an estimated " << cost
                       << " bytes, over the " << cert_page_limit
                       << "-byte microcontroller page limit, and offers no "
                          "split point (a split must fall inside a job, after "
                          "at least one of its ops); break the oversized job "
                          "into smaller jobs";
        *hadError = true;
      }
      return failure();
    }

    // never split a local_barrier participant set across an .eop
    // (G-localbar). If the cost-chosen split point would separate a barrier
    // group, retarget to a legal split point; if none exists the page cannot be
    // made legal, so fail the compile rather than emit something broken.
    // an implicit page (formed by formImplicitPages) groups jobs meant to
    // be cooperatively scheduled together; an explicit page is a user-authored
    // boundary. This governs the wording of the diagnostics below (the remedy
    // differs), not whether they fire.
    bool isImplicit = op->hasAttr("cert.implicit");

    std::map<Operation *, int> flatIndex;
    std::map<int, std::pair<int, int>> barrierSpans;
    computeBarrierSpans(op, flatIndex, barrierSpans);
    if (!barrierSpans.empty()) {
      auto fi = flatIndex.find(&*split_iter);
      int p = fi == flatIndex.end() ? 0 : fi->second;
      if (cutSeparatesBarrier(barrierSpans, p)) {
        if (!findLegalSplitPoint(op, cert_page_size / 2, flatIndex,
                                 barrierSpans, split_job, split_iter)) {
          // No split keeps every local_barrier group intact. Leave the
          // page whole: severing a barrier hangs the firmware, so an oversized
          // page is the better of the two. That is only survivable while the
          // page still fits, though -- past cert_page_limit neither option
          // works, and both fail silently at runtime, so diagnose and fail the
          // compile. The remedy differs by page kind: an implicit page just
          // needs a user-authored boundary in the right place, while an
          // explicit page is already where the user put it and has to be made
          // smaller or its participants regrouped. Raise `hadError` so the pass
          // actually fails -- returning failure() alone only tells the greedy
          // driver this pattern did not apply.
          if (cost > cert_page_limit) {
            if (isImplicit)
              op.emitError()
                  << "implicit cert.page is an estimated " << cost
                  << " bytes, over the " << cert_page_limit
                  << "-byte microcontroller page limit, and cannot be split "
                     "without separating a local_barrier participant set "
                     "across pages; insert an explicit cert.page boundary to "
                     "co-locate the barrier participants";
            else
              op.emitError()
                  << "cert.page is an estimated " << cost << " bytes, over the "
                  << cert_page_limit
                  << "-byte microcontroller page limit, and cannot be split "
                     "without separating a local_barrier participant set "
                     "across pages; reduce the page's contents or regroup its "
                     "local_barrier participants so that a legal split point "
                     "exists";
            *hadError = true;
          }
          return failure();
        }
      }
    }

    // splitting an implicit page turns jobs that were meant to run
    // cooperatively on one page into strictly sequential pages. Warn so the
    // user can insert an explicit boundary if the serialization is unintended.
    if (isImplicit)
      op.emitRemark()
          << "auto-split of implicit cert.page serializes cooperatively-"
             "scheduled jobs into separate sequential pages; insert an "
             "explicit "
             "cert.page boundary to control where the split occurs";

    auto loc = op.getLoc();

    // the split preserves IR (textual) order both within and across the
    // two result pages. Jobs before split_job are cloned whole onto the earlier
    // page; jobs after it onto the later page; split_job's own ops are
    // partitioned at split_iter into [begin, split_iter) -> earlier page and
    // [split_iter, end) -> later page, each cloned in order (see the
    // distribution loop below). Because nothing is reordered, every producer
    // stays before its consumer, so any dependency that survives across the new
    // .eop is backward (earlier page -> later page), which is legal (G-fwd). In
    // particular splitting a wait_tcts from its earlier uc_dma enqueues is
    // allowed: the enqueues land on the earlier page and the wait on the later
    // page (a backward dependency). split_iter is always interior to split_job
    // (estimateCost / findLegalSplitPoint guarantee at least one op precedes
    // it), so the earlier page is never empty.
    assert(split_iter != split_job.getBody().front().begin() &&
           "split_iter must be interior: at least one op stays on the earlier "
           "page, preserving producer-before-consumer order");

    // NOTE: the device-wide job-id shift below only keeps job ids unique per
    // page; job_id is a label, not an order, so it does not affect
    // emission/execution order, which is textual. A device-wide job-id renumber
    // is left for a follow-up.
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

    // propagate the implicit tag so a later re-split of either fragment
    // still knows it came from an implicit page and diagnoses consistently.
    if (isImplicit) {
      newPageOp0->setAttr("cert.implicit", rewriter.getUnitAttr());
      newPageOp1->setAttr("cert.implicit", rewriter.getUnitAttr());
    }

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

  bool *hadError;
};

// Form implicit pages: group maximal runs of contiguous top-level
// cert.job ops into a single cert.page. Only an explicit cert.page delimits a
// run; structural ops (tiles, buffers, locks, switchboxes, ...) do NOT. The
// implicit configuration job (tagged {cert.configure}) is kept isolated in its
// own page so it can be forced first on uC0 and never merged into a user
// implicit page. CertJobOp legally parents under DeviceOp, so bare device-level
// jobs are valid input here (produced by RuntimeSequenceToCertJob).
// attach_to_group(0) names uC 0, which is exactly the default (unspecified)
// group. It is therefore a placement no-op, not a page boundary: flatten it so
// its content joins the device-level group-0 stream and participates in
// implicit-page formation with adjacent default-group-0 content (rather than
// being isolated on its own page). Move each group-0 op out before the
// attach_to_group, preserving order, then erase the now-empty op.
static void inlineDefaultGroup(AIE::DeviceOp dev) {
  Block *body = dev.getBody();
  SmallVector<AIEX::CertAttachToGroupOp> zeroGroups;
  for (auto grp : body->getOps<AIEX::CertAttachToGroupOp>())
    if (grp.getGroupId() == 0)
      zeroGroups.push_back(grp);
  for (auto grp : zeroGroups) {
    SmallVector<Operation *> inner;
    for (Operation &op : grp.getBody().front())
      if (!isa<AIE::EndOp>(op))
        inner.push_back(&op);
    for (Operation *op : inner)
      op->moveBefore(grp.getOperation());
    grp.erase();
  }
}

// Group a block's maximal runs of contiguous top-level cert.job ops into
// implicit cert.page ops (only an explicit cert.page delimits; the
// {cert.configure} job stays isolated in its own page). Applied per uC stream.
static void formImplicitPagesInBlock(Block *body, OpBuilder &builder) {
  // Snapshot top-level ops; the body is mutated while iterating.
  SmallVector<Operation *> topOps;
  for (Operation &op : *body)
    topOps.push_back(&op);

  // `cert.implicit` marks a page formed from a run of *user* jobs -- a
  // boundary the author could have drawn but didn't. The splitter's diagnostics
  // are keyed off it and all advise editing page boundaries, so it must not be
  // set on a page the user cannot restructure. The compiler-synthesized
  // @configure page is exactly that case, and it is isolated by construction,
  // so the tag would buy nothing there anyway.
  auto openPage = [&](Location loc, Operation *before,
                      bool implicit) -> AIEX::CertPageOp {
    builder.setInsertionPoint(before);
    auto page = AIEX::CertPageOp::create(builder, loc);
    if (implicit)
      page->setAttr("cert.implicit", builder.getUnitAttr());
    Block *b = new Block();
    page.getBody().push_back(b);
    AIEX::CertPageOp::ensureTerminator(page.getBody(), builder, loc);
    return page;
  };

  AIEX::CertPageOp currentPage; // null => no open implicit page
  for (Operation *op : topOps) {
    if (auto job = dyn_cast<AIEX::CertJobOp>(op)) {
      if (job->hasAttr("cert.configure")) {
        // Keep the configuration job isolated in its own page. Not tagged
        // implicit: the user did not author this job and cannot add a boundary
        // inside it, so the auto-split remark would be noise on every compile
        // whose
        // config transaction is large enough to split (and it usually is).
        auto page = openPage(job.getLoc(), job, /*implicit=*/false);
        job->moveBefore(page.getBody().front().getTerminator());
        currentPage = nullptr;
        continue;
      }
      if (!currentPage)
        currentPage = openPage(job.getLoc(), job, /*implicit=*/true);
      job->moveBefore(currentPage.getBody().front().getTerminator());
    } else if (isa<AIEX::CertPageOp>(op)) {
      // An explicit page delimits the current implicit run.
      currentPage = nullptr;
    }
    // All other ops (structural / control containers) do not delimit a run.
  }
}

static void formImplicitPages(AIE::DeviceOp dev) {
  OpBuilder builder(dev.getContext());
  // uC 0 (device-level) stream. attach_to_group(0) has already been inlined
  // here by inlineDefaultGroup.
  formImplicitPagesInBlock(dev.getBody(), builder);
  // Each other uC (attach_to_group) forms implicit pages within its own stream,
  // so adjacent jobs on that uC are cooperatively co-located too.
  SmallVector<AIEX::CertAttachToGroupOp> groups;
  for (auto grp : dev.getBody()->getOps<AIEX::CertAttachToGroupOp>())
    groups.push_back(grp);
  for (auto grp : groups)
    formImplicitPagesInBlock(&grp.getBody().front(), builder);
}

// Force the implicit configuration page to be the first page on uC0:
// @configure sets up the device (DMA/lock/core config), so it must strictly
// precede all other uC0 pages regardless of where it appeared textually. The
// config job is already isolated in its own page by formImplicitPages; here we
// just hoist that page to the front of the device body (structural ops are not
// control code, so position relative to them does not matter).
static void moveConfigPageFirst(AIE::DeviceOp dev) {
  Block *body = dev.getBody();
  AIEX::CertPageOp configPage;
  for (auto page : body->getOps<AIEX::CertPageOp>()) {
    for (auto job : page.getBody().front().getOps<AIEX::CertJobOp>()) {
      if (job->hasAttr("cert.configure")) {
        configPage = page;
        break;
      }
    }
    if (configPage)
      break;
  }
  if (configPage && &body->front() != configPage.getOperation())
    configPage->moveBefore(&body->front());
}

// Lower the `placement` attribute on top-level cert.page ops to enclosing
// cert.attach_to_group ops. Placed pages (placement present and != 0)
// are moved, preserving IR order, under one cert.attach_to_group per distinct
// group id. Unplaced pages (and placement 0) stay at device level and are
// emitted as the implicit group 0. Nesting stays attach_to_group -> page -> job
// as the emitter expects.
static void lowerPagePlacementToGroups(AIE::DeviceOp dev) {
  Block *body = dev.getBody();

  // Collect placed top-level pages by resolved group id, in first-seen order.
  std::map<int32_t, SmallVector<AIEX::CertPageOp, 4>> byGroup;
  for (auto page : body->getOps<AIEX::CertPageOp>()) {
    if (auto placement = page.getPlacementAttr()) {
      auto gid = static_cast<int32_t>(placement.getInt());
      if (gid != 0)
        byGroup[gid].push_back(page);
    }
  }

  OpBuilder builder(dev.getContext());
  for (auto &entry : byGroup) {
    int32_t gid = entry.first;
    SmallVector<AIEX::CertPageOp, 4> &pages = entry.second;

    // Create the group op where the first placed page currently lives (avoids
    // assuming anything about a device terminator).
    builder.setInsertionPoint(pages.front());
    auto grp = AIEX::CertAttachToGroupOp::create(builder, dev.getLoc(), gid);
    Block *grpBlock = new Block();
    grp.getBody().push_back(grpBlock);

    // Move the placed pages into the group, in order, dropping the
    // now-redundant placement attribute (the group encodes it).
    for (AIEX::CertPageOp page : pages) {
      page->removeAttr("placement");
      page->moveBefore(grpBlock, grpBlock->end());
    }
    AIEX::CertAttachToGroupOp::ensureTerminator(grp.getBody(), builder,
                                                dev.getLoc());
  }
}

struct AIECertPagesPass
    : xilinx::AIEX::impl::AIECertPagesBase<AIECertPagesPass> {
  void runOnOperation() override {
    // Flatten attach_to_group(0) into the device-level (default group-0) stream
    // so it is a placement no-op, not a page boundary.
    inlineDefaultGroup(getOperation());

    // Form implicit pages: group contiguous top-level jobs into pages; only an
    // explicit cert.page delimits. Replaces the old one-page-per-job
    // wrapping.
    formImplicitPages(getOperation());

    // Force the @configure page to be first on uC0.
    moveConfigPageFirst(getOperation());

    // Lower cert.page placement to cert.attach_to_group BEFORE isolate
    // and split. Those rewrites create replacement pages adjacent to the
    // original (same parent block) with the no-placement builder; running the
    // placement grouping first means placed pages already live inside their
    // cert.attach_to_group, so any pages the later rewrites spawn stay in the
    // same group by position. (Doing this last silently dropped placement from
    // split/isolated pages, emitting them on group 0.)
    lowerPagePlacementToGroups(getOperation());

    // First, isolate load_pdi and preempt operations into their own job/page
    RewritePatternSet isolatePatterns(&getContext());
    isolatePatterns.insert<IsolateFullPageOpsPattern>(&getContext());
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(isolatePatterns))))
      signalPassFailure();

    // Then apply the page splitting pattern. `splitError` catches diagnostics
    // the pattern emits while declining to split (see SplitCertPageOpPattern):
    // applyPatternsGreedily reports only whether the rewrite converged, so an
    // illegal-to-split page would otherwise print an error and still compile.
    bool splitError = false;
    RewritePatternSet p1(&getContext());
    p1.insert<SplitCertPageOpPattern>(&getContext(), &splitError);
    if (failed(applyPatternsGreedily(getOperation(), std::move(p1))))
      signalPassFailure();
    if (splitError)
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
