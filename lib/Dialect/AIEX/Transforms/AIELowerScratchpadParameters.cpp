//===- AIELowerScratchpadParameters.cpp - Lower scratchpad parameter ops --===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

#include <numeric>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIELOWERSCRATCHPADPARAMETERS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

namespace {

/// Returns true iff the parameter-sync preamble (use_lock acquire) should be
/// emitted for this core. Reads the explicit attribute if present; otherwise
/// defaults to true iff any 'aiex.read_scratchpad_parameter' op is present in
/// the core body.
static bool shouldEmitParameterSyncPreamble(CoreOp coreOp) {
  if (auto attr = coreOp.getEmitParameterSyncPreambleAttr()) {
    return attr.getValue();
  }
  bool found = false;
  coreOp.getBody().walk([&](ReadScratchpadParameterOp) { found = true; });
  return found;
}

// Warned here, not at NPU lowering, which inlines a sequence at every call
// site and so would warn once per call.
static void warnIfRuntimeOffsetMayRound(Operation *op, Type bufType) {
  uint32_t elemBytes =
      llvm::cast<BaseMemRefType>(bufType).getElementTypeBitWidth() / 8;
  if (elemBytes == 0 || elemBytes % 4 == 0) {
    return;
  }
  mlir::emitWarning(op->getLoc())
      << "runtime offset parameter on a " << (elemBytes * 8)
      << "-bit element type: the firmware masks the BD address register with "
         "0xFFFFFFFC, so a value that is not a multiple of "
      << (4 / std::gcd(4u, elemBytes))
      << " elements is silently rounded down to the 4-byte boundary below "
         "instead of being rejected";
}

/// A DMA with runtime offset `o` and length `l` touches elements
/// [offset + o, last + o + lengthStep * l].
struct TransferExtent {
  int64_t bufferElems;
  int64_t offset;
  int64_t last;
  int64_t lengthStep;
};

/// `sizes`/`strides` are innermost-first and empty for a linear BD.
static std::optional<TransferExtent>
getTransferExtent(Type bufType, int64_t offset, int64_t lenElems,
                  SmallVector<int64_t> sizes, SmallVector<int64_t> strides,
                  std::optional<int64_t> lengthUnit) {
  auto buffer = llvm::cast<BaseMemRefType>(bufType);
  if (!buffer.hasStaticShape())
    return std::nullopt;
  TransferExtent extent{buffer.getNumElements(), offset, offset, 0};
  if (!lengthUnit) {
    if (sizes.empty())
      extent.last += lenElems - 1;
    for (auto [size, stride] : llvm::zip(sizes, strides))
      extent.last += (size - 1) * stride;
    return extent;
  }
  bool linear = sizes.empty();
  sizes.resize(std::max<size_t>(sizes.size(), 3), 1);
  strides.resize(sizes.size(), 0);
  for (size_t i = 3; i < sizes.size(); ++i)
    extent.last += (sizes[i] - 1) * strides[i];
  if (linear || AIEX::isContiguousTransfer(sizes, strides)) {
    extent.last += lenElems - 1;
    extent.lengthStep = *lengthUnit;
    return extent;
  }
  AIE::placeRuntimeLengthDimension(sizes, strides);
  int64_t rowElems = sizes[0] * sizes[1];
  // A static length of 0 counts -1 steps, so that the first added unit's
  // steps start at 0.
  extent.last += (sizes[0] - 1) * strides[0] + (sizes[1] - 1) * strides[1] +
                 (llvm::divideCeilSigned(lenElems, rowElems) - 1) * strides[2];
  extent.lengthStep = *lengthUnit / rowElems * strides[2];
  return extent;
}

static std::optional<TransferExtent>
getTransferExtent(NpuDmaMemcpyNdOp op, std::optional<int64_t> lengthUnit) {
  std::optional<SmallVector<int64_t>> offsets =
      getConstantIntValues(op.getMixedOffsets());
  std::optional<SmallVector<int64_t>> sizes =
      getConstantIntValues(op.getMixedSizes());
  std::optional<SmallVector<int64_t>> strides =
      getConstantIntValues(op.getMixedStrides());
  int64_t elemBits = op.getElementTypeBitwidth();
  if (!offsets || !sizes || !strides || elemBits % 8 != 0)
    return std::nullopt;
  std::reverse(sizes->begin(), sizes->end());
  std::reverse(strides->begin(), strides->end());
  return getTransferExtent(
      op.getMemref().getType(), op.getOffsetInBytes() / (elemBits / 8),
      (*sizes)[0] * (*sizes)[1] * (*sizes)[2], *sizes, *strides, lengthUnit);
}

/// The range of a BD's runtime offset, if it is known to stay within the
/// buffer. An offset computed from a runtime-sequence loop's induction variable
/// is only folded once the loop is unrolled, after this pass.
static std::optional<std::pair<int64_t, int64_t>>
getOffsetRange(Value offset, int64_t bufferElems, DataFlowSolver *solver) {
  if (!solver)
    return std::nullopt;
  auto *state = solver->lookupState<dataflow::IntegerValueRangeLattice>(offset);
  if (!state || state->getValue().isUninitialized())
    return std::nullopt;
  const ConstantIntRanges &range = state->getValue().getValue();
  // The offset may be wider than 64 bits; an extremum that does not fit is
  // outside the buffer anyway.
  std::optional<int64_t> lo = range.smin().trySExtValue();
  std::optional<int64_t> hi = range.smax().trySExtValue();
  if (!lo || !hi || *lo < 0 || *hi >= bufferElems)
    return std::nullopt;
  return std::make_pair(*lo, *hi);
}

/// With a runtime offset, the extent starts at its smallest value and ends
/// at its largest.
static std::optional<TransferExtent>
getTransferExtent(AIE::DMABDOp op, std::optional<int64_t> lengthUnit,
                  DataFlowSolver *solver) {
  auto buffer = llvm::cast<BaseMemRefType>(op.getBuffer().getType());
  std::optional<int64_t> offset = 0, offsetMin = 0;
  if (op.hasOffset()) {
    offset = offsetMin = op.getConstantOffset();
    if (!offset && buffer.hasStaticShape()) {
      if (auto range =
              getOffsetRange(op.getOffset(), buffer.getNumElements(), solver)) {
        offsetMin = range->first;
        offset = range->second;
      }
    }
  }
  std::optional<int64_t> len;
  if (op.hasLen())
    len = op.getConstantLen();
  else if (buffer.hasStaticShape())
    len = buffer.getNumElements();
  std::optional<SmallVector<int64_t>> sizes =
      getConstantIntValues(op.getMixedSizes());
  std::optional<SmallVector<int64_t>> strides =
      getConstantIntValues(op.getMixedStrides());
  if (!offset || !len || !sizes || !strides)
    return std::nullopt;
  std::reverse(sizes->begin(), sizes->end());
  std::reverse(strides->begin(), strides->end());
  std::optional<TransferExtent> extent =
      getTransferExtent(buffer, *offset, *len, *sizes, *strides, lengthUnit);
  if (extent)
    extent->offset = *offsetMin;
  return extent;
}

/// The largest length parameter value that keeps the shim BD's Buffer_Length,
/// the static length plus that many units in 32-bit words, within its field.
static int64_t getMaxLengthSteps(Operation *op, int64_t lenBytes,
                                 int64_t unitBytes) {
  int64_t maxWords =
      getTargetModel(op).getDmaBdMaxLen(AIETileType::ShimNOCTile);
  return (maxWords - lenBytes / 4) / (unitBytes / 4);
}

static std::optional<int64_t> getMaxLengthSteps(NpuDmaMemcpyNdOp op) {
  std::optional<SmallVector<int64_t>> sizes =
      getConstantIntValues(op.getMixedSizes());
  std::optional<int64_t> lengthUnit = op.getLengthUnit();
  if (!sizes || !lengthUnit)
    return std::nullopt;
  int64_t elemBytes = op.getElementTypeBitwidth() / 8;
  return getMaxLengthSteps(op,
                           (*sizes)[1] * (*sizes)[2] * (*sizes)[3] * elemBytes,
                           *lengthUnit * elemBytes);
}

static std::optional<int64_t> getMaxLengthSteps(AIE::DMABDOp op) {
  std::optional<int64_t> lengthUnit = op.getLengthUnit();
  if (!lengthUnit)
    return std::nullopt;
  return getMaxLengthSteps(op, op.getLenInBytes(),
                           *lengthUnit * op.getBufferElementTypeWidthInBytes());
}

/// Returns true iff the parameter-sync preamble (create_scratchpad + set_lock)
/// should be emitted into this sequence. Reads the explicit attribute if
/// present; otherwise defaults to true iff the parent device uses any
/// scratchpad parameter, from a core or a DMA BD, and the runtime sequence does
/// not already contain a `aiex.sync_scratchpad_parameters_from_host` marker.
static bool shouldEmitParameterSyncPreamble(RuntimeSequenceOp seqOp) {
  if (auto attr = seqOp.getEmitParameterSyncPreambleAttr()) {
    return attr.getValue();
  }
  bool hasManualSync = false;
  seqOp.walk([&](SyncScratchpadParametersFromHostOp) { hasManualSync = true; });
  if (hasManualSync) {
    return false;
  }
  auto device = seqOp->getParentOfType<DeviceOp>();
  if (!device) {
    return false;
  }
  bool found = false;
  device.walk([&](Operation *op) {
    if (found) {
      return;
    }
    if (llvm::isa<ReadScratchpadParameterOp>(op) ||
        op->hasAttr("offset_parameter") ||
        op->hasAttr("offset_state_table_idx") ||
        op->hasAttr("length_parameter") ||
        op->hasAttr("length_state_table_idx")) {
      found = true;
    }
  });
  return found;
}

struct AIELowerScratchpadParametersPass
    : public xilinx::AIEX::impl::AIELowerScratchpadParametersBase<
          AIELowerScratchpadParametersPass> {
  using AIELowerScratchpadParametersBase::AIELowerScratchpadParametersBase;

  // For each read_scratchpad_parameter of a unique parameter, create a 2xi32
  // buffer and store a reference to it on the ReadScratchpadParameterOp as
  // the `buffer` attribute.
  void allocateBuffers(DeviceOp device, OpBuilder &builder) {
    MLIRContext *ctx = device.getContext();
    unsigned uniquingCounter = 0;

    DenseMap<std::pair<StringRef, Operation *>, BufferOp> seen;

    device.walk([&](ReadScratchpadParameterOp readOp) {
      auto coreOp = readOp->getParentOfType<CoreOp>();
      TileOp tile = coreOp.getTileOp();
      StringRef paramName = readOp.getParameter();
      auto key = std::make_pair(paramName, tile.getOperation());

      if (seen.count(key)) {
        readOp.setBufferAttr(
            FlatSymbolRefAttr::get(ctx, *seen[key].getSymName()));
        return;
      }

      builder.setInsertionPointAfter(tile);
      // Buffer must be 8 bytes: update_from_scratchpad always writes a 48-bit
      // value across two 32-bit registers at [RegOff] and [RegOff+4]. The
      // firmware masks Reg[0] with 0xFFFFFFFC (lower 2 bits forced to 0)
      // because it was designed for 4-byte-aligned DMA BD addresses. The host
      // library's `ParameterScratchpad::write` left-shifts by 2 before
      // writing to the scratchpad, and the core right-shifts by 2 after
      // loading. Note that this limits effective parameter values to 30
      // bits.
      auto bufType = MemRefType::get({2}, builder.getI32Type());
      std::string prefix =
          ("__param_" + paramName + "_" + std::to_string(tile.getCol()) + "_" +
           std::to_string(tile.getRow()) + "_")
              .str();
      std::string bufName =
          AIE::generateUniqueSymbolName(device, prefix, uniquingCounter);
      auto buf =
          BufferOp::create(builder, readOp.getLoc(), bufType, tile,
                           builder.getStringAttr(bufName), /*address=*/nullptr,
                           /*initial_value=*/nullptr, /*mem_bank=*/nullptr);
      seen[key] = buf;

      readOp.setBufferAttr(
          FlatSymbolRefAttr::get(ctx, *seen[key].getSymName()));
    });
  }

  // Lower each read_scratchpad_parameter to: load from buffer[0], shift right
  // by 2, and cast to the result type.  The buffer to use comes from the
  // `buffer` attribute set by allocateBuffers().
  void lowerReadParameters(DeviceOp device, OpBuilder &builder) {
    SmallVector<ReadScratchpadParameterOp> readOps;
    device.walk([&](ReadScratchpadParameterOp op) { readOps.push_back(op); });

    for (auto readOp : readOps) {
      FlatSymbolRefAttr bufRef = readOp.getBufferAttr();
      auto buf = AIE::lookupNamedOpIn<BufferOp>(device, bufRef.getAttr());

      builder.setInsertionPoint(readOp);
      Value c0 = arith::ConstantIndexOp::create(builder, readOp.getLoc(), 0);
      Value raw = memref::LoadOp::create(builder, readOp.getLoc(), buf, c0);
      Value c2 = arith::ConstantOp::create(builder, readOp.getLoc(),
                                           builder.getI32IntegerAttr(2));
      Value decoded = arith::ShRUIOp::create(builder, readOp.getLoc(), raw, c2);

      Type resultType = readOp.getResult().getType();
      Value result = decoded;
      if (resultType.isInteger() && resultType != builder.getI32Type()) {
        result = arith::TruncIOp::create(builder, readOp.getLoc(), resultType,
                                         decoded);
      } else if (resultType.isBF16()) {
        Value masked = arith::TruncIOp::create(builder, readOp.getLoc(),
                                               builder.getI16Type(), decoded);
        result = arith::BitcastOp::create(builder, readOp.getLoc(), resultType,
                                          masked);
      }

      readOp.getResult().replaceAllUsesWith(result);
      readOp.erase();
    }
  }

  // For each core in `device` with shouldEmitParameterSyncPreamble()==true,
  // create an aie.lock (no lockID; AIEAssignLockIDs will assign one later) and
  // insert aie.use_lock(Acquire, 1) at the top of the core body.  Returns the
  // list of lock values created (one per qualifying core, in walk order).
  SmallVector<Value> emitCorePreambles(DeviceOp device, OpBuilder &builder) {
    SmallVector<Value> syncLocks;
    device.walk([&](CoreOp coreOp) {
      if (!shouldEmitParameterSyncPreamble(coreOp)) {
        return;
      }

      TileOp tile = coreOp.getTileOp();
      builder.setInsertionPointAfter(tile);
      auto lockOp = LockOp::create(
          builder, coreOp.getLoc(), builder.getIndexType(), tile.getResult(),
          /*lockID=*/IntegerAttr{}, builder.getI32IntegerAttr(0),
          /*sym_name=*/StringAttr{});
      syncLocks.push_back(lockOp.getResult());

      Block &bodyBlock = coreOp.getBody().front();
      builder.setInsertionPointToStart(&bodyBlock);
      UseLockOp::create(builder, coreOp.getLoc(), lockOp.getResult(),
                        LockAction::Acquire, 1);

      // Mark as done so the pass is idempotent.
      coreOp.setEmitParameterSyncPreambleAttr(builder.getBoolAttr(false));
    });
    return syncLocks;
  }

  // For each runtime sequence in `device` with shouldEmitParameterSyncPreamble
  // ()==true, insert a marker SyncScratchpadParametersFromHostOp after the
  // last top-level load_pdi of the sequence body, or at its start if there is
  // none.
  LogicalResult emitSequencePreambles(DeviceOp device, OpBuilder &builder) {
    WalkResult result = device.walk([&](RuntimeSequenceOp seqOp) {
      if (!shouldEmitParameterSyncPreamble(seqOp)) {
        return WalkResult::advance();
      }

      Region &region = seqOp.getBody();
      if (region.empty())
        region.emplaceBlock();
      Block &body = region.front();
      builder.setInsertionPointToStart(&body);
      // Loading a PDI resets the core buffers and locks the sync writes.
      for (NpuLoadPdiOp loadPdi : body.getOps<NpuLoadPdiOp>())
        builder.setInsertionPointAfter(loadPdi);
      for (Operation &op :
           llvm::make_range(body.begin(), builder.getInsertionPoint())) {
        WalkResult early = op.walk([&](Operation *inner) {
          auto dmaOp = dyn_cast<NpuDmaMemcpyNdOp>(inner);
          auto bdOp = dyn_cast<AIE::DMABDOp>(inner);
          if ((dmaOp && (dmaOp.getOffsetStateTableIdxAttr() ||
                         dmaOp.getLengthStateTableIdxAttr())) ||
              (bdOp && (bdOp.getOffsetStateTableIdxAttr() ||
                        bdOp.getLengthStateTableIdxAttr()))) {
            inner->emitOpError(
                "reads a scratchpad parameter before the last "
                "aiex.npu.load_pdi of its runtime sequence, which is where "
                "the scratchpad is created. Load the PDIs first, or place "
                "aiex.sync_scratchpad_parameters_from_host before this op.");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (early.wasInterrupted())
          return WalkResult::interrupt();
      }
      SyncScratchpadParametersFromHostOp::create(builder, seqOp.getLoc());

      // Mark as done so the pass is idempotent.
      seqOp.setEmitParameterSyncPreambleAttr(builder.getBoolAttr(false));
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  // Lower a sync_scratchpad_parameters_from_host marker op in place to its
  // component ops, using the parameter information for the device that contains
  // it:
  //   1. npu.create_scratchpad(scratchpadSize)
  //   2. For each (state_idx, buffer_ref) in paramEntries: npu.write32(0) +
  //      npu.update_from_scratchpad
  //   3. set_lock(lock, 1) for each lock in syncLocks
  void lowerSyncParametersOp(
      SyncScratchpadParametersFromHostOp syncOp, uint32_t scratchpadSize,
      const SmallVector<std::pair<uint8_t, FlatSymbolRefAttr>> &paramEntries,
      const SmallVector<Value> &syncLocks) {
    OpBuilder builder(syncOp);
    Location loc = syncOp.getLoc();

    // A module with no parameters has nothing to stage, and size 0 is invalid.
    if (scratchpadSize > 0)
      NpuCreateScratchpadOp::create(builder, loc, scratchpadSize);

    for (auto &[stateIdx, bufRef] : paramEntries) {
      // Zero the destination before the additive UpdateScratchpad.
      NpuWrite32Op::create(builder, loc,
                           /*address=*/createConstantI32(builder, loc, 0),
                           /*value=*/createConstantI32(builder, loc, 0), bufRef,
                           /*column=*/nullptr, /*row=*/nullptr);
      NpuUpdateFromScratchpadOp::create(
          builder, loc, stateIdx, StateTableFunc::Incr,
          /*func_arg=*/static_cast<uint32_t>(0),
          /*address=*/static_cast<uint32_t>(0), bufRef,
          /*column=*/nullptr, /*row=*/nullptr);
    }

    for (Value lock : syncLocks) {
      SetLockOp::create(builder, loc, lock, createConstantI32(builder, loc, 1));
    }

    syncOp.erase();
  }

  // Lower every sync_scratchpad_parameters_from_host op in `device` to its
  // component ops in place, using parameter info for this device only.
  void lowerSyncParametersOps(
      DeviceOp device, uint32_t scratchpadSize,
      const SmallVector<std::pair<uint8_t, FlatSymbolRefAttr>> &paramEntries,
      const SmallVector<Value> &syncLocks) {
    SmallVector<SyncScratchpadParametersFromHostOp> syncOps;
    device.walk(
        [&](SyncScratchpadParametersFromHostOp op) { syncOps.push_back(op); });
    for (SyncScratchpadParametersFromHostOp syncOp : syncOps) {
      lowerSyncParametersOp(syncOp, scratchpadSize, paramEntries, syncLocks);
    }
  }

  // Emit a single params.txt for the whole module to `outputParamsFile`.
  LogicalResult emitParamsFile() {
    if (outputParamsFile.empty())
      return success();

    std::error_code ec;
    llvm::raw_fd_ostream out(outputParamsFile, ec);
    if (ec) {
      return emitError(UnknownLoc::get(&getContext()),
                       "failed to open params output file '")
             << outputParamsFile << "': " << ec.message();
    }

    emitScratchpadParamsFile(getOperation(), out);
    return success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());

    // Ranges are only needed for a parameterized BD with a runtime offset.
    std::unique_ptr<DataFlowSolver> solver;
    WalkResult needRanges = moduleOp.walk([](AIE::DMABDOp bd) {
      bool param = bd.getLengthParameterAttr() || bd.getOffsetParameterAttr();
      return param && bd.getOffset() && !bd.getConstantOffset()
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (needRanges.wasInterrupted()) {
      solver = std::make_unique<DataFlowSolver>();
      solver->load<dataflow::DeadCodeAnalysis>();
      solver->load<dataflow::SparseConstantPropagation>();
      solver->load<dataflow::IntegerRangeAnalysis>();
      if (failed(solver->initializeAndRun(moduleOp)))
        solver.reset();
    }

    // Step 1: collect every parameter in the module.
    SmallVector<ScratchpadParameterOp> allParams;
    moduleOp.walk([&](ScratchpadParameterOp p) { allParams.push_back(p); });

    if (allParams.size() > 32) {
      InFlightDiagnostic diag =
          moduleOp.emitError("Module declares ")
          << allParams.size()
          << " parameters but the scratchpad supports at most 32. The "
             "scratchpad is a single hardware resource shared by all PDIs "
             "loaded by a runtime sequence.";
      for (auto p : allParams)
        diag.attachNote(p.getLoc()) << "parameter '" << p.getSymName() << "'";
      return signalPassFailure();
    }

    // Step 2: determine each parameter's kind from its usage, erroring on
    // mixed use.  A parameter is "core" if any aiex.read_scratchpad_parameter
    // references it; "addr" if any DMA op references it via offset_parameter.
    // If both, emit an error. A DMA length_parameter works with either kind,
    // so it takes the kind of the parameter's other uses, and "addr" if it is
    // the only use; the lowering compensates in the update's multiplier.
    DenseMap<StringRef, bool> usedAsCore;
    DenseMap<StringRef, bool> usedAsLength;
    DenseMap<StringRef, bool> usedAsAddr;
    moduleOp.walk([&](ReadScratchpadParameterOp op) {
      usedAsCore[op.getParameter()] = true;
    });
    auto markDmaUses = [&](FlatSymbolRefAttr offsetRef,
                           FlatSymbolRefAttr lengthRef) {
      if (offsetRef)
        usedAsAddr[offsetRef.getValue()] = true;
      if (lengthRef)
        usedAsLength[lengthRef.getValue()] = true;
    };
    moduleOp.walk([&](NpuDmaMemcpyNdOp op) {
      markDmaUses(op.getOffsetParameterAttr(), op.getLengthParameterAttr());
    });
    moduleOp.walk([&](AIE::DMABDOp op) {
      markDmaUses(op.getOffsetParameterAttr(), op.getLengthParameterAttr());
    });

    for (auto p : allParams) {
      StringRef name = p.getSymName();
      bool core = usedAsCore.lookup(name);
      bool length = usedAsLength.lookup(name);
      bool addr = usedAsAddr.lookup(name);
      if (core && addr) {
        p.emitError("parameter '")
            << name
            << "' is used both as an aiex.read_scratchpad_parameter source "
               "(core) and as a DMA offset_parameter (addr); a parameter "
               "must have a single kind";
        return signalPassFailure();
      }
      p.setKindAttr(ScratchpadParameterKindAttr::get(
          &getContext(), addr || (length && !core)
                             ? ScratchpadParameterKind::Addr
                             : ScratchpadParameterKind::Core));
    }

    // Step 3: assign global state_table_idx in walk order, 0..N-1.
    for (auto [i, p] : llvm::enumerate(allParams)) {
      p.setStateTableIdxAttr(builder.getIntegerAttr(
          builder.getIntegerType(8, /*isSigned=*/false), i));
    }

    // Step 3b: rewrite DMA `offset_parameter` / `length_parameter` symbol
    // references to plain `offset_state_table_idx` / `length_state_table_idx`
    // integer attributes, so downstream `aie`-dialect passes do not need to
    // resolve `aiex.scratchpad_parameter` symbols.
    auto rewriteParam =
        [&](Operation *op, FlatSymbolRefAttr ref, StringRef attrName,
            StringRef idxAttrName) -> FailureOr<ScratchpadParameterOp> {
      auto paramOp =
          moduleOp.lookupSymbol<ScratchpadParameterOp>(ref.getAttr());
      if (!paramOp) {
        op->emitOpError() << attrName << " '" << ref.getValue()
                          << "' not found. Declare it at module scope with "
                             "aiex.scratchpad_parameter.";
        return failure();
      }
      if (!paramOp.getType().isInteger(32)) {
        auto err = op->emitOpError()
                   << attrName << " '" << ref.getValue()
                   << "' must have type i32, got " << paramOp.getType() << ".";
        err.attachNote(paramOp.getLoc()) << "Parameter declared here.";
        return failure();
      }
      uint8_t stateIdx =
          static_cast<uint8_t>(paramOp.getStateTableIdx().value());
      op->setAttr(idxAttrName,
                  builder.getIntegerAttr(
                      builder.getIntegerType(8, /*isSigned=*/false), stateIdx));
      op->removeAttr(attrName);
      return paramOp;
    };
    // The values of each DMA offset or length parameter that keep every
    // transfer using it within its buffer, judged with the transfer's other
    // parameter at 0.
    DenseMap<StringRef, std::pair<int64_t, int64_t>> bounds;
    // (length, offset, length step) -> max of offset + length step * length.
    llvm::MapVector<std::tuple<StringRef, StringRef, int64_t>, int64_t>
        jointBounds;
    auto rewriteParams =
        [&](Operation *op, FlatSymbolRefAttr offsetRef,
            FlatSymbolRefAttr lengthRef, Type bufType,
            std::optional<TransferExtent> extent,
            std::optional<int64_t> maxLengthSteps) -> LogicalResult {
      bool joint = offsetRef == lengthRef;
      int64_t room = extent ? extent->bufferElems - 1 - extent->last : 0;
      if (offsetRef) {
        if (failed(rewriteParam(op, offsetRef, "offset_parameter",
                                "offset_state_table_idx")))
          return failure();
        warnIfRuntimeOffsetMayRound(op, bufType);
        auto &[lo, hi] =
            bounds.try_emplace(offsetRef.getValue(), INT32_MIN, INT32_MAX)
                .first->second;
        if (extent) {
          lo = std::max(lo, -extent->offset);
          hi = std::min(hi, llvm::divideFloorSigned(
                                room, 1 + (joint ? extent->lengthStep : 0)));
        }
      }
      if (lengthRef) {
        FailureOr<ScratchpadParameterOp> lengthParam = rewriteParam(
            op, lengthRef, "length_parameter", "length_state_table_idx");
        if (failed(lengthParam))
          return failure();
        auto &[lo, hi] =
            bounds.try_emplace(lengthRef.getValue(), INT32_MIN, INT32_MAX)
                .first->second;
        lo = std::max<int64_t>(lo, 0);
        if (maxLengthSteps)
          hi = std::min(hi, *maxLengthSteps);
        if (lengthParam->getKind() == ScratchpadParameterKind::Core) {
          op->setAttr("length_core_encoded", builder.getUnitAttr());
          hi = std::min<int64_t>(hi, (1 << 30) - 1);
        }
        if (extent && !joint && extent->lengthStep > 0)
          hi = std::min(hi, llvm::divideFloorSigned(room, extent->lengthStep));
        if (extent && offsetRef && !joint && extent->lengthStep > 0) {
          auto [it, inserted] = jointBounds.try_emplace(
              {lengthRef.getValue(), offsetRef.getValue(), extent->lengthStep},
              room);
          it->second = std::min(it->second, room);
        }
      }
      for (FlatSymbolRefAttr ref : {offsetRef, lengthRef}) {
        if (!ref)
          continue;
        auto [lo, hi] = bounds.lookup(ref.getValue());
        if (lo > hi)
          return op->emitOpError("no value of parameter '")
                 << ref.getValue()
                 << "' keeps every transfer using it within its buffer";
      }
      return success();
    };
    WalkResult rewriteResult = moduleOp.walk([&](Operation *op) {
      if (auto dmaOp = dyn_cast<NpuDmaMemcpyNdOp>(op)) {
        FlatSymbolRefAttr offsetRef = dmaOp.getOffsetParameterAttr();
        FlatSymbolRefAttr lengthRef = dmaOp.getLengthParameterAttr();
        if (!offsetRef && !lengthRef)
          return WalkResult::advance();
        if (failed(rewriteParams(
                op, offsetRef, lengthRef, dmaOp.getMemref().getType(),
                getTransferExtent(dmaOp, lengthRef ? dmaOp.getLengthUnit()
                                                   : std::nullopt),
                getMaxLengthSteps(dmaOp))))
          return WalkResult::interrupt();
      } else if (auto bdOp = dyn_cast<AIE::DMABDOp>(op)) {
        FlatSymbolRefAttr offsetRef = bdOp.getOffsetParameterAttr();
        FlatSymbolRefAttr lengthRef = bdOp.getLengthParameterAttr();
        if (!offsetRef && !lengthRef)
          return WalkResult::advance();
        std::optional<int64_t> lengthUnit;
        if (lengthRef)
          lengthUnit = bdOp.getLengthUnit();
        if (failed(rewriteParams(
                op, offsetRef, lengthRef, bdOp.getBuffer().getType(),
                getTransferExtent(bdOp, lengthUnit, solver.get()),
                getMaxLengthSteps(bdOp))))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (rewriteResult.wasInterrupted()) {
      return signalPassFailure();
    }
    for (ScratchpadParameterOp p : allParams) {
      auto it = bounds.find(p.getSymName());
      if (it == bounds.end())
        continue;
      p.setMinValueAttr(builder.getI32IntegerAttr(it->second.first));
      p.setMaxValueAttr(builder.getI32IntegerAttr(it->second.second));
    }
    llvm::MapVector<StringRef, SmallVector<Attribute>> jointBoundAttrs;
    for (auto [key, max] : jointBounds) {
      auto [length, offset, step] = key;
      jointBoundAttrs[length].push_back(JointBoundAttr::get(
          &getContext(), FlatSymbolRefAttr::get(&getContext(), offset), step,
          max));
    }
    for (ScratchpadParameterOp p : allParams) {
      auto *it = jointBoundAttrs.find(p.getSymName());
      if (it != jointBoundAttrs.end())
        p.setJointBoundsAttr(builder.getArrayAttr(it->second));
    }

    // Step 4: per-device lowering.
    SmallVector<DeviceOp> devices;
    moduleOp.walk([&](DeviceOp d) { devices.push_back(d); });
    for (auto d : devices) {
      allocateBuffers(d, builder);

      // Collect unique (stateIdx, bufferRef) pairs for core-kind parameters in
      // this device.  Buffer attrs are set by allocateBuffers() above.
      SmallVector<std::pair<uint8_t, FlatSymbolRefAttr>> paramEntries;
      DenseSet<StringRef> seenBufs;
      d.walk([&](ReadScratchpadParameterOp readOp) {
        FlatSymbolRefAttr bufRef = readOp.getBufferAttr();
        if (!bufRef || !seenBufs.insert(bufRef.getValue()).second) {
          return;
        }
        auto paramOp =
            moduleOp.lookupSymbol<ScratchpadParameterOp>(readOp.getParameter());
        uint8_t stateIdx =
            static_cast<uint8_t>(paramOp.getStateTableIdx().value());
        paramEntries.push_back({stateIdx, bufRef});
      });

      // Emit lock + use_lock preambles in cores, then insert a marker
      // SyncScratchpadParametersFromHostOp at the start of each qualifying
      // runtime sequence.  Immediately lower every
      // sync_scratchpad_parameters_from_host op in the device (both the
      // preamble-inserted ones and any user-written ones) using this device's
      // parameter info.
      SmallVector<Value> syncLocks = emitCorePreambles(d, builder);
      if (failed(emitSequencePreambles(d, builder)))
        return signalPassFailure();
      uint32_t scratchpadSize = static_cast<uint32_t>(allParams.size() * 4);
      lowerSyncParametersOps(d, scratchpadSize, paramEntries, syncLocks);

      lowerReadParameters(d, builder);
    }

    // Step 5: emit the single params.txt for the module.
    if (failed(emitParamsFile())) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIELowerScratchpadParametersPass() {
  return std::make_unique<AIELowerScratchpadParametersPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIELowerScratchpadParametersPass(
    AIELowerScratchpadParametersOptions options) {
  return std::make_unique<AIELowerScratchpadParametersPass>(std::move(options));
}
