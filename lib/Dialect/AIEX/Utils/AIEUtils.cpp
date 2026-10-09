//===- AIEUtils.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace xilinx;

std::optional<AIEX::SubviewTraceResult>
AIEX::traceSubviewToBlockArgument(Value value) {
  int64_t offsetInBytes = 0;
  Value current = value;

  // Walk through the chain of operations until we reach a block argument
  while (current) {
    // Check if we've reached a block argument
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      return SubviewTraceResult{blockArg, offsetInBytes};
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      return std::nullopt;
    }

    // Handle memref.cast (just pass through)
    if (auto castOp = dyn_cast<memref::CastOp>(defOp)) {
      current = castOp.getSource();
      continue;
    }

    // Handle memref.reinterpret_cast (validate and pass through)
    if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      auto sourceType =
          dyn_cast<MemRefType>(reinterpretOp.getSource().getType());
      if (!sourceType) {
        return std::nullopt;
      }

      // Validate that source is contiguous (all strides must be 1)
      if (auto strided = dyn_cast<StridedLayoutAttr>(sourceType.getLayout())) {
        for (int64_t stride : strided.getStrides()) {
          if (stride != 1) {
            return std::nullopt; // Non-contiguous memory, cannot safely
                                 // reinterpret
          }
        }
      }

      current = reinterpretOp.getSource();
      continue;
    }

    // Handle memref.subview. Accepts any source rank; byte-offset delta is
    // (resultOffset - sourceOffset) from the strided layouts. The result
    // slice must remain row-major contiguous so callers can treat it as
    // linear from the returned base offset.
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      auto sourceType = subviewOp.getSourceType();
      auto resultType = subviewOp.getType();

      if (llvm::any_of(subviewOp.getStaticOffsets(), ShapedType::isDynamic) ||
          llvm::any_of(subviewOp.getStaticSizes(), ShapedType::isDynamic) ||
          llvm::any_of(subviewOp.getStaticStrides(), ShapedType::isDynamic))
        return std::nullopt;

      // No skipping in source.
      if (llvm::any_of(subviewOp.getStaticStrides(),
                       [](int64_t s) { return s != 1; }))
        return std::nullopt;

      unsigned elemSizeInBits =
          sourceType.getElementType().getIntOrFloatBitWidth();
      if (elemSizeInBits % 8 != 0)
        return std::nullopt;

      llvm::SmallVector<int64_t> srcStrides, resStrides;
      int64_t srcOff, resOff;
      if (failed(sourceType.getStridesAndOffset(srcStrides, srcOff)) ||
          failed(resultType.getStridesAndOffset(resStrides, resOff)))
        return std::nullopt;
      if (srcOff == ShapedType::kDynamic || resOff == ShapedType::kDynamic)
        return std::nullopt;

      // Result must be row-major contiguous: innermost stride == 1, and each
      // outer stride equals the product of all inner sizes. (Size-1 dims are
      // free: their stride is never stepped.) Without this, e.g. a column
      // slice of a 2D row-major memref would be accepted and patched as if
      // linear.
      ArrayRef<int64_t> resShape = resultType.getShape();
      if (!resShape.empty()) {
        if (resStrides.back() != 1)
          return std::nullopt;
        uint64_t product = 1;
        for (int d = static_cast<int>(resShape.size()) - 1; d > 0; --d) {
          product *= static_cast<uint64_t>(resShape[d]);
          if (resShape[d - 1] > 1 &&
              static_cast<uint64_t>(resStrides[d - 1]) != product)
            return std::nullopt;
        }
      }

      offsetInBytes += (resOff - srcOff) * (elemSizeInBits / 8);

      current = subviewOp.getSource();
      continue;
    }

    // Handle memref.view. The fused full-ELF path materializes every DMA
    // buffer as a typed, contiguous view sliced out of a single flat byte
    // arena (a runtime-sequence block argument) at a constant byte offset:
    //   %v = memref.view %arena[%byte_off][] : memref<Nxi8> to memref<...>
    // A memref.view result always has an identity (contiguous) layout, so only
    // the byte offset needs to be accumulated before following the source.
    if (auto viewOp = dyn_cast<memref::ViewOp>(defOp)) {
      if (!viewOp.getSizes().empty())
        return std::nullopt; // dynamic result sizes unsupported
      std::optional<int64_t> byteShift =
          getConstantIntValue(viewOp.getByteShift());
      if (!byteShift)
        return std::nullopt; // non-constant byte offset
      offsetInBytes += *byteShift;
      current = viewOp.getSource();
      continue;
    }

    // Encountered an unsupported operation
    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<unsigned> AIEX::getHostBufferArgIndex(BlockArgument arg) {
  if (!isa<BaseMemRefType>(arg.getType()))
    return std::nullopt;
  unsigned index = 0;
  for (BlockArgument other : arg.getOwner()->getArguments()) {
    if (other == arg)
      return index;
    if (isa<BaseMemRefType>(other.getType()))
      index++;
  }
  return std::nullopt;
}

AIEX::BlockwriteData::BlockwriteData(AIE::DeviceOp dev, StringRef prefix)
    : dev(dev), prefix(prefix) {
  for (Operation &op : dev.getBody()->getOperations()) {
    if (auto global = dyn_cast<memref::GlobalOp>(op))
      if (auto initVal = global.getInitialValue();
          initVal && global.getConstant())
        byValue.try_emplace(*initVal, global);
    auto symName =
        op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (!symName)
      continue;
    StringRef suffix = symName.getValue();
    unsigned idx;
    if (suffix.consume_front(prefix) && !suffix.getAsInteger(10, idx) &&
        idx >= nextId)
      nextId = idx + 1;
  }
}

memref::GlobalOp AIEX::BlockwriteData::getOrCreate(OpBuilder &builder,
                                                   Location loc,
                                                   ArrayRef<uint32_t> words) {
  int64_t numWords = words.size();
  MemRefType memrefType = MemRefType::get({numWords}, builder.getI32Type());
  TensorType tensorType =
      RankedTensorType::get({numWords}, builder.getI32Type());
  auto initVal = DenseElementsAttr::get<uint32_t>(tensorType, words);

  // The attribute is uniqued and carries the shape, but not the memref's
  // layout or memory space.
  memref::GlobalOp &global = byValue[initVal];
  if (global && global.getType() == memrefType)
    return global;
  global = memref::GlobalOp::create(
      builder, loc, prefix + std::to_string(nextId++),
      builder.getStringAttr("private"), memrefType, initVal, true, nullptr);
  return global;
}

LogicalResult AIEX::emitUpdateBdAddressFromOffsetParameter(
    OpBuilder &builder, Operation *bdOp, BaseMemRefType bufType,
    uint64_t registerAddr) {
  auto idxAttr = bdOp->getAttrOfType<IntegerAttr>("offset_state_table_idx");
  assert(idxAttr && "emitUpdateBdAddressFromOffsetParameter called without "
                    "offset_state_table_idx attribute");

  uint8_t stateIdx = static_cast<uint8_t>(idxAttr.getUInt());
  uint32_t elemBytes = bufType.getElementTypeBitWidth() / 8;

  // Use func=mul with func_arg=elemBytes so the firmware computes
  // StateTable[idx] * elemBytes = byte offset, added into the BD address
  // register.
  AIEX::NpuUpdateFromScratchpadOp::create(
      builder, bdOp->getLoc(), stateIdx, AIEX::StateTableFunc::Mul,
      /*func_arg=*/elemBytes,
      /*address=*/static_cast<uint32_t>(registerAddr),
      /*buffer=*/nullptr, /*column=*/nullptr, /*row=*/nullptr);
  return success();
}

LogicalResult AIEX::emitUpdateBdLengthFromParameter(
    OpBuilder &builder, Operation *bdOp, BaseMemRefType bufType,
    int64_t lengthUnit, const AIE::AIETargetModel &targetModel, int col,
    int row, int bdId) {
  auto idxAttr = bdOp->getAttrOfType<IntegerAttr>("length_state_table_idx");
  assert(idxAttr && "emitUpdateBdLengthFromParameter called without "
                    "length_state_table_idx attribute");

  if (failed(AIE::verifyLengthParameterTile(bdOp, targetModel, col, row)))
    return failure();
  uint64_t registerAddr =
      targetModel.getDmaBdAddress(col, row, bdId) +
      4 * targetModel.getDmaBdLayout(col, row)->bufferLength.word;

  uint8_t stateIdx = static_cast<uint8_t>(idxAttr.getUInt());
  FailureOr<int64_t> unitBytes =
      AIE::getLengthUnitBytes(bdOp, lengthUnit, bufType);
  if (failed(unitBytes))
    return failure();

  // The length register counts 32-bit words, so func=mul adds n * unitBytes / 4
  // words: func_arg is unitBytes / 4 for an addr parameter (StateTable[idx]
  // holds n) and unitBytes / 16 for a core parameter (it holds n << 2).
  int64_t bytesPerStateUnit = bdOp->hasAttr("length_core_encoded") ? 16 : 4;
  AIEX::NpuUpdateFromScratchpadOp::create(
      builder, bdOp->getLoc(), stateIdx, AIEX::StateTableFunc::Mul,
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      /*func_arg=*/static_cast<uint32_t>(*unitBytes / bytesPerStateUnit),
      /*address=*/static_cast<uint32_t>(registerAddr),
      /*buffer=*/nullptr, /*column=*/nullptr, /*row=*/nullptr);
  return success();
}

void AIEX::emitScratchpadParamsFile(ModuleOp moduleOp, llvm::raw_ostream &os) {
  SmallVector<AIEX::ScratchpadParameterOp> allParams;
  moduleOp.walk([&](AIEX::ScratchpadParameterOp p) { allParams.push_back(p); });

  os << allParams.size() << "\n";
  for (auto p : allParams) {
    std::string typeStr;
    llvm::raw_string_ostream ts(typeStr);
    p.getType().print(ts);
    ts.flush();
    // --aie-lower-scratchpad-parameters assigns kind and state_table_idx to
    // every ScratchpadParameterOp unconditionally (defaulting unused
    // parameters to Core), so by the time this dump runs both are set.
    auto kind = p.getKind();
    auto stateTableIdx = p.getStateTableIdx();
    assert(kind && stateTableIdx &&
           "expected --aie-lower-scratchpad-parameters to have assigned "
           "kind/state_table_idx to every parameter");
    StringRef kindStr =
        *kind == AIEX::ScratchpadParameterKind::Addr ? "addr" : "core";
    os << p.getSymName() << " " << static_cast<unsigned>(*stateTableIdx) << " "
       << typeStr << " " << kindStr;
    auto minValue = p.getMinValue();
    auto maxValue = p.getMaxValue();
    if (minValue && maxValue)
      os << " " << static_cast<int32_t>(*minValue) << " "
         << static_cast<int32_t>(*maxValue);
    else
      os << " - -";
    os << "\n";
  }

  SmallVector<std::pair<StringRef, AIEX::JointBoundAttr>> jointBounds;
  for (auto p : allParams)
    if (ArrayAttr bounds = p.getJointBoundsAttr())
      for (auto bound : bounds.getAsRange<AIEX::JointBoundAttr>())
        jointBounds.push_back({p.getSymName(), bound});
  os << jointBounds.size() << "\n";
  for (auto [length, bound] : jointBounds)
    os << length << " " << bound.getOffset().getValue() << " "
       << bound.getLengthStep() << " " << bound.getMax() << "\n";
}

bool AIEX::getReachableConfigures(
    Value task, SmallVectorImpl<DMAConfigureTaskOp> &configures) {
  bool complete = true;
  llvm::SmallPtrSet<Value, 8> seen;
  SmallVector<Value> worklist{task};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!v || !seen.insert(v).second)
      continue;
    if (auto cfg = v.getDefiningOp<DMAConfigureTaskOp>()) {
      configures.push_back(cfg);
      continue;
    }

    Operation *regionBranchOp;
    if (auto res = dyn_cast<OpResult>(v))
      regionBranchOp = res.getOwner();
    else
      regionBranchOp = cast<BlockArgument>(v).getOwner()->getParentOp();
    auto rbi = dyn_cast_or_null<RegionBranchOpInterface>(regionBranchOp);
    if (!rbi) {
      complete = false;
      continue;
    }

    RegionBranchInverseSuccessorMapping mapping;
    rbi.getSuccessorInputOperandMapping(mapping);
    auto operands = mapping.lookup(v);
    if (operands.empty())
      complete = false;
    for (OpOperand *operand : operands)
      worklist.push_back(operand->get());
  }
  return complete;
}

AIEX::DMAConfigureTaskOp AIEX::getUniqueReachableConfigure(Value task) {
  SmallVector<DMAConfigureTaskOp> configures;
  if (!getReachableConfigures(task, configures) || configures.size() != 1)
    return nullptr;
  return configures.front();
}
