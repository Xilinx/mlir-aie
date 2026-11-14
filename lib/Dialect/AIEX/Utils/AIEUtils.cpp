//===- AIEUtils.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <set>

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

unsigned AIEX::getHostBufferCount(Block &block) {
  return llvm::count_if(block.getArguments(), [](BlockArgument arg) {
    return isa<BaseMemRefType>(arg.getType());
  });
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

uint64_t AIEX::getBdAddressUnit(const AIE::AIETargetModel &targetModel, int col,
                                int row) {
  return targetModel.isShimNOCTile(col, row)
             ? 1
             : targetModel.getAddressGenGranularity() / 8;
}

LogicalResult AIEX::emitUpdateBdAddressFromOffsetParameter(
    OpBuilder &builder, Operation *bdOp, BaseMemRefType bufType,
    uint64_t registerAddr, const AIE::AIETargetModel &targetModel, int col,
    int row, std::optional<uint64_t> staticByteAddress,
    OpFoldResult lengthField) {
  auto idxAttr = bdOp->getAttrOfType<IntegerAttr>("offset_state_table_idx");
  assert(idxAttr && "emitUpdateBdAddressFromOffsetParameter called without "
                    "offset_state_table_idx attribute");

  uint8_t stateIdx = static_cast<uint8_t>(idxAttr.getUInt());
  uint32_t elemBytes = bufType.getElementTypeBitWidth() / 8;

  const AIE::DmaBdLayout *layout = targetModel.getDmaBdLayout(col, row);
  if (!layout)
    return bdOp->emitOpError("has no buffer descriptor layout on this tile");
  const AIE::DmaBdField &addressField = layout->bufferOffset;
  // The firmware adds StateTable[idx] * func_arg into the register holding the
  // address field.
  uint64_t unit = getBdAddressUnit(targetModel, col, row);
  uint64_t perElement = uint64_t{elemBytes} << addressField.shift;
  if (perElement % unit || perElement / unit > UINT32_MAX)
    return bdOp->emitOpError("a runtime offset into ")
           << elemBytes
           << "-byte elements cannot be added to this tile's buffer "
              "descriptor address, which counts "
           << unit << "-byte units from bit " << int{addressField.shift};
  // It also clears bits [1:0] of that register. An address field from bit 0
  // loses its own low bits, so the buffer must start on that step, as the
  // offset moves by it (offset_multiple); one higher up loses another field's,
  // restored below.
  uint64_t step = addressField.shift < 2 ? unit << (2 - addressField.shift) : 0;
  if (step && unit > 1 && (!staticByteAddress || *staticByteAddress % step))
    return bdOp->emitOpError("a runtime offset needs the buffer at a multiple "
                             "of ")
           << step
           << " bytes, as the firmware clears the address's low two bits "
              "when it adds the offset";
  Location loc = bdOp->getLoc();
  AIEX::NpuUpdateFromScratchpadOp::create(
      builder, loc, stateIdx, AIEX::StateTableFunc::Mul,
      /*func_arg=*/static_cast<uint32_t>(perElement / unit),
      /*address=*/static_cast<uint32_t>(registerAddr),
      /*buffer=*/nullptr, /*column=*/nullptr, /*row=*/nullptr);
  if (step)
    return success();
  const AIE::DmaBdField &lengthBits = layout->bufferLength;
  if (lengthBits.word != addressField.word || lengthBits.shift != 0 ||
      !lengthField)
    return bdOp->emitOpError("a runtime offset would clear bits [1:0] of a "
                             "buffer descriptor register it cannot restore");
  // Created one at a time: the order of a call's arguments is unspecified, and
  // the constants' order shows in the output.
  Value address = arith::ConstantOp::create(
      builder, loc,
      builder.getI32IntegerAttr(static_cast<int32_t>(registerAddr)));
  Value lowBits, mask;
  if (std::optional<int64_t> length = getConstantIntValue(lengthField)) {
    lowBits = arith::ConstantOp::create(
        builder, loc,
        builder.getI32IntegerAttr(static_cast<int32_t>(
            lengthBits.place(static_cast<uint64_t>(*length)) & 0x3)));
    mask =
        arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0x3));
  } else {
    mask =
        arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0x3));
    lowBits =
        arith::AndIOp::create(builder, loc, cast<Value>(lengthField), mask);
  }
  AIEX::NpuMaskWrite32Op::create(builder, loc, address, lowBits, mask, nullptr,
                                 nullptr, nullptr);
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

std::optional<int64_t> AIEX::staticExecutions(Operation *op) {
  int64_t count = 1;
  for (Operation *cur = op; !isa<AIE::CoreOp, AIE::RuntimeSequenceOp>(cur);
       cur = cur->getParentOp()) {
    if (cur->getBlock() != &cur->getParentRegion()->front())
      return std::nullopt;
    Operation *parent = cur->getParentOp();
    if (isa<AIE::CoreOp, AIE::RuntimeSequenceOp>(parent))
      break;
    auto forOp = dyn_cast<scf::ForOp>(parent);
    std::optional<APInt> trips =
        forOp ? forOp.getStaticTripCount() : std::nullopt;
    if (!trips || trips->isNegative() || trips->sge(kRunsForever))
      return std::nullopt;
    std::optional<int64_t> product =
        llvm::checkedMul(count, trips->getSExtValue());
    if (!product)
      return std::nullopt;
    count = *product;
  }
  return count;
}

// Every parameter is at least 0, so it is enough to compare the terms one by
// one, a term missing from either side counting as 0.
bool AIEX::atMost(const Volume &a, int64_t scaleA, const Volume &b,
                  int64_t scaleB) {
  std::set<int64_t> keys;
  for (const auto &term :
       llvm::concat<const std::pair<const int64_t, int64_t>>(a, b))
    keys.insert(term.first);
  for (int64_t key : keys) {
    auto x = a.find(key), y = b.find(key);
    int64_t lhs = 0, rhs = 0;
    if ((x != a.end() && llvm::MulOverflow(x->second, scaleA, lhs)) ||
        (y != b.end() && llvm::MulOverflow(y->second, scaleB, rhs)) ||
        lhs > rhs)
      return false;
  }
  return true;
}

bool AIEX::wholeGranules(const Volume &bytes, int64_t granule) {
  return !granule || llvm::all_of(bytes, [&](const auto &term) {
    return term.second % granule == 0;
  });
}

AIEX::Unbalanced AIEX::balanceUnder(AIE::ConsumedByAttr proof, const Volume &in,
                                    const Volume &out) {
  if (!wholeGranules(out, proof.getGranuleBytes()))
    return Unbalanced::PartGranule;
  if (!atMost(in, proof.getOutputBytes(), out, proof.getInputBytes()))
    return Unbalanced::Excess;
  return Unbalanced::No;
}

bool AIEX::namesFifo(Operation *op, StringRef fifo,
                     SymbolTableCollection &symbols) {
  auto uses = SymbolTable::getSymbolUses(op);
  return uses && llvm::any_of(*uses, [&](const SymbolTable::SymbolUse &use) {
           auto end = dyn_cast_or_null<AIE::RouteEndpoint>(
               symbols.lookupNearestSymbolFrom(op, use.getSymbolRef()));
           return end && end.getFifoName() == fifo;
         });
}

LogicalResult AIEX::checkNoSharedResident(Operation *reload,
                                          AIE::DeviceOp before,
                                          AIE::DeviceOp after) {
  for (auto earlier : before.getOps<AIE::BufferOp>()) {
    FlatSymbolRefAttr resident = earlier.getResidentAttr();
    if (!resident)
      continue;
    for (auto later : after.getOps<AIE::BufferOp>()) {
      if (later.getResidentAttr() != resident)
        continue;
      InFlightDiagnostic diag =
          reload->emitOpError("resets the array between @")
          << before.getSymName() << " and @" << after.getSymName()
          << ", which both hold a buffer resident on " << resident
          << "; needs --reconfig-teardown";
      diag.attachNote(earlier.getLoc()) << "resident here";
      diag.attachNote(later.getLoc()) << "resident here";
      return diag;
    }
  }
  return success();
}

std::string AIEX::scratchpadCommandName(AIE::RuntimeSequenceOp seq) {
  auto device = seq->getParentOfType<AIE::DeviceOp>();
  return (device.getSymName() + ":" + seq.getSymName()).str();
}

void AIEX::emitScratchpadParamsFile(ModuleOp moduleOp, llvm::raw_ostream &os) {
  SmallVector<AIEX::ScratchpadParameterOp> allParams;
  moduleOp.walk([&](AIEX::ScratchpadParameterOp p) { allParams.push_back(p); });

  unsigned numSlots = 0;
  for (auto p : allParams)
    if (auto idx = p.getStateTableIdx())
      numSlots = std::max(numSlots, static_cast<unsigned>(*idx) + 1);
  os << allParams.size() << " " << numSlots << "\n";
  for (auto p : allParams) {
    std::string typeStr;
    llvm::raw_string_ostream ts(typeStr);
    p.getType().print(ts);
    ts.flush();
    auto kind = p.getKind();
    assert(kind && "expected --aie-lower-scratchpad-parameters to have "
                   "assigned every parameter a kind");
    StringRef kindStr = AIEX::stringifyScratchpadParameterKind(*kind);
    os << p.getSymName() << " ";
    if (std::optional<uint8_t> stateTableIdx = p.getStateTableIdx())
      os << static_cast<unsigned>(*stateTableIdx);
    else
      os << "-";
    os << " " << typeStr << " " << kindStr;
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

  SmallVector<AIE::RuntimeSequenceOp> commands;
  moduleOp.walk([&](AIE::RuntimeSequenceOp seq) {
    if (seq.getScratchpadParametersAttr())
      commands.push_back(seq);
  });
  os << commands.size() << "\n";
  for (AIE::RuntimeSequenceOp seq : commands) {
    ArrayAttr reads = seq.getScratchpadParametersAttr();
    os << scratchpadCommandName(seq) << " " << reads.size();
    for (auto ref : reads.getAsRange<FlatSymbolRefAttr>())
      os << " " << ref.getValue();
    os << "\n";
  }

  // Last, and only when there are any, so a reader that predates it stops
  // before it.
  SmallVector<std::pair<StringRef, uint32_t>> multiples;
  for (auto p : allParams)
    if (std::optional<uint32_t> multiple = p.getOffsetMultiple())
      multiples.push_back({p.getSymName(), *multiple});
  if (multiples.empty())
    return;
  os << multiples.size() << "\n";
  for (auto [name, multiple] : multiples)
    os << name << " " << multiple << "\n";
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

static StringAttr runtimeDmaSymbol(Operation *op) {
  if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op))
    return task.getAlloc().getRootReference();
  if (auto copy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op))
    return copy.getMetadata().getRootReference();
  return {};
}

Operation *AIEX::getRuntimeDmaTile(Operation *op, SymbolTable &symbols) {
  if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op))
    return task.getTile().getDefiningOp();
  StringAttr symbol = runtimeDmaSymbol(op);
  Operation *named = symbol ? symbols.lookup(symbol) : nullptr;
  if (auto alloc = dyn_cast_or_null<AIE::ShimDMAAllocationOp>(named))
    return alloc.getTile().getDefiningOp();
  if (auto endpoint = dyn_cast_or_null<AIE::RouteEndpointOp>(named))
    return endpoint.getTile().getDefiningOp();
  auto fifo = dyn_cast_or_null<AIE::ObjectFifoCreateOp>(named);
  if (!fifo)
    return nullptr;
  SmallVector<Value> ends{fifo.getProducerTile()};
  llvm::append_range(ends, fifo.getConsumerTiles());
  for (Value end : ends)
    if (auto tile = dyn_cast_or_null<AIE::TileLike>(end.getDefiningOp());
        tile && tile.isShimNOCTile())
      return tile;
  return nullptr;
}

void AIEX::forEachRuntimePinnedBdId(
    AIE::DeviceOp device,
    llvm::function_ref<void(Operation *op, Operation *tile,
                            std::optional<int> channel, int32_t id)>
        pinned) {
  SymbolTable symbols(device);
  device.walk([&](Operation *op) {
    Operation *tile = getRuntimeDmaTile(op, symbols);
    if (!tile)
      return;
    std::optional<int> channel;
    if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op))
      channel = task.getChannel();
    else if (auto alloc =
                 symbols.lookup<AIE::ShimDMAAllocationOp>(runtimeDmaSymbol(op)))
      channel = alloc.getChannelIndex();
    if (auto copy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op)) {
      pinned(op, tile, channel, static_cast<int32_t>(copy.getId()));
      return;
    }
    cast<AIEX::DMAConfigureTaskLike>(op).getBody().walk([&](AIE::DMABDOp bd) {
      if (std::optional<int32_t> id = bd.getBdId())
        pinned(op, tile, channel, *id);
    });
  });
}
