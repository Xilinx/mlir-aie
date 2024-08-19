//===- AIEXDialect.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

#include <algorithm>

using namespace mlir;
using namespace xilinx;

#include "aie/Dialect/AIEX/IR/AIEXDialect.cpp.inc"

namespace xilinx::AIEX {

// FIXME: use Tablegen'd dialect class
void AIEXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"
      >();
}

uint64_t getBufferDescriptorAddressRegisterAddress(
    const AIE::AIETargetModel &tm, unsigned bd_id, unsigned col, unsigned row) {
  assert(bd_id < tm.getNumBDs(col, row));
  return ((col & 0xff) << tm.getColumnShift()) |
         ((row & 0xff) << tm.getRowShift()) | (0x1D004 + bd_id * 0x20);
}

/* Return the correct values to write to the hardware registers to configure
  strides and wraps given the input user-facing strides and wraps.

  In the IR, we express strides in units of element data type, but the hardware
  requires it in units of address granularity. Address granularity currently is
  4 bytes for all hardware.


  User-facing strides/wraps relate to hardware as follows:

   - By default, stride 0 and size 1 is assumed if unspecified.
   - If only N strides/wraps are defined, those define the lowest N dimensions.

  inputStride[3]        == iteration_stride / elemSizeFac + 1
  inputWrap[3]          == iteration_size + 1
    Highest-dimension stride/wrap is iteration count / iteration stride.
  inputStride[2]        == d2_stride / elemSizeFac + 1
                           Note: d2_size is not specified in hardware as it is
                           implicit from the total buffer transfer length
  inputStride[1]        == d1_stride / elemSizeFac + 1
  inputSize[1]          == d1_size
  inputStride[0]        == d0_stride / elemSizeFac + 1
  inputSize[0]          == d0_size / elemSizeFac

  where elemSizeFac == bufferElementSize / addressGranularity
  where bufferElementSize == size in bytes of elements in buffer,
                             e.g. 4 for int32
  where addressGranularity == transfer granularity in hardware, which is
                              4 bytes for all current hardware

  Note: strides are expressed offset by one from user input strides, because the
  hardware does not support a 0 stride (repeat).
  */
void getHardwareStridesWraps(const AIE::AIETargetModel &targetModel,
                             mlir::MemRefType referencedBufType,
                             llvm::SmallVector<int64_t, 4> inputSizes,
                             llvm::SmallVector<int64_t, 4> inputStrides,
                             llvm::SmallVector<int64_t, 4> &sizes,
                             llvm::SmallVector<int64_t, 4> &strides) {
  assert(inputSizes.size() == inputStrides.size());
  assert(sizes.size() == 4);
  assert(strides.size() == 4);

  auto elemWidth = referencedBufType.getElementTypeBitWidth();
  auto addressGranularity = targetModel.getAddressGenGranularity();

  // Output strides and sizes are default-initialized to 0
  std::fill(sizes.begin(), sizes.end(), 0);
  std::fill(strides.begin(), strides.end(), 0);

  if (inputSizes[0] == 0) {
    // Illegal input, this won't transfer anything at all.
    // Leave it to the verification functions to complain to the user.
    return;
  }

  // d0_size, d0_stride
  sizes[0] = inputSizes[0] * elemWidth / addressGranularity;
  if (inputStrides[0] * elemWidth < addressGranularity) {
    // While the hardware cannot transfer less than addressGranularity bits at
    // a time, the user may expresses a contiguous transfer of multiple
    // elements with a stride smaller than addressGranularity. We can thus set
    // the stride to 1 (encoded in hardware as 0) here to allow such transfers.
    // The verification function should ensure that
    //    inputStrides[0] * elemWidth < addressGranularity
    //    iff. inputSize[0] * elemWidth > addressGranularity.
    strides[0] = 0;
  } else {
    strides[0] = inputStrides[0] * elemWidth / addressGranularity - 1;
  }

  // d1_size, d1_stride
  sizes[1] = inputSizes[1];
  if (inputSizes[1] > 1) {
    // Stride only matters if we have more than one iteration.
    strides[1] = inputStrides[1] * elemWidth / addressGranularity - 1;
  }

  // d2_size, d2_stride
  sizes[2] = inputSizes[2];
  if (inputSizes[2] > 1) {
    // Stride only matters if we have more than one iteration.
    strides[2] = inputStrides[2] * elemWidth / addressGranularity - 1;
  }

  // iteration_size, iteration_stride
  if (inputSizes[3] > 1) {
    // Stride only matters if we have more than one iteration.
    sizes[3] = inputSizes[3] - 1;
    // Note that the iteration_stride must be positive, just like the other
    // dimensions. However, one can encode a zero-stride "repeat" of the same
    // transfer by setting a positive repeat_count on the pushToQueue instr,
    // and setting the size here to 1. This causes the BD to "wrap" at every
    // single iteration, effectively never adding the specified stride, in turn
    // equalling a repeat without stride.
    if (inputStrides[3] > 0) {
      strides[3] = inputStrides[3] * elemWidth / addressGranularity - 1;
    }
  }
}

mlir::LogicalResult
verifyStridesWraps(mlir::Operation *forOp, mlir::MemRefType referencedBufType,
                   int tileCol, int tileRow,
                   llvm::SmallVector<int64_t, 4> inputSizes,
                   llvm::SmallVector<int64_t, 4> inputStrides,
                   llvm::SmallVector<int64_t, 4> hardwareSizes,
                   llvm::SmallVector<int64_t, 4> hardwareStrides,
                   bool skipTransformationChecks) {
  const auto &targetModel = AIE::getTargetModel(forOp);
  auto addressGranularity = targetModel.getAddressGenGranularity();
  auto elemWidth = referencedBufType.getElementTypeBitWidth();

  uint32_t wrap_bits = 0;
  uint32_t step_bits = 0;
  uint32_t iter_bits = 6;
  if (targetModel.isShimNOCTile(tileCol, tileRow)) {
    step_bits = 20; // XAIEMLGBL_NOC_MODULE_DMA_BD0_3_D0_STEPSIZE_WIDTH
    wrap_bits = 10; // XAIEMLGBL_NOC_MODULE_DMA_BD0_3_D0_WRAP_WIDTH
  } else if (targetModel.isMemTile(tileCol, tileRow)) {
    step_bits = 17; // XAIEMLGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_STEPSIZE_WIDTH
    wrap_bits = 10; // XAIEMLGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_WRAP_WIDTH
  } else if (targetModel.isCoreTile(tileCol, tileRow)) {
    step_bits = 13; // XAIEMLGBL_MEMORY_MODULE_DMA_BD0_2_D0_STEPSIZE_WIDTH
    wrap_bits = 8;  // XAIEMLGBL_MEMORY_MODULE_DMA_BD0_3_D0_WRAP_WIDTH
  } else {
    return forOp->emitOpError(
        "Unsupported tile type at (" + std::to_string(tileCol) + ", " +
        std::to_string(tileRow) + ") Must be ShimNOC, Mem or Core.");
  }

  for (int i = 0; i < 4; i++) {
    if (inputSizes[i] <= 0) {
      return forOp->emitOpError("Size ") << i << " must be a positive integer.";
    }
  }

  if (inputSizes[0] * elemWidth % addressGranularity != 0) {
    std::stringstream msg;
    msg << "Transfer sizes must be multiples of " << (addressGranularity / 8)
        << " bytes. " << inputSizes[0] << " elements at " << (elemWidth / 8)
        << " bytes each equal " << (inputSizes[0] * elemWidth / 8)
        << " bytes, which is not divisible by " << (addressGranularity / 8)
        << ". ";
    return forOp->emitOpError(msg.str());
  }

  if (skipTransformationChecks) {
    return success();
  }

  for (int i = 0; i < 3; i++) {
    if (inputSizes[i] > 1 && inputStrides[i] < 1) {
      // If inputSize[i] == 1, anything is allowable in the stride, since that
      // stride will never be applied. For any larger size, we must verify that
      // the stride is positive.
      return forOp->emitOpError("Stride ")
             << i << " must be a positive integer.";
    }
  }
  // A value of zero is allowable for the fourth-dimension stride, as such a
  // "repeat" can be accomplished by setting size==1 and repeat_count=size.
  if (inputSizes[3] > 1 && inputStrides[3] < 0) {
    return forOp->emitOpError("Stride 3 must be a non-negative integer.");
  }

  for (int i = 0; i < 4; i++) {
    // strides[0] == 1 is ok iff the tranfer size is a multiple of
    // addressGranularity, which is checked below
    if (i == 0 && inputStrides[i] == 1)
      continue;
    if (inputStrides[i] * elemWidth % addressGranularity != 0) {
      std::stringstream msg;
      msg << "Stride " << i << " is " << inputStrides[i] << " elements * "
          << (elemWidth / 8) << " bytes = " << (inputStrides[i] * elemWidth / 8)
          << " bytes, which is not divisible by " << (addressGranularity / 8)
          << ". ";
      return forOp->emitOpError(msg.str());
    }
  }

  if (hardwareSizes[0] > (1 << wrap_bits) - 1)
    return forOp->emitOpError(
        "Size 0 exceeds the [0:" + std::to_string((1 << wrap_bits) - 1) +
        "] range.");
  if (hardwareSizes[1] > (1 << wrap_bits) - 1)
    return forOp->emitOpError(
        "Size 1 exceeds the [0:" + std::to_string((1 << wrap_bits) - 1) +
        "] range.");
  if (hardwareSizes[3] > (1 << iter_bits))
    return forOp->emitOpError(
        "Size 3 exceeds the [1:" + std::to_string(1 << iter_bits) + "] range.");
  if (hardwareStrides[0] > (1 << step_bits))
    return forOp->emitOpError("Stride 0 exceeds the [1:" +
                              std::to_string(1 << step_bits) + "] range.");
  if (hardwareStrides[1] > (1 << step_bits))
    return forOp->emitOpError("Stride 1 exceeds the [1:" +
                              std::to_string(1 << step_bits) + "] range.");
  if (hardwareStrides[2] > (1 << step_bits))
    return forOp->emitOpError("Stride 2 exceeds the [1:" +
                              std::to_string(1 << step_bits) + "] range.");
  // strides[3] exceeding the range is ok iff the sizes[3] is one, which is
  // checked below
  if (hardwareStrides[3] > (1 << step_bits) && hardwareSizes[3] > 0)
    return forOp->emitOpError("Stride 3 exceeds the [1:" +
                              std::to_string(1 << step_bits) + "] range.");

  return success();
}

} // namespace xilinx::AIEX

#define GET_OP_CLASSES
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"

//===----------------------------------------------------------------------===//
// UseTokenOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::UseTokenOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (isa<func::FuncOp>(parentOp) || isa<AIE::CoreOp>(parentOp) ||
      isa<AIE::MemOp>(parentOp) || isa<AIE::ShimDMAOp>(parentOp))
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// MulticastOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::MulticastOp::verify() {
  Region &body = getPorts();
  assert(getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front())
    if (!isa<MultiDestOp, AIE::EndOp>(ops))
      return ops.emitOpError("cannot be contained in a Multicast op");

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastPacketOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::BroadcastPacketOp::verify() {
  Region &body = getPorts();
  assert(getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front())
    if (!isa<BPIDOp, AIE::EndOp>(ops))
      return ops.emitOpError("cannot be contained in a BroadcastPacket op");

  return success();
}

//===----------------------------------------------------------------------===//
// NpuDmaMemcpyNdOp
//===----------------------------------------------------------------------===//

/* Calculates the offset value to be written to the
 */
int64_t AIEX::NpuDmaMemcpyNdOp::getOffsetInBytes() {
  llvm::SmallVector<int64_t, 4> offsets =
      llvm::map_to_vector(llvm::reverse(getMixedOffsets()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  size_t stride = 1;
  size_t offset = 0;
  MemRefType my_memref = getMemref().getType();
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
  return offset;
}

// dma_memcpy_nd transfers of the form [1, 1, 1, len][0, 0, 0, 1] do not
// specify any data layout transformation, but simply express a contiguous
// transfer of `len`.
bool AIEX::NpuDmaMemcpyNdOp::isLinearTransferWithoutTransformation() {
  llvm::SmallVector<int64_t, 4> inputSizes =
      llvm::map_to_vector(llvm::reverse(getMixedSizes()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  llvm::SmallVector<int64_t, 4> inputStrides =
      llvm::map_to_vector(llvm::reverse(getMixedStrides()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  return (inputSizes[1] == 1 && inputSizes[2] == 1 && inputSizes[3] == 1 &&
          inputStrides[0] == 1 && inputStrides[1] == 0 &&
          inputStrides[2] == 0 && inputStrides[3] == 0);
}

LogicalResult AIEX::NpuDmaMemcpyNdOp::verify() {
  MemRefType buffer = getMemref().getType();
  const auto &targetModel = AIE::getTargetModel(*this);
  auto addressGranularity = targetModel.getAddressGenGranularity();

  if (buffer.getElementTypeBitWidth() > addressGranularity) {
    return emitOpError("Maximum element bit width allowed is ")
           << addressGranularity << "bits. ";
  } else if ((buffer.getNumElements() * buffer.getElementTypeBitWidth()) <
             addressGranularity) {
    return emitOpError("Minimum data transfer size required is ")
           << addressGranularity << "bits. ";
  }
  if (!llvm::all_of(getMixedStrides(), [](OpFoldResult s) {
        return getConstantIntValue(s).has_value();
      }))
    return emitOpError("Only constant strides currently supported.");
  if (!llvm::all_of(getMixedSizes(), [](OpFoldResult s) {
        return getConstantIntValue(s).has_value();
      }))
    return emitOpError("Only constant sizes currently supported.");
  if (!llvm::all_of(getMixedOffsets(), [](OpFoldResult s) {
        return getConstantIntValue(s).has_value();
      }))
    return emitOpError("Only constant offsets currently supported.");

  llvm::SmallVector<int64_t, 4> inputSizes =
      llvm::map_to_vector(llvm::reverse(getMixedSizes()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  llvm::SmallVector<int64_t, 4> inputStrides =
      llvm::map_to_vector(llvm::reverse(getMixedStrides()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  llvm::SmallVector<int64_t, 4> hardwareSizes(4);
  llvm::SmallVector<int64_t, 4> hardwareStrides(4);
  getHardwareStridesWraps(targetModel, buffer, inputSizes, inputStrides,
                          hardwareSizes, hardwareStrides);
  int64_t offset = getOffsetInBytes();

  // The experimental HSA target uses this op on AIE1, skip all the AIE2
  // specific checks
  if (targetModel.getTargetArch() == AIE::AIEArch::AIE1)
    return success();

  if (offset % 4 != 0) {
    return emitOpError("Offset must be 4-byte-aligned.");
  }

  // dma_memcpy_nd transfers of the form [1, 1, 1, len][0, 0, 0, 1] do not
  // specify any data layout transformation, but simply express a contiguous
  // transfer of `len`. For backwards compatibility, we allow this to proceed
  // even if it exceeds the maximum stride/wrap size of any one dimension,
  // and simply do not lower any data layout transformations, since there is
  // no other way to express this at the dma_memcpy_nd interface otherwise.
  bool skipTransformationChecks = isLinearTransferWithoutTransformation();
  if (failed(verifyStridesWraps(*this, buffer, getX(), getY(), inputSizes,
                                inputStrides, hardwareSizes, hardwareStrides,
                                skipTransformationChecks))) {
    return failure();
  }

  // packet header
  if (auto packetInfo = getPacket()) {
    if (packetInfo->getPktType() > 7)
      return emitOpError("Packet type field can only hold 3 bits.");
    if (packetInfo->getPktId() > 31)
      return emitOpError("Packet ID field can only hold 5 bits.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// NpuDmaWaitOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuDmaWaitOp::verify() {
  AIE::DeviceOp dev = (*this)->getParentOfType<AIE::DeviceOp>();
  // Some passes (e.g. aie-standard-lowering) use aiex ops outside a DeviceOp,
  // so we can't expect the device to always exist.
  if (dev && !dev.lookupSymbol(getSymbol()))
    return emitOpError("couldn't find symbol in parent device");
  return success();
}

//===----------------------------------------------------------------------===//
// NpuPushQueueOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuPushQueueOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);
  auto numBds = targetModel.getNumBDs(getColumn(), getRow());
  if (getBdId() > numBds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (getRepeatCount() > 255)
    return emitOpError("Repeat count exceeds the [0:255] range.");
  return success();
}

//===----------------------------------------------------------------------===//
// NpuWriteBdOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuWriteBdOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);
  auto numBds = targetModel.getNumBDs(getColumn(), getRow());
  if (getBdId() > numBds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (getD0Size() > 0x3FF)
    return emitOpError("D0 Size exceeds the [0:1023] range.");
  if (getD0Stride() > 0xFFFFF)
    return emitOpError("D0 Stride exceeds the [0:1M-1] range.");
  if (getD1Size() > 0x3FF)
    return emitOpError("D1 Size exceeds the [0:1023] range.");
  if (getD1Stride() > 0xFFFFF)
    return emitOpError("D1 Stride exceeds the [0:1M-1] range.");
  if (getD2Stride() > 0xFFFFF)
    return emitOpError("D2 Stride exceeds the [0:1M-1] range.");
  if (getIterationSize() > 0x3F)
    return emitOpError("Iteration Size exceeds the [0:63] range.");
  if (getIterationStride() > 0xFFFFF)
    return emitOpError("Iteration Stride exceeds the [0:1M-1] range.");
  return success();
}

//===----------------------------------------------------------------------===//
// RuntimeSequenceOp
//===----------------------------------------------------------------------===//

ParseResult AIEX::RuntimeSequenceOp::parse(OpAsmParser &parser,
                                           OperationState &result) {

  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  SmallVector<OpAsmParser::Argument> entryArgs;

  // Entry arguments,  e.g. (%addr: memref<1xi32>)
  ParseResult argParseResult = parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        OpAsmParser::Argument argument;
        if (parser.parseArgument(argument, true, true)) {
          return failure();
        }
        entryArgs.push_back(argument);
        return success();
      });
  if (argParseResult) {
    return argParseResult;
  }

  // Body
  auto *body = result.addRegion();
  ParseResult bodyParseResult = parser.parseRegion(*body, entryArgs, false);
  if (bodyParseResult) {
    return bodyParseResult;
  }

  return success();
}

void AIEX::RuntimeSequenceOp::print(OpAsmPrinter &printer) {
  Region &body = getRegion();

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    printer << ' ';
    printer.printSymbolName(nameAttr);
  }

  printer << '(';
  for (unsigned i = 0, n = body.getNumArguments(); i < n; i++) {
    if (i > 0) {
      printer << ", ";
    }
    printer.printRegionArgument(body.getArgument(i));
  }
  printer << ')';

  printer << ' ';
  printer.printRegion(body, false, true);
}

LogicalResult AIEX::RuntimeSequenceOp::verify() {
  AIE::DeviceOp device = (*this)->getParentOfType<AIE::DeviceOp>();
  if (!device) {
    // this check is redudnant with the HasParent trait, but can't hurt
    (*this)->emitOpError() << "must be inside AIE device operation.";
    return failure();
  }
  auto seq_ops = device.getOps<AIEX::RuntimeSequenceOp>();
  if (std::distance(seq_ops.begin(), seq_ops.end()) > 1) {
    auto err = device.emitOpError()
               << "Cannot have more than one runtime sequence per device.";
    for (auto it = seq_ops.begin(); it != seq_ops.end(); ++it) {
      AIEX::RuntimeSequenceOp seq_op = *it;
      err.attachNote(seq_op.getLoc()) << "Sequence operation definition here.";
    }
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DMAConfigureTaskOp
//===----------------------------------------------------------------------===//

std::optional<uint32_t> AIEX::DMAConfigureTaskOp::getFirstBdId() {
  Region &body = getBody();
  if (body.empty()) {
    return std::nullopt;
  }
  auto bd_ops = body.front().getOps<AIE::DMABDOp>();
  if (bd_ops.empty()) {
    return std::nullopt;
  }
  AIE::DMABDOp bd = *bd_ops.begin();
  if (!bd.getBdId().has_value()) {
    return std::nullopt;
  }
  return bd.getBdId().value();
}

LogicalResult
AIEX::DMAConfigureTaskOp::canonicalize(AIEX::DMAConfigureTaskOp op,
                                       PatternRewriter &rewriter) {
  // Remove blocks that contain nothing but a terminator
  Region &body = op.getBody();
  bool did_rewrite = false;
  for (auto it = body.begin(); it != body.end(); ++it) {
    Block &block = *it;
    if (block.empty()) {
      continue;
    }
    auto ops_it = block.without_terminator();
    if (std::distance(ops_it.begin(), ops_it.end()) == 0) {
      rewriter.eraseOp(block.getTerminator());
      did_rewrite = true;
    }
  }
  if (did_rewrite) {
    return success();
  }
  return failure();
}

LogicalResult AIEX::DMAConfigureTaskOp::verify() {
  Region &body = getBody();
  for (auto it = body.begin(); it != body.end(); ++it) {
    Block &block = *it;
    if (block.empty()) {
      continue;
    }
    if (block.hasNoPredecessors() && !block.isEntryBlock()) {
      auto error = block.getTerminator()->emitError(
          "Block ending in this terminator does not form a chain with "
          "entry block.");
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DMAStartBdChainOp
//===----------------------------------------------------------------------===//

AIE::BDChainOp AIEX::DMAStartBdChainOp::getBDChainOp() {
  AIE::DeviceOp device = (*this)->getParentOfType<AIE::DeviceOp>();
  AIE::BDChainOp chain = device.lookupSymbol<AIE::BDChainOp>(getSymbol());
  return chain;
}

LogicalResult AIEX::DMAStartBdChainOp::verify() {
  AIE::BDChainOp chain = getBDChainOp();
  if (!chain) {
    return emitOpError("symbol does not reference valid BD chain");
  }

  auto actualArgTypes = getArgs().getTypes();
  auto expectedArgTypes = chain.getRegion().getArgumentTypes();
  if (actualArgTypes.size() != expectedArgTypes.size()) {
    return emitOpError("Number of arguments mismatches.");
  }
  for (unsigned i = 0, n = expectedArgTypes.size(); i < n; i++) {
    if (actualArgTypes[i] != expectedArgTypes[i]) {
      return emitOpError("Argument ") << (i + 1) << " types mismatch: "
                                      << "expected " << expectedArgTypes[i]
                                      << " but got " << actualArgTypes[i];
    }
  }
  return success();
}