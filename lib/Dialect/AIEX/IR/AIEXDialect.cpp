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
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/TypeSize.h"

#include <cstdint>
#include <numeric>

using namespace mlir;
using namespace xilinx;

#include "aie/Dialect/AIEX/IR/AIEXDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/AIEX/IR/AIEXTypes.cpp.inc"

namespace xilinx::AIEX {

// FIXME: use Tablegen'd dialect class
void AIEXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "aie/Dialect/AIEX/IR/AIEXTypes.cpp.inc"
      >();
}

} // namespace xilinx::AIEX

#define GET_OP_CLASSES
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"

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
void AIEX::getHardwareStridesWraps(const AIE::AIETargetModel &targetModel,
                                   mlir::Operation *op,
                                   mlir::BaseMemRefType referencedBufType,
                                   llvm::SmallVector<int64_t, 4> inputSizes,
                                   llvm::SmallVector<int64_t, 4> inputStrides,
                                   llvm::SmallVector<int64_t, 4> &sizes,
                                   llvm::SmallVector<int64_t, 4> &strides) {
  assert(inputSizes.size() == inputStrides.size());
  assert(sizes.size() == 4);
  assert(strides.size() == 4);

  DataLayout dataLayout = DataLayout::closest(op);
  auto elemWidth =
      dataLayout.getTypeSizeInBits(referencedBufType.getElementType());
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
  if (inputStrides[0] * elemWidth < addressGranularity ||
      (elemWidth > addressGranularity)) {
    // First check:
    // While the hardware cannot transfer less than addressGranularity bits at
    // a time, the user may expresses a contiguous transfer of multiple
    // elements with a stride smaller than addressGranularity. We can thus set
    // the stride to 1 (encoded in hardware as 0) here to allow such transfers.
    // The verification function should ensure that
    //    inputStrides[0] * elemWidth < addressGranularity
    //    iff. inputSize[0] * elemWidth > addressGranularity.
    // Second check:
    // If the element width is larger than addressGranularity, we need to make
    // sure that all bytes are properly copied and therefore the stride must be
    // set to 1 (encoded in hardware as 0).
    // The verification function should ensure that
    //     inputStrides[0] * elemWidth % addressGranularity == 0
    //     && inputStrides[0] == 1 if elemWidth > addressGranularity
    // This makes it impossible to have a stride greater than 1 for
    // elemWidths bigger than addressGranularity, even if they are a multiple of
    // it. Such operations should make use of an additional dimension instead.
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
AIEX::verifyStridesWraps(mlir::Operation *forOp,
                         mlir::BaseMemRefType referencedBufType, int tileCol,
                         int tileRow, llvm::SmallVector<int64_t, 4> inputSizes,
                         llvm::SmallVector<int64_t, 4> inputStrides,
                         llvm::SmallVector<int64_t, 4> hardwareSizes,
                         llvm::SmallVector<int64_t, 4> hardwareStrides,
                         bool skipTransformationChecks) {
  const auto &targetModel = AIE::getTargetModel(forOp);
  auto addressGranularity = targetModel.getAddressGenGranularity();
  DataLayout dataLayout = DataLayout::closest(forOp);
  auto elemWidth =
      dataLayout.getTypeSizeInBits(referencedBufType.getElementType());

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

  for (int i = 0; i < 3; i++) {
    if (inputSizes[i] > 1 && inputStrides[i] < 1) {
      // If inputSize[i] == 1, anything is allowable in the stride, since that
      // stride will never be applied. For any larger size, we must verify that
      // the stride is positive.
      return forOp->emitOpError("Stride ")
             << i << " must be a positive integer.";
    }
  }
  // A value of zero is allowable for the fourth-dimension stride
  // (this indicates an interation stride for the repeat of 0)
  if (inputSizes[3] > 1 && inputStrides[3] < 0) {
    return forOp->emitOpError("Stride 3 must be a non-negative integer.");
  }

  for (int i = 0; i < 4; i++) {
    // strides[0] == 1 is ok iff the transfer size is a multiple of
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

  if (!skipTransformationChecks && hardwareSizes[0] > (1 << wrap_bits) - 1)
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
  llvm::SmallVector<int64_t, 4> strides =
      llvm::map_to_vector(llvm::reverse(getMixedStrides()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  size_t offset = 0;
  size_t R = offsets.size();
  size_t el_bit_width = getElementTypeBitwidth();
  assert(el_bit_width % 8 == 0 &&
         "Expected Memref element bitwidth to be multiple of 8.");
  size_t S = el_bit_width / 8;
  for (size_t i = 0; i < R; i++)
    offset += offsets[i] * strides[i] * S;
  return offset;
}

// dma_memcpy_nd transfers of the form [*, 1, 1, len][*, 0, 0, 1] do not
// specify any data layout transformation, but simply express a contiguous
// transfer of `len`. We exclude checks to 4th dimension, because repeat count
// is still possible without a data layout transformation.
bool AIEX::NpuDmaMemcpyNdOp::isLinearTransferWithoutTransformation() {
  llvm::SmallVector<int64_t, 4> inputSizes =
      llvm::map_to_vector(llvm::reverse(getMixedSizes()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  llvm::SmallVector<int64_t, 4> inputStrides =
      llvm::map_to_vector(llvm::reverse(getMixedStrides()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  return (inputSizes[1] == 1 && inputSizes[2] == 1 && inputStrides[0] == 1 &&
          inputStrides[1] == 0 && inputStrides[2] == 0);
}

// Helper method to check if a requested burst length is supported by the target
// model. Returns an error message if the burst length is not supported or an
// empty option otherwise.
static std::optional<std::string>
checkBurstLength(const xilinx::AIE::AIETargetModel &targetModel,
                 uint32_t requestedBurstLength) {
  if (requestedBurstLength != 0) {
    auto bel = targetModel.getShimBurstEncodingsAndLengths();
    auto pair = std::find_if(bel.begin(), bel.end(),
                             [=](const std::pair<uint32_t, uint32_t> &p) {
                               return p.second == requestedBurstLength;
                             });

    if (pair == bel.end()) {
      std::string errorMessage =
          "Requested burst length is not supported by the target. "
          "Supported burst lengths:";

      errorMessage =
          std::accumulate(bel.begin(), bel.end(), errorMessage,
                          [](const std::string &a, auto b) {
                            return a + " " + std::to_string(b.second);
                          });

      return errorMessage;
    }
  }

  return std::nullopt;
}

LogicalResult AIEX::NpuDmaMemcpyNdOp::verify() {
  BaseMemRefType buffer = getMemref().getType();
  const auto &targetModel = AIE::getTargetModel(*this);
  auto addressGranularity = targetModel.getAddressGenGranularity();

  if (getElementTypeBitwidth() > addressGranularity) {
    return emitOpError("Maximum element bit width allowed is ")
           << addressGranularity << "bits. ";
  }
  if (buffer.hasStaticShape() &&
      (buffer.getNumElements() * getElementTypeBitwidth()) <
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
  getHardwareStridesWraps(targetModel, getOperation(), buffer, inputSizes,
                          inputStrides, hardwareSizes, hardwareStrides);
  int64_t offset = getOffsetInBytes();

  auto errorMessage = checkBurstLength(targetModel, getBurstLength());
  if (errorMessage.has_value()) {
    return emitOpError(errorMessage.value());
  }

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
  AIE::ShimDMAllocationGetter allocGetter;
  AIE::DeviceOp dev = getOperation()->getParentOfType<AIE::DeviceOp>();
  if (auto allocOp = allocGetter.get(dev, getMetadata())) {
    int col = allocOp->getCol();
    bool skipTransformationChecks = isLinearTransferWithoutTransformation();
    if (failed(verifyStridesWraps(*this, buffer, col, 0, inputSizes,
                                  inputStrides, hardwareSizes, hardwareStrides,
                                  skipTransformationChecks))) {
      return failure();
    }
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
  bool isLinearTransfer =
      (getD0Size() >= 1) && (getD1Size() == 1) && (getIterationSize() == 0);
  if (getBdId() > numBds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (getPacketId() > 31)
    return emitOpError("Packet ID exceeds the maximum supported by 5 bits.");
  if (getPacketType() > 7)
    return emitOpError("Packet Type exceeds the maximum supported by 3 bits.");
  if (!isLinearTransfer && getD0Size() > 0x3FF)
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
  if (targetModel.isShimNOCTile(getColumn(), getRow()) && getD2Size() != 0)
    return emitOpError("ShimTile only supports 3 dimensions of sizes.");
  if (targetModel.isShimNOCTile(getColumn(), getRow()) &&
      (getD0ZeroBefore() != 0 || getD0ZeroAfter() != 0 ||
       getD1ZeroBefore() != 0 || getD1ZeroAfter() != 0 ||
       getD2ZeroBefore() != 0 || getD2ZeroAfter() != 0))
    return emitOpError("ShimTile doesn't support zero padding.");
  if (!targetModel.isShimNOCTile(getColumn(), getRow()) &&
      getBurstLength() != 0)
    return emitOpError("Only ShimTiles support burst length.");
  auto errorMessage = checkBurstLength(targetModel, getBurstLength());
  if (errorMessage.has_value()) {
    return emitOpError(errorMessage.value());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// NpuWrite32Op
//===----------------------------------------------------------------------===//

template <typename T>
static std::optional<uint32_t> getAbsoluteAddress(T *op) {
  AIE::DeviceOp device =
      op->getOperation()->template getParentOfType<AIE::DeviceOp>();
  if (!device) {
    op->emitError("Must be inside a device.");
    return std::nullopt;
  }
  const AIE::AIETargetModel &tm = device.getTargetModel();

  uint32_t address = 0;

  // If blockwrite references a buffer, the given address is understood to be
  // relative to the buffer's start address.
  if (op->getBuffer()) {
    AIE::BufferOp buffer = device.lookupSymbol<AIE::BufferOp>(*op->getBuffer());
    if (!buffer) {
      op->emitError() << "buffer '" << *op->getBuffer()
                      << "' not found in device";
      return std::nullopt;
    }

    if (!buffer.getAddress()) {
      mlir::InFlightDiagnostic err =
          op->emitError("referenced buffer must have address assigned");
      err.attachNote(buffer.getLoc()) << "This buffer must have an address.";
      return std::nullopt;
    }

    uint32_t col = buffer.getTileOp().getCol();
    uint32_t row = buffer.getTileOp().getRow();
    address = static_cast<uint32_t>(*buffer.getAddress()) +
              op->getAddress() * sizeof(uint32_t);
    address = ((col & 0xff) << tm.getColumnShift()) |
              ((row & 0xff) << tm.getRowShift()) | (address & 0xfffff);
  } else { // otherwise, the given address is absolute
    address = op->getAddress();
    std::optional<uint32_t> col = op->getColumn();
    std::optional<uint32_t> row = op->getRow();
    if (col && row) {
      // If col and row are set, only the lower 20 bits of the address are
      // used, and col and row dictate the upper bits (ignored)
      address = ((*col & 0xff) << tm.getColumnShift()) |
                ((*row & 0xff) << tm.getRowShift()) | (address & 0xfffff);
    }
  }

  return address;
}

std::optional<uint32_t> AIEX::NpuWrite32Op::getAbsoluteAddress() {
  return ::getAbsoluteAddress(this);
}

//===----------------------------------------------------------------------===//
// NpuMaskWrite32Op
//===----------------------------------------------------------------------===//

std::optional<uint32_t> AIEX::NpuMaskWrite32Op::getAbsoluteAddress() {
  return ::getAbsoluteAddress(this);
}

//===----------------------------------------------------------------------===//
// NpuBlockWriteOp
//===----------------------------------------------------------------------===//

std::optional<uint32_t> AIEX::NpuBlockWriteOp::getAbsoluteAddress() {
  return ::getAbsoluteAddress(this);
}

DenseIntElementsAttr AIEX::NpuBlockWriteOp::getDataWords() {
  Value memref = this->getData();
  DataLayout dataLayout = DataLayout::closest(*this);
  int64_t width = dataLayout.getTypeSizeInBits(
      cast<MemRefType>(memref.getType()).getElementType());
  if (width != 32) {
    emitWarning("Only 32-bit data type is supported for now");
    return nullptr;
  }

  memref::GetGlobalOp getGlobal = memref.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal) {
    emitError("Only MemRefs from memref.get_global are supported");
    return nullptr;
  }

  auto global = dyn_cast_if_present<memref::GlobalOp>(
      (*this)->getParentOfType<AIE::DeviceOp>().lookupSymbol(
          getGlobal.getName()));
  if (!global) {
    emitError("Global symbol not found");
    return nullptr;
  }

  auto initVal = global.getInitialValue();
  if (!initVal) {
    emitError("Global symbol has no initial value");
    return nullptr;
  }

  auto data = dyn_cast<DenseIntElementsAttr>(*initVal);
  if (!data) {
    emitError("Global symbol initial value is not a dense int array");
    return nullptr;
  }

  return data;
}

//===----------------------------------------------------------------------===//
// RuntimeSequenceOp
//===----------------------------------------------------------------------===//

ParseResult AIEX::RuntimeSequenceOp::parse(OpAsmParser &parser,
                                           OperationState &result) {

  // Name of this runtime sequence
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
  if (nameAttr &&
      nameAttr != ::mlir::OpBuilder((*this)->getContext())
                      .getStringAttr(getDefaultRuntimeSequenceName())) {
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
  return success();
}

AIEX::RuntimeSequenceOp
AIEX::RuntimeSequenceOp::getForSymbolInDevice(AIE::DeviceOp deviceOp,
                                              llvm::StringRef symbol) {
  AIEX::RuntimeSequenceOp runtimeSequenceOp;
  if (!symbol.size()) {
    runtimeSequenceOp = *deviceOp.getOps<AIEX::RuntimeSequenceOp>().begin();
  } else {
    Operation *maybeRuntimeSequenceOp =
        mlir::SymbolTable::lookupSymbolIn(deviceOp, symbol);
    if (!maybeRuntimeSequenceOp) {
      return nullptr;
    }
    runtimeSequenceOp =
        llvm::dyn_cast<AIEX::RuntimeSequenceOp>(maybeRuntimeSequenceOp);
  }
  return runtimeSequenceOp;
}

AIEX::RuntimeSequenceOp
AIEX::RuntimeSequenceOp::getForSymbolInDeviceOrError(AIE::DeviceOp deviceOp,
                                                     llvm::StringRef symbol) {
  AIEX::RuntimeSequenceOp runtimeSequenceOp =
      getForSymbolInDevice(deviceOp, symbol);
  if (!runtimeSequenceOp) {
    if (!symbol.empty()) {
      deviceOp.emitError("No such runtime sequence: ") << symbol;
    } else {
      deviceOp.emitError("No runtime sequence in device");
    }
  }
  return runtimeSequenceOp;
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
  if (bd_ops.empty() && body.front().getNumSuccessors() == 1) {
    // Allow the first block to be empty and point to the entry point of the
    // chain. This allows for specifying cyclying BD chains (infinite loops)
    // within the constraints of MLIR syntax.
    Block &chain_entry = *body.front().getSuccessor(0);
    bd_ops = chain_entry.getOps<AIE::DMABDOp>();
  }
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

    const AIE::AIETargetModel &targetModel =
        AIE::getTargetModel(getOperation());

    // This is a layering violation on the DMABDOps, but they are never verified
    // otherwise Because DMAConfigureTaskOps are not yet merged into the AIE
    // dialect. The normal DMABDOp verify operation will skip over any BD inside
    // a DMAConfigureTaskOp
    LogicalResult result = success();
    block.walk([&](AIE::DMABDOp bd) {
      if (bd.getBurstLength() != 0 &&
          !targetModel.isShimNOCTile(getTileID().col, getTileID().row)) {
        bd.emitOpError("Burst length is only supported in Shim NOC tiles that "
                       "are connected to the memory-mapped NOC.");
        result = failure();
      }
    });
    if (failed(result)) {
      return result;
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

//===----------------------------------------------------------------------===//
// NpuControlPacketOp
//===----------------------------------------------------------------------===//

uint32_t AIEX::NpuControlPacketOp::getRowFromAddr() {
  const auto &targetModel = AIE::getTargetModel(*this);
  uint32_t addr = getAddress();
  uint32_t rowInt = (addr >> targetModel.getRowShift()) & 0x1f;
  return rowInt;
}

uint32_t AIEX::NpuControlPacketOp::getColumnFromAddr() {
  const auto &targetModel = AIE::getTargetModel(*this);
  uint32_t addr = getAddress();
  uint32_t colInt = (addr >> targetModel.getColumnShift()) & 0x1f;
  return colInt;
}

//===----------------------------------------------------------------------===//
// SetLockOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::SetLockOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);

  if (targetModel.getTargetArch() == AIE::AIEArch::AIE1)
    return emitOpError("SetLockOp is not supported on AIE1.");

  if (getValue() > targetModel.getMaxLockValue())
    return emitOpError("Lock value exceeds the maximum value of " +
                       std::to_string(targetModel.getMaxLockValue()));

  auto lockOp = getLockOp();
  auto lockIDOpt = getLockOp().getLockID();
  // Note that the lockID may not be assigned initially, so lets wait until it
  // is to verify the lockID dependent conditions
  if (!lockIDOpt) {
    return success();
  }

  auto col = lockOp.colIndex();
  auto row = lockOp.rowIndex();
  uint32_t lockID = lockOp.getLockIDValue();

  if (lockID >= targetModel.getNumLocks(col, row)) {
    return emitOpError("Lock ID out of range for given tile. Max ID: " +
                       std::to_string(targetModel.getNumLocks(col, row) - 1));
  }

  if (!targetModel.getLocalLockAddress(lockID, lockOp.getTileID())) {
    return emitOpError("Invalid lock ID and tile combination when trying to "
                       "retrieve the local lock address.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BlockFloatingPointType
//===----------------------------------------------------------------------===//
uint64_t AIEX::BlockFloatType::getTotalSizeInBits() const {
  return getBlockSize() * getMantissaBits() + getExponentBits() +
         getSubtileShiftBits();
}

llvm::TypeSize AIEX::BlockFloatType::getTypeSizeInBits(
    const mlir::DataLayout &dataLayout,
    mlir::DataLayoutEntryListRef params) const {
  return llvm::TypeSize::getFixed(getTotalSizeInBits());
}

uint64_t AIEX::BlockFloatType::getABIAlignment(
    const mlir::DataLayout &dataLayout,
    mlir::DataLayoutEntryListRef params) const {
  // For the purposes of the data movement operations, we want all types to be
  // packed <=> ABI alignment is 1.
  return 1;
}

std::optional<AIEX::BlockFloatType::BlockFormat>
AIEX::BlockFloatType::getBlockFormat(StringRef blockType) {
  static const llvm::StringMap<AIEX::BlockFloatType::BlockFormat>
      blockFormatsMap = {
          {"v8bfp16ebs8", {8, 8, 8, 0}},
          {"v16bfp16ebs16", {16, 8, 8, 0}},
      };

  auto it = blockFormatsMap.find(blockType);
  if (it != blockFormatsMap.end()) {
    return it->second;
  }

  return std::nullopt;
}

LogicalResult
AIEX::BlockFloatType::verify(function_ref<InFlightDiagnostic()> emitError,
                             StringRef block_type) {
  if (!getBlockFormat(block_type))
    return emitError() << "Invalid block type: " << block_type
                       << ". Known types are: v8bfp16ebs8, v16bfp16ebs16.";

  return success();
}

//===----------------------------------------------------------------------===//
// ConfigureOp
//===----------------------------------------------------------------------===//

AIE::DeviceOp AIEX::ConfigureOp::getReferencedDeviceOp() {
  ModuleOp moduleOp = this->getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    emitError("aiex.configure must be inside of a module");
    return nullptr;
  }
  Operation *maybeReferencedDevice =
      SymbolTable::lookupSymbolIn(moduleOp.getOperation(), getSymbolAttr());
  if (!maybeReferencedDevice) {
    emitError("No such device: '") << getSymbolAttr() << "'";
    return nullptr;
  }
  AIE::DeviceOp referencedDevice =
      llvm::dyn_cast<AIE::DeviceOp>(maybeReferencedDevice);
  if (!referencedDevice) {
    emitError("Not a device: '") << getSymbolAttr() << "'";
    return nullptr;
  }
  return referencedDevice;
}

LogicalResult AIEX::ConfigureOp::verify() {
  AIE::DeviceOp parentDev = getOperation()->getParentOfType<AIE::DeviceOp>();
  AIE::DeviceOp referencedDev = getReferencedDeviceOp();
  if (!referencedDev) {
    return failure();
  }
  if (parentDev.getDevice() != referencedDev.getDevice()) {
    emitError("Device types do not match: '")
        << AIE::stringifyAIEDevice(parentDev.getDevice()) << "' vs. '"
        << AIE::stringifyAIEDevice(referencedDev.getDevice()) << "'";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RunOp
//===----------------------------------------------------------------------===//

AIE::DeviceOp AIEX::RunOp::getCalleeDeviceOp() {
  AIEX::ConfigureOp configureOp =
      getOperation()->getParentOfType<AIEX::ConfigureOp>();
  if (!configureOp) {
    return nullptr;
  }
  AIE::DeviceOp referencedDevice = configureOp.getReferencedDeviceOp();
  return referencedDevice;
}

AIEX::RuntimeSequenceOp AIEX::RunOp::getCalleeRuntimeSequenceOp() {
  AIEX::ConfigureOp configureOp =
      getOperation()->getParentOfType<AIEX::ConfigureOp>();
  if (!configureOp) {
    return nullptr;
  }
  AIE::DeviceOp referencedDevice = configureOp.getReferencedDeviceOp();
  if (!referencedDevice) {
    return nullptr;
  }

  Operation *maybeRuntimeSequence =
      SymbolTable::lookupSymbolIn(referencedDevice, getRuntimeSequenceSymbol());

  if (!maybeRuntimeSequence) {
    auto err = emitError() << "No such runtime sequence for device '"
                           << referencedDevice.getSymName() << "': '"
                           << getRuntimeSequenceSymbol() << "'";
    err.attachNote(referencedDevice.getLoc())
        << "This device does not have a '" << getRuntimeSequenceSymbol()
        << "' runtime sequence";
    return nullptr;
  }
  AIEX::RuntimeSequenceOp runtimeSequence =
      llvm::dyn_cast<AIEX::RuntimeSequenceOp>(maybeRuntimeSequence);
  if (!runtimeSequence) {
    emitError() << "Not a runtime sequence: '" << getRuntimeSequenceSymbol()
                << "'";
    return nullptr;
  }

  return runtimeSequence;
}

LogicalResult AIEX::RunOp::verify() {
  if (getCalleeDeviceOp() && getCalleeRuntimeSequenceOp()) {
    return success();
  }
  return failure();
}