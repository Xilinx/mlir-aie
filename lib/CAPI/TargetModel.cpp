//===- TargetModel.cpp - C API for AIE TargetModel ------------------------===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie-c/TargetModel.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Utils/DmaDecomposition.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "llvm/Support/MathExtras.h"

using namespace mlir;

static inline AieTargetModel wrap(const xilinx::AIE::AIETargetModel &tm) {
  return AieTargetModel{reinterpret_cast<std::uintptr_t>(&tm)};
}

static inline const xilinx::AIE::AIETargetModel &unwrap(AieTargetModel tm) {
  return *reinterpret_cast<const xilinx::AIE::AIETargetModel *>(tm.d);
}

AieTargetModel aieGetTargetModel(uint32_t device) {
  return wrap(
      xilinx::AIE::getTargetModel(static_cast<xilinx::AIE::AIEDevice>(device)));
}

uint32_t aieGetTargetModelAddressGenGranularity(AieTargetModel targetModel) {
  return unwrap(targetModel).getAddressGenGranularity();
}

int aieTargetModelColumns(AieTargetModel targetModel) {
  return unwrap(targetModel).columns();
}

int aieTargetModelRows(AieTargetModel targetModel) {
  return unwrap(targetModel).rows();
}

uint32_t aieTargetModelGetTileType(AieTargetModel targetModel, int col,
                                   int row) {
  return static_cast<uint32_t>(unwrap(targetModel).getTileType(col, row));
}

bool aieTargetModelIsCoreTile(AieTargetModel targetModel, int col, int row) {
  return unwrap(targetModel).isCoreTile(col, row);
}

bool aieTargetModelIsMemTile(AieTargetModel targetModel, int col, int row) {
  return unwrap(targetModel).isMemTile(col, row);
}

bool aieTargetModelIsShimNOCTile(AieTargetModel targetModel, int col, int row) {
  return unwrap(targetModel).isShimNOCTile(col, row);
}

bool aieTargetModelIsShimPLTile(AieTargetModel targetModel, int col, int row) {
  return unwrap(targetModel).isShimPLTile(col, row);
}

bool aieTargetModelIsShimNOCorPLTile(AieTargetModel targetModel, int col,
                                     int row) {
  return unwrap(targetModel).isShimNOCorPLTile(col, row);
}

bool aieTargetModelIsInternal(AieTargetModel targetModel, int src_col,
                              int src_row, int dst_col, int dst_row) {
  return unwrap(targetModel).isInternal(src_col, src_row, dst_col, dst_row);
}

bool aieTargetModelIsWest(AieTargetModel targetModel, int src_col, int src_row,
                          int dst_col, int dst_row) {
  return unwrap(targetModel).isWest(src_col, src_row, dst_col, dst_row);
}

bool aieTargetModelIsEast(AieTargetModel targetModel, int src_col, int src_row,
                          int dst_col, int dst_row) {
  return unwrap(targetModel).isEast(src_col, src_row, dst_col, dst_row);
}

bool aieTargetModelIsNorth(AieTargetModel targetModel, int src_col, int src_row,
                           int dst_col, int dst_row) {
  return unwrap(targetModel).isNorth(src_col, src_row, dst_col, dst_row);
}

bool aieTargetModelIsSouth(AieTargetModel targetModel, int src_col, int src_row,
                           int dst_col, int dst_row) {
  return unwrap(targetModel).isSouth(src_col, src_row, dst_col, dst_row);
}

bool aieTargetModelIsMemWest(AieTargetModel targetModel, int src_col,
                             int src_row, int dst_col, int dst_row) {
  return unwrap(targetModel).isMemWest(src_col, src_row, dst_col, dst_row);
}

bool aieTargetModelIsMemEast(AieTargetModel targetModel, int src_col,
                             int src_row, int dst_col, int dst_row) {
  return unwrap(targetModel).isMemEast(src_col, src_row, dst_col, dst_row);
}

bool aieTargetModelIsMemNorth(AieTargetModel targetModel, int src_col,
                              int src_row, int dst_col, int dst_row) {
  return unwrap(targetModel).isMemNorth(src_col, src_row, dst_col, dst_row);
}

bool aieTargetModelIsMemSouth(AieTargetModel targetModel, int src_col,
                              int src_row, int dst_col, int dst_row) {
  return unwrap(targetModel).isMemSouth(src_col, src_row, dst_col, dst_row);
}

bool aieTargetModelIsLegalMemAffinity(AieTargetModel targetModel, int src_col,
                                      int src_row, int dst_col, int dst_row) {
  return unwrap(targetModel)
      .isLegalMemAffinity(src_col, src_row, dst_col, dst_row);
}

uint32_t aieTargetModelGetMemSouthBaseAddress(AieTargetModel targetModel) {
  return unwrap(targetModel).getMemSouthBaseAddress();
}

uint32_t aieTargetModelGetMemNorthBaseAddress(AieTargetModel targetModel) {
  return unwrap(targetModel).getMemNorthBaseAddress();
}

uint32_t aieTargetModelGetMemEastBaseAddress(AieTargetModel targetModel) {
  return unwrap(targetModel).getMemEastBaseAddress();
}

uint32_t aieTargetModelGetMemWestBaseAddress(AieTargetModel targetModel) {
  return unwrap(targetModel).getMemWestBaseAddress();
}

uint32_t aieTargetModelGetLocalMemorySize(AieTargetModel targetModel) {
  return unwrap(targetModel).getLocalMemorySize();
}

uint32_t aieTargetModelGetDefaultCoreStackSize(AieTargetModel targetModel) {
  return unwrap(targetModel).getDefaultCoreStackSize();
}

uint32_t aieTargetModelGetNumLocks(AieTargetModel targetModel, int col,
                                   int row) {
  return unwrap(targetModel).getNumLocks(col, row);
}

uint32_t aieTargetModelGetNumBDs(AieTargetModel targetModel, int col, int row) {
  return unwrap(targetModel).getNumBDs(col, row);
}

uint32_t aieTargetModelGetMaxLockValue(AieTargetModel targetModel) {
  return unwrap(targetModel).getMaxLockValue();
}

uint32_t aieTargetModelGetMaxRepeatCount(AieTargetModel targetModel) {
  return unwrap(targetModel).getMaxRepeatCount();
}

uint32_t aieTargetModelGetDmaTaskQueueDepth(AieTargetModel targetModel) {
  return unwrap(targetModel).getDmaTaskQueueDepth();
}

uint32_t aieTargetModelGetDmaBdWrapBits(AieTargetModel targetModel, int col,
                                        int row) {
  return unwrap(targetModel).getDmaBdWrapBits(col, row);
}

uint32_t aieTargetModelGetDmaBdStepBits(AieTargetModel targetModel, int col,
                                        int row) {
  return unwrap(targetModel).getDmaBdStepBits(col, row);
}

uint32_t aieTargetModelGetDmaBdIterBits(AieTargetModel targetModel, int col,
                                        int row) {
  return unwrap(targetModel).getDmaBdIterBits(col, row);
}

uint32_t aieTargetModelGetNumMemTileRows(AieTargetModel targetModel) {
  return unwrap(targetModel).getNumMemTileRows();
}

uint32_t aieTargetModelGetMemTileSize(AieTargetModel targetModel) {
  return unwrap(targetModel).getMemTileSize();
}

uint32_t aieTargetModelGetNumBanks(AieTargetModel targetModel, int col,
                                   int row) {
  return unwrap(targetModel).getNumBanks(col, row);
}

bool aieDmaBdFits(uint32_t device, int col, int row, MlirType elementType,
                  int64_t offset, const int64_t *sizes, const int64_t *strides,
                  int64_t len) {
  Type elemType = unwrap(elementType);
  MLIRContext *ctx = elemType.getContext();
  ctx->getOrLoadDialect<xilinx::AIE::AIEDialect>();
  OpBuilder builder(ctx);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());
  auto dev = xilinx::AIE::DeviceOp::create(
      builder, loc, static_cast<xilinx::AIE::AIEDevice>(device),
      xilinx::AIE::DeviceOp::getDefaultDeviceName());
  const xilinx::AIE::AIETargetModel &tm = dev.getTargetModel();

  int64_t elemBits = DataLayout::closest(dev).getTypeSizeInBits(elemType);
  if (offset * elemBits % tm.getAddressGenGranularity())
    return false;

  SmallVector<xilinx::AIE::BDDimLayoutAttr> dims;
  for (unsigned d = 0; d < xilinx::AIEX::kNdDmaDims; d++) {
    if (sizes[d] < 0 || sizes[d] > UINT32_MAX || strides[d] < 0 ||
        strides[d] > UINT32_MAX)
      return false;
    dims.push_back(
        xilinx::AIE::BDDimLayoutAttr::get(ctx, sizes[d], strides[d]));
  }
  // The dimensions the lowerings see: a descriptor the normalize pass makes
  // one linear transfer is held to its length alone.
  if (std::optional<SmallVector<xilinx::AIE::BDDimLayoutAttr>> normalized =
          xilinx::AIE::normalizedBDDims(dims, len)) {
    if (normalized->empty()) {
      int64_t bits;
      return !llvm::MulOverflow(len, elemBits, bits) &&
             static_cast<uint64_t>(bits / tm.getAddressGenGranularity()) <=
                 tm.getDmaBdMaxLen(col, row);
    }
    dims = *normalized;
  }
  // Innermost first, the dimensions the descriptor lacks filled with units.
  xilinx::AIEX::NdDmaPattern pattern;
  for (unsigned d = 0; d < xilinx::AIEX::kNdDmaDims; d++) {
    bool held = d < dims.size();
    pattern.offsets.push_back(0);
    pattern.sizes.push_back(held ? dims[dims.size() - 1 - d].getSize() : 1);
    pattern.strides.push_back(held ? dims[dims.size() - 1 - d].getStride() : 0);
  }
  bool linear = xilinx::AIEX::isLinearModeTransfer(tm, col, row, pattern.sizes,
                                                   pattern.strides);
  if (!xilinx::AIEX::patternPassesVerification(
          dev, MemRefType::get({ShapedType::kDynamic}, elemType), tm, col, row,
          pattern, linear))
    return false;

  // The wrap checks leave the length field, d0 * d1 * d2 in address granules,
  // to NpuWriteBdOp's verifier.
  int64_t lenBits = elemBits;
  for (int d = 0; d < 3; d++)
    if (llvm::MulOverflow(lenBits, pattern.sizes[d], lenBits))
      return false;
  return static_cast<uint64_t>(lenBits / tm.getAddressGenGranularity()) <=
         tm.getDmaBdMaxLen(col, row);
}

uint32_t aieTargetModelGetNumDestSwitchboxConnections(
    AieTargetModel targetModel, int col, int row, uint32_t bundle) {
  xilinx::AIE::WireBundle wireBundle =
      static_cast<xilinx::AIE::WireBundle>(bundle);
  return unwrap(targetModel)
      .getNumDestSwitchboxConnections(col, row, wireBundle);
}

uint32_t aieTargetModelGetNumSourceSwitchboxConnections(
    AieTargetModel targetModel, int col, int row, uint32_t bundle) {
  xilinx::AIE::WireBundle wireBundle =
      static_cast<xilinx::AIE::WireBundle>(bundle);
  return unwrap(targetModel)
      .getNumSourceSwitchboxConnections(col, row, wireBundle);
}

uint32_t aieTargetModelGetNumDestShimMuxConnections(AieTargetModel targetModel,
                                                    int col, int row,
                                                    uint32_t bundle) {
  xilinx::AIE::WireBundle wireBundle =
      static_cast<xilinx::AIE::WireBundle>(bundle);
  return unwrap(targetModel).getNumDestShimMuxConnections(col, row, wireBundle);
}

uint32_t aieTargetModelGetNumSourceShimMuxConnections(
    AieTargetModel targetModel, int col, int row, uint32_t bundle) {
  xilinx::AIE::WireBundle wireBundle =
      static_cast<xilinx::AIE::WireBundle>(bundle);
  return unwrap(targetModel)
      .getNumSourceShimMuxConnections(col, row, wireBundle);
}

bool aieTargetModelIsNPU(AieTargetModel targetModel) {
  return unwrap(targetModel).hasProperty(xilinx::AIE::AIETargetModel::IsNPU);
}

uint32_t aieTargetModelGetTargetArch(AieTargetModel targetModel) {
  return static_cast<uint32_t>(unwrap(targetModel).getTargetArch());
}

uint32_t aieTargetModelGetColumnShift(AieTargetModel targetModel) {
  return unwrap(targetModel).getColumnShift();
}

uint32_t aieTargetModelGetRowShift(AieTargetModel targetModel) {
  return unwrap(targetModel).getRowShift();
}
