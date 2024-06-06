//===- TargetModel.cpp - C API for AIE TargetModel ------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie-c/TargetModel.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

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

uint32_t aieGetTargetModelAddressGenGranularity(AieTargetModel targetModel){
  return unwrap(targetModel).getAddressGenGranularity();
}

int aieTargetModelColumns(AieTargetModel targetModel) {
  return unwrap(targetModel).columns();
}

int aieTargetModelRows(AieTargetModel targetModel) {
  return unwrap(targetModel).rows();
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

uint32_t aieTargetModelGetNumLocks(AieTargetModel targetModel, int col,
                                   int row) {
  return unwrap(targetModel).getNumLocks(col, row);
}

uint32_t aieTargetModelGetNumBDs(AieTargetModel targetModel, int col, int row) {
  return unwrap(targetModel).getNumBDs(col, row);
}

uint32_t aieTargetModelGetNumMemTileRows(AieTargetModel targetModel) {
  return unwrap(targetModel).getNumMemTileRows();
}

uint32_t aieTargetModelGetMemTileSize(AieTargetModel targetModel) {
  return unwrap(targetModel).getMemTileSize();
}

bool aieTargetModelIsNPU(AieTargetModel targetModel) {
  return unwrap(targetModel).isNPU();
}