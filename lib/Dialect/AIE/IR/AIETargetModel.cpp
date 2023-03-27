//===- AIETargetModel.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"

using namespace llvm;

namespace xilinx {
namespace AIE {
AIETargetModel::~AIETargetModel() {}

///
/// AIE1 TargetModel
///

AIEArch AIE1TargetModel::getTargetArch() const { return AIEArch::AIE1; }

// Return the tile ID of the memory to the west of the given tile, if it exists.
Optional<TileID> AIE1TargetModel::getMemWest(TileID src) const {
  bool isEvenRow = ((src.second % 2) == 0);
  Optional<TileID> ret;
  if (isEvenRow)
    ret = src;
  else
    ret = std::make_pair(src.first - 1, src.second);
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
Optional<TileID> AIE1TargetModel::getMemEast(TileID src) const {
  bool isEvenRow = ((src.second % 2) == 0);
  Optional<TileID> ret;
  if (isEvenRow)
    ret = std::make_pair(src.first + 1, src.second);
  else
    ret = src;
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
Optional<TileID> AIE1TargetModel::getMemNorth(TileID src) const {
  Optional<TileID> ret = std::make_pair(src.first, src.second + 1);
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}
Optional<TileID> AIE1TargetModel::getMemSouth(TileID src) const {
  Optional<TileID> ret = std::make_pair(src.first, src.second - 1);
  // The first row doesn't have a tile memory south
  if (!isValidTile(*ret) || ret->second == 0)
    ret.reset();
  return ret;
}

bool AIE1TargetModel::isInternal(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return ((srcCol == dstCol) && (srcRow == dstRow));
}

bool AIE1TargetModel::isWest(int srcCol, int srcRow, int dstCol,
                             int dstRow) const {
  return ((srcCol == dstCol + 1) && (srcRow == dstRow));
}

bool AIE1TargetModel::isMemWest(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  bool IsEvenRow = ((srcRow % 2) == 0);
  return (IsEvenRow && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (!IsEvenRow && isWest(srcCol, srcRow, dstCol, dstRow));
}

bool AIE1TargetModel::isEast(int srcCol, int srcRow, int dstCol,
                             int dstRow) const {
  return ((srcCol == dstCol - 1) && (srcRow == dstRow));
}

bool AIE1TargetModel::isMemEast(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  bool IsEvenRow = ((srcRow % 2) == 0);
  return (!IsEvenRow && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (IsEvenRow && isEast(srcCol, srcRow, dstCol, dstRow));
}

bool AIE1TargetModel::isNorth(int srcCol, int srcRow, int dstCol,
                              int dstRow) const {
  return ((srcCol == dstCol) && (srcRow == dstRow - 1));
}

bool AIE1TargetModel::isMemNorth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE1TargetModel::isSouth(int srcCol, int srcRow, int dstCol,
                              int dstRow) const {
  return ((srcCol == dstCol) && (srcRow == dstRow + 1));
}

bool AIE1TargetModel::isMemSouth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isSouth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE1TargetModel::isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                                         int memRow) const {
  bool IsEvenRow = ((coreRow % 2) == 0);

  bool IsMemWest = (isWest(coreCol, coreRow, memCol, memRow) && !IsEvenRow) ||
                   (isInternal(coreCol, coreRow, memCol, memRow) && IsEvenRow);

  bool IsMemEast = (isEast(coreCol, coreRow, memCol, memRow) && IsEvenRow) ||
                   (isInternal(coreCol, coreRow, memCol, memRow) && !IsEvenRow);

  bool IsMemNorth = isNorth(coreCol, coreRow, memCol, memRow);
  bool IsMemSouth = isSouth(coreCol, coreRow, memCol, memRow);

  return IsMemSouth || IsMemNorth || IsMemWest || IsMemEast;
}

///
/// AIE2 TargetModel
///

AIEArch AIE2TargetModel::getTargetArch() const { return AIEArch::AIE2; }

// Return the tile ID of the memory to the west of the given tile, if it exists.
Optional<TileID> AIE2TargetModel::getMemWest(TileID src) const {
  Optional<TileID> ret = std::make_pair(src.first - 1, src.second);
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
Optional<TileID> AIE2TargetModel::getMemEast(TileID src) const {
  Optional<TileID> ret = src;
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}
// Return the tile ID of the memory to the west of the given tile, if it exists.
Optional<TileID> AIE2TargetModel::getMemNorth(TileID src) const {
  Optional<TileID> ret = std::make_pair(src.first, src.second + 1);
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}
Optional<TileID> AIE2TargetModel::getMemSouth(TileID src) const {
  Optional<TileID> ret = std::make_pair(src.first, src.second - 1);
  // The first row doesn't have a tile memory south
  if (!isValidTile(*ret) || ret->second == 0)
    ret.reset();
  return ret;
}

bool AIE2TargetModel::isInternal(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return ((srcCol == dstCol) && (srcRow == dstRow));
}

bool AIE2TargetModel::isWest(int srcCol, int srcRow, int dstCol,
                             int dstRow) const {
  return isInternal(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isMemWest(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isInternal(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isEast(int srcCol, int srcRow, int dstCol,
                             int dstRow) const {
  return ((srcCol == dstCol - 1) && (srcRow == dstRow));
}

bool AIE2TargetModel::isMemEast(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isEast(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isNorth(int srcCol, int srcRow, int dstCol,
                              int dstRow) const {
  return ((srcCol == dstCol) && (srcRow == dstRow - 1));
}

bool AIE2TargetModel::isMemNorth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isSouth(int srcCol, int srcRow, int dstCol,
                              int dstRow) const {
  return ((srcCol == dstCol) && (srcRow == dstRow + 1));
}

bool AIE2TargetModel::isMemSouth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isSouth(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                                         int memRow) const {
  bool IsMemWest = isInternal(coreCol, coreRow, memCol, memRow);
  bool IsMemEast = isEast(coreCol, coreRow, memCol, memRow);
  bool IsMemNorth = isNorth(coreCol, coreRow, memCol, memRow);
  bool IsMemSouth = isSouth(coreCol, coreRow, memCol, memRow);

  return IsMemSouth || IsMemNorth || IsMemWest || IsMemEast;
}

} // namespace AIE
} // namespace xilinx
