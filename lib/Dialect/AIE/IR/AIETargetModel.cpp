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
uint32_t
AIE1TargetModel::getNumDestSwitchboxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      return 6;
    case WireBundle::West:
      if (col == 0)
        return 0;
      else
        return 4;
    case WireBundle::South:
      return 6;
    case WireBundle::East:
      if (col == columns() - 1)
        return 0;
      else
        return 4;
    default:
      return 0;
    }
  else
    switch (bundle) {
    case WireBundle::Core:
      return 2;
    case WireBundle::DMA:
      return 2;
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      if (row == rows() - 1)
        return 0;
      else
        return 6;
    case WireBundle::West:
      if (col == 0)
        return 0;
      else
        return 4;
    case WireBundle::South:
      return 4;
    case WireBundle::East:
      if (col == columns() - 1)
        return 0;
      else
        return 4;
    default:
      return 0;
    }
}
uint32_t
AIE1TargetModel::getNumSourceSwitchboxConnections(int col, int row,
                                                  WireBundle bundle) const {
  if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      return 4;
    case WireBundle::West:
      if (col == 0)
        return 0;
      else
        return 4;
    case WireBundle::South:
      return 8;
    case WireBundle::East:
      if (col == columns() - 1)
        return 0;
      else
        return 4;
    case WireBundle::Trace:
      return 1;
    default:
      return 0;
    }
  else
    switch (bundle) {
    case WireBundle::Core:
      return 2;
    case WireBundle::DMA:
      return 2;
    case WireBundle::FIFO:
      return 2;
    case WireBundle::North:
      if (row == rows() - 1)
        return 0;
      else
        return 4;
    case WireBundle::West:
      if (col == 0)
        return 0;
      else
        return 4;
    case WireBundle::South:
      return 6;
    case WireBundle::East:
      if (col == columns() - 1)
        return 0;
      else
        return 4;
    case WireBundle::Trace:
      return 2;
    default:
      return 0;
    }
}
uint32_t
AIE1TargetModel::getNumDestShimMuxConnections(int col, int row,
                                              WireBundle bundle) const {
  if (isShimNOCTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 6;
    case WireBundle::South:
      return 8; // Connection to the south port of the stream switch
    default:
      return 0;
    }
  else
    return 0;
}
uint32_t
AIE1TargetModel::getNumSourceShimMuxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isShimNOCTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 6;
    case WireBundle::South:
      return 6;
    default:
      return 0;
    }
  else
    return 0;
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
  // Memtiles don't have memory adjacency to neighboring core tiles.
  if (!isValidTile(*ret) || ret->second == 0 ||
      isMemTile(ret->first, ret->second))
    ret.reset();
  return ret;
}

bool AIE2TargetModel::isInternal(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return ((srcCol == dstCol) && (srcRow == dstRow));
}

bool AIE2TargetModel::isWest(int srcCol, int srcRow, int dstCol,
                             int dstRow) const {
  return ((srcCol == dstCol + 1) && (srcRow == dstRow));
}

bool AIE2TargetModel::isMemWest(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isWest(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isEast(int srcCol, int srcRow, int dstCol,
                             int dstRow) const {
  return isInternal(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isMemEast(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isInternal(srcCol, srcRow, dstCol, dstRow);
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

  bool IsMemWest = isMemWest(coreCol, coreRow, memCol, memRow);
  bool IsMemEast = isMemEast(coreCol, coreRow, memCol, memRow);
  bool IsMemNorth = isMemNorth(coreCol, coreRow, memCol, memRow);
  bool IsMemSouth = isMemSouth(coreCol, coreRow, memCol, memRow);

  if (isMemTile(coreCol, coreRow))
    return IsMemSouth || (IsMemNorth && isMemTile(memCol, memRow)) ||
           IsMemWest || IsMemEast;
  else
    return (IsMemSouth && !isMemTile(memCol, memRow)) || IsMemNorth ||
           IsMemWest || IsMemEast;
}
uint32_t
AIE2TargetModel::getNumDestSwitchboxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isMemTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 6;
    case WireBundle::North:
      return 6;
    case WireBundle::South:
      return 4;
    default:
      return 0;
    }
  else if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 1;
    case WireBundle::North:
      return 6;
    case WireBundle::West:
      if (col == 0)
        return 0;
      else
        return 4;
    case WireBundle::South:
      return 6;
    case WireBundle::East:
      if (col == columns() - 1)
        return 0;
      else
        return 4;
    default:
      return 0;
    }
  else
    switch (bundle) {
    case WireBundle::Core:
      return 1;
    case WireBundle::DMA:
      return 2;
    case WireBundle::FIFO:
      return 1;
    case WireBundle::North:
      if (row == rows() - 1)
        return 0;
      else
        return 6;
    case WireBundle::West:
      if (col == 0)
        return 0;
      else
        return 4;
    case WireBundle::South:
      return 4;
    case WireBundle::East:
      if (col == columns() - 1)
        return 0;
      else
        return 4;
    default:
      return 0;
    }
}
uint32_t
AIE2TargetModel::getNumSourceSwitchboxConnections(int col, int row,
                                                  WireBundle bundle) const {
  if (isMemTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 6;
    case WireBundle::North:
      return 4;
    case WireBundle::South:
      return 6;
    case WireBundle::Trace:
      return 1;
    default:
      return 0;
    }
  else if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 1;
    case WireBundle::North:
      return 4;
    case WireBundle::West:
      if (col == 0)
        return 0;
      else
        return 4;
    case WireBundle::South:
      return 8;
    case WireBundle::East:
      if (col == columns() - 1)
        return 0;
      else
        return 4;
    case WireBundle::Trace:
      return 1;
    default:
      return 0;
    }
  else
    switch (bundle) {
    case WireBundle::Core:
      return 1;
    case WireBundle::DMA:
      return 2;
    case WireBundle::FIFO:
      return 1;
    case WireBundle::North:
      if (row == rows() - 1)
        return 0;
      else
        return 4;
    case WireBundle::West:
      if (col == 0)
        return 0;
      else
        return 4;
    case WireBundle::South:
      return 6;
    case WireBundle::East:
      if (col == columns() - 1)
        return 0;
      else
        return 4;
    case WireBundle::Trace:
      return 1;
    default:
      return 0;
    }
}
uint32_t
AIE2TargetModel::getNumDestShimMuxConnections(int col, int row,
                                              WireBundle bundle) const {
  if (isShimNOCTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 6;
    case WireBundle::South:
      return 8;
    default:
      return 0;
    }
  else
    return 0;
}
uint32_t
AIE2TargetModel::getNumSourceShimMuxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isShimNOCTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 6;
    case WireBundle::South:
      return 6;
    default:
      return 0;
    }
  else
    return 0;
}

void AIETargetModel::validate() const {
  // Every tile in a shimtile row must be a shimtile, and can only be one type
  // of shim tile.
  for (int j = 0; j < columns(); j++) {
    assert(!isMemTile(j, 0) && (isShimPLTile(j, 0) || isShimNOCTile(j, 0)) &&
           !isCoreTile(j, 0));
    assert(isShimPLTile(j, 0) ^ isShimNOCTile(j, 0));
  }

  // Every tile in a memtile row must be a memtile.
  for (int i = 1; i < 1 + (int)getNumMemTileRows(); i++)
    for (int j = 0; j < columns(); j++)
      assert(isMemTile(j, i) && !isShimPLTile(j, i) && !isShimNOCTile(j, i) &&
             !isCoreTile(j, i));

  // Every other tile is a coretile.
  for (int i = 1 + (int)getNumMemTileRows(); i < rows(); i++)
    for (int j = 0; j < columns(); j++)
      assert(!isMemTile(j, i) && !isShimPLTile(j, i) && !isShimNOCTile(j, i) &&
             isCoreTile(j, i));

  // Looking North, busses must match
  for (int i = 0; i < rows() - 1; i++)
    for (int j = 0; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::North) ==
             getNumDestSwitchboxConnections(j, i + 1, WireBundle::South));
  // Looking South, busses must match
  for (int i = 1; i < rows(); i++)
    for (int j = 0; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::South) ==
             getNumDestSwitchboxConnections(j, i - 1, WireBundle::North));
  // Looking East, busses must match
  for (int i = 0; i < rows(); i++)
    for (int j = 0; j < columns() - 1; j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::East) ==
             getNumDestSwitchboxConnections(j + 1, i, WireBundle::West));
  // Looking West, busses must match
  for (int i = 0; i < rows(); i++)
    for (int j = 1; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::West) ==
             getNumDestSwitchboxConnections(j - 1, i, WireBundle::East));
  // Edges have no connections
  for (int j = 0; j < columns(); j++)
    assert(getNumSourceSwitchboxConnections(j, rows() - 1, WireBundle::North) ==
           0);
  for (int i = 0; i < rows(); i++)
    assert(getNumSourceSwitchboxConnections(columns() - 1, i,
                                            WireBundle::East) == 0);
  for (int i = 0; i < rows(); i++)
    assert(getNumSourceSwitchboxConnections(0, i, WireBundle::West) == 0);

  // FIFOS are consistent
  for (int i = 0; i < rows(); i++)
    for (int j = 0; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::FIFO) ==
             getNumDestSwitchboxConnections(j, i, WireBundle::FIFO));
}

} // namespace AIE
} // namespace xilinx
