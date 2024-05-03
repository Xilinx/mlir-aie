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
#include "llvm/ADT/SmallSet.h"

using namespace llvm;

namespace xilinx {
namespace AIE {
AIETargetModel::~AIETargetModel() = default;

///
/// AIE1 TargetModel
///

AIEArch AIE1TargetModel::getTargetArch() const { return AIEArch::AIE1; }

// Return the tile ID of the memory to the west of the given tile, if it exists.
std::optional<TileID> AIE1TargetModel::getMemWest(TileID src) const {
  bool isEvenRow = ((src.row % 2) == 0);
  std::optional<TileID> ret;
  if (isEvenRow)
    ret = src;
  else
    ret = {src.col - 1, src.row};
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the west of the given tile, if it exists.
std::optional<TileID> AIE1TargetModel::getMemEast(TileID src) const {
  bool isEvenRow = (src.row % 2) == 0;
  std::optional<TileID> ret;
  if (isEvenRow)
    ret = {src.col + 1, src.row};
  else
    ret = src;
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the west of the given tile, if it exists.
std::optional<TileID> AIE1TargetModel::getMemNorth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row + 1});
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

std::optional<TileID> AIE1TargetModel::getMemSouth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row - 1});
  // The first row doesn't have a tile memory south
  if (!isValidTile(*ret) || ret->row == 0)
    ret.reset();
  return ret;
}

bool AIE1TargetModel::isMemWest(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  bool IsEvenRow = (srcRow % 2) == 0;
  return (IsEvenRow && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (!IsEvenRow && isWest(srcCol, srcRow, dstCol, dstRow));
}

bool AIE1TargetModel::isMemEast(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  bool IsEvenRow = (srcRow % 2) == 0;
  return (!IsEvenRow && isInternal(srcCol, srcRow, dstCol, dstRow)) ||
         (IsEvenRow && isEast(srcCol, srcRow, dstCol, dstRow));
}

bool AIE1TargetModel::isMemNorth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
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
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 4;
    }
    case WireBundle::South:
      return 6;
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 4;
    }
    case WireBundle::Ctrl:
      return isShimNOCTile(col, row) ? 1 : 0;
    default:
      return 0;
    }

  switch (bundle) {
  case WireBundle::Core:
  case WireBundle::DMA:
  case WireBundle::FIFO:
    return 2;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 6;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 4;
  }
  case WireBundle::South:
    return 4;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 4;
  }
  case WireBundle::Ctrl:
    return 1;
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
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 4;
    }
    case WireBundle::South:
      return 8;
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 4;
    }
    case WireBundle::Trace:
      return 1;
    case WireBundle::Ctrl:
      return isShimNOCTile(col, row) ? 1 : 0;
    default:
      return 0;
    }

  switch (bundle) {
  case WireBundle::Core:
  case WireBundle::DMA:
  case WireBundle::FIFO:
    return 2;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 4;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 4;
  }
  case WireBundle::South:
    return 6;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 4;
  }
  case WireBundle::Trace:
    return 2;
  case WireBundle::Ctrl:
    return 1;
  default:
    return 0;
  }
}
uint32_t
AIE1TargetModel::getNumDestShimMuxConnections(int col, int row,
                                              WireBundle bundle) const {
  if (isShimNOCorPLTile(col, row))
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
    case WireBundle::South:
      return 6;
    default:
      return 0;
    }
  return 0;
}

bool AIE1TargetModel::isLegalMemtileConnection(WireBundle srcBundle,
                                               int srcChan,
                                               WireBundle dstBundle,
                                               int dstChan) const {
  return false;
}

///
/// AIE2 TargetModel
///

AIEArch AIE2TargetModel::getTargetArch() const { return AIEArch::AIE2; }

// Return the tile ID of the memory to the west of the given tile, if it exists.
std::optional<TileID> AIE2TargetModel::getMemWest(TileID src) const {
  std::optional<TileID> ret({src.col - 1, src.row});
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the east of the given tile (ie self), if
// it exists.
std::optional<TileID> AIE2TargetModel::getMemEast(TileID src) const {
  std::optional ret = src;
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

// Return the tile ID of the memory to the north of the given tile, if it
// exists.
std::optional<TileID> AIE2TargetModel::getMemNorth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row + 1});
  if (!isValidTile(*ret))
    ret.reset();
  return ret;
}

std::optional<TileID> AIE2TargetModel::getMemSouth(TileID src) const {
  std::optional<TileID> ret({src.col, src.row - 1});
  // The first row doesn't have a tile memory south
  // Memtiles don't have memory adjacency to neighboring core tiles.
  if (!isValidTile(*ret) || ret->row == 0 || isMemTile(ret->col, ret->row))
    ret.reset();
  return ret;
}

bool AIE2TargetModel::isMemWest(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isWest(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isMemEast(int srcCol, int srcRow, int dstCol,
                                int dstRow) const {
  return isInternal(srcCol, srcRow, dstCol, dstRow);
}

bool AIE2TargetModel::isMemNorth(int srcCol, int srcRow, int dstCol,
                                 int dstRow) const {
  return isNorth(srcCol, srcRow, dstCol, dstRow);
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
    return isEast(coreCol, coreRow, memCol, memRow) ||
           isInternal(coreCol, coreRow, memCol, memRow) ||
           isWest(coreCol, coreRow, memCol, memRow);
  return (IsMemSouth && !isMemTile(memCol, memRow)) || IsMemNorth ||
         IsMemWest || IsMemEast;
}

uint32_t
AIE2TargetModel::getNumDestSwitchboxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isMemTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
    case WireBundle::North:
      return 6;
    case WireBundle::South:
      return 4;
    case WireBundle::Ctrl:
      return 1;
    default:
      return 0;
    }

  if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 1;
    case WireBundle::North:
      return 6;
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 4;
    }
    case WireBundle::South:
      return 6;
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 4;
    }
    case WireBundle::Ctrl:
      return isShimNOCTile(col, row) ? 1 : 0;
    default:
      return 0;
    }

  switch (bundle) {
  case WireBundle::Core:
    return 1;
  case WireBundle::DMA:
    return 2;
  case WireBundle::FIFO:
    return 1;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 6;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 4;
  }
  case WireBundle::South:
    return 4;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 4;
  }
  case WireBundle::Ctrl:
    return 1;
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
    case WireBundle::Ctrl:
      return 1;
    default:
      return 0;
    }

  if (isShimNOCTile(col, row) || isShimPLTile(col, row))
    switch (bundle) {
    case WireBundle::FIFO:
      return 1;
    case WireBundle::North:
      return 4;
    case WireBundle::West: {
      if (col == 0)
        return 0;
      return 4;
    }
    case WireBundle::South:
      return 8;
    case WireBundle::East: {
      if (col == columns() - 1)
        return 0;
      return 4;
    }
    case WireBundle::Trace:
      return 1;
    case WireBundle::Ctrl:
      return isShimNOCTile(col, row) ? 1 : 0;
    default:
      return 0;
    }

  // compute/core tile
  switch (bundle) {
  case WireBundle::Core:
    return 1;
  case WireBundle::DMA:
    return 2;
  case WireBundle::FIFO:
    return 1;
  case WireBundle::North: {
    if (row == rows() - 1)
      return 0;
    return 4;
  }
  case WireBundle::West: {
    if (col == 0)
      return 0;
    return 4;
  }
  case WireBundle::South:
    return 6;
  case WireBundle::East: {
    if (col == columns() - 1)
      return 0;
    return 4;
  }
  case WireBundle::Trace:
    // Port 0: core trace. Port 1: memory trace.
    return 2;
  case WireBundle::Ctrl:
    return 1;
  default:
    return 0;
  }
}

uint32_t
AIE2TargetModel::getNumDestShimMuxConnections(int col, int row,
                                              WireBundle bundle) const {
  if (isShimNOCorPLTile(col, row))
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
    case WireBundle::South:
      return 6;
    default:
      return 0;
    }

  return 0;
}

bool AIE2TargetModel::isLegalMemtileConnection(WireBundle srcBundle,
                                               int srcChan,
                                               WireBundle dstBundle,
                                               int dstChan) const {
  // Memtile north-south stream switch constraint
  // Memtile stream interconnect master South and slave North must have equal
  // channel indices
  if (srcBundle == WireBundle::North && dstBundle == WireBundle::South &&
      srcChan != dstChan)
    return false;
  if (srcBundle == WireBundle::South && dstBundle == WireBundle::North &&
      srcChan != dstChan)
    return false;
  // Memtile has no east or west connections
  if (srcBundle == WireBundle::East)
    return false;
  if (srcBundle == WireBundle::West)
    return false;
  if (dstBundle == WireBundle::East)
    return false;
  if (dstBundle == WireBundle::West)
    return false;
  return true;
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
  for (int i = 1; i < 1 + static_cast<int>(getNumMemTileRows()); i++)
    for (int j = 0; j < columns(); j++)
      assert(isMemTile(j, i) && !isShimPLTile(j, i) && !isShimNOCTile(j, i) &&
             !isCoreTile(j, i));

  // Every other tile is a coretile.
  for (int i = 1 + getNumMemTileRows(); i < rows(); i++)
    for (int j = 0; j < columns(); j++)
      assert(!isMemTile(j, i) && !isShimPLTile(j, i) && !isShimNOCTile(j, i) &&
             isCoreTile(j, i));

  // Looking North, buses must match
  for (int i = 0; i < rows() - 1; i++)
    for (int j = 0; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::North) ==
             getNumDestSwitchboxConnections(j, i + 1, WireBundle::South));
  // Looking South, buses must match
  for (int i = 1; i < rows(); i++)
    for (int j = 0; j < columns(); j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::South) ==
             getNumDestSwitchboxConnections(j, i - 1, WireBundle::North));
  // Looking East, buses must match
  for (int i = 0; i < rows(); i++)
    for (int j = 0; j < columns() - 1; j++)
      assert(getNumSourceSwitchboxConnections(j, i, WireBundle::East) ==
             getNumDestSwitchboxConnections(j + 1, i, WireBundle::West));
  // Looking West, buses must match
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
