//===- AIETargetModel.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIE_DEVICEMODEL_H
#define MLIR_AIE_DEVICEMODEL_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"

#include "aie/Dialect/AIE/IR/AIEEnums.h"

namespace xilinx {
namespace AIE {

typedef struct TileID {
  uint32_t col;
  uint32_t row;

  inline bool operator==(const TileID &rhs) const {
    return std::tie(col, row) == std::tie(rhs.col, rhs.row);
  }

  inline bool operator!=(const TileID &rhs) const {
    return std::tie(col, row) != std::tie(rhs.col, rhs.row);
  }

  inline bool operator<(const TileID &rhs) const {
    return col == rhs.col ? row < rhs.row : col < rhs.col;
  }
} TileID;

class AIETargetModel {
public:
  AIETargetModel() {}
  virtual ~AIETargetModel();

  /// Return the target architecture.
  virtual AIEArch getTargetArch() const = 0;

  /// Return the number of columns in the device.
  virtual uint32_t columns() const = 0;

  /// Return the number of rows in the device.
  virtual uint32_t rows() const = 0;

  /// Return true if the given tile is a 'Core' tile.  These tiles
  /// include a Core, TileDMA, tile memory, and stream connections.
  virtual bool isCoreTile(uint32_t col, uint32_t row) const = 0;

  /// Return true if the given tile is an AIE2 'Memory' tile.  These tiles
  /// include a TileDMA, tile memory, and stream connections, but no core.
  virtual bool isMemTile(uint32_t col, uint32_t row) const = 0;

  /// Return true if the given tile is a Shim NOC tile.  These tiles include a
  /// ShimDMA and a connection to the memory-mapped NOC.  They do not contain
  /// any memory.
  virtual bool isShimNOCTile(uint32_t col, uint32_t row) const = 0;

  /// Return true if the given tile is a Shim PL uint32_terface tile.  These
  /// tiles do not include a ShimDMA and instead include connections to the PL.
  /// They do not contain any memory.
  virtual bool isShimPLTile(uint32_t col, uint32_t row) const = 0;

  /// Return true if the given tile is either a Shim NOC or a Shim PL interface
  /// tile.
  virtual bool isShimNOCorPLTile(uint32_t col, uint32_t row) const = 0;

  /// Return true if the given tile ID is valid.
  virtual bool isValidTile(TileID src) const {
    return (src.col >= 0) && (src.col < columns()) && (src.row >= 0) &&
           (src.row < rows());
  }

  /// Return true if the given port in the given tile is a valid destination for
  /// traces
  virtual bool isValidTraceMaster(uint32_t col, uint32_t row,
                                  WireBundle destBundle,
                                  uint32_t destIndex) const = 0;

  /// Return the tile ID of the memory to the west of the given tile, if it
  /// exists.
  virtual std::optional<TileID> getMemWest(TileID src) const = 0;
  /// Return the tile ID of the memory to the east of the given tile, if it
  /// exists.
  virtual std::optional<TileID> getMemEast(TileID src) const = 0;
  /// Return the tile ID of the memory to the north of the given tile, if it
  /// exists.
  virtual std::optional<TileID> getMemNorth(TileID src) const = 0;
  /// Return the tile ID of the memory to the south of the given tile, if it
  /// exists.
  virtual std::optional<TileID> getMemSouth(TileID src) const = 0;

  /// Return true if src is the internal memory of dst
  bool isInternal(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                  uint32_t dstRow) const {
    return ((srcCol == dstCol) && (srcRow == dstRow));
  }
  /// Return true if src is West of dst
  bool isWest(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
              uint32_t dstRow) const {
    return ((srcCol == dstCol + 1) && (srcRow == dstRow));
  }
  /// Return true if src is East of dst
  bool isEast(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
              uint32_t dstRow) const {
    return ((srcCol == dstCol - 1) && (srcRow == dstRow));
  }
  /// Return true if src is North of dst
  bool isNorth(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
               uint32_t dstRow) const {
    return ((srcCol == dstCol) && (srcRow == dstRow - 1));
  }
  /// Return true if src is South of dst
  bool isSouth(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
               uint32_t dstRow) const {
    return ((srcCol == dstCol) && (srcRow == dstRow + 1));
  }

  /// Return true if src has a memory tile which is West of dst
  virtual bool isMemWest(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                         uint32_t dstRow) const = 0;
  /// Return true if src has a memory tile which is East of dst
  virtual bool isMemEast(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                         uint32_t dstRow) const = 0;
  /// Return true if src has a memory tile which is North of dst
  virtual bool isMemNorth(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                          uint32_t dstRow) const = 0;
  /// Return true if src has a memory tile which is South of dst
  virtual bool isMemSouth(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                          uint32_t dstRow) const = 0;

  /// Return true if core can access the memory in mem
  virtual bool isLegalMemAffinity(uint32_t coreCol, uint32_t coreRow,
                                  uint32_t memCol, uint32_t memRow) const = 0;

  /// Return the base address in the local address map of differnet memories.
  virtual uint32_t getMemInternalBaseAddress(TileID src) const = 0;
  virtual uint32_t getMemSouthBaseAddress() const = 0;
  virtual uint32_t getMemWestBaseAddress() const = 0;
  virtual uint32_t getMemNorthBaseAddress() const = 0;
  virtual uint32_t getMemEastBaseAddress() const = 0;

  /// Return the size (in bytes) of the local data memory of a core.
  virtual uint32_t getLocalMemorySize() const = 0;

  /// Return the number of lock objects
  virtual uint32_t getNumLocks(uint32_t col, uint32_t row) const = 0;

  /// Return the number of buffer descriptors supported by the DMA in the given
  /// tile.
  virtual uint32_t getNumBDs(uint32_t col, uint32_t row) const = 0;

  virtual uint32_t getNumMemTileRows() const = 0;
  /// Return the size (in bytes) of a MemTile.
  virtual uint32_t getMemTileSize() const = 0;
  /// Return the number of destinations of connections inside a switchbox. These
  /// are the targets of connect operations in the switchbox.
  virtual uint32_t getNumDestSwitchboxConnections(uint32_t col, uint32_t row,
                                                  WireBundle bundle) const = 0;
  /// Return the number of sources of connections inside a switchbox.  These are
  /// the origins of connect operations in the switchbox.
  virtual uint32_t
  getNumSourceSwitchboxConnections(uint32_t col, uint32_t row,
                                   WireBundle bundle) const = 0;
  /// Return the number of destinations of connections inside a shimmux.  These
  /// are the targets of connect operations in the switchbox.
  virtual uint32_t getNumDestShimMuxConnections(uint32_t col, uint32_t row,
                                                WireBundle bundle) const = 0;
  /// Return the number of sources of connections inside a shimmux.  These are
  /// the origins of connect operations in the switchbox.
  virtual uint32_t getNumSourceShimMuxConnections(uint32_t col, uint32_t row,
                                                  WireBundle bundle) const = 0;

  // Return true if the stream switch connection is legal, false otherwise.
  virtual bool isLegalMemtileConnection(WireBundle srcBundle, uint32_t srcChan,
                                        WireBundle dstBundle,
                                        uint32_t dstChan) const = 0;

  // Run consistency checks on the target model.
  void validate() const;
};

class AIE1TargetModel : public AIETargetModel {
public:
  AIE1TargetModel() {}

  bool isCoreTile(uint32_t col, uint32_t row) const override { return row > 0; }
  bool isMemTile(uint32_t col, uint32_t row) const override { return false; }

  AIEArch getTargetArch() const override;

  std::optional<TileID> getMemWest(TileID src) const override;
  std::optional<TileID> getMemEast(TileID src) const override;
  std::optional<TileID> getMemNorth(TileID src) const override;
  std::optional<TileID> getMemSouth(TileID src) const override;

  bool isMemWest(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                 uint32_t dstRow) const override;
  bool isMemEast(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                 uint32_t dstRow) const override;
  bool isMemNorth(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                  uint32_t dstRow) const override;
  bool isMemSouth(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                  uint32_t dstRow) const override;

  bool isLegalMemAffinity(uint32_t coreCol, uint32_t coreRow, uint32_t memCol,
                          uint32_t memRow) const override;

  uint32_t getMemInternalBaseAddress(TileID src) const override {
    bool IsEvenRow = ((src.row % 2) == 0);
    if (IsEvenRow)
      // Internal is West
      return getMemWestBaseAddress();
    else
      // Internal is East
      return getMemEastBaseAddress();
  }
  uint32_t getMemSouthBaseAddress() const override { return 0x00020000; }
  uint32_t getMemWestBaseAddress() const override { return 0x00028000; }
  uint32_t getMemNorthBaseAddress() const override { return 0x00030000; }
  uint32_t getMemEastBaseAddress() const override { return 0x00038000; }
  uint32_t getLocalMemorySize() const override { return 0x00008000; }
  uint32_t getNumLocks(uint32_t col, uint32_t row) const override { return 16; }
  uint32_t getNumBDs(uint32_t col, uint32_t row) const override { return 16; }
  uint32_t getNumMemTileRows() const override { return 0; }
  uint32_t getMemTileSize() const override { return 0; }

  uint32_t getNumDestSwitchboxConnections(uint32_t col, uint32_t row,
                                          WireBundle bundle) const override;
  uint32_t getNumSourceSwitchboxConnections(uint32_t col, uint32_t row,
                                            WireBundle bundle) const override;
  uint32_t getNumDestShimMuxConnections(uint32_t col, uint32_t row,
                                        WireBundle bundle) const override;
  uint32_t getNumSourceShimMuxConnections(uint32_t col, uint32_t row,
                                          WireBundle bundle) const override;
  bool isLegalMemtileConnection(WireBundle srcBundle, uint32_t srcChan,
                                WireBundle dstBundle,
                                uint32_t dstChan) const override;

  bool isValidTraceMaster(uint32_t col, uint32_t row, WireBundle destBundle,
                          uint32_t destIndex) const override {
    if (isCoreTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::South)
      return true;
    return false;
  }
};

class AIE2TargetModel : public AIETargetModel {
public:
  AIE2TargetModel() {}

  AIEArch getTargetArch() const override;

  std::optional<TileID> getMemWest(TileID src) const override;
  std::optional<TileID> getMemEast(TileID src) const override;
  std::optional<TileID> getMemNorth(TileID src) const override;
  std::optional<TileID> getMemSouth(TileID src) const override;

  bool isMemWest(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                 uint32_t dstRow) const override;
  bool isMemEast(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                 uint32_t dstRow) const override;
  bool isMemNorth(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                  uint32_t dstRow) const override;
  bool isMemSouth(uint32_t srcCol, uint32_t srcRow, uint32_t dstCol,
                  uint32_t dstRow) const override;

  bool isLegalMemAffinity(uint32_t coreCol, uint32_t coreRow, uint32_t memCol,
                          uint32_t memRow) const override;

  uint32_t getMemInternalBaseAddress(TileID src) const override {
    return getMemEastBaseAddress();
  }
  uint32_t getMemSouthBaseAddress() const override { return 0x00040000; }
  uint32_t getMemWestBaseAddress() const override { return 0x00050000; }
  uint32_t getMemNorthBaseAddress() const override { return 0x00060000; }
  uint32_t getMemEastBaseAddress() const override { return 0x00070000; }
  uint32_t getLocalMemorySize() const override { return 0x00010000; }
  uint32_t getNumLocks(uint32_t col, uint32_t row) const override {
    return isMemTile(col, row) ? 64 : 16;
  }
  uint32_t getNumBDs(uint32_t col, uint32_t row) const override {
    return isMemTile(col, row) ? 48 : 16;
  }
  uint32_t getMemTileSize() const override { return 0x00080000; }

  uint32_t getNumDestSwitchboxConnections(uint32_t col, uint32_t row,
                                          WireBundle bundle) const override;
  uint32_t getNumSourceSwitchboxConnections(uint32_t col, uint32_t row,
                                            WireBundle bundle) const override;
  uint32_t getNumDestShimMuxConnections(uint32_t col, uint32_t row,
                                        WireBundle bundle) const override;
  uint32_t getNumSourceShimMuxConnections(uint32_t col, uint32_t row,
                                          WireBundle bundle) const override;
  bool isLegalMemtileConnection(WireBundle srcBundle, uint32_t srcChan,
                                WireBundle dstBundle,
                                uint32_t dstChan) const override;
};

class VC1902TargetModel : public AIE1TargetModel {
  llvm::SmallDenseSet<unsigned, 16> noc_columns = {
      2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47};

public:
  VC1902TargetModel() {}

  uint32_t columns() const override { return 50; }
  uint32_t rows() const override {
    return 9; /* One Shim row and 8 Core rows. */
  }

  bool isShimNOCTile(uint32_t col, uint32_t row) const override {
    return row == 0 && noc_columns.contains(col);
  }
  bool isShimPLTile(uint32_t col, uint32_t row) const override {
    return row == 0 && !noc_columns.contains(col);
  }
  bool isShimNOCorPLTile(uint32_t col, uint32_t row) const override {
    return isShimNOCTile(col, row) || isShimPLTile(col, row);
  }
};

class VE2302TargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 8> noc_columns = {2, 3, 6, 7, 10, 11};

public:
  VE2302TargetModel() {}

  uint32_t columns() const override { return 17; }
  uint32_t rows() const override {
    return 4; /* One Shim row, 1 memtile rows, and 2 Core rows. */
  }

  bool isCoreTile(uint32_t col, uint32_t row) const override { return row > 1; }
  bool isMemTile(uint32_t col, uint32_t row) const override { return row == 1; }
  bool isShimNOCTile(uint32_t col, uint32_t row) const override {
    return row == 0 && noc_columns.contains(col);
  }
  bool isShimPLTile(uint32_t col, uint32_t row) const override {
    return row == 0 && !noc_columns.contains(col);
  }
  bool isShimNOCorPLTile(uint32_t col, uint32_t row) const override {
    return isShimNOCTile(col, row) || isShimPLTile(col, row);
  }
  uint32_t getNumMemTileRows() const override { return 1; }
  bool isValidTraceMaster(uint32_t col, uint32_t row, WireBundle destBundle,
                          uint32_t destIndex) const override {
    if (isCoreTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isCoreTile(col, row) && destBundle == WireBundle::DMA &&
             destIndex == 0)
      return true;
    else if (isMemTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isMemTile(col, row) && destBundle == WireBundle::DMA &&
             destIndex == 5)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::West &&
             destIndex == 0)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::East &&
             destIndex == 0)
      return true;
    return false;
  }
};

class VE2802TargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 16> noc_columns = {2,  3,  6,  7,  14, 15,
                                                   22, 23, 30, 31, 34, 35};

public:
  VE2802TargetModel() {}

  uint32_t columns() const override { return 38; }
  uint32_t rows() const override {
    return 11; /* One Shim row, 2 memtile rows, and 8 Core rows. */
  }

  bool isCoreTile(uint32_t col, uint32_t row) const override { return row > 2; }
  bool isMemTile(uint32_t col, uint32_t row) const override {
    return (row == 1) || (row == 2);
  }
  bool isShimNOCTile(uint32_t col, uint32_t row) const override {
    return row == 0 && noc_columns.contains(col);
  }
  bool isShimPLTile(uint32_t col, uint32_t row) const override {
    return row == 0 && !noc_columns.contains(col);
  }
  bool isShimNOCorPLTile(uint32_t col, uint32_t row) const override {
    return isShimNOCTile(col, row) || isShimPLTile(col, row);
  }
  uint32_t getNumMemTileRows() const override { return 2; }
  bool isValidTraceMaster(uint32_t col, uint32_t row, WireBundle destBundle,
                          uint32_t destIndex) const override {
    if (isCoreTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isCoreTile(col, row) && destBundle == WireBundle::DMA &&
             destIndex == 0)
      return true;
    else if (isMemTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isMemTile(col, row) && destBundle == WireBundle::DMA &&
             destIndex == 5)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::West &&
             destIndex == 0)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::East &&
             destIndex == 0)
      return true;
    return false;
  }
};

class IPUTargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 16> noc_columns = {0, 1, 2, 3};

public:
  IPUTargetModel() {}

  uint32_t columns() const override { return 5; }
  uint32_t rows() const override {
    return 6; /* 1 Shim row, 1 memtile row, and 4 Core rows. */
  }

  bool isCoreTile(uint32_t col, uint32_t row) const override { return row > 1; }
  bool isMemTile(uint32_t col, uint32_t row) const override { return row == 1; }

  bool isShimNOCTile(uint32_t col, uint32_t row) const override {
    return row == 0 && noc_columns.contains(col);
  }
  bool isShimPLTile(uint32_t col, uint32_t row) const override {
    return row == 0 && !noc_columns.contains(col);
  }
  bool isShimNOCorPLTile(uint32_t col, uint32_t row) const override {
    return isShimNOCTile(col, row) || isShimPLTile(col, row);
  }
  uint32_t getNumMemTileRows() const override { return 1; }
  bool isValidTraceMaster(uint32_t col, uint32_t row, WireBundle destBundle,
                          uint32_t destIndex) const override {
    if (isCoreTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isCoreTile(col, row) && destBundle == WireBundle::DMA &&
             destIndex == 0)
      return true;
    else if (isMemTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isMemTile(col, row) && destBundle == WireBundle::DMA &&
             destIndex == 5)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::South)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::West &&
             destIndex == 0)
      return true;
    else if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::East &&
             destIndex == 0)
      return true;
    return false;
  }
};

} // namespace AIE
} // namespace xilinx

namespace llvm {
using namespace xilinx::AIE;
template <> struct DenseMapInfo<TileID> {
  using FirstInfo = DenseMapInfo<uint32_t>;
  using SecondInfo = DenseMapInfo<uint32_t>;

  static inline TileID getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static inline TileID getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const TileID &t) {
    return detail::combineHashValue(FirstInfo::getHashValue(t.col),
                                    SecondInfo::getHashValue(t.row));
  }

  static bool isEqual(const TileID &lhs, const TileID &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif
