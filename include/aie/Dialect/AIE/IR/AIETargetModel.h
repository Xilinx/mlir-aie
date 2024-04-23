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

#include "aie/Dialect/AIE/IR/AIEEnums.h"

#include "llvm/ADT/DenseSet.h"

#include <iostream>

namespace xilinx::AIE {

using TileID = struct TileID {
  // friend definition (will define the function as a non-member function in the
  // namespace surrounding the class).
  friend std::ostream &operator<<(std::ostream &os, const TileID &s) {
    os << "TileID(" << s.col << ", " << s.row << ")";
    return os;
  }

  friend std::string to_string(const TileID &s) {
    std::ostringstream ss;
    ss << s;
    return ss.str();
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TileID &s) {
    os << to_string(s);
    return os;
  }

  // Imposes a lexical order on TileIDs.
  inline bool operator<(const TileID &rhs) const {
    return std::tie(col, row) < std::tie(rhs.col, rhs.row);
  }

  bool operator==(const TileID &rhs) const {
    return std::tie(col, row) == std::tie(rhs.col, rhs.row);
  }

  bool operator!=(const TileID &rhs) const { return !(*this == rhs); }

  int col, row;
};

class AIETargetModel {
public:
  AIETargetModel() = default;

  virtual ~AIETargetModel();

  /// Return the target architecture.
  virtual AIEArch getTargetArch() const = 0;

  /// Return the number of columns in the device.
  virtual int columns() const = 0;

  /// Return the number of rows in the device.
  virtual int rows() const = 0;

  /// Return true if the given tile is a 'Core' tile.  These tiles
  /// include a Core, TileDMA, tile memory, and stream connections.
  virtual bool isCoreTile(int col, int row) const = 0;

  /// Return true if the given tile is an AIE2 'Memory' tile.  These tiles
  /// include a TileDMA, tile memory, and stream connections, but no core.
  virtual bool isMemTile(int col, int row) const = 0;

  /// Return true if the given tile is a Shim NOC tile.  These tiles include a
  /// ShimDMA and a connection to the memory-mapped NOC.  They do not contain
  /// any memory.
  virtual bool isShimNOCTile(int col, int row) const = 0;

  /// Return true if the given tile is a Shim PL interface tile.  These
  /// tiles do not include a ShimDMA and instead include connections to the PL.
  /// They do not contain any memory.
  virtual bool isShimPLTile(int col, int row) const = 0;

  /// Return true if the given tile is either a Shim NOC or a Shim PL interface
  /// tile.
  virtual bool isShimNOCorPLTile(int col, int row) const = 0;

  /// Return true if the given tile ID is valid.
  virtual bool isValidTile(TileID src) const {
    return src.col >= 0 && src.col < columns() && src.row >= 0 &&
           src.row < rows();
  }

  /// Return true if the given port in the given tile is a valid destination for
  /// traces
  virtual bool isValidTraceMaster(int col, int row, WireBundle destBundle,
                                  int destIndex) const = 0;

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
  bool isInternal(int srcCol, int srcRow, int dstCol, int dstRow) const {
    return srcCol == dstCol && srcRow == dstRow;
  }

  /// Return true if src is West of dst
  bool isWest(int srcCol, int srcRow, int dstCol, int dstRow) const {
    return srcCol == dstCol + 1 && srcRow == dstRow;
  }

  /// Return true if src is East of dst
  bool isEast(int srcCol, int srcRow, int dstCol, int dstRow) const {
    return srcCol == dstCol - 1 && srcRow == dstRow;
  }

  /// Return true if src is North of dst
  bool isNorth(int srcCol, int srcRow, int dstCol, int dstRow) const {
    return srcCol == dstCol && srcRow == dstRow - 1;
  }

  /// Return true if src is South of dst
  bool isSouth(int srcCol, int srcRow, int dstCol, int dstRow) const {
    return srcCol == dstCol && srcRow == dstRow + 1;
  }

  /// Return true if src has a memory tile which is West of dst
  virtual bool isMemWest(int srcCol, int srcRow, int dstCol,
                         int dstRow) const = 0;
  /// Return true if src has a memory tile which is East of dst
  virtual bool isMemEast(int srcCol, int srcRow, int dstCol,
                         int dstRow) const = 0;
  /// Return true if src has a memory tile which is North of dst
  virtual bool isMemNorth(int srcCol, int srcRow, int dstCol,
                          int dstRow) const = 0;
  /// Return true if src has a memory tile which is South of dst
  virtual bool isMemSouth(int srcCol, int srcRow, int dstCol,
                          int dstRow) const = 0;

  /// Return true if core can access the memory in mem
  virtual bool isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                                  int memRow) const = 0;

  /// Return the base address in the local address map of different memories.
  virtual uint32_t getMemInternalBaseAddress(TileID src) const = 0;
  virtual uint32_t getMemSouthBaseAddress() const = 0;
  virtual uint32_t getMemWestBaseAddress() const = 0;
  virtual uint32_t getMemNorthBaseAddress() const = 0;
  virtual uint32_t getMemEastBaseAddress() const = 0;

  /// Return the size (in bytes) of the local data memory of a core.
  virtual uint32_t getLocalMemorySize() const = 0;

  /// Return the number of lock objects
  virtual uint32_t getNumLocks(int col, int row) const = 0;

  /// Return the number of buffer descriptors supported by the DMA in the given
  /// tile.
  virtual uint32_t getNumBDs(int col, int row) const = 0;

  virtual uint32_t getNumMemTileRows() const = 0;
  /// Return the size (in bytes) of a MemTile.
  virtual uint32_t getMemTileSize() const = 0;
  /// Return the number of destinations of connections inside a switchbox. These
  /// are the targets of connect operations in the switchbox.
  virtual uint32_t getNumDestSwitchboxConnections(int col, int row,
                                                  WireBundle bundle) const = 0;
  /// Return the number of sources of connections inside a switchbox.  These are
  /// the origins of connect operations in the switchbox.
  virtual uint32_t
  getNumSourceSwitchboxConnections(int col, int row,
                                   WireBundle bundle) const = 0;
  /// Return the number of destinations of connections inside a shimmux.  These
  /// are the targets of connect operations in the switchbox.
  virtual uint32_t getNumDestShimMuxConnections(int col, int row,
                                                WireBundle bundle) const = 0;
  /// Return the number of sources of connections inside a shimmux.  These are
  /// the origins of connect operations in the switchbox.
  virtual uint32_t getNumSourceShimMuxConnections(int col, int row,
                                                  WireBundle bundle) const = 0;

  // Return true if the stream switch connection is legal, false otherwise.
  virtual bool isLegalMemtileConnection(WireBundle srcBundle, int srcChan,
                                        WireBundle dstBundle,
                                        int dstChan) const = 0;

  // Run consistency checks on the target model.
  void validate() const;
};

class AIE1TargetModel : public AIETargetModel {
public:
  AIE1TargetModel() = default;

  bool isCoreTile(int col, int row) const override { return row > 0; }
  bool isMemTile(int col, int row) const override { return false; }

  AIEArch getTargetArch() const override;

  std::optional<TileID> getMemWest(TileID src) const override;
  std::optional<TileID> getMemEast(TileID src) const override;
  std::optional<TileID> getMemNorth(TileID src) const override;
  std::optional<TileID> getMemSouth(TileID src) const override;

  bool isMemWest(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemEast(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemNorth(int srcCol, int srcRow, int dstCol,
                  int dstRow) const override;
  bool isMemSouth(int srcCol, int srcRow, int dstCol,
                  int dstRow) const override;

  bool isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                          int memRow) const override;

  uint32_t getMemInternalBaseAddress(TileID src) const override {
    if (src.row % 2 == 0)
      // Internal is West
      return getMemWestBaseAddress();
    // Internal is East
    return getMemEastBaseAddress();
  }

  uint32_t getMemSouthBaseAddress() const override { return 0x00020000; }
  uint32_t getMemWestBaseAddress() const override { return 0x00028000; }
  uint32_t getMemNorthBaseAddress() const override { return 0x00030000; }
  uint32_t getMemEastBaseAddress() const override { return 0x00038000; }
  uint32_t getLocalMemorySize() const override { return 0x00008000; }
  uint32_t getNumLocks(int col, int row) const override { return 16; }
  uint32_t getNumBDs(int col, int row) const override { return 16; }
  uint32_t getNumMemTileRows() const override { return 0; }
  uint32_t getMemTileSize() const override { return 0; }

  uint32_t getNumDestSwitchboxConnections(int col, int row,
                                          WireBundle bundle) const override;
  uint32_t getNumSourceSwitchboxConnections(int col, int row,
                                            WireBundle bundle) const override;
  uint32_t getNumDestShimMuxConnections(int col, int row,
                                        WireBundle bundle) const override;
  uint32_t getNumSourceShimMuxConnections(int col, int row,
                                          WireBundle bundle) const override;
  bool isLegalMemtileConnection(WireBundle srcBundle, int srcChan,
                                WireBundle dstBundle,
                                int dstChan) const override;

  bool isValidTraceMaster(int col, int row, WireBundle destBundle,
                          int destIndex) const override {
    if (isCoreTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::South)
      return true;
    return false;
  }
};

class AIE2TargetModel : public AIETargetModel {
public:
  AIE2TargetModel() = default;

  AIEArch getTargetArch() const override;

  std::optional<TileID> getMemWest(TileID src) const override;
  std::optional<TileID> getMemEast(TileID src) const override;
  std::optional<TileID> getMemNorth(TileID src) const override;
  std::optional<TileID> getMemSouth(TileID src) const override;

  bool isMemWest(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemEast(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemNorth(int srcCol, int srcRow, int dstCol,
                  int dstRow) const override;
  bool isMemSouth(int srcCol, int srcRow, int dstCol,
                  int dstRow) const override;

  bool isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                          int memRow) const override;

  uint32_t getMemInternalBaseAddress(TileID src) const override {
    return getMemEastBaseAddress();
  }

  uint32_t getMemSouthBaseAddress() const override { return 0x00040000; }
  uint32_t getMemWestBaseAddress() const override { return 0x00050000; }
  uint32_t getMemNorthBaseAddress() const override { return 0x00060000; }
  uint32_t getMemEastBaseAddress() const override { return 0x00070000; }
  uint32_t getLocalMemorySize() const override { return 0x00010000; }

  uint32_t getNumLocks(int col, int row) const override {
    return isMemTile(col, row) ? 64 : 16;
  }

  uint32_t getNumBDs(int col, int row) const override {
    return isMemTile(col, row) ? 48 : 16;
  }

  uint32_t getMemTileSize() const override { return 0x00080000; }

  uint32_t getNumDestSwitchboxConnections(int col, int row,
                                          WireBundle bundle) const override;
  uint32_t getNumSourceSwitchboxConnections(int col, int row,
                                            WireBundle bundle) const override;
  uint32_t getNumDestShimMuxConnections(int col, int row,
                                        WireBundle bundle) const override;
  uint32_t getNumSourceShimMuxConnections(int col, int row,
                                          WireBundle bundle) const override;
  bool isLegalMemtileConnection(WireBundle srcBundle, int srcChan,
                                WireBundle dstBundle,
                                int dstChan) const override;
};

class VC1902TargetModel : public AIE1TargetModel {
  llvm::SmallDenseSet<unsigned, 16> nocColumns = {
      2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47};

public:
  VC1902TargetModel() = default;

  int columns() const override { return 50; }

  int rows() const override { return 9; /* One Shim row and 8 Core rows. */ }

  bool isShimNOCTile(int col, int row) const override {
    return row == 0 && nocColumns.contains(col);
  }

  bool isShimPLTile(int col, int row) const override {
    return row == 0 && !nocColumns.contains(col);
  }

  bool isShimNOCorPLTile(int col, int row) const override {
    return isShimNOCTile(col, row) || isShimPLTile(col, row);
  }
};

class VE2302TargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 8> nocColumns = {2, 3, 6, 7, 10, 11};

public:
  VE2302TargetModel() = default;

  int columns() const override { return 17; }

  int rows() const override {
    return 4; /* One Shim row, 1 memtile rows, and 2 Core rows. */
  }

  bool isCoreTile(int col, int row) const override { return row > 1; }
  bool isMemTile(int col, int row) const override { return row == 1; }

  bool isShimNOCTile(int col, int row) const override {
    return row == 0 && nocColumns.contains(col);
  }

  bool isShimPLTile(int col, int row) const override {
    return row == 0 && !nocColumns.contains(col);
  }

  bool isShimNOCorPLTile(int col, int row) const override {
    return isShimNOCTile(col, row) || isShimPLTile(col, row);
  }

  uint32_t getNumMemTileRows() const override { return 1; }

  bool isValidTraceMaster(int col, int row, WireBundle destBundle,
                          int destIndex) const override {
    if (isCoreTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isCoreTile(col, row) && destBundle == WireBundle::DMA && destIndex == 0)
      return true;
    if (isMemTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isMemTile(col, row) && destBundle == WireBundle::DMA && destIndex == 5)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::West &&
        destIndex == 0)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::East &&
        destIndex == 0)
      return true;
    return false;
  }
};

class VE2802TargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 16> nocColumns = {2,  3,  6,  7,  14, 15,
                                                  22, 23, 30, 31, 34, 35};

public:
  VE2802TargetModel() = default;

  int columns() const override { return 38; }

  int rows() const override {
    return 11; /* One Shim row, 2 memtile rows, and 8 Core rows. */
  }

  bool isCoreTile(int col, int row) const override { return row > 2; }

  bool isMemTile(int col, int row) const override {
    return row == 1 || row == 2;
  }

  bool isShimNOCTile(int col, int row) const override {
    return row == 0 && nocColumns.contains(col);
  }

  bool isShimPLTile(int col, int row) const override {
    return row == 0 && !nocColumns.contains(col);
  }

  bool isShimNOCorPLTile(int col, int row) const override {
    return isShimNOCTile(col, row) || isShimPLTile(col, row);
  }

  uint32_t getNumMemTileRows() const override { return 2; }

  bool isValidTraceMaster(int col, int row, WireBundle destBundle,
                          int destIndex) const override {
    if (isCoreTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isCoreTile(col, row) && destBundle == WireBundle::DMA && destIndex == 0)
      return true;
    if (isMemTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isMemTile(col, row) && destBundle == WireBundle::DMA && destIndex == 5)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::West &&
        destIndex == 0)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::East &&
        destIndex == 0)
      return true;
    return false;
  }
};

class NPUTargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 16> nocColumns = {0, 1, 2, 3};

public:
  NPUTargetModel() = default;

  int columns() const override { return 5; }

  int rows() const override {
    return 6; /* 1 Shim row, 1 memtile row, and 4 Core rows. */
  }

  bool isCoreTile(int col, int row) const override { return row > 1; }
  bool isMemTile(int col, int row) const override { return row == 1; }

  bool isShimNOCTile(int col, int row) const override {
    return row == 0 && nocColumns.contains(col);
  }

  bool isShimPLTile(int col, int row) const override {
    return row == 0 && !nocColumns.contains(col);
  }

  bool isShimNOCorPLTile(int col, int row) const override {
    return isShimNOCTile(col, row) || isShimPLTile(col, row);
  }

  uint32_t getNumMemTileRows() const override { return 1; }

  bool isValidTraceMaster(int col, int row, WireBundle destBundle,
                          int destIndex) const override {
    if (isCoreTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isCoreTile(col, row) && destBundle == WireBundle::DMA && destIndex == 0)
      return true;
    if (isMemTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isMemTile(col, row) && destBundle == WireBundle::DMA && destIndex == 5)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::South)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::West &&
        destIndex == 0)
      return true;
    if (isShimNOCorPLTile(col, row) && destBundle == WireBundle::East &&
        destIndex == 0)
      return true;
    return false;
  }
};

} // namespace xilinx::AIE

namespace llvm {
template <>
struct DenseMapInfo<xilinx::AIE::TileID> {
  using FirstInfo = DenseMapInfo<int>;
  using SecondInfo = DenseMapInfo<int>;

  static xilinx::AIE::TileID getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static xilinx::AIE::TileID getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const xilinx::AIE::TileID &t) {
    return detail::combineHashValue(FirstInfo::getHashValue(t.col),
                                    SecondInfo::getHashValue(t.row));
  }

  static bool isEqual(const xilinx::AIE::TileID &lhs,
                      const xilinx::AIE::TileID &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

template <>
struct std::hash<xilinx::AIE::TileID> {
  std::size_t operator()(const xilinx::AIE::TileID &s) const noexcept {
    std::size_t h1 = std::hash<int>{}(s.col);
    std::size_t h2 = std::hash<int>{}(s.row);
    return h1 ^ (h2 << 1);
  }
};

#endif
