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

  /// Return the data bus width of the device.
  virtual uint32_t getAddressGenGranularity() const = 0;

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

  /// Return the size (in bits) of the accumulator/cascade.
  virtual uint32_t getAccumulatorCascadeSize() const = 0;

  /// Return the number of lock objects
  virtual uint32_t getNumLocks(int col, int row) const = 0;

  /// Return the number of buffer descriptors supported by the DMA in the given
  /// tile.
  virtual uint32_t getNumBDs(int col, int row) const = 0;

  /// Return true iff buffer descriptor `bd_id` on tile (`col`, `row`) can be
  /// submitted on channel `channel`.
  virtual bool isBdChannelAccessible(int col, int row, uint32_t bd_id,
                                     int channel) const = 0;

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
  virtual bool isLegalTileConnection(int col, int row, WireBundle srcBundle,
                                     int srcChan, WireBundle dstBundle,
                                     int dstChan) const = 0;

  // Run consistency checks on the target model.
  void validate() const;

  // Return true if this is an NPU-based device
  // There are several special cases for handling the NPU at the moment.
  virtual bool isNPU() const { return false; }

  // Return the bit offset of the column within a tile address.
  // This is used to compute the control address of a tile from it's column
  // location.
  virtual uint32_t getColumnShift() const = 0;

  // Return the bit offset of the row within a tile address.
  // This is used to compute the control address of a tile from it's row
  // location.
  virtual uint32_t getRowShift() const = 0;
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
  uint32_t getAccumulatorCascadeSize() const override { return 384; }
  uint32_t getNumLocks(int col, int row) const override { return 16; }
  uint32_t getNumBDs(int col, int row) const override { return 16; }
  bool isBdChannelAccessible(int col, int row, uint32_t bd_id,
                             int channel) const override {
    return true;
  }
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
  bool isLegalTileConnection(int col, int row, WireBundle srcBundle,
                             int srcChan, WireBundle dstBundle,
                             int dstChan) const override;

  uint32_t getColumnShift() const override { return 23; }
  uint32_t getRowShift() const override { return 18; }
};

class AIE2TargetModel : public AIETargetModel {
public:
  AIE2TargetModel() = default;

  AIEArch getTargetArch() const override;

  uint32_t getAddressGenGranularity() const override { return 32; }

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
  uint32_t getAccumulatorCascadeSize() const override { return 512; }

  uint32_t getNumLocks(int col, int row) const override {
    return isMemTile(col, row) ? 64 : 16;
  }

  uint32_t getNumBDs(int col, int row) const override {
    return isMemTile(col, row) ? 48 : 16;
  }

  bool isBdChannelAccessible(int col, int row, uint32_t bd_id,
                             int channel) const override {
    if (!isMemTile(col, row)) {
      return true;
    } else {
      if ((channel & 1) == 0) { // even channel number
        return bd_id < 24;
      } else {
        return bd_id >= 24;
      }
    }
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
  bool isLegalTileConnection(int col, int row, WireBundle srcBundle,
                             int srcChan, WireBundle dstBundle,
                             int dstChan) const override;

  uint32_t getColumnShift() const override { return 25; }
  uint32_t getRowShift() const override { return 20; }
};

class VC1902TargetModel : public AIE1TargetModel {
  llvm::SmallDenseSet<unsigned, 16> nocColumns = {
      2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47};

public:
  VC1902TargetModel() = default;

  uint32_t getAddressGenGranularity() const override { return 32; }

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
};

class BaseNPUTargetModel : public AIE2TargetModel {
public:
  BaseNPUTargetModel() = default;

  int rows() const override {
    return 6; /* 1 Shim row, 1 memtile row, and 4 Core rows. */
  }

  bool isCoreTile(int col, int row) const override { return row > 1; }
  bool isMemTile(int col, int row) const override { return row == 1; }

  bool isShimPLTile(int col, int row) const override {
    return false; // No PL
  }

  bool isShimNOCorPLTile(int col, int row) const override {
    return isShimNOCTile(col, row) || isShimPLTile(col, row);
  }

  uint32_t getNumMemTileRows() const override { return 1; }

  // Return true if the device model is virtualized.  This is used
  // during CDO code generation to configure aie-rt properly.
  virtual bool isVirtualized() const = 0;

  virtual bool isNPU() const override { return true; }
};

// The full Phoenix NPU
class NPUTargetModel : public BaseNPUTargetModel {
public:
  NPUTargetModel() = default;

  int columns() const override { return 5; }

  bool isShimNOCTile(int col, int row) const override {
    return row == 0 && col > 0;
  }

  bool isShimPLTile(int col, int row) const override {
    // This isn't useful because it's not connected to anything.
    return row == 0 && col == 0;
  }

  bool isVirtualized() const override { return false; }
};

// A sub-portion of the NPU
class VirtualizedNPUTargetModel : public BaseNPUTargetModel {
  int cols;

public:
  VirtualizedNPUTargetModel(int _cols) : cols(_cols) {}

  uint32_t getAddressGenGranularity() const override { return 32; }

  int columns() const override { return cols; }

  bool isShimNOCTile(int col, int row) const override { return row == 0; }

  bool isVirtualized() const override { return true; }
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
