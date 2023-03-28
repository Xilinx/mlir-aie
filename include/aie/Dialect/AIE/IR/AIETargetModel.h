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
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"

namespace xilinx {
namespace AIE {

typedef std::pair<int, int> TileID;

enum AIEArch {
  AIE1 = 1,
  AIE2 = 2,
};

class AIETargetModel {
public:
  AIETargetModel() {}
  virtual ~AIETargetModel();

  /// Return the target architecture.
  virtual AIEArch getTargetArch() const = 0;

  /// Return the number of columns in the device.
  virtual int columns() const = 0;

  /// Return the number of rows in the device.
  virtual int rows() const = 0;

  /// Return true if the given tile is a Shim NOC tile.  These tiles include a
  /// ShimDMA and a connection to the memory-mapped NOC.
  virtual bool isShimNOCTile(int col, int row) const = 0;

  /// Return true if the given tile is a Shim PL interface tile.  These tiles do
  /// not include a ShimDMA and instead include connections to the PL.
  virtual bool isShimPLTile(int col, int row) const = 0;

  /// Return true if the given tile ID is valid.
  virtual bool isValidTile(TileID src) const {
    return (src.first >= 0) && (src.first < columns()) && (src.second >= 0) &&
           (src.second < rows());
  }

  /// Return the tile ID of the memory to the west of the given tile, if it
  /// exists.
  virtual llvm::Optional<TileID> getMemWest(TileID src) const = 0;
  /// Return the tile ID of the memory to the east of the given tile, if it
  /// exists.
  virtual llvm::Optional<TileID> getMemEast(TileID src) const = 0;
  /// Return the tile ID of the memory to the north of the given tile, if it
  /// exists.
  virtual llvm::Optional<TileID> getMemNorth(TileID src) const = 0;
  /// Return the tile ID of the memory to the south of the given tile, if it
  /// exists.
  virtual llvm::Optional<TileID> getMemSouth(TileID src) const = 0;

  /// Return true if src is the internal memory of dst
  virtual bool isInternal(int srcCol, int srcRow, int dstCol,
                          int dstRow) const = 0;
  /// Return true if src is West of dst
  virtual bool isWest(int srcCol, int srcRow, int dstCol, int dstRow) const = 0;
  /// Return true if src has a memory tile which is West of dst
  virtual bool isMemWest(int srcCol, int srcRow, int dstCol,
                         int dstRow) const = 0;
  /// Return true if src is East of dst
  virtual bool isEast(int srcCol, int srcRow, int dstCol, int dstRow) const = 0;
  /// Return true if src has a memory tile which is East of dst
  virtual bool isMemEast(int srcCol, int srcRow, int dstCol,
                         int dstRow) const = 0;
  /// Return true if src is North of dst
  virtual bool isNorth(int srcCol, int srcRow, int dstCol,
                       int dstRow) const = 0;
  /// Return true if src has a memory tile which is North of dst
  virtual bool isMemNorth(int srcCol, int srcRow, int dstCol,
                          int dstRow) const = 0;
  /// Return true if src is South of dst
  virtual bool isSouth(int srcCol, int srcRow, int dstCol,
                       int dstRow) const = 0;
  /// Return true if src has a memory tile which is South of dst
  virtual bool isMemSouth(int srcCol, int srcRow, int dstCol,
                          int dstRow) const = 0;

  /// Return true if core can access the memory in mem
  virtual bool isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                                  int memRow) const = 0;

  /// Return the base address in the local address map of differnet memories.
  virtual uint32_t getMemInternalBaseAddress(TileID src) const = 0;
  virtual uint32_t getMemSouthBaseAddress() const = 0;
  virtual uint32_t getMemWestBaseAddress() const = 0;
  virtual uint32_t getMemNorthBaseAddress() const = 0;
  virtual uint32_t getMemEastBaseAddress() const = 0;

  /// Return the size (in bytes) of the local data memory of a core.
  virtual uint32_t getLocalMemorySize() const = 0;
};

class AIE1TargetModel : public AIETargetModel {
public:
  AIE1TargetModel() {}

  AIEArch getTargetArch() const override;

  llvm::Optional<TileID> getMemWest(TileID src) const override;
  llvm::Optional<TileID> getMemEast(TileID src) const override;
  llvm::Optional<TileID> getMemNorth(TileID src) const override;
  llvm::Optional<TileID> getMemSouth(TileID src) const override;

  bool isInternal(int srcCol, int srcRow, int dstCol,
                  int dstRow) const override;
  bool isWest(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemWest(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isEast(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemEast(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isNorth(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemNorth(int srcCol, int srcRow, int dstCol,
                  int dstRow) const override;
  bool isSouth(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemSouth(int srcCol, int srcRow, int dstCol,
                  int dstRow) const override;

  bool isLegalMemAffinity(int coreCol, int coreRow, int memCol,
                          int memRow) const override;

  uint32_t getMemInternalBaseAddress(TileID src) const override {
    bool IsEvenRow = ((src.second % 2) == 0);
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
};

class AIE2TargetModel : public AIETargetModel {
public:
  AIE2TargetModel() {}

  AIEArch getTargetArch() const override;

  llvm::Optional<TileID> getMemWest(TileID src) const override;
  llvm::Optional<TileID> getMemEast(TileID src) const override;
  llvm::Optional<TileID> getMemNorth(TileID src) const override;
  llvm::Optional<TileID> getMemSouth(TileID src) const override;

  bool isInternal(int srcCol, int srcRow, int dstCol,
                  int dstRow) const override;
  bool isWest(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemWest(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isEast(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemEast(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isNorth(int srcCol, int srcRow, int dstCol, int dstRow) const override;
  bool isMemNorth(int srcCol, int srcRow, int dstCol,
                  int dstRow) const override;
  bool isSouth(int srcCol, int srcRow, int dstCol, int dstRow) const override;
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
};

class VC1902TargetModel : public AIE1TargetModel {
  llvm::SmallDenseSet<unsigned, 16> noc_columns = {
      2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47};

public:
  VC1902TargetModel() {}

  int columns() const override { return 50; }
  int rows() const override { return 9; /* One Shim row and 8 Core rows. */ }

  bool isShimNOCTile(int col, int row) const override {
    return row == 0 && noc_columns.contains(col);
  }
  bool isShimPLTile(int col, int row) const override {
    return row == 0 && !noc_columns.contains(col);
  }
};

class VE2302TargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 16> noc_columns = {}; // FIXME
public:
  VE2302TargetModel() {}

  int columns() const override { return 17; }
  int rows() const override {
    return 4; /* One Shim row, 1 memtile rows, and 2 Core rows. */
  }

  bool isShimNOCTile(int col, int row) const override {
    return row == 0 && noc_columns.contains(col);
  }
  bool isShimPLTile(int col, int row) const override {
    return row == 0 && !noc_columns.contains(col);
  }
};

class VE2802TargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 16> noc_columns = {}; // FIXME
public:
  VE2802TargetModel() {}

  int columns() const override { return 38; }
  int rows() const override {
    return 11; /* One Shim row, 2 memtile rows, and 8 Core rows. */
  }

  bool isShimNOCTile(int col, int row) const override {
    return row == 0 && noc_columns.contains(col);
  }
  bool isShimPLTile(int col, int row) const override {
    return row == 0 && !noc_columns.contains(col);
  }
};

} // namespace AIE
} // namespace xilinx

#endif