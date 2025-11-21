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
#include "aie/Dialect/AIE/IR/AIERegisterDatabase.h"

#include "llvm/ADT/DenseSet.h"

#include <iostream>
#include <memory>

namespace xilinx::AIE {

// Forward declarations
class TileOp;

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
  enum TargetModelKind {
    TK_AIE1_VC1902,
    TK_AIE1_Last,
    TK_AIE2_VE2302,
    TK_AIE2_VE2802,
    TK_AIE2_NPU1_1Col,
    TK_AIE2_NPU1_2Col,
    TK_AIE2_NPU1_3Col,
    TK_AIE2_NPU1_4Col, // whole array must be last because of how we
                       // cast/initialize the VirtualizedNPU1TargetModel class
    TK_AIE2_NPU1_Last,
    TK_AIE2_NPU2 = TK_AIE2_NPU1_Last,
    TK_AIE2_NPU2_1Col,
    TK_AIE2_NPU2_2Col,
    TK_AIE2_NPU2_3Col,
    TK_AIE2_NPU2_4Col,
    TK_AIE2_NPU2_5Col,
    TK_AIE2_NPU2_6Col,
    TK_AIE2_NPU2_7Col,
    TK_AIE2_NPU2_Last,
    TK_AIE2_Last = TK_AIE2_NPU2_Last,
  };

  // One-hot encoded list of target model properties.
  enum ModelProperty {
    // Device uses semaphore locks.
    UsesSemaphoreLocks = 1U << 0,
    // Device is an NPU-based device.
    // There are several special cases for handling the NPU at the moment.
    IsNPU = 1U << 1,
    // Device model is virtualized.
    // This is used during CDO code generation to configure aie-rt properly.
    IsVirtualized = 1U << 2,
    // Device uses multi-dimensional buffer descriptors.
    UsesMultiDimensionalBDs = 1U << 3,
  };

private:
  const TargetModelKind kind;

  uint32_t ModelProperties = 0;

  // Register database (loaded lazily on first access)
  mutable std::unique_ptr<RegisterDatabase> regDB;
  mutable bool regDBLoadAttempted = false;

protected:
  /// Subclasses override to provide architecture-specific database loading.
  /// Returns nullptr if register database is not available for this
  /// architecture.
  virtual std::unique_ptr<RegisterDatabase> loadRegisterDatabase() const;

  /// Get the register database, loading it lazily on first access.
  /// Throws fatal error if database is required but unavailable.
  const RegisterDatabase *getRegisterDatabase() const;

public:
  TargetModelKind getKind() const { return kind; }

  AIETargetModel(TargetModelKind k) : kind(k) {}

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

  /// Return the base address in the local address map for a core.
  virtual uint32_t getMemInternalBaseAddress(TileID src) const = 0;
  /// Return the base address in the local address map for a core.
  virtual uint32_t getMemSouthBaseAddress() const = 0;
  /// Return the base address in the local address map for a core.
  virtual uint32_t getMemWestBaseAddress() const = 0;
  /// Return the base address in the local address map for a core.
  virtual uint32_t getMemNorthBaseAddress() const = 0;
  /// Return the base address in the local address map for a core.
  virtual uint32_t getMemEastBaseAddress() const = 0;

  /// Return the lock base index (or offset) in the local tile when accessing a
  /// neighbor's lock or an empty optional if an invalid neighbor is given
  /// Takes into account differences between Memory and Core tiles
  std::optional<uint32_t> getLockLocalBaseIndex(int localCol, int localRow,
                                                int lockCol, int lockRow) const;

  /// Return the memory base address (or offset) in the local tile when
  /// accessing a neighbor's memory or an empty optional if an invalid neighbor
  /// is given
  /// Takes into account differences between Memory and Core tiles
  std::optional<uint32_t> getMemLocalBaseAddress(int localCol, int localRow,
                                                 int memCol, int memRow) const;

  /// Return the size (in bytes) of the local data memory of a core.
  virtual uint32_t getLocalMemorySize() const = 0;

  /// Return the size (in bits) of the accumulator/cascade.
  virtual uint32_t getAccumulatorCascadeSize() const = 0;

  /// Return the number of lock objects
  virtual uint32_t getNumLocks(int col, int row) const = 0;

  /// Return the maximum value that can be stored in a lock register
  virtual uint32_t getMaxLockValue() const = 0;

  // Return the lock address for the lock ID in the local memory for a given
  // tile or a nullopt if invalid arguments are given.
  virtual std::optional<uint32_t> getLocalLockAddress(uint32_t lockId,
                                                      TileID tile) const = 0;

  /// Get stream switch port index for a given port specification
  /// @param col Tile column
  /// @param row Tile row
  /// @param bundle Port type (WireBundle enum: DMA, FIFO, North, South, East,
  /// West, Core, etc.)
  /// @param channel Channel/port number within the bundle
  /// @param master True for master port, false for slave port
  /// @return Port index for Stream_Switch_Event_Port_Selection register, or
  /// nullopt if invalid
  virtual std::optional<uint32_t>
  getStreamSwitchPortIndex(int col, int row, WireBundle bundle,
                           uint32_t channel, bool master) const = 0;

  /// Check if a stream switch port is valid for the given tile
  /// @param col Tile column
  /// @param row Tile row
  /// @param bundle Port type
  /// @param channel Channel/port number
  /// @param master Master/slave direction
  /// @return True if the port configuration is valid
  virtual bool isValidStreamSwitchPort(int col, int row, WireBundle bundle,
                                       uint32_t channel, bool master) const = 0;

  /// Return the number of buffer descriptors supported by the DMA in the given
  /// tile.
  virtual uint32_t getNumBDs(int col, int row) const = 0;

  /// Return true iff buffer descriptor `bd_id` on tile (`col`, `row`) can be
  /// submitted on channel `channel`.
  virtual bool isBdChannelAccessible(int col, int row, uint32_t bd_id,
                                     int channel) const = 0;

  /// Return the array address of the dma buffer descriptor for the given
  /// col, row, buffer descriptor id, channel and direction. Not all
  /// architecture variants will use channel and direction so these have default
  /// values.
  virtual uint64_t getDmaBdAddress(
      int col, int row, uint32_t bd_id, int channel = -1,
      AIE::DMAChannelDir direction = AIE::DMAChannelDir::MM2S) const = 0;

  /// Return the offset of the base address field within the shim dma buffer
  /// descriptor.
  virtual uint32_t getDmaBdAddressOffset(int col, int row) const = 0;

  /// Return the array address of the dma task queue register for the given
  /// col, row, channel and direction
  virtual uint32_t getDmaControlAddress(int col, int row, int channel,
                                        AIE::DMAChannelDir direction) const = 0;

  virtual uint32_t getNumMemTileRows() const = 0;
  /// Return the size (in bytes) of a MemTile.
  virtual uint32_t getMemTileSize() const = 0;
  /// Return the number of memory banks of a given tile.
  virtual uint32_t getNumBanks(int col, int row) const = 0;

  virtual uint32_t getMaxChannelNumForAdjacentMemTile(int col,
                                                      int row) const = 0;

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

  uint32_t getModelProperties() const { return ModelProperties; }
  void addModelProperty(uint32_t prop) { ModelProperties |= prop; }
  // Return true if this device has a given property.
  bool hasProperty(ModelProperty Prop) const {
    return (getModelProperties() & Prop) == Prop;
  }

  // Return the bit offset of the column within a tile address.
  // This is used to compute the control address of a tile from it's column
  // location.
  virtual uint32_t getColumnShift() const = 0;

  // Return the bit offset of the row within a tile address.
  // This is used to compute the control address of a tile from it's row
  // location.
  virtual uint32_t getRowShift() const = 0;

  // Returns the list of possible burst encodings (first) and
  // their corresponding lengths in bytes (second).
  virtual std::vector<std::pair<uint32_t, uint32_t>>
  getShimBurstEncodingsAndLengths() const = 0;

  // Returns true if the target model supports the given block format.
  virtual bool isSupportedBlockFormat(std::string const &format) const;

  /// Register Database API - provides access to register and event information
  /// for trace configuration and low-level register access.

  /// Lookup register information by name and tile.
  /// @param name Register name (e.g., "Trace_Control0")
  /// @param tile Tile operation to determine module context
  /// @param isMem True for memory module registers, false for core/shim
  /// @return Pointer to register info, or nullptr if not found
  /// @throws fatal_error if database required but unavailable
  const RegisterInfo *lookupRegister(llvm::StringRef name, TileOp tile,
                                     bool isMem = false) const;

  /// Lookup event number by name and tile.
  /// @param name Event name (e.g., "INSTR_EVENT_0", "DMA_S2MM_0_START_TASK")
  /// @param tile Tile operation to determine module context
  /// @param isMem True for memory module events, false for core/shim
  /// @return Event number if found, nullopt otherwise
  /// @throws fatal_error if database required but unavailable
  std::optional<uint32_t> lookupEvent(llvm::StringRef name, TileOp tile,
                                      bool isMem = false) const;

  /// Encode a field value with proper bit shifting.
  /// @param field Bit field information
  /// @param value Value to encode
  /// @return Value shifted to correct bit position
  /// @throws fatal_error if database required but unavailable
  uint32_t encodeFieldValue(const BitFieldInfo &field, uint32_t value) const;

  /// Resolve stream switch port specification to port index.
  /// @param value Port specification string (e.g., "NORTH:1", "DMA:0")
  /// @param tile Tile operation for context
  /// @param master True for master port, false for slave
  /// @return Port index for stream switch register, or nullopt if invalid
  /// @throws fatal_error if database required but unavailable
  std::optional<uint32_t> resolvePortValue(llvm::StringRef value, TileOp tile,
                                           bool master) const;
};

class AIE1TargetModel : public AIETargetModel {
public:
  AIE1TargetModel(TargetModelKind k) : AIETargetModel(k) {}

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
  uint32_t getMaxLockValue() const override { return 1; }
  std::optional<uint32_t> getLocalLockAddress(uint32_t lockId,
                                              TileID tile) const override;
  uint32_t getNumBDs(int col, int row) const override { return 16; }
  bool isBdChannelAccessible(int col, int row, uint32_t bd_id,
                             int channel) const override {
    return true;
  }

  uint64_t getDmaBdAddress(int col, int row, uint32_t bd_id, int channel,
                           AIE::DMAChannelDir direction) const override;

  uint32_t getDmaBdAddressOffset(int col, int row) const override;

  uint32_t getDmaControlAddress(int col, int row, int channel,
                                AIE::DMAChannelDir direction) const override;

  uint32_t getNumMemTileRows() const override { return 0; }
  uint32_t getMemTileSize() const override { return 0; }
  uint32_t getNumBanks(int col, int row) const override { return 4; }

  uint32_t getMaxChannelNumForAdjacentMemTile(int col, int row) const override {
    return 0;
  }

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

  std::optional<uint32_t> getStreamSwitchPortIndex(int col, int row,
                                                   WireBundle bundle,
                                                   uint32_t channel,
                                                   bool master) const override;
  bool isValidStreamSwitchPort(int col, int row, WireBundle bundle,
                               uint32_t channel, bool master) const override;

  uint32_t getColumnShift() const override { return 23; }
  uint32_t getRowShift() const override { return 18; }

  static bool classof(const AIETargetModel *model) {
    return model->getKind() >= TK_AIE1_VC1902 &&
           model->getKind() < TK_AIE1_Last;
  }

  std::vector<std::pair<uint32_t, uint32_t>>
  getShimBurstEncodingsAndLengths() const override;
};

class AIE2TargetModel : public AIETargetModel {
protected:
  std::unique_ptr<RegisterDatabase> loadRegisterDatabase() const override;

public:
  AIE2TargetModel(TargetModelKind k) : AIETargetModel(k) {
    // Device properties initialization
    addModelProperty(AIETargetModel::UsesSemaphoreLocks);
    addModelProperty(AIETargetModel::UsesMultiDimensionalBDs);
  }

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

  uint32_t getMaxLockValue() const override { return 0x3F; }

  std::optional<uint32_t> getLocalLockAddress(uint32_t lockId,
                                              TileID tile) const override;

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

  uint64_t getDmaBdAddress(int col, int row, uint32_t bd_id, int channel,
                           AIE::DMAChannelDir direction) const override;

  uint32_t getDmaBdAddressOffset(int col, int row) const override;

  uint32_t getDmaControlAddress(int col, int row, int channel,
                                AIE::DMAChannelDir direction) const override;

  uint32_t getMemTileSize() const override { return 0x00080000; }

  uint32_t getNumBanks(int col, int row) const override {
    return isMemTile(col, row) ? 8 : 4;
  }

  uint32_t getMaxChannelNumForAdjacentMemTile(int col, int row) const override {
    return 4;
  }

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

  std::optional<uint32_t> getStreamSwitchPortIndex(int col, int row,
                                                   WireBundle bundle,
                                                   uint32_t channel,
                                                   bool master) const override;
  bool isValidStreamSwitchPort(int col, int row, WireBundle bundle,
                               uint32_t channel, bool master) const override;

  uint32_t getColumnShift() const override { return 25; }
  uint32_t getRowShift() const override { return 20; }

  static bool classof(const AIETargetModel *model) {
    return model->getKind() >= TK_AIE2_VE2302 &&
           model->getKind() < TK_AIE2_Last;
  }

  std::vector<std::pair<uint32_t, uint32_t>>
  getShimBurstEncodingsAndLengths() const override;
};

class VC1902TargetModel : public AIE1TargetModel {
  llvm::SmallDenseSet<unsigned, 16> nocColumns = {
      2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47};

public:
  VC1902TargetModel() : AIE1TargetModel(TK_AIE1_VC1902) {}

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

  static bool classof(const AIETargetModel *model) {
    return model->getKind() == TK_AIE1_VC1902;
  }
};

class VE2302TargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 8> nocColumns = {2, 3, 6, 7, 10, 11};

public:
  VE2302TargetModel() : AIE2TargetModel(TK_AIE2_VE2302) {}

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

  static bool classof(const AIETargetModel *model) {
    return model->getKind() == TK_AIE2_VE2302;
  }
};

class VE2802TargetModel : public AIE2TargetModel {
  llvm::SmallDenseSet<unsigned, 16> nocColumns = {2,  3,  6,  7,  14, 15,
                                                  22, 23, 30, 31, 34, 35};

public:
  VE2802TargetModel() : AIE2TargetModel(TK_AIE2_VE2802) {}

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

  static bool classof(const AIETargetModel *model) {
    return model->getKind() == TK_AIE2_VE2802;
  }
};

class BaseNPU1TargetModel : public AIE2TargetModel {
public:
  BaseNPU1TargetModel(TargetModelKind k) : AIE2TargetModel(k) {
    // Device properties initialization
    addModelProperty(AIETargetModel::IsNPU);
  }

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

  static bool classof(const AIETargetModel *model) {
    return model->getKind() >= TK_AIE2_NPU1_1Col &&
           model->getKind() < TK_AIE2_NPU1_Last;
  }
};

// A sub-portion of the Phoenix NPU
class VirtualizedNPU1TargetModel : public BaseNPU1TargetModel {
  int cols;

public:
  VirtualizedNPU1TargetModel(int _cols)
      : BaseNPU1TargetModel(static_cast<TargetModelKind>(
            static_cast<std::underlying_type_t<TargetModelKind>>(
                TK_AIE2_NPU1_1Col) +
            _cols - 1)),
        cols(_cols) {
    // Device properties initialization
    addModelProperty(AIETargetModel::IsVirtualized);
  }

  int columns() const override { return cols; }

  bool isShimNOCTile(int col, int row) const override { return row == 0; }

  static bool classof(const AIETargetModel *model) {
    return model->getKind() >= TK_AIE2_NPU1_1Col &&
           model->getKind() < TK_AIE2_NPU1_Last;
  }
};

class BaseNPU2TargetModel : public AIE2TargetModel {
public:
  BaseNPU2TargetModel(TargetModelKind k) : AIE2TargetModel(k) {
    // Device properties initialization
    addModelProperty(AIETargetModel::IsNPU);
  }

  AIEArch getTargetArch() const override;

  int rows() const override {
    return 6; /* 1 Shim row, 1 memtile row, and 4 Core rows. */
  }

  bool isCoreTile(int col, int row) const override { return row > 1; }
  bool isMemTile(int col, int row) const override { return row == 1; }

  bool isShimPLTile(int col, int row) const override {
    return false; // No PL tiles
  }

  bool isShimNOCTile(int col, int row) const override { return row == 0; }

  bool isShimNOCorPLTile(int col, int row) const override {
    return isShimNOCTile(col, row);
  }

  uint32_t getNumMemTileRows() const override { return 1; }

  std::vector<std::pair<uint32_t, uint32_t>>
  getShimBurstEncodingsAndLengths() const override;

  bool isSupportedBlockFormat(std::string const &format) const override;

  static bool classof(const AIETargetModel *model) {
    return model->getKind() >= TK_AIE2_NPU2 &&
           model->getKind() < TK_AIE2_NPU2_Last;
  }
};

// The full Strix NPU
class NPU2TargetModel : public BaseNPU2TargetModel {
public:
  NPU2TargetModel() : BaseNPU2TargetModel(TK_AIE2_NPU2) {}

  int columns() const override { return 8; }

  static bool classof(const AIETargetModel *model) {
    return model->getKind() == TK_AIE2_NPU2;
  }
};

// A sub-portion of the Strix NPU
class VirtualizedNPU2TargetModel : public BaseNPU2TargetModel {
  int cols;

public:
  VirtualizedNPU2TargetModel(int _cols)
      : BaseNPU2TargetModel(static_cast<TargetModelKind>(
            static_cast<std::underlying_type_t<TargetModelKind>>(TK_AIE2_NPU2) +
            _cols)),
        cols(_cols) {
    // Device properties initialization
    addModelProperty(AIETargetModel::IsVirtualized);
  }

  int columns() const override { return cols; }

  static bool classof(const AIETargetModel *model) {
    return model->getKind() >= TK_AIE2_NPU2_1Col &&
           model->getKind() < TK_AIE2_NPU2_Last;
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
