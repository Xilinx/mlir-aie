//===- Device.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Array geometry and address decode.
//
// The simulator is driven entirely by the flat addresses that arrive at
// ess_Write32 / ess_Read32, so turning an address back into (column, row,
// register offset) is the first thing it has to get right. aie-rt builds those
// addresses as
//
//     (Col << ColShift) | (Row << RowShift) | RegOff
//
// (see _XAie_GetTileAddr and the inverse helpers in
// third_party/aie-rt/driver/src/common/xaie_helper.c:920-931), with the shifts
// coming from the XAie_Config the host program passes to XAie_CfgInitialize.
// The values here must match those configs; DeviceModel::checkAgainstConfig
// exists so a mismatch is a loud failure at simulation start rather than a
// silent decode into the wrong tile.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_DEVICE_H
#define AIESIM_DEVICE_H

#include <cstdint>
#include <string>

namespace aiesim {

/// The kind of tile at a given row. Rows are numbered from the shim row up,
/// matching aie-rt and the AIE dialect.
enum class TileType {
  Shim,    ///< Row 0: PL/NoC interface, shim DMA, no core.
  MemTile, ///< Rows 1..N: large data memory + DMA, no core.
  Core,    ///< Remaining rows: core, data memory, DMA, locks.
  Invalid,
};

/// Which AIE generation the array implements. This selects both the register
/// map and the core ISA.
enum class Generation {
  AIE2,  ///< npu1 / Phoenix
  AIE2P, ///< npu2 / Strix
};

/// Static description of one array: geometry, address shifts, memory sizes and
/// resource counts. All sizes are in bytes.
struct DeviceModel {
  Generation generation;

  /// Number of columns actually present in the simulated partition.
  uint32_t numCols;
  /// Total rows including the shim row.
  uint32_t numRows;
  /// Rows 1..numMemTileRows are memtiles.
  uint32_t numMemTileRows;

  /// Address shifts, as configured in XAie_Config.
  uint32_t colShift;
  uint32_t rowShift;
  /// Base address the host adds to every tile address.
  uint64_t baseAddr;

  /// Per-tile memory sizes.
  uint32_t coreDataMemSize; ///< Core tile local data memory.
  uint32_t coreProgMemSize; ///< Core tile program memory.
  uint32_t memTileMemSize;  ///< Memtile data memory.

  /// Offset of program memory inside a core tile's register window. aie-rt
  /// calls this ProgMemHostOffset and uses it when loading ELF sections.
  uint32_t progMemHostOffset;

  /// Resource counts, per tile of the relevant type.
  uint32_t numCoreLocks;
  uint32_t numMemTileLocks;
  uint32_t numShimLocks;
  uint32_t numCoreDmaChannelsS2MM;
  uint32_t numCoreDmaChannelsMM2S;
  uint32_t numMemTileDmaChannelsS2MM;
  uint32_t numMemTileDmaChannelsMM2S;
  uint32_t numShimDmaChannelsS2MM;
  uint32_t numShimDmaChannelsMM2S;
  uint32_t numCoreBds;
  uint32_t numMemTileBds;
  uint32_t numShimBds;

  TileType tileTypeAt(uint32_t row) const {
    if (row == 0)
      return TileType::Shim;
    if (row <= numMemTileRows)
      return TileType::MemTile;
    if (row < numRows)
      return TileType::Core;
    return TileType::Invalid;
  }

  /// True if (col, row) is inside the simulated partition.
  bool contains(uint32_t col, uint32_t row) const {
    return col < numCols && row < numRows;
  }
};

/// Where a decoded flat address points.
struct DecodedAddress {
  uint32_t col;
  uint32_t row;
  /// Byte offset inside the tile's register window.
  uint32_t regOff;
  /// False if the address fell outside the partition, in which case the
  /// simulator reports an error instead of guessing.
  bool valid;
};

/// Split a flat address into (col, row, regOff). This is the exact inverse of
/// aie-rt's address construction, so it must stay in sync with colShift /
/// rowShift.
DecodedAddress decodeAddress(const DeviceModel &dev, uint64_t addr);

/// The two supported devices. `numCols` is the partition width, which the
/// caller takes from the design (npu1_1col, npu2_4col and so on).
DeviceModel makeAIE2Device(uint32_t numCols);
DeviceModel makeAIE2PDevice(uint32_t numCols);

/// Parse a device name as spelled in the AIE dialect ("npu1", "npu1_1col",
/// "npu2", "npu2_4col", ...). Returns false and fills `error` if the name is
/// not one of the supported devices.
bool makeDeviceFromName(const std::string &name, DeviceModel &out,
                        std::string &error);

} // namespace aiesim

#endif // AIESIM_DEVICE_H
