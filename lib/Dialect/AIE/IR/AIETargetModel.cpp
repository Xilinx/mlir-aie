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
#include "aie/Dialect/AIE/Util/AIERegisterDatabase.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <utility>

using namespace llvm;

namespace xilinx {
namespace AIE {

namespace {

std::string getModuleForTile(const AIETargetModel &model, TileID tile,
                             bool isMem) {
  if (model.isShimNOCorPLTile(tile.col, tile.row))
    return "shim";
  if (model.isMemTile(tile.col, tile.row))
    return "memory_tile";
  return isMem ? std::string("memory") : std::string("core");
}

} // namespace

AIETargetModel::~AIETargetModel() = default;

// Base class implementations for register database

std::unique_ptr<RegisterDatabase> AIETargetModel::loadRegisterDatabase() const {
  // Default: no register database available
  return nullptr;
}

const RegisterDatabase *AIETargetModel::getRegisterDatabase() const {
  std::call_once(regDBInitFlag, [this]() { regDB = loadRegisterDatabase(); });
  return regDB.get();
}

const RegisterInfo *AIETargetModel::lookupRegister(llvm::StringRef name,
                                                   TileID tile,
                                                   bool isMem) const {
  const auto *db = getRegisterDatabase();
  if (!db)
    return nullptr;
  return db->lookupRegister(name, getModuleForTile(*this, tile, isMem));
}

std::optional<uint32_t> AIETargetModel::lookupEvent(llvm::StringRef name,
                                                    TileID tile,
                                                    bool isMem) const {
  const auto *db = getRegisterDatabase();
  if (!db)
    return std::nullopt;
  return db->lookupEvent(name, getModuleForTile(*this, tile, isMem));
}

uint32_t AIETargetModel::encodeFieldValue(const BitFieldInfo &field,
                                          uint32_t value) const {
  const auto *db = getRegisterDatabase();
  if (!db)
    return 0;
  return db->encodeFieldValue(field, value);
}

std::optional<uint32_t> AIETargetModel::resolvePortValue(llvm::StringRef value,
                                                         TileID tile,
                                                         bool master) const {
  auto colonPos = value.find(':');
  if (colonPos == StringRef::npos)
    return std::nullopt;

  StringRef portName = value.substr(0, colonPos);
  StringRef channelStr = value.substr(colonPos + 1);

  int channel;
  if (channelStr.getAsInteger(10, channel) || channel < 0)
    return std::nullopt;

  WireBundle bundle;
  if (portName.equals_insensitive("north")) {
    bundle = WireBundle::North;
  } else if (portName.equals_insensitive("south")) {
    bundle = WireBundle::South;
  } else if (portName.equals_insensitive("east")) {
    bundle = WireBundle::East;
  } else if (portName.equals_insensitive("west")) {
    bundle = WireBundle::West;
  } else if (portName.equals_insensitive("dma")) {
    bundle = WireBundle::DMA;
  } else if (portName.equals_insensitive("fifo")) {
    bundle = WireBundle::FIFO;
  } else if (portName.equals_insensitive("core")) {
    bundle = WireBundle::Core;
  } else if (portName.equals_insensitive("ctrl")) {
    bundle = WireBundle::TileControl;
  } else {
    return std::nullopt;
  }

  return getStreamSwitchPortIndex(tile.col, tile.row, bundle,
                                  static_cast<uint32_t>(channel), master);
}

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

uint64_t AIE1TargetModel::getDmaBdAddress(int col, int row, uint32_t bd_id,
                                          int channel,
                                          AIE::DMAChannelDir direction) const {
  uint32_t offset = 0;
  if (isShimNOCTile(col, row)) {
    offset = 0x0001D000 + (bd_id * 0x14);
  } else if (isCoreTile(col, row)) {
    offset = 0x0001D000 + (bd_id * 0x20);
  } else {
    llvm_unreachable(
        "AIE1TargetModel::getDmaBdAddress called for non-DMA tile");
  }
  return ((col & 0xff) << getColumnShift()) | ((row & 0xff) << getRowShift()) |
         offset;
}

uint32_t AIE1TargetModel::getDmaBdAddressOffset(int col, int row) const {
  if (isShimNOCTile(col, row) || isCoreTile(col, row)) {
    return 0;
  }
  llvm_unreachable(
      "AIE1TargetModel::getDmaBdAddressOffset called for non-DMA tile");
}

uint32_t
AIE1TargetModel::getDmaControlAddress(int col, int row, int channel,
                                      AIE::DMAChannelDir direction) const {
  uint32_t offset = 0;
  if (isShimNOCTile(col, row))
    offset = 0x0001D140 + (channel * 0x8);
  else if (isCoreTile(col, row))
    offset = 0x0001DE00 + (channel * 0x8);
  else
    llvm_unreachable(
        "AIE1TargetModel::getDmaControlAddress called for non-DMA tile");

  if (direction == AIE::DMAChannelDir::MM2S)
    offset += 010;

  return ((col & 0xff) << getColumnShift()) | ((row & 0xff) << getRowShift()) |
         offset;
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
    case WireBundle::TileControl:
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
  case WireBundle::TileControl:
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
    case WireBundle::TileControl:
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
  case WireBundle::TileControl:
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
  if (isShimNOCorPLTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 8;
    case WireBundle::South:
      return 6; // Connection to the south port of the stream switch
    default:
      return 0;
    }
  return 0;
}

bool AIE1TargetModel::isLegalTileConnection(int col, int row,
                                            WireBundle srcBundle, int srcChan,
                                            WireBundle dstBundle,
                                            int dstChan) const {
  // Check Channel Id within the range
  if (srcChan >= int(getNumSourceSwitchboxConnections(col, row, srcBundle)))
    return false;
  if (dstChan >= int(getNumDestSwitchboxConnections(col, row, dstBundle)))
    return false;

  // Memtile
  if (isMemTile(col, row)) {
    return false;
  }
  // Shimtile
  else if (isShimNOCorPLTile(col, row)) {
    if (srcBundle == WireBundle::Trace)
      return dstBundle == WireBundle::South;
    else
      return true;
  }
  // Coretile
  else if (isCoreTile(col, row)) {
    if (srcBundle == WireBundle::Trace)
      return dstBundle == WireBundle::South;
    else
      return true;
  }
  return false;
}

std::vector<std::pair<uint32_t, uint32_t>>
AIE1TargetModel::getShimBurstEncodingsAndLengths() const {
  return {std::pair(0, 64), std::pair(1, 128), std::pair(2, 256)};
}

std::optional<uint32_t>
AIE1TargetModel::getLocalLockAddress(uint32_t lockId, TileID tile) const {
  // This function is currently not supported for AIE1.
  // In order to be implemented for this target model, the interface
  // would need to change given the different way locks are written to in AIE1.
  return std::nullopt;
}

namespace {
namespace aie1_port_id {
namespace core {

// Slave port offset/size constants
static constexpr uint32_t S_CORE_OFFSET = 0;
static constexpr uint32_t S_CORE_SIZE = 2;
static constexpr uint32_t S_CTRL_OFFSET = 4;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_DMA_OFFSET = 2;
static constexpr uint32_t S_DMA_SIZE = 2;
static constexpr uint32_t S_EAST_OFFSET = 21;
static constexpr uint32_t S_EAST_SIZE = 4;
static constexpr uint32_t S_FIFO_OFFSET = 5;
static constexpr uint32_t S_FIFO_SIZE = 2;
static constexpr uint32_t S_NORTH_OFFSET = 17;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 7;
static constexpr uint32_t S_SOUTH_SIZE = 6;
static constexpr uint32_t S_TRACE_OFFSET = 25;
static constexpr uint32_t S_TRACE_SIZE = 2;
static constexpr uint32_t S_WEST_OFFSET = 13;
static constexpr uint32_t S_WEST_SIZE = 4;

// Master port offset/size constants
static constexpr uint32_t M_CORE_OFFSET = 0;
static constexpr uint32_t M_CORE_SIZE = 2;
static constexpr uint32_t M_CTRL_OFFSET = 4;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_DMA_OFFSET = 2;
static constexpr uint32_t M_DMA_SIZE = 2;
static constexpr uint32_t M_EAST_OFFSET = 21;
static constexpr uint32_t M_EAST_SIZE = 4;
static constexpr uint32_t M_FIFO_OFFSET = 5;
static constexpr uint32_t M_FIFO_SIZE = 2;
static constexpr uint32_t M_NORTH_OFFSET = 15;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 7;
static constexpr uint32_t M_SOUTH_SIZE = 4;
static constexpr uint32_t M_WEST_OFFSET = 11;
static constexpr uint32_t M_WEST_SIZE = 4;

} // namespace core

namespace shim {

// Slave port offset/size constants
static constexpr uint32_t S_CTRL_OFFSET = 0;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_EAST_OFFSET = 19;
static constexpr uint32_t S_EAST_SIZE = 4;
static constexpr uint32_t S_FIFO_OFFSET = 1;
static constexpr uint32_t S_FIFO_SIZE = 2;
static constexpr uint32_t S_NORTH_OFFSET = 15;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 3;
static constexpr uint32_t S_SOUTH_SIZE = 8;
static constexpr uint32_t S_TRACE_OFFSET = 23;
static constexpr uint32_t S_TRACE_SIZE = 1;
static constexpr uint32_t S_WEST_OFFSET = 11;
static constexpr uint32_t S_WEST_SIZE = 4;

// Master port offset/size constants
static constexpr uint32_t M_CTRL_OFFSET = 0;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_EAST_OFFSET = 19;
static constexpr uint32_t M_EAST_SIZE = 4;
static constexpr uint32_t M_FIFO_OFFSET = 1;
static constexpr uint32_t M_FIFO_SIZE = 2;
static constexpr uint32_t M_NORTH_OFFSET = 13;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 3;
static constexpr uint32_t M_SOUTH_SIZE = 6;
static constexpr uint32_t M_WEST_OFFSET = 9;
static constexpr uint32_t M_WEST_SIZE = 4;

} // namespace shim
} // namespace aie1_port_id
} // anonymous namespace

std::optional<uint32_t> AIE1TargetModel::getStreamSwitchPortIndex(
    int col, int row, WireBundle bundle, uint32_t port_num, bool master) const {
  if (master) {
    if (isCoreTile(col, row)) {
      switch (bundle) {
      case WireBundle::Core:
        if (port_num >= aie1_port_id::core::M_CORE_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_CORE_OFFSET + port_num;
      case WireBundle::TileControl:
        if (port_num >= aie1_port_id::core::M_CTRL_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie1_port_id::core::M_DMA_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_DMA_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie1_port_id::core::M_EAST_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie1_port_id::core::M_FIFO_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie1_port_id::core::M_NORTH_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie1_port_id::core::M_SOUTH_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_SOUTH_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie1_port_id::core::M_WEST_SIZE)
          return std::nullopt;
        return aie1_port_id::core::M_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isShimNOCorPLTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie1_port_id::shim::M_CTRL_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_CTRL_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie1_port_id::shim::M_EAST_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie1_port_id::shim::M_FIFO_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie1_port_id::shim::M_NORTH_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie1_port_id::shim::M_SOUTH_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_SOUTH_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie1_port_id::shim::M_WEST_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::M_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  } else {
    if (isCoreTile(col, row)) {
      switch (bundle) {
      case WireBundle::Core:
        if (port_num >= aie1_port_id::core::S_CORE_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_CORE_OFFSET + port_num;
      case WireBundle::TileControl:
        if (port_num >= aie1_port_id::core::S_CTRL_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie1_port_id::core::S_DMA_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_DMA_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie1_port_id::core::S_EAST_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie1_port_id::core::S_FIFO_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie1_port_id::core::S_NORTH_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie1_port_id::core::S_SOUTH_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie1_port_id::core::S_TRACE_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_TRACE_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie1_port_id::core::S_WEST_SIZE)
          return std::nullopt;
        return aie1_port_id::core::S_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isShimNOCorPLTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie1_port_id::shim::S_CTRL_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_CTRL_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie1_port_id::shim::S_EAST_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie1_port_id::shim::S_FIFO_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie1_port_id::shim::S_NORTH_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie1_port_id::shim::S_SOUTH_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie1_port_id::shim::S_TRACE_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_TRACE_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie1_port_id::shim::S_WEST_SIZE)
          return std::nullopt;
        return aie1_port_id::shim::S_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }
}

bool AIE1TargetModel::isValidStreamSwitchPort(int col, int row,
                                              WireBundle bundle,
                                              uint32_t channel,
                                              bool master) const {
  // TODO: Add proper validation
  // For now, accept reasonable-looking configurations
  if (channel > 7)
    return false;

  // Accept common port types
  switch (bundle) {
  case WireBundle::DMA:
  case WireBundle::FIFO:
  case WireBundle::North:
  case WireBundle::South:
  case WireBundle::East:
  case WireBundle::West:
    return true;
  default:
    return false;
  }
}

///
/// AIE2 TargetModel
///

std::unique_ptr<RegisterDatabase>
AIE2TargetModel::loadRegisterDatabase() const {
  return RegisterDatabase::loadAIE2();
}

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

uint64_t AIE2TargetModel::getDmaBdAddress(int col, int row, uint32_t bd_id,
                                          int channel,
                                          AIE::DMAChannelDir direction) const {
  uint64_t offset = 0;
  if (isShimNOCTile(col, row)) {
    offset = 0x0001D000 + bd_id * 0x20;
  } else if (isMemTile(col, row)) {
    offset = 0x000A0000 + bd_id * 0x20;
  } else if (isCoreTile(col, row)) {
    offset = 0x0001D000 + bd_id * 0x20;
  } else {
    llvm_unreachable(
        "AIE2TargetModel::getDmaBdAddress called for non-DMA tile");
  }
  return ((col & 0xff) << getColumnShift()) | ((row & 0xff) << getRowShift()) |
         offset;
}

uint32_t AIE2TargetModel::getDmaBdAddressOffset(int col, int row) const {
  if (isCoreTile(col, row))
    return 0x0;
  return 0x4;
}

uint32_t
AIE2TargetModel::getDmaControlAddress(int col, int row, int channel,
                                      AIE::DMAChannelDir direction) const {
  uint32_t offset = 0;
  if (isShimNOCTile(col, row)) {
    offset = 0x0001D200 + (channel * 0x8);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x10;
  } else if (isMemTile(col, row)) {
    offset = 0x000A0600 + (channel * 0x8);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x30;
  } else if (isCoreTile(col, row)) {
    offset = 0x0001DE00 + (channel * 0x8);
    if (direction == AIE::DMAChannelDir::MM2S)
      offset += 0x10;
  } else {
    llvm_unreachable(
        "AIE2TargetModel::getDmaControlAddress called for non-DMA tile");
  }

  return ((col & 0xff) << getColumnShift()) | ((row & 0xff) << getRowShift()) |
         offset;
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
    case WireBundle::TileControl:
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
    case WireBundle::TileControl:
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
  case WireBundle::TileControl:
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
    case WireBundle::TileControl:
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
    case WireBundle::TileControl:
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
  case WireBundle::TileControl:
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
      return 8; // Connection to the south port of the stream switch
    default:
      return 0;
    }

  return 0;
}

uint32_t
AIE2TargetModel::getNumSourceShimMuxConnections(int col, int row,
                                                WireBundle bundle) const {
  if (isShimNOCorPLTile(col, row))
    switch (bundle) {
    case WireBundle::DMA:
      return 2;
    case WireBundle::NOC:
      return 4;
    case WireBundle::PLIO:
      return 8;
    case WireBundle::South:
      return 6; // Connection to the south port of the stream switch
    default:
      return 0;
    }

  return 0;
}

bool AIE2TargetModel::isLegalTileConnection(int col, int row,
                                            WireBundle srcBundle, int srcChan,
                                            WireBundle dstBundle,
                                            int dstChan) const {
  // Check Channel Id within the range
  if (srcChan >= int(getNumSourceSwitchboxConnections(col, row, srcBundle)))
    return false;
  if (dstChan >= int(getNumDestSwitchboxConnections(col, row, dstBundle)))
    return false;

  // Lambda function to check if a bundle is in a list
  auto isBundleInList = [](WireBundle bundle,
                           std::initializer_list<WireBundle> bundles) {
    return std::find(bundles.begin(), bundles.end(), bundle) != bundles.end();
  };

  // Memtile
  if (isMemTile(col, row)) {
    if (srcBundle == WireBundle::DMA) {
      if (dstBundle == WireBundle::DMA)
        return srcChan == dstChan;
      if (isBundleInList(dstBundle, {WireBundle::TileControl, WireBundle::South,
                                     WireBundle::North}))
        return true;
    }
    if (srcBundle == WireBundle::TileControl) {
      if (dstBundle == WireBundle::DMA)
        return dstChan == 5;
      if (isBundleInList(dstBundle, {WireBundle::South, WireBundle::North}))
        return true;
    }
    if (isBundleInList(srcBundle, {WireBundle::South, WireBundle::North})) {
      if (isBundleInList(dstBundle, {WireBundle::DMA, WireBundle::TileControl}))
        return true;
      if (isBundleInList(dstBundle, {WireBundle::South, WireBundle::North}))
        return srcChan == dstChan;
    }
    if (srcBundle == WireBundle::Trace) {
      if (dstBundle == WireBundle::DMA)
        return dstChan == 5;
      if (dstBundle == WireBundle::South)
        return true;
    }
  }
  // Shimtile
  else if (isShimNOCorPLTile(col, row)) {
    if (srcBundle == WireBundle::TileControl)
      return dstBundle != WireBundle::TileControl;
    if (isBundleInList(srcBundle, {WireBundle::FIFO, WireBundle::South}))
      return isBundleInList(dstBundle,
                            {WireBundle::TileControl, WireBundle::FIFO,
                             WireBundle::South, WireBundle::West,
                             WireBundle::North, WireBundle::East});
    if (isBundleInList(srcBundle,
                       {WireBundle::West, WireBundle::North, WireBundle::East}))
      return (srcBundle == dstBundle)
                 ? (srcChan == dstChan)
                 : isBundleInList(dstBundle,
                                  {WireBundle::TileControl, WireBundle::FIFO,
                                   WireBundle::South, WireBundle::West,
                                   WireBundle::North, WireBundle::East});
    if (srcBundle == WireBundle::Trace) {
      if (isBundleInList(dstBundle, {WireBundle::FIFO, WireBundle::South}))
        return true;
      if (isBundleInList(dstBundle, {WireBundle::West, WireBundle::East}))
        return dstChan == 0;
    }
  }
  // Coretile
  else if (isCoreTile(col, row)) {
    if (isBundleInList(srcBundle,
                       {WireBundle::DMA, WireBundle::FIFO, WireBundle::South,
                        WireBundle::West, WireBundle::North, WireBundle::East}))
      if (isBundleInList(dstBundle, {WireBundle::Core, WireBundle::DMA,
                                     WireBundle::TileControl, WireBundle::FIFO,
                                     WireBundle::South, WireBundle::West,
                                     WireBundle::North, WireBundle::East}))
        return (srcBundle == dstBundle) ? (srcChan == dstChan) : true;
    if (srcBundle == WireBundle::Core)
      return dstBundle != WireBundle::Core;
    if (srcBundle == WireBundle::TileControl)
      return dstBundle != WireBundle::TileControl &&
             dstBundle != WireBundle::DMA;
    if (srcBundle == WireBundle::Trace) {
      if (dstBundle == WireBundle::DMA)
        return dstChan == 0;
      if (isBundleInList(dstBundle, {WireBundle::FIFO, WireBundle::South}))
        return true;
    }
  }
  return false;
}

std::vector<std::pair<uint32_t, uint32_t>>
AIE2TargetModel::getShimBurstEncodingsAndLengths() const {
  return {std::pair(0, 64), std::pair(1, 128), std::pair(2, 256)};
}

std::optional<uint32_t>
AIE2TargetModel::getLocalLockAddress(uint32_t lockId, TileID tile) const {
  auto computeTileBaseAddress = 0x0001F000;
  auto memTileBaseAddress = 0x000C0000;
  auto shimTileBaseAddress = 0x00014000;
  auto lockAddrOffset = 0x10;

  if (isCoreTile(tile.col, tile.row) &&
      lockId < getNumLocks(tile.col, tile.row))
    return computeTileBaseAddress + lockAddrOffset * lockId;

  if (isMemTile(tile.col, tile.row) && lockId < getNumLocks(tile.col, tile.row))
    return memTileBaseAddress + lockAddrOffset * lockId;

  if (isShimNOCorPLTile(tile.col, tile.row) &&
      lockId < getNumLocks(tile.col, tile.row))
    return shimTileBaseAddress + lockAddrOffset * lockId;

  return std::nullopt;
}

namespace {
namespace aie2_port_id {
namespace core {

// Slave port offset/size constants
static constexpr uint32_t S_CORE_OFFSET = 0;
static constexpr uint32_t S_CORE_SIZE = 1;
static constexpr uint32_t S_CTRL_OFFSET = 3;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_DMA_OFFSET = 1;
static constexpr uint32_t S_DMA_SIZE = 2;
static constexpr uint32_t S_EAST_OFFSET = 19;
static constexpr uint32_t S_EAST_SIZE = 4;
static constexpr uint32_t S_FIFO_OFFSET = 4;
static constexpr uint32_t S_FIFO_SIZE = 1;
static constexpr uint32_t S_NORTH_OFFSET = 15;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 5;
static constexpr uint32_t S_SOUTH_SIZE = 6;
static constexpr uint32_t S_TRACE_OFFSET = 23;
static constexpr uint32_t S_TRACE_SIZE = 2;
static constexpr uint32_t S_WEST_OFFSET = 11;
static constexpr uint32_t S_WEST_SIZE = 4;

// Master port offset/size constants
static constexpr uint32_t M_CORE_OFFSET = 0;
static constexpr uint32_t M_CORE_SIZE = 1;
static constexpr uint32_t M_CTRL_OFFSET = 3;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_DMA_OFFSET = 1;
static constexpr uint32_t M_DMA_SIZE = 2;
static constexpr uint32_t M_EAST_OFFSET = 19;
static constexpr uint32_t M_EAST_SIZE = 4;
static constexpr uint32_t M_FIFO_OFFSET = 4;
static constexpr uint32_t M_FIFO_SIZE = 1;
static constexpr uint32_t M_NORTH_OFFSET = 13;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 5;
static constexpr uint32_t M_SOUTH_SIZE = 4;
static constexpr uint32_t M_WEST_OFFSET = 9;
static constexpr uint32_t M_WEST_SIZE = 4;

} // namespace core

namespace mem {

// Slave port offset/size constants
static constexpr uint32_t S_CTRL_OFFSET = 6;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_DMA_OFFSET = 0;
static constexpr uint32_t S_DMA_SIZE = 6;
static constexpr uint32_t S_NORTH_OFFSET = 13;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 7;
static constexpr uint32_t S_SOUTH_SIZE = 6;
static constexpr uint32_t S_TRACE_OFFSET = 17;
static constexpr uint32_t S_TRACE_SIZE = 1;

// Master port offset/size constants
static constexpr uint32_t M_CTRL_OFFSET = 6;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_DMA_OFFSET = 0;
static constexpr uint32_t M_DMA_SIZE = 6;
static constexpr uint32_t M_NORTH_OFFSET = 11;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 7;
static constexpr uint32_t M_SOUTH_SIZE = 4;

} // namespace mem

namespace shim {

// Slave port offset/size constants
static constexpr uint32_t S_CTRL_OFFSET = 0;
static constexpr uint32_t S_CTRL_SIZE = 1;
static constexpr uint32_t S_EAST_OFFSET = 18;
static constexpr uint32_t S_EAST_SIZE = 4;
static constexpr uint32_t S_FIFO_OFFSET = 1;
static constexpr uint32_t S_FIFO_SIZE = 1;
static constexpr uint32_t S_NORTH_OFFSET = 14;
static constexpr uint32_t S_NORTH_SIZE = 4;
static constexpr uint32_t S_SOUTH_OFFSET = 2;
static constexpr uint32_t S_SOUTH_SIZE = 8;
static constexpr uint32_t S_TRACE_OFFSET = 22;
static constexpr uint32_t S_TRACE_SIZE = 2;
static constexpr uint32_t S_WEST_OFFSET = 10;
static constexpr uint32_t S_WEST_SIZE = 4;

// Master port offset/size constants
static constexpr uint32_t M_CTRL_OFFSET = 0;
static constexpr uint32_t M_CTRL_SIZE = 1;
static constexpr uint32_t M_EAST_OFFSET = 18;
static constexpr uint32_t M_EAST_SIZE = 4;
static constexpr uint32_t M_FIFO_OFFSET = 1;
static constexpr uint32_t M_FIFO_SIZE = 1;
static constexpr uint32_t M_NORTH_OFFSET = 12;
static constexpr uint32_t M_NORTH_SIZE = 6;
static constexpr uint32_t M_SOUTH_OFFSET = 2;
static constexpr uint32_t M_SOUTH_SIZE = 6;
static constexpr uint32_t M_WEST_OFFSET = 8;
static constexpr uint32_t M_WEST_SIZE = 4;

} // namespace shim
} // namespace aie2_port_id
} // namespace

std::optional<uint32_t> AIE2TargetModel::getStreamSwitchPortIndex(
    int col, int row, WireBundle bundle, uint32_t port_num, bool master) const {

  if (master) {
    if (isCoreTile(col, row)) {
      switch (bundle) {
      case WireBundle::Core:
        if (port_num >= aie2_port_id::core::M_CORE_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_CORE_OFFSET + port_num;
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::core::M_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie2_port_id::core::M_DMA_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_DMA_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie2_port_id::core::M_EAST_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie2_port_id::core::M_FIFO_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::core::M_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::core::M_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_SOUTH_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie2_port_id::core::M_WEST_SIZE)
          return std::nullopt;
        return aie2_port_id::core::M_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isMemTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::mem::M_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::M_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie2_port_id::mem::M_DMA_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::M_DMA_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::mem::M_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::mem::M_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::M_SOUTH_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isShimNOCTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::shim::M_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_CTRL_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie2_port_id::shim::M_EAST_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie2_port_id::shim::M_FIFO_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::shim::M_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::shim::M_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_SOUTH_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie2_port_id::shim::M_WEST_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::M_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    }
    // Slave ports
  } else {

    if (isCoreTile(col, row)) {
      switch (bundle) {
      case WireBundle::Core:
        if (port_num >= aie2_port_id::core::S_CORE_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_CORE_OFFSET + port_num;
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::core::S_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie2_port_id::core::S_DMA_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_DMA_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie2_port_id::core::S_EAST_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie2_port_id::core::S_FIFO_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::core::S_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::core::S_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie2_port_id::core::S_TRACE_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_TRACE_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie2_port_id::core::S_WEST_SIZE)
          return std::nullopt;
        return aie2_port_id::core::S_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isMemTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::mem::S_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_CTRL_OFFSET + port_num;
      case WireBundle::DMA:
        if (port_num >= aie2_port_id::mem::S_DMA_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_DMA_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::mem::S_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::mem::S_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie2_port_id::mem::S_TRACE_SIZE)
          return std::nullopt;
        return aie2_port_id::mem::S_TRACE_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    } else if (isShimNOCTile(col, row)) {
      switch (bundle) {
      case WireBundle::TileControl:
        if (port_num >= aie2_port_id::shim::S_CTRL_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_CTRL_OFFSET + port_num;
      case WireBundle::East:
        if (port_num >= aie2_port_id::shim::S_EAST_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_EAST_OFFSET + port_num;
      case WireBundle::FIFO:
        if (port_num >= aie2_port_id::shim::S_FIFO_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_FIFO_OFFSET + port_num;
      case WireBundle::North:
        if (port_num >= aie2_port_id::shim::S_NORTH_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_NORTH_OFFSET + port_num;
      case WireBundle::South:
        if (port_num >= aie2_port_id::shim::S_SOUTH_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_SOUTH_OFFSET + port_num;
      case WireBundle::Trace:
        if (port_num >= aie2_port_id::shim::S_TRACE_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_TRACE_OFFSET + port_num;
      case WireBundle::West:
        if (port_num >= aie2_port_id::shim::S_WEST_SIZE)
          return std::nullopt;
        return aie2_port_id::shim::S_WEST_OFFSET + port_num;
      default:
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

bool AIE2TargetModel::isValidStreamSwitchPort(int col, int row,
                                              WireBundle bundle,
                                              uint32_t channel,
                                              bool master) const {
  // TODO: Add proper validation
  // For now, accept reasonable-looking configurations
  if (channel > 7)
    return false;

  // Accept common port types
  switch (bundle) {
  case WireBundle::DMA:
  case WireBundle::FIFO:
  case WireBundle::North:
  case WireBundle::South:
  case WireBundle::East:
  case WireBundle::West:
    return true;
  default:
    return false;
  }
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

std::optional<uint32_t>
AIETargetModel::getLockLocalBaseIndex(int localCol, int localRow, int lockCol,
                                      int lockRow) const {
  if (isCoreTile(localCol, localRow)) {
    if (isMemSouth(localCol, localRow, lockCol, lockRow))
      return 0;
    if (isMemWest(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow);
    if (isMemNorth(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow) * 2;
    if (isMemEast(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow) * 3;
  }

  if (isMemTile(localCol, localRow)) {
    if (isWest(localCol, localRow, lockCol, lockRow))
      return 0;
    if (isInternal(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow);
    if (isEast(localCol, localRow, lockCol, lockRow))
      return getNumLocks(localCol, localRow) * 2;
  }

  return std::nullopt;
}

std::optional<uint32_t>
AIETargetModel::getMemLocalBaseAddress(int localCol, int localRow, int memCol,
                                       int memRow) const {
  if (isCoreTile(localCol, localRow)) {
    if (isMemSouth(localCol, localRow, memCol, memRow))
      return getMemSouthBaseAddress();
    if (isMemWest(localCol, localRow, memCol, memRow))
      return getMemWestBaseAddress();
    if (isMemNorth(localCol, localRow, memCol, memRow))
      return getMemNorthBaseAddress();
    if (isMemEast(localCol, localRow, memCol, memRow))
      return getMemEastBaseAddress();
  }

  if (isMemTile(localCol, localRow)) {
    if (isWest(localCol, localRow, memCol, memRow))
      return 0;
    if (isInternal(localCol, localRow, memCol, memRow))
      return getMemTileSize();
    if (isEast(localCol, localRow, memCol, memRow))
      return getMemTileSize() * 2;
  }

  return std::nullopt;
}

bool AIETargetModel::isSupportedBlockFormat(std::string const &format) const {
  return false;
}

AIEArch BaseNPU2TargetModel::getTargetArch() const { return AIEArch::AIE2p; }

std::vector<std::pair<uint32_t, uint32_t>>
BaseNPU2TargetModel::getShimBurstEncodingsAndLengths() const {
  return {std::pair(0, 64), std::pair(1, 128), std::pair(2, 256),
          std::pair(3, 512)};
}

bool BaseNPU2TargetModel::isSupportedBlockFormat(
    std::string const &format) const {
  std::set<std::string> supportedTypes = {"v8bfp16ebs8", "v16bfp16ebs16"};
  return static_cast<bool>(supportedTypes.find(format) != supportedTypes.end());
}

} // namespace AIE
} // namespace xilinx
