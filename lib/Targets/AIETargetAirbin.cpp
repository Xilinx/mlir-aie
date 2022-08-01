//===- AIETargetAirbin.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <elf.h>
#include <fcntl.h>
#include <sstream>
#include <unistd.h>

#include "aie/AIEDialect.h"
#include "aie/AIENetlistAnalysis.h"

#include "AIETargets.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx {
namespace AIE {

static constexpr auto disable = 0u;
static constexpr auto enable = 1u;
static constexpr auto MAX_CHANNEL_COUNT = 4u;

static constexpr auto TILE_ADDR_OFF_WITDH = 18u;

static constexpr auto TILE_ADDR_ROW_SHIFT = TILE_ADDR_OFF_WITDH;
static constexpr auto TILE_ADDR_ROW_WIDTH = 5u;

static constexpr auto TILE_ADDR_COL_SHIFT =
    TILE_ADDR_ROW_SHIFT + TILE_ADDR_ROW_WIDTH;
static constexpr auto TILE_ADDR_COL_WIDTH = 7u;

static constexpr auto TILE_ADDR_ARR_SHIFT =
    TILE_ADDR_COL_SHIFT + TILE_ADDR_COL_WIDTH;

static constexpr auto CORE_CORECTRL = 0x00032000u;
static constexpr auto CORE_CTRL_ENABLE_SHIFT = 0u;
static constexpr auto CORE_CTRL_ENABLE_MASK = 1u;
static constexpr auto CORE_CTRL_RESET_SHIFT = 1u;
static constexpr auto CORE_CTRL_RESET_MASK = 2u;

/*
 * Tile address format:
 * --------------------------------------------
 * |                7 bits  5 bits   18 bits  |
 * --------------------------------------------
 * | Array offset | Column | Row | Tile addr  |
 * --------------------------------------------
 */
class TileAddress {
public:
  TileAddress(uint8_t column, uint8_t row, uint64_t array_offset = 0x800u)
      : array_offset{array_offset}, column{column}, row{row} {}

  uint64_t fullAddress(uint32_t register_offset) const {
    return (0x800ul << TILE_ADDR_ARR_SHIFT) |
           (static_cast<uint64_t>(column) << TILE_ADDR_COL_SHIFT) |
           (static_cast<uint64_t>(row) << TILE_ADDR_ROW_SHIFT) |
           register_offset;
  }

private:
  uint64_t array_offset : 34;
  uint8_t column : TILE_ADDR_COL_WIDTH;
  uint8_t row : TILE_ADDR_ROW_WIDTH;
};

class Address {
public:
  Address(TileAddress tile, uint32_t register_offset)
      : tile{tile}, register_offset{register_offset} {}

  operator uint64_t() const { return tile.fullAddress(register_offset); }

private:
  TileAddress tile;
  uint32_t register_offset : TILE_ADDR_ROW_SHIFT;
};

class Write {
public:
  Write(Address addr, uint32_t data) : address{addr}, data{data} {
    assert(address % 4 == 0);
  }

private:
  Address address;
  uint32_t data;
};

static std::vector<Write> writes;

static void write32(Address addr, uint32_t value) {
  writes.emplace_back(addr, value);
}

// Inclusive on both ends
static void clearRange(TileAddress tile, uint32_t range_start,
                       uint32_t range_end) {
  assert(range_start % 4 == 0);
  assert(range_end % 4 == 0);

  for (auto off = range_start; off <= range_end; off += 4u) {
    write32({tile, off}, 0);
  }
}

enum class ShimTileType { OnlyShim, ShimNOC, ShimNOCOrPL };

// The SHIM row is always 0
static void generateShimConfig(uint8_t col, ShimTileType tile_type) {

  // XAieTile_ShimColumnReset(&(TileInst[col][0]), XAIE_RESETENABLE);
  static constexpr auto PL_AIETILCOLRST = 0x00036048u;
  static constexpr auto RESET_ENABLE = 1u;

  TileAddress tile{col, 0};
  write32({tile, PL_AIETILCOLRST}, RESET_ENABLE);

  switch (tile_type) {
  case ShimTileType::ShimNOC:
    clearRange(tile, 0x1D000, 0x1D13C);
    clearRange(tile, 0x1D140, 0x1D140);
    clearRange(tile, 0x1D148, 0x1D148);
    clearRange(tile, 0x1D150, 0x1D150);
    clearRange(tile, 0x1D158, 0x1D158);
    break;
  case ShimTileType::ShimNOCOrPL:
    // output << "// Stream Switch master config\n";
    clearRange(tile, 0x3F000, 0x3F058);
    // output << "// Stream Switch slave config\n";
    clearRange(tile, 0x3F100, 0x3F15C);
    // output << "// Stream Switch slave slot config\n";
    clearRange(tile, 0x3F200, 0x3F37C);
    break;
  case ShimTileType::OnlyShim:
    break;
  }
  // XAieTile_ShimColumnReset(&(TileInst[col][0]), XAIE_RESETDISABLE);
  write32({tile, PL_AIETILCOLRST}, 0);
}

static constexpr uint64_t setField(uint64_t value, uint8_t shift,
                                   uint64_t mask) {
  return (value << shift) & mask;
}

static bool loadElf(TileAddress tile, const std::string &filename) {

  llvm::dbgs() << "Reading ELF file " << filename;

  int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    return false;
  }

  static constexpr auto READ_SIZE = sizeof(uint32_t);
  unsigned char num[READ_SIZE] = {0};

  auto read_at = [](int fd, unsigned char bytes[READ_SIZE], uint64_t addr) {
    auto ret = lseek(fd, addr, SEEK_SET);
    assert(ret >= 0 and static_cast<uint64_t>(ret) == addr);
    return read(fd, bytes, READ_SIZE) == READ_SIZE;
  };

  auto parse_little_endian = [](unsigned char bytes[READ_SIZE]) {
    return bytes[0] | (((uint32_t)bytes[1]) << 8u) |
           (((uint32_t)bytes[2]) << 16u) | (((uint32_t)bytes[3]) << 24u);
  };

  // check that the elf is LSB
  if (!read_at(fd, num, 4)) {
    return false;
  }

  // Read data as 32-bit little endian
  assert(num[0] == ELFCLASS32);
  assert(num[1] == ELFDATA2LSB);

#define PROGRAM_HEADER_OFFSET                                                  \
  (EI_NIDENT + sizeof(uint16_t) * 2 + sizeof(uint32_t) + sizeof(Elf32_Addr))

  if (!read_at(fd, num, PROGRAM_HEADER_OFFSET)) {
    return false;
  }
  uint32_t phstart = parse_little_endian(num);
  assert(phstart != 0);

#define PROGRAM_HEADER_COUNT_OFFSET                                            \
  (PROGRAM_HEADER_OFFSET + sizeof(Elf32_Off) * 2 + sizeof(uint32_t) +          \
   sizeof(uint16_t))

  if (!read_at(fd, num, PROGRAM_HEADER_COUNT_OFFSET)) {
    return false;
  }

  {
    uint16_t prog_header_size = (((uint16_t)num[1]) << 8u) | num[0];
    assert(prog_header_size > 0);

    uint16_t prog_header_count = (((uint16_t)num[3]) << 8u) | num[2];
    assert(prog_header_count > 0);
  }

  if (!read_at(fd, num, phstart)) {
    return false;
  }
  {
    uint32_t header_type = parse_little_endian(num);
    assert(header_type == PT_LOAD);
  }

  if (!read_at(fd, num, phstart + sizeof(num))) {
    return false;
  }
  uint32_t start = parse_little_endian(num);

  if (!read_at(fd, num, phstart + 2 * sizeof(num))) {
    return false;
  }
  uint32_t dest = parse_little_endian(num);

  if (!read_at(fd, num, phstart + 4 * sizeof(num))) {
    return false;
  }
  uint32_t stop = parse_little_endian(num);

  llvm::dbgs() << "Loading " << filename << " tile @ offset "
               << ((tile.fullAddress(0) >> TILE_ADDR_ROW_SHIFT) & 0xFFFu);

  assert(lseek(fd, start, SEEK_SET) == start);
  for (auto i = 0u; i < stop / sizeof(num); i++) {
    if (read(fd, num, sizeof(num)) < static_cast<long>(sizeof(num))) {
      return false;
    }
    static constexpr auto PROG_MEM_OFFSET = 0x20000u;
    Address dest_addr{
        tile, static_cast<uint32_t>(dest + PROG_MEM_OFFSET + i * READ_SIZE)};
    // The data needs to be read in little endian
    uint32_t data = parse_little_endian(num);
    write32(dest_addr, data);
  }

  return true;
}

static void configure_cores(mlir::ModuleOp module) {

  for (auto tileOp : module.getOps<TileOp>()) {
    auto col = tileOp.colIndex();
    if (tileOp.isShimTile()) {
      auto tile_type = [&tileOp] {
        if (tileOp.isShimNOCorPLTile())
          return ShimTileType::ShimNOCOrPL;
        if (tileOp.isShimNOCTile())
          return ShimTileType::ShimNOC;
        return ShimTileType::OnlyShim;
      }();
      generateShimConfig(col, tile_type);
    } else {
      TileAddress tile{static_cast<uint8_t>(col),
                       static_cast<uint8_t>(tileOp.rowIndex())};

      write32(
          {tile, CORE_CORECTRL},
          setField(disable, CORE_CTRL_ENABLE_SHIFT, CORE_CTRL_ENABLE_MASK) |
              setField(enable, CORE_CTRL_RESET_SHIFT, CORE_CTRL_RESET_MASK));

      // Reset configuration
      // Program Memory
      clearRange(tile, 0x20000, 0x23FFF);
      // TileDMA
      clearRange(tile, 0x1D000, 0x1D1F8);
      clearRange(tile, 0x1DE00, 0x1DE00);
      clearRange(tile, 0x1DE08, 0x1DE08);
      clearRange(tile, 0x1DE10, 0x1DE10);
      clearRange(tile, 0x1DE18, 0x1DE18);
      // Stream Switch master config
      clearRange(tile, 0x3F000, 0x3F060);
      // Stream Switch slave config
      clearRange(tile, 0x3F100, 0x3F168);
      // Stream Switch slave slot config
      clearRange(tile, 0x3F200, 0x3F3AC);

      // NOTE: Here is usually where locking is done.
      // However, the runtime will handle that when loading the airbin.

      if (auto coreOp = tileOp.getCoreOp()) {
        std::string fileName;
        if (auto fileAttr = coreOp->getAttrOfType<StringAttr>("elf_file")) {
          fileName = std::string(fileAttr.getValue());
        } else {
          std::stringstream ss;
          ss << "core_" << col << '_' << tileOp.rowIndex() << ".elf";
          fileName = ss.str();
        }
        if (not loadElf(tile, fileName)) {
          llvm::outs() << "Error loading " << fileName;
        }
      }
    }
  }
}

// Start execution of all the cores.
static void start_cores(mlir::ModuleOp module) {

  for (auto tileOp : module.getOps<TileOp>()) {
    if (!tileOp.isShimTile()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();

      write32(
          {TileAddress{static_cast<uint8_t>(col), static_cast<uint8_t>(row)},
           CORE_CORECTRL},
          setField(enable, CORE_CTRL_ENABLE_SHIFT, CORE_CTRL_ENABLE_MASK) |
              setField(disable, CORE_CTRL_RESET_SHIFT, CORE_CTRL_RESET_MASK));
    }
  }
}

struct BDInfo {
  bool foundBdPacket = false;
  int packetType = 0;
  int packetID = 0;
  bool foundBd = false;
  int lenA = 0;
  int lenB = 0;
  int bytesA = 0;
  int bytesB = 0;
  int offsetA = 0;
  int offsetB = 0;
  int BaseAddrA = 0;
  int BaseAddrB = 0;
  bool hasA = false;
  bool hasB = false;
  std::string bufA = "0";
  std::string bufB = "0";
  uint32_t AbMode = disable;
  uint32_t FifoMode = disable; // FIXME: when to enable FIFO mode?
};

static BDInfo getBDInfo(Block &block, NetlistAnalysis &NL) {
  BDInfo bdInfo;
  for (auto op : block.getOps<DMABDOp>()) {
    bdInfo.foundBd = true;
    auto bufferType = op.buffer().getType().cast<::mlir::MemRefType>();

    if (op.isA()) {
      bdInfo.BaseAddrA = NL.getBufferBaseAddress(op.buffer().getDefiningOp());
      bdInfo.lenA = op.getLenValue();
      bdInfo.bytesA = bufferType.getElementTypeBitWidth() / 8;
      bdInfo.offsetA = op.getOffsetValue();
      bdInfo.bufA = "XAIEDMA_TILE_BD_ADDRA";
      bdInfo.hasA = true;
    }

    if (op.isB()) {
      bdInfo.BaseAddrB = NL.getBufferBaseAddress(op.buffer().getDefiningOp());
      bdInfo.lenB = op.getLenValue();
      bdInfo.bytesB = bufferType.getElementTypeBitWidth() / 8;
      bdInfo.offsetB = op.getOffsetValue();
      bdInfo.bufB = "XAIEDMA_TILE_BD_ADDRB";
      bdInfo.hasB = true;
    }
  }
  return bdInfo;
}

static void configure_dmas(mlir::ModuleOp module, NetlistAnalysis &NL) {
  static constexpr uint32_t dmaChannelCtrlOffsets[4]{
      0x1de00,
      0x1de08,
      0x1de10,
      0x1de18,
  };
  static constexpr uint32_t dmaChannelQueueOffsets[4]{
      0x1de04,
      0x1de0C,
      0x1de14,
      0x1de1C,
  };

  static constexpr auto dmaChannelResetLSB = 1u;
  static constexpr auto dmaChannelResetMask = 0x2u;
  static constexpr auto dmaChannelEnableLSB = 0u;
  static constexpr auto dmaChannelEnableMask = 0x1u;

  /* clang-format off
     DMA configuration
     XAieDma_TileSetStartBd(DmaInstPtr, ChNum, BdStart)
     u32 XAieDma_TileSoftInitialize(XAieGbl_Tile *TileInstPtr, XAieDma_Tile *DmaInstPtr)
     u32 XAieDma_TileInitialize(XAieGbl_Tile *TileInstPtr, XAieDma_Tile *DmaInstPtr);
     void XAieDma_TileBdSetLock(XAieDma_Tile *DmaInstPtr, u8 BdNum, u8 AbType, u8 LockId, u8 LockRelEn, u8 LockRelVal, u8 LockAcqEn, u8 LockAcqVal)
     void XAieDma_TileBdSetXy2d(XAieDma_Tile *DmaInstPtr, u8 BdNum, u8 XyType, u16 Incr, u16 Wrap, u16 Offset);
     void XAieDma_TileBdSetIntlv(XAieDma_Tile *DmaInstPtr, u8 BdNum, u8 IntlvMode, u8 IntlvDb, u8 IntlvCnt, u16 IntlvCur)
     void XAieDma_TileBdSetPkt(XAieDma_Tile *DmaInstPtr, u8 BdNum, u8 PktEn, u8 PktType, u8 PktId)
     void XAieDma_TileBdSetAdrLenMod(XAieDma_Tile *DmaInstPtr, u8 BdNum, u16 BaseAddrA, u16 BaseAddrB, u16 Length, u8 AbMode, u8 FifoMode)
     void XAieDma_TileBdSetNext(XAieDma_Tile *DmaInstPtr, u8 BdNum, u8 NextBd)
     void XAieDma_TileBdWrite(XAieDma_Tile *DmaInstPtr, u8 BdNum)
     void XAieDma_TileBdClear(XAieDma_Tile *DmaInstPtr, u8 BdNum)
     void XAieDma_TileBdClearAll(XAieDma_Tile *DmaInstPtr)
     u32 XAieDma_TileChControl(XAieDma_Tile *DmaInstPtr, u8 ChNum, u8 Reset, u8 Enable)
     u32 XAieDma_TileChReset(XAieDma_Tile *DmaInstPtr, u8 ChNum)
     u32 XAieDma_TileChResetAll(XAieDma_Tile *DmaInstPtr)
     clang-format on
     */

  for (auto memOp : module.getOps<MemOp>()) {
    int col = memOp.colIndex();
    int row = memOp.rowIndex();
    /* clang-format off
    output << "XAieDma_TileInitialize(" << tileInstStr(col, row) << ", " << tileDMAInstStr(col, row) << ");\n";
    ----
    DmaInstPtr->BaseAddress = TileInstPtr->TileAddr;

    Clear the BD entries in the DMA instance structure
    for(BdIdx = 0U; BdIdx < XAIEDMA_TILE_MAX_NUM_DESCRS; BdIdx++) {
      XAieDma_TileBdClear(DmaInstPtr, BdIdx);
    }
    clang-format on */

    // Note: `TileInitialize` already clears the bds,
    // so calling `TileBdClearAll` is unneeded.

    // Note: `TileBdClear` does not call `Write32`

    /* clang-format off
    output << "XAieDma_TileChResetAll(" << tileDMAInstStr(col, row) << ");\n";
    ----
    for (ChNum = 0U; ChNum < XAIEDMA_TILE_MAX_NUM_CHANNELS; ChNum++) {
      Status = XAieDma_TileChReset(DmaInstPtr, ChNum);
      if (Status == XAIE_FAILURE) {
        Ret = XAIE_FAILURE;
      }
    }
    ----
    for (ChNum = 0U; ChNum < XAIEDMA_TILE_MAX_NUM_CHANNELS; ChNum++) {
      // Reset the channel
      Status = XAieDma_TileChControl(DmaInstPtr, ChNum, XAIE_RESETENABLE, XAIE_DISABLE);
      if (Status == XAIE_FAILURE) {
        return Status;
      }

      // Unreset and Disable the channel
      Status = XAieDma_TileChControl(DmaInstPtr, ChNum, XAIE_RESETDISABLE, XAIE_DISABLE);
      if (Status == XAIE_FAILURE) {
        return Status;
      }

      // Set Start BD to the reset value
      XAieDma_TileSetStartBd(DmaInstPtr, ChNum, XAIEDMA_TILE_STARTBD_RESET);
    }
    ----
    for (ChNum = 0U; ChNum < XAIEDMA_TILE_MAX_NUM_CHANNELS; ChNum++) {
      // NOTE: the first call to `TileChControl` is overwritten by the second.

      // Unreset and Disable the channel
      RegAddr = DmaInstPtr->BaseAddress + TileDmaCh[ChNum].CtrlOff;
      // Frame the register value
      RegVal = XAie_SetField(XAIE_RESETDISABLE, TileDmaCh[ChNum].Rst.Lsb, TileDmaCh[ChNum].Rst.Mask) |
              XAie_SetField(XAIE_ENABLE, TileDmaCh[ChNum].En.Lsb, TileDmaCh[ChNum].En.Mask);

      // Write to channel control register
      XAieGbl_Write32(RegAddr, RegVal);
      DmaInstPtr->StartBd[ChNum] = BdStart;
    }
    clang-format on */

    TileAddress tile{static_cast<uint8_t>(col), static_cast<uint8_t>(row)};
    for (auto chNum = 0u; chNum < MAX_CHANNEL_COUNT; ++chNum) {
      write32({tile, dmaChannelCtrlOffsets[chNum]},
              setField(disable, dmaChannelResetLSB, dmaChannelResetMask) |
                  setField(enable, dmaChannelEnableLSB, dmaChannelEnableMask));
    }

    DenseMap<Block *, int> blockMap;

    {
      // Assign each block a BD number
      auto bdNum = 0;
      for (auto &block : memOp.body()) {
        if (not block.getOps<DMABDOp>().empty()) {
          blockMap[&block] = bdNum;
          bdNum++;
        }
      }
    }

    for (auto &block : memOp.body()) {
      auto bdInfo = getBDInfo(block, NL);

      if (bdInfo.hasA and bdInfo.hasB) {
        bdInfo.AbMode = enable;
        if (bdInfo.lenA != bdInfo.lenB)
          llvm::errs() << "ABmode must have matching lengths.\n";
        if (bdInfo.bytesA != bdInfo.bytesB)
          llvm::errs() << "ABmode must have matching element data types.\n";
      }
      int acqValue = 0, relValue = 0;
      auto acqEnable = disable;
      auto relEnable = disable;
      int lockID;

      for (auto op : block.getOps<UseLockOp>()) {
        LockOp lock = dyn_cast<LockOp>(op.lock().getDefiningOp());
        lockID = lock.getLockID();
        if (op.acquire()) {
          acqEnable = enable;
          acqValue = op.getLockValue();
        } else if (op.release()) {
          relEnable = enable;
          relValue = op.getLockValue();
        }
      }

      for (auto op : block.getOps<DMABDPACKETOp>()) {
        bdInfo.foundBdPacket = true;
        bdInfo.packetType = op.getPacketType();
        bdInfo.packetID = op.getPacketID();
      }

      auto bdNum = blockMap[&block];
      if (bdInfo.foundBd) {
        if (bdInfo.hasA) {
          /*
          output << "XAieDma_TileBdSetLock(" << tileDMAInstStr(col, row) << ", "
                 << bdNum << ", " << bufA << ", " << lockID << ", " << relEnable
                 << ", " << relValue << ", " << acqEnable << ", " << acqValue
                 << ");\n";
                 */
        }
        if (bdInfo.hasB) {
          /*
          output << "XAieDma_TileBdSetLock(" << tileDMAInstStr(col, row) << ", "
                 << bdNum << ", " << bufB << ", " << lockID << ", " << relEnable
                 << ", " << relValue << ", " << acqEnable << ", " << acqValue
                 << ");\n";
                 */
        }

        /*
                output << "XAieDma_TileBdSetAdrLenMod(" << tileDMAInstStr(col,
           row)
                       << ", " << bdNum << ", "
                       << "0x" << llvm::utohexstr(BaseAddrA + offsetA) << ", "
                       << "0x" << llvm::utohexstr(BaseAddrB + offsetB) << ", "
           << lenA
                       << " * " << bytesA << ", " << AbMode << ", " << FifoMode
                       << ");\n";
                       */

        if (block.getNumSuccessors() > 0) {
          // should have only one successor block
          assert(block.getNumSuccessors() == 1);
          auto *nextBlock = block.getSuccessors()[0];
          auto nextBdNum = blockMap[nextBlock];

          /*
          output << "XAieDma_TileBdSetNext(" << tileDMAInstStr(col, row) << ", "
                           << bdNum << ", " << nextBdNum << ");\n";
          */
        }

        if (bdInfo.foundBdPacket) {
          /*
          output << "XAieDma_TileBdSetPkt(" << tileDMAInstStr(col, row) << ", "
                 << bdNum << ", " << 1 << ", " << packetType << ", " << packetID
                 << ");\n";
                 */
        }
        /*
        output << "XAieDma_TileBdWrite(" << tileDMAInstStr(col, row) << ", "
               << bdNum << ");\n";
               */
      }
    }

    for (auto &block : memOp.body()) {
      for (auto op : block.getOps<DMAStartOp>()) {
        auto bdNum = blockMap[op.dest()];

        /*
        output << "XAieDma_TileSetStartBd("
               << "(" << tileDMAInstStr(col, row) << ")"
               << ", "
               << "XAIEDMA_TILE_CHNUM_" << stringifyDMAChan(op.dmaChan())
               << ", " << bdNum << ");\n";
               */

        if (bdNum != 0xFFU) {

          uint32_t chNum;
          switch (op.dmaChan()) {
          case DMAChan::S2MM0:
            chNum = 0;
            break;
          case DMAChan::S2MM1:
            chNum = 1;
            break;
          case DMAChan::MM2S0:
            chNum = 2;
            break;
          case DMAChan::MM2S1:
            chNum = 3;
            break;
          default:
            assert(false);
          }

          write32(Address{tile, dmaChannelQueueOffsets[chNum]},
                  setField(bdNum, 0u, 0xFu));

          write32(
              {tile, dmaChannelCtrlOffsets[chNum]},
              setField(disable, dmaChannelResetLSB, dmaChannelResetMask) |
                  setField(enable, dmaChannelEnableLSB, dmaChannelEnableMask));
        }
      }
    }

    // XAieDma_Shim ShimDmaInst1;
    // u32 XAieDma_ShimSoftInitialize(XAieGbl_Tile *TileInstPtr,
    // XAieDma_Shim *DmaInstPtr); void XAieDma_ShimInitialize(XAieGbl_Tile
    // *TileInstPtr, XAieDma_Shim *DmaInstPtr); u32
    // XAieDma_ShimChReset(XAieDma_Shim *DmaInstPtr, u8 ChNum); u32
    // XAieDma_ShimChResetAll(XAieDma_Shim *DmaInstPtr); void
    // XAieDma_ShimBdSetLock(XAieDma_Shim *DmaInstPtr, u8 BdNum, u8 LockId,
    // u8 LockRelEn, u8 LockRelVal, u8 LockAcqEn, u8 LockAcqVal); void
    // XAieDma_ShimBdSetAxi(XAieDma_Shim *DmaInstPtr, u8 BdNum, u8 Smid, u8
    // BurstLen, u8 Qos, u8 Cache, u8 Secure); void
    // XAieDma_ShimBdSetPkt(XAieDma_Shim *DmaInstPtr, u8 BdNum, u8 PktEn, u8
    // PktType, u8 PktId); void XAieDma_ShimBdSetNext(XAieDma_Shim
    // *DmaInstPtr, u8 BdNum, u8 NextBd); void
    // XAieDma_ShimBdSetAddr(XAieDma_Shim *DmaInstPtr, u8 BdNum, u16
    // AddrHigh, u32 AddrLow, u32 Length); void
    // XAieDma_ShimBdWrite(XAieDma_Shim *DmaInstPtr, u8 BdNum); void
    // XAieDma_ShimBdClear(XAieDma_Shim *DmaInstPtr, u8 BdNum); void
    // XAieDma_ShimBdClearAll(XAieDma_Shim *DmaInstPtr); u8
    // XAieDma_ShimWaitDone(XAieDma_Shim *DmaInstPtr, u32 ChNum, u32
    // TimeOut); u8 XAieDma_ShimPendingBdCount(XAieDma_Shim *DmaInstPtr, u32
    // ChNum);
    auto index = 0;
    for (auto op : module.getOps<ShimDMAOp>()) {
      auto col = op.colIndex();
      auto row = op.rowIndex();
      /*
      std::string dmaName =
          shimDMAInstStr(std::to_string(col), std::to_string(index));
      output << "XAieDma_Shim " << dmaName << ";\n";
      output << "XAieDma_ShimInitialize(" << tileInstStr(col, row) << ", &"
             << dmaName << ");\n";
             */

      DenseMap<Block *, int> blockMap;

      {
        // Assign each block a BD number
        int bdNum = 0;
        for (auto &block : op.body()) {
          if (!block.getOps<DMABDOp>().empty()) {
            blockMap[&block] = bdNum;
            bdNum++;
          }
        }
      }

      for (auto &block : op.body()) {
        bool foundBd = false;
        int len = 0;
        uint64_t bytes = 0;
        uint64_t offset = 0;
        uint64_t BaseAddr = 0;

        for (auto op : block.getOps<DMABDOp>()) {
          foundBd = true;
          len = op.getLenValue();
          auto bufferType = op.buffer().getType().cast<::mlir::MemRefType>();
          bytes = bufferType.getElementTypeBitWidth() / 8u;
          BaseAddr = NL.getBufferBaseAddress(op.buffer().getDefiningOp());
          offset = op.getOffsetValue();
        }

        int acqValue = 0, relValue = 0;
        bool hasLock = false;
        auto acqEnable = disable;
        auto relEnable = disable;
        int lockID = 0;
        for (auto op : block.getOps<UseLockOp>()) {
          auto lock = dyn_cast<LockOp>(op.lock().getDefiningOp());
          lockID = lock.getLockID();
          hasLock = true;
          if (op.acquire()) {
            acqEnable = enable;
            acqValue = op.getLockValue();
          } else if (op.release()) {
            relEnable = enable;
            relValue = op.getLockValue();
          }
        }

        int bdNum = blockMap[&block];
        if (foundBd) {
          // void XAieDma_ShimBdSetLock(XAieDma_Shim *DmaInstPtr, u8 BdNum,
          // u8 LockId, u8 LockRelEn, u8 LockRelVal, u8 LockAcqEn, u8
          // LockAcqVal);
          if (hasLock) {
            /*
            output << "XAieDma_ShimBdSetLock(&" << dmaName << ", " << bdNum
                   << ", " << lockID << ", " << relEnable << ", " << relValue
                   << ", " << acqEnable << ", " << acqValue << ");\n";
                   */
          }
          // void XAieDma_ShimBdSetAddr(XAieDma_Shim *DmaInstPtr, u8 BdNum,
          // u16 AddrHigh, u32 AddrLow, u32 Length);
          uint64_t address = BaseAddr + offset;
          /*
          output << "XAieDma_ShimBdSetAddr(&" << dmaName << ", " << bdNum << ",
          "
                 << "HIGH_ADDR((u64)0x" << llvm::utohexstr(address) << "), "
                 << "LOW_ADDR((u64)0x" << llvm::utohexstr(address) << "), " <<
          len
                 << " * " << bytes << ");\n";
                 */

          // void XAieDma_ShimBdSetAxi(XAieDma_Shim *DmaInstPtr, u8 BdNum,
          // u8 Smid, u8 BurstLen, u8 Qos, u8 Cache, u8 Secure);
          /*
          output << "XAieDma_ShimBdSetAxi(&" << dmaName << ", " << bdNum << ", "
                 << " 0, "
                 << " 4, "
                 << " 0, "
                 << " 0, "
                 << " " << enable << ");\n";
                 */

          if (block.getNumSuccessors() > 0) {
            // should have only one successor block
            assert(block.getNumSuccessors() == 1);
            Block *nextBlock = block.getSuccessors()[0];
            int nextBdNum = blockMap[nextBlock];
            // void XAieDma_ShimBdSetNext(XAieDma_Shim *DmaInstPtr, u8
            // BdNum, u8 NextBd);
            /*
            output << "XAieDma_ShimBdSetNext(&" << dmaName << ", "
                   << "  " << bdNum << ", "
                   << "  " << nextBdNum << ");\n";
                   */
          }
          /*
          output << "XAieDma_ShimBdWrite(&" << dmaName << ", "
                 << "  " << bdNum << ");\n";
                 */
        }
      }

      for (auto &block : op.body()) {
        for (auto op : block.getOps<DMAStartOp>()) {
          int bdNum = blockMap[op.dest()];

          /*
          output << "XAieDma_ShimSetStartBd(&" << dmaName << ", "
                 << "XAIEDMA_SHIM_CHNUM_" << stringifyDMAChan(op.dmaChan())
                 << ", "
                 << "  " << bdNum << ");\n";
          // #define XAieDma_ShimChControl(DmaInstPtr, ChNum, PauseStrm,
          // PauseMm, Enable)
          output << "XieDma_ShimChControl(&" << dmaName << ", "
                 << "XAIEDMA_TILE_CHNUM_" << stringifyDMAChan(op.dmaChan())
                 << ",  " << disable << ",  " << disable << ", " << enable
                 << ");\n";
                 */
        }
      }
    }
  }
}

mlir::LogicalResult AIETranslateToAirbin(mlir::ModuleOp module,
                                         llvm::raw_ostream &output) {

  assert(not output.is_displayed());

  DenseMap<std::pair<int, int>, Operation *> tiles;
  DenseMap<Operation *, CoreOp> cores;
  DenseMap<Operation *, MemOp> mems;
  DenseMap<std::pair<Operation *, int>, LockOp> locks;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;
  DenseMap<Operation *, SwitchboxOp> switchboxes;

  NetlistAnalysis NL(module, tiles, cores, mems, locks, buffers, switchboxes);
  NL.collectTiles(tiles);
  NL.collectBuffers(buffers);

  configure_cores(module);
  start_cores(module);
  configure_dmas(module, NL);

  assert(false);

  /*

    output << "void mlir_aie_initialize_locks(" << ctx_p << ") {\n";
    // Lock configuration
    // u8 XAieTile_LockAcquire(XAieGbl_Tile *TileInstPtr, u8 LockId, u8
    LockVal,
    // u32 TimeOut); u8 XAieTile_LockRelease(XAieGbl_Tile *TileInstPtr, u8
    LockId,
    // u8 LockVal, u32 TimeOut);
    for (auto op : module.getOps<UseLockOp>()) {
      int lockVal = op.getLockValue();
      int timeOut = op.getTimeout();
      LockOp lock = dyn_cast<LockOp>(op.lock().getDefiningOp());
      TileOp tile = dyn_cast<TileOp>(lock.tile().getDefiningOp());
      int col = tile.colIndex();
      int row = tile.rowIndex();
      int lockID = lock.getLockID();
      if (op.acquire()) {
        output << "XAieTile_LockAcquire(" << tileDMAInstStr(col, row) << ", "
               << lockID << ", " << lockVal << ", " << timeOut << ");\n";
      } else if (op.release()) {
        output << "XAieTile_LockRelease(" << tileDMAInstStr(col, row) << ", "
               << lockID << ", " << lockVal << ", " << timeOut << ");\n";
      }
    }
    output << "} // mlir_aie_initialize_locks\n";

    output << "void mlir_aie_configure_switchboxes(" << ctx_p << ") {\n";
    output << "  int x, y;\n";

    // StreamSwitch (switchbox) configuration
    // void XAieTile_StrmConnectCct(XAieGbl_Tile *TileInstPtr, u8 Slave, u8
    // Master, u8 SlvEnable); void XAieTile_StrmConfigMstr(XAieGbl_Tile
    // *TileInstPtr, u8 Master, u8 Enable, u8 PktEnable, u8 Config); void
    // XAieTile_StrmConfigSlv(XAieGbl_Tile *TileInstPtr, u8 Slave, u8 Enable, u8
    // PktEnable); void XAieTile_StrmConfigSlvSlot(XAieGbl_Tile *TileInstPtr, u8
    // Slave, u8 Slot, u8 Enable, u32 RegVal); void
    // XAieTile_ShimStrmMuxConfig(XAieGbl_Tile *TileInstPtr, u32 Port, u32
    Input);
    // void XAieTile_ShimStrmDemuxConfig(XAieGbl_Tile *TileInstPtr, u32 Port,
    u32
    // Output); void XAieTile_StrmEventPortSelect(XAieGbl_Tile *TileInstPtr, u8
    // Port, u8 Master, u8 Id);

    // XAieTile_StrmConnectCct(&(TileInst[col+i][row]),
    // XAIETILE_STRSW_SPORT_TRACE((&(TileInst[col+i][row])),
    //                         1),
    // XAIETILE_STRSW_MPORT_NORTH((&(TileInst[col+i][row])),
    //                         0), XAIE_ENABLE);
    for (auto switchboxOp : module.getOps<SwitchboxOp>()) {
      Region &r = switchboxOp.connections();
      Block &b = r.front();
      bool isEmpty = b.getOps<ConnectOp>().empty() &&
                     b.getOps<MasterSetOp>().empty() &&
                     b.getOps<PacketRulesOp>().empty();
      bool isParam = false;

      if (isa<TileOp>(switchboxOp.tile().getDefiningOp())) {
        int col = switchboxOp.colIndex();
        int row = switchboxOp.rowIndex();
        if (!isEmpty) {
          output << "// Core Stream Switch column " << col << " row " << row
                 << "\n";
          output << "x = " << col << ";\n";
          output << "y = " << row << ";\n";
        }
      } else if (AIE::SelectOp sel = dyn_cast<AIE::SelectOp>(
                     switchboxOp.tile().getDefiningOp())) {
        // parameterize streamswitch's configuration
        isParam = true;
        HerdOp sourceHerd = dyn_cast<HerdOp>(sel.startHerd().getDefiningOp());
        std::string sourceHerdName(sourceHerd.name().getValue());

        IterOp iterX = dyn_cast<IterOp>(sel.iterX().getDefiningOp());
        IterOp iterY = dyn_cast<IterOp>(sel.iterY().getDefiningOp());
        int startXValue = iterX.getStartValue();
        int endXValue = iterX.getEndValue();
        int strideXValue = iterX.getStrideValue();
        int startYValue = iterY.getStartValue();
        int endYValue = iterY.getEndValue();
        int strideYValue = iterY.getStrideValue();

        std::string startX(sourceHerdName + "_X + " +
                           std::to_string(startXValue));
        std::string endX(sourceHerdName + "_X + " + std::to_string(endXValue));
        std::string startY(sourceHerdName + "_Y + " +
                           std::to_string(startYValue));
        std::string endY(sourceHerdName + "_Y + " + std::to_string(endYValue));

        output << "for (x = " << startX << "; x < " << endX
               << "; x += " << strideXValue << ") {\n";
        output << "for (y = " << startY << "; y < " << endY
               << "; y += " << strideYValue << ") {\n";
      }

      for (auto connectOp : b.getOps<ConnectOp>()) {
        output << "XAieTile_StrmConnectCct(" << tileInstStr("x", "y") << ",\n";
        output << "\tXAIETILE_STRSW_SPORT_"
               << stringifyWireBundle(connectOp.sourceBundle()).upper() << "("
               << tileInstStr("x", "y") << ", " << connectOp.sourceIndex()
               << "),\n";
        output << "\tXAIETILE_STRSW_MPORT_"
               << stringifyWireBundle(connectOp.destBundle()).upper() << "("
               << tileInstStr("x", "y") << ", " << connectOp.destIndex()
               << "),\n";
        output << "\t" << enable << ");\n";
      }

      for (auto connectOp : b.getOps<MasterSetOp>()) {
        int mask = 0;
        int arbiter = -1;
        for (auto val : connectOp.amsels()) {
          AMSelOp amsel = dyn_cast<AMSelOp>(val.getDefiningOp());
          arbiter = amsel.arbiterIndex();
          int msel = amsel.getMselValue();
          mask |= (1 << msel);
        }

        output << "XAieTile_StrmConfigMstr(" << tileInstStr("x", "y") << ",\n";
        output << "\tXAIETILE_STRSW_MPORT_"
               << stringifyWireBundle(connectOp.destBundle()).upper() << "("
               << tileInstStr("x", "y") << ", " << connectOp.destIndex()
               << "),\n";
        output << "\t" << enable << ",\n"; // port enable
        output << "\t" << enable << ",\n"; // packet enable
        output << "\tXAIETILE_STRSW_MPORT_CFGPKT(" << tileInstStr("x", "y")
               << ",\n";
        output << "\t\tXAIETILE_STRSW_MPORT_"
               << stringifyWireBundle(connectOp.destBundle()).upper() << "("
               << tileInstStr("x", "y") << ", " << connectOp.destIndex()
               << "),\n";
        bool isdma = (connectOp.destBundle() == WireBundle::DMA);
        output << "\t\t" << (isdma ? enable : disable) << " ,\n";
        output << "\t\t"
               << "0x" << llvm::utohexstr(mask)
               << " ,\n"; // FIXME: compute mask for msel
        output << "\t\t" << arbiter << " ));\n";
      }

      for (auto connectOp : b.getOps<PacketRulesOp>()) {
        int slot = 0;
        Block &block = connectOp.rules().front();
        for (auto slotOp : block.getOps<PacketRuleOp>()) {
          AMSelOp amselOp = dyn_cast<AMSelOp>(slotOp.amsel().getDefiningOp());
          int arbiter = amselOp.arbiterIndex();
          int msel = amselOp.getMselValue();
          output << "XAieTile_StrmConfigSlv(" << tileInstStr("x", "y") << ",\n";
          output << "\tXAIETILE_STRSW_SPORT_"
                 << stringifyWireBundle(connectOp.sourceBundle()).upper() << "("
                 << tileInstStr("x", "y") << ", " << connectOp.sourceIndex()
                 << "),\n";
          output << "\t" << enable << ", " << enable << ");\n";
          output << "XAieTile_StrmConfigSlvSlot(" << tileInstStr("x", "y")
                 << ",\n";
          output << "\tXAIETILE_STRSW_SPORT_"
                 << stringifyWireBundle(connectOp.sourceBundle()).upper() << "("
                 << tileInstStr("x", "y") << ", " << connectOp.sourceIndex()
                 << "),\n";
          output << "\t" << slot << " ,\n";
          output << "\t" << enable << ",\n";
          output << "\tXAIETILE_STRSW_SLVSLOT_CFG(" << tileInstStr("x", "y")
                 << ",\n";
          output << "\t\t(XAIETILE_STRSW_SPORT_"
                 << stringifyWireBundle(connectOp.sourceBundle()).upper() << "("
                 << tileInstStr("x", "y") << ", " << connectOp.sourceIndex()
                 << ")),\n";
          output << "\t\t" << slot << " ,\n";
          output << "\t\t"
                 << "0x" << llvm::utohexstr(slotOp.valueInt())
                 << " ,\n";
          output << "\t\t"
                 << "0x" << llvm::utohexstr(slotOp.maskInt()) << " ,\n";
          output << "\t\t" << enable << ",\n";
          output << "\t\t" << msel << " ,\n";
          output << "\t\t" << arbiter << " ));\n";
          slot++;
        }
      }

      if (isParam) {
        output << "}\n";
        output << "}\n";
      }
    }
    for (auto op : module.getOps<ShimMuxOp>()) {
      Region &r = op.connections();
      Block &b = r.front();
      bool isEmpty = b.getOps<ConnectOp>().empty();

      if (isa<TileOp>(op.tile().getDefiningOp())) {
        int col = op.colIndex();
        int row = op.rowIndex();
        if (!isEmpty) {
          output << "// ShimMux column " << col << " row " << row << "\n";
          output << "// NOTE ShimMux always connects from the south as "
                 << "directions are defined relative to the tile stream "
                 << "switch\n";
          output << "x = " << col << ";\n";
          output << "y = " << row << ";\n";
        }
      }

      // XAieTile_ShimStrmMuxConfig(&(TileInst[col][0]),
      // XAIETILE_SHIM_STRM_MUX_SOUTH3, XAIETILE_SHIM_STRM_MUX_DMA);
      // XAieTile_ShimStrmDemuxConfig(&(TileInst[col][0]),
      // XAIETILE_SHIM_STRM_DEM_SOUTH3, XAIETILE_SHIM_STRM_DEM_DMA);
      for (auto connectOp : b.getOps<ConnectOp>()) {
        if (connectOp.sourceBundle() == WireBundle::North) {
          // demux!
          output << "XAieTile_ShimStrmDemuxConfig(" << tileInstStr("x", "y")
                 << ",\n";
          output << "\tXAIETILE_SHIM_STRM_DEM_SOUTH"
                 << // NOTE hardcoded to SOUTH to match definitions
                    //      from libxaie
              connectOp.sourceIndex() << ",\n";
          if (connectOp.destBundle() == WireBundle::PLIO) {
            output << "\tXAIETILE_SHIM_STRM_DEM_PL);\n";
          } else {
            output << "\tXAIETILE_SHIM_STRM_DEM_"
                   << stringifyWireBundle(connectOp.destBundle()).upper()
                   << ");\n";
          }

        } else if (connectOp.destBundle() == WireBundle::North) {
          // mux
          output << "XAieTile_ShimStrmMuxConfig(" << tileInstStr("x", "y")
                 << ",\n";
          output << "\tXAIETILE_SHIM_STRM_MUX_SOUTH"
                 << // NOTE hardcoded to SOUTH to match definitions
                    //      from libxaie
              connectOp.destIndex() << ",\n";
          if (connectOp.sourceBundle() == WireBundle::PLIO) {
            output << "\tXAIETILE_SHIM_STRM_MUX_PL);\n";
          } else {
            output << "\tXAIETILE_SHIM_STRM_MUX_"
                   << stringifyWireBundle(connectOp.sourceBundle()).upper()
                   << ");\n";
          }
        }
      }
    }
    for (auto switchboxOp : module.getOps<ShimSwitchboxOp>()) {
      Region &r = switchboxOp.connections();
      Block &b = r.front();
      bool isEmpty = b.getOps<ConnectOp>().empty();
      int col = switchboxOp.col();
      if (!isEmpty) {
        output << "// Shim Switch column " << col << "\n";
      }
      for (auto connectOp : b.getOps<ConnectOp>()) {
        output << "XAieTile_StrmConnectCct(" << tileInstStr(col, 0) << ",\n";
        output << "\tXAIETILE_STRSW_SPORT_"
               << stringifyWireBundle(connectOp.sourceBundle()).upper() << "("
               << tileInstStr(col, 0) << ", " << connectOp.sourceIndex()
               << "),\n";
        output << "\tXAIETILE_STRSW_MPORT_"
               << stringifyWireBundle(connectOp.destBundle()).upper() << "("
               << tileInstStr(col, 0) << ", " << connectOp.destIndex() <<
    "),\n"; output << "\t" << enable << ");\n";
      }
    }

    output << "} // mlir_aie_configure_switchboxes\n\n";

    // Output Buffer Accessors
    for (auto tile : tiles) {
      Operation *tileOp = tile.second;
      std::pair<int, int> coord = NL.getCoord(tileOp);
      int col = coord.first;
      int row = coord.second;
      auto tileInst = tileInstStr(col, row);

      auto bufferAccessor = [&](Optional<TileID> tile, BufferOp buf) {
        // int32_t mlir_aie_read_buffer_a13(int index) {
        //     return XAieTile_DmReadWord(&(TileInst[1][3]), a13_offset +
        //     (index*4));
        // }
        // void mlir_aie_write_buffer_a13(int index, int32_t value) {
        //     XAieTile_DmWriteWord(&(TileInst[1][3]), a13_offset + (index*4),
        //     value);
        // }
        std::string bufName(buf.name().getValue());
        Type t = buf.getType();
        Type et;
        std::string typestr;
        if (auto memrefType = t.dyn_cast<MemRefType>()) {
          et = memrefType.getElementType();
          if (et.isInteger(32))
            typestr = "int32_t";
          else if (et.isF32())
            typestr = "float";
          else {
            output << "// buffer " << bufName << " with unsupported type " << t
                   << ";\n";
            return; // Unsupported type
          }

        } else {
          output << "// buffer " << bufName << " with unsupported type " << t
                 << ";\n";
          return; // Unsupported type
        }

        output << "const int " << bufName
               << "_offset = " << NL.getBufferBaseAddress(buf) << ";\n";
        output << typestr << " mlir_aie_read_buffer_" << bufName << "(" << ctx_p
               << ", int index) {\n";
        output << "  int32_t value = XAieTile_DmReadWord(" << tileInst << ", "
               << bufName << "_offset + (index*4));\n";
        if (et.isInteger(32))
          output << "  return value;\n";
        else if (et.isF32()) {
          output << "  union caster { int32_t i; float f; };\n";
          output << "  caster c; c.i = value;\n";
          output << "  return c.f;\n";
        }
        output << "}\n";
        output << "void mlir_aie_write_buffer_" << bufName << "(" << ctx_p
               << ", int index, " << typestr << " value) {\n";
        if (et.isInteger(32))
          output << "  int32_t int_value = value;\n";
        else if (et.isF32()) {
          output << "  union caster { int32_t i; float f; };\n";
          output << "  caster c; c.f = value;\n";
          output << "  int32_t int_value = c.i;\n";
        }
        output << "  return XAieTile_DmWriteWord(" << tileInst << ", " <<
    bufName
               << "_offset + (index*4), int_value);\n";
        output << "}\n";
      };

      // if(tiles.count(tile.getValue()))
      for (auto buf : buffers[tileOp])
        bufferAccessor(coord, buf);
      // };
    }
    */
  return success();
}
} // namespace AIE
} // namespace xilinx
