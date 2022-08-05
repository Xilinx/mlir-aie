//===- AIETargetAirbin.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/None.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm> // find_if
#include <array>
#include <cstdint>
#include <elf.h>
#include <fcntl.h> // open
#include <sstream>
#include <type_traits> // enable_if_t
#include <unistd.h>    // read
#include <utility>     // pair
#include <vector>

#include "aie/AIENetlistAnalysis.h"

#include "AIETargets.h"

// This target is a "flattening" of AIETargetXAIEV1 thru libXAIE.
// All recorded writes are time/order invariant.
// This allows sorting to compact the airbin,
// but translating from XAIE is more difficult,
// as some writes are handled by the runtime that loads our resulting airbin.

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

  // SFINAE is used here to choose the copy constructor for `TileAddress`,
  // and this constructor for all other classes.
  template <
      typename Op,
      std::enable_if_t<not std::is_same<Op, TileAddress>::value, bool> = true>
  TileAddress(Op &op)
      : TileAddress{static_cast<uint8_t>(op.colIndex()),
                    static_cast<uint8_t>(op.rowIndex())} {}

  uint64_t fullAddress(uint32_t register_offset) const {
    return (array_offset << TILE_ADDR_ARR_SHIFT) |
           (static_cast<uint64_t>(column) << TILE_ADDR_COL_SHIFT) |
           (static_cast<uint64_t>(row) << TILE_ADDR_ROW_SHIFT) |
           register_offset;
  }

  bool isShim() const { return row == 0; }

  operator uint16_t() const {
    return (static_cast<uint16_t>(column) << TILE_ADDR_ROW_WIDTH) | row;
  }

  uint8_t col() const { return column; }

private:
  uint64_t array_offset : 34;
  uint8_t column : TILE_ADDR_COL_WIDTH;
  uint8_t row : TILE_ADDR_ROW_WIDTH;
};

static_assert(sizeof(TileAddress) <= sizeof(uint64_t),
              "Tile addresses are at most 64-bits");

class Address {
public:
  Address(TileAddress tile, uint32_t register_offset)
      : tile{tile}, register_offset{register_offset} {}

  operator uint64_t() const { return tile.fullAddress(register_offset); }

  TileAddress destTile() const { return tile; }
  uint32_t offset() const { return register_offset; }

private:
  TileAddress tile;
  uint32_t register_offset : TILE_ADDR_ROW_SHIFT;
};

class Write {
public:
  Write(Address addr, uint32_t data) : address{addr}, data{data} {
    assert(address % 4 == 0);
  }

  bool isJustBefore(const Write &rhs) const {
    return static_cast<uint64_t>(this->address) + sizeof(data) ==
           static_cast<uint64_t>(rhs.address);
  }

  uint64_t destination() const { return address; }
  uint32_t relativeDest() const { return address.offset(); }

  uint16_t tile() const { return address.destTile(); }
  uint32_t value() const { return data; }

  bool operator<(const Write &rhs) const {
    return static_cast<uint64_t>(address) < static_cast<uint64_t>(rhs.address);
  }

private:
  Address address;
  uint32_t data;
};

static std::vector<Write> writes;

static void write32(Address addr, uint32_t value) {
  assert(addr.destTile().col() > 0);

  auto iter =
      std::find_if(writes.begin(), writes.end(), [addr](const auto &x) -> bool {
        return addr == x.destination();
      });

  if (iter != writes.end()) {
    // Overwrite
    writes.erase(iter);
  }

  writes.emplace_back(addr, value);
}

static uint32_t read32(Address addr) {
  auto iter =
      std::find_if(writes.begin(), writes.end(), [addr](const auto &x) -> bool {
        return x.destination() == addr;
      });

  return (iter != writes.end()) ? iter->value() : 0;
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

static constexpr auto BD_BASE = 0x1D000u;
static constexpr auto BD_END = 0x1D1F8u;
static constexpr auto SHIM_BD_END = 0x1D15Cu;

// The SHIM row is always 0.
// SHIM resets are handled by the runtime.
static void generateShimConfig(TileOp &tileOp) {

  TileAddress tileAddress{tileOp};

  if (tileOp.isShimNOCTile()) {
    clearRange(tileAddress, BD_BASE, SHIM_BD_END);
  }
  if (tileOp.isShimNOCTile() or tileOp.isShimTile()) {
    // output << "// Stream Switch master config\n";
    clearRange(tileAddress, 0x3F000, 0x3F058);
    // output << "// Stream Switch slave config\n";
    clearRange(tileAddress, 0x3F100, 0x3F15C);
    // output << "// Stream Switch slave slot config\n";
    clearRange(tileAddress, 0x3F200, 0x3F368);
  }
}

static constexpr uint64_t setField(uint64_t value, uint8_t shift,
                                   uint64_t mask) {
  return (value << shift) & mask;
}

// LLVM may have a way to read ELF files,
// but for our purposes, manually parsing is fine for now.
static bool loadElf(TileAddress tile, const std::string &filename) {

  // Clear program memory
  clearRange(tile, 0, 0x7ffcu);

  llvm::dbgs() << "Reading ELF file " << filename << '\n';

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
    // TODO: What if the first header is not the one to load?
    // TODO: What if there are multiple sections to load?
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
               << ((tile.fullAddress(0) >> TILE_ADDR_ROW_SHIFT) & 0xFFFu)
               << '\n';

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
    if (tileOp.isShimTile()) {
      generateShimConfig(tileOp);
    } else {
      TileAddress tileAddress{tileOp};

      // Reset configuration
      // Program Memory
      clearRange(tileAddress, 0x20000, 0x23FFC);
      // TileDMA
      clearRange(tileAddress, BD_BASE, BD_END);
      clearRange(tileAddress, 0x1de00, 0x1de1C);
      // Stream Switch master config
      clearRange(tileAddress, 0x3F000, 0x3F060);
      // Stream Switch slave config
      clearRange(tileAddress, 0x3F100, 0x3F168);
      // Stream Switch slave slot config
      clearRange(tileAddress, 0x3F200, 0x3F38C);

      // NOTE: Here is usually where locking is done.
      // However, the runtime will handle that when loading the airbin.

      if (auto coreOp = tileOp.getCoreOp()) {
        std::string fileName;
        if (auto fileAttr = coreOp->getAttrOfType<StringAttr>("elf_file")) {
          fileName = std::string(fileAttr.getValue());
        } else {
          std::stringstream ss;
          ss << "core_" << tileOp.colIndex() << '_' << tileOp.rowIndex()
             << ".elf";
          fileName = ss.str();
        }
        if (not loadElf(tileAddress, fileName)) {
          llvm::outs() << "Error loading " << fileName;
        }
      }
    }
  }
}

// Start execution of all the cores.
// This is by the runtime. We do not need to record it.
/*
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
*/

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

static BDInfo getBDInfo(Block &block, const NetlistAnalysis &NL) {
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

  for (auto memOp : module.getOps<MemOp>()) {
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

    TileAddress tile{memOp};
    // Clear the CTRL and QUEUE registers for the DMA channels.
    for (auto chNum = 0u; chNum < MAX_CHANNEL_COUNT; ++chNum) {
      write32({tile, dmaChannelCtrlOffsets[chNum]},
              setField(disable, dmaChannelResetLSB, dmaChannelResetMask) |
                  setField(disable, dmaChannelEnableLSB, dmaChannelEnableMask));
      write32({tile, dmaChannelQueueOffsets[chNum]}, 0);
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

      struct BdData {
        uint32_t addr_a{0};
        uint32_t addr_b{0};
        // The X register has some fields
        // which need to be nonzero in the default state.
        uint32_t x{0x00ff0001u};
        // The Y register has some fields
        // which need to be nonzero in the default state.
        uint32_t y{0xffff0100u};
        uint32_t packet{0};
        uint32_t interleave{0};
        uint32_t control{0};
      };

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
      Optional<int> lockID = llvm::NoneType::None;

      for (auto op : block.getOps<UseLockOp>()) {
        LockOp lock = dyn_cast<LockOp>(op.lock().getDefiningOp());
        lockID = lock.getLockID();
        if (op.acquire()) {
          acqEnable = enable;
          acqValue = op.getLockValue();
        } else if (op.release()) {
          relEnable = enable;
          relValue = op.getLockValue();
        } else
          assert(false);
      }

      // We either
      //  a. went thru the loop once (`lockID` should be something) xor
      //  b. did not enter the loop (the enables should be both disable)
      assert(lockID.hasValue() xor
             (acqEnable == disable and relEnable == disable));

      for (auto op : block.getOps<DMABDPACKETOp>()) {
        bdInfo.foundBdPacket = true;
        bdInfo.packetType = op.getPacketType();
        bdInfo.packetID = op.getPacketID();
      }

      auto bdNum = blockMap[&block];
      BdData bdData;
      if (bdInfo.foundBd) {
        static constexpr auto BD_ADDR_LOCKID_SHIFT = 22u;
        static constexpr auto BD_ADDR_LOCKID_MASK = 0xFu
                                                    << BD_ADDR_LOCKID_SHIFT;
        static constexpr auto BD_ADDR_RELEN_SHIFT = 21u;
        static constexpr auto BD_ADDR_RELEN_MASK = 1u << BD_ADDR_RELEN_SHIFT;

        static constexpr auto BD_ADDR_RELVAL_SHIFT = 20u;
        static constexpr auto BD_ADDR_RELVAL_MASK = 1u << BD_ADDR_RELVAL_SHIFT;

        static constexpr auto BD_ADDR_RELVALEN_SHIFT = 19u;
        static constexpr auto BD_ADDR_RELVALEN_MASK = 1u
                                                      << BD_ADDR_RELVALEN_SHIFT;

        static constexpr auto BD_ADDR_ACQEN_SHIFT = 18u;
        static constexpr auto BD_ADDR_ACQEN_MASK = 1u << BD_ADDR_ACQEN_SHIFT;

        static constexpr auto BD_ADDR_ACQVAL_SHIFT = 17u;
        static constexpr auto BD_ADDR_ACQVAL_MASK = 1u << BD_ADDR_ACQVAL_SHIFT;

        static constexpr auto BD_ADDR_ACQVALEN_SHIFT = 16u;
        static constexpr auto BD_ADDR_ACQVALEN_MASK = 1u
                                                      << BD_ADDR_ACQVALEN_SHIFT;

        if (bdInfo.hasA) {
          /* clang-format off
          output << "XAieDma_TileBdSetLock(" << tileDMAInstStr(col, row) << ", "
                 << bdNum << ", " << bufA << ", " << lockID << ", " << relEnable
                 << ", " << relValue << ", " << acqEnable << ", " << acqValue
                 << ");\n";
void XAieDma_TileBdSetLock(XAieDma_Tile *DmaInstPtr, u8 BdNum, u8 AbType, u8 LockId, u8 LockRelEn, u8 LockRelVal, u8 LockAcqEn, u8 LockAcqVal)
   with AbType = bufA
        LockId = lockID
        LockRelEn = relEnable
        LockRelVal = relValue
   clang-format on */
          bdData.addr_a =
              setField(lockID.getValue(), BD_ADDR_LOCKID_SHIFT,
                       BD_ADDR_LOCKID_MASK) |
              setField(relEnable, BD_ADDR_RELEN_SHIFT, BD_ADDR_RELEN_MASK) |
              setField(acqEnable, BD_ADDR_ACQEN_SHIFT, BD_ADDR_ACQEN_MASK);

          if (relValue != 0xFFu) {
            bdData.addr_a |=
                setField(1, BD_ADDR_RELVALEN_SHIFT, BD_ADDR_RELVALEN_MASK) |
                setField(relValue, BD_ADDR_RELVAL_SHIFT, BD_ADDR_RELVAL_MASK);
          }
          if (acqValue != 0xFFu) {
            bdData.addr_a |=
                setField(1, BD_ADDR_ACQVALEN_SHIFT, BD_ADDR_ACQVALEN_MASK) |
                setField(acqValue, BD_ADDR_ACQVAL_SHIFT, BD_ADDR_ACQVAL_MASK);
          }
        }
        if (bdInfo.hasB) {
          assert(false);
          /*
          output << "XAieDma_TileBdSetLock(" << tileDMAInstStr(col, row) << ", "
                 << bdNum << ", " << bufB << ", " << lockID << ", " << relEnable
                 << ", " << relValue << ", " << acqEnable << ", " << acqValue
                 << ");\n";
                 */
        }

        /* clang-format off
        output << "XAieDma_TileBdSetAdrLenMod(" << tileDMAInstStr(col, row)
               << ", " << bdNum << ", "
               << "0x" << llvm::utohexstr(BaseAddrA + offsetA) << ", "
               << "0x" << llvm::utohexstr(BaseAddrB + offsetB) << ", " << lenA
               << " * " << bytesA << ", " << AbMode << ", " << FifoMode
               << ");\n";
void XAieDma_TileBdSetAdrLenMod(XAieDma_Tile *DmaInstPtr, u8 BdNum, u16 BaseAddrA, u16 BaseAddrB, u16 Length, u8 AbMode, u8 FifoMode)
with BaseAddrA = BaseAddr + offsetA
     Length = lenA

	Length = Length >> XAIEDMA_TILE_LENGTH32_OFFSET;
	if(FifoMode != 0U) {
		LenMask = XAIEDMA_TILE_LENGTH128_MASK;
	}

	DescrPtr->AddrA.BaseAddr = BaseAddrA >>XAIEDMA_TILE_ADDRAB_ALIGN_OFFSET;
	DescrPtr->AddrB.BaseAddr = BaseAddrB >>XAIEDMA_TILE_ADDRAB_ALIGN_OFFSET;
	DescrPtr->Length = Length - 1U;
	DescrPtr->AbMode = AbMode;
	DescrPtr->FifoMode = FifoMode;
        clang-format on */

        auto addr_a = bdInfo.BaseAddrA + bdInfo.offsetA;
        auto addr_b = bdInfo.BaseAddrB + bdInfo.offsetB;

        static constexpr auto BD_ADDR_BASE_MASK = 0x1FFFu;
        static constexpr auto BD_CTRL_LEN_MASK = 0x1FFFu;

        static constexpr auto BD_CTRL_ABMODE_SHIFT = 30u;
        static constexpr auto BD_CTRL_ABMODE_MASK = 1u << BD_CTRL_ABMODE_SHIFT;

        static constexpr auto BD_CTRL_FIFO_SHIFT = 28u;
        static constexpr auto BD_CTRL_FIFO_MASK = 3u << BD_CTRL_FIFO_SHIFT;

        bdData.addr_a |= setField(addr_a >> 2u, 0, BD_ADDR_BASE_MASK);
        bdData.addr_b |= setField(addr_b >> 2u, 0, BD_ADDR_BASE_MASK);
        bdData.control |=
            setField(bdInfo.lenA - 1, 0, BD_CTRL_LEN_MASK) |
            setField(bdInfo.FifoMode, BD_CTRL_FIFO_SHIFT, BD_CTRL_FIFO_MASK) |
            setField(bdInfo.AbMode, BD_CTRL_ABMODE_SHIFT, BD_CTRL_ABMODE_MASK);

        if (block.getNumSuccessors() > 0) {
          // should have only one successor block
          assert(block.getNumSuccessors() == 1);
          auto *nextBlock = block.getSuccessors()[0];
          auto nextBdNum = blockMap[nextBlock];

          /*
          output << "XAieDma_TileBdSetNext(" << tileDMAInstStr(col, row) << ", "
                           << bdNum << ", " << nextBdNum << ");\n";
        DescrPtr->NextBd = NextBd;

        // Use next BD only if the Next BD value is not invalid
        if(NextBd != XAIEDMA_TILE_BD_NEXTBD_INVALID) {
                DescrPtr->NextBdEn = XAIE_ENABLE;
        } else {
                DescrPtr->NextBdEn = XAIE_DISABLE;
        }
          */

          static constexpr auto BD_CTRL_NEXTBD_SHIFT = 13u;
          static constexpr auto BD_CTRL_NEXTBD_MASK = 0xFu
                                                      << BD_CTRL_NEXTBD_SHIFT;

          static constexpr auto BD_CTRL_NEXTBDEN_SHIFT = 17u;
          static constexpr auto BD_CTRL_NEXTBDEN_MASK =
              1u << BD_CTRL_NEXTBDEN_SHIFT;

          bdData.control |=
              setField(nextBdNum, BD_CTRL_NEXTBD_SHIFT, BD_CTRL_NEXTBD_MASK) |
              setField(nextBdNum != 0xFFu, BD_CTRL_NEXTBDEN_SHIFT,
                       BD_CTRL_NEXTBDEN_MASK);
        }

        if (bdInfo.foundBdPacket) {
          assert(false);
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

        static constexpr auto BD_CTRL_VALID_SHIFT = 31u;
        static constexpr auto BD_CTRL_VALID_MASK = 1u << BD_CTRL_VALID_SHIFT;

        auto bdOffset = BD_BASE + bdNum * 0x20u;
        assert(bdOffset <= BD_END);

        write32({tile, bdOffset}, bdData.addr_a);
        write32({tile, bdOffset + 4u}, bdData.addr_b);
        write32({tile, bdOffset + 8u}, bdData.x);
        write32({tile, bdOffset + 0xCu}, bdData.y);
        write32({tile, bdOffset + 0x10u}, bdData.packet);
        write32({tile, bdOffset + 0x14u}, bdData.interleave);
        write32({tile, bdOffset + 0x18u},
                bdData.control |
                    setField(1u, BD_CTRL_VALID_SHIFT, BD_CTRL_VALID_MASK));
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

#define XAieDma_TileSetStartBd(DmaInstPtr, ChNum, BdStart)
                        if(BdStart != 0xFFU) {
                                XAieGbl_Write32((DmaInstPtr->BaseAddress +
                                TileDmaCh[ChNum].StatQOff),
                                (XAie_SetField(BdStart,
                                TileDmaCh[ChNum].StatQ.Lsb,
                                TileDmaCh[ChNum].StatQ.Mask)));
                        }
                         DmaInstPtr->StartBd[ChNum] = BdStart
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

    // auto index = 0;
    for (auto op : module.getOps<ShimDMAOp>()) {
      assert(false);
      // auto col = op.colIndex();
      // auto row = op.rowIndex();
      /* TODO: Implement the following
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
        assert(false);
        bool foundBd = false;
        /*
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
        */

        /*
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
        */

        // int bdNum = blockMap[&block];
        if (foundBd) {
          assert(false);
          // void XAieDma_ShimBdSetLock(XAieDma_Shim *DmaInstPtr, u8 BdNum,
          // u8 LockId, u8 LockRelEn, u8 LockRelVal, u8 LockAcqEn, u8
          // LockAcqVal);
          /* TODO: Implement the following
          if (hasLock) {
            output << "XAieDma_ShimBdSetLock(&" << dmaName << ", " << bdNum
                   << ", " << lockID << ", " << relEnable << ", " << relValue
                   << ", " << acqEnable << ", " << acqValue << ");\n";
          }
          */
          // void XAieDma_ShimBdSetAddr(XAieDma_Shim *DmaInstPtr, u8 BdNum,
          // u16 AddrHigh, u32 AddrLow, u32 Length);
          // uint64_t address = BaseAddr + offset;
          /* TODO: Implement the following
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
            // Block *nextBlock = block.getSuccessors()[0];
            // int nextBdNum = blockMap[nextBlock];
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
          // int bdNum = blockMap[op.dest()];
          assert(false);
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

/*
Initialize locks seems to not ever call `Write32`, making it a no-op.
static void initialize_locks(mlir::ModuleOp module) {

  // Lock configuration
  // u8 XAieTile_LockAcquire(XAieGbl_Tile *TileInstPtr, u8 LockId, u8 LockVal,
  // u32 TimeOut) u8 XAieTile_LockRelease(XAieGbl_Tile *TileInstPtr, u8 LockId,
  // u8 LockVal, u32 TimeOut)
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
}
*/

static uint8_t computeSlavePort(WireBundle bundle, int index, bool isShim) {
  assert(index >= 0);
  assert(index < UINT8_MAX - 17);

  switch (bundle) {
  case WireBundle::DMA:
    return 2u + index;
  case WireBundle::South:
    if (isShim)
      return 3u + index;
    else
      return 7u + index;
  case WireBundle::North:
    if (isShim)
      return 15u + index;
    else
      return 17u + index;
  default:
    // To implement a new WireBundle,
    // look in libXAIE for the macros that handle the port.
    assert(false);
  }
}

static uint8_t computeMasterPort(WireBundle bundle, int index, bool isShim) {
  assert(index >= 0);
  assert(index < UINT8_MAX - 17);

  switch (bundle) {
  case WireBundle::DMA:
    return 2u + index;
  case WireBundle::South:
    if (isShim)
      return 3u + index;
    else
      return 7u + index;
  case WireBundle::North:
    if (isShim)
      return 13u + index;
    else
      return 15u + index;
  default:
    // To implement a new WireBundle,
    // look in libXAIE for the macros that handle the port.
    assert(false);
  }
}

static void configure_switchboxes(mlir::ModuleOp &module) {

  // StreamSwitch (switchbox) configuration
  // void XAieTile_StrmConnectCct(XAieGbl_Tile *TileInstPtr, u8 Slave, u8
  // Master, u8 SlvEnable); void XAieTile_StrmConfigMstr(XAieGbl_Tile
  // *TileInstPtr, u8 Master, u8 Enable, u8 PktEnable, u8 Config); void
  // XAieTile_StrmConfigSlv(XAieGbl_Tile *TileInstPtr, u8 Slave, u8 Enable, u8
  // PktEnable); void XAieTile_StrmConfigSlvSlot(XAieGbl_Tile *TileInstPtr, u8
  // Slave, u8 Slot, u8 Enable, u32 RegVal); void
  // XAieTile_ShimStrmMuxConfig(XAieGbl_Tile *TileInstPtr, u32 Port, u32 Input);
  // void XAieTile_ShimStrmDemuxConfig(XAieGbl_Tile *TileInstPtr, u32 Port, u32
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

    // NOTE: may not be needed
    auto switchbox_set = [&] {
      std::set<std::pair<int, int>> result;
      if (isa<TileOp>(switchboxOp.tile().getDefiningOp())) {
        int col = switchboxOp.colIndex();
        int row = switchboxOp.rowIndex();
        if (!isEmpty) {
          result.emplace(col, row);
        }
      } else if (AIE::SelectOp sel = dyn_cast<AIE::SelectOp>(
                     switchboxOp.tile().getDefiningOp())) {
        // TODO: Use XAIEV1 target and translate into write32s
        assert(false);
      }

      return result;
    }();

    for (auto connectOp : b.getOps<ConnectOp>()) {
      /*
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
      */
      /* clang-format off
      void XAieTile_StrmConnectCct(
          XAieGbl_Tile *TileInstPtr,
          u8 Slave, u8 Master, u8 SlvEnable)
        with TileInstPtr @ (x, y)
             Slave @ XAIETILE_STRSW_SPORT_ << stringifyWireBundle(connectOp.sourceBundle()).upper()  ( tile @ (x, y) , connectOp.sourceIndex() )
             Master @ XAIETILE_STRSW_MPORT_ << stringifyWireBundle(connectOp.destBundle()).upper() ( tile @ (x, y), connectOp.destIndex())
        clang-format on
      */
      for (auto tile_coords : switchbox_set) {
        TileAddress tile{static_cast<uint8_t>(tile_coords.first),
                         static_cast<uint8_t>(tile_coords.second)};

        auto slave_port = computeSlavePort(
            connectOp.sourceBundle(), connectOp.sourceIndex(), tile.isShim());

        auto master_port = computeMasterPort(
            connectOp.destBundle(), connectOp.destIndex(), tile.isShim());

        /* clang-format off
          Enable the master port in circuit switched mode and specify the slave port it is connected to
          XAieTile_StrmConfigMstr(TileInstPtr, Master, XAIE_ENABLE, XAIE_DISABLE, Slave);

          Enable the slave port in circuit switched mode
          if(SlvEnable == XAIE_ENABLE) XAieTile_StrmConfigSlv(TileInstPtr, Slave, SlvEnable, XAIE_DISABLE);
          clang-format on
        */

        // Configure master side
        {
          /* clang-format off
          void XAieTile_StrmConfigMstr(XAieGbl_Tile *TileInstPtr, u8 Master, u8 Enable, u8 PktEnable, u8 Config)
          ----
          RegAddr = TileInstPtr->TileAddr + RegPtr->RegOff;

          DropHdr = XAie_GetField(Config, RegPtr->DrpHdr.Lsb, RegPtr->DrpHdr.Mask)
          RegVal =
              XAie_SetField(Enable, RegPtr->MstrEn.Lsb, RegPtr->MstrEn.Mask) |
              XAie_SetField(PktEnable, RegPtr->PktEn.Lsb, RegPtr->PktEn.Mask) |
              XAie_SetField(DropHdr, RegPtr->DrpHdr.Lsb, RegPtr->DrpHdr.Mask) |
              XAie_SetField(Config, RegPtr->Config.Lsb, RegPtr->Config.Mask);
          clang-format on
        */
          Address address{tile, 0x3F000u + master_port * 4u};

          auto drop_header = (slave_port & 0x80u) >> 7u;

          auto value =
              setField(1, 31, 0x80000000u) | setField(0, 30, 0x40000000u) |
              setField(drop_header, 7, 0x80u) | setField(slave_port, 0, 0x7Fu);
          assert(value < UINT32_MAX);
          write32(address, value);
        }

        // Configure slave side
        {
          /*
void XAieTile_StrmConfigSlv(XAieGbl_Tile *TileInstPtr, u8 Slave, u8 Enable,
                                                                u8 PktEnable)
----
          // Get the address of Slave port config reg
          RegAddr = TileInstPtr->TileAddr + RegPtr->RegOff;

          // Frame the 32-bit reg value
          RegVal = XAie_SetField(Enable, RegPtr->SlvEn.Lsb,
                                          RegPtr->SlvEn.Mask) |
                  XAie_SetField(PktEnable, RegPtr->PktEn.Lsb,
                                          RegPtr->PktEn.Mask);
            */
          Address address{tile, 0x3F100u + slave_port * 4u};

          write32(address,
                  setField(1, 31, 0x80000000u) | setField(0, 30, 0x40000000u));
        }
      }
    }

    for (auto connectOp : b.getOps<MasterSetOp>()) {
      /*
      int mask = 0;
      int arbiter = -1;
      for (auto val : connectOp.amsels()) {
        auto amsel = dyn_cast<AMSelOp>(val.getDefiningOp());
        arbiter = amsel.arbiterIndex();
        int msel = amsel.getMselValue();
        mask |= (1u << msel);
      }
      */
      assert(false);

      /* clang-format off
      TODO: Implement the following
      bool isdma = (connectOp.destBundle() == WireBundle::DMA);
      XAieTile_StrmConfigMstr(tile @ (x, y),
        XAIETILE_STRSW_MPORT_ << stringifyWireBundle(connectOp.destBundle()).upper() ( tile @ (x, y)), connectOp.destIndex() ),
        enable, // port enable output
        enable, // packet enable output
        XAIETILE_STRSW_MPORT_CFGPKT( tile @ (x, y),
        XAIETILE_STRSW_MPORT_ << stringifyWireBundle(connectOp.destBundle()).upper() ( tile @ (x, y), connectOp.destIndex() ),
        (isdma ? enable : disable), mask, arbiter)
      clang-format on
      */
    }

    for (auto connectOp : b.getOps<PacketRulesOp>()) {
      int slot = 0;
      Block &block = connectOp.rules().front();
      for (auto slotOp : block.getOps<PacketRuleOp>()) {
        /*
        AMSelOp amselOp = dyn_cast<AMSelOp>(slotOp.amsel().getDefiningOp());
        int arbiter = amselOp.arbiterIndex();
        int msel = amselOp.getMselValue();
        */
        assert(false);
        /* TODO
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
               << "0x" << llvm::utohexstr(slotOp.valueInt()) << " ,\n";
        output << "\t\t"
               << "0x" << llvm::utohexstr(slotOp.maskInt()) << " ,\n";
        output << "\t\t" << enable << ",\n";
        output << "\t\t" << msel << " ,\n";
        output << "\t\t" << arbiter << " ));\n";
        */
        slot++;
      }
    }
  }

  Optional<TileAddress> currentTile = llvm::NoneType::None;
  for (auto op : module.getOps<ShimMuxOp>()) {
    Region &r = op.connections();
    Block &b = r.front();

    if (isa<TileOp>(op.tile().getDefiningOp())) {
      bool isEmpty = b.getOps<ConnectOp>().empty();
      if (!isEmpty) {
        currentTile = op;
      }
    }

    for (auto connectOp : b.getOps<ConnectOp>()) {

      const auto inputMaskFor = [](WireBundle bundle, uint8_t shiftAmt) {
        switch (bundle) {
        case WireBundle::PLIO:
          return 0u << shiftAmt;
        case WireBundle::DMA:
          return 1u << shiftAmt;
        case WireBundle::NOC:
          return 2u << shiftAmt;
        default:
          assert(false);
        }
      };

      if (connectOp.sourceBundle() == WireBundle::North) {
        // demux!
        // XAieTile_ShimStrmDemuxConfig(&(TileInst[col][0]),
        // XAIETILE_SHIM_STRM_DEM_SOUTH3, XAIETILE_SHIM_STRM_DEM_DMA);
        assert(currentTile.hasValue());

        auto shiftAmt = [index = connectOp.sourceIndex()] {
          // NOTE: hardcoded to SOUTH to match definitions from libxaie
          switch (index) {
          case 2:
            return 4u;
          case 3:
            return 6u;
          case 6:
            return 8u;
          case 7:
            return 10u;
          default:
            assert(false);
          }
        }();

        // We need to add to the possibly preexisting mask.
        Address addr{currentTile.value(), 0x1F004u};
        auto currentMask = read32(addr);

        write32(addr,
                currentMask | inputMaskFor(connectOp.destBundle(), shiftAmt));

      } else if (connectOp.destBundle() == WireBundle::North) {
        // mux
        // XAieTile_ShimStrmMuxConfig(&(TileInst[col][0]),
        // XAIETILE_SHIM_STRM_MUX_SOUTH3, XAIETILE_SHIM_STRM_MUX_DMA);
        assert(currentTile.hasValue());

        auto shiftAmt = [index = connectOp.destIndex()] {
          // NOTE: hardcoded to SOUTH to match definitions from libxaie
          switch (index) {
          case 2:
            return 8u;
          case 3:
            return 10u;
          case 6:
            return 12u;
          case 7:
            return 14u;
          default:
            assert(false);
          }
        }();

        Address addr{currentTile.value(), 0x1F000u};
        auto currentMask = read32(addr);

        write32(addr,
                currentMask | inputMaskFor(connectOp.sourceBundle(), shiftAmt));
      }
    }
  }

  for (auto switchboxOp : module.getOps<ShimSwitchboxOp>()) {
    Region &r = switchboxOp.connections();
    Block &b = r.front();
    /*
    bool isEmpty = b.getOps<ConnectOp>().empty();
    int col = switchboxOp.col();
    */
    for (auto connectOp : b.getOps<ConnectOp>()) {
      assert(false);
      /* TODO: Implement the following
      output << "XAieTile_StrmConnectCct(" << tileInstStr(col, 0) << ",\n";
      output << "\tXAIETILE_STRSW_SPORT_"
             << stringifyWireBundle(connectOp.sourceBundle()).upper() << "("
             << tileInstStr(col, 0) << ", " << connectOp.sourceIndex()
             << "),\n";
      output << "\tXAIETILE_STRSW_MPORT_"
             << stringifyWireBundle(connectOp.destBundle()).upper() << "("
             << tileInstStr(col, 0) << ", " << connectOp.destIndex() << "),\n";
      output << "\t" << enable << ");\n";
      */
    }
  }
}

static std::vector<std::vector<Write>> group_sections() {
  std::vector<std::vector<Write>> sections{{}};

  for (auto write : writes) {
    if (sections.back().empty()) {
      sections.back().emplace_back(write);
      continue;
    }

    auto &last_write = sections.back().back();
    if (last_write.isJustBefore(write)) {
      sections.back().emplace_back(write);
    } else {
      sections.emplace_back(std::vector<decltype(write)>{write});
    }
  }

  return sections;
}

struct [[gnu::packed]] SectionHeader {
  uint8_t padding;
  uint16_t tile;
  uint8_t name;
  uint32_t type;
  uint64_t address;
  uint64_t offset;
  uint64_t size;
};
static_assert(
    sizeof(SectionHeader) == sizeof(uint64_t) * 4u,
    "SectionHeaders should have the same in-memory and in-file sizes");

// TODO: Use `sizeof(SectionHeader)` when in binary mode
static constexpr auto SECTION_SIZE = 9u * 8u;

static constexpr auto HEADER_SIZE = 16u + 16u;

struct [[gnu::packed]] FileHeader {
  std::array<uint8_t, 16> ident{'~' + 1, 'A', 'I', 'R', 2, 2, 1};
  uint16_t type{1};
  uint16_t machine{1};
  uint16_t version{1};
  uint16_t chnum;
  uint64_t choff = HEADER_SIZE;
};

static_assert(sizeof(FileHeader) == sizeof(uint64_t) * 2 + 16,
              "FileHeader should have the same size in-file and in-memory");

static std::vector<SectionHeader>
make_section_headers(const std::vector<std::vector<Write>> &group_writes) {
  std::vector<SectionHeader> headers;

  uint64_t seen_size = 0;
  uint8_t leftMostColumn = UINT8_MAX;

  for (const auto &section : group_writes) {
    assert(not section.empty());
    SectionHeader header;
    header.type = 1;
    header.address = section.front().relativeDest();
    // TODO: The size is for binary mode,
    // but the offset must account for ascii mode and newlines
    header.offset = seen_size;
    header.size = section.size() * sizeof(uint32_t);
    seen_size += section.size() * 2 * sizeof(uint32_t) + section.size();

    header.tile = section.front().tile();
    leftMostColumn =
        std::min(leftMostColumn,
                 static_cast<uint8_t>(header.tile >> TILE_ADDR_ROW_WIDTH));

    headers.emplace_back(std::move(header));
  }

  assert(leftMostColumn < UINT8_MAX);

  llvm::dbgs() << "Leftmost column: " << int{leftMostColumn} << '\n';

  for (auto &header : headers) {
    header.offset += HEADER_SIZE + SECTION_SIZE * headers.size();
    header.tile -= leftMostColumn << TILE_ADDR_ROW_WIDTH;
  }

  return headers;
}

static void output_sections(llvm::raw_ostream &output,
                            const std::vector<SectionHeader> &headers,
                            const std::vector<std::vector<Write>> &writes) {

  FileHeader fileHeader;
  fileHeader.chnum = headers.size();

  llvm::dbgs() << "Found " << fileHeader.chnum << " sections\n";

  output.write(reinterpret_cast<const char *>(fileHeader.ident.data()),
               fileHeader.ident.size())
      << llvm::format("%02X%02X%02X%02X%08X\n", fileHeader.type,
                      fileHeader.machine, fileHeader.version, fileHeader.chnum,
                      fileHeader.choff);

  for (const auto &header : headers) {
    output << llvm::format(
        "%08X\n"
        "%08X\n"
        "%08X\n"
        "%08X\n"
        "%08X\n"
        "%08X\n"
        "%08X\n"
        "%08X\n",
        header.name | (static_cast<uint32_t>(header.tile) << 8u), header.type,
        0, header.address, 0, header.offset, 0, header.size);
  }

  for (auto &section : writes) {
    for (auto &write : section) {
      output << llvm::format("%08X\n", write.value());
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
  configure_switchboxes(module);
  configure_dmas(module, NL);

  std::sort(writes.begin(), writes.end());

  auto section_writes = group_sections();
  auto sections = make_section_headers(section_writes);

  output_sections(output, sections, section_writes);

  return success();
}
} // namespace AIE
} // namespace xilinx
