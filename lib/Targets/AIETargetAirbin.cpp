//===- AIETargetAirbin.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <fcntl.h> // open
#include <gelf.h>
#include <iostream>
#include <libelf.h>
#include <set>
#include <sys/stat.h>
#include <unistd.h> // read
#include <utility>  // pair
#include <vector>

#define DEBUG_TYPE "aie-generate-airbin"

#define EM_AMDAIR 225 /* AMD AIR */

using namespace mlir;

namespace xilinx::AIE {

enum {
  SEC_IDX_NULL,
  SEC_IDX_SSMAST,
  SEC_IDX_SSSLVE,
  SEC_IDX_SSPCKT,
  SEC_IDX_SDMA_BD,
  SEC_IDX_SHMMUX,
  SEC_IDX_SDMA_CTL,
  SEC_IDX_PRGM_MEM,
  SEC_IDX_TDMA_BD,
  SEC_IDX_TDMA_CTL,
  SEC_IDX_DEPRECATED,
  SEC_IDX_DATA_MEM,
  SEC_IDX_MAX
};

static constexpr auto DISABLE = 0u;
static constexpr auto ENABLE = 1u;

static constexpr auto TILE_ADDR_OFF_WIDTH = 18u;

static constexpr auto TILE_ADDR_ROW_SHIFT = TILE_ADDR_OFF_WIDTH;
static constexpr auto TILE_ADDR_ROW_WIDTH = 5u;

static constexpr auto TILE_ADDR_COL_SHIFT =
    TILE_ADDR_ROW_SHIFT + TILE_ADDR_ROW_WIDTH;
static constexpr auto TILE_ADDR_COL_WIDTH = 7u;

static constexpr auto TILE_ADDR_ARR_SHIFT =
    TILE_ADDR_COL_SHIFT + TILE_ADDR_COL_WIDTH;

/********
  ME Tile
********/
static constexpr auto ME_DATA_MEM_BASE = 0x00000u;
static constexpr auto ME_PROG_MEM_BASE = 0x20000u;
static constexpr auto ME_DMA_BD_BASE = 0x1D000u;
static constexpr auto ME_DMA_S2MM_BASE = 0x1DE00u;
static constexpr auto ME_DMA_MM2S_BASE = 0x1DE10u;
static constexpr auto ME_SS_MASTER_BASE = 0x3F000u;
static constexpr auto ME_SS_SLAVE_CFG_BASE = 0x3F100u;
static constexpr auto ME_SS_SLAVE_SLOT_BASE = 0x3F200u;

/*
  Tile DMA
*/
static constexpr auto ME_DMA_BD_COUNT = 48;
static constexpr auto ME_DMA_BD_SIZE = 0x20;

struct MERegDMABD {
  uint32_t addrA;
  uint32_t addrB;
  uint32_t x2d{0xff0000u | 0x001u}; // wrap at 256, increment by 1
  uint32_t y2d{0xff000000u | 0xff0000u |
               0x100u}; // wrap at 256, increment by 256 every 256 streams
  uint32_t packet;
  uint32_t interleave;
  uint32_t control;
  uint32_t padding;
};

static_assert(sizeof(MERegDMABD) == ME_DMA_BD_SIZE,
              "Size of me_reg_dma_bd is incorrect");

using DMABDRegBlock = MERegDMABD[ME_DMA_BD_COUNT];
static const MERegDMABD *
    DMABdRegs(reinterpret_cast<MERegDMABD *>(ME_DMA_BD_BASE));

static_assert(sizeof(DMABDRegBlock) == (ME_DMA_BD_COUNT * sizeof(MERegDMABD)),
              "Size of dma_bd_reg_block is incorrect");

auto regDMAAddrABD = [](auto idx) {
  return reinterpret_cast<uint64_t>(&DMABdRegs[idx].addrA);
};

auto regDMAAddrBBD = [](auto idx) {
  return reinterpret_cast<uint64_t>(&DMABdRegs[idx].addrB);
};

auto regDMA2DXBD = [](auto idx) {
  return reinterpret_cast<uint64_t>(&DMABdRegs[idx].x2d);
};

auto regDMA2DYBD = [](auto idx) {
  return reinterpret_cast<uint64_t>(&DMABdRegs[idx].y2d);
};

auto regDMAPktBD = [](auto idx) {
  return reinterpret_cast<uint64_t>(&DMABdRegs[idx].packet);
};

auto regDMAIntStateBD = [](auto idx) {
  return reinterpret_cast<uint64_t>(&DMABdRegs[idx].interleave);
};

auto regDMACtrlBD = [](auto idx) {
  return reinterpret_cast<uint64_t>(&DMABdRegs[idx].control);
};

/*
  DMA S2MM channel control
*/
static constexpr auto DMA_S2MM_CHANNEL_COUNT = 2u;
static constexpr auto REG_DMA_S2MM_BLOCK_SIZE = 0x08;

struct RegDMAS2MM {
  uint32_t ctrl;
  uint32_t queue;
};

static_assert(sizeof(RegDMAS2MM) == REG_DMA_S2MM_BLOCK_SIZE,
              "Size of reg_dma_s2mm is incorrect");

using DMAS2MMRegBlock = RegDMAS2MM[DMA_S2MM_CHANNEL_COUNT];
static const RegDMAS2MM *
    DMAS2MMRegs(reinterpret_cast<RegDMAS2MM *>(ME_DMA_S2MM_BASE));

auto regDMAS2MMCtrl = [](auto ch) {
  return reinterpret_cast<uint64_t>(&DMAS2MMRegs[ch].ctrl);
};

auto regDMAS2MMQueue = [](auto ch) {
  return reinterpret_cast<uint64_t>(&DMAS2MMRegs[ch].queue);
};

/*
  DMA MM2S channel control
*/
static constexpr auto DMA_MM2S_CHANNEL_COUNT = 2u;
static constexpr auto REG_DMA_MM2S_BLOCK_SIZE = 0x08;

struct RegDMAMM2S {
  uint32_t ctrl;
  uint32_t queue;
};

static_assert(sizeof(RegDMAMM2S) == REG_DMA_MM2S_BLOCK_SIZE,
              "Size of reg_dma_mm2s is incorrect");

using DMAMM2SRegBlock = RegDMAMM2S[DMA_MM2S_CHANNEL_COUNT];
static const RegDMAMM2S *
    DMAMM2SRegs(reinterpret_cast<RegDMAMM2S *>(ME_DMA_MM2S_BASE));

auto regDMAMM2SCtrl = [](auto ch) {
  return reinterpret_cast<uint64_t>(&DMAMM2SRegs[ch].ctrl);
};

auto regDMAMM2SQueue = [](auto ch) {
  return reinterpret_cast<uint64_t>(&DMAMM2SRegs[ch].queue);
};

/*
  ME stream switches
*/
static constexpr auto ME_SS_MASTER_COUNT = 25;
static constexpr auto ME_SS_SLAVE_CFG_COUNT = 27;
static constexpr auto ME_SS_SLAVE_SLOT_COUNT = 108;
static constexpr auto SS_SLOT_NUM_PORTS = 4u;

using MESSMasterBlock = uint32_t[ME_SS_MASTER_COUNT];
static const MESSMasterBlock *
    MESSMaster(reinterpret_cast<MESSMasterBlock *>(ME_SS_MASTER_BASE));

static_assert(sizeof(MESSMasterBlock) ==
                  (ME_SS_MASTER_COUNT * sizeof(uint32_t)),
              "Size of me_ss_master_block is incorrect");

auto regMESSMaster = [](auto idx) {
  return reinterpret_cast<uint64_t>(&MESSMaster[idx]);
};

using MESSSlaveCfgBlock = uint32_t[ME_SS_SLAVE_CFG_COUNT];
static const MESSSlaveCfgBlock *
    MESSSlaveCfg(reinterpret_cast<MESSSlaveCfgBlock *>(ME_SS_SLAVE_CFG_BASE));

static_assert(sizeof(MESSSlaveCfgBlock) ==
                  (ME_SS_SLAVE_CFG_COUNT * sizeof(uint32_t)),
              "Size of me_ss_slave_cfg_block is incorrect");

auto regMESSSlaveCfg = [](auto idx) {
  return reinterpret_cast<uint64_t>(&MESSSlaveCfg[idx]);
};

using MESSSlaveSlotBlock = uint32_t[ME_SS_SLAVE_SLOT_COUNT][SS_SLOT_NUM_PORTS];
static const MESSSlaveSlotBlock *MESSSlaveSlot(
    reinterpret_cast<MESSSlaveSlotBlock *>(ME_SS_SLAVE_SLOT_BASE));

static_assert(sizeof(MESSSlaveSlotBlock) ==
                  (ME_SS_SLAVE_SLOT_COUNT * SS_SLOT_NUM_PORTS *
                   sizeof(uint32_t)),
              "Size of me_ss_slave_slot_block is incorrect");

auto regMESSSlaveSlot = [](auto port, auto slot) {
  return reinterpret_cast<uint64_t>(&MESSSlaveSlot[slot][port]);
};

// ME data memory
static constexpr auto DATA_MEM_SIZE = 0x08000u; // 32KB

// ME program memory
static constexpr auto PROG_MEM_SIZE = 0x4000u; // 16KB

/**********
  Shim Tile
**********/
static constexpr auto SHIM_DMA_BD_BASE = 0x1D000u;
static constexpr auto SHIM_DMA_S2MM_BASE = 0x1D140u;
static constexpr auto SHIM_SS_MASTER_BASE = 0x3F000u;
static constexpr auto SHIM_SS_SLAVE_CFG_BASE = 0x3F100u;
static constexpr auto SHIM_SS_SLAVE_SLOT_BASE = 0x3F200u;

/*
  Shim DMA
*/
static constexpr auto SHIM_DMA_BD_COUNT = 16;
static constexpr auto REG_SHIM_DMA_BD_SIZE = 0x14;

struct ShimDMABD {
  uint32_t addrLow;
  uint32_t len;
  uint32_t control;
  uint32_t axiCfg;
  uint32_t packet;
};

static_assert(sizeof(struct ShimDMABD) == REG_SHIM_DMA_BD_SIZE,
              "Size of shim_dma_bd is incorrect");

using ShimDMABDBlock = ShimDMABD[SHIM_DMA_BD_COUNT];

/*
  Mux/demux
*/
static constexpr auto SHIM_MUX_BASE = 0x1F000u;

/*
  Shim stream switches
*/
static constexpr auto SHIM_SS_MASTER_COUNT = 23;
static constexpr auto SHIM_SS_SLAVE_CFG_COUNT = 24;
static constexpr auto SHIM_SS_SLAVE_SLOT_COUNT = 96;

using ShimSSMasterBlock = uint32_t[SHIM_SS_MASTER_COUNT];
using ShimSSSlaveCfgBlock = uint32_t[SHIM_SS_SLAVE_CFG_COUNT];
using ShimSSSlaveSlotBlock = uint32_t[SHIM_SS_SLAVE_SLOT_COUNT];

// section names
static uint8_t secNameOffset[SEC_IDX_MAX];

static const char *secNameStr[SEC_IDX_MAX] = {
    "null",     ".ssmast",   ".ssslve",    ".sspckt",
    ".sdma.bd", ".shmmux",   ".sdma.ctl",  ".prgm.mem",
    ".tdma.bd", ".tdma.ctl", "deprecated", ".data.mem"};

static size_t stridx;

/*
   Holds a sorted list of all writes made to device memory
   All recorded writes are time/order invariant. This allows sorting to
   compact the airbin.
*/
static std::map<uint64_t, uint32_t> memWrites;

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
  TileAddress(uint8_t column, uint8_t row, uint64_t arrayOffset = 0x000u)
      : arrayOffset{arrayOffset}, column{column}, row{row} {}

  // SFINAE is used here to choose the copy constructor for `TileAddress`,
  // and this constructor for all other classes.
  template <typename Op,
            std::enable_if_t<!std::is_same_v<Op, TileAddress>, bool> = true>
  TileAddress(Op &op)
      : TileAddress{static_cast<uint8_t>(op.colIndex()),
                    static_cast<uint8_t>(op.rowIndex())} {}

  uint64_t fullAddress(uint64_t registerOffset) const {
    return (arrayOffset << TILE_ADDR_ARR_SHIFT) |
           (static_cast<uint64_t>(column) << TILE_ADDR_COL_SHIFT) |
           (static_cast<uint64_t>(row) << TILE_ADDR_ROW_SHIFT) | registerOffset;
  }

  bool isShim() const { return row == 0; }

  operator uint16_t() const {
    return (static_cast<uint16_t>(column) << TILE_ADDR_ROW_WIDTH) | row;
  }

  uint8_t col() const { return column; }

  void clearRange(uint32_t rangeStart, uint32_t length);

private:
  uint64_t arrayOffset : 34;
  uint8_t column : TILE_ADDR_COL_WIDTH;
  uint8_t row : TILE_ADDR_ROW_WIDTH;
};

static_assert(sizeof(TileAddress) <= sizeof(uint64_t),
              "Tile addresses are at most 64-bits");

class Address {
public:
  Address(TileAddress tile, uint64_t offset) : tile{tile}, offset{offset} {}

  operator uint64_t() const { return tile.fullAddress(offset); }

  TileAddress destTile() const { return tile; }
  uint32_t getOffset() const { return offset; }

private:
  TileAddress tile;
  uint64_t offset : TILE_ADDR_OFF_WIDTH;
};

using Write = std::pair<uint64_t, uint32_t>;

class Section {
public:
  Section(uint64_t addr) : address(addr){};
  uint64_t getAddr() const { return address; }
  size_t getLength() const { return data.size() * sizeof(uint32_t); }
  void addData(uint32_t value) { data.push_back(value); }
  const uint32_t *getData() const { return data.data(); }

private:
  uint64_t address;           // start address of this section
  std::vector<uint32_t> data; // data to be written starting at 'address'
};

// This template can be instantiated to represent a bitfield in a register.
template <uint8_t highBit, uint8_t lowBit = highBit>
class Field final {
public:
  static_assert(highBit >= lowBit,
                "The high bit should be higher than the low bit");
  static_assert(highBit < sizeof(uint32_t) * 8u,
                "The field must live in a 32-bit register");

  static constexpr auto NUM_BITS_USED = (highBit - lowBit) + 1u;
  static constexpr auto UNSHIFTED_MASK = (1u << NUM_BITS_USED) - 1u;
  static_assert((lowBit != highBit) ^ (UNSHIFTED_MASK == 1),
                "1 is a valid mask iff the field is 1 bit wide");

  static constexpr auto SHIFTED_MASK = UNSHIFTED_MASK << lowBit;

  [[nodiscard]] constexpr uint32_t operator()(uint32_t value) const {
    return (value << lowBit) & SHIFTED_MASK;
  }
};

/*
        Add or replace a register value in 'mem_writes'
*/
static void write32(Address addr, uint32_t value) {
  if (addr.destTile().col() <= 0)
    llvm::report_fatal_error(
        llvm::Twine("address of destination tile <= 0 : ") +
        std::to_string(addr.destTile().col()));

  auto ret = memWrites.emplace(addr, value);
  if (!ret.second)
    ret.first->second = value;
}

/*
        Look up a value for a given address

        If the address is found return the value, otherwise 0
*/
static uint32_t read32(Address addr) {
  auto ret = memWrites.find(addr);
  if (ret != memWrites.end())
    return ret->second;

  return 0;
}

/*
        Set every address in the range to 0
*/
void TileAddress::clearRange(uint32_t start, uint32_t length) {
  if (start % 4 != 0)
    llvm::report_fatal_error(llvm::Twine("start address ") +
                             std::to_string(start) +
                             " must word 4 byte aligned");
  if (start % 4 != 0)
    llvm::report_fatal_error(llvm::Twine("length ") + std::to_string(start) +
                             " must be a multiple of 4 bytes");

  LLVM_DEBUG(llvm::dbgs() << llvm::format("<%u,%u> 0x%x - 0x%x (len: %u)\n",
                                          column, row, start, start + length,
                                          length));
  for (auto off = start; off < start + length; off += 4u)
    write32(Address{*this, off}, 0);
}

/*
   Read the ELF produced by the AIE compiler and include its loadable
   output in the airbin ELF
*/
static void loadElf(TileAddress tile, const std::string &filename) {
  LLVM_DEBUG(llvm::dbgs() << "Reading ELF file " << filename << " for tile "
                          << tile << '\n');

  int elfFd = open(filename.c_str(), O_RDONLY);
  if (elfFd < 0)
    llvm::report_fatal_error(llvm::Twine("Can't open elf file ") + filename);

  elf_version(EV_CURRENT);
  Elf *inElf = elf_begin(elfFd, ELF_C_READ, nullptr);

  // check the characteristics
  GElf_Ehdr *ehdr;
  GElf_Ehdr ehdrMem;
  ehdr = gelf_getehdr(inElf, &ehdrMem);
  if (!ehdr)
    llvm::report_fatal_error(llvm::Twine("cannot get ELF header: ") +
                             elf_errmsg(-1));

  // Read data as 32-bit little endian
  assert(ehdr->e_ident[EI_CLASS] == ELFCLASS32 &&
         "(ehdr->e_ident[EI_CLASS] != ELFCLASS32");
  assert(ehdr->e_ident[EI_DATA] == ELFDATA2LSB &&
         "ehdr->e_ident[EI_DATA] != ELFDATA2LSB");

  size_t phnum;
  if (elf_getphdrnum(inElf, &phnum) != 0)
    llvm::report_fatal_error(llvm::Twine("cannot get program header count: ") +
                             elf_errmsg(-1));

  // iterate through all program headers
  for (unsigned int ndx = 0; ndx < phnum; ndx++) {
    GElf_Phdr phdrMem;
    GElf_Phdr *phdr = gelf_getphdr(inElf, ndx, &phdrMem);
    if (!phdr)
      llvm::report_fatal_error(llvm::Twine("cannot get program header entry ") +
                               std::to_string(ndx) + ": " + elf_errmsg(-1));

    // for each loadable program header
    if (phdr->p_type != PT_LOAD)
      continue;

    // decide destination address based on header attributes
    uint32_t dest;
    if (phdr->p_flags & PF_X)
      dest = ME_PROG_MEM_BASE + phdr->p_vaddr;
    else
      dest = ME_DATA_MEM_BASE + (phdr->p_vaddr & (DATA_MEM_SIZE - 1));

    LLVM_DEBUG(llvm::dbgs()
               << llvm::format("ELF flags=0x%x vaddr=0x%lx dest=0x%x\r\n",
                               phdr->p_flags, phdr->p_vaddr, dest));
    // read data one word at a time and write it to the output list
    // TODO since we know these are data and not registers, we could likely
    // bypass the output list and write a section directly into the AIRBIN
    size_t elfSize;
    uint32_t offset;
    char *raw = elf_rawfile(inElf, &elfSize);

    for (offset = phdr->p_offset; offset < phdr->p_offset + phdr->p_filesz;
         offset += 4) {
      Address destAddr{tile, dest};
      uint32_t data = *reinterpret_cast<uint32_t *>(raw + offset);
      write32(destAddr, data);
      dest += 4;
    }
  }

  elf_end(inElf);
  close(elfFd);
}

/*
  The SHIM row is always 0.
  SHIM resets are handled by the runtime.
*/
static void configShimTile(TileOp &tileOp) {
  assert(tileOp.isShimTile() &&
         "The tile must be a Shim to generate Shim Config");

  TileAddress tileAddress{tileOp};

  if (tileOp.isShimNOCTile())
    tileAddress.clearRange(SHIM_DMA_BD_BASE, sizeof(ShimDMABDBlock));

  tileAddress.clearRange(SHIM_SS_MASTER_BASE, sizeof(ShimSSMasterBlock));
  tileAddress.clearRange(SHIM_SS_SLAVE_CFG_BASE, sizeof(ShimSSSlaveCfgBlock));
  tileAddress.clearRange(SHIM_SS_SLAVE_SLOT_BASE, sizeof(ShimSSSlaveSlotBlock));
}

/*
  Generate the config for an ME tile
*/
static void configMETile(TileOp tileOp, const std::string &coreFilesDir) {
  TileAddress tileAddress{tileOp};
  // Reset configuration

  // clear program and data memory
  tileAddress.clearRange(ME_PROG_MEM_BASE, PROG_MEM_SIZE);
  tileAddress.clearRange(ME_DATA_MEM_BASE, DATA_MEM_SIZE);

  // TileDMA
  tileAddress.clearRange(ME_DMA_BD_BASE, sizeof(DMABDRegBlock));
  tileAddress.clearRange(ME_DMA_S2MM_BASE, sizeof(DMAS2MMRegBlock));
  tileAddress.clearRange(ME_DMA_MM2S_BASE, sizeof(DMAMM2SRegBlock));

  // Stream Switches
  tileAddress.clearRange(ME_SS_MASTER_BASE, sizeof(MESSMasterBlock));
  tileAddress.clearRange(ME_SS_SLAVE_CFG_BASE, sizeof(MESSSlaveCfgBlock));
  tileAddress.clearRange(ME_SS_SLAVE_SLOT_BASE, sizeof(MESSSlaveSlotBlock));

  // NOTE: Here is usually where locking is done.
  // However, the runtime will handle that when loading the airbin.

  // read the AIE executable and copy the loadable parts
  if (auto coreOp = tileOp.getCoreOp()) {
    std::string fileName;
    if (auto fileAttr = coreOp->getAttrOfType<StringAttr>("elf_file"))
      fileName = fileAttr.str();
    else
      fileName = llvm::formatv("{0}/core_{1}_{2}.elf", coreFilesDir,
                               tileOp.colIndex(), tileOp.rowIndex());
    loadElf(tileAddress, fileName);
  }
}

struct BDInfo {
  bool foundBDPacket = false;
  int packetType = 0;
  int packetID = 0;
  bool foundBD = false;
  int lenA = 0;
  int lenB = 0;
  unsigned bytesA = 0;
  unsigned bytesB = 0;
  int offsetA = 0;
  int offsetB = 0;
  uint64_t baseAddrA = 0;
  uint64_t baseAddrB = 0;
  bool hasA = false;
  bool hasB = false;
  std::string bufA = "0";
  std::string bufB = "0";
  uint32_t abMode = DISABLE;
  uint32_t fifoMode = DISABLE; // FIXME: when to enable FIFO mode?
};

static BDInfo getBDInfo(Block &block) {
  BDInfo bdInfo;
  for (auto op : block.getOps<DMABDOp>()) {
    bdInfo.foundBD = true;
    assert(op.getBufferOp().getAddress().has_value() &&
           "buffer op should have address");
    bdInfo.baseAddrA = op.getBufferOp().getAddress().value();
    bdInfo.lenA = op.getLenIn32bWords() * 4;
    bdInfo.bytesA = op.getBufferElementTypeWidthInBytes();
    bdInfo.offsetA = op.getOffset() * 4;
    bdInfo.bufA = "XAIEDMA_TILE_BD_ADDRA";
    bdInfo.hasA = true;
  }
  return bdInfo;
}

static void configureDMAs(DeviceOp &targetOp) {
  Field<1> dmaChannelReset;
  Field<0> dmaChannelEnable;

  for (auto memOp : targetOp.getOps<MemOp>()) {
    TileAddress tile{memOp};
    LLVM_DEBUG(llvm::dbgs() << "DMA: tile=" << memOp.getTile());
    // Clear the CTRL and QUEUE registers for the DMA channels.
    for (auto chNum = 0u; chNum < DMA_S2MM_CHANNEL_COUNT; ++chNum) {
      write32({tile, regDMAS2MMCtrl(chNum)},
              dmaChannelReset(DISABLE) | dmaChannelEnable(DISABLE));
      write32({tile, regDMAS2MMQueue(chNum)}, 0);
    }
    for (auto chNum = 0u; chNum < DMA_MM2S_CHANNEL_COUNT; ++chNum) {
      write32({tile, regDMAMM2SCtrl(chNum)},
              dmaChannelReset(DISABLE) | dmaChannelEnable(DISABLE));
      write32({tile, regDMAMM2SQueue(chNum)}, 0);
    }

    DenseMap<Block *, int> blockMap;

    {
      // Assign each block a BD number
      auto bdNum = 0;
      for (auto &block : memOp.getBody()) {
        if (!block.getOps<DMABDOp>().empty()) {
          blockMap[&block] = bdNum;
          bdNum++;
        }
      }
    }

    for (auto &block : memOp.getBody()) {
      auto bdInfo = getBDInfo(block);

      if (bdInfo.hasA and bdInfo.hasB) {
        bdInfo.abMode = ENABLE;
        if (bdInfo.lenA != bdInfo.lenB)
          llvm::errs() << "ABmode must have matching lengths.\n";
        if (bdInfo.bytesA != bdInfo.bytesB)
          llvm::errs() << "ABmode must have matching element data types.\n";
      }

      int acqValue = 0, relValue = 0;
      auto acqEnable = DISABLE;
      auto relEnable = DISABLE;
      std::optional<int> lockID = std::nullopt;

      for (auto op : block.getOps<UseLockOp>()) {
        LockOp lock = dyn_cast<LockOp>(op.getLock().getDefiningOp());
        lockID = lock.getLockIDValue();
        if (op.acquire()) {
          acqEnable = ENABLE;
          acqValue = op.getLockValue();
        } else {
          relEnable = ENABLE;
          relValue = op.getLockValue();
        }
      }

      // We either
      //  a. went thru the loop once (`lockID` should be something) xor
      //  b. did not enter the loop (the enables should be both disable)
      assert(lockID.has_value() ^
                 (acqEnable == DISABLE and relEnable == DISABLE) &&
             "lock invariants not satisfied");

      for (auto op : block.getOps<DMABDPACKETOp>()) {
        bdInfo.foundBDPacket = true;
        bdInfo.packetType = op.getPacketType();
        bdInfo.packetID = op.getPacketID();
      }

      auto bdNum = blockMap[&block];
      MERegDMABD bdData;
      if (bdInfo.foundBD) {
        Field<25, 22> bdAddressLockID;
        Field<21> bdAddressReleaseEnable;
        Field<20> bdAddressReleaseValue;
        Field<19> bdAddressReleaseValueEnable;
        Field<18> bdAddressAcquireEnable;
        Field<17> bdAddressAcquireValue;
        Field<16> bdAddressAcquireValueEnable;

        if (bdInfo.hasA) {
          bdData.addrA = bdAddressLockID(lockID.value()) |
                         bdAddressReleaseEnable(relEnable) |
                         bdAddressAcquireEnable(acqEnable);
          if (relValue != 0xFFu)
            bdData.addrA |= bdAddressReleaseValueEnable(true) |
                            bdAddressReleaseValue(relValue);
          if (acqValue != 0xFFu)
            bdData.addrA |= bdAddressAcquireValueEnable(true) |
                            bdAddressAcquireValue(acqValue);
        }
        if (bdInfo.hasB)
          llvm::report_fatal_error("bdInfo.hasB not supported");

        auto addrA = bdInfo.baseAddrA + bdInfo.offsetA;
        auto addrB = bdInfo.baseAddrB + bdInfo.offsetB;

        Field<12, 0> bdAddressBase, bdControlLength;
        Field<30> bdControlABMode;
        Field<28> bdControlFifo;

        bdData.addrA |= bdAddressBase(addrA >> 2u);
        bdData.addrB |= bdAddressBase(addrB >> 2u);
        bdData.control |= bdControlLength(bdInfo.lenA - 1) |
                          bdControlFifo(bdInfo.fifoMode) |
                          bdControlABMode(bdInfo.abMode);

        if (block.getNumSuccessors() > 0) {
          // should have only one successor block
          assert(block.getNumSuccessors() == 1 &&
                 "block.getNumSuccessors() != 1");
          auto *nextBlock = block.getSuccessors()[0];
          auto nextBDNum = blockMap[nextBlock];

          Field<16, 13> bdControlNextBD;
          Field<17> bdControlEnableNextBD;

          bdData.control |= bdControlEnableNextBD(nextBDNum != 0xFFu) |
                            bdControlNextBD(nextBDNum);
        }

        if (bdInfo.foundBDPacket) {
          Field<14, 12> bdPacketType;
          Field<4, 0> bdPacketID;
          Field<27> bdControlEnablePacket;

          bdData.packet =
              bdPacketID(bdInfo.packetID) | bdPacketType(bdInfo.packetType);
          bdData.control |= bdControlEnablePacket(ENABLE);
        }

        Field<31> bdControlValid;

        assert(bdNum < ME_DMA_BD_COUNT && "bdNum >= ME_DMA_BD_COUNT");
        uint64_t bdOffset = regDMAAddrABD(bdNum);

        write32({tile, bdOffset}, bdData.addrA);
        write32({tile, regDMAAddrBBD(bdNum)}, bdData.addrB);
        write32({tile, regDMA2DXBD(bdNum)}, bdData.x2d);
        write32({tile, regDMA2DYBD(bdNum)}, bdData.y2d);
        write32({tile, regDMAPktBD(bdNum)}, bdData.packet);
        write32({tile, regDMAIntStateBD(bdNum)}, bdData.interleave);
        write32({tile, regDMACtrlBD(bdNum)},
                bdData.control | bdControlValid(true));
      }
    }

    for (auto &block : memOp.getBody()) {
      for (auto op : block.getOps<DMAStartOp>()) {
        auto bdNum = blockMap[op.getDest()];
        if (bdNum != 0xFFU) {
          Field<4, 0> dmaChannelQueueStartBd;

          uint32_t chNum = op.getChannelIndex();
          if (op.getChannelDir() == DMAChannelDir::MM2S) {
            write32(Address{tile, regDMAMM2SQueue(chNum)},
                    dmaChannelQueueStartBd(bdNum));
            write32({tile, regDMAMM2SCtrl(chNum)},
                    dmaChannelEnable(ENABLE) | dmaChannelReset(DISABLE));
          } else {
            write32(Address{tile, regDMAS2MMQueue(chNum)},
                    dmaChannelQueueStartBd(bdNum));
            write32({tile, regDMAS2MMCtrl(chNum)},
                    dmaChannelEnable(ENABLE) | dmaChannelReset(DISABLE));
          }
        }
      }
    }
  }
}

static uint8_t computeSlavePort(WireBundle bundle, int index, bool isShim) {
  assert(index >= 0 && "index < 0");
  assert(index < UINT8_MAX - 21 && "index >= UINT8_MAX - 21");

  switch (bundle) {
  case WireBundle::DMA:
    return 2u + index;
  case WireBundle::East: {
    if (isShim)
      return 19u + index;
    return 21u + index;
  }
  case WireBundle::North: {
    if (isShim)
      return 15u + index;
    return 17u + index;
  }
  case WireBundle::South: {
    if (isShim)
      return 3u + index;
    return 7u + index;
  }
  case WireBundle::West: {
    if (isShim)
      return 11u + index;
    return 13u + index;
  }
  default:
    // To implement a new WireBundle,
    // look in libXAIE for the macros that handle the port.
    llvm::report_fatal_error("unexpected bundle");
  }
}

static uint8_t computeMasterPort(WireBundle bundle, int index, bool isShim) {
  assert(index >= 0 && "index < 0");
  assert(index < UINT8_MAX - 21 && "index >= UINT8_MAX - 21");

  switch (bundle) {
  case WireBundle::DMA:
    return 2u + index;
  case WireBundle::East: {
    if (isShim)
      return 19u + index;
    return 21u + index;
  }
  case WireBundle::North: {
    if (isShim)
      return 13u + index;
    return 15u + index;
  }
  case WireBundle::South: {
    if (isShim)
      return 3u + index;
    return 7u + index;
  }
  case WireBundle::West: {
    if (isShim)
      return 9u + index;
    return 11u + index;
  }
  default:
    // To implement a new WireBundle,
    // look in libXAIE for the macros that handle the port.
    llvm::report_fatal_error(llvm::Twine("unexpected bundle") +
                             std::to_string(static_cast<uint32_t>(bundle)));
  }
}

static void configureSwitchBoxes(DeviceOp &targetOp) {
  for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
    Region &r = switchboxOp.getConnections();
    Block &b = r.front();
    bool isEmpty = b.getOps<ConnectOp>().empty() &&
                   b.getOps<MasterSetOp>().empty() &&
                   b.getOps<PacketRulesOp>().empty();

    // NOTE: may not be needed
    std::set<TileAddress> switchboxSet;
    if (isa<TileOp>(switchboxOp.getTile().getDefiningOp())) {
      if (!isEmpty)
        switchboxSet.emplace(switchboxOp);
    } else if (AIEX::SelectOp sel = dyn_cast<AIEX::SelectOp>(
                   switchboxOp.getTile().getDefiningOp()))
      // TODO: Use XAIEV1 target and translate into write32s
      llvm::report_fatal_error("select op not supported");

    constexpr Field<31> STREAM_ENABLE;
    constexpr Field<30> STREAM_PACKET_ENABLE;
    for (auto connectOp : b.getOps<ConnectOp>()) {
      for (auto tile : switchboxSet) {
        auto slavePort =
            computeSlavePort(connectOp.getSourceBundle(),
                             connectOp.sourceIndex(), tile.isShim());
        auto masterPort = computeMasterPort(
            connectOp.getDestBundle(), connectOp.destIndex(), tile.isShim());

        Field<7> streamMasterDropHeader;
        Field<6, 0> streamMasterConfig;

        // Configure master side
        {
          Address address{tile, regMESSMaster(masterPort)};
          // TODO: `Field::extract(uint32_t)`?
          auto dropHeader = (slavePort & 0x80u) >> 7u;
          auto value = STREAM_ENABLE(true) | STREAM_PACKET_ENABLE(false) |
                       streamMasterDropHeader(dropHeader) |
                       streamMasterConfig(slavePort);
          assert(value < UINT32_MAX);
          write32(address, value);
        }

        // Configure slave side
        {
          Address address{tile, regMESSSlaveCfg(slavePort)};
          write32(address, STREAM_ENABLE(true) | STREAM_PACKET_ENABLE(false));
        }

        for (auto connectOp : b.getOps<MasterSetOp>()) {
          auto mask = 0u;
          int arbiter = -1;
          for (auto val : connectOp.getAmsels()) {
            auto amsel = dyn_cast<AMSelOp>(val.getDefiningOp());
            arbiter = amsel.arbiterIndex();
            int msel = amsel.getMselValue();
            mask |= 1u << msel;
          }

          static constexpr auto STREAM_SWITCH_MSEL_SHIFT = 3u;
          static constexpr auto STREAM_SWITCH_ARB_SHIFT = 0u;

          const auto DROP_HEADER = connectOp.getDestBundle() == WireBundle::DMA;
          auto config = streamMasterDropHeader(DROP_HEADER) |
                        (mask << STREAM_SWITCH_MSEL_SHIFT) |
                        (arbiter << STREAM_SWITCH_ARB_SHIFT);
          Address dest{tile, regMESSMaster(masterPort)};
          write32(dest, STREAM_ENABLE(ENABLE) | STREAM_PACKET_ENABLE(ENABLE) |
                            streamMasterDropHeader(DROP_HEADER) |
                            streamMasterConfig(config));
        }
      }
    }

    for (auto connectOp : b.getOps<PacketRulesOp>()) {
      int slot = 0;
      Block &block = connectOp.getRules().front();
      for (auto slotOp : block.getOps<PacketRuleOp>()) {
        AMSelOp amselOp = dyn_cast<AMSelOp>(slotOp.getAmsel().getDefiningOp());
        int arbiter = amselOp.arbiterIndex();
        int msel = amselOp.getMselValue();

        for (auto tile : switchboxSet) {
          auto slavePort =
              computeSlavePort(connectOp.getSourceBundle(),
                               connectOp.sourceIndex(), tile.isShim());
          write32({tile, regMESSSlaveCfg(slavePort)},
                  STREAM_ENABLE(ENABLE) | STREAM_PACKET_ENABLE(ENABLE));

          Field<28, 24> streamSlotId;
          Field<20, 16> streamSlotMask;
          Field<8> streamSlotEnable;
          Field<5, 4> streamSlotMSel;
          Field<2, 0> streamSlotArbit;

          auto config = streamSlotId(slotOp.valueInt()) |
                        streamSlotMask(slotOp.maskInt()) |
                        streamSlotEnable(ENABLE) | streamSlotMSel(msel) |
                        streamSlotArbit(arbiter);
          write32({tile, regMESSSlaveSlot(slavePort, slot)}, config);
          slot++;
        }
      }
    }
  }

  const auto INPUT_MASK_FOR = [](WireBundle bundle, uint8_t shiftAmt) {
    switch (bundle) {
    case WireBundle::PLIO:
      return 0u << shiftAmt;
    case WireBundle::DMA:
      return 1u << shiftAmt;
    case WireBundle::NOC:
      return 2u << shiftAmt;
    default:
      llvm::report_fatal_error(llvm::Twine("unexpected bundle: ") +
                               std::to_string(static_cast<uint32_t>(bundle)));
    }
  };

  std::optional<TileAddress> currentTile = std::nullopt;
  for (auto op : targetOp.getOps<ShimMuxOp>()) {
    Region &r = op.getConnections();
    Block &b = r.front();

    if (isa<TileOp>(op.getTile().getDefiningOp())) {
      bool isEmpty = b.getOps<ConnectOp>().empty();
      if (!isEmpty)
        currentTile = op;
    }

    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.getSourceBundle() == WireBundle::North) {
        // demux!
        // XAieTile_ShimStrmDemuxConfig(&(TileInst[col][0]),
        // XAIETILE_SHIM_STRM_DEM_SOUTH3, XAIETILE_SHIM_STRM_DEM_DMA);
        assert(currentTile.has_value() && "current tile not set");
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
          default: // Unsure about this, but seems safe to assume
            llvm::report_fatal_error(llvm::Twine("unexpected source index: ") +
                                     std::to_string(index));
          }
        }();

        // We need to add to the possibly preexisting mask.
        Address addr{currentTile.value(), 0x1F004u};
        auto currentMask = read32(addr);
        write32(addr, currentMask |
                          INPUT_MASK_FOR(connectOp.getDestBundle(), shiftAmt));
      } else if (connectOp.getDestBundle() == WireBundle::North) {
        // mux
        // XAieTile_ShimStrmMuxConfig(&(TileInst[col][0]),
        // XAIETILE_SHIM_STRM_MUX_SOUTH3, XAIETILE_SHIM_STRM_MUX_DMA);
        assert(currentTile.has_value() && "no current tile");
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
          default: // Unsure about this, but seems safe to assume
            llvm::report_fatal_error(llvm::Twine("unexpected dest index ") +
                                     std::to_string(index));
          }
        }();

        Address addr{currentTile.value(), 0x1F000u};
        auto currentMask = read32(addr);
        write32(addr, currentMask | INPUT_MASK_FOR(connectOp.getSourceBundle(),
                                                   shiftAmt));
      }
    }
  }

  /* TODO: Implement the following
  for (auto switchboxOp : targetOp.getOps<ShimSwitchboxOp>()) {
    Region &r = switchboxOp.getConnections();
    Block &b = r.front();
    for (auto connectOp : b.getOps<ConnectOp>()) {
      output << "XAieTile_StrmConnectCct(" << tileInstStr(col, 0) << ",\n";
      output << "\tXAIETILE_STRSW_SPORT_"
             << stringifyWireBundle(connectOp.sourceBundle()).upper() << "("
             << tileInstStr(col, 0) << ", " << connectOp.sourceIndex()
             << "),\n";
      output << "\tXAIETILE_STRSW_MPORT_"
             << stringifyWireBundle(connectOp.destBundle()).upper() << "("
             << tileInstStr(col, 0) << ", " << connectOp.destIndex() << "),\n";
      output << "\t" << enable << ");\n";
    }
  }
  */
}

static void configureCascade(DeviceOp &targetOp) {
  const auto &target_model = xilinx::AIE::getTargetModel(targetOp);
  if (target_model.getTargetArch() == AIEArch::AIE2) {
    for (auto configOp : targetOp.getOps<ConfigureCascadeOp>()) {
      TileOp tile = cast<TileOp>(configOp.getTile().getDefiningOp());
      auto inputDir = stringifyCascadeDir(configOp.getInputDir()).upper();
      auto outputDir = stringifyCascadeDir(configOp.getOutputDir()).upper();

      Address address{tile, 0x36060u};

      /*
       *  * Register value for output BIT 1: 0 == SOUTH, 1 == EAST
       *  * Register value for input BIT 0: 0 == NORTH, 1 == WEST
       */
      uint8_t outputValue = (outputDir == "SOUTH") ? 0 : 1;
      uint8_t inputValue = (inputDir == "NORTH") ? 0 : 1;

      constexpr Field<1> Output;
      constexpr Field<0> Input;

      auto regValue = Output(outputValue) | Input(inputValue);

      write32(address, regValue);
    }
  }
}

/*
        Convert memory address to index

        Used to look up register/region name
*/
static uint8_t secAddr2Index(uint64_t in) {
  switch (in & ((1 << TILE_ADDR_OFF_WIDTH) - 1)) {
  case 0:
    return SEC_IDX_DATA_MEM;
  case ME_SS_MASTER_BASE:
    return SEC_IDX_SSMAST;
  case ME_SS_SLAVE_CFG_BASE:
    return SEC_IDX_SSSLVE;
  case ME_SS_SLAVE_SLOT_BASE:
    return SEC_IDX_SSPCKT;
  case ME_DMA_BD_BASE:
    return SEC_IDX_SDMA_BD;
  case SHIM_MUX_BASE:
    return SEC_IDX_SHMMUX;
  case SHIM_DMA_S2MM_BASE:
    return SEC_IDX_SDMA_CTL;
  case ME_PROG_MEM_BASE:
    return SEC_IDX_PRGM_MEM;
  case ME_DMA_S2MM_BASE:
    return SEC_IDX_TDMA_CTL;
  default:
    return 0;
  }
}

/*
        Group the writes into contiguous sections
*/
static void groupSections(std::vector<Section *> &sections) {
  uint64_t lastAddr = 0;
  Section *section = nullptr;

  for (auto write : memWrites) {
    if (write.first != lastAddr + 4) {
      if (section)
        sections.push_back(section);
      section = new Section(write.first);
      LLVM_DEBUG(llvm::dbgs() << "Starting new section @ "
                              << llvm::format("0x%lx (last=0x%lx)\n",
                                              write.first, lastAddr));
    }
    assert(section && "section is null");
    section->addData(write.second);
    lastAddr = write.first;
  }

  sections.push_back(section);
}

/*
   Add a string to the section header string table and return the offset of
   the start of the string
*/
static size_t addString(Elf_Scn *scn, const char *str) {
  size_t lastidx = stridx;
  size_t size = strlen(str) + 1;

  Elf_Data *data = elf_newdata(scn);
  data->d_buf = (void *)str;
  data->d_type = ELF_T_BYTE;
  data->d_size = size;
  data->d_align = 1;
  data->d_version = EV_CURRENT;

  stridx += size;
  return lastidx;
}

Elf_Data *sectionAddData(Elf_Scn *scn, const Section *section) {
  size_t size = section->getLength();
  auto *buf = static_cast<uint32_t *>(malloc(size));

  // create a data object for the section
  Elf_Data *data = elf_newdata(scn);
  data->d_buf = buf;
  data->d_type = ELF_T_BYTE;
  data->d_size = size;
  data->d_off = 0;
  data->d_align = 1;
  data->d_version = EV_CURRENT;

  // fill the data
  memcpy(buf, section->getData(), size);

  return data;
}

mlir::LogicalResult AIETranslateToAirbin(mlir::ModuleOp module,
                                         const std::string &outputFilename,
                                         const std::string &coreFilesDir,
                                         bool testAirBin) {
  int tmpElfFD;
  Elf *outElf;
  GElf_Ehdr ehdrMem;
  GElf_Ehdr *ehdr;
  GElf_Shdr *shdr;
  GElf_Shdr shdrMem;
  char emptyStr[] = "";
  char strTabName[] = ".shstrtab";
  std::vector<Section *> sections;

  if (module.getOps<DeviceOp>().empty()) {
    LLVM_DEBUG(llvm::dbgs() << "no device ops found");
    return success();
  }

  DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

  // Write the initial configuration for every tile specified in the MLIR.
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "CC: tile=" << tileOp.getTileID());
    if (tileOp.isShimTile())
      configShimTile(tileOp);
    else
      configMETile(tileOp, coreFilesDir);
  }

  configureSwitchBoxes(targetOp);
  configureCascade(targetOp);
  configureDMAs(targetOp);
  groupSections(sections);

  LLVM_DEBUG(llvm::dbgs() << llvm::format("mem_writes: %lu in %lu sections\n",
                                          memWrites.size(), sections.size()));

  elf_version(EV_CURRENT);
  tmpElfFD =
      open(outputFilename.c_str(), O_RDWR | O_CREAT | O_TRUNC, DEFFILEMODE);
  outElf = elf_begin(tmpElfFD, ELF_C_WRITE, nullptr);

  if (!gelf_newehdr(outElf, ELFCLASS64))
    llvm::report_fatal_error(llvm::Twine("Error creating ELF64 header: ") +
                             elf_errmsg(-1));

  ehdr = gelf_getehdr(outElf, &ehdrMem);
  if (!ehdr)
    llvm::report_fatal_error(llvm::Twine("cannot get ELF header: ") +
                             elf_errmsg(-1));

  // Initialize header.
  ehdr->e_ident[EI_DATA] = ELFDATA2LSB;
  ehdr->e_ident[EI_OSABI] = ELFOSABI_GNU;
  ehdr->e_type = ET_NONE;
  ehdr->e_machine = EM_AMDAIR;
  ehdr->e_version = EV_CURRENT;
  if (gelf_update_ehdr(outElf, ehdr) == 0)
    llvm::report_fatal_error(llvm::Twine("cannot update ELF header: ") +
                             elf_errmsg(-1));

  // Create new section for the 'section header string table'
  Elf_Scn *shStrTabScn = elf_newscn(outElf);
  if (!shStrTabScn)
    llvm::report_fatal_error(
        llvm::Twine("cannot create new shstrtab section: ") + elf_errmsg(-1));

  // the first entry in the string table must be a NULL string
  addString(shStrTabScn, emptyStr);

  shdr = gelf_getshdr(shStrTabScn, &shdrMem);
  if (!shdr)
    llvm::report_fatal_error(
        llvm::Twine("cannot get header for sh_strings section: ") +
        elf_errmsg(-1));

  shdr->sh_type = SHT_STRTAB;
  shdr->sh_flags = 0;
  shdr->sh_addr = 0;
  shdr->sh_link = SHN_UNDEF;
  shdr->sh_info = SHN_UNDEF;
  shdr->sh_addralign = 1;
  shdr->sh_entsize = 0;
  shdr->sh_name = addString(shStrTabScn, strTabName);

  // add all the AIRBIN-specific section names up front and index them
  for (uint8_t secIdx = SEC_IDX_SSMAST; secIdx < SEC_IDX_MAX; secIdx++)
    secNameOffset[secIdx] = addString(shStrTabScn, secNameStr[secIdx]);
  secNameOffset[SEC_IDX_NULL] = 0;

  // We have to store the section strtab index in the ELF header so sections
  // have actual names.
  int ndx = elf_ndxscn(shStrTabScn);
  ehdr->e_shstrndx = ndx;

  if (!gelf_update_ehdr(outElf, ehdr))
    llvm::report_fatal_error(llvm::Twine("cannot update ELF header: ") +
                             elf_errmsg(-1));

  // Finished new shstrtab section, update the header.
  if (!gelf_update_shdr(shStrTabScn, shdr))
    llvm::report_fatal_error(
        llvm::Twine("cannot update new shstrtab section header: ") +
        elf_errmsg(-1));

  // output the rest of the sections
  for (const Section *section : sections) {
    uint64_t addr = section->getAddr();
    Elf_Scn *scn = elf_newscn(outElf);
    if (!scn)
      llvm::report_fatal_error(llvm::Twine("cannot create new ") +
                               secNameStr[secAddr2Index(addr)] +
                               "section: " + elf_errmsg(-1));

    shdr = gelf_getshdr(scn, &shdrMem);
    if (!shdr)
      llvm::report_fatal_error(llvm::Twine("cannot get header for ") +
                               secNameStr[secAddr2Index(addr)] +
                               "section: " + elf_errmsg(-1));

    Elf_Data *data = sectionAddData(scn, section);

    shdr->sh_type = SHT_PROGBITS;
    shdr->sh_flags = SHF_ALLOC;
    shdr->sh_addr = section->getAddr();
    shdr->sh_link = SHN_UNDEF;
    shdr->sh_info = SHN_UNDEF;
    shdr->sh_addralign = 1;
    shdr->sh_entsize = 0;
    shdr->sh_size = data->d_size;
    shdr->sh_name = secNameOffset[secAddr2Index(addr)];

    if (!gelf_update_shdr(scn, shdr))
      llvm::report_fatal_error(llvm::Twine("cannot update section header: ") +
                               elf_errmsg(-1));
  }

  if (elf_update(outElf, ELF_C_WRITE) < 0)
    llvm::report_fatal_error(llvm::Twine("failure in elf_update: ") +
                             elf_errmsg(-1));
  elf_end(outElf);
  close(tmpElfFD);

  return success();
}
} // namespace xilinx::AIE
