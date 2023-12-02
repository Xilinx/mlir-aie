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

static constexpr auto disable = 0u;
static constexpr auto enable = 1u;

static constexpr auto TILE_ADDR_OFF_WIDTH = 18u;

static constexpr auto TILE_ADDR_ROW_SHIFT = TILE_ADDR_OFF_WIDTH;
static constexpr auto TILE_ADDR_ROW_WIDTH = 5u;

static constexpr auto TILE_ADDR_COL_SHIFT =
    TILE_ADDR_ROW_SHIFT + TILE_ADDR_ROW_WIDTH;
static constexpr auto TILE_ADDR_COL_WIDTH = 7u;

static constexpr auto TILE_ADDR_ARR_SHIFT =
    TILE_ADDR_COL_SHIFT + TILE_ADDR_COL_WIDTH;

static bool TEST_AIRBIN = false;

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
static constexpr auto ME_DMA_BD_COUNT = 16;
static constexpr auto ME_DMA_BD_SIZE = 0x20;

struct me_reg_dma_bd {
  uint32_t addr_a;
  uint32_t addr_b;
  uint32_t x_2d{0xff0000u | 0x001u}; // wrap at 256, increment by 1
  uint32_t y_2d{0xff000000u | 0xff0000u |
                0x100u}; // wrap at 256, increment by 256 every 256 streams
  uint32_t packet;
  uint32_t interleave;
  uint32_t control;
  uint32_t padding;
};

static_assert(sizeof(me_reg_dma_bd) == ME_DMA_BD_SIZE,
              "Size of me_reg_dma_bd is incorrect");

typedef me_reg_dma_bd dma_bd_reg_block[ME_DMA_BD_COUNT];
static const me_reg_dma_bd *
    dma_bd_regs(reinterpret_cast<me_reg_dma_bd *>(ME_DMA_BD_BASE));

static_assert(sizeof(dma_bd_reg_block) ==
                  (ME_DMA_BD_COUNT * sizeof(struct me_reg_dma_bd)),
              "Size of dma_bd_reg_block is incorrect");

auto reg_dma_addr_a_bd = [](auto _idx) {
  return reinterpret_cast<uint64_t>(&dma_bd_regs[_idx].addr_a);
};
auto reg_dma_addr_b_bd = [](auto _idx) {
  return reinterpret_cast<uint64_t>(&dma_bd_regs[_idx].addr_b);
};
auto reg_dma_2d_x_bd = [](auto _idx) {
  return reinterpret_cast<uint64_t>(&dma_bd_regs[_idx].x_2d);
};
auto reg_dma_2d_y_bd = [](auto _idx) {
  return reinterpret_cast<uint64_t>(&dma_bd_regs[_idx].y_2d);
};
auto reg_dma_pkt_bd = [](auto _idx) {
  return reinterpret_cast<uint64_t>(&dma_bd_regs[_idx].packet);
};
auto reg_dma_int_state_bd = [](auto _idx) {
  return reinterpret_cast<uint64_t>(&dma_bd_regs[_idx].interleave);
};
auto reg_dma_ctrl_bd = [](auto _idx) {
  return reinterpret_cast<uint64_t>(&dma_bd_regs[_idx].control);
};

/*
  DMA S2MM channel control
*/
static constexpr auto DMA_S2MM_CHANNEL_COUNT = 2u;
static constexpr auto REG_DMA_S2MM_BLOCK_SIZE = 0x08;

struct reg_dma_s2mm {
  uint32_t ctrl;
  uint32_t queue;
};

static_assert(sizeof(struct reg_dma_s2mm) == REG_DMA_S2MM_BLOCK_SIZE,
              "Size of reg_dma_s2mm is incorrect");

typedef reg_dma_s2mm dma_s2mm_reg_block[DMA_S2MM_CHANNEL_COUNT];
static const reg_dma_s2mm *
    dma_s2mm_regs(reinterpret_cast<reg_dma_s2mm *>(ME_DMA_S2MM_BASE));

auto reg_dma_s2mm_ctrl = [](auto _ch) {
  return reinterpret_cast<uint64_t>(&dma_s2mm_regs[_ch].ctrl);
};
auto reg_dma_s2mm_queue = [](auto _ch) {
  return reinterpret_cast<uint64_t>(&dma_s2mm_regs[_ch].queue);
};

/*
  DMA MM2S channel control
*/
static constexpr auto DMA_MM2S_CHANNEL_COUNT = 2u;
static constexpr auto REG_DMA_MM2S_BLOCK_SIZE = 0x08;

struct reg_dma_mm2s {
  uint32_t ctrl;
  uint32_t queue;
};

static_assert(sizeof(struct reg_dma_mm2s) == REG_DMA_MM2S_BLOCK_SIZE,
              "Size of reg_dma_mm2s is incorrect");

typedef reg_dma_mm2s dma_mm2s_reg_block[DMA_MM2S_CHANNEL_COUNT];
static const reg_dma_mm2s *
    dma_mm2s_regs(reinterpret_cast<reg_dma_mm2s *>(ME_DMA_MM2S_BASE));

auto reg_dma_mm2s_ctrl = [](auto _ch) {
  return reinterpret_cast<uint64_t>(&dma_mm2s_regs[_ch].ctrl);
};
auto reg_dma_mm2s_queue = [](auto _ch) {
  return reinterpret_cast<uint64_t>(&dma_mm2s_regs[_ch].queue);
};

/*
  ME stream switches
*/
static constexpr auto ME_SS_MASTER_COUNT = 25;
static constexpr auto ME_SS_SLAVE_CFG_COUNT = 27;
static constexpr auto ME_SS_SLAVE_SLOT_COUNT = 108;
static constexpr auto SS_SLOT_NUM_PORTS = 4u;

typedef uint32_t me_ss_master_block[ME_SS_MASTER_COUNT];
static const me_ss_master_block *
    me_ss_master(reinterpret_cast<me_ss_master_block *>(ME_SS_MASTER_BASE));

static_assert(sizeof(me_ss_master_block) ==
                  (ME_SS_MASTER_COUNT * sizeof(uint32_t)),
              "Size of me_ss_master_block is incorrect");

auto reg_me_ss_master = [](auto _idx) {
  return reinterpret_cast<uint64_t>(&me_ss_master[_idx]);
};

typedef uint32_t me_ss_slave_cfg_block[ME_SS_SLAVE_CFG_COUNT];
static const me_ss_slave_cfg_block *me_ss_slave_cfg(
    reinterpret_cast<me_ss_slave_cfg_block *>(ME_SS_SLAVE_CFG_BASE));

static_assert(sizeof(me_ss_slave_cfg_block) ==
                  (ME_SS_SLAVE_CFG_COUNT * sizeof(uint32_t)),
              "Size of me_ss_slave_cfg_block is incorrect");

auto reg_me_ss_slave_cfg = [](auto _idx) {
  return reinterpret_cast<uint64_t>(&me_ss_slave_cfg[_idx]);
};

typedef uint32_t me_ss_slave_slot_block[ME_SS_SLAVE_SLOT_COUNT]
                                       [SS_SLOT_NUM_PORTS];
static const me_ss_slave_slot_block *me_ss_slave_slot(
    reinterpret_cast<me_ss_slave_slot_block *>(ME_SS_SLAVE_SLOT_BASE));

static_assert(sizeof(me_ss_slave_slot_block) ==
                  (ME_SS_SLAVE_SLOT_COUNT * SS_SLOT_NUM_PORTS *
                   sizeof(uint32_t)),
              "Size of me_ss_slave_slot_block is incorrect");

auto reg_me_ss_slave_slot = [](auto _port, auto _slot) {
  return reinterpret_cast<uint64_t>(&me_ss_slave_slot[_slot][_port]);
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

struct shim_dma_bd {
  uint32_t addr_low;
  uint32_t len;
  uint32_t control;
  uint32_t axi_cfg;
  uint32_t packet;
};

static_assert(sizeof(struct shim_dma_bd) == REG_SHIM_DMA_BD_SIZE,
              "Size of shim_dma_bd is incorrect");

typedef shim_dma_bd shim_dma_bd_block[SHIM_DMA_BD_COUNT];

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

typedef uint32_t shim_ss_master_block[SHIM_SS_MASTER_COUNT];
typedef uint32_t shim_ss_slave_cfg_block[SHIM_SS_SLAVE_CFG_COUNT];
typedef uint32_t shim_ss_slave_slot_block[SHIM_SS_SLAVE_SLOT_COUNT];

// section names
static uint8_t sec_name_offset[SEC_IDX_MAX];

static const char *sec_name_str[SEC_IDX_MAX] = {
    "null",     ".ssmast",   ".ssslve",    ".sspckt",
    ".sdma.bd", ".shmmux",   ".sdma.ctl",  ".prgm.mem",
    ".tdma.bd", ".tdma.ctl", "deprecated", ".data.mem"};

static size_t stridx;

/*
   Holds a sorted list of all writes made to device memory
   All recorded writes are time/order invariant. This allows sorting to
   compact the airbin.
*/
static std::map<uint64_t, uint32_t> mem_writes;

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
  TileAddress(uint8_t column, uint8_t row, uint64_t array_offset = 0x000u)
      : array_offset{array_offset}, column{column}, row{row} {}

  // SFINAE is used here to choose the copy constructor for `TileAddress`,
  // and this constructor for all other classes.
  template <typename Op,
            std::enable_if_t<!std::is_same_v<Op, TileAddress>, bool> = true>
  TileAddress(Op &op)
      : TileAddress{static_cast<uint8_t>(op.colIndex()),
                    static_cast<uint8_t>(op.rowIndex())} {}

  uint64_t fullAddress(uint64_t register_offset) const {
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

  void clearRange(uint32_t range_start, uint32_t length);

private:
  uint64_t array_offset : 34;
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
  uint32_t get_offset() const { return offset; }

private:
  TileAddress tile;
  uint64_t offset : TILE_ADDR_OFF_WIDTH;
};

typedef std::pair<uint64_t, uint32_t> Write;

class Section {
public:
  Section(uint64_t addr) : address(addr){};
  uint64_t get_addr() const { return address; }
  size_t get_length() const { return data.size() * sizeof(uint32_t); }
  void add_data(uint32_t value) { data.push_back(value); }
  const uint32_t *get_data() const { return data.data(); }

private:
  uint64_t address;           // start address of this section
  std::vector<uint32_t> data; // data to be written starting at 'address'
};

// This template can be instantiated to represent a bitfield in a register.
template <uint8_t high_bit, uint8_t low_bit = high_bit> class Field final {
public:
  static_assert(high_bit >= low_bit,
                "The high bit should be higher than the low bit");
  static_assert(high_bit < sizeof(uint32_t) * 8u,
                "The field must live in a 32-bit register");

  static constexpr auto num_bits_used = (high_bit - low_bit) + 1u;
  static constexpr auto unshifted_mask = (1u << num_bits_used) - 1u;
  static_assert((low_bit != high_bit) ^ (unshifted_mask == 1),
                "1 is a valid mask iff the field is 1 bit wide");

  static constexpr auto shifted_mask = unshifted_mask << low_bit;

  [[nodiscard]] constexpr uint32_t operator()(uint32_t value) const {
    return (value << low_bit) & shifted_mask;
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

  auto ret = mem_writes.emplace(addr, value);
  if (!ret.second)
    (ret.first)->second = value;
}

/*
        Look up a value for a given address

        If the address is found return the value, otherwise 0
*/
static uint32_t read32(Address addr) {
  auto ret = mem_writes.find(addr);
  if (ret != mem_writes.end())
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
  for (auto off = start; off < start + length; off += 4u) {
    write32(Address{*this, off}, 0);
    if (TEST_AIRBIN)
      break;
  }
}

/*
   Read the ELF produced by the AIE compiler and include its loadable
   output in the airbin ELF
*/
static void loadElf(TileAddress tile, const std::string &filename) {
  LLVM_DEBUG(llvm::dbgs() << "Reading ELF file " << filename << " for tile "
                          << tile << '\n');

  int elf_fd = open(filename.c_str(), O_RDONLY);
  if (elf_fd < 0)
    llvm::report_fatal_error(llvm::Twine("Can't open elf file ") + filename);

  elf_version(EV_CURRENT);
  Elf *inelf = elf_begin(elf_fd, ELF_C_READ, nullptr);

  // check the characteristics
  GElf_Ehdr *ehdr;
  GElf_Ehdr ehdr_mem;
  ehdr = gelf_getehdr(inelf, &ehdr_mem);
  if (!ehdr)
    llvm::report_fatal_error(llvm::Twine("cannot get ELF header: ") +
                             elf_errmsg(-1));

  // Read data as 32-bit little endian
  assert(ehdr->e_ident[EI_CLASS] == ELFCLASS32);
  assert(ehdr->e_ident[EI_DATA] == ELFDATA2LSB);

  size_t phnum;
  if (elf_getphdrnum(inelf, &phnum) != 0)
    llvm::report_fatal_error(llvm::Twine("cannot get program header count: ") +
                             elf_errmsg(-1));

  // iterate through all program headers
  for (unsigned int ndx = 0; ndx < phnum; ndx++) {
    GElf_Phdr phdr_mem;
    GElf_Phdr *phdr = gelf_getphdr(inelf, ndx, &phdr_mem);
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
    size_t elfsize;
    uint32_t offset;
    char *raw = elf_rawfile(inelf, &elfsize);

    for (offset = phdr->p_offset; offset < phdr->p_offset + phdr->p_filesz;
         offset += 4) {
      Address dest_addr{tile, dest};
      uint32_t data = *(uint32_t *)(raw + offset);
      write32(dest_addr, data);
      dest += 4;
    }
  }

  elf_end(inelf);
  close(elf_fd);
}

/*
  The SHIM row is always 0.
  SHIM resets are handled by the runtime.
*/
static void config_shim_tile(TileOp &tileOp) {
  assert(tileOp.isShimTile() &&
         "The tile must be a Shim to generate Shim Config");

  TileAddress tileAddress{tileOp};

  if (tileOp.isShimNOCTile()) {
    tileAddress.clearRange(SHIM_DMA_BD_BASE, sizeof(shim_dma_bd_block));
  }

  tileAddress.clearRange(SHIM_SS_MASTER_BASE, sizeof(shim_ss_master_block));
  tileAddress.clearRange(SHIM_SS_SLAVE_CFG_BASE,
                         sizeof(shim_ss_slave_cfg_block));
  tileAddress.clearRange(SHIM_SS_SLAVE_SLOT_BASE,
                         sizeof(shim_ss_slave_slot_block));
}

/*
  Generate the config for an ME tile
*/
static void config_ME_tile(TileOp tileOp, const std::string &coreFilesDir) {
  TileAddress tileAddress{tileOp};
  // Reset configuration

  // clear program and data memory
  tileAddress.clearRange(ME_PROG_MEM_BASE, PROG_MEM_SIZE);
  tileAddress.clearRange(ME_DATA_MEM_BASE, DATA_MEM_SIZE);

  // TileDMA
  tileAddress.clearRange(ME_DMA_BD_BASE, sizeof(dma_bd_reg_block));
  tileAddress.clearRange(ME_DMA_S2MM_BASE, sizeof(dma_s2mm_reg_block));
  tileAddress.clearRange(ME_DMA_MM2S_BASE, sizeof(dma_mm2s_reg_block));

  // Stream Switches
  tileAddress.clearRange(ME_SS_MASTER_BASE, sizeof(me_ss_master_block));
  tileAddress.clearRange(ME_SS_SLAVE_CFG_BASE, sizeof(me_ss_slave_cfg_block));
  tileAddress.clearRange(ME_SS_SLAVE_SLOT_BASE, sizeof(me_ss_slave_slot_block));

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
  bool foundBdPacket = false;
  int packetType = 0;
  int packetID = 0;
  bool foundBd = false;
  int lenA = 0;
  int lenB = 0;
  unsigned bytesA = 0;
  unsigned bytesB = 0;
  int offsetA = 0;
  int offsetB = 0;
  uint64_t BaseAddrA = 0;
  uint64_t BaseAddrB = 0;
  bool hasA = false;
  bool hasB = false;
  std::string bufA = "0";
  std::string bufB = "0";
  uint32_t AbMode = disable;
  uint32_t FifoMode = disable; // FIXME: when to enable FIFO mode?
};

static BDInfo getBDInfo(Block &block) {
  BDInfo bdInfo;
  for (auto op : block.getOps<DMABDOp>()) {
    bdInfo.foundBd = true;
    auto bufferType = op.getBuffer().getType().cast<::mlir::MemRefType>();

    if (op.isA()) {
      bdInfo.BaseAddrA = op.getBufferOp().address();
      bdInfo.lenA = op.getLenValue();
      bdInfo.bytesA = bufferType.getElementTypeBitWidth() / 8u;
      bdInfo.offsetA = op.getOffsetValue();
      bdInfo.bufA = "XAIEDMA_TILE_BD_ADDRA";
      bdInfo.hasA = true;
    }

    if (op.isB()) {
      bdInfo.BaseAddrB = op.getBufferOp().address();
      bdInfo.lenB = op.getLenValue();
      bdInfo.bytesB = bufferType.getElementTypeBitWidth() / 8u;
      bdInfo.offsetB = op.getOffsetValue();
      bdInfo.bufB = "XAIEDMA_TILE_BD_ADDRB";
      bdInfo.hasB = true;
    }
  }
  return bdInfo;
}

static void configure_dmas(DeviceOp &targetOp) {
  Field<1> dmaChannelReset;
  Field<0> dmaChannelEnable;

  for (auto memOp : targetOp.getOps<MemOp>()) {
    TileAddress tile{memOp};
    LLVM_DEBUG(llvm::dbgs() << "DMA: tile=" << memOp.getTile());
    // Clear the CTRL and QUEUE registers for the DMA channels.
    for (auto chNum = 0u; chNum < DMA_S2MM_CHANNEL_COUNT; ++chNum) {
      write32({tile, reg_dma_s2mm_ctrl(chNum)},
              dmaChannelReset(disable) | dmaChannelEnable(disable));
      write32({tile, reg_dma_s2mm_queue(chNum)}, 0);
    }
    for (auto chNum = 0u; chNum < DMA_MM2S_CHANNEL_COUNT; ++chNum) {
      write32({tile, reg_dma_mm2s_ctrl(chNum)},
              dmaChannelReset(disable) | dmaChannelEnable(disable));
      write32({tile, reg_dma_mm2s_queue(chNum)}, 0);
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
        bdInfo.AbMode = enable;
        if (bdInfo.lenA != bdInfo.lenB)
          llvm::errs() << "ABmode must have matching lengths.\n";
        if (bdInfo.bytesA != bdInfo.bytesB)
          llvm::errs() << "ABmode must have matching element data types.\n";
      }

      int acqValue = 0, relValue = 0;
      auto acqEnable = disable;
      auto relEnable = disable;
      std::optional<int> lockID = std::nullopt;

      for (auto op : block.getOps<UseLockOp>()) {
        LockOp lock = dyn_cast<LockOp>(op.getLock().getDefiningOp());
        lockID = lock.getLockIDValue();
        if (op.acquire()) {
          acqEnable = enable;
          acqValue = op.getLockValue();
        } else {
          relEnable = enable;
          relValue = op.getLockValue();
        }
      }

      // We either
      //  a. went thru the loop once (`lockID` should be something) xor
      //  b. did not enter the loop (the enables should be both disable)
      assert(lockID.has_value() ^
                 (acqEnable == disable and relEnable == disable) &&
             "lock invariants not satisfied");

      for (auto op : block.getOps<DMABDPACKETOp>()) {
        bdInfo.foundBdPacket = true;
        bdInfo.packetType = op.getPacketType();
        bdInfo.packetID = op.getPacketID();
      }

      auto bdNum = blockMap[&block];
      me_reg_dma_bd bdData;
      if (bdInfo.foundBd) {
        Field<25, 22> bdAddressLockID;
        Field<21> bdAddressReleaseEnable;
        Field<20> bdAddressReleaseValue;
        Field<19> bdAddressReleaseValueEnable;
        Field<18> bdAddressAcquireEnable;
        Field<17> bdAddressAcquireValue;
        Field<16> bdAddressAcquireValueEnable;

        if (bdInfo.hasA) {
          bdData.addr_a = bdAddressLockID(lockID.value()) |
                          bdAddressReleaseEnable(relEnable) |
                          bdAddressAcquireEnable(acqEnable);
          if (relValue != 0xFFu)
            bdData.addr_a |= bdAddressReleaseValueEnable(true) |
                             bdAddressReleaseValue(relValue);
          if (acqValue != 0xFFu)
            bdData.addr_a |= bdAddressAcquireValueEnable(true) |
                             bdAddressAcquireValue(acqValue);
        }
        if (bdInfo.hasB)
          llvm::report_fatal_error("bdInfo.hasB not supported");

        auto addr_a = bdInfo.BaseAddrA + bdInfo.offsetA;
        auto addr_b = bdInfo.BaseAddrB + bdInfo.offsetB;

        Field<12, 0> bdAddressBase, bdControlLength;
        Field<30> bdControlABMode;
        Field<28> bdControlFifo;

        bdData.addr_a |= bdAddressBase(addr_a >> 2u);
        bdData.addr_b |= bdAddressBase(addr_b >> 2u);
        bdData.control |= bdControlLength(bdInfo.lenA - 1) |
                          bdControlFifo(bdInfo.FifoMode) |
                          bdControlABMode(bdInfo.AbMode);

        if (block.getNumSuccessors() > 0) {
          // should have only one successor block
          assert(block.getNumSuccessors() == 1 &&
                 "block.getNumSuccessors() != 1");
          auto *nextBlock = block.getSuccessors()[0];
          auto nextBdNum = blockMap[nextBlock];

          Field<16, 13> bdControlNextBD;
          Field<17> bdControlEnableNextBD;

          bdData.control |= bdControlEnableNextBD(nextBdNum != 0xFFu) |
                            bdControlNextBD(nextBdNum);
        }

        if (bdInfo.foundBdPacket) {
          Field<14, 12> bdPacketType;
          Field<4, 0> bdPacketID;
          Field<27> bdControlEnablePacket;

          bdData.packet =
              bdPacketID(bdInfo.packetID) | bdPacketType(bdInfo.packetType);
          bdData.control |= bdControlEnablePacket(enable);
        }

        Field<31> bdControlValid;

        assert(bdNum < ME_DMA_BD_COUNT && "bdNum >= ME_DMA_BD_COUNT");
        uint64_t bdOffset = reg_dma_addr_a_bd(bdNum);

        write32({tile, bdOffset}, bdData.addr_a);
        write32({tile, reg_dma_addr_b_bd(bdNum)}, bdData.addr_b);
        write32({tile, reg_dma_2d_x_bd(bdNum)}, bdData.x_2d);
        write32({tile, reg_dma_2d_y_bd(bdNum)}, bdData.y_2d);
        write32({tile, reg_dma_pkt_bd(bdNum)}, bdData.packet);
        write32({tile, reg_dma_int_state_bd(bdNum)}, bdData.interleave);
        write32({tile, reg_dma_ctrl_bd(bdNum)},
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
            write32(Address{tile, reg_dma_mm2s_queue(chNum)},
                    dmaChannelQueueStartBd(bdNum));
            write32({tile, reg_dma_mm2s_ctrl(chNum)},
                    dmaChannelEnable(enable) | dmaChannelReset(disable));
          } else {
            write32(Address{tile, reg_dma_s2mm_queue(chNum)},
                    dmaChannelQueueStartBd(bdNum));
            write32({tile, reg_dma_s2mm_ctrl(chNum)},
                    dmaChannelEnable(enable) | dmaChannelReset(disable));
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

static void configure_switchboxes(DeviceOp &targetOp) {
  for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
    Region &r = switchboxOp.getConnections();
    Block &b = r.front();
    bool isEmpty = b.getOps<ConnectOp>().empty() &&
                   b.getOps<MasterSetOp>().empty() &&
                   b.getOps<PacketRulesOp>().empty();

    // NOTE: may not be needed
    std::set<TileAddress> switchbox_set;
    if (isa<TileOp>(switchboxOp.getTile().getDefiningOp())) {
      if (!isEmpty)
        switchbox_set.emplace(switchboxOp);
    } else if (AIEX::SelectOp sel = dyn_cast<AIEX::SelectOp>(
                   switchboxOp.getTile().getDefiningOp()))
      // TODO: Use XAIEV1 target and translate into write32s
      llvm::report_fatal_error("select op not supported");

    constexpr Field<31> streamEnable;
    constexpr Field<30> streamPacketEnable;
    for (auto connectOp : b.getOps<ConnectOp>()) {
      for (auto tile : switchbox_set) {
        auto slave_port =
            computeSlavePort(connectOp.getSourceBundle(),
                             connectOp.sourceIndex(), tile.isShim());
        auto master_port = computeMasterPort(
            connectOp.getDestBundle(), connectOp.destIndex(), tile.isShim());

        Field<7> streamMasterDropHeader;
        Field<6, 0> streamMasterConfig;

        // Configure master side
        {
          Address address{tile, reg_me_ss_master(master_port)};
          // TODO: `Field::extract(uint32_t)`?
          auto drop_header = (slave_port & 0x80u) >> 7u;
          auto value = streamEnable(true) | streamPacketEnable(false) |
                       streamMasterDropHeader(drop_header) |
                       streamMasterConfig(slave_port);
          assert(value < UINT32_MAX);
          write32(address, value);
        }

        // Configure slave side
        {
          Address address{tile, reg_me_ss_slave_cfg(slave_port)};
          write32(address, streamEnable(true) | streamPacketEnable(false));
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

          const auto dropHeader = connectOp.getDestBundle() == WireBundle::DMA;
          auto config = streamMasterDropHeader(dropHeader) |
                        (mask << STREAM_SWITCH_MSEL_SHIFT) |
                        (arbiter << STREAM_SWITCH_ARB_SHIFT);
          Address dest{tile, reg_me_ss_master(master_port)};
          write32(dest, streamEnable(enable) | streamPacketEnable(enable) |
                            streamMasterDropHeader(dropHeader) |
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

        for (auto tile : switchbox_set) {
          auto slavePort =
              computeSlavePort(connectOp.getSourceBundle(),
                               connectOp.sourceIndex(), tile.isShim());
          write32({tile, reg_me_ss_slave_cfg(slavePort)},
                  streamEnable(enable) | streamPacketEnable(enable));

          Field<28, 24> streamSlotId;
          Field<20, 16> streamSlotMask;
          Field<8> streamSlotEnable;
          Field<5, 4> streamSlotMSel;
          Field<2, 0> streamSlotArbit;

          auto config = streamSlotId(slotOp.valueInt()) |
                        streamSlotMask(slotOp.maskInt()) |
                        streamSlotEnable(enable) | streamSlotMSel(msel) |
                        streamSlotArbit(arbiter);
          write32({tile, reg_me_ss_slave_slot(slavePort, slot)}, config);
          slot++;
        }
      }
    }
  }

  const auto inputMaskFor = [](WireBundle bundle, uint8_t shiftAmt) {
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
                          inputMaskFor(connectOp.getDestBundle(), shiftAmt));
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
        write32(addr, currentMask |
                          inputMaskFor(connectOp.getSourceBundle(), shiftAmt));
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

/*
        Convert memory address to index

        Used to look up register/region name
*/
static uint8_t sec_addr2index(uint64_t in) {
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
static void group_sections(std::vector<Section *> &sections) {
  uint64_t last_addr = 0;
  Section *section = nullptr;

  for (auto write : mem_writes) {
    if (write.first != last_addr + 4) {
      if (section)
        sections.push_back(section);
      section = new Section(write.first);
      LLVM_DEBUG(llvm::dbgs() << "Starting new section @ "
                              << llvm::format("0x%lx (last=0x%lx)\n",
                                              write.first, last_addr));
    }
    assert(section && "section is null");
    section->add_data(write.second);
    last_addr = write.first;
  }

  sections.push_back(section);
}

/*
   Add a string to the section header string table and return the offset of
   the start of the string
*/
static size_t add_string(Elf_Scn *scn, const char *str) {
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

Elf_Data *section_add_data(Elf_Scn *scn, const Section *section) {
  size_t size = section->get_length();
  if (TEST_AIRBIN)
    size = 4;
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
  memcpy(buf, section->get_data(), size);

  return data;
}

mlir::LogicalResult AIETranslateToAirbin(mlir::ModuleOp module,
                                         const std::string &outputFilename,
                                         const std::string &coreFilesDir,
                                         bool testAirBin) {

  TEST_AIRBIN = testAirBin;

  int tmp_elf_fd;
  Elf *outelf;
  GElf_Ehdr ehdr_mem;
  GElf_Ehdr *ehdr;
  GElf_Shdr *shdr;
  GElf_Shdr shdr_mem;
  char empty_str[] = "";
  char strtab_name[] = ".shstrtab";
  std::vector<Section *> sections;

  DenseMap<std::pair<int, int>, Operation *> tiles;
  DenseMap<Operation *, CoreOp> cores;
  DenseMap<Operation *, MemOp> mems;
  DenseMap<std::pair<Operation *, int>, LockOp> locks;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;
  DenseMap<Operation *, SwitchboxOp> switchboxes;

  if (module.getOps<DeviceOp>().empty()) {
    LLVM_DEBUG(llvm::dbgs() << "no device ops found");
    return success();
  }

  DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

  // Write the initial configuration for every tile specified in the MLIR.
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "CC: tile=" << tileOp.getTileID());
    if (tileOp.isShimTile())
      config_shim_tile(tileOp);
    else
      config_ME_tile(tileOp, coreFilesDir);
  }
  configure_switchboxes(targetOp);
  configure_dmas(targetOp);

  group_sections(sections);

  LLVM_DEBUG(llvm::dbgs() << llvm::format("mem_writes: %lu in %lu sections\n",
                                          mem_writes.size(), sections.size()));

  elf_version(EV_CURRENT);
  tmp_elf_fd =
      open(outputFilename.c_str(), O_RDWR | O_CREAT | O_TRUNC, DEFFILEMODE);
  outelf = elf_begin(tmp_elf_fd, ELF_C_WRITE, nullptr);

  if (!gelf_newehdr(outelf, ELFCLASS64))
    llvm::report_fatal_error(llvm::Twine("Error creating ELF64 header: ") +
                             elf_errmsg(-1));

  ehdr = gelf_getehdr(outelf, &ehdr_mem);
  if (!ehdr)
    llvm::report_fatal_error(llvm::Twine("cannot get ELF header: ") +
                             elf_errmsg(-1));

  // Initialize header.
  ehdr->e_ident[EI_DATA] = ELFDATA2LSB;
  ehdr->e_ident[EI_OSABI] = ELFOSABI_GNU;
  ehdr->e_type = ET_NONE;
  ehdr->e_machine = EM_AMDAIR;
  ehdr->e_version = EV_CURRENT;
  if (gelf_update_ehdr(outelf, ehdr) == 0)
    llvm::report_fatal_error(llvm::Twine("cannot update ELF header: ") +
                             elf_errmsg(-1));

  // Create new section for the 'section header string table'
  Elf_Scn *shstrtab_scn = elf_newscn(outelf);
  if (!shstrtab_scn)
    llvm::report_fatal_error(
        llvm::Twine("cannot create new shstrtab section: ") + elf_errmsg(-1));

  // the first entry in the string table must be a NULL string
  add_string(shstrtab_scn, empty_str);

  shdr = gelf_getshdr(shstrtab_scn, &shdr_mem);
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
  shdr->sh_name = add_string(shstrtab_scn, strtab_name);

  // add all the AIRBIN-specific section names up front and index them
  for (uint8_t sec_idx = SEC_IDX_SSMAST; sec_idx < SEC_IDX_MAX; sec_idx++)
    sec_name_offset[sec_idx] = add_string(shstrtab_scn, sec_name_str[sec_idx]);
  sec_name_offset[SEC_IDX_NULL] = 0;

  // We have to store the section strtab index in the ELF header so sections
  // have actual names.
  int ndx = elf_ndxscn(shstrtab_scn);
  ehdr->e_shstrndx = ndx;

  if (!gelf_update_ehdr(outelf, ehdr))
    llvm::report_fatal_error(llvm::Twine("cannot update ELF header: ") +
                             elf_errmsg(-1));

  // Finished new shstrtab section, update the header.
  if (!gelf_update_shdr(shstrtab_scn, shdr))
    llvm::report_fatal_error(
        llvm::Twine("cannot update new shstrtab section header: ") +
        elf_errmsg(-1));

  // output the rest of the sections
  for (const Section *section : sections) {
    uint64_t addr = section->get_addr();
    Elf_Scn *scn = elf_newscn(outelf);
    if (!scn)
      llvm::report_fatal_error(llvm::Twine("cannot create new ") +
                               sec_name_str[sec_addr2index(addr)] +
                               "section: " + elf_errmsg(-1));

    shdr = gelf_getshdr(scn, &shdr_mem);
    if (!shdr)
      llvm::report_fatal_error(llvm::Twine("cannot get header for ") +
                               sec_name_str[sec_addr2index(addr)] +
                               "section: " + elf_errmsg(-1));

    Elf_Data *data = section_add_data(scn, section);

    shdr->sh_type = SHT_PROGBITS;
    shdr->sh_flags = SHF_ALLOC;
    shdr->sh_addr = section->get_addr();
    shdr->sh_link = SHN_UNDEF;
    shdr->sh_info = SHN_UNDEF;
    shdr->sh_addralign = 1;
    shdr->sh_entsize = 0;
    shdr->sh_size = data->d_size;
    shdr->sh_name = sec_name_offset[sec_addr2index(addr)];

    if (!gelf_update_shdr(scn, shdr))
      llvm::report_fatal_error(llvm::Twine("cannot update section header: ") +
                               elf_errmsg(-1));
  }

  // Write everything to disk.
  if (elf_update(outelf, ELF_C_WRITE) < 0)
    llvm::report_fatal_error(llvm::Twine("failure in elf_update: ") +
                             elf_errmsg(-1));

  // close the elf object
  elf_end(outelf);

  // copy the file to the compiler's output stream
  close(tmp_elf_fd);

  return success();
}
} // namespace xilinx::AIE
