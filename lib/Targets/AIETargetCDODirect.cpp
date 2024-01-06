//===- AIETargetCDODirect.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"
#include "aie/Targets/cdo_driver.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstddef> // size_t
#include <cstdint> // uint
#include <cstdlib> // calloc
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>

#ifndef NDEBUG
#define XAIE_DEBUG
#endif

extern "C" {
#include "xaiengine/xaie_core.h"
#include "xaiengine/xaie_dma.h"
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaie_locks.h"
#include "xaiengine/xaie_plif.h"
#include "xaiengine/xaie_ss.h"
#include "xaiengine/xaiegbl.h"
#include "xaiengine/xaiegbl_defs.h"
}

#define DEBUG_TYPE "aie-generate-cdo-direct"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define AIERC_STR(x) x, #x
static const std::map<AieRC, std::string> AIERCTOSTR = {
    {AIERC_STR(XAIE_OK)},
    {AIERC_STR(XAIE_ERR)},
    {AIERC_STR(XAIE_INVALID_DEVICE)},
    {AIERC_STR(XAIE_INVALID_RANGE)},
    {AIERC_STR(XAIE_INVALID_ARGS)},
    {AIERC_STR(XAIE_INVALID_TILE)},
    {AIERC_STR(XAIE_ERR_STREAM_PORT)},
    {AIERC_STR(XAIE_INVALID_DMA_TILE)},
    {AIERC_STR(XAIE_INVALID_BD_NUM)},
    {AIERC_STR(XAIE_ERR_OUTOFBOUND)},
    {AIERC_STR(XAIE_INVALID_DATA_MEM_ADDR)},
    {AIERC_STR(XAIE_INVALID_ELF)},
    {AIERC_STR(XAIE_CORE_STATUS_TIMEOUT)},
    {AIERC_STR(XAIE_INVALID_CHANNEL_NUM)},
    {AIERC_STR(XAIE_INVALID_LOCK)},
    {AIERC_STR(XAIE_INVALID_DMA_DIRECTION)},
    {AIERC_STR(XAIE_INVALID_PLIF_WIDTH)},
    {AIERC_STR(XAIE_INVALID_LOCK_ID)},
    {AIERC_STR(XAIE_INVALID_LOCK_VALUE)},
    {AIERC_STR(XAIE_LOCK_RESULT_FAILED)},
    {AIERC_STR(XAIE_INVALID_DMA_DESC)},
    {AIERC_STR(XAIE_INVALID_ADDRESS)},
    {AIERC_STR(XAIE_FEATURE_NOT_SUPPORTED)},
    {AIERC_STR(XAIE_INVALID_BURST_LENGTH)},
    {AIERC_STR(XAIE_INVALID_BACKEND)},
    {AIERC_STR(XAIE_INSUFFICIENT_BUFFER_SIZE)},
    {AIERC_STR(XAIE_ERR_MAX)}};
#undef AIERC_STR

static const std::map<WireBundle, StrmSwPortType>
    WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE = {
        {WireBundle::Core, StrmSwPortType::CORE},
        {WireBundle::DMA, StrmSwPortType::DMA},
        // missing control from StrmSwPortType
        {WireBundle::FIFO, StrmSwPortType::FIFO},
        {WireBundle::South, StrmSwPortType::SOUTH},
        {WireBundle::West, StrmSwPortType::WEST},
        {WireBundle::North, StrmSwPortType::NORTH},
        {WireBundle::East, StrmSwPortType::EAST},
        // missing PLIO from WireBundle
        // missing NOC from WireBundle
        {WireBundle::Trace, StrmSwPortType::TRACE},
};

// So that we can use the pattern if(auto r = TRY_XAIE_API...) { // r is nonzero
// }
static_assert(XAIE_OK == 0);

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                                     \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << "\n");           \
    if (auto r = API(__VA_ARGS__))                                             \
      report_fatal_error(llvm::Twine(#API " failed with ") +                   \
                         AIERCTOSTR.at(r));                                    \
  } while (0)

#define TRY_XAIE_API_EMIT_ERROR(OP, API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << "\n");           \
    if (auto r = API(__VA_ARGS__))                                             \
      return OP.emitOpError() << #API " failed with " << AIERCTOSTR.at(r);     \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << "\n");           \
    if (auto r = API(__VA_ARGS__)) {                                           \
      llvm::errs() << #API " failed with " << AIERCTOSTR.at(r);                \
      return failure();                                                        \
    }                                                                          \
  } while (0)

auto ps = std::filesystem::path::preferred_separator;

#define HW_GEN XAIE_DEV_GEN_AIEML
#define XAIE_NUM_ROWS 6
#define XAIE_NUM_COLS 5
#define XAIE_BASE_ADDR 0x40000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_MEM_TILE_NUM_ROWS 1
#define XAIE_AIE_TILE_ROW_START 2
#define XAIE_AIE_TILE_NUM_ROWS 4
#define XAIE_PARTITION_BASE_ADDR 0x0

#define NPI_ADDR 0x0
#define NUM_LOCKS 16
#define EVEN_BD_NUM_START 0
#define ODD_BD_NUM_START 24
#define ACQ_LOCK_ID_INCR 64
#define REL_LOCK_ID_INCR 64
#define BASE_ADDR_A_INCR 0x80000
#define PARTITION_START_COL 1
#define PARTITION_NUM_COLS 1

namespace xilinx::AIE {

struct AIEControl {
  XAie_Config configPtr;
  XAie_DevInst devInst;

  AIEControl(uint8_t hwGen = XAIE_DEV_GEN_AIEML,
             uint64_t xaieBaseAddr = XAIE_BASE_ADDR,
             uint8_t xaieColShift = XAIE_COL_SHIFT,
             uint8_t xaieRowShift = XAIE_ROW_SHIFT,
             uint8_t xaieNumCols = XAIE_NUM_COLS,
             uint8_t xaieNumRows = XAIE_NUM_ROWS,
             uint8_t xaieShimRow = XAIE_SHIM_ROW,
             uint8_t xaieMemTileRowStart = XAIE_MEM_TILE_ROW_START,
             uint8_t xaieMemTileNumRows = XAIE_MEM_TILE_NUM_ROWS,
             uint8_t xaieAieTileRowStart = XAIE_AIE_TILE_ROW_START,
             uint8_t xaieAieTileNumRows = XAIE_AIE_TILE_NUM_ROWS,
             uint64_t xaiePartitionBaseAddr = XAIE_PARTITION_BASE_ADDR,
             uint64_t npiAddr = NPI_ADDR)
      : configPtr({
            .AieGen = hwGen,
            .BaseAddr = xaieBaseAddr,
            .ColShift = xaieColShift,
            .RowShift = xaieRowShift,
            .NumRows = xaieNumRows,
            .NumCols = xaieNumCols,
            .ShimRowNum = xaieShimRow,
            .MemTileRowStart = xaieMemTileRowStart,
            .MemTileNumRows = xaieMemTileNumRows,
            .AieTileRowStart = xaieAieTileRowStart,
            .AieTileNumRows = xaieAieTileNumRows,
            .PartProp = {},
        }) {
    // Quoting: The instance of a device must be always declared using this
    //		macro. In future, the same macro will be expanded to allocate
    //		more memory from the user application for resource management.
    XAie_InstDeclare(_devInst, &configPtr);
    devInst = _devInst;
    // TODO(max): what is the "partition"?
    TRY_XAIE_API_FATAL_ERROR(XAie_SetupPartitionConfig, &devInst,
                             xaiePartitionBaseAddr, PARTITION_START_COL,
                             PARTITION_NUM_COLS);
    TRY_XAIE_API_FATAL_ERROR(XAie_CfgInitialize, &devInst, &configPtr);
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst, XAIE_IO_BACKEND_CDO);
    TRY_XAIE_API_FATAL_ERROR(XAie_UpdateNpiAddr, &devInst, npiAddr);
  }

  LogicalResult addErrorHandlingToCDO() {
    TRY_XAIE_API_LOGICAL_RESULT(XAie_ErrorHandlingInit, &devInst);
    return success();
  }

  LogicalResult addAieElfToCDO(uint8_t col, uint8_t row,
                               const StringRef elfPath) {
    // loadSym: Load symbols from .map file. This argument is not used when
    // __AIESIM__ is not defined.
    bool loadSym = false;
    TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadElf, &devInst, XAie_TileLoc(col, row),
                                elfPath.str().c_str(), loadSym);
    return success();
  }

  LogicalResult addAieElfsToCDO(DeviceOp &targetOp,
                                const StringRef workDirPath) {
    for (auto tileOp : targetOp.getOps<TileOp>())
      if (tileOp.isShimNOCorPLTile()) {
        // Resets no needed with V2 kernel driver
      } else {
        int col = tileOp.colIndex();
        int row = tileOp.rowIndex();
        if (auto coreOp = tileOp.getCoreOp()) {
          std::string fileName;
          if (auto fileAttr = coreOp.getElfFile())
            fileName = fileAttr->str();
          else
            fileName = std::string("core_") + std::to_string(col) + "_" +
                       std::to_string(row) + ".elf";
          if (failed(
                  addAieElfToCDO(col, row, workDirPath.str() + ps + fileName)))
            return failure();
        }
      }
    return success();
  }

  LogicalResult addInitConfigToCDO(DeviceOp &targetOp) {
    for (auto tileOp : targetOp.getOps<TileOp>()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      if (!tileOp.isShimTile() && tileOp.getCoreOp()) {
        TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreReset, &devInst,
                                XAie_TileLoc(col, row));
        TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreUnreset, &devInst,
                                XAie_TileLoc(col, row));
        // Set locks to zero
        for (uint8_t l = 0; l < NUM_LOCKS; l++)
          TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_LockSetValue, &devInst,
                                  XAie_TileLoc(col, row), XAie_LockInit(l, 0));
      }
    }

    // Set locks with explicit initializers
    for (auto lockOp : targetOp.getOps<LockOp>()) {
      auto tileOp = lockOp.getTileOp();
      assert(lockOp.getLockID() && lockOp.getInit() &&
             "locks must be fully initialized");
      TRY_XAIE_API_EMIT_ERROR(
          lockOp, XAie_LockSetValue, &devInst,
          XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex()),
          XAie_LockInit(*lockOp.getLockID(), *lockOp.getInit()));
    }

    auto &targetModel = targetOp.getTargetModel();
    auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
    llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
    for (TileElement memOp : memOps) {
      int col = memOp.getTileID().col;
      int row = memOp.getTileID().row;

      DenseMap<Block *, int> blockMap;
      DenseMap<Block *, int> channelMap;

      for (Block &block : memOp.getOperation()->getRegion(0))
        for (auto op : block.getOps<DMAStartOp>()) {
          int chNum = op.getChannelIndex();
          channelMap[&block] = chNum;
          Block *dest = op.getDest();
          while (dest) {
            channelMap[dest] = chNum;
            dest = dest->getSuccessors()[0];
            if (channelMap.count(dest))
              dest = nullptr;
          }
        }

      // Assign each block a BD number
      int evenBdNum = EVEN_BD_NUM_START;
      int oddBdNum = ODD_BD_NUM_START;
      for (Block &block : memOp.getOperation()->getRegion(0)) {
        if (block.getOps<DMABDOp>().empty())
          continue;
        assert(channelMap.count(&block));
        if (targetModel.isMemTile(col, row) && (channelMap[&block] & 1))
          blockMap[&block] = oddBdNum++;
        else
          blockMap[&block] = evenBdNum++;
      }

      for (Block &block : memOp.getOperation()->getRegion(0)) {
        bool foundBd = false;
        int lenA = 0, bytesA = 0, offsetA = 0, baseAddrA = 0;
        // StringRef FifoMode = disable; // FIXME: when to enable FIFO mode?
        std::optional<ArrayRef<BDDimLayoutAttr>> dims;
        for (auto op : block.getOps<DMABDOp>()) {
          foundBd = true;
          ShapedType bufferType =
              op.getBuffer().getType().cast<::mlir::MemRefType>();
          baseAddrA =
              cast<AIE::BufferOp>(op.getBuffer().getDefiningOp()).address();
          lenA = op.getLenValue();
          bytesA = bufferType.getElementTypeBitWidth() / 8;
          offsetA = op.getOffsetValue();
          dims = op.getDimensions();
        }

        int acqValue = 0, relValue = 0, acqLockId = 0, relLockId = 0;
        for (auto op : block.getOps<UseLockOp>()) {
          LockOp lock = dyn_cast<LockOp>(op.getLock().getDefiningOp());
          if (op.acquire() || op.acquireGE()) {
            acqLockId = lock.getLockIDValue();
            acqValue = op.getLockValue();
            if (op.acquireGE())
              acqValue = -acqValue;
          } else if (op.release()) {
            relLockId = lock.getLockIDValue();
            relValue = op.getLockValue();
          } else
            return memOp.emitError("unsupported lock action");
        }

        if (targetModel.isMemTile(col, row)) {
          acqLockId += ACQ_LOCK_ID_INCR;
          relLockId += REL_LOCK_ID_INCR;
          baseAddrA += BASE_ADDR_A_INCR;
        }

        bool foundBdPacket = false;
        int packetType = 0;
        int packetID = 0;
        for (auto op : block.getOps<DMABDPACKETOp>()) {
          foundBdPacket = true;
          packetType = op.getPacketType();
          packetID = op.getPacketID();
        }

        int bdNum = blockMap[&block];
        if (foundBd) {
          // TODO For now, we are going to name each dma desc with loc and bd
          // which we assume is unique. This is strictly not enforced but in
          // practice, this is true
          XAie_DmaDesc dmaTileBd;
          TRY_XAIE_API_EMIT_ERROR(memOp, XAie_DmaDescInit, &devInst, &dmaTileBd,
                                  XAie_TileLoc(col, row));
          TRY_XAIE_API_EMIT_ERROR(memOp, XAie_DmaSetLock, &dmaTileBd,
                                  XAie_LockInit(acqLockId, acqValue),
                                  XAie_LockInit(relLockId, relValue));
          if (!dims) {
            TRY_XAIE_API_EMIT_ERROR(memOp, XAie_DmaSetAddrLen, &dmaTileBd,
                                    baseAddrA + offsetA, lenA * bytesA);
          } else {
            XAie_DmaTensor dmaTileBdTensor = {};
            dmaTileBdTensor.NumDim = dims->size();
            dmaTileBdTensor.Dim = static_cast<XAie_DmaDimDesc *>(
                calloc(dims->size(), sizeof(XAie_DmaDimDesc)));
            if (!dmaTileBdTensor.Dim)
              return memOp.emitError(
                  "couldn't allocate array of XAie_DmaDimDesc");
            // TODO(max): rethink this?
            for (size_t i = 0; i < dims->size(); i++) {
              // Pass down dimensions in reverse order; in the MLIR, this allows
              // us to specify step sizes/wraps in the same order as we would
              // access a multi-dim C array, with the highest dimension first.
              int j = dims->size() - i - 1;
              // Assume AIE-ML architecture; we assert this above
              // TODO(max): no we don't
              dmaTileBdTensor.Dim[j].AieMlDimDesc = {dims.value()[i].getStep(),
                                                     dims.value()[i].getWrap()};
            }
            // TODO: Probably need special handling for NOC
            // TODO: Might need to adjust step sizes / wraps by -1
            TRY_XAIE_API_EMIT_ERROR(memOp, XAie_DmaSetMultiDimAddr, &dmaTileBd,
                                    &dmaTileBdTensor, baseAddrA + offsetA,
                                    lenA * bytesA);
          }

          if (block.getNumSuccessors()) {
            assert(llvm::range_size(block.getSuccessors()) == 1 &&
                   "should have only one successor block");
            Block *nextBlock = block.getSuccessor(0);
            int nextBdNum = blockMap[nextBlock];
            // TODO Check if br ^end: to disable this?
            TRY_XAIE_API_EMIT_ERROR(memOp, XAie_DmaSetNextBd, &dmaTileBd,
                                    nextBdNum,
                                    /* enableNextBd */ 1);
          }

          if (foundBdPacket)
            TRY_XAIE_API_EMIT_ERROR(memOp, XAie_DmaSetPkt, &dmaTileBd,
                                    XAie_PacketInit(packetID, packetType));
          TRY_XAIE_API_EMIT_ERROR(memOp, XAie_DmaEnableBd, &dmaTileBd);
          TRY_XAIE_API_EMIT_ERROR(memOp, XAie_DmaWriteBd, &devInst, &dmaTileBd,
                                  XAie_TileLoc(col, row), bdNum);
        }
      }

      for (Block &block : memOp.getOperation()->getRegion(0))
        for (auto op : block.getOps<DMAStartOp>()) {
          int bdNum = blockMap[op.getDest()];
          int chNum = op.getChannelIndex();
          TRY_XAIE_API_EMIT_ERROR(
              memOp, XAie_DmaChannelPushBdToQueue, &devInst,
              XAie_TileLoc(col, row), chNum,
              // TODO hack until physical dialect changes
              op.getChannelDir() == DMAChannelDir::S2MM ? DMA_S2MM : DMA_MM2S,
              bdNum);
          TRY_XAIE_API_EMIT_ERROR(
              memOp, XAie_DmaChannelEnable, &devInst, XAie_TileLoc(col, row),
              chNum,
              // TODO hack until physical dialect changes
              op.getChannelDir() == DMAChannelDir::S2MM ? DMA_S2MM : DMA_MM2S);
        }
    }

    // StreamSwitch (switchbox) configuration
    for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
      XAie_LocType tileLoc = XAie_TileLoc(switchboxOp.getTileOp().getCol(),
                                          switchboxOp.getTileOp().getRow());
      if (switchboxOp.rowIndex() == 0) {
        // FIXME hack for TCT routing
        // TODO Support both channels
        TRY_XAIE_API_EMIT_ERROR(switchboxOp, XAie_StrmConnCctEnable, &devInst,
                                tileLoc, CTRL,
                                /*SlvPortNum*/ 0, SOUTH,
                                /*MstrPortNum*/ 0);
        // configure DMA_<S2MM/MM2S>_<chNum>_Ctrl register
        for (int chNum = 0; chNum <= 1; ++chNum) {
          XAie_DmaChannelDesc dmaChannelDescInst;
          TRY_XAIE_API_EMIT_ERROR(switchboxOp, XAie_DmaChannelDescInit,
                                  &devInst, &dmaChannelDescInst, tileLoc);
          TRY_XAIE_API_EMIT_ERROR(switchboxOp, XAie_DmaChannelSetControllerId,
                                  &dmaChannelDescInst,
                                  /*ControllerId*/ 0);
          TRY_XAIE_API_EMIT_ERROR(switchboxOp, XAie_DmaWriteChannel, &devInst,
                                  &dmaChannelDescInst, tileLoc, chNum,
                                  DMA_S2MM);
        }

        TRY_XAIE_API_EMIT_ERROR(switchboxOp, XAie_AieToPlIntfEnable, &devInst,
                                tileLoc, /*PortNum*/ 0, PLIF_WIDTH_32);
      }

      Block &b = switchboxOp.getConnections().front();
      for (auto connectOp : b.getOps<ConnectOp>())
        TRY_XAIE_API_EMIT_ERROR(
            switchboxOp, XAie_StrmConnCctEnable, &devInst, tileLoc,
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
            connectOp.sourceIndex(),
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
            connectOp.destIndex());

      for (auto connectOp : b.getOps<MasterSetOp>()) {
        int mask = 0;
        int arbiter = -1;

        for (auto val : connectOp.getAmsels()) {
          AMSelOp amsel = dyn_cast<AMSelOp>(val.getDefiningOp());
          arbiter = amsel.arbiterIndex();
          int msel = amsel.getMselValue();
          mask |= (1 << msel);
        }

        bool isdma = connectOp.getDestBundle() == WireBundle::DMA;
        // assume a connection going south from row zero gets wired to shimdma
        // by a shimmux. TODO: fix the assumption
        if (!isdma && (switchboxOp.rowIndex() == 0))
          isdma = connectOp.getDestBundle() == WireBundle::South;
        // Flag for overriding DROP_HEADER. TODO: Formalize this in tablegen
        isdma &= !connectOp->hasAttr("keep_pkt_header");
        TRY_XAIE_API_EMIT_ERROR(
            connectOp, XAie_StrmPktSwMstrPortEnable, &devInst, tileLoc,
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
            connectOp.destIndex(),
            isdma ? XAIE_SS_PKT_DROP_HEADER : XAIE_SS_PKT_DONOT_DROP_HEADER,
            arbiter, mask);
      }

      for (auto connectOp : b.getOps<PacketRulesOp>()) {
        int slot = 0;
        Block &block = connectOp.getRules().front();
        for (auto slotOp : block.getOps<PacketRuleOp>()) {
          AMSelOp amselOp =
              dyn_cast<AMSelOp>(slotOp.getAmsel().getDefiningOp());
          int arbiter = amselOp.arbiterIndex();
          int msel = amselOp.getMselValue();
          TRY_XAIE_API_EMIT_ERROR(
              connectOp, XAie_StrmPktSwSlavePortEnable, &devInst, tileLoc,
              WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
              connectOp.sourceIndex());
          // TODO Need to better define packet id,type used here
          TRY_XAIE_API_EMIT_ERROR(
              connectOp, XAie_StrmPktSwSlaveSlotEnable, &devInst, tileLoc,
              WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
              connectOp.sourceIndex(), slot,
              XAie_PacketInit(slotOp.valueInt(), /*PktType*/ 0),
              slotOp.maskInt(), msel, arbiter);
          slot++;
        }
      }
    }

    for (auto muxOp : targetOp.getOps<ShimMuxOp>()) {
      // NOTE ShimMux always connects from the south as directions are
      // defined relative to the tile stream switch.
      Block &b = muxOp.getConnections().front();
      for (auto connectOp : b.getOps<ConnectOp>()) {
        // demux!
        if (connectOp.getSourceBundle() == WireBundle::North)
          TRY_XAIE_API_EMIT_ERROR(muxOp, XAie_EnableAieToShimDmaStrmPort,
                                  &devInst,
                                  XAie_TileLoc(muxOp.getTileOp().getCol(),
                                               muxOp.getTileOp().getRow()),
                                  connectOp.sourceIndex());
        // mux
        if (connectOp.getDestBundle() == WireBundle::North)
          TRY_XAIE_API_EMIT_ERROR(muxOp, XAie_EnableShimDmaToAieStrmPort,
                                  &devInst,
                                  XAie_TileLoc(muxOp.getTileOp().getCol(),
                                               muxOp.getTileOp().getRow()),
                                  connectOp.destIndex());
      }
    }

    for (auto switchboxOp : targetOp.getOps<ShimSwitchboxOp>()) {
      Block &b = switchboxOp.getConnections().front();
      for (auto connectOp : b.getOps<ConnectOp>())
        TRY_XAIE_API_EMIT_ERROR(
            switchboxOp, XAie_StrmConnCctEnable, &devInst,
            XAie_TileLoc(switchboxOp.getCol(), 0),
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
            connectOp.sourceIndex(),
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
            connectOp.destIndex());
    }
    return success();
  }

  LogicalResult addCoreEnableToCDO(DeviceOp &targetOp) {
    // Start execution of all the cores.
    for (auto tileOp : targetOp.getOps<TileOp>())
      if (!tileOp.isShimTile() && tileOp.getCoreOp())
        TRY_XAIE_API_EMIT_ERROR(
            targetOp, XAie_CoreEnable, &devInst,
            XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex()));
    return success();
  }
};
} // namespace xilinx::AIE

void initializeCDOGenerator(byte_ordering endianness, bool axiDebug) {
  // Enables AXI-MM prints for configs being added in CDO
  if (axiDebug)
    EnAXIdebug();
  setEndianness(endianness);
};

LogicalResult generateCDOBinary(const StringRef outputPath,
                                const std::function<LogicalResult()> &cb) {
  startCDOFileStream(outputPath.str().c_str());
  FileHeader();
  if (failed(cb()))
    return failure();
  configureHeader();
  endCurrentCDOFileStream();
  return success();
}

LogicalResult generateCDOBinariesSeparately(AIEControl &ctl,
                                            const StringRef workDirPath,
                                            DeviceOp &targetOp) {
  if (failed(generateCDOBinary(
          workDirPath.str() + ps + "aie_cdo_error_handling.bin",
          std::bind(&AIEControl::addErrorHandlingToCDO, ctl))))
    return failure();

  if (failed(generateCDOBinary(workDirPath.str() + ps + "aie_cdo_elfs.bin",
                               [&ctl, &targetOp, &workDirPath] {
                                 return ctl.addAieElfsToCDO(targetOp,
                                                            workDirPath);
                               })))
    return failure();

  if (failed(generateCDOBinary(
          workDirPath.str() + ps + "aie_cdo_init.bin",
          [&ctl, &targetOp] { return ctl.addInitConfigToCDO(targetOp); })))
    return failure();

  if (failed(generateCDOBinary(
          workDirPath.str() + ps + "aie_cdo_enable.bin",
          [&ctl, &targetOp] { return ctl.addCoreEnableToCDO(targetOp); })))
    return failure();

  return success();
}

LogicalResult generateCDOUnified(AIEControl &ctl, const StringRef workDirPath,
                                 DeviceOp &targetOp) {
  return generateCDOBinary(
      workDirPath.str() + ps + "aie_cdo.bin", [&ctl, &targetOp, &workDirPath] {
        if (failed(ctl.addErrorHandlingToCDO()))
          return failure();
        if (failed(ctl.addAieElfsToCDO(targetOp, workDirPath)))
          return failure();
        if (failed(ctl.addInitConfigToCDO(targetOp)))
          return failure();
        if (failed(ctl.addCoreEnableToCDO(targetOp)))
          return failure();
        return success();
      });
}

// Not sure why but defining this with xilinx::AIE will create a duplicate
// symbol in libAIETargets.a that then doesn't actually match the header?
namespace xilinx::AIE {
LogicalResult AIETranslateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                                      byte_ordering endianness,
                                      bool emitUnified, bool axiDebug) {
  auto devOps = m.getOps<DeviceOp>();
  assert(llvm::range_size(devOps) == 1 &&
         "only exactly 1 device op supported.");
  DeviceOp targetOp = *devOps.begin();
  AIEControl ctl;
  initializeCDOGenerator(endianness, axiDebug);
  if (emitUnified)
    return generateCDOUnified(ctl, workDirPath, targetOp);
  return generateCDOBinariesSeparately(ctl, workDirPath, targetOp);
}
} // namespace xilinx::AIE
