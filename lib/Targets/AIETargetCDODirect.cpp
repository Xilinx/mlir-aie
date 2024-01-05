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
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstddef> // size_t
#include <cstdlib> // calloc
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <stdint.h> // uint
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
using namespace xilinx::AIEX;

// So that we can use the pattern if(auto r = ...) { // r is nonzero }
static_assert(XAIE_OK == 0);

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

#define TRY_XAIE_API(API, ...)                                                 \
  if (auto r = API(__VA_ARGS__))                                               \
  report_fatal_error(llvm::Twine(#API " failed with ") + AIERCTOSTR.at(r))

auto ps = std::filesystem::path::preferred_separator;

#define NUM_LOCKS 16
#define EVEN_BD_NUM_START 0
#define ODD_BD_NUM_START 24
#define ACQ_LOCK_ID_INCR 64
#define REL_LOCK_ID_INCR 64
#define BASE_ADDR_A_INCR 0x80000

namespace xilinx::AIE {

struct AIEControl {
  XAie_Config configPtr;
  XAie_DevInst devInst;

  AIEControl(uint8_t hwGen = XAIE_DEV_GEN_AIEML,
             uint64_t xaieBaseAddr = 0x40000000, uint8_t xaieColShift = 25,
             uint8_t xaieRowShift = 20, uint8_t xaieNumCols = 5,
             uint8_t xaieNumRows = 6, uint8_t xaieShimRow = 0,
             uint8_t xaieMemTileRowStart = 1, uint8_t xaieMemTileNumRows = 1,
             uint8_t xaieAieTileRowStart = 2, uint8_t xaieAieTileNumRows = 4,
             uint64_t xaiePartitionBaseAddr = 0x0, uint64_t npiAddr = 0x0)
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
            .PartProp = {0},
        }) {
    // Quoting: The instance of a device must be always declared using this
    //		macro. In future, the same macro will be expanded to allocate
    //		more memory from the user application for resource management.
    XAie_InstDeclare(_devInst, &configPtr); // Declare global device instance
    devInst = _devInst;
    // TODO(max): what is the "partition"?
    TRY_XAIE_API(XAie_SetupPartitionConfig, &devInst, xaiePartitionBaseAddr,
                 /*PartStartCol=*/1,
                 /*PartNumCols=*/1);
    TRY_XAIE_API(XAie_CfgInitialize, &devInst, &configPtr);
    TRY_XAIE_API(XAie_SetIOBackend, &devInst, XAIE_IO_BACKEND_CDO);
    TRY_XAIE_API(XAie_UpdateNpiAddr, &devInst, npiAddr);
  }

  void addErrorHandlingToCDO() {
    TRY_XAIE_API(XAie_ErrorHandlingInit, &devInst);
  }

  void addAieElfToCDO(uint8_t col, uint8_t row, const std::string &elfPath) {
    // loadSym: Load symbols from .map file. This argument is not used when
    // __AIESIM__ is not defined.
    bool loadSym = false;
    TRY_XAIE_API(XAie_LoadElf, &devInst, XAie_TileLoc(col, row),
                 elfPath.c_str(), loadSym);
  }

  void addAieElfsToCDO(DeviceOp &targetOp, const std::string &workDirPath) {
    for (auto tileOp : targetOp.getOps<TileOp>()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      if (tileOp.isShimNOCorPLTile()) {
        // Resets no needed with V2 kernel driver
      } else {
        if (auto coreOp = tileOp.getCoreOp()) {
          std::string fileName;
          if (auto fileAttr = coreOp.getElfFile())
            fileName = fileAttr->str();
          else
            fileName = std::string("core_") + std::to_string(col) + "_" +
                       std::to_string(row) + ".elf";
          addAieElfToCDO(col, row, workDirPath + ps + fileName);
        }
      }
    }
  }

  void addInitConfigToCDO(DeviceOp &targetOp) {
    for (auto tileOp : targetOp.getOps<TileOp>()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      if (!tileOp.isShimTile() && tileOp.getCoreOp()) {
        TRY_XAIE_API(XAie_CoreReset, &devInst, XAie_TileLoc(col, row));
        TRY_XAIE_API(XAie_CoreUnreset, &devInst, XAie_TileLoc(col, row));
        // Set locks to zero
        for (uint8_t l = 0; l < NUM_LOCKS; l++)
          TRY_XAIE_API(XAie_LockSetValue, &devInst, XAie_TileLoc(col, row),
                       XAie_LockInit(l, 0));
      }
    }

    // Set locks with explicit initializers
    for (auto lockOp : targetOp.getOps<LockOp>()) {
      auto tileOp = lockOp.getTileOp();
      assert(lockOp.getLockID() && lockOp.getInit() &&
             "locks must be fully initialized");
      TRY_XAIE_API(XAie_LockSetValue, &devInst,
                   XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex()),
                   XAie_LockInit(*lockOp.getLockID(), *lockOp.getInit()));
    }

    auto &targetModel = targetOp.getTargetModel();
    // llvm::concat<Operation>(targetOp.getOps<MemOp>(),
    //                         targetOp.getOps<MemTileDMAOp>());
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
        bool foundBdPacket = false;
        int packetType = 0;
        int packetID = 0;
        bool foundBd = false;
        int lenA = 0;
        int bytesA = 0;
        int offsetA = 0;
        int baseAddrA = 0;
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

        int acqValue = 0, relValue = 0;
        int acqLockId = 0;
        int relLockId = 0;
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
          } else {
            op.emitOpError("unsupported lock action");
            llvm::report_fatal_error("unsupported lock action");
          }
        }

        if (targetModel.isMemTile(col, row)) {
          acqLockId += ACQ_LOCK_ID_INCR;
          relLockId += REL_LOCK_ID_INCR;
          baseAddrA += BASE_ADDR_A_INCR;
        }

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
          TRY_XAIE_API(XAie_DmaDescInit, &devInst, &dmaTileBd,
                       XAie_TileLoc(col, row));
          TRY_XAIE_API(XAie_DmaSetLock, &dmaTileBd,
                       XAie_LockInit(acqLockId, acqValue),
                       XAie_LockInit(relLockId, relValue));
          if (!dims)
            TRY_XAIE_API(XAie_DmaSetAddrLen, &dmaTileBd, baseAddrA + offsetA,
                         lenA * bytesA);
          else {
            XAie_DmaTensor dmaTileBdTensor = {};
            dmaTileBdTensor.NumDim = dims->size();
            dmaTileBdTensor.Dim = static_cast<XAie_DmaDimDesc *>(
                calloc(dims->size(), sizeof(XAie_DmaDimDesc)));
            if (!dmaTileBdTensor.Dim)
              llvm::report_fatal_error(
                  "couldn't allocate array of XAie_DmaDimDesc");
            for (size_t i = 0; i < dims->size(); i++) {
              // Pass down dimensions in reverse order; in the MLIR, this allows
              // us to specify step sizes/wraps in the same order as we would
              // access a multi-dim C array, with the highest dimension first.
              int j = dims->size() - i - 1;
              // Assume AIE-ML architecture; we assert this above
              dmaTileBdTensor.Dim[j].AieMlDimDesc = {dims.value()[i].getStep(),
                                                     dims.value()[i].getWrap()};
            }
            TRY_XAIE_API(XAie_DmaSetMultiDimAddr, &dmaTileBd, &dmaTileBdTensor,
                         baseAddrA + offsetA, lenA * bytesA);
            // TODO: Probably need special handling for NOC
            // TODO: Might need to adjust step sizes / wraps by -1
          }

          if (block.getNumSuccessors()) {
            assert(llvm::range_size(block.getSuccessors()) == 1 &&
                   "should have only one successor block");
            Block *nextBlock =
                block.getSuccessor(0); // should have only one successor block
            int nextBdNum = blockMap[nextBlock];
            // TODO Check if br ^end: to disable this?
            TRY_XAIE_API(XAie_DmaSetNextBd, &dmaTileBd, nextBdNum,
                         /* enableNextBd */ 1);
          }

          if (foundBdPacket)
            TRY_XAIE_API(XAie_DmaSetPkt, &dmaTileBd,
                         XAie_PacketInit(packetID, packetType));
          TRY_XAIE_API(XAie_DmaEnableBd, &dmaTileBd);
          TRY_XAIE_API(XAie_DmaWriteBd, &devInst, &dmaTileBd,
                       XAie_TileLoc(col, row), bdNum);
        }
      }

      for (Block &block : memOp.getOperation()->getRegion(0))
        for (auto op : block.getOps<DMAStartOp>()) {
          int bdNum = blockMap[op.getDest()];
          int chNum = op.getChannelIndex();
          TRY_XAIE_API(XAie_DmaChannelPushBdToQueue, &devInst,
                       XAie_TileLoc(col, row), chNum,
                       // TODO hack until physical dialect changes
                       op.getChannelDir() == DMAChannelDir::S2MM ? DMA_S2MM
                                                                 : DMA_MM2S,
                       bdNum);
          TRY_XAIE_API(
              XAie_DmaChannelEnable, &devInst, XAie_TileLoc(col, row), chNum,
              // TODO hack until physical dialect changes
              op.getChannelDir() == DMAChannelDir::S2MM ? DMA_S2MM : DMA_MM2S);
        }
    }

    int x, y;
    // StreamSwitch (switchbox) configuration
    for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
      Region &r = switchboxOp.getConnections();
      Block &b = r.front();
      bool isEmpty = b.getOps<ConnectOp>().empty() &&
                     b.getOps<MasterSetOp>().empty() &&
                     b.getOps<PacketRulesOp>().empty();
      bool isParam = false;
      if (isa<TileOp>(switchboxOp.getTile().getDefiningOp())) {
        int col = switchboxOp.colIndex();
        int row = switchboxOp.rowIndex();
        if (!isEmpty) {
          // Core Stream Switch column for col, row
          x = col;
          y = row;
        }
      } else if (AIEX::SelectOp sel = dyn_cast<AIEX::SelectOp>(
                     switchboxOp.getTile().getDefiningOp())) {
        // parameterize streamswitch's configuration
        isParam = true;
        HerdOp sourceHerd =
            dyn_cast<HerdOp>(sel.getStartHerd().getDefiningOp());

        IterOp iterX = dyn_cast<IterOp>(sel.getIterX().getDefiningOp());
        IterOp iterY = dyn_cast<IterOp>(sel.getIterY().getDefiningOp());
        int startXValue = iterX.getStartValue();
        int endXValue = iterX.getEndValue();
        int strideXValue = iterX.getStrideValue();
        int startYValue = iterY.getStartValue();
        int endYValue = iterY.getEndValue();
        int strideYValue = iterY.getStrideValue();

        llvm::report_fatal_error("HerdOp not supported");
        // output << "for (x = " << startX << "; x < " << endX
        //        << "; x += " << strideXValue << ") {\n";
        // output << "for (y = " << startY << "; y < " << endY
        //        << "; y += " << strideYValue << ") {\n";
      }

      if (switchboxOp.rowIndex() == 0) {
        // FIXME hack for TCT routing
        // TODO Support both channels
        TRY_XAIE_API(XAie_StrmConnCctEnable, &devInst, XAie_TileLoc(x, y), CTRL,
                     0, SOUTH, 0);
        {
          // configure DMA_<S2MM/MM2S>_<N>_Ctrl register
          XAie_DmaChannelDesc dmaChannelDescInst;
          TRY_XAIE_API(XAie_DmaChannelDescInit, &devInst, &dmaChannelDescInst,
                       XAie_TileLoc(x, y));
          TRY_XAIE_API(XAie_DmaChannelSetControllerId, &dmaChannelDescInst, 0);
          TRY_XAIE_API(XAie_DmaWriteChannel, &devInst, &dmaChannelDescInst,
                       XAie_TileLoc(x, y), 0, DMA_S2MM);
        }

        {
          // configure DMA_<S2MM/MM2S>_<N>_Ctrl register
          XAie_DmaChannelDesc dmaChannelDescInst;
          TRY_XAIE_API(XAie_DmaChannelDescInit, &devInst, &dmaChannelDescInst,
                       XAie_TileLoc(x, y));
          TRY_XAIE_API(XAie_DmaChannelSetControllerId, &dmaChannelDescInst, 0);
          TRY_XAIE_API(XAie_DmaWriteChannel, &devInst, &dmaChannelDescInst,
                       XAie_TileLoc(x, y), 1, DMA_S2MM);
        }
        TRY_XAIE_API(XAie_AieToPlIntfEnable, &devInst, XAie_TileLoc(x, y), 0,
                     PLIF_WIDTH_32);
      }

      for (auto connectOp : b.getOps<ConnectOp>())
        TRY_XAIE_API(
            XAie_StrmConnCctEnable, &devInst, XAie_TileLoc(x, y),
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
        TRY_XAIE_API(
            XAie_StrmPktSwMstrPortEnable, &devInst, XAie_TileLoc(x, y),
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
          TRY_XAIE_API(
              XAie_StrmPktSwSlavePortEnable, &devInst, XAie_TileLoc(x, y),
              WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
              connectOp.sourceIndex());
          // TODO Need to better define packet id,type used here
          TRY_XAIE_API(
              XAie_StrmPktSwSlaveSlotEnable, &devInst, XAie_TileLoc(x, y),
              WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
              connectOp.sourceIndex(), slot,
              XAie_PacketInit(slotOp.valueInt(), /*type*/ 0), slotOp.maskInt(),
              msel, arbiter);
          slot++;
        }
      }

      if (isParam)
        llvm::report_fatal_error("HerdOp not supported");
    }
    for (auto op : targetOp.getOps<ShimMuxOp>()) {
      Region &r = op.getConnections();
      Block &b = r.front();
      bool isEmpty = b.getOps<ConnectOp>().empty();

      if (isa<TileOp>(op.getTile().getDefiningOp())) {
        int col = op.colIndex();
        int row = op.rowIndex();
        if (!isEmpty) {
          // NOTE ShimMux always connects from the south as directions are
          // defined relative to the tile stream switch\n";
          x = col;
          y = row;
        }
      }

      for (auto connectOp : b.getOps<ConnectOp>()) {
        // demux!
        if (connectOp.getSourceBundle() == WireBundle::North)
          TRY_XAIE_API(XAie_EnableAieToShimDmaStrmPort, &devInst,
                       XAie_TileLoc(x, y), connectOp.sourceIndex());
        // mux
        if (connectOp.getDestBundle() == WireBundle::North)
          TRY_XAIE_API(XAie_EnableShimDmaToAieStrmPort, &devInst,
                       XAie_TileLoc(x, y), connectOp.destIndex());
      }
    }

    for (auto switchboxOp : targetOp.getOps<ShimSwitchboxOp>()) {
      Block &b = switchboxOp.getConnections().front();
      int col = switchboxOp.getCol();
      for (auto connectOp : b.getOps<ConnectOp>())
        TRY_XAIE_API(
            XAie_StrmConnCctEnable, &devInst, XAie_TileLoc(col, 0),
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
            connectOp.sourceIndex(),
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
            connectOp.destIndex());
    }
  }

  void addCoreEnableToCDO(DeviceOp &targetOp) {
    // Start execution of all the cores.
    for (auto tileOp : targetOp.getOps<TileOp>())
      if (!tileOp.isShimTile() && tileOp.getCoreOp())
        TRY_XAIE_API(XAie_CoreEnable, &devInst,
                     XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex()));
  }
};

} // namespace xilinx::AIE

void initializeCDOGenerator(byte_ordering endianness) {
#ifndef NDEBUG
  EnAXIdebug(); // Enables AXI-MM prints for configs being added in CDO,
#endif
  setEndianness(endianness);
};

void generateCDOBinary(const std::string &outputPath,
                       const std::function<void(void)> &cb) {
  startCDOFileStream(outputPath.c_str());
  FileHeader();
  cb();
  configureHeader();
  endCurrentCDOFileStream();
}

void generateCDOBinariesSeparately(AIEControl &ctl,
                                   const std::string &workDirPath,
                                   DeviceOp &targetOp) {
  generateCDOBinary(workDirPath + ps + "aie_cdo_error_handling.bin",
                    std::bind(&AIEControl::addErrorHandlingToCDO, ctl));
  generateCDOBinary(workDirPath + ps + "aie_cdo_elfs.bin",
                    [&ctl, &targetOp, &workDirPath] {
                      ctl.addAieElfsToCDO(targetOp, workDirPath);
                    });
  generateCDOBinary(workDirPath + ps + "aie_cdo_init.bin",
                    [&ctl, &targetOp] { ctl.addInitConfigToCDO(targetOp); });
  generateCDOBinary(workDirPath + ps + "aie_cdo_enable.bin",
                    [&ctl, &targetOp] { ctl.addCoreEnableToCDO(targetOp); });
}

LogicalResult AIE::AIETranslateToCDODirect(ModuleOp &m,
                                           const std::string &workDirPath,
                                           byte_ordering endianness) {
  auto devOps = m.getOps<DeviceOp>();
  assert(llvm::range_size(devOps) == 1 &&
         "only exactly 1 device op supported.");
  DeviceOp targetOp = *devOps.begin();

  AIEControl ctl;
  initializeCDOGenerator(endianness);
  generateCDOBinariesSeparately(ctl, workDirPath, targetOp);
  return success();
}
