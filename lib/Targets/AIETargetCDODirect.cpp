//===- AIETargetCDODirect.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Targets/AIETargets.h"
extern "C" {
#include "cdo-driver/cdo_driver.h"
}

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
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

#define DEBUG_TYPE "aie-generate-cdo"

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

// https://stackoverflow.com/a/32230306
template <typename H1>
raw_ostream &showArgs(raw_ostream &out, const char *label, H1 &&value) {
  return out << label << "=" << std::forward<H1>(value);
}

template <typename H1, typename... T>
raw_ostream &showArgs(raw_ostream &out, const char *label, H1 &&value,
                      T &&...rest) {
  const char *pcomma = strchr(label, ',');
  return showArgs(out.write(label, pcomma - label)
                      << "=" << std::forward<H1>(value) << ',',
                  pcomma + 1, std::forward<T>(rest)...);
}

#define SHOW_ARGS(os, ...) showArgs(os, #__VA_ARGS__, __VA_ARGS__)

raw_ostream &operator<<(raw_ostream &os, const XAie_LocType &loc) {
  os << "XAie_LocType(col: " << std::to_string(loc.Col)
     << ", row: " << std::to_string(loc.Row) << ")";
  return os;
}

raw_ostream &operator<<(raw_ostream &os, const XAie_Lock &lock) {
  os << "XAie_Lock(id: " << std::to_string(lock.LockId)
     << ", val: " << std::to_string(lock.LockVal) << ")";
  return os;
}

raw_ostream &operator<<(raw_ostream &os, const XAie_Packet &packet) {
  os << "XAie_Packet(id: " << std::to_string(packet.PktId)
     << ", type: " << std::to_string(packet.PktType) << ")";
  return os;
}

// So that we can use the pattern if(auto r = TRY_XAIE_API...) { // r is nonzero
// }
static_assert(XAIE_OK == 0);

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                                     \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                          \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__))                                             \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +             \
                               AIERCTOSTR.at(r));                              \
  } while (0)

#define TRY_XAIE_API_EMIT_ERROR(OP, API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                          \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__))                                             \
      return OP.emitOpError() << #API " failed with " << AIERCTOSTR.at(r);     \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                          \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__)) {                                           \
      llvm::errs() << #API " failed with " << AIERCTOSTR.at(r);                \
      return failure();                                                        \
    }                                                                          \
  } while (0)

auto ps = std::filesystem::path::preferred_separator;

#define XAIE_BASE_ADDR 0x40000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_PARTITION_BASE_ADDR 0x0

#define NPI_ADDR 0x0
#define NUM_LOCKS 16
#define EVEN_BD_NUM_START 0
#define ODD_BD_NUM_START 24
#define MEM_TILE_LOCK_ID_INCR 64
#define BASE_ADDR_A_INCR 0x80000

namespace xilinx::AIE {

LogicalResult configureLocksInBdBlock(XAie_DmaDesc &dmaTileBd, Block &block,
                                      const AIETargetModel &targetModel,
                                      XAie_LocType &tileLoc) {
  assert(!block.getOps<UseLockOp>().empty() &&
         "expected use_lock op in bb with dma_db op");
  std::optional<int> acqValue, relValue, acqLockId, relLockId;
  bool acqEn;
  // switch (lock->getAc)
  for (auto op : block.getOps<UseLockOp>()) {
    // Only dyn_cast if you are going to check if it was of the type
    // expected; if you aren't checking use cast instead as it will at
    // least assert in debug mode with an easier to understand error than
    // dereferencing.
    LockOp lock = cast<LockOp>(op.getLock().getDefiningOp());
    switch (op.getAction()) {
    case LockAction::Acquire:
    case LockAction::AcquireGreaterEqual:
      acqEn = op.getAcqEn();
      acqLockId = lock.getLockIDValue();
      acqValue = op.getLockValue();
      if (op.acquireGE())
        acqValue.value() = -acqValue.value();
      break;
    case LockAction::Release:
      relLockId = lock.getLockIDValue();
      relValue = op.getLockValue();
      break;
    }
  }

  assert(acqValue && relValue && acqLockId && relLockId &&
         "expected both use_lock(acquire) and use_lock(release) with bd");

  if (targetModel.isMemTile(tileLoc.Col, tileLoc.Row)) {
    if (acqLockId)
      acqLockId.value() += MEM_TILE_LOCK_ID_INCR;
    if (relLockId)
      relLockId.value() += MEM_TILE_LOCK_ID_INCR;
  }

  // no RelEn in the arch spec even though the API requires you to set it?
  bool relEn = true;
  XAie_Lock acqLock = XAie_LockInit(acqLockId.value(), acqValue.value());
  XAie_Lock relLock = XAie_LockInit(relLockId.value(), relValue.value());
  TRY_XAIE_API_EMIT_ERROR((*block.getOps<UseLockOp>().begin()),
                          dmaTileBd.DmaMod->SetLock, &dmaTileBd, acqLock,
                          relLock, acqEn, relEn);
  return success();
}

LogicalResult configureBdInBlock(XAie_DevInst &devInst, XAie_DmaDesc &dmaTileBd,
                                 Block &block,
                                 const AIETargetModel &targetModel,
                                 XAie_LocType &tileLoc, int bdNum,
                                 std::optional<int> nextBdNum) {
  assert(!block.getOps<DMABDOp>().empty() && "expected bd ops in block");

  std::optional<int> packetType;
  std::optional<int> packetID;
  auto maybePacketOps = block.getOps<DMABDPACKETOp>();
  if (!maybePacketOps.empty()) {
    assert(llvm::range_size(maybePacketOps) == 1 &&
           "expected only one dma_bd_packet");
    auto packetOp = *maybePacketOps.begin();
    packetType = packetOp.getPacketType();
    packetID = packetOp.getPacketID();
  }

  // deref here because this is a const iter and the various getters below
  // aren't const (even though they probably should be...)
  auto bdOp = *block.getOps<DMABDOp>().begin();
  // StringRef FifoMode = disable; // FIXME: when to enable FIFO mode?
  ShapedType bufferType = bdOp.getBuffer().getType().cast<::mlir::MemRefType>();
  int bytes = bufferType.getElementTypeBitWidth() / 8;
  int baseAddr = 0;
  if (!targetModel.isShimNOCTile(tileLoc.Col, tileLoc.Row)) {
    auto bufferOp = cast<AIE::BufferOp>(bdOp.getBuffer().getDefiningOp());
    assert(bufferOp.getAddress().has_value() && "buffer must have address");
    baseAddr = bufferOp.getAddress().value();
    if (targetModel.isMemTile(tileLoc.Col, tileLoc.Row))
      baseAddr += BASE_ADDR_A_INCR;
  }

  std::optional<llvm::ArrayRef<BDDimLayoutAttr>> dims = bdOp.getDimensions();
  if (!dims) {
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAddrLen, &dmaTileBd,
                            baseAddr + bdOp.getOffsetValue(),
                            bdOp.getLenValue() * bytes);
    if (targetModel.isShimNOCTile(tileLoc.Col, tileLoc.Row)) {
      // write them out like this so they show up with names in debug prints
      size_t smid = 0;
      size_t burstLen = 4; // 00 config, 64 bytes?
      size_t qOs = 0;
      size_t cache = 0;
      // uint8_t secure = XAIE_ENABLE;
      size_t secure = 0;
      TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAxi, &dmaTileBd, smid, burstLen,
                              qOs, cache, secure);
    }

  } else {
    XAie_DmaTensor dmaTileBdTensor = {};
    dmaTileBdTensor.NumDim = dims->size();
    dmaTileBdTensor.Dim = static_cast<XAie_DmaDimDesc *>(
        calloc(dims->size(), sizeof(XAie_DmaDimDesc)));
    if (!dmaTileBdTensor.Dim)
      return bdOp.emitError("couldn't allocate array of XAie_DmaDimDesc");
    // TODO(max): rethink this?
    for (size_t i = 0; i < dims->size(); i++) {
      // Pass down dimensions in reverse order; in the MLIR, this allows
      // us to specify step sizes/wraps in the same order as we would
      // access a multi-dim C array, with the highest dimension first.
      int j = dims->size() - i - 1;
      // Assume AIE-ML architecture; we assert this above
      // TODO(max): no we don't
      dmaTileBdTensor.Dim[j].AieMlDimDesc = {dims.value()[i].getStride(),
                                             dims.value()[i].getSize()};
    }
    // TODO: Probably need special handling for NOC
    // TODO: Might need to adjust step sizes / wraps by -1
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetMultiDimAddr, &dmaTileBd,
                            &dmaTileBdTensor, baseAddr + bdOp.getOffsetValue(),
                            bdOp.getLenValue() * bytes);
  }

  if (nextBdNum) {
    auto enableNextBd = 1;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetNextBd, &dmaTileBd,
                            nextBdNum.value(), enableNextBd);
  }

  if (packetID) {
    assert(packetType && "must have packetType with packetID");
    if (bdOp.getLenValue() == 0)
      return bdOp.emitOpError(
          "For MM2S channels, if Buffer_Length=0 then Enable_Packet must be "
          "set to 0, otherwise behavior is undefined (3.7.8 arch spec)");
    TRY_XAIE_API_EMIT_ERROR(
        bdOp, XAie_DmaSetPkt, &dmaTileBd,
        XAie_PacketInit(packetID.value(), packetType.value()));
  }
  TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaEnableBd, &dmaTileBd);
  TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaWriteBd, &devInst, &dmaTileBd, tileLoc,
                          bdNum);
  return success();
};

struct AIEControl {
  XAie_Config configPtr;
  XAie_DevInst devInst;

  AIEControl(size_t partitionStartCol, size_t partitionNumCols, bool aieSim,
             const AIETargetModel &tm) {
    configPtr = XAie_Config{
        /*AieGen*/ XAIE_DEV_GEN_AIEML,
        /*BaseAddr*/ XAIE_BASE_ADDR,
        /*ColShift*/ XAIE_COL_SHIFT,
        /*RowShift*/ XAIE_ROW_SHIFT,
        /*NumRows*/ static_cast<uint8_t>(tm.rows()),
        /*NumCols*/ static_cast<uint8_t>(tm.columns()),
        /*ShimRowNum*/ XAIE_SHIM_ROW,
        /*MemTileRowStart*/ XAIE_MEM_TILE_ROW_START,
        /*MemTileNumRows*/ static_cast<uint8_t>(tm.getNumMemTileRows()),
        /*AieTileRowStart*/
        static_cast<uint8_t>(XAIE_MEM_TILE_ROW_START + tm.getNumMemTileRows()),
        /*AieTileNumRows*/
        static_cast<uint8_t>(tm.rows() - tm.getNumMemTileRows() - 1),
        /*PartProp*/ {},
        /*Backend*/ XAIE_IO_BACKEND_CDO};

    // Quoting: The instance of a device must be always declared using this
    //		macro. In future, the same macro will be expanded to allocate
    //		more memory from the user application for resource management.
    XAie_InstDeclare(_devInst, &configPtr);

    devInst = _devInst;
    // TODO(max): what is the "partition"?
    TRY_XAIE_API_FATAL_ERROR(XAie_SetupPartitionConfig, &devInst,
                             XAIE_PARTITION_BASE_ADDR, partitionStartCol,
                             partitionNumCols);
    TRY_XAIE_API_FATAL_ERROR(XAie_CfgInitialize, &devInst, &configPtr);
    if (aieSim) {
      TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst,
                               XAIE_IO_BACKEND_CDO);
    }
    TRY_XAIE_API_FATAL_ERROR(XAie_UpdateNpiAddr, &devInst, NPI_ADDR);
  }

  LogicalResult addErrorHandlingToCDO() {
    TRY_XAIE_API_LOGICAL_RESULT(XAie_ErrorHandlingInit, &devInst);
    return success();
  }

  LogicalResult addAieElfToCDO(uint8_t col, uint8_t row,
                               const StringRef elfPath, bool aieSim) {
    // loadSym: Load symbols from .map file. This argument is not used when
    // __AIESIM__ is not defined.
    TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadElf, &devInst, XAie_TileLoc(col, row),
                                elfPath.str().c_str(), /*loadSym*/ aieSim);
    return success();
  }

  LogicalResult addAieElfsToCDO(DeviceOp &targetOp, const StringRef workDirPath,
                                bool aieSim) {
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
            fileName = (llvm::Twine("core_") + std::to_string(col) + "_" +
                        std::to_string(row) + ".elf")
                           .str();
          if (failed(addAieElfToCDO(
                  col, row,
                  (llvm::Twine(workDirPath) + std::string(1, ps) + fileName)
                      .str(),
                  aieSim)))
            return failure();
        }
      }
    return success();
  }

  LogicalResult addInitConfigToCDO(DeviceOp &targetOp) {
    for (auto tileOp : targetOp.getOps<TileOp>()) {
      auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
      if (!tileOp.isShimTile() && tileOp.getCoreOp()) {
        TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreReset, &devInst, tileLoc);
        TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreUnreset, &devInst, tileLoc);
        // Set locks to zero
        for (uint8_t l = 0; l < NUM_LOCKS; l++) {
          auto locInit = XAie_LockInit(l, 0);
          TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_LockSetValue, &devInst, tileLoc,
                                  locInit);
        }
      }
    }

    // Set locks with explicit initializers
    targetOp.walk<WalkOrder::PreOrder>([&](LockOp lockOp) {
      if (lockOp.getLockID() && lockOp.getInit()) {
        auto tileLoc = XAie_TileLoc(lockOp.getTileOp().colIndex(),
                                    lockOp.getTileOp().rowIndex());
        auto locInit = XAie_LockInit(*lockOp.getLockID(), *lockOp.getInit());
        TRY_XAIE_API_FATAL_ERROR(XAie_LockSetValue, &devInst, tileLoc, locInit);
      } else
        LLVM_DEBUG(llvm::dbgs()
                   << "lock op missing either id or init" << lockOp << "\n");
    });

    auto pushToBdQueueAndEnable =
        [this](Operation &op, XAie_LocType &tileLoc, int chNum,
               const DMAChannelDir &channelDir, int bdNum,
               int repeatCount) -> LogicalResult {
      XAie_DmaDirection direction =
          channelDir == DMAChannelDir::S2MM ? DMA_S2MM : DMA_MM2S;
      auto enTokenIssue = tileLoc.Row == 0 && direction == DMA_S2MM;
      // XAIE for whatever reason does not adopt the same convention as the arch
      // ie repeat_count = 0 => do the thing once; instead it takes
      // repeat_count = 1 => do the thing once and then subtracts one off
      // when doing the register write. so we keep the arch convention
      repeatCount += 1;
      TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelSetStartQueue, &devInst,
                              tileLoc, chNum, direction, bdNum, repeatCount,
                              enTokenIssue);
      TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelEnable, &devInst, tileLoc,
                              chNum, direction);
      return success();
    };

    const AIETargetModel &targetModel = targetOp.getTargetModel();
    auto isMemTileOddBd = [&targetModel](int col, int row, int channelIndex) {
      return targetModel.isMemTile(col, row) && channelIndex & 1;
    };
    auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
    llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
    llvm::append_range(memOps, targetOp.getOps<ShimDMAOp>());
    for (TileElement memOp : memOps) {
      int col = memOp.getTileID().col;
      int row = memOp.getTileID().row;
      auto tileLoc = XAie_TileLoc(col, row);
      DenseMap<Block *, int> blockBdNumMap;

      // handle DMA ops separately
      auto dmaOps = llvm::to_vector_of<DMAOp>(
          memOp.getOperation()->getRegion(0).getOps<DMAOp>());
      if (!dmaOps.empty()) {
        int oddBdNum = ODD_BD_NUM_START;
        int evenBdNum = EVEN_BD_NUM_START;
        for (auto dmaOp : dmaOps) {
          auto bdRegions = dmaOp.getBds();
          for (auto *bdRegionIt = bdRegions.begin();
               bdRegionIt != bdRegions.end();) {
            auto &block = bdRegionIt->getBlocks().front();
            blockBdNumMap[&block] =
                isMemTileOddBd(col, row, dmaOp.getChannelIndex()) ? oddBdNum++
                                                                  : evenBdNum++;
            std::optional<int> nextBdNum;
            if (++bdRegionIt != bdRegions.end()) {
              nextBdNum = isMemTileOddBd(col, row, dmaOp.getChannelIndex())
                              ? oddBdNum
                              : evenBdNum;
            } else if (dmaOp.getLoop()) {
              assert(blockBdNumMap.contains(
                  &bdRegions.front().getBlocks().front()));
              nextBdNum = blockBdNumMap[&bdRegions.front().getBlocks().front()];
            }
            assert(blockBdNumMap.contains(&block));
            XAie_DmaDesc dmaTileBd;
            TRY_XAIE_API_EMIT_ERROR(dmaOp, XAie_DmaDescInit, &devInst,
                                    &dmaTileBd, tileLoc);
            if (!block.getOps<UseLockOp>().empty() &&
                failed(configureLocksInBdBlock(dmaTileBd, block, targetModel,
                                               tileLoc)))
              return failure();
            if (!block.getOps<DMABDOp>().empty() &&
                failed(configureBdInBlock(devInst, dmaTileBd, block,
                                          targetModel, tileLoc,
                                          blockBdNumMap[&block], nextBdNum)))
              return failure();
          }
        }
      } else {
        DenseMap<Block *, int> blockChannelMap;
        // Assign each block a BD number
        for (Block &block : memOp.getOperation()->getRegion(0))
          for (auto op : block.getOps<DMAStartOp>()) {
            int chNum = op.getChannelIndex();
            blockChannelMap[&block] = chNum;
            Block *dest = op.getDest();
            while (dest) {
              blockChannelMap[dest] = chNum;
              if (dest->hasNoSuccessors())
                break;
              dest = dest->getSuccessors()[0];
              if (blockChannelMap.contains(dest))
                dest = nullptr;
            }
          }

        // Assign each block a BD number
        int evenBdNum = EVEN_BD_NUM_START;
        int oddBdNum = ODD_BD_NUM_START;
        for (Block &block : memOp.getOperation()->getRegion(0)) {
          if (block.getOps<DMABDOp>().empty())
            continue;
          assert(blockChannelMap.count(&block));
          if (isMemTileOddBd(col, row, blockChannelMap[&block]))
            blockBdNumMap[&block] = oddBdNum++;
          else
            blockBdNumMap[&block] = evenBdNum++;
        }

        for (Block &block : memOp.getOperation()->getRegion(0)) {
          if (block.getOps<DMABDOp>().empty())
            continue;
          assert(blockBdNumMap.contains(&block));
          int bdNum = blockBdNumMap[&block];

          std::optional<int> nextBdNum;
          if (block.getNumSuccessors()) {
            assert(llvm::range_size(block.getSuccessors()) == 1 &&
                   "should have only one successor block");
            Block *nextBlock = block.getSuccessor(0);
            if (!blockBdNumMap.contains(nextBlock))
              assert(nextBlock->getOperations().size() == 1 &&
                     isa<EndOp>(nextBlock->getOperations().front()) &&
                     "bb that's not in blockMap can only have aie.end");
            else
              nextBdNum = blockBdNumMap[nextBlock];
          }

          XAie_DmaDesc dmaTileBd;
          TRY_XAIE_API_EMIT_ERROR(memOp, XAie_DmaDescInit, &devInst, &dmaTileBd,
                                  tileLoc);
          if (!block.getOps<UseLockOp>().empty() &&
              failed(configureLocksInBdBlock(dmaTileBd, block, targetModel,
                                             tileLoc)))
            return failure();
          if (!block.getOps<DMABDOp>().empty() &&
              failed(configureBdInBlock(devInst, dmaTileBd, block, targetModel,
                                        tileLoc, bdNum, nextBdNum)))
            return failure();
        }
      }

      if (!dmaOps.empty())
        for (auto dmaOp : dmaOps) {
          auto &block = dmaOp.getBds().front().getBlocks().front();
          assert(blockBdNumMap.contains(&block));
          if (failed(pushToBdQueueAndEnable(
                  *dmaOp.getOperation(), tileLoc, dmaOp.getChannelIndex(),
                  dmaOp.getChannelDir(), blockBdNumMap[&block],
                  dmaOp.getRepeatCount())))
            return failure();
        }
      else
        for (Block &block : memOp.getOperation()->getRegion(0)) {
          for (auto op : block.getOps<DMAStartOp>()) {
            assert(blockBdNumMap.contains(op.getDest()));
            int bdNum = blockBdNumMap[op.getDest()];
            int chNum = op.getChannelIndex();
            auto channelDir = op.getChannelDir();
            if (failed(pushToBdQueueAndEnable(*op.getOperation(), tileLoc,
                                              chNum, channelDir, bdNum,
                                              op.getRepeatCount())))
              return failure();
          }
        }
    }

    // StreamSwitch (switchbox) configuration
    for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
      int32_t col = switchboxOp.colIndex();
      int32_t row = switchboxOp.rowIndex();
      XAie_LocType tileLoc = XAie_TileLoc(col, row);
      assert(targetOp.getDevice() == AIEDevice::ipu &&
             "Only IPU currently supported");
      if (row == 0) {
        // FIXME hack for TCT routing
        // TODO Support both channels
        auto slvPortNum = 0;
        auto mstrPortNum = 0;
        TRY_XAIE_API_EMIT_ERROR(switchboxOp, XAie_StrmConnCctEnable, &devInst,
                                tileLoc, CTRL, slvPortNum, SOUTH, mstrPortNum);
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
          AMSelOp amsel = cast<AMSelOp>(val.getDefiningOp());
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
        auto dropHeader =
            isdma ? XAIE_SS_PKT_DROP_HEADER : XAIE_SS_PKT_DONOT_DROP_HEADER;
        TRY_XAIE_API_EMIT_ERROR(
            connectOp, XAie_StrmPktSwMstrPortEnable, &devInst, tileLoc,
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
            connectOp.destIndex(), dropHeader, arbiter, mask);
      }

      for (auto connectOp : b.getOps<PacketRulesOp>()) {
        int slot = 0;
        Block &block = connectOp.getRules().front();
        for (auto slotOp : block.getOps<PacketRuleOp>()) {
          AMSelOp amselOp = cast<AMSelOp>(slotOp.getAmsel().getDefiningOp());
          int arbiter = amselOp.arbiterIndex();
          int msel = amselOp.getMselValue();
          TRY_XAIE_API_EMIT_ERROR(
              connectOp, XAie_StrmPktSwSlavePortEnable, &devInst, tileLoc,
              WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
              connectOp.sourceIndex());
          auto packetInit = XAie_PacketInit(slotOp.valueInt(), /*PktType*/ 0);
          // TODO Need to better define packet id,type used here
          TRY_XAIE_API_EMIT_ERROR(
              connectOp, XAie_StrmPktSwSlaveSlotEnable, &devInst, tileLoc,
              WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
              connectOp.sourceIndex(), slot, packetInit, slotOp.maskInt(), msel,
              arbiter);
          slot++;
        }
      }
    }

    for (auto muxOp : targetOp.getOps<ShimMuxOp>()) {
      // NOTE ShimMux always connects from the south as directions are
      // defined relative to the tile stream switch.
      auto tileLoc =
          XAie_TileLoc(muxOp.getTileOp().getCol(), muxOp.getTileOp().getRow());
      Block &b = muxOp.getConnections().front();
      for (auto connectOp : b.getOps<ConnectOp>()) {
        // demux!
        if (connectOp.getSourceBundle() == WireBundle::North)
          TRY_XAIE_API_EMIT_ERROR(muxOp, XAie_EnableAieToShimDmaStrmPort,
                                  &devInst, tileLoc, connectOp.sourceIndex());
        // mux
        if (connectOp.getDestBundle() == WireBundle::North)
          TRY_XAIE_API_EMIT_ERROR(muxOp, XAie_EnableShimDmaToAieStrmPort,
                                  &devInst, tileLoc, connectOp.destIndex());
      }
    }

    for (auto switchboxOp : targetOp.getOps<ShimSwitchboxOp>()) {
      Block &b = switchboxOp.getConnections().front();
      auto tileLoc = XAie_TileLoc(switchboxOp.getCol(), 0);
      for (auto connectOp : b.getOps<ConnectOp>())
        TRY_XAIE_API_EMIT_ERROR(
            switchboxOp, XAie_StrmConnCctEnable, &devInst, tileLoc,
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
            connectOp.sourceIndex(),
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
            connectOp.destIndex());
    }

    // Cascade configuration
    if (targetModel.getTargetArch() == AIEArch::AIE2) {
      for (auto configOp : targetOp.getOps<ConfigureCascadeOp>()) {
        TileOp tile = cast<TileOp>(configOp.getTile().getDefiningOp());
        auto tileLoc = XAie_TileLoc(tile.getCol(), tile.getRow());
        TRY_XAIE_API_EMIT_ERROR(
            targetOp, XAie_CoreConfigAccumulatorControl, &devInst, tileLoc,
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
                static_cast<WireBundle>(configOp.getInputDir())),
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
                static_cast<WireBundle>(configOp.getOutputDir())));
      }
    }

    return success();
  }

  LogicalResult addCoreEnableToCDO(DeviceOp &targetOp) {
    // Start execution of all the cores.
    for (auto tileOp : targetOp.getOps<TileOp>()) {
      auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
      if (!tileOp.isShimTile() && tileOp.getCoreOp())
        TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreEnable, &devInst, tileLoc);
    }
    return success();
  }

  void dmaUpdateBdAddr(DeviceOp &targetOp, int col, int row, size_t addr,
                       size_t bdNum) {
    auto tileLoc = XAie_TileLoc(col, row);
    TRY_XAIE_API_FATAL_ERROR(XAie_DmaUpdateBdAddr, &devInst, tileLoc, addr,
                             bdNum);
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
                                            DeviceOp &targetOp, bool aieSim) {
  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) +
           "aie_cdo_error_handling.bin")
              .str(),
          std::bind(&AIEControl::addErrorHandlingToCDO, ctl))))
    return failure();

  if (!targetOp.getOps<CoreOp>().empty() &&
      failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_elfs.bin")
              .str(),
          [&ctl, &targetOp, &workDirPath, &aieSim] {
            return ctl.addAieElfsToCDO(targetOp, workDirPath, aieSim);
          })))
    return failure();

  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_init.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.addInitConfigToCDO(targetOp); })))
    return failure();

  if (!targetOp.getOps<CoreOp>().empty() &&
      failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_enable.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.addCoreEnableToCDO(targetOp); })))
    return failure();

  return success();
}

LogicalResult generateCDOUnified(AIEControl &ctl, const StringRef workDirPath,
                                 DeviceOp &targetOp, bool aieSim) {
  return generateCDOBinary(
      (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo.bin").str(),
      [&ctl, &targetOp, &workDirPath, &aieSim] {
        if (failed(ctl.addErrorHandlingToCDO()))
          return failure();
        if (!targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addAieElfsToCDO(targetOp, workDirPath, aieSim)))
          return failure();
        if (failed(ctl.addInitConfigToCDO(targetOp)))
          return failure();
        if (!targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addCoreEnableToCDO(targetOp)))
          return failure();
        return success();
      });
}

LogicalResult AIETranslateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                                      byte_ordering endianness,
                                      bool emitUnified, bool axiDebug,
                                      bool aieSim, size_t partitionStartCol) {
  auto devOps = m.getOps<DeviceOp>();
  assert(llvm::range_size(devOps) == 1 &&
         "only exactly 1 device op supported.");
  DeviceOp targetOp = *devOps.begin();
  // things like XAIE_MEM_TILE_ROW_START and the missing
  // shim dma on tile (0,0) are hard-coded assumptions about IPU...
  assert(targetOp.getDevice() == AIEDevice::ipu &&
         "Only IPU currently supported");
  int maxCol = 0, minCol = 0;
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    minCol = std::min(tileOp.getCol(), minCol);
    maxCol = std::max(tileOp.getCol(), maxCol);
  }
  size_t partitionNumCols = maxCol - minCol + 1;
  AIEControl ctl(partitionStartCol, partitionNumCols, aieSim,
                 targetOp.getTargetModel());
  initializeCDOGenerator(endianness, axiDebug);
  if (emitUnified)
    return generateCDOUnified(ctl, workDirPath, targetOp, aieSim);
  return generateCDOBinariesSeparately(ctl, workDirPath, targetOp, aieSim);
}
// Not sure why but defining this with xilinx::AIE will create a duplicate
// symbol in libAIETargets.a that then doesn't actually match the header?
namespace xilinx::AIE {
LogicalResult AIETranslateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                                      bool bigEndian, bool emitUnified,
                                      bool axiDebug, bool aieSim,
                                      size_t partitionStartCol) {
  byte_ordering endianness =
      bigEndian ? byte_ordering::Big_Endian : byte_ordering::Little_Endian;
  return AIETranslateToCDODirect(m, workDirPath, endianness, emitUnified,
                                 axiDebug, aieSim, partitionStartCol);
}
} // namespace xilinx::AIE
