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
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

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
#include <vector>

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
#include "xaiengine/xaie_txn.h"
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
        {WireBundle::Ctrl, StrmSwPortType::CTRL},
        {WireBundle::FIFO, StrmSwPortType::FIFO},
        {WireBundle::South, StrmSwPortType::SOUTH},
        {WireBundle::West, StrmSwPortType::WEST},
        {WireBundle::North, StrmSwPortType::NORTH},
        {WireBundle::East, StrmSwPortType::EAST},
        // missing PLIO from WireBundle
        // missing NOC from WireBundle
        {WireBundle::Trace, StrmSwPortType::TRACE},
};

#ifndef NDEBUG

// https://stackoverflow.com/a/32230306
template <typename H1>
static raw_ostream &showArgs(raw_ostream &out, const char *label, H1 &&value) {
  return out << label << "=" << std::forward<H1>(value);
}

template <typename H1, typename... T>
static raw_ostream &showArgs(raw_ostream &out, const char *label, H1 &&value,
                             T &&...rest) {
  const char *pcomma = strchr(label, ',');
  return showArgs(out.write(label, pcomma - label)
                      << "=" << std::forward<H1>(value) << ',',
                  pcomma + 1, std::forward<T>(rest)...);
}

#define SHOW_ARGS(os, ...) showArgs(os, #__VA_ARGS__, __VA_ARGS__)

static raw_ostream &operator<<(raw_ostream &os, const XAie_LocType &loc) {
  os << "XAie_LocType(col: " << std::to_string(loc.Col)
     << ", row: " << std::to_string(loc.Row) << ")";
  return os;
}

static raw_ostream &operator<<(raw_ostream &os, const XAie_Lock &lock) {
  os << "XAie_Lock(id: " << std::to_string(lock.LockId)
     << ", val: " << std::to_string(lock.LockVal) << ")";
  return os;
}

static raw_ostream &operator<<(raw_ostream &os, const XAie_Packet &packet) {
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

#else

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                                     \
  do {                                                                         \
    if (auto r = API(__VA_ARGS__))                                             \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +             \
                               AIERCTOSTR.at(r));                              \
  } while (0)

#define TRY_XAIE_API_EMIT_ERROR(OP, API, ...)                                  \
  do {                                                                         \
    if (auto r = API(__VA_ARGS__))                                             \
      return OP.emitOpError() << #API " failed with " << AIERCTOSTR.at(r);     \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                                  \
  do {                                                                         \
    if (auto r = API(__VA_ARGS__)) {                                           \
      llvm::errs() << #API " failed with " << AIERCTOSTR.at(r);                \
      return failure();                                                        \
    }                                                                          \
  } while (0)

#endif

auto ps = std::filesystem::path::preferred_separator;

#define XAIE_BASE_ADDR 0x40000000
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_PARTITION_BASE_ADDR 0x0

#define NPI_ADDR 0x0
#define NUM_LOCKS 16
#define EVEN_BD_NUM_START 0
#define ODD_BD_NUM_START 24
#define MEM_TILE_LOCK_ID_INCR 64
#define BASE_ADDR_A_INCR 0x80000

static LogicalResult configureLocksInBdBlock(XAie_DmaDesc &dmaTileBd,
                                             Block &block,
                                             const AIETargetModel &targetModel,
                                             XAie_LocType &tileLoc) {
  LLVM_DEBUG(llvm::dbgs() << "\nstart configuring bds\n");
  std::optional<int> acqValue, relValue, acqLockId, relLockId;
  bool acqEn = false;

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
  bool relEn = false;
  XAie_Lock acqLock = XAie_LockInit(acqLockId.value(), acqValue.value());
  XAie_Lock relLock = XAie_LockInit(relLockId.value(), relValue.value());
  TRY_XAIE_API_EMIT_ERROR((*block.getOps<UseLockOp>().begin()),
                          dmaTileBd.DmaMod->SetLock, &dmaTileBd, acqLock,
                          relLock, acqEn, relEn);
  return success();
}

static LogicalResult configureBdInBlock(XAie_DevInst &devInst,
                                        XAie_DmaDesc &dmaTileBd, Block &block,
                                        const AIETargetModel &targetModel,
                                        XAie_LocType &tileLoc, int bdId,
                                        std::optional<int> nextBdId) {
  std::optional<int> packetType;
  std::optional<int> packetID;

  // Below should go
  auto maybePacketOps = block.getOps<DMABDPACKETOp>();
  if (!maybePacketOps.empty()) {
    assert(llvm::range_size(maybePacketOps) == 1 &&
           "expected only one dma_bd_packet");
    auto packetOp = *maybePacketOps.begin();
    packetType = packetOp.getPacketType();
    packetID = packetOp.getPacketID();
  }

  auto bdOp = *block.getOps<DMABDOp>().begin();

  if (targetModel.isShimNOCTile(tileLoc.Col, tileLoc.Row)) {
    // write them out like this so they show up with names in debug prints
    size_t smid = 0;
    size_t burstLen = 16; // (10):BLEN=16 (256Byte) (corresponds to
                          // 0x800000000 from target)
    size_t qOs = 0;
    size_t cache = 0;
    size_t secure = 0;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAxi, &dmaTileBd, smid, burstLen,
                            qOs, cache, secure);
  }

  // StringRef FifoMode = disable; // FIXME: when to enable FIFO mode?
  int baseAddr = 0;
  if (!targetModel.isShimNOCTile(tileLoc.Col, tileLoc.Row)) {
    auto bufferOp = cast<AIE::BufferOp>(bdOp.getBuffer().getDefiningOp());
    if (!bufferOp.getAddress())
      return bufferOp.emitError("buffer must have address assigned");
    baseAddr = bufferOp.getAddress().value();
    if (targetModel.isMemTile(tileLoc.Col, tileLoc.Row))
      baseAddr += BASE_ADDR_A_INCR;
  }

  std::optional<llvm::ArrayRef<BDDimLayoutAttr>> dims = bdOp.getDimensions();
  int lenInBytes = bdOp.getLenInBytes();
  int basePlusOffsetInBytes = baseAddr + bdOp.getOffsetInBytes();
  if (!dims) {
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAddrLen, &dmaTileBd,
                            basePlusOffsetInBytes, lenInBytes);
  } else {
    XAie_DmaTensor dmaTileBdTensor = {};
    dmaTileBdTensor.NumDim = dims->size();
    dmaTileBdTensor.Dim = static_cast<XAie_DmaDimDesc *>(
        calloc(dmaTileBdTensor.NumDim, sizeof(XAie_DmaDimDesc)));
    if (!dmaTileBdTensor.Dim)
      return bdOp.emitError("couldn't allocate array of XAie_DmaDimDesc");
    // libxaie requires stride in multiples of 32b
    double elementWidthIn32bWords =
        static_cast<double>(bdOp.getBufferElementTypeWidthInBytes()) / 4.0;
    for (size_t i = 0; i < dims->size(); i++) {
      // Pass down dimensions in reverse order; in the MLIR, this allows
      // us to specify step sizes/wraps in the same order as we would
      // access a multi-dim C array, with the highest dimension first.
      int j = dims->size() - i - 1;
      uint16_t size;
      uint32_t stride;
      if (j > 0) {
        stride = static_cast<uint32_t>(dims.value()[i].getStride() *
                                       elementWidthIn32bWords);
        size = dims.value()[i].getSize();
      } else {
        stride = dims.value()[i].getStride();
        size = static_cast<uint16_t>(dims.value()[i].getSize() *
                                     elementWidthIn32bWords);
      }
      stride = stride > 0 ? stride : 1;
      // Assume AIE-ML architecture (ie use AieMlDimDesc instead of AieDimDesc);
      // asserted in AIETranslateToCDODirect).
      dmaTileBdTensor.Dim[j].AieMlDimDesc = {stride, size};
    }
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetMultiDimAddr, &dmaTileBd,
                            &dmaTileBdTensor, basePlusOffsetInBytes,
                            lenInBytes);
  }

  // ND zero padding.
  std::optional<llvm::ArrayRef<BDPadLayoutAttr>> padDims =
      bdOp.getPadDimensions();

  if (padDims) {
    XAie_DmaPadTensor dmaPadTensor = {};
    dmaPadTensor.NumDim = padDims->size();
    dmaPadTensor.PadDesc = static_cast<XAie_PadDesc *>(
        calloc(dmaPadTensor.NumDim, sizeof(XAie_PadDesc)));
    if (!dmaPadTensor.PadDesc)
      return bdOp.emitError("couldn't allocate array of XAie_PadDesc");
    // libxaie requires stride in multiples of 32b
    double elementWidthIn32bWords =
        static_cast<double>(bdOp.getBufferElementTypeWidthInBytes()) / 4.0;
    for (size_t i = 0; i < padDims->size(); i++) {
      // Pass down dimensions in reverse order.
      int j = padDims->size() - i - 1;
      uint8_t before;
      uint8_t after;
      if (j > 0) {
        before = static_cast<uint8_t>(padDims.value()[i].getConstPadBefore());
        after = static_cast<uint8_t>(padDims.value()[i].getConstPadAfter());
      } else {
        before = static_cast<uint8_t>(padDims.value()[i].getConstPadBefore() *
                                      elementWidthIn32bWords);
        after = static_cast<uint8_t>(padDims.value()[i].getConstPadAfter() *
                                     elementWidthIn32bWords);
      }
      dmaPadTensor.PadDesc[j] = {before, after};
    }
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetPadding, &dmaTileBd,
                            &dmaPadTensor);
  }
  if (nextBdId) {
    auto enableNextBd = 1;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetNextBd, &dmaTileBd,
                            nextBdId.value(), enableNextBd);
  }

  if (auto packetInfo = bdOp.getPacket()) {
    packetType = packetInfo->getPktType();
    packetID = packetInfo->getPktId();
  }

  if (packetID) {
    if (!packetType)
      bdOp.emitError("must have packetType with packetID");
    if (bdOp.getLen() == 0)
      return bdOp.emitOpError(
          "For MM2S channels, if Buffer_Length=0 then Enable_Packet must be "
          "set to 0, otherwise behavior is undefined (3.7.8 arch spec)");
    TRY_XAIE_API_EMIT_ERROR(
        bdOp, XAie_DmaSetPkt, &dmaTileBd,
        XAie_PacketInit(packetID.value(), packetType.value()));
  }
  TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaEnableBd, &dmaTileBd);
  TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaWriteBd, &devInst, &dmaTileBd, tileLoc,
                          bdId);
  LLVM_DEBUG(llvm::dbgs() << "\nend configuring bds\n");
  return success();
};

static LogicalResult pushToBdQueueAndEnable(XAie_DevInst &devInst,
                                            Operation &op,
                                            XAie_LocType &tileLoc, int chNum,
                                            const DMAChannelDir &channelDir,
                                            int bdId, int repeatCount) {
  XAie_DmaDirection direction =
      channelDir == DMAChannelDir::S2MM ? DMA_S2MM : DMA_MM2S;
  auto enTokenIssue = tileLoc.Row == 0 && direction == DMA_S2MM;
  // in english repeat_count==0 means "do it once" and don't repeat but
  // libxaie treats repeat_count=1 as do it once.
  repeatCount += 1;
  TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelSetStartQueue, &devInst, tileLoc,
                          chNum, direction, bdId, repeatCount, enTokenIssue);
  TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelEnable, &devInst, tileLoc, chNum,
                          direction);
  return success();
};

static LogicalResult configureLocksAndBd(XAie_DevInst &devInst, Block &block,
                                         XAie_LocType tileLoc,
                                         const AIETargetModel &targetModel) {
  DMABDOp bd = *block.getOps<DMABDOp>().begin();
  assert(bd.getBdId().has_value() &&
         "DMABDOp must have assigned bd_id; did you forget to run "
         "aie-assign-bd-ids?");
  XAie_DmaDesc dmaTileBd;
  TRY_XAIE_API_EMIT_ERROR(bd, XAie_DmaDescInit, &devInst, &dmaTileBd, tileLoc);
  if (!block.getOps<UseLockOp>().empty() &&
      failed(configureLocksInBdBlock(dmaTileBd, block, targetModel, tileLoc)))
    return failure();
  if (!block.getOps<DMABDOp>().empty() &&
      failed(configureBdInBlock(devInst, dmaTileBd, block, targetModel, tileLoc,
                                bd.getBdId().value(), bd.getNextBdId())))
    return failure();
  return success();
};

namespace {
struct AIEControl {
  XAie_Config configPtr;
  XAie_DevInst devInst;

  AIEControl(bool aieSim, bool xaieDebug, const BaseNPUTargetModel &tm) {
    // The first column in the NPU lacks a shim tile.  AIE-RT exposes some of
    // the internals about how this is modeled in a somewhat awkward way.
    size_t partitionStartCol = tm.isVirtualized() ? 1 : 0;
    size_t partitionNumCols = tm.columns();
    size_t deviceRows = tm.rows();
    size_t deviceCols = tm.columns() + partitionStartCol;

    // Don't put this in the target model, because it's XAIE specific.
    unsigned char devGen;
    switch (tm.getTargetArch()) {
    case AIEArch::AIE1: // probably unreachable.
      devGen = XAIE_DEV_GEN_AIE;
      break;
    case AIEArch::AIE2:
      devGen = XAIE_DEV_GEN_AIEML;
      break;
    default:
      assert(false);
    }
    configPtr = XAie_Config{
        /*AieGen*/ devGen,
        /*BaseAddr*/ XAIE_BASE_ADDR,
        /*ColShift*/ (uint8_t)tm.getColumnShift(),
        /*RowShift*/ (uint8_t)tm.getRowShift(),
        /*NumRows*/ static_cast<uint8_t>(deviceRows),
        /*NumCols*/ static_cast<uint8_t>(deviceCols),
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
    TRY_XAIE_API_FATAL_ERROR(XAie_SetupPartitionConfig, &devInst,
                             XAIE_PARTITION_BASE_ADDR, partitionStartCol,
                             partitionNumCols);
    TRY_XAIE_API_FATAL_ERROR(XAie_CfgInitialize, &devInst, &configPtr);
    if (aieSim) {
      TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst,
                               XAIE_IO_BACKEND_SIM);
    } else if (xaieDebug)
      TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst,
                               XAIE_IO_BACKEND_DEBUG);
    else
      TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst,
                               XAIE_IO_BACKEND_CDO);

    TRY_XAIE_API_FATAL_ERROR(XAie_UpdateNpiAddr, &devInst, NPI_ADDR);
  }

  LogicalResult addAieElf(uint8_t col, uint8_t row, const StringRef elfPath,
                          bool aieSim) {
    TRY_XAIE_API_LOGICAL_RESULT(XAie_CoreDisable, &devInst,
                                XAie_TileLoc(col, row));
    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelResetAll, &devInst,
                                XAie_TileLoc(col, row),
                                XAie_DmaChReset::DMA_CHANNEL_RESET);

    // loadSym: Load symbols from .map file. This argument is not used when
    // __AIESIM__ is not defined.
    TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadElf, &devInst, XAie_TileLoc(col, row),
                                elfPath.str().c_str(), /*loadSym*/ aieSim);

    TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelResetAll, &devInst,
                                XAie_TileLoc(col, row),
                                XAie_DmaChReset::DMA_CHANNEL_UNRESET);

    return success();
  }

  LogicalResult addAieElfs(DeviceOp &targetOp, const StringRef workDirPath,
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
          if (failed(addAieElf(
                  col, row,
                  (llvm::Twine(workDirPath) + std::string(1, ps) + fileName)
                      .str(),
                  aieSim)))
            return failure();
        }
      }
    return success();
  }

  LogicalResult addInitConfig(DeviceOp &targetOp) {
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

    const AIETargetModel &targetModel = targetOp.getTargetModel();

    auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
    llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
    llvm::append_range(memOps, targetOp.getOps<ShimDMAOp>());
    for (TileElement memOp : memOps) {
      int col = memOp.getTileID().col;
      int row = memOp.getTileID().row;
      XAie_LocType tileLoc = XAie_TileLoc(col, row);

      // handle DMA ops separately
      auto dmaOps = llvm::to_vector_of<DMAOp>(
          memOp.getOperation()->getRegion(0).getOps<DMAOp>());
      if (!dmaOps.empty()) {
        for (auto dmaOp : dmaOps)
          for (auto &bdRegion : dmaOp.getBds()) {
            Block &block = bdRegion.getBlocks().front();
            if (failed(
                    configureLocksAndBd(devInst, block, tileLoc, targetModel)))
              return failure();
          }
      } else {
        for (Block &block : memOp.getOperation()->getRegion(0)) {
          if (block.getOps<DMABDOp>().empty())
            continue;
          if (failed(configureLocksAndBd(devInst, block, tileLoc, targetModel)))
            return failure();
        }
      }

      if (!dmaOps.empty())
        for (auto dmaOp : dmaOps) {
          auto &block = dmaOp.getBds().front().getBlocks().front();
          DMABDOp bd = *block.getOps<DMABDOp>().begin();
          if (failed(pushToBdQueueAndEnable(
                  devInst, *dmaOp.getOperation(), tileLoc,
                  dmaOp.getChannelIndex(), dmaOp.getChannelDir(),
                  bd.getBdId().value(), dmaOp.getRepeatCount())))
            return failure();
        }
      else
        for (Block &block : memOp.getOperation()->getRegion(0)) {
          for (auto op : block.getOps<DMAStartOp>()) {
            DMABDOp bd = *op.getDest()->getOps<DMABDOp>().begin();
            int chNum = op.getChannelIndex();
            auto channelDir = op.getChannelDir();
            if (failed(pushToBdQueueAndEnable(
                    devInst, *bd.getOperation(), tileLoc, chNum, channelDir,
                    bd.getBdId().value(), op.getRepeatCount())))
              return failure();
          }
        }
    }

    // StreamSwitch (switchbox) configuration
    for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
      int32_t col = switchboxOp.colIndex();
      int32_t row = switchboxOp.rowIndex();
      XAie_LocType tileLoc = XAie_TileLoc(col, row);
      assert(targetModel.isNPU() && "Only NPU currently supported");

      Block &b = switchboxOp.getConnections().front();
      for (auto connectOp : b.getOps<ConnectOp>())
        TRY_XAIE_API_EMIT_ERROR(
            switchboxOp, XAie_StrmConnCctEnable, &devInst, tileLoc,
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
            connectOp.sourceIndex(),
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
            connectOp.destIndex());

      for (auto masterSetOp : b.getOps<MasterSetOp>()) {
        int mask = 0;
        int arbiter = -1;

        for (auto val : masterSetOp.getAmsels()) {
          AMSelOp amsel = cast<AMSelOp>(val.getDefiningOp());
          arbiter = amsel.arbiterIndex();
          int msel = amsel.getMselValue();
          mask |= (1 << msel);
        }

        // the default is to keep header
        bool keepHeader = true;
        // the default for dma destinations is to drop the header
        if (masterSetOp.getDestBundle() == WireBundle::DMA)
          keepHeader = false;
        // assume a connection going south from row zero gets wired to shimdma
        // by a shimmux.
        if (switchboxOp.rowIndex() == 0 &&
            masterSetOp.getDestBundle() == WireBundle::South)
          keepHeader = false;

        // "keep_pkt_header" attribute overrides the above defaults, if set
        if (auto keep = masterSetOp.getKeepPktHeader())
          keepHeader = *keep;

        auto dropHeader = keepHeader ? XAIE_SS_PKT_DONOT_DROP_HEADER
                                     : XAIE_SS_PKT_DROP_HEADER;
        TRY_XAIE_API_EMIT_ERROR(
            masterSetOp, XAie_StrmPktSwMstrPortEnable, &devInst, tileLoc,
            WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(masterSetOp.getDestBundle()),
            masterSetOp.destIndex(), dropHeader, arbiter, mask);
      }

      for (auto packetRulesOp : b.getOps<PacketRulesOp>()) {
        int slot = 0;
        Block &block = packetRulesOp.getRules().front();
        for (auto slotOp : block.getOps<PacketRuleOp>()) {
          AMSelOp amselOp = cast<AMSelOp>(slotOp.getAmsel().getDefiningOp());
          int arbiter = amselOp.arbiterIndex();
          int msel = amselOp.getMselValue();
          TRY_XAIE_API_EMIT_ERROR(packetRulesOp, XAie_StrmPktSwSlavePortEnable,
                                  &devInst, tileLoc,
                                  WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
                                      packetRulesOp.getSourceBundle()),
                                  packetRulesOp.sourceIndex());
          auto packetInit = XAie_PacketInit(slotOp.valueInt(), /*PktType*/ 0);
          // TODO Need to better define packet id,type used here
          TRY_XAIE_API_EMIT_ERROR(packetRulesOp, XAie_StrmPktSwSlaveSlotEnable,
                                  &devInst, tileLoc,
                                  WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
                                      packetRulesOp.getSourceBundle()),
                                  packetRulesOp.sourceIndex(), slot, packetInit,
                                  slotOp.maskInt(), msel, arbiter);
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

  LogicalResult addCoreEnable(DeviceOp &targetOp) {
    // Start execution of all the cores.
    for (auto tileOp : targetOp.getOps<TileOp>()) {
      auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
      if (!tileOp.isShimTile() && tileOp.getCoreOp())
        TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreEnable, &devInst, tileLoc);
    }
    return success();
  }
};

} // namespace

static void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug) {
  // Enables AXI-MM prints for configs being added in CDO
  if (cdoDebug)
    EnAXIdebug();
  setEndianness(endianness);
};

static LogicalResult
generateCDOBinary(const StringRef outputPath,
                  const std::function<LogicalResult()> &cb) {

  // TODO(newling): Get bootgen team to remove print statement in this function.
  startCDOFileStream(outputPath.str().c_str());
  FileHeader();
  // Never generate a completely empty CDO file.  If the file only contains a
  // header, then bootgen flags it as invalid.
  insertNoOpCommand(4);
  if (failed(cb()))
    return failure();
  configureHeader();
  endCurrentCDOFileStream();
  return success();
}

static LogicalResult generateCDOBinariesSeparately(AIEControl &ctl,
                                                   const StringRef workDirPath,
                                                   DeviceOp &targetOp,
                                                   bool aieSim,
                                                   bool enableCores) {

  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_elfs.bin")
              .str(),
          [&ctl, &targetOp, &workDirPath, &aieSim] {
            return ctl.addAieElfs(targetOp, workDirPath, aieSim);
          })))
    return failure();

  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_init.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.addInitConfig(targetOp); })))
    return failure();

  if (enableCores &&
      failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_enable.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.addCoreEnable(targetOp); })))
    return failure();

  return success();
}

static LogicalResult generateCDOUnified(AIEControl &ctl,
                                        const StringRef workDirPath,
                                        DeviceOp &targetOp, bool aieSim,
                                        bool enableCores) {
  return generateCDOBinary(
      (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo.bin").str(),
      [&ctl, &targetOp, &workDirPath, &aieSim, &enableCores] {
        if (!targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addAieElfs(targetOp, workDirPath, aieSim)))
          return failure();
        if (failed(ctl.addInitConfig(targetOp)))
          return failure();
        if (enableCores && !targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addCoreEnable(targetOp)))
          return failure();
        return success();
      });
}

static LogicalResult
translateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                     byte_ordering endianness, bool emitUnified, bool cdoDebug,
                     bool aieSim, bool xaieDebug, bool enableCores) {

  auto devOps = m.getOps<DeviceOp>();
  assert(llvm::range_size(devOps) == 1 &&
         "only exactly 1 device op supported.");
  DeviceOp targetOp = *devOps.begin();
  const BaseNPUTargetModel &targetModel =
      (const BaseNPUTargetModel &)targetOp.getTargetModel();

  // things like XAIE_MEM_TILE_ROW_START and the missing
  // shim dma on tile (0,0) are hard-coded assumptions about NPU...
  assert(targetModel.isNPU() && "Only NPU currently supported");

  AIEControl ctl(aieSim, xaieDebug, targetModel);
  initializeCDOGenerator(endianness, cdoDebug);

  auto result = [&]() {
    if (emitUnified) {
      return generateCDOUnified(ctl, workDirPath, targetOp, aieSim,
                                enableCores);
    }
    return generateCDOBinariesSeparately(ctl, workDirPath, targetOp, aieSim,
                                         enableCores);
  }();
  return result;
}

namespace {

// An TransactionBinaryOperation encapulates an aie-rt TnxCmd struct
struct TransactionBinaryOperation {
  struct XAie_TxnCmd cmd;
  TransactionBinaryOperation(XAie_TxnOpcode opc, uint32_t mask, uint64_t addr,
                             uint32_t value, const uint8_t *data,
                             uint32_t size) {
    cmd.Opcode = opc;
    cmd.Mask = mask;
    cmd.RegOff = addr;
    cmd.Value = value;
    cmd.DataPtr = reinterpret_cast<uint64_t>(data);
    cmd.Size = size;
  }
};
} // namespace

// Parse a TXN binary blob. On success return the number of columns from the
// header and a vector of parsed operations. On failure return std::nullopt.
static std::optional<int>
parseTransactionBinary(const std::vector<uint8_t> &data,
                       std::vector<TransactionBinaryOperation> &ops) {

  uint32_t major = data[0];
  uint32_t minor = data[1];
  uint32_t num_cols = data[4];

  uint32_t num_ops, txn_size;
  std::memcpy(&num_ops, &data[8], 4);
  std::memcpy(&txn_size, &data[12], 4);

  LLVM_DEBUG(llvm::dbgs() << "Major: " << major << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Minor: " << minor << "\n");
  LLVM_DEBUG(llvm::dbgs() << "DevGen: " << data[2] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumRows: " << data[3] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumCols: " << num_cols << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumMemTileRows: " << data[5] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumOps: " << num_ops << "\n");
  LLVM_DEBUG(llvm::dbgs() << "TxnSize: " << txn_size << " bytes\n");

  size_t i = 16;

  // Convert opcode from uint8 to enum
  auto convertOpcode = [](uint8_t opc) {
    switch (opc) {
    case 0:
      return XAie_TxnOpcode::XAIE_IO_WRITE;
    case 1:
      return XAie_TxnOpcode::XAIE_IO_BLOCKWRITE;
    case 3:
      return XAie_TxnOpcode::XAIE_IO_MASKWRITE;
    default:
      llvm::errs() << "Unhandled opcode: " << std::to_string(opc) << "\n";
      return XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_MAX;
    }
  };

  // Parse the binary blob. There are two versions supported, 0.1 and 1.0.
  // For both versions, build a list of TransactionBinaryOperation objects
  // representing the parsed operations.
  if (major == 0 && minor == 1) {
    while (i < data.size()) {

      XAie_TxnOpcode opc = convertOpcode(data[i]);
      LLVM_DEBUG(llvm::dbgs() << "opcode: " + std::to_string(opc) + "\n");

      uint64_t addr = 0;
      uint32_t value = 0;
      uint32_t size = 0;
      uint32_t mask = 0;
      const uint8_t *data_ptr = nullptr;

      if (opc == XAie_TxnOpcode::XAIE_IO_WRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: WRITE (0x00)\n");
        uint32_t addr0, addr1;
        std::memcpy(&addr0, &data[i + 8], 4);
        std::memcpy(&addr1, &data[i + 12], 4);
        std::memcpy(&value, &data[i + 16], 4);
        std::memcpy(&size, &data[i + 20], 4);
        addr = static_cast<uint64_t>(addr1) << 32 | addr0;
        i += size;
      } else if (opc == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: BLOCKWRITE (0x01)\n");
        std::memcpy(&addr, &data[i + 8], 4);
        std::memcpy(&size, &data[i + 12], 4);
        data_ptr = data.data() + i + 16;
        i += size;
        size = size - 16;
      } else if (opc == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: MASKWRITE (0x03)\n");
        uint32_t addr0, addr1;
        std::memcpy(&addr0, &data[i + 8], 4);
        std::memcpy(&addr1, &data[i + 12], 4);
        std::memcpy(&value, &data[i + 16], 4);
        std::memcpy(&mask, &data[i + 20], 4);
        std::memcpy(&size, &data[i + 24], 4);
        addr = static_cast<uint64_t>(addr1) << 32 | addr0;
        i += size;
      } else {
        llvm::errs() << "Unhandled opcode: " << std::to_string(opc) << "\n";
        return std::nullopt;
      }
      ops.emplace_back(opc, mask, addr, value, data_ptr, size);
      LLVM_DEBUG(llvm::dbgs() << "addr: " << addr << "\n");
      LLVM_DEBUG(llvm::dbgs() << "value: " << value << "\n");
      LLVM_DEBUG(llvm::dbgs() << "size: " << size << "\n");
      LLVM_DEBUG(llvm::dbgs() << "mask: " << mask << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "data: " << reinterpret_cast<uintptr_t>(data_ptr) << "\n");
    }
  } else if (major == 1 && minor == 0) {
    while (i < data.size()) {

      XAie_TxnOpcode opc = convertOpcode(data[i]);
      LLVM_DEBUG(llvm::dbgs() << "opcode: " + std::to_string(opc) + "\n");

      uint64_t addr = 0;
      uint32_t value = 0;
      uint32_t size = 0;
      uint32_t mask = 0;
      const uint8_t *data_ptr = nullptr;

      if (opc == XAie_TxnOpcode::XAIE_IO_WRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: WRITE (0x00)\n");
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        i += 12;
      } else if (opc == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: BLOCKWRITE (0x01)\n");
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&size, &data[i + 8], 4);
        data_ptr = data.data() + i + 12;
        i += size;
        size = size - 12;
      } else if (opc == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: MASKWRITE (0x03)\n");
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        std::memcpy(&mask, &data[i + 12], 4);
        i += 16;
      } else {
        llvm::errs() << "Unhandled opcode: " << std::to_string(opc) << "\n";
        return std::nullopt;
      }
      LLVM_DEBUG(llvm::dbgs() << "addr: " << addr << "\n");
      LLVM_DEBUG(llvm::dbgs() << "value: " << value << "\n");
      LLVM_DEBUG(llvm::dbgs() << "size: " << size << "\n");
      LLVM_DEBUG(llvm::dbgs() << "mask: " << mask << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "data: " << reinterpret_cast<uintptr_t>(data_ptr) << "\n");
      ops.emplace_back(opc, mask, addr, value, data_ptr, size);
    }
  } else {
    llvm::errs() << "Unsupported TXN binary version: " << major << "." << minor
                 << "\n";
    return std::nullopt;
  }

  return num_cols;
}

static LogicalResult generateTxn(AIEControl &ctl, const StringRef workDirPath,
                                 DeviceOp &targetOp, bool aieSim,
                                 bool enableElfs, bool enableInit,
                                 bool enableCores) {
  if (enableElfs && !targetOp.getOps<CoreOp>().empty() &&
      failed(ctl.addAieElfs(targetOp, workDirPath, aieSim)))
    return failure();
  if (enableInit && failed(ctl.addInitConfig(targetOp)))
    return failure();
  if (enableCores && !targetOp.getOps<CoreOp>().empty() &&
      failed(ctl.addCoreEnable(targetOp)))
    return failure();
  return success();
}

static LogicalResult translateToTxn(ModuleOp m, std::vector<uint8_t> &output,
                                    llvm::StringRef workDirPath, bool aieSim,
                                    bool xaieDebug, bool enableCores) {

  auto devOps = m.getOps<DeviceOp>();
  if (llvm::range_size(devOps) > 1)
    return m.emitError("only exactly 1 device op supported.");

  DeviceOp targetOp = *devOps.begin();
  const BaseNPUTargetModel &targetModel =
      (const BaseNPUTargetModel &)targetOp.getTargetModel();

  if (!targetModel.isNPU())
    return failure();

  AIEControl ctl(aieSim, xaieDebug, targetModel);

  // start collecting transations
  XAie_StartTransaction(&ctl.devInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);

  auto result =
      generateTxn(ctl, workDirPath, targetOp, aieSim, true, true, true);
  if (failed(result))
    return result;

  // Export the transactions to a buffer
  uint8_t *txn_ptr = XAie_ExportSerializedTransaction(&ctl.devInst, 0, 0);
  XAie_TxnHeader *hdr = (XAie_TxnHeader *)txn_ptr;
  std::vector<uint8_t> txn_data(txn_ptr, txn_ptr + hdr->TxnSize);
  output.swap(txn_data);

  return success();
}

LogicalResult xilinx::AIE::AIETranslateToCDODirect(
    ModuleOp m, llvm::StringRef workDirPath, bool bigEndian, bool emitUnified,
    bool cdoDebug, bool aieSim, bool xaieDebug, bool enableCores) {
  byte_ordering endianness =
      bigEndian ? byte_ordering::Big_Endian : byte_ordering::Little_Endian;
  return translateToCDODirect(m, workDirPath, endianness, emitUnified, cdoDebug,
                              aieSim, xaieDebug, enableCores);
}

std::optional<mlir::ModuleOp>
xilinx::AIE::AIETranslateBinaryToTxn(mlir::MLIRContext *ctx,
                                     std::vector<uint8_t> &binary) {

  // parse the binary
  std::vector<TransactionBinaryOperation> operations;
  auto c = parseTransactionBinary(binary, operations);
  if (!c) {
    llvm::errs() << "Failed to parse binary\n";
    return std::nullopt;
  }
  int columns = *c;

  auto loc = mlir::UnknownLoc::get(ctx);

  // create a new ModuleOp and set the insertion point
  auto module = ModuleOp::create(loc);
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());

  // create aie.device
  std::vector<AIEDevice> devices{AIEDevice::npu1_1col, AIEDevice::npu1_2col,
                                 AIEDevice::npu1_3col, AIEDevice::npu1_4col,
                                 AIEDevice::npu1};
  auto device = builder.create<DeviceOp>(loc, devices[columns - 1]);
  device.getRegion().emplaceBlock();
  builder.setInsertionPointToStart(device.getBody());

  // for each blockwrite in the binary, create a GlobalOp with the data
  std::vector<memref::GlobalOp> global_data;
  for (auto &op : operations) {
    if (op.cmd.Opcode != XAIE_IO_BLOCKWRITE) {
      global_data.push_back(nullptr);
      continue;
    }
    uint32_t size = op.cmd.Size / 4;
    const uint32_t *d = reinterpret_cast<const uint32_t *>(op.cmd.DataPtr);
    std::vector<uint32_t> data32(d, d + size);

    int id = 0;
    std::string name = "blockwrite_data";
    while (device.lookupSymbol(name))
      name = "blockwrite_data_" + std::to_string(id++);

    MemRefType memrefType = MemRefType::get({size}, builder.getI32Type());
    TensorType tensorType = RankedTensorType::get({size}, builder.getI32Type());
    auto global = builder.create<memref::GlobalOp>(
        loc, name, builder.getStringAttr("private"), memrefType,
        DenseElementsAttr::get<uint32_t>(tensorType, data32), true, nullptr);
    global_data.push_back(global);
  }

  // create aiex.runtime_sequence
  auto seq = builder.create<AIEX::RuntimeSequenceOp>(loc, nullptr);
  seq.getBody().push_back(new Block);

  // create the txn ops
  builder.setInsertionPointToStart(&seq.getBody().front());
  for (auto p : llvm::zip(operations, global_data)) {
    auto op = std::get<0>(p);
    memref::GlobalOp payload = std::get<1>(p);

    if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_WRITE) {
      builder.create<AIEX::NpuWrite32Op>(loc, op.cmd.RegOff, op.cmd.Value,
                                         nullptr, nullptr, nullptr);
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
      auto memref = builder.create<memref::GetGlobalOp>(loc, payload.getType(),
                                                        payload.getName());
      builder.create<AIEX::NpuBlockWriteOp>(
          loc, builder.getUI32IntegerAttr(op.cmd.RegOff), memref.getResult(),
          nullptr, nullptr, nullptr);
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
      builder.create<AIEX::NpuMaskWrite32Op>(loc, op.cmd.RegOff, op.cmd.Value,
                                             op.cmd.Mask, nullptr, nullptr,
                                             nullptr);
    } else {
      llvm::errs() << "Unhandled txn opcode: " << op.cmd.Opcode << "\n";
      return std::nullopt;
    }
  }

  return module;
}

std::optional<mlir::ModuleOp>
xilinx::AIE::AIETranslateBinaryToCtrlpkt(mlir::MLIRContext *ctx,
                                         std::vector<uint8_t> &binary) {

  // parse the binary
  std::vector<TransactionBinaryOperation> operations;
  auto c = parseTransactionBinary(binary, operations);
  if (!c) {
    llvm::errs() << "Failed to parse binary\n";
    return std::nullopt;
  }
  int columns = *c;

  auto loc = mlir::UnknownLoc::get(ctx);

  // create a new ModuleOp and set the insertion point
  auto module = ModuleOp::create(loc);
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());

  // create aie.device
  std::vector<AIEDevice> devices{AIEDevice::npu1_1col, AIEDevice::npu1_2col,
                                 AIEDevice::npu1_3col, AIEDevice::npu1_4col,
                                 AIEDevice::npu1};
  auto device = builder.create<DeviceOp>(loc, devices[columns - 1]);
  device.getRegion().emplaceBlock();
  builder.setInsertionPointToStart(device.getBody());

  // for each blockwrite in the binary, create a GlobalOp with the data
  std::vector<memref::GlobalOp> global_data;
  for (auto &op : operations) {
    if (op.cmd.Opcode != XAIE_IO_BLOCKWRITE) {
      global_data.push_back(nullptr);
      continue;
    }
    uint32_t size = op.cmd.Size / 4;
    const uint32_t *d = reinterpret_cast<const uint32_t *>(op.cmd.DataPtr);
    std::vector<uint32_t> data32(d, d + size);

    int id = 0;
    std::string name = "blockwrite_data";
    while (device.lookupSymbol(name))
      name = "blockwrite_data_" + std::to_string(id++);

    MemRefType memrefType = MemRefType::get({size}, builder.getI32Type());
    TensorType tensorType = RankedTensorType::get({size}, builder.getI32Type());
    auto global = builder.create<memref::GlobalOp>(
        loc, name, builder.getStringAttr("private"), memrefType,
        DenseElementsAttr::get<uint32_t>(tensorType, data32), true, nullptr);
    global_data.push_back(global);
  }

  // create aiex.runtime_sequence
  auto seq = builder.create<AIEX::RuntimeSequenceOp>(loc, nullptr);
  seq.getBody().push_back(new Block);

  // create the txn ops
  builder.setInsertionPointToStart(&seq.getBody().front());
  for (auto p : llvm::zip(operations, global_data)) {
    auto op = std::get<0>(p);
    memref::GlobalOp payload = std::get<1>(p);

    if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_WRITE) {
      builder.create<AIEX::NpuControlPacketOp>(
          loc, builder.getUI32IntegerAttr(op.cmd.RegOff), nullptr,
          /*opcode*/ builder.getI32IntegerAttr(0),
          /*stream_id*/ builder.getI32IntegerAttr(0),
          DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(op.cmd.Value)));
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
      if (!std::get<1>(p).getInitialValue())
        continue;
      auto blockWriteData =
          dyn_cast<DenseIntElementsAttr>(*std::get<1>(p).getInitialValue());
      if (!blockWriteData) {
        payload.emitError(
            "Global symbol initial value is not a dense int array");
        break;
      }
      auto blockWriteDataValues = blockWriteData.getValues<int32_t>();
      // Split block write data into beats of 4 or less, in int32_t.
      int currAddr = op.cmd.RegOff;
      for (size_t i = 0; i < blockWriteDataValues.size(); i += 4) {
        auto last = std::min(blockWriteDataValues.size(), i + 4);
        SmallVector<int32_t> splitData =
            SmallVector<int32_t>(blockWriteDataValues.begin() + i,
                                 blockWriteDataValues.begin() + last);
        builder.create<AIEX::NpuControlPacketOp>(
            loc, builder.getUI32IntegerAttr(currAddr), nullptr,
            /*opcode*/ builder.getI32IntegerAttr(0),
            /*stream_id*/ builder.getI32IntegerAttr(0),
            DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(splitData)));
        currAddr += splitData.size() * sizeof(int32_t);
      }

    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
      builder.create<AIEX::NpuControlPacketOp>(
          loc, builder.getUI32IntegerAttr(op.cmd.RegOff), nullptr,
          /*opcode*/ builder.getI32IntegerAttr(0),
          /*stream_id*/ builder.getI32IntegerAttr(0),
          DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(op.cmd.Value)));
    } else {
      llvm::errs() << "Unhandled txn opcode: " << op.cmd.Opcode << "\n";
      return std::nullopt;
    }
  }

  return module;
}

LogicalResult xilinx::AIE::AIETranslateToTxn(ModuleOp m,
                                             llvm::raw_ostream &output,
                                             llvm::StringRef workDirPath,
                                             bool outputBinary, bool enableSim,
                                             bool xaieDebug, bool enableCores) {
  std::vector<uint8_t> bin;
  auto result =
      translateToTxn(m, bin, workDirPath, enableSim, xaieDebug, enableCores);
  if (failed(result))
    return result;

  if (outputBinary) {
    output.write(reinterpret_cast<const char *>(bin.data()), bin.size());
    return success();
  }

  auto new_module = AIETranslateBinaryToTxn(m.getContext(), bin);
  if (!new_module)
    return failure();
  new_module->print(output);
  return success();
}

LogicalResult xilinx::AIE::AIETranslateToControlPackets(
    ModuleOp m, llvm::raw_ostream &output, llvm::StringRef workDirPath,
    bool outputBinary, bool enableSim, bool xaieDebug, bool enableCores) {
  std::vector<uint8_t> bin;
  auto result =
      translateToTxn(m, bin, workDirPath, enableSim, xaieDebug, enableCores);
  if (failed(result))
    return result;

  if (outputBinary) {
    output.write(reinterpret_cast<const char *>(bin.data()), bin.size());
    return success();
  }

  auto new_module = AIETranslateBinaryToCtrlpkt(m.getContext(), bin);
  if (!new_module)
    return failure();
  new_module->print(output);
  return success();
}
