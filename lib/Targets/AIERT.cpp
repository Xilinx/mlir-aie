//===- AIERT.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIERT.h"
#include "aie/Targets/AIETargetShared.h"

#include "mlir/Support/LogicalResult.h"

extern "C" {
#include "xaiengine/xaie_core.h"
#include "xaiengine/xaie_dma.h"
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaie_locks.h"
#include "xaiengine/xaie_mem.h"
#include "xaiengine/xaie_plif.h"
#include "xaiengine/xaie_ss.h"
#include "xaiengine/xaie_txn.h"
#include "xaiengine/xaiegbl.h"
#include "xaiengine/xaiegbl_defs.h"
}

#include <filesystem>

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

static const std::map<xilinx::AIE::WireBundle, StrmSwPortType>
    WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE = {
        {xilinx::AIE::WireBundle::Core, StrmSwPortType::CORE},
        {xilinx::AIE::WireBundle::DMA, StrmSwPortType::DMA},
        {xilinx::AIE::WireBundle::TileControl, StrmSwPortType::CTRL},
        {xilinx::AIE::WireBundle::FIFO, StrmSwPortType::FIFO},
        {xilinx::AIE::WireBundle::South, StrmSwPortType::SOUTH},
        {xilinx::AIE::WireBundle::West, StrmSwPortType::WEST},
        {xilinx::AIE::WireBundle::North, StrmSwPortType::NORTH},
        {xilinx::AIE::WireBundle::East, StrmSwPortType::EAST},
        // missing PLIO from WireBundle
        // missing NOC from WireBundle
        {xilinx::AIE::WireBundle::Trace, StrmSwPortType::TRACE},
};

#ifndef NDEBUG

// https://stackoverflow.com/a/32230306
template <typename H1>
llvm::raw_ostream &showAIEXRTArgs(llvm::raw_ostream &out, const char *label,
                                  H1 &&value) {
  return out << label << "=" << std::forward<H1>(value);
}

template <typename H1, typename... T>
llvm::raw_ostream &showAIEXRTArgs(llvm::raw_ostream &out, const char *label,
                                  H1 &&value, T &&...rest) {
  const char *pcomma = strchr(label, ',');
  return showAIEXRTArgs(out.write(label, pcomma - label)
                            << "=" << std::forward<H1>(value) << ',',
                        pcomma + 1, std::forward<T>(rest)...);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_LocType &loc);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_Lock &lock);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_Packet &packet);

#define SHOW_AIERT_ARGS(os, ...) showAIEXRTArgs(os, #__VA_ARGS__, __VA_ARGS__)

// So that we can use the pattern if(auto r = TRY_XAIE_API...) { // r is nonzero
// }
static_assert(XAIE_OK == 0);

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                                     \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_AIERT_ARGS(llvm::dbgs(), __VA_ARGS__));                    \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__))                                             \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +             \
                               AIERCTOSTR.at(r));                              \
  } while (0)

#define TRY_XAIE_API_EMIT_ERROR(OP, API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_AIERT_ARGS(llvm::dbgs(), __VA_ARGS__));                    \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                          \
    if (auto r = API(__VA_ARGS__))                                             \
      return OP.emitOpError() << #API " failed with " << AIERCTOSTR.at(r);     \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                                  \
  do {                                                                         \
    LLVM_DEBUG(llvm::dbgs() << "trying XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_AIERT_ARGS(llvm::dbgs(), __VA_ARGS__));                    \
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

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "aie-aiert"

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_LocType &loc) {
  os << "XAie_LocType(col: " << std::to_string(loc.Col)
     << ", row: " << std::to_string(loc.Row) << ")";
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XAie_Lock &lock) {
  os << "XAie_Lock(id: " << std::to_string(lock.LockId)
     << ", val: " << std::to_string(lock.LockVal) << ")";
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const XAie_Packet &packet) {
  os << "XAie_Packet(id: " << std::to_string(packet.PktId)
     << ", type: " << std::to_string(packet.PktType) << ")";
  return os;
}

#define XAIE_BASE_ADDR 0x40000000
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_PARTITION_BASE_ADDR 0x0

#define NPI_ADDR 0x0
#define NUM_LOCKS 16
#define EVEN_BD_NUM_START 0
#define ODD_BD_NUM_START 24

struct xilinx::AIE::AIERTControl::AIERtImpl {
  XAie_Config configPtr;
  XAie_DevInst devInst;
};

xilinx::AIE::AIERTControl::~AIERTControl() = default;

xilinx::AIE::AIERTControl::AIERTControl(const AIE::AIETargetModel &tm)
    : targetModel(tm), aiert(std::make_unique<AIERtImpl>()) {
  // The first column in the NPU lacks a shim tile.  AIE-RT exposes some of
  // the internals about how this is modeled in a somewhat awkward way.
  size_t partitionStartCol =
      tm.hasProperty(AIE::AIETargetModel::IsVirtualized) ? 1 : 0;
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
    // FIXME: What if we don't have an IPU?  aie-rt
    // models non-IPU devices differently.
    devGen = XAIE_DEV_GEN_AIE2IPU;
    break;
  case AIEArch::AIE2p:
    devGen = XAIE_DEV_GEN_AIE2P_STRIX_B0;
    break;
  }
  aiert->configPtr = XAie_Config{
      /*AieGen*/ devGen,
      /*BaseAddr*/ XAIE_BASE_ADDR,
      /*ColShift*/ static_cast<uint8_t>(tm.getColumnShift()),
      /*RowShift*/ static_cast<uint8_t>(tm.getRowShift()),
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
  XAie_InstDeclare(_devInst, &aiert->configPtr);
  aiert->devInst = _devInst;
  TRY_XAIE_API_FATAL_ERROR(XAie_SetupPartitionConfig, &aiert->devInst,
                           XAIE_PARTITION_BASE_ADDR, partitionStartCol,
                           partitionNumCols);
  TRY_XAIE_API_FATAL_ERROR(XAie_CfgInitialize, &aiert->devInst,
                           &aiert->configPtr);
  TRY_XAIE_API_FATAL_ERROR(XAie_UpdateNpiAddr, &aiert->devInst, NPI_ADDR);
}

LogicalResult xilinx::AIE::AIERTControl::setIOBackend(bool aieSim,
                                                      bool xaieDebug) {
  // Quoting: The instance of a device must be always declared using this
  // macro. In the future, the same macro will be expanded to
  // allocate more memory from the user application for resource
  // management.
  if (aieSim) {
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &aiert->devInst,
                             XAIE_IO_BACKEND_SIM);
  } else if (xaieDebug)
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &aiert->devInst,
                             XAIE_IO_BACKEND_DEBUG);
  else
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &aiert->devInst,
                             XAIE_IO_BACKEND_CDO);
  return success();
}

LogicalResult configureLocksInBdBlock(const AIE::AIETargetModel &targetModel,
                                      XAie_DmaDesc &dmaTileBd, Block &block,
                                      int col, int row) {
  LLVM_DEBUG(llvm::dbgs() << "\nstart configuring bds\n");
  std::optional<int> acqValue, relValue, acqLockId, relLockId;
  bool acqEn = false;

  // switch (lock->getAc)
  AIE::LockOp lock;
  for (auto op : block.getOps<AIE::UseLockOp>()) {
    // Only dyn_cast if you are going to check if it was of the type
    // expected; if you aren't checking use cast instead as it will at
    // least assert in debug mode with an easier to understand error than
    // dereferencing.
    lock = cast<AIE::LockOp>(op.getLock().getDefiningOp());
    switch (op.getAction()) {
    case AIE::LockAction::Acquire:
    case AIE::LockAction::AcquireGreaterEqual:
      acqEn = op.getAcqEn();
      acqLockId = lock.getLockIDValue();
      acqValue = op.getLockValue();
      if (op.acquireGE())
        acqValue.value() = -acqValue.value();
      break;
    case AIE::LockAction::Release:
      relLockId = lock.getLockIDValue();
      relValue = op.getLockValue();
      break;
    }
  }

  assert(acqValue && relValue && acqLockId && relLockId &&
         "expected both use_lock(acquire) and use_lock(release) with bd");

  if (targetModel.isMemTile(col, row)) {
    auto lockOffset = targetModel.getLockLocalBaseIndex(
        col, row, lock.colIndex(), lock.rowIndex());
    if (lockOffset && acqLockId)
      acqLockId.value() += lockOffset.value();
    if (lockOffset && relLockId)
      relLockId.value() += lockOffset.value();
  }

  // no RelEn in the arch spec even though the API requires you to set it?
  bool relEn = false;
  XAie_Lock acqLock = XAie_LockInit(acqLockId.value(), acqValue.value());
  XAie_Lock relLock = XAie_LockInit(relLockId.value(), relValue.value());
  TRY_XAIE_API_EMIT_ERROR((*block.getOps<AIE::UseLockOp>().begin()),
                          dmaTileBd.DmaMod->SetLock, &dmaTileBd, acqLock,
                          relLock, acqEn, relEn);
  return success();
}

LogicalResult configureBdInBlock(const AIE::AIETargetModel &targetModel,
                                 XAie_DevInst *devInst, XAie_DmaDesc &dmaTileBd,
                                 Block &block, int col, int row, int bdId,
                                 std::optional<int> nextBdId) {
  std::optional<int> packetType;
  std::optional<int> packetID;

  // Below should go
  auto maybePacketOps = block.getOps<AIE::DMABDPACKETOp>();
  if (!maybePacketOps.empty()) {
    assert(llvm::range_size(maybePacketOps) == 1 &&
           "expected only one dma_bd_packet");
    auto packetOp = *maybePacketOps.begin();
    packetType = packetOp.getPacketType();
    packetID = packetOp.getPacketID();
  }

  auto bdOp = *block.getOps<AIE::DMABDOp>().begin();

  if (targetModel.isShimNOCTile(col, row)) {
    // write them out like this so they show up with names in debug prints
    uint8_t smid = 0;
    uint32_t burstLen =
        getShimBurstLengthBytes(targetModel, bdOp.getBurstLength());
    uint8_t qOs = 0;
    uint8_t cache = 0;
    uint8_t secure = 0;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAxi, &dmaTileBd, smid,
                            burstLen / 16, qOs, cache, secure);
  }

  // get address from BufferOp (core,mem) or ExternalBufferOp (shim)
  uint64_t baseAddr = 0;
  if (targetModel.isShimNOCTile(col, row)) {
    auto bufferOp =
        cast<AIE::ExternalBufferOp>(bdOp.getBuffer().getDefiningOp());
    // external buffers aren't required to have an address here because the
    // address might get patched later or the default of zero might be a valid
    // address.
    if (bufferOp.getAddress())
      baseAddr = bufferOp.getAddress().value();
  } else {
    auto bufferOp = cast<AIE::BufferOp>(bdOp.getBuffer().getDefiningOp());
    if (!bufferOp.getAddress())
      return bufferOp.emitError("buffer must have address assigned");
    baseAddr = bufferOp.getAddress().value();
  }

  if (targetModel.isMemTile(col, row)) {
    // check if buffer is allocated on the same memtile, the west, or the east
    // one
    auto bufferOp = cast<AIE::BufferOp>(bdOp.getBuffer().getDefiningOp());
    auto bufferRow = bufferOp.getTileOp().getRow();
    auto bufferCol = bufferOp.getTileOp().getCol();
    auto addrOffset =
        targetModel.getMemLocalBaseAddress(col, row, bufferCol, bufferRow);
    if (addrOffset)
      baseAddr += addrOffset.value();
  }

  std::optional<llvm::ArrayRef<AIE::BDDimLayoutAttr>> dims =
      bdOp.getDimensions();
  uint64_t lenInBytes = bdOp.getLenInBytes();
  uint64_t basePlusOffsetInBytes = baseAddr + bdOp.getOffsetInBytes();
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
  std::optional<llvm::ArrayRef<AIE::BDPadLayoutAttr>> padDims =
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
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaWriteBd, devInst, &dmaTileBd, tileLoc,
                          bdId);
  LLVM_DEBUG(llvm::dbgs() << "\nend configuring bds\n");
  return success();
};

LogicalResult xilinx::AIE::AIERTControl::pushToBdQueueAndEnable(
    Operation &op, int col, int row, int chNum, const DMAChannelDir &channelDir,
    int bdId, int repeatCount) {
  XAie_DmaDirection direction =
      channelDir == DMAChannelDir::S2MM ? DMA_S2MM : DMA_MM2S;
  auto tileLoc = XAie_TileLoc(col, row);
  auto enTokenIssue = tileLoc.Row == 0 && direction == DMA_S2MM;
  // in english repeat_count==0 means "do it once" and don't repeat but
  // libxaie treats repeat_count=1 as do it once.
  repeatCount += 1;
  TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelSetStartQueue, &aiert->devInst,
                          tileLoc, chNum, direction, bdId, repeatCount,
                          enTokenIssue);
  TRY_XAIE_API_EMIT_ERROR(op, XAie_DmaChannelEnable, &aiert->devInst, tileLoc,
                          chNum, direction);
  return success();
};

LogicalResult xilinx::AIE::AIERTControl::configureLocksAndBd(Block &block,
                                                             int col, int row) {
  DMABDOp bd = *block.getOps<DMABDOp>().begin();
  assert(bd.getBdId().has_value() &&
         "DMABDOp must have assigned bd_id; did you forget to run "
         "aie-assign-bd-ids?");
  XAie_DmaDesc dmaTileBd;
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_EMIT_ERROR(bd, XAie_DmaDescInit, &aiert->devInst, &dmaTileBd,
                          tileLoc);
  if (!block.getOps<UseLockOp>().empty() &&
      failed(configureLocksInBdBlock(targetModel, dmaTileBd, block, col, row)))
    return failure();
  if (!block.getOps<DMABDOp>().empty() &&
      failed(configureBdInBlock(targetModel, &aiert->devInst, dmaTileBd, block,
                                col, row, bd.getBdId().value(),
                                bd.getNextBdId())))
    return failure();
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::initLocks(DeviceOp &targetOp) {
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
    if (!tileOp.isShimTile() && tileOp.getCoreOp()) {
      TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreReset, &aiert->devInst, tileLoc);
      TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_CoreUnreset, &aiert->devInst,
                              tileLoc);
      // Set locks to zero
      for (uint8_t l = 0; l < NUM_LOCKS; l++) {
        auto locInit = XAie_LockInit(l, 0);
        TRY_XAIE_API_EMIT_ERROR(tileOp, XAie_LockSetValue, &aiert->devInst,
                                tileLoc, locInit);
      }
    }
  }

  // Set locks with explicit initializers
  targetOp.walk<WalkOrder::PreOrder>([&](LockOp lockOp) {
    if (lockOp.getLockID() && lockOp.getInit()) {
      auto tileLoc = XAie_TileLoc(lockOp.getTileOp().colIndex(),
                                  lockOp.getTileOp().rowIndex());
      auto locInit = XAie_LockInit(*lockOp.getLockID(), *lockOp.getInit());
      TRY_XAIE_API_FATAL_ERROR(XAie_LockSetValue, &aiert->devInst, tileLoc,
                               locInit);
    } else
      LLVM_DEBUG(llvm::dbgs()
                 << "lock op missing either id or init" << lockOp << "\n");
  });
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::initBuffers(DeviceOp &targetOp) {
  // Set buffers with explicit initializers
  targetOp.walk<WalkOrder::PreOrder>([&](BufferOp bufferOp) {
    auto initialValue = bufferOp.getInitialValue();
    if (!initialValue)
      return;
    mlir::DenseElementsAttr denseInit =
        dyn_cast<mlir::DenseElementsAttr>(initialValue.value());
    if (!denseInit)
      return;
    auto tileLoc = XAie_TileLoc(bufferOp.getTileOp().colIndex(),
                                bufferOp.getTileOp().rowIndex());
    std::vector<char> byteVec;
    if (denseInit.getElementType().isIntOrIndex()) {
      for (auto intVal : denseInit.getValues<APInt>()) {
        // Get the size in bytes
        size_t byteSize = (intVal.getBitWidth() + 7) / 8;
        // Create a buffer for the integer bytes and copy
        std::vector<char> bytes(byteSize);
        std::copy(
            static_cast<const char *>(static_cast<const void *>(&intVal)),
            static_cast<const char *>(static_cast<const void *>(&intVal)) +
                byteSize,
            bytes.begin());
        byteVec.insert(byteVec.end(), bytes.begin(), bytes.end());
      }
    } else if (isa<FloatType>(denseInit.getElementType())) {
      for (auto floatVal : denseInit.getValues<APFloat>()) {
        APInt floatInt = floatVal.bitcastToAPInt();
        // Get the size in bytes
        size_t byteSize = (floatInt.getBitWidth() + 7) / 8;
        // Create a buffer for the float bytes and copy
        std::vector<char> bytes(byteSize);
        std::copy(
            static_cast<const char *>(static_cast<const void *>(&floatInt)),
            static_cast<const char *>(static_cast<const void *>(&floatInt)) +
                byteSize,
            bytes.begin());
        byteVec.insert(byteVec.end(), bytes.begin(), bytes.end());
      }
    } else {
      llvm::outs() << "buffer op type not supported for initialization "
                   << bufferOp << "\n";
      return;
    }
    TRY_XAIE_API_FATAL_ERROR(XAie_DataMemBlockWrite, &aiert->devInst, tileLoc,
                             bufferOp.getAddress().value(), byteVec.data(),
                             byteVec.size());
  });
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::configureSwitches(DeviceOp &targetOp) {

  // StreamSwitch (switchbox) configuration
  for (auto switchboxOp : targetOp.getOps<SwitchboxOp>()) {
    int32_t col = switchboxOp.colIndex();
    int32_t row = switchboxOp.rowIndex();
    XAie_LocType tileLoc = XAie_TileLoc(col, row);
    assert(targetModel.hasProperty(AIETargetModel::IsNPU) &&
           "Only NPU currently supported");

    Block &b = switchboxOp.getConnections().front();
    for (auto connectOp : b.getOps<ConnectOp>())
      TRY_XAIE_API_EMIT_ERROR(
          switchboxOp, XAie_StrmConnCctEnable, &aiert->devInst, tileLoc,
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

      auto dropHeader =
          keepHeader ? XAIE_SS_PKT_DONOT_DROP_HEADER : XAIE_SS_PKT_DROP_HEADER;
      TRY_XAIE_API_EMIT_ERROR(
          masterSetOp, XAie_StrmPktSwMstrPortEnable, &aiert->devInst, tileLoc,
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
                                &aiert->devInst, tileLoc,
                                WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
                                    packetRulesOp.getSourceBundle()),
                                packetRulesOp.sourceIndex());
        auto packetInit = XAie_PacketInit(slotOp.valueInt(), /*PktType*/ 0);
        // TODO Need to better define packet id,type used here
        TRY_XAIE_API_EMIT_ERROR(packetRulesOp, XAie_StrmPktSwSlaveSlotEnable,
                                &aiert->devInst, tileLoc,
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
                                &aiert->devInst, tileLoc,
                                connectOp.sourceIndex());
      // mux
      if (connectOp.getDestBundle() == WireBundle::North)
        TRY_XAIE_API_EMIT_ERROR(muxOp, XAie_EnableShimDmaToAieStrmPort,
                                &aiert->devInst, tileLoc,
                                connectOp.destIndex());
    }
  }

  for (auto switchboxOp : targetOp.getOps<ShimSwitchboxOp>()) {
    Block &b = switchboxOp.getConnections().front();
    auto tileLoc = XAie_TileLoc(switchboxOp.getCol(), 0);
    for (auto connectOp : b.getOps<ConnectOp>())
      TRY_XAIE_API_EMIT_ERROR(
          switchboxOp, XAie_StrmConnCctEnable, &aiert->devInst, tileLoc,
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getSourceBundle()),
          connectOp.sourceIndex(),
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(connectOp.getDestBundle()),
          connectOp.destIndex());
  }

  // Cascade configuration
  if (isa<AIE2TargetModel>(targetModel)) {
    for (auto configOp : targetOp.getOps<ConfigureCascadeOp>()) {
      TileOp tile = cast<TileOp>(configOp.getTile().getDefiningOp());
      auto tileLoc = XAie_TileLoc(tile.getCol(), tile.getRow());
      TRY_XAIE_API_EMIT_ERROR(
          targetOp, XAie_CoreConfigAccumulatorControl, &aiert->devInst, tileLoc,
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
              static_cast<WireBundle>(configOp.getInputDir())),
          WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(
              static_cast<WireBundle>(configOp.getOutputDir())));
    }
  }

  return success();
}

LogicalResult xilinx::AIE::AIERTControl::addInitConfig(DeviceOp &targetOp) {

  if (failed(initLocks(targetOp))) {
    return failure();
  }

  if (failed(initBuffers(targetOp))) {
    return failure();
  }

  auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
  llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
  llvm::append_range(memOps, targetOp.getOps<ShimDMAOp>());
  for (TileElement memOp : memOps) {
    int col = memOp.getTileID().col;
    int row = memOp.getTileID().row;

    // Get the region's entry block, then start traversing through the chain of
    // blocks.
    llvm::SetVector<Block *> blockVector =
        getOrderedChainOfBlocks(&memOp.getOperation()->getRegion(0));

    // handle DMA ops separately
    auto dmaOps = llvm::to_vector_of<DMAOp>(
        memOp.getOperation()->getRegion(0).getOps<DMAOp>());
    if (!dmaOps.empty()) {
      for (auto dmaOp : dmaOps)
        for (auto &bdRegion : dmaOp.getBds()) {
          Block &block = bdRegion.getBlocks().front();
          if (failed(configureLocksAndBd(block, col, row)))
            return failure();
        }
    } else {
      for (Block *block : blockVector) {
        if (block->getOps<DMABDOp>().empty())
          continue;
        if (failed(configureLocksAndBd(*block, col, row)))
          return failure();
      }
    }

    if (!dmaOps.empty())
      for (auto dmaOp : dmaOps) {
        auto &block = dmaOp.getBds().front().getBlocks().front();
        DMABDOp bd = *block.getOps<DMABDOp>().begin();
        if (failed(pushToBdQueueAndEnable(
                *dmaOp.getOperation(), col, row, dmaOp.getChannelIndex(),
                dmaOp.getChannelDir(), bd.getBdId().value(),
                dmaOp.getRepeatCount())))
          return failure();
      }
    else
      for (Block *block : blockVector) {
        for (auto op : block->getOps<DMAStartOp>()) {
          DMABDOp bd = *op.getDest()->getOps<DMABDOp>().begin();
          int chNum = op.getChannelIndex();
          auto channelDir = op.getChannelDir();
          if (failed(pushToBdQueueAndEnable(*bd.getOperation(), col, row, chNum,
                                            channelDir, bd.getBdId().value(),
                                            op.getRepeatCount())))
            return failure();
        }
      }
  }

  if (failed(configureSwitches(targetOp))) {
    return failure();
  }

  return success();
}

LogicalResult xilinx::AIE::AIERTControl::addCoreEnable(DeviceOp &targetOp) {
  // Start execution of all the cores.
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
    if (!tileOp.isShimTile() && tileOp.getCoreOp())
      TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreEnable, &aiert->devInst,
                              tileLoc);
  }
  return success();
}

LogicalResult xilinx::AIE::AIERTControl::addAieElf(uint8_t col, uint8_t row,
                                                   const StringRef elfPath,
                                                   bool aieSim) {
  TRY_XAIE_API_LOGICAL_RESULT(XAie_CoreDisable, &aiert->devInst,
                              XAie_TileLoc(col, row));
  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelResetAll, &aiert->devInst,
                              XAie_TileLoc(col, row),
                              XAie_DmaChReset::DMA_CHANNEL_RESET);

  // loadSym: Load symbols from .map file. This argument is not used when
  // __AIESIM__ is not defined.
  TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadElf, &aiert->devInst,
                              XAie_TileLoc(col, row), elfPath.str().c_str(),
                              /*loadSym*/ aieSim);

  TRY_XAIE_API_LOGICAL_RESULT(XAie_DmaChannelResetAll, &aiert->devInst,
                              XAie_TileLoc(col, row),
                              XAie_DmaChReset::DMA_CHANNEL_UNRESET);

  return success();
}

LogicalResult xilinx::AIE::AIERTControl::addAieElfs(DeviceOp &targetOp,
                                                    const StringRef elfPath,
                                                    bool aieSim) {
  for (auto tileOp : targetOp.getOps<TileOp>())
    if (tileOp.isShimNOCorPLTile()) {
      // Resets no needed with V2 kernel driver
    } else {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      if (auto coreOp = tileOp.getCoreOp()) {
        std::string fileName;
        if (auto fileAttr = coreOp.getElfFile()) {
          fileName = fileAttr->str();
        } else {
          coreOp.emitOpError()
              << "Expected lowered ELF file to be given as attribute "
                 "`elf_file` for this core. Compile cores first.";
          return failure();
        }
        // Check if fileName is already an absolute path.
        // If so, use it directly. Otherwise, concatenate with elfPath.
        std::string fullPath;
        if (std::filesystem::path(fileName).is_absolute()) {
          fullPath = fileName;
        } else {
          auto ps = std::filesystem::path::preferred_separator;
          fullPath =
              (llvm::Twine(elfPath) + std::string(1, ps) + fileName).str();
        }
        if (failed(addAieElf(col, row, fullPath, aieSim)))
          return failure();
      }
    }
  return success();
}

void xilinx::AIE::AIERTControl::dmaUpdateBdAddr(int col, int row, size_t addr,
                                                size_t bdId) {
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_FATAL_ERROR(XAie_DmaUpdateBdAddr, &aiert->devInst, tileLoc, addr,
                           bdId);
}

void xilinx::AIE::AIERTControl::startTransaction() {
  TRY_XAIE_API_FATAL_ERROR(XAie_StartTransaction, &aiert->devInst,
                           XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
}

std::vector<uint8_t> xilinx::AIE::AIERTControl::exportSerializedTransaction() {
  // Export the transactions to a binary buffer
  uint8_t *txn_ptr = XAie_ExportSerializedTransaction(&aiert->devInst, 0, 0);
  XAie_TxnHeader *hdr = (XAie_TxnHeader *)txn_ptr;
  std::vector<uint8_t> txn_data(txn_ptr, txn_ptr + hdr->TxnSize);
  return txn_data;
}
