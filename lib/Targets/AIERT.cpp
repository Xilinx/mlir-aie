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

#include "mlir/Support/LogicalResult.h"

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

#include <filesystem>

using namespace mlir;

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

namespace xilinx::AIE {

AIERTControl::AIERTControl(const AIE::BaseNPUTargetModel &tm)
    : targetModel(tm) {
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
  XAie_InstDeclare(_devInst, &configPtr);
  devInst = _devInst;
  TRY_XAIE_API_FATAL_ERROR(XAie_SetupPartitionConfig, &devInst,
                           XAIE_PARTITION_BASE_ADDR, partitionStartCol,
                           partitionNumCols);
  TRY_XAIE_API_FATAL_ERROR(XAie_CfgInitialize, &devInst, &configPtr);
  TRY_XAIE_API_FATAL_ERROR(XAie_UpdateNpiAddr, &devInst, NPI_ADDR);
}

LogicalResult AIERTControl::setIOBackend(bool aieSim, bool xaieDebug) {
  // Quoting: The instance of a device must be always declared using this
  // macro. In the future, the same macro will be expanded to
  // allocate more memory from the user application for resource
  // management.
  if (aieSim) {
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst, XAIE_IO_BACKEND_SIM);
  } else if (xaieDebug)
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst,
                             XAIE_IO_BACKEND_DEBUG);
  else
    TRY_XAIE_API_FATAL_ERROR(XAie_SetIOBackend, &devInst, XAIE_IO_BACKEND_CDO);
  return success();
}

LogicalResult AIERTControl::configureLocksInBdBlock(XAie_DmaDesc &dmaTileBd,
                                                    Block &block,
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

LogicalResult AIERTControl::configureBdInBlock(XAie_DmaDesc &dmaTileBd,
                                               Block &block,
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

LogicalResult
AIERTControl::pushToBdQueueAndEnable(Operation &op, XAie_LocType &tileLoc,
                                     int chNum, const DMAChannelDir &channelDir,
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

LogicalResult AIERTControl::configureLocksAndBd(Block &block,
                                                XAie_LocType tileLoc) {
  DMABDOp bd = *block.getOps<DMABDOp>().begin();
  assert(bd.getBdId().has_value() &&
         "DMABDOp must have assigned bd_id; did you forget to run "
         "aie-assign-bd-ids?");
  XAie_DmaDesc dmaTileBd;
  TRY_XAIE_API_EMIT_ERROR(bd, XAie_DmaDescInit, &devInst, &dmaTileBd, tileLoc);
  if (!block.getOps<UseLockOp>().empty() &&
      failed(configureLocksInBdBlock(dmaTileBd, block, tileLoc)))
    return failure();
  if (!block.getOps<DMABDOp>().empty() &&
      failed(configureBdInBlock(dmaTileBd, block, tileLoc, bd.getBdId().value(),
                                bd.getNextBdId())))
    return failure();
  return success();
}

LogicalResult AIERTControl::initLocks(DeviceOp &targetOp) {
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
  return success();
}

LogicalResult AIERTControl::configureSwitches(DeviceOp &targetOp) {

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

      auto dropHeader =
          keepHeader ? XAIE_SS_PKT_DONOT_DROP_HEADER : XAIE_SS_PKT_DROP_HEADER;
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

LogicalResult AIERTControl::addInitConfig(DeviceOp &targetOp) {

  if (failed(initLocks(targetOp))) {
    return failure();
  }

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
          if (failed(configureLocksAndBd(block, tileLoc)))
            return failure();
        }
    } else {
      for (Block &block : memOp.getOperation()->getRegion(0)) {
        if (block.getOps<DMABDOp>().empty())
          continue;
        if (failed(configureLocksAndBd(block, tileLoc)))
          return failure();
      }
    }

    if (!dmaOps.empty())
      for (auto dmaOp : dmaOps) {
        auto &block = dmaOp.getBds().front().getBlocks().front();
        DMABDOp bd = *block.getOps<DMABDOp>().begin();
        if (failed(pushToBdQueueAndEnable(
                *dmaOp.getOperation(), tileLoc, dmaOp.getChannelIndex(),
                dmaOp.getChannelDir(), bd.getBdId().value(),
                dmaOp.getRepeatCount())))
          return failure();
      }
    else
      for (Block &block : memOp.getOperation()->getRegion(0)) {
        for (auto op : block.getOps<DMAStartOp>()) {
          DMABDOp bd = *op.getDest()->getOps<DMABDOp>().begin();
          int chNum = op.getChannelIndex();
          auto channelDir = op.getChannelDir();
          if (failed(pushToBdQueueAndEnable(*bd.getOperation(), tileLoc, chNum,
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

LogicalResult AIERTControl::addCoreEnable(DeviceOp &targetOp) {
  // Start execution of all the cores.
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
    if (!tileOp.isShimTile() && tileOp.getCoreOp())
      TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreEnable, &devInst, tileLoc);
  }
  return success();
}

LogicalResult AIERTControl::addAieElf(uint8_t col, uint8_t row,
                                      const StringRef elfPath, bool aieSim) {
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

LogicalResult AIERTControl::addAieElfs(DeviceOp &targetOp,
                                       const StringRef elfPath, bool aieSim) {
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
        auto ps = std::filesystem::path::preferred_separator;
        if (failed(addAieElf(
                col, row,
                (llvm::Twine(elfPath) + std::string(1, ps) + fileName).str(),
                aieSim)))
          return failure();
      }
    }
  return success();
}

void AIERTControl::dmaUpdateBdAddr(int col, int row, size_t addr, size_t bdId) {
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_FATAL_ERROR(XAie_DmaUpdateBdAddr, &devInst, tileLoc, addr, bdId);
}

void AIERTControl::startTransaction() {
  TRY_XAIE_API_FATAL_ERROR(XAie_StartTransaction, &devInst,
                           XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
}

void AIERTControl::exportSerializedTransaction() {
  XAie_TxnInst *txnInst = XAie_ExportTransactionInstance(&devInst);
  std::ios_base::fmtflags f(std::cout.flags());
  for (size_t i = 0; i < txnInst->NumCmds; ++i) {
    std::cout.flags(f);
    std::cout << "Txn OpCode: " << std::hex
              << AIETXNOPCODETOSTR.at(txnInst->CmdBuf[i].Opcode) << "\n";
    std::cout.flags(f);
    std::cout << "RegOff: 0x" << std::hex << txnInst->CmdBuf[i].RegOff << "\n";
    std::cout.flags(f);
    std::cout << "Value: 0x" << std::hex << txnInst->CmdBuf[i].Value << "\n";
    std::cout.flags(f);
    std::cout << "Mask: 0x" << std::hex << txnInst->CmdBuf[i].Mask << "\n";
  }
}

} // namespace xilinx::AIE
