//===- AIERTX.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIERTX.h"

#include "mlir/Support/LogicalResult.h"

using namespace mlir;

#define DEBUG_TYPE "aie-aiertx"

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
AIERTXControl::AIERTXControl(size_t partitionStartCol, size_t partitionNumCols,
                       const AIETargetModel &tm)
    : targetModel(tm) {
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
  XAie_InstDeclare(_devInst, &configPtr);
  devInst = _devInst;
  TRY_XAIE_API_FATAL_ERROR(XAie_SetupPartitionConfig, &devInst,
                           XAIE_PARTITION_BASE_ADDR, partitionStartCol,
                           partitionNumCols);
  TRY_XAIE_API_FATAL_ERROR(XAie_CfgInitialize, &devInst, &configPtr);
  TRY_XAIE_API_FATAL_ERROR(XAie_UpdateNpiAddr, &devInst, NPI_ADDR);
}

LogicalResult AIERTXControl::setIOBackend(bool aieSim, bool xaieDebug) {
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

LogicalResult AIERTXControl::configureLocksInBdBlock(XAie_DmaDesc &dmaTileBd,
                                                  Block &block,
                                                  XAie_LocType &tileLoc) {
  LLVM_DEBUG(llvm::dbgs() << "\nstart configuring bds\n");
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
  bool relEn = false;
  XAie_Lock acqLock = XAie_LockInit(acqLockId.value(), acqValue.value());
  XAie_Lock relLock = XAie_LockInit(relLockId.value(), relValue.value());
  TRY_XAIE_API_EMIT_ERROR((*block.getOps<UseLockOp>().begin()),
                          dmaTileBd.DmaMod->SetLock, &dmaTileBd, acqLock,
                          relLock, acqEn, relEn);
  return success();
}

LogicalResult AIERTXControl::configureBdInBlock(XAie_DmaDesc &dmaTileBd,
                                             Block &block,
                                             XAie_LocType &tileLoc, int bdId,
                                             std::optional<int> nextBdId) {
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

  auto bdOp = *block.getOps<DMABDOp>().begin();

  if (targetModel.isShimNOCTile(tileLoc.Col, tileLoc.Row)) {
    // write them out like this so they show up with names in debug prints
    size_t smid = 0;
    size_t burstLen = 16; // (10):BLEN=16 (256Byte) (corresponds to
                          // 0x800000000 from targetipu)
    size_t qOs = 0;
    size_t cache = 0;
    size_t secure = 0;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAxi, &dmaTileBd, smid, burstLen,
                            qOs, cache, secure);
  }

  // deref here because this is a const iter and the various getters below
  // aren't const (even though they probably should be...)
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
  int lenInBytes = bdOp.getLenValue() * bytes;
  int basePlusOffset = baseAddr + bdOp.getOffsetValue();
  if (!dims) {
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetAddrLen, &dmaTileBd,
                            basePlusOffset, lenInBytes);
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
                            &dmaTileBdTensor, basePlusOffset, lenInBytes);
  }

  if (nextBdId) {
    auto enableNextBd = 1;
    TRY_XAIE_API_EMIT_ERROR(bdOp, XAie_DmaSetNextBd, &dmaTileBd,
                            nextBdId.value(), enableNextBd);
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
                          bdId);
  LLVM_DEBUG(llvm::dbgs() << "\nend configuring bds\n");
  return success();
};

LogicalResult
AIERTXControl::pushToBdQueueAndEnable(Operation &op, XAie_LocType &tileLoc,
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

LogicalResult AIERTXControl::configureLocksAndBd(Block &block,
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

LogicalResult AIERTXControl::initLocks(DeviceOp &targetOp) {
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

LogicalResult AIERTXControl::configureSwitches(DeviceOp &targetOp) {

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

LogicalResult AIERTXControl::enableCoresInDevice(DeviceOp &targetOp) {
  // Start execution of all the cores.
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    auto tileLoc = XAie_TileLoc(tileOp.colIndex(), tileOp.rowIndex());
    if (!tileOp.isShimTile() && tileOp.getCoreOp())
      TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_CoreEnable, &devInst, tileLoc);
  }
  return success();
}

LogicalResult AIERTXControl::dmaUpdateBdAddr(DeviceOp &targetOp, int col, int row,
                                          size_t addr, size_t bdId) {
  auto tileLoc = XAie_TileLoc(col, row);
  TRY_XAIE_API_EMIT_ERROR(targetOp, XAie_DmaUpdateBdAddr, &devInst, tileLoc,
                          addr, bdId);
  return success();
}

} // namespace xilinx::AIE
