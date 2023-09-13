//===- AIEObjectFifoStatefulTransform.cpp ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: October 18th 2021
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <numeric>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-objectFifo-stateful-transform"

#define LOOP_VAR_DEPENDENCY -2

//===----------------------------------------------------------------------===//
// Conversion Pattern
//===----------------------------------------------------------------------===//
template <typename MyOp>
struct AIEOpRemoval : public OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  using OpAdaptor = typename MyOp::Adaptor;

  AIEOpRemoval(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(MyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    rewriter.eraseOp(Op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lock Analysis
//===----------------------------------------------------------------------===//
class LockAnalysis {
  DenseMap<std::pair<Value, int>, int> locksPerTile;

public:
  LockAnalysis(DeviceOp &device) {
    // go over the locks created for each tile and update the index in
    // locksPerTile
    for (auto lockOp : device.getOps<LockOp>()) {
      auto tile = lockOp.getTile();
      auto lockID = lockOp.getLockIDValue();
      locksPerTile[std::make_pair(tile, lockID)] = 1;
    }
  }

  /// Given a tile, returns next usable lockID for that tile.
  int getLockID(TileOp &tileOp) {
    const auto &target_model = xilinx::AIE::getTargetModel(tileOp);
    for (unsigned i = 0;
         i < target_model.getNumLocks(tileOp.getCol(), tileOp.getRow()); i++) {
      int usageCnt = locksPerTile[std::make_pair(tileOp, i)];
      if (usageCnt == 0) {
        locksPerTile[std::make_pair(tileOp, i)] = 1;
        return i;
      }
    }
    return -1;
  }
};

//===----------------------------------------------------------------------===//
// TileDMA Channel Analysis
//===----------------------------------------------------------------------===//
class DMAChannelAnalysis {
  DenseMap<Value, int> masterChannelsPerTile;
  DenseMap<Value, int> slaveChannelsPerTile;

public:
  DMAChannelAnalysis(DeviceOp &device) {
    // go over the channels used for each tile and update the master/slave
    // channel maps
    for (auto memOp : device.getOps<MemOp>()) {
      Region &r = memOp.getBody();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          if (op.isSend())
            getMasterDMAChannel(memOp.getTile());
          else
            getSlaveDMAChannel(memOp.getTile());
        }
      }
    }
  }

  /// Given an AIE tile, returns its next usable master channel.
  xilinx::AIE::DMAChannel getMasterDMAChannel(Value tile) {
    xilinx::AIE::DMAChannel dmaChan;
    if (masterChannelsPerTile.find(tile) == masterChannelsPerTile.end()) {
      masterChannelsPerTile[tile] = 0;
    } else {
      assert([&] {
        TileOp tileOp = tile.getDefiningOp<TileOp>();
        int numChannels = tileOp.getNumSourceConnections(WireBundle::DMA);
        if (masterChannelsPerTile[tile] >= (numChannels - 1)) {
          printf("All tile DMA master channels are already in use.\n");
          return false;
        }
        return true;
      }());
      masterChannelsPerTile[tile]++;
    }
    dmaChan = std::make_pair(DMAChannelDir::MM2S, masterChannelsPerTile[tile]);
    return dmaChan;
  }

  /// Given an AIE tile, returns its next usable slave channel.
  xilinx::AIE::DMAChannel getSlaveDMAChannel(Value tile) {
    xilinx::AIE::DMAChannel dmaChan;
    if (slaveChannelsPerTile.find(tile) == slaveChannelsPerTile.end()) {
      slaveChannelsPerTile[tile] = 0;
    } else {
      assert([&] {
        TileOp tileOp = tile.getDefiningOp<TileOp>();
        int numChannels = tileOp.getNumDestConnections(WireBundle::DMA);
        if (slaveChannelsPerTile[tile] >= (numChannels - 1)) {
          printf("All tile DMA slave channels are already in use.\n");
          return false;
        }
        return true;
      }());
      slaveChannelsPerTile[tile]++;
    }
    dmaChan = std::make_pair(DMAChannelDir::S2MM, slaveChannelsPerTile[tile]);
    return dmaChan;
  }
};

//===----------------------------------------------------------------------===//
// Create objectFifos Pass
//===----------------------------------------------------------------------===//
struct AIEObjectFifoStatefulTransformPass
    : public AIEObjectFifoStatefulTransformBase<
          AIEObjectFifoStatefulTransformPass> {
  DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>>
      buffersPerFifo; // maps each objFifo to its corresponding buffer
  DenseMap<ObjectFifoCreateOp, std::vector<ExternalBufferOp>>
      externalBuffersPerFifo; // maps each objFifo to its corresponding
                              // external buffers
  DenseMap<ObjectFifoCreateOp, std::vector<LockOp>>
      locksPerFifo; // maps each objFifo to its corresponding locks
  std::vector<std::pair<ObjectFifoCreateOp, std::vector<ObjectFifoCreateOp>>>
      splitFifos; // maps each objFifo between non-adjacent tiles to its
                  // corresponding consumer objectFifos
  DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp>
      objFifoLinks; // maps each ObjectFifoLinkOp to objFifo whose elements
                    // have been created and should be used

  /// Function that returns true if two tiles in the AIE array share a memory
  /// module. share_direction is equal to:
  ///   * -1 if the shared memory module is that of the first input tile,
  ///   * 1 if it is that of the second input tile,
  ///   * 0 is no memory module is shared.
  bool isSharedMemory(TileOp a, TileOp b, int *share_direction) {
    const auto &target_model = getTargetModel(a.getOperation());

    if ((a.isShimTile() && !b.isShimTile()) ||
        (!a.isShimTile() && b.isShimTile())) {
      *share_direction = 0;
      return false;
    }
    if ((target_model.isMemTile(a.getCol(), a.getRow()) &&
         !target_model.isMemTile(b.getCol(), b.getRow())) ||
        (!target_model.isMemTile(a.getCol(), a.getRow()) &&
         target_model.isMemTile(b.getCol(), b.getRow()))) {
      *share_direction = 0;
      return false;
    }
    bool rightShared = target_model.isLegalMemAffinity(
        a.colIndex(), a.rowIndex(), b.colIndex(), b.rowIndex());

    bool leftShared = target_model.isLegalMemAffinity(
        b.colIndex(), b.rowIndex(), a.colIndex(), a.rowIndex());

    if (leftShared)
      *share_direction = -1;
    else if (rightShared)
      *share_direction = 1;
    else
      *share_direction = 0;

    return leftShared || rightShared;
  }

  /// Function to multiply all dimensions of a memref.
  int64_t getMemrefTypeSize(MemRefType memref) {
    int64_t size = 1;
    for (auto dim : memref.getShape())
      size *= dim;
    return size;
  }

  /// Function to retrieve ObjectFifoLinkOp of ObjectFifoCreateOp,
  /// if it belongs to one.
  std::optional<ObjectFifoLinkOp> getOptionalLinkOp(ObjectFifoCreateOp op) {
    auto device = op->getParentOfType<DeviceOp>();
    for (auto linkOp : device.getOps<ObjectFifoLinkOp>()) {
      for (auto in : linkOp.getInputObjectFifos())
        if (in.name() == op.name())
          return {linkOp};
      for (auto out : linkOp.getOutputObjectFifos())
        if (out.name() == op.name())
          return {linkOp};
    }
    return {};
  }

  ObjectFifoCreateOp createObjectFifo(OpBuilder &builder,
                                      AIEObjectFifoType datatype,
                                      std::string name, Value prodTile,
                                      Value consTile, Attribute depth) {
    auto ofName = builder.getStringAttr(name);
    ObjectFifoCreateOp fifo = builder.create<ObjectFifoCreateOp>(
        builder.getUnknownLoc(), ofName, prodTile, consTile, depth, datatype);
    return fifo;
  }

  /// Function used to create objectFifo locks based on target architecture.
  /// Called by createObjectFifoElements().
  std::vector<LockOp> createObjectFifoLocks(OpBuilder &builder,
                                            LockAnalysis &lockAnalysis,
                                            ObjectFifoCreateOp op, int numElem,
                                            TileOp creation_tile) {
    std::vector<LockOp> locks;
    auto dev = op->getParentOfType<xilinx::AIE::DeviceOp>();
    auto &target = dev.getTargetModel();
    if (creation_tile.isShimTile())
      numElem = externalBuffersPerFifo[op].size();
    if (target.getTargetArch() == xilinx::AIE::AIEArch::AIE1) {
      int of_elem_index = 0; // used to give objectFifo elements a symbolic name
      for (int i = 0; i < numElem; i++) {
        // create corresponding aie1 locks
        int lockID = lockAnalysis.getLockID(creation_tile);
        assert(lockID >= 0 && "No more locks to allocate!");
        LockOp lock = builder.create<LockOp>(builder.getUnknownLoc(),
                                             creation_tile, lockID, 0);
        lock.getOperation()->setAttr(
            mlir::SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name().str() + "_lock_" +
                                  std::to_string(of_elem_index)));
        locks.push_back(lock);
        of_elem_index++;
      }
    } else {
      // create corresponding aie2 locks
      int prodLockID = lockAnalysis.getLockID(creation_tile);
      assert(prodLockID >= 0 && "No more locks to allocate!");
      LockOp prodLock = builder.create<LockOp>(
          builder.getUnknownLoc(), creation_tile, prodLockID, numElem);
      prodLock.getOperation()->setAttr(
          mlir::SymbolTable::getSymbolAttrName(),
          builder.getStringAttr(op.name().str() + "_prod_lock"));
      locks.push_back(prodLock);

      int consLockID = lockAnalysis.getLockID(creation_tile);
      assert(consLockID >= 0 && "No more locks to allocate!");
      LockOp consLock = builder.create<LockOp>(builder.getUnknownLoc(),
                                               creation_tile, consLockID, 0);
      consLock.getOperation()->setAttr(
          mlir::SymbolTable::getSymbolAttrName(),
          builder.getStringAttr(op.name().str() + "_cons_lock"));
      locks.push_back(consLock);
    }
    return locks;
  }

  /// Function used to create objectFifo elements and their locks.
  /// It maps the input objectFifo to associated buffers and locks.
  void createObjectFifoElements(OpBuilder &builder, LockAnalysis &lockAnalysis,
                                ObjectFifoCreateOp op, int share_direction) {
    if (!op.size())
      return;

    std::vector<BufferOp> buffers;
    std::vector<LockOp> locks;
    AIEObjectFifoType fifo = op.getElemType().cast<AIEObjectFifoType>();
    MemRefType elemType = fifo.getElementType().cast<MemRefType>();
    int numElem = op.size();
    int of_elem_index = 0; // used to give objectFifo elements a symbolic name

    // if this objectFifo is linked to another, check if the other's elements
    // have already been created (the elements that are created are those of
    // the objFifo with elements of bigger size)
    bool linked = false;
    auto linkOp = getOptionalLinkOp(op);
    if (linkOp) {
      auto fifoIn = linkOp->getInputObjectFifos()[0];
      auto fifoOut = linkOp->getOutputObjectFifos()[0];
      linked = true;
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
        return; // elements have already been created
      if (linkOp->isJoin()) {
        // if join, fifoOut has bigger size
        if (op.name() != fifoOut.name())
          return;
      } else if (linkOp->isDistribute()) {
        // if distribute, fifoIn has bigger size
        if (op.name() != fifoIn.name())
          return;
      } else {
        AIEObjectFifoType fifoInType = linkOp->getInputObjectFifos()[0]
                                           .getElemType()
                                           .cast<AIEObjectFifoType>();
        MemRefType elemInType = fifoInType.getElementType().cast<MemRefType>();
        int inSize = getMemrefTypeSize(elemInType);

        AIEObjectFifoType fifoOutType = linkOp->getOutputObjectFifos()[0]
                                            .getElemType()
                                            .cast<AIEObjectFifoType>();
        MemRefType elemOutType =
            fifoOutType.getElementType().cast<MemRefType>();
        int outSize = getMemrefTypeSize(elemOutType);

        if (inSize >= outSize) {
          if (op.name() != fifoIn.name())
            return;
        } else {
          if (linkOp->getOutputObjectFifos()[0] != op)
            return;
        }
      }
    }

    TileOp creation_tile;
    if (share_direction == 0 || share_direction == -1)
      creation_tile = op.getProducerTileOp();
    else {
      TileOp consumerTileOp =
          dyn_cast<TileOp>(op.getConsumerTiles()[0].getDefiningOp());
      creation_tile = consumerTileOp;
    }

    builder.setInsertionPointAfter(op);
    for (int i = 0; i < numElem; i++) {
      // if shimTile external buffers are collected from input code
      // create as many locks as there are external buffers
      if (!creation_tile.isShimTile()) {
        BufferOp buff = builder.create<BufferOp>(builder.getUnknownLoc(),
                                                 elemType, creation_tile);
        buff.getOperation()->setAttr(
            mlir::SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name().str() + "_buff_" +
                                  std::to_string(of_elem_index)));
        buffers.push_back(buff);
      }
      of_elem_index++;
    }
    if (linked) {
      if (linkOp->isDistribute())
        numElem *= linkOp->getFifoOuts().size();
      else if (linkOp->isJoin())
        numElem *= linkOp->getFifoIns().size();
      objFifoLinks[*linkOp] = op;
    }
    locks = createObjectFifoLocks(builder, lockAnalysis, op, numElem,
                                  creation_tile);
    buffersPerFifo[op] = buffers;
    locksPerFifo[op] = locks;
  }

  /// Function that returns a pointer to the block of a Region
  /// that contains the AIEEndOp.
  Block *findEndOpBlock(Region &r) {
    Block *endBlock = nullptr;
    for (auto &bl : r.getBlocks())
      if (!bl.getOps<EndOp>().empty())
        endBlock = &bl;
    return endBlock;
  }

  /// Function used to create a Bd block.
  template <typename MyOp>
  void createBd(OpBuilder &builder, LockOp acqLock, int acqMode,
                LockAction acqLockAction, LockOp relLock, int relMode,
                MyOp buff, int offset, int len, Block *succ) {
    builder.create<UseLockOp>(builder.getUnknownLoc(), acqLock, acqMode,
                              acqLockAction);
    builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, 0);
    builder.create<UseLockOp>(builder.getUnknownLoc(), relLock, relMode,
                              LockAction::Release);
    builder.create<NextBDOp>(builder.getUnknownLoc(), succ);
  }

  /// Function used to create a Bd block.
  /// If lockMode is 0 we create a consumerDMA (i.e. on producer tile) else a
  /// producerDMA (i.e. on consumer tile).
  template <typename MyOp>
  void createBdBlock(OpBuilder &builder, ObjectFifoCreateOp op, int lockMode,
                     int acqNum, int relNum, MyOp buff, int offset, int len,
                     DMAChannelDir channelDir, int blockIndex, Block *succ) {
    LockOp acqLock;
    LockOp relLock;
    int acqMode = 1;
    int relMode = 1;
    LockAction acqLockAction = LockAction::Acquire;
    auto dev = op->getParentOfType<xilinx::AIE::DeviceOp>();
    auto &target = dev.getTargetModel();
    if (target.getTargetArch() == xilinx::AIE::AIEArch::AIE1) {
      acqMode = lockMode == 0 ? 1 : 0;
      relMode = lockMode == 0 ? 0 : 1;
      acqLock = locksPerFifo[op][blockIndex];
      relLock = locksPerFifo[op][blockIndex];
    } else {
      acqMode = acqNum;
      relMode = relNum;
      acqLockAction = LockAction::AcquireGreaterEqual;
      acqLock = (channelDir == DMAChannelDir::S2MM) ? locksPerFifo[op][0]
                                                    : locksPerFifo[op][1];
      relLock = (channelDir == DMAChannelDir::S2MM) ? locksPerFifo[op][1]
                                                    : locksPerFifo[op][0];
    }
    createBd(builder, acqLock, acqMode, acqLockAction, relLock, relMode, buff,
             offset, len, succ);
  }

  /// Function that either calls createAIETileDMA(), createShimDMA() or
  /// createMemTileDMA() based on op tile row value.
  void createDMA(DeviceOp &device, OpBuilder &builder, ObjectFifoCreateOp op,
                 DMAChannelDir channelDir, int channelIndex, int lockMode) {
    if (op.getProducerTileOp().isShimTile())
      createShimDMA(device, builder, op, channelDir, channelIndex, lockMode);
    else if (op.getProducerTileOp().isMemTile())
      createMemTileDMA(device, builder, op, channelDir, channelIndex, lockMode);
    else
      createAIETileDMA(device, builder, op, channelDir, channelIndex, lockMode);
  }

  /// Function used to create a MemOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createAIETileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode) {
    int numBlocks = op.size();
    if (numBlocks == 0)
      return;

    int acqNum = 1;
    int relNum = 1;
    int offset = 0;

    AIEObjectFifoType fifo = op.getElemType().cast<AIEObjectFifoType>();
    MemRefType elemType = fifo.getElementType().cast<MemRefType>();
    int len = getMemrefTypeSize(elemType);

    // search for the buffers/locks (based on if this objFifo has a link)
    ObjectFifoCreateOp target = op;
    auto linkOp = getOptionalLinkOp(op);
    if (linkOp)
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
        target = objFifoLinks[*linkOp];

    // search for MemOp
    Operation *producerMem = nullptr;
    for (auto memOp : device.getOps<MemOp>()) {
      if (memOp.getTile() == op.getProducerTile()) {
        producerMem = memOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = target.getProducerTileOp();
    if (producerMem == nullptr) {
      if (device->getNumRegions() != 1)
        assert(false && "expected num regions for device op");
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(device.getBody());
      MemOp newMemOp =
          builder.create<MemOp>(builder.getUnknownLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newMemOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerMem = newMemOp.getOperation();
    }
    Block *endBlock = findEndOpBlock(producerMem->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, bdBlock, endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ = nullptr;
    Block *curr = bdBlock;
    int blockIndex = 0;
    for (int i = 0; i < numBlocks; i++) {
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      builder.setInsertionPointToStart(curr);
      createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                              buffersPerFifo[target][blockIndex], offset, len,
                              channelDir, blockIndex, succ);
      curr = succ;
      blockIndex++;
    }
  }

  /// Function used to create a ShimDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createShimDMA(DeviceOp &device, OpBuilder &builder,
                     ObjectFifoCreateOp op, DMAChannelDir channelDir,
                     int channelIndex, int lockMode) {
    int numBlocks = externalBuffersPerFifo[op].size();
    if (numBlocks == 0)
      return;

    int acqNum = 1;
    int relNum = 1;
    int offset = 0;

    // search for ShimDMAOp
    Operation *producerDMA = nullptr;
    for (auto dmaOp : device.getOps<ShimDMAOp>()) {
      if (dmaOp.getTile() == op.getProducerTile()) {
        producerDMA = dmaOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = op.getProducerTileOp();
    if (producerDMA == nullptr) {
      if (device->getNumRegions() != 1)
        assert(false && "expected num regions for device op");
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(device.getBody());
      ShimDMAOp newDMAOp = builder.create<ShimDMAOp>(
          builder.getUnknownLoc(), builder.getIndexType(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, bdBlock, endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    int blockIndex = 0;
    for (int i = 0; i < numBlocks; i++) {
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      MemRefType buffer = externalBuffersPerFifo[op][blockIndex].getType();
      int len = getMemrefTypeSize(buffer);
      builder.setInsertionPointToStart(curr);
      createBdBlock<ExternalBufferOp>(builder, op, lockMode, acqNum, relNum,
                                      externalBuffersPerFifo[op][blockIndex],
                                      offset, len, channelDir, blockIndex,
                                      succ);
      curr = succ;
      blockIndex++;
    }
  }

  /// Function used to create a MemTileDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createMemTileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode) {
    int numBlocks = op.size();
    if (numBlocks == 0)
      return;

    int offset = 0;
    AIEObjectFifoType fifo = op.getElemType().cast<AIEObjectFifoType>();
    MemRefType elemType = fifo.getElementType().cast<MemRefType>();
    int lenOut = getMemrefTypeSize(elemType);
    int bytes = elemType.getElementTypeBitWidth() / 8;
    int acqNum = 1;
    int relNum = 1;

    // search for the buffers/locks (based on if this objFifo has a link)
    // identify size difference between input and output memrefs
    ObjectFifoCreateOp target = op;
    bool isDistribute = false;
    bool isJoin = false;
    int extraOffset = 0;
    auto linkOp = getOptionalLinkOp(op);
    if (linkOp) {
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end()) {
        target = objFifoLinks[*linkOp];

        if (linkOp->isJoin()) {
          // find offset based on order of this op in join list
          isJoin = true;
          if (target == op) {
            acqNum = linkOp->getFifoIns().size();
            relNum = linkOp->getFifoIns().size();
          } else {
            for (auto fifoIn : linkOp->getInputObjectFifos()) {
              AIEObjectFifoType fifoType =
                  fifoIn.getElemType().cast<AIEObjectFifoType>();
              MemRefType elemType =
                  fifoType.getElementType().cast<MemRefType>();
              if (fifoIn.name() == op.name())
                break;
              else
                extraOffset += (int)elemType.getShape()[0];
            }
          }
        } else if (linkOp->isDistribute()) {
          // find offset based on order of this op in distribute list
          isDistribute = true;
          if (target == op) {
            acqNum = linkOp->getFifoOuts().size();
            relNum = linkOp->getFifoOuts().size();
          } else {
            for (auto fifoOut : linkOp->getOutputObjectFifos()) {
              AIEObjectFifoType fifoType =
                  fifoOut.getElemType().cast<AIEObjectFifoType>();
              MemRefType elemType =
                  fifoType.getElementType().cast<MemRefType>();
              if (fifoOut.name() == op.name())
                break;
              else
                extraOffset += (int)elemType.getShape()[0];
            }
          }
        } else {
          if (target != op) {
            AIEObjectFifoType targetFifo =
                target.getElemType().cast<AIEObjectFifoType>();
            MemRefType targetElemType =
                targetFifo.getElementType().cast<MemRefType>();
            lenOut = getMemrefTypeSize(targetElemType);
          }
        }

        // check if current op is of smaller size in link
        if (target != op)
          numBlocks = target.size();
      }
    }

    // search for MemTileDMAOp
    Operation *producerDMA = nullptr;
    for (auto dmaOp : device.getOps<MemTileDMAOp>()) {
      if (dmaOp.getTile() == target.getProducerTile()) {
        producerDMA = dmaOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = target.getProducerTileOp();
    if (producerDMA == nullptr) {
      if (device->getNumRegions() != 1)
        assert(false && "expected num regions for device op");
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(device.getBody());
      MemTileDMAOp newDMAOp =
          builder.create<MemTileDMAOp>(builder.getUnknownLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, bdBlock, endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ = nullptr;
    Block *curr = bdBlock;
    int blockIndex = 0;
    for (int i = 0; i < numBlocks; i++) {
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      builder.setInsertionPointToStart(curr);
      if (isDistribute || isJoin)
        offset = extraOffset * bytes;
      createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                              buffersPerFifo[target][blockIndex], offset,
                              lenOut, channelDir, blockIndex, succ);
      curr = succ;
      blockIndex++;
    }
  }

  // Function that computes the Least Common Multiplier of the values
  // of a vector.
  int computeLCM(std::set<int> values) {
    int lcm = 1;
    for (int i : values)
      lcm = (i * lcm) / std::gcd(i, lcm);
    return lcm;
  }

  // Recursively calls itself if it finds a nested for loop.
  // Returns the next index to use to uniquely identify operations
  // on the body of the innerLoop.
  int identifyDependencies(mlir::scf::ForOp outerLoop,
                           mlir::scf::ForOp innerLoop,
                           std::vector<Operation *> &operations,
                           DenseMap<Operation *, int> &opIndex,
                           std::vector<std::vector<int>> &dependencies,
                           int startIndex) {
    Block *body = innerLoop.getBody();
    auto withoutTerminator = --body->end();
    int index = startIndex;
    for (auto op = body->begin(); op != withoutTerminator; op++) {
      operations.push_back(&(*op));
      opIndex[&(*op)] = index;

      // identify dependencies
      auto numOperands = (&(*op))->getNumOperands();
      std::vector<int> dependecyIndices;
      for (int i = 0; (unsigned)i < numOperands; i++) {
        auto operand = (&(*op))->getOperand(i);
        int dependencyIndex = -1;

        if (operand == outerLoop.getInductionVar()) {
          dependencyIndex = LOOP_VAR_DEPENDENCY;
        } else {
          auto definingOp = operand.getDefiningOp();
          if (opIndex.find(definingOp) != opIndex.end())
            dependencyIndex = opIndex[definingOp];
        }
        dependecyIndices.push_back(dependencyIndex);
      }
      dependencies.push_back(dependecyIndices);

      index++;

      // if op was a nested for-loop, also keep track of dependencies inside it
      if (auto nestedLoop = dyn_cast<mlir::scf::ForOp>(op))
        index = identifyDependencies(outerLoop, nestedLoop, operations, opIndex,
                                     dependencies, index);
    }
    return index;
  }

  // Replace operands of cloned operation with results from other
  // duplicated operations based on the index of the original
  // operation and its dependencies.
  void replaceOperands(OpBuilder &builder, Operation *clone,
                       int originalOpIndex, mlir::Value base, int64_t step,
                       bool inLoop, int currentDuplication,
                       std::vector<std::vector<int>> &dependencies,
                       std::vector<Operation *> &duplicatedOperations) {
    auto numOperands = clone->getNumOperands();
    for (int operandIndex = 0; (unsigned)operandIndex < numOperands;
         operandIndex++) {
      int originalDependencyIndex = dependencies[originalOpIndex][operandIndex];

      if (originalDependencyIndex >= 0) {
        // replace the operand with the result of operation with
        // same index in current duplication
        auto duplicatedOp = duplicatedOperations[originalDependencyIndex];
        mlir::Value result = duplicatedOp->getResult(0);
        clone->setOperand(operandIndex, result);

      } else if (originalDependencyIndex == LOOP_VAR_DEPENDENCY) {
        int64_t increment_value = 0;
        if (inLoop)
          // +1 because we do not duplicate original loop body
          increment_value = (currentDuplication + 1) * step;
        else
          increment_value = currentDuplication * step;

        arith::ConstantOp increment = builder.create<arith::ConstantOp>(
            builder.getUnknownLoc(), builder.getIndexAttr(increment_value));
        arith::AddIOp sum = builder.create<arith::AddIOp>(
            builder.getUnknownLoc(), builder.getIndexType(), base,
            increment->getResult(0));
        clone->setOperand(operandIndex, sum->getResult(0));
      }
    }
    duplicatedOperations.push_back(clone);
  }

  // Function that duplicates given operations for the given number
  // of times. !!! Assumes builder insertion point is set. !!!
  // If there is a dependency on a loop induction variable, the given
  // base mlir::Value is used to resolve it.
  void duplicateBlock(OpBuilder &builder, int numDuplications,
                      std::vector<Operation *> &operations,
                      std::vector<std::vector<int>> &dependencies,
                      mlir::Value base, int64_t step, bool inLoop) {
    std::vector<Operation *> duplicatedOperations; // operations in current
                                                   // duplication iteration
    for (int i = 0; i < numDuplications; i++) {
      duplicatedOperations.clear();
      for (unsigned opIndex = 0; opIndex < operations.size(); opIndex++) {
        // for each operand, check whether there was a dependecy
        auto op = operations[opIndex];
        auto clone = op->clone();
        replaceOperands(builder, clone, opIndex, base, step, inLoop, i,
                        dependencies, duplicatedOperations);
        builder.insert(clone);

        if (auto nestedLoop = dyn_cast<mlir::scf::ForOp>(clone)) {
          Block *body = nestedLoop.getBody();
          auto withoutTerminator = --body->end();
          for (auto loopOp = body->begin(); loopOp != withoutTerminator;
               loopOp++) {
            opIndex++;
            replaceOperands(builder, &(*loopOp), opIndex, base, step, inLoop, i,
                            dependencies, duplicatedOperations);
          }
        }
      }
    }
  }

  // Function that unrolls for-loops that contain objectFifo operations.
  void unrollForLoops(DeviceOp &device, OpBuilder &builder,
                      std::set<TileOp> objectFifoTiles) {
    for (auto coreOp : device.getOps<CoreOp>()) {
      if (objectFifoTiles.count(coreOp.getTileOp()) > 0) {
        coreOp.walk([&](mlir::scf::ForOp forLoop) {
          // look for operations on objectFifos
          // when multiple fifos in same loop, must use the smallest
          // common multiplier as the unroll factor
          bool found = false;
          std::set<int> objFifoSizes;
          int unrollFactor = 0;
          Block *body = forLoop.getBody();

          for (auto acqOp : body->getOps<ObjectFifoAcquireOp>()) {
            if (acqOp.getOperation()->getParentOp() == forLoop) {
              found = true;
              ObjectFifoCreateOp op = acqOp.getObjectFifo();
              objFifoSizes.insert(op.size());
            }
          }

          unrollFactor =
              computeLCM(objFifoSizes); // also counts original loop body

          if (found) {
            std::vector<Operation *>
                operations; // operations in original loop body, without
                            // terminator operation
            DenseMap<Operation *, int>
                opIndex; // maps operations of original loop body to their
                         // position in it
            std::vector<std::vector<int>>
                dependencies; // index in first vecotr corresponds to position
                              // in original loop body dependency vector has
                              // size equal to number of operands of that
                              // operation:
                              //    * if LOOP_VAR_DEPENDENCY : operand is
                              //    dependent on loop induction variable
                              //    * if -1 : operand is not dependent on any
                              //    operation in loop body
                              //    * if >=0: operand is dependent on operation
                              //    with that index in original loop body

            // find new loop size and step
            auto old_upper_bound = forLoop.getUpperBound()
                                       .getDefiningOp<arith::ConstantOp>()
                                       .getValue();
            int64_t old_upper_value =
                old_upper_bound.dyn_cast<IntegerAttr>().getInt();
            auto old_lower_bound = forLoop.getLowerBound()
                                       .getDefiningOp<arith::ConstantOp>()
                                       .getValue();
            int64_t old_lower_value =
                old_lower_bound.dyn_cast<IntegerAttr>().getInt();
            auto old_step =
                forLoop.getStep().getDefiningOp<arith::ConstantOp>().getValue();
            int64_t old_step_value = old_step.dyn_cast<IntegerAttr>().getInt();
            int64_t num_iter =
                (old_upper_value - old_lower_value) / old_step_value;

            int64_t num_unrolls =
                0; // number of times to unroll loop, not counting original body

            identifyDependencies(forLoop, forLoop, operations, opIndex,
                                 dependencies, 0);

            if (num_iter <= unrollFactor) {
              // duplicate loop body and remove loop
              num_unrolls = num_iter;
              builder.setInsertionPointAfter(forLoop);
              duplicateBlock(builder, num_unrolls, operations, dependencies,
                             forLoop.getLowerBound(), old_step_value, false);
              forLoop.getOperation()->erase();

            } else {
              num_unrolls = unrollFactor - 1; // -1 without original loop body

              // create new upper bound and step
              int64_t new_step_value = (int64_t)unrollFactor * old_step_value;
              int64_t remainder =
                  ((old_upper_value - old_lower_value) % new_step_value) /
                  old_step_value;
              builder.setInsertionPoint(forLoop);
              if (remainder > 0) {
                int64_t new_upper_bound =
                    ((old_upper_value - old_lower_value) / new_step_value) *
                    new_step_value;
                arith::ConstantOp uBound = builder.create<arith::ConstantOp>(
                    builder.getUnknownLoc(),
                    builder.getIndexAttr(new_upper_bound));
                forLoop.setUpperBound(uBound);
              }
              arith::ConstantOp new_step = builder.create<arith::ConstantOp>(
                  builder.getUnknownLoc(),
                  builder.getIndexAttr(new_step_value));
              forLoop.setStep(new_step);

              // duplicate loop body, insert before terminator operation
              builder.setInsertionPoint(&(body->back()));
              duplicateBlock(builder, num_unrolls, operations, dependencies,
                             forLoop.getInductionVar(), old_step_value, true);
              // duplicate remainder operations after loop body
              builder.setInsertionPointAfter(forLoop);
              duplicateBlock(builder, remainder, operations, dependencies,
                             forLoop.getUpperBound(), old_step_value, false);
            }
          }
        });
      }
    }
  }

  /// Function used to create a UseLockOp based on input parameters.
  /// acc is an accumulator map that tracks the indices of the next locks to
  /// acquire (or release). Uses op to find index of acc for next lockID.
  /// Updates acc.
  void createUseLocks(OpBuilder &builder, ObjectFifoCreateOp op,
                      ObjectFifoPort port,
                      DenseMap<std::pair<ObjectFifoCreateOp, int>, int> &acc,
                      int numLocks, LockAction lockAction) {
    ObjectFifoCreateOp target = op;
    auto portNum = (port == ObjectFifoPort::Produce) ? 0 : 1;
    auto linkOp = getOptionalLinkOp(op);
    if (linkOp)
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
        target = objFifoLinks[*linkOp];

    auto dev = op->getParentOfType<xilinx::AIE::DeviceOp>();
    auto &targetArch = dev.getTargetModel();
    if (targetArch.getTargetArch() == xilinx::AIE::AIEArch::AIE1) {
      int lockMode = 0;
      if ((port == ObjectFifoPort::Produce &&
           lockAction == LockAction::Release) ||
          (port == ObjectFifoPort::Consume &&
           lockAction == LockAction::Acquire))
        lockMode = 1;
      for (int i = 0; i < numLocks; i++) {
        int lockID = acc[{op, portNum}];
        builder.create<UseLockOp>(builder.getUnknownLoc(),
                                  locksPerFifo[target][lockID], lockMode,
                                  lockAction);
        acc[{op, portNum}] =
            (lockID + 1) % op.size(); // update to next objFifo elem
      }
    } else {
      if (numLocks == 0)
        return;
      // search for the correct lock based on the port of the acq/rel
      // operation e.g. acq as consumer is the read lock (second)
      LockOp lock;
      if (lockAction == LockAction::AcquireGreaterEqual) {
        if (port == ObjectFifoPort::Produce)
          lock = locksPerFifo[target][0];
        else
          lock = locksPerFifo[target][1];
      } else {
        if (port == ObjectFifoPort::Produce)
          lock = locksPerFifo[target][1];
        else
          lock = locksPerFifo[target][0];
      }
      builder.create<UseLockOp>(builder.getUnknownLoc(), lock, numLocks,
                                lockAction);
      acc[{op, portNum}] = (acc[{op, portNum}] + numLocks) %
                           op.size(); // update to next objFifo elem
    }
  }

  /// Function used to check whether op is already contained in map.
  /// If it is then return the associated int, if not create new entry and
  /// return 0.
  int updateAndReturnIndex(
      DenseMap<std::pair<ObjectFifoCreateOp, int>, int> &map,
      std::pair<ObjectFifoCreateOp, int> pair) {
    if (map.find(pair) == map.end()) {
      map[pair] = 0;
      return 0;
    }
    return map[pair];
  }

  /// Function used to add an external buffer to the externalBuffersPerFifo map.
  void addExternalBuffer(ObjectFifoCreateOp fifo, ExternalBufferOp buff) {
    if (externalBuffersPerFifo.find(fifo) == externalBuffersPerFifo.end()) {
      std::vector<ExternalBufferOp> buffs;
      externalBuffersPerFifo[fifo] = buffs;
    }
    externalBuffersPerFifo[fifo].push_back(buff);
  }

  /// Function used to detect all external buffers associated with parent
  /// objectFifo and tile then map them to child objectFifo.
  void detectExternalBuffers(DeviceOp &device, ObjectFifoCreateOp parent,
                             ObjectFifoCreateOp child, Value tile) {
    for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>()) {
      auto objFifo = regOp.getObjectFifo();
      if (regOp.getTile() == tile && objFifo == parent) {
        for (auto extBuff : regOp.getExternalBuffers())
          addExternalBuffer(child, extBuff.getDefiningOp<ExternalBufferOp>());
      }
    }
  }

  /// Function used to replace uses of split objectFifos.
  void replaceSplitFifo(ObjectFifoCreateOp originalOp, ObjectFifoCreateOp newOp,
                        TileOp tile) {
    auto original =
        originalOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    auto newSymbol =
        newOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    for (auto user : tile->getUsers())
      if (auto coreOp = dyn_cast<CoreOp>(user)) {
        auto res =
            mlir::SymbolTable::replaceAllSymbolUses(original, newSymbol, user);
        if (res.failed())
          llvm_unreachable("unreachable");
      }
  }

  /// Function used to find the size of an objectFifo after split based on
  /// the maximum number of elements (of the original objectFifo) acquired
  /// by a process running on given tile. If no CoreOp exists for this tile
  /// return 0.
  int findObjectFifoSize(DeviceOp &device, Value tile,
                         ObjectFifoCreateOp objFifo) {
    if (objFifo.size() == 0)
      return 0;

    // if memTile, size is equal to objFifo size
    if (tile.getDefiningOp<TileOp>().isMemTile())
      return objFifo.size();

    // if shimTile, size is equal to number of external buffers
    if (tile.getDefiningOp<TileOp>().isShimTile()) {
      for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>()) {
        auto objFifo = regOp.getObjectFifo();
        if (regOp.getTile() == tile && objFifo == objFifo)
          return regOp.getExternalBuffers().size();
      }
    }

    int maxAcquire = 0;
    for (auto coreOp : device.getOps<CoreOp>()) {
      if (coreOp.getTile() == tile) {
        coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
          auto createOp = acqOp.getObjectFifo();
          if (createOp == objFifo)
            if (acqOp.acqNumber() > maxAcquire)
              maxAcquire = acqOp.acqNumber();
        });
      }
    }

    if (maxAcquire > 0) {
      if ((maxAcquire == 1) && (objFifo.size() == 1))
        return 1;
      return maxAcquire + 1;
      // +1 because objectFifo size is always 1 bigger than maxAcquire to allow
      // for prefetching: simplest case scenario is at least a ping-pong buffer
    }

    return objFifo.size();
  }

  /// Function used to generate, from an objectFifo with a shimTile endpoint, a
  /// shimDMAAllocationOp containing the channelDir, channelIndex and
  /// shimTile col assigned by the objectFifo lowering.
  void createObjectFifoAllocationInfo(OpBuilder &builder, MLIRContext *ctx,
                                      FlatSymbolRefAttr obj_fifo, int colIndex,
                                      DMAChannelDir channelDir,
                                      int channelIndex) {
    builder.create<ShimDMAAllocationOp>(builder.getUnknownLoc(), obj_fifo,
                                        DMAChannelDirAttr::get(ctx, channelDir),
                                        builder.getI64IntegerAttr(channelIndex),
                                        builder.getI64IntegerAttr(colIndex));
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    LockAnalysis lockAnalysis(device);
    DMAChannelAnalysis dmaAnalysis(device);
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());
    auto ctx = device->getContext();

    //===------------------------------------------------------------------===//
    // Create objectFifos
    //===------------------------------------------------------------------===//
    std::set<TileOp>
        objectFifoTiles; // track cores to check for loops during unrolling

    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      objectFifoTiles.insert(createOp.getProducerTileOp());
      bool shared = false;
      std::vector<ObjectFifoCreateOp> splitConsumerFifos;
      int share_direction = 0;
      int consumerIndex = 0;
      int consumerDepth = createOp.size();

      for (auto consumerTile : createOp.getConsumerTiles()) {
        TileOp consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());
        objectFifoTiles.insert(consumerTileOp);

        // if there is no broadcast, we can optimize in shared memory case
        if (createOp.getConsumerTiles().size() == 1) {
          bool memoryAdjacent = isSharedMemory(
              createOp.getProducerTileOp(), consumerTileOp, &share_direction);
          if (memoryAdjacent) {
            shared = true;
            break;
          }
        }

        // objectFifos between non-adjacent tiles must be split into two,
        // their elements will be created in next iterations
        if (isa<ArrayAttr>(createOp.getElemNumber()))
          // +1 to account for 1st depth (producer)
          consumerDepth = createOp.size(consumerIndex + 1);
        else
          consumerDepth = findObjectFifoSize(device, consumerTileOp, createOp);

        builder.setInsertionPointAfter(createOp);
        AIEObjectFifoType datatype =
            createOp.getElemType().cast<AIEObjectFifoType>();
        auto consumerObjFifoSize =
            builder.getIntegerAttr(builder.getI32Type(), consumerDepth);
        // rename and replace split objectFifo
        std::string consumerFifoName;
        if (createOp.getConsumerTiles().size() > 1) {
          consumerFifoName = createOp.name().str() + "_" +
                             std::to_string(consumerIndex) + "_cons";
          consumerIndex++;
        } else {
          consumerFifoName = createOp.name().str() + "_cons";
        }
        ObjectFifoCreateOp consumerFifo =
            createObjectFifo(builder, datatype, consumerFifoName, consumerTile,
                             consumerTile, consumerObjFifoSize);
        replaceSplitFifo(createOp, consumerFifo, consumerTileOp);

        // identify external buffers that were registered to
        // the consumer objectFifo
        if (consumerTile.getDefiningOp<TileOp>().isShimTile())
          detectExternalBuffers(device, createOp, consumerFifo, consumerTile);

        // record that this objectFifo was split
        splitConsumerFifos.push_back(consumerFifo);

        // update the linkOp if the split objFifo was originally its start point
        auto linkOp = getOptionalLinkOp(createOp);
        if (linkOp)
          for (auto fifoIn : linkOp->getInputObjectFifos())
            if (fifoIn.name() == createOp.name())
              if (consumerTile == *(linkOp->getOptionalSharedTile())) {
                auto res = mlir::SymbolTable::replaceAllSymbolUses(
                    createOp.name(), consumerFifo.name(),
                    linkOp->getOperation());
                if (res.failed())
                  llvm_unreachable("unreachable");
              }
      }

      // identify external buffers that were registered to
      // the producer objectFifo
      if (createOp.getProducerTileOp().isShimTile())
        detectExternalBuffers(device, createOp, createOp,
                              createOp.getProducerTile());

      // if split, the necessary size for producer fifo might change
      if (shared) {
        createObjectFifoElements(builder, lockAnalysis, createOp,
                                 share_direction);
      } else {
        if (isa<ArrayAttr>(createOp.getElemNumber())) {
          createOp->setAttr("elemNumber",
                            builder.getI32IntegerAttr(createOp.size()));
        } else {
          int prodMaxAcquire = findObjectFifoSize(
              device, createOp.getProducerTileOp(), createOp);
          createOp->setAttr("elemNumber",
                            builder.getI32IntegerAttr(prodMaxAcquire));
        }
        createObjectFifoElements(builder, lockAnalysis, createOp,
                                 share_direction);
        // register split consumer objectFifos
        splitFifos.push_back({createOp, splitConsumerFifos});
      }
    }

    //===------------------------------------------------------------------===//
    // Create flows and tile DMAs
    //===------------------------------------------------------------------===//
    for (auto &[producer, consumers] : splitFifos) {
      // create producer tile DMA
      xilinx::AIE::DMAChannel producerChan =
          dmaAnalysis.getMasterDMAChannel(producer.getProducerTile());
      createDMA(device, builder, producer, producerChan.first,
                producerChan.second, 0);
      // generate objectFifo allocation info
      builder.setInsertionPoint(&device.getBody()->back());
      if (producer.getProducerTileOp().isShimTile())
        createObjectFifoAllocationInfo(
            builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
            producer.getProducerTileOp().colIndex(), producerChan.first,
            producerChan.second);

      for (auto consumer : consumers) {
        // create consumer tile DMA
        xilinx::AIE::DMAChannel consumerChan =
            dmaAnalysis.getSlaveDMAChannel(consumer.getProducerTile());
        createDMA(device, builder, consumer, consumerChan.first,
                  consumerChan.second, 1);
        // generate objectFifo allocation info
        builder.setInsertionPoint(&device.getBody()->back());
        if (consumer.getProducerTileOp().isShimTile())
          createObjectFifoAllocationInfo(
              builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
              consumer.getProducerTileOp().colIndex(), consumerChan.first,
              consumerChan.second);

        // create flow
        builder.setInsertionPointAfter(producer);
        builder.create<FlowOp>(builder.getUnknownLoc(),
                               producer.getProducerTile(), WireBundle::DMA,
                               producerChan.second, consumer.getProducerTile(),
                               WireBundle::DMA, consumerChan.second);
      }
    }

    //===------------------------------------------------------------------===//
    // Unroll for loops
    //===------------------------------------------------------------------===//
    unrollForLoops(device, builder, objectFifoTiles);

    //===------------------------------------------------------------------===//
    // Replace ops
    //===------------------------------------------------------------------===//
    for (auto coreOp : device.getOps<CoreOp>()) {
      DenseMap<ObjectFifoAcquireOp, std::vector<BufferOp *>>
          subviews; // maps each "subview" to its buffer references (subviews
                    // are created by AcquireOps)
      DenseMap<std::pair<ObjectFifoCreateOp, int>, std::vector<int>>
          acquiresPerFifo; // maps each objFifo to indices of buffers acquired
                           // in latest subview of that objFifo (useful to
                           // cascade acquired elements to next AcquireOp)
      DenseMap<std::pair<ObjectFifoCreateOp, int>,
               std::vector<ObjectFifoReleaseOp>>
          releaseOps; // useful to check which ReleaseOp has taken place before
                      // an AcquireOp per objFifo
      DenseMap<std::pair<ObjectFifoCreateOp, int>, int>
          acqPerFifo; // maps each objFifo to its next index to acquire within
                      // this CoreOp
      DenseMap<std::pair<ObjectFifoCreateOp, int>, int>
          relPerFifo; // maps each objFifo to its next index to release within
                      // this CoreOp

      //===----------------------------------------------------------------===//
      // Replace objectFifo.release ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoReleaseOp releaseOp) {
        builder.setInsertionPointAfter(releaseOp);
        ObjectFifoCreateOp op = releaseOp.getObjectFifo();
        auto port = releaseOp.getPort();
        auto portNum = (port == ObjectFifoPort::Produce) ? 0 : 1;
        auto core = releaseOp->getParentOfType<CoreOp>();

        auto linkOp = getOptionalLinkOp(op);
        if (linkOp) {
          if (core.getTile() == *(linkOp->getOptionalSharedTile())) {
            releaseOp->emitOpError("currently cannot access objectFifo used in "
                                   "ObjectFifoLinkOp");
            return;
          }
        }

        // update index of next element to release for this objectFifo
        updateAndReturnIndex(relPerFifo, {op, portNum});

        // release locks
        int numLocks = releaseOp.relNumber();
        createUseLocks(builder, op, port, relPerFifo, numLocks,
                       LockAction::Release);

        // register release op
        if (releaseOps.find({op, portNum}) != releaseOps.end()) {
          releaseOps[{op, portNum}].push_back(releaseOp);
        } else {
          std::vector<ObjectFifoReleaseOp> release = {releaseOp};
          releaseOps[{op, portNum}] = release;
        }
      });

      //===----------------------------------------------------------------===//
      // Replace objectFifo.acquire ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoAcquireOp acquireOp) {
        ObjectFifoCreateOp op = acquireOp.getObjectFifo();
        builder.setInsertionPointAfter(acquireOp);
        auto port = acquireOp.getPort();
        auto portNum = (port == ObjectFifoPort::Produce) ? 0 : 1;
        auto core = acquireOp->getParentOfType<CoreOp>();

        auto linkOp = getOptionalLinkOp(op);
        if (linkOp) {
          if (core.getTile() == *(linkOp->getOptionalSharedTile())) {
            acquireOp->emitOpError("currently cannot access objectFifo used in "
                                   "ObjectFifoLinkOp");
            return;
          }
        }

        // index of next element to acquire for this objectFifo
        int start = updateAndReturnIndex(
            acqPerFifo, {op, portNum}); // useful for keeping track of which
                                        // indices are acquired

        // check how many elements have been released in between this AcquireOp
        // and the previous one
        int numRel = 0;
        for (auto relOp : releaseOps[{op, portNum}]) {
          ObjectFifoCreateOp otherOp = relOp.getObjectFifo();
          // TODO: operations may not be in the same block: currently only
          // support one block level of difference

          if (op == otherOp) {
            // if they are already in the same block, check if releaseOp
            // happened before
            if (acquireOp.getOperation()->getBlock() ==
                relOp.getOperation()->getBlock()) {
              if (!acquireOp->isBeforeInBlock(relOp)) {
                releaseOps[{op, portNum}].erase(
                    releaseOps[{op, portNum}].begin());
                // to ensure that we do not account
                // the ReleaseOps again later,
                // after the subview is created
                numRel += relOp.relNumber();
              }
            } else {
              Operation *acqBlockDefOp =
                  acquireOp.getOperation()->getBlock()->getParentOp();

              // else, check if releaseOp happened before the block region
              // with the acquireOp
              if (relOp.getOperation()->getBlock() ==
                  acqBlockDefOp->getBlock()) {
                if (!acqBlockDefOp->isBeforeInBlock(relOp)) {
                  releaseOps[{op, portNum}].erase(
                      releaseOps[{op, portNum}]
                          .begin()); // to ensure that we do not account
                                     // the ReleaseOps again later, after
                                     // the subview is created
                  numRel += relOp.relNumber();
                }

                // else, check if the block region with releaseOp happened
                // before...
              } else {
                Operation *relBlockDefOp =
                    relOp.getOperation()->getBlock()->getParentOp();

                // ...the acquireOp
                if (acquireOp.getOperation()->getBlock() ==
                    relBlockDefOp->getBlock()) {
                  if (!acquireOp->isBeforeInBlock(relBlockDefOp)) {
                    releaseOps[{op, portNum}].erase(
                        releaseOps[{op, portNum}]
                            .begin()); // to ensure that we do not account
                                       // the ReleaseOps again later,
                                       // after the subview is created
                    numRel += relOp.relNumber();
                  }

                  // ...the block region with the acquireOp
                } else if (acqBlockDefOp->getBlock() ==
                           relBlockDefOp->getBlock()) {
                  if (!acqBlockDefOp->isBeforeInBlock(relBlockDefOp)) {
                    releaseOps[{op, portNum}].erase(
                        releaseOps[{op, portNum}]
                            .begin()); // to ensure that we do not account
                                       // the ReleaseOps again later,
                                       // after the subview is created
                    numRel += relOp.relNumber();
                  }
                }
              }
            }
          }
        }

        // track indices of elements to acquire
        std::vector<int> acquiredIndices;
        if (acquiresPerFifo[{op, portNum}].size() != 0) {
          // take into account what has already been acquired by previous
          // AcquireOp in program order
          acquiredIndices = acquiresPerFifo[{op, portNum}];
          // take into account what has been released in-between
          assert((size_t)numRel <= acquiredIndices.size() &&
                 "Cannot release more elements than are already acquired.");
          for (int i = 0; i < numRel; i++)
            acquiredIndices.erase(acquiredIndices.begin());
        }

        // acquire locks
        int numLocks = acquireOp.acqNumber();
        int alreadyAcq = acquiredIndices.size();
        int numCreate;
        if (numLocks > alreadyAcq)
          numCreate = numLocks - alreadyAcq;
        else
          numCreate = 0;

        auto dev = op->getParentOfType<xilinx::AIE::DeviceOp>();
        auto &targetArch = dev.getTargetModel();
        if (targetArch.getTargetArch() == xilinx::AIE::AIEArch::AIE1)
          createUseLocks(builder, op, port, acqPerFifo, numCreate,
                         LockAction::Acquire);
        else
          createUseLocks(builder, op, port, acqPerFifo, numCreate,
                         LockAction::AcquireGreaterEqual);

        // if objFifo was linked with others, find which objFifos
        // elements to use
        ObjectFifoCreateOp target = op;
        if (linkOp)
          if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
            target = objFifoLinks[*linkOp];

        // create subview: buffers that were already acquired + new acquires
        for (int i = 0; i < numCreate; i++) {
          acquiredIndices.push_back(start);
          start = (start + 1) % op.size();
        }
        std::vector<BufferOp *> subviewRefs;
        for (auto index : acquiredIndices)
          subviewRefs.push_back(&buffersPerFifo[target][index]);

        subviews[acquireOp] = subviewRefs;
        acquiresPerFifo[{op, portNum}] = acquiredIndices;
      });

      //===----------------------------------------------------------------===//
      // Replace subview.access ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoSubviewAccessOp accessOp) {
        ObjectFifoAcquireOp acqOp =
            accessOp.getSubview().getDefiningOp<ObjectFifoAcquireOp>();
        ObjectFifoCreateOp op = acqOp.getObjectFifo();
        auto linkOp = getOptionalLinkOp(op);
        if (linkOp) {
          accessOp->emitOpError("currently cannot access objectFifo used in "
                                "ObjectFifoLinkOp");
          return;
        }
        accessOp.getOutput().replaceAllUsesWith(
            subviews[acqOp][accessOp.getIndex()]->getBuffer());
      });
    }

    // make global symbols to replace the to be erased ObjectFifoCreateOps
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      builder.setInsertionPointToStart(&(device.getBodyRegion().front()));
      auto sym_name = createOp.getName();
      createOp->setAttr(mlir::SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr("__erase_" + sym_name));
      auto memrefType = cast<MemRefType>(
          cast<AIEObjectFifoType>(createOp.getElemType()).getElementType());
      builder.create<memref::GlobalOp>(builder.getUnknownLoc(), sym_name,
                                       builder.getStringAttr("public"),
                                       memrefType, nullptr, false, nullptr);
    }

    //===------------------------------------------------------------------===//
    // Remove old ops
    //===------------------------------------------------------------------===//
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<AIEOpRemoval<ObjectFifoCreateOp>>(device.getContext());
    patterns.add<AIEOpRemoval<ObjectFifoLinkOp>>(device.getContext());
    patterns.add<AIEOpRemoval<ObjectFifoRegisterExternalBuffersOp>>(
        device.getContext());
    patterns.add<AIEOpRemoval<ObjectFifoAcquireOp>>(device.getContext());
    patterns.add<AIEOpRemoval<ObjectFifoSubviewAccessOp>>(device.getContext());
    patterns.add<AIEOpRemoval<ObjectFifoReleaseOp>>(device.getContext());
    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoStatefulTransformPass() {
  return std::make_unique<AIEObjectFifoStatefulTransformPass>();
}
