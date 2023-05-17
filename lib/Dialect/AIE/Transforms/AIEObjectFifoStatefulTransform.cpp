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
          if (op.isSend()) {
            getMasterDMAChannel(memOp.getTile());
          } else {
            getSlaveDMAChannel(memOp.getTile());
          }
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
  DenseMap<ObjectFifoCreateOp, std::vector<ObjectFifoCreateOp>>
      splitFifos;     // maps each objFifo between non-adjacent tiles to its
                      // corresponding consumer objectFifos

  /// Function that returns true if two tiles in the AIE array share a memory
  /// module. share_direction is equal to:
  ///   * -1 if the shared memory module is that of the first input tile,
  ///   * 1 if it is that of the second input tile,
  ///   * 0 is no memory module is shared.
  bool isSharedMemory(TileOp a, TileOp b, int *share_direction) {
    if ((a.isShimTile() && !b.isShimTile()) ||
        (!a.isShimTile() && b.isShimTile())) {
      *share_direction = 0;
      return false;
    }

    const auto &target_model = getTargetModel(a.getOperation());
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

  ObjectFifoCreateOp createObjectFifo(OpBuilder &builder,
                                      AIEObjectFifoType datatype,
                                      Value prodTile, Value consTile,
                                      int depth) {
    ObjectFifoCreateOp fifo = builder.create<ObjectFifoCreateOp>(
        builder.getUnknownLoc(), datatype, prodTile, consTile, depth);
    return fifo;
  }

  /// Function used to create objectFifo locks based on target architecture.
  /// Called by createObjectFifoElements().
  std::vector<LockOp> createObjectFifoLocks(OpBuilder &builder,
                                            LockAnalysis &lockAnalysis,
                                            ObjectFifoCreateOp op,
                                            TileOp creation_tile) {
    std::vector<LockOp> locks;
    auto dev = op->getParentOfType<xilinx::AIE::DeviceOp>();
    auto &target = dev.getTargetModel();
    if (target.getTargetArch() == xilinx::AIE::AIEArch::AIE1) {
      int of_elem_index = 0; // used to give objectFifo elements a symbolic name
      for (int i = 0; i < op.size(); i++) {
        // create corresponding aie1 locks
        int lockID = lockAnalysis.getLockID(creation_tile);
        assert(lockID >= 0 && "No more locks to allocate!");
        LockOp lock = builder.create<LockOp>(builder.getUnknownLoc(),
                                             creation_tile, lockID, 0);
        lock.getOperation()->setAttr(
            mlir::SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name()->getValue() + "_lock_" +
                                  std::to_string(of_elem_index)));
        locks.push_back(lock);
        of_elem_index++;
      }
    } else {
      // create corresponding aie2 locks
      int prodLockID = lockAnalysis.getLockID(creation_tile);
      assert(prodLockID >= 0 && "No more locks to allocate!");
      LockOp prodLock = builder.create<LockOp>(
          builder.getUnknownLoc(), creation_tile, prodLockID, op.size());
      prodLock.getOperation()->setAttr(
          mlir::SymbolTable::getSymbolAttrName(),
          builder.getStringAttr(op.name()->getValue() + "_prod_lock"));
      locks.push_back(prodLock);

      int consLockID = lockAnalysis.getLockID(creation_tile);
      assert(consLockID >= 0 && "No more locks to allocate!");
      LockOp consLock = builder.create<LockOp>(builder.getUnknownLoc(),
                                               creation_tile, consLockID, 0);
      consLock.getOperation()->setAttr(
          mlir::SymbolTable::getSymbolAttrName(),
          builder.getStringAttr(op.name()->getValue() + "_cons_lock"));
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
    AIEObjectFifoType fifo = op.getType().cast<AIEObjectFifoType>();
    MemRefType elemType = fifo.getElementType().cast<MemRefType>();
    int of_elem_index = 0; // used to give objectFifo elements a symbolic name

    TileOp creation_tile;
    if (share_direction == 0 || share_direction == -1)
      creation_tile = op.getProducerTileOp();
    else {
      TileOp consumerTileOp =
          dyn_cast<TileOp>(op.getConsumerTiles()[0].getDefiningOp());
      creation_tile = consumerTileOp;
    }

    builder.setInsertionPointAfter(op);
    for (int i = 0; i < op.size(); i++) {
      // if shimTile external buffers are collected from input code
      // create as many locks as there are external buffers
      if (!creation_tile.isShimTile()) {
        BufferOp buff = builder.create<BufferOp>(builder.getUnknownLoc(),
                                                 elemType, creation_tile);
        buff.getOperation()->setAttr(
            mlir::SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name()->getValue() + "_buff_" +
                                  std::to_string(of_elem_index)));
        buffers.push_back(buff);
      }
      of_elem_index++;
    }
    locks = createObjectFifoLocks(builder, lockAnalysis, op, creation_tile);

    buffersPerFifo[op] = buffers;
    locksPerFifo[op] = locks;
  }

  /// Function that returns a pointer to the block of a Region
  /// that contains the AIEEndOp.
  Block *findEndOpBlock(Region *r) {
    Block *endBlock = nullptr;
    for (auto &bl : r->getBlocks()) {
      if (!bl.getOps<EndOp>().empty())
        endBlock = &bl;
    }
    return endBlock;
  }

  /// Function used to create a Bd block.
  /// If lockMode is 0 we create a consumerDMA (i.e. on producer tile) else a
  /// producerDMA (i.e. on consumer tile).
  template <typename MyOp>
  void createBdBlock(OpBuilder &builder, ObjectFifoCreateOp op, int lockMode,
                     MyOp buff, DMAChannelDir channelDir, int blockIndex,
                     Block *succ) {
    int offset = 0;
    MemRefType buffer = buff.getType();
    int len = 1;
    for (auto i : buffer.getShape())
      len *= i;

    int acqMode;
    int relMode;
    auto dev = op->getParentOfType<xilinx::AIE::DeviceOp>();
    auto &target = dev.getTargetModel();
    if (target.getTargetArch() == xilinx::AIE::AIEArch::AIE1) {
      acqMode = lockMode == 0 ? 1 : 0;
      relMode = lockMode == 0 ? 0 : 1;
      builder.create<UseLockOp>(builder.getUnknownLoc(),
                                locksPerFifo[op][blockIndex], acqMode,
                                LockAction::Acquire);
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, 0);
      builder.create<UseLockOp>(builder.getUnknownLoc(),
                                locksPerFifo[op][blockIndex], relMode,
                                LockAction::Release);
      builder.create<NextBDOp>(builder.getUnknownLoc(), succ);
    } else {
      acqMode = 1;
      relMode = 1;
      LockOp prodLock;
      LockOp consLock;
      if (channelDir == DMAChannelDir::S2MM) {
        prodLock = locksPerFifo[op][0];
        consLock = locksPerFifo[op][1];
      } else {
        prodLock = locksPerFifo[op][1];
        consLock = locksPerFifo[op][0];
      }
      builder.create<UseLockOp>(builder.getUnknownLoc(), prodLock, acqMode,
                                LockAction::AcquireGreaterEqual);
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, 0);
      builder.create<UseLockOp>(builder.getUnknownLoc(), consLock, relMode,
                                LockAction::Release);
      builder.create<NextBDOp>(builder.getUnknownLoc(), succ);
    }
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
    unsigned numBlocks = op.size();
    if (numBlocks == 0)
      return;
    TileOp objFifoTileOp = op.getProducerTileOp();
    assert([&] {
      const auto &target_model = xilinx::AIE::getTargetModel(objFifoTileOp);
      if (numBlocks > target_model.getNumBDs(objFifoTileOp.getCol(),
                                             objFifoTileOp.getRow())) {
        printf("Max number of BDs in a DMA channel exceeded.\n");
        return false;
      }
      return true;
    }());

    // search for MemOp
    MemOp *producerMem = nullptr;
    for (auto memOp : device.getOps<MemOp>()) {
      if (memOp.getTile() == op.getProducerTile()) {
        producerMem = &memOp;
        break;
      }
    }

    // if none exists, create one
    if (producerMem == nullptr) {
      builder.setInsertionPointToEnd(device.getBody());
      MemOp newMemOp =
          builder.create<MemOp>(builder.getUnknownLoc(), objFifoTileOp);
      producerMem = &newMemOp;
      Region &r = producerMem->getBody();
      r.push_back(new Block);
      // add terminator operation to end block
      Block &endBlock = r.back();
      builder.setInsertionPointToStart(&endBlock);
      builder.create<EndOp>(builder.getUnknownLoc());
    }

    Block *endBlock = findEndOpBlock(&(producerMem->getBody()));
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
    for (unsigned i = 0; i < numBlocks; i++) {
      if (i == numBlocks - 1) {
        succ = bdBlock;
      } else {
        succ = builder.createBlock(endBlock);
      }
      builder.setInsertionPointToStart(curr);
      createBdBlock<BufferOp>(builder, op, lockMode,
                              buffersPerFifo[op][blockIndex], channelDir,
                              blockIndex, succ);
      curr = succ;
      blockIndex++;
    }
  }

  /// Function used to create a ShimDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createShimDMA(DeviceOp &device, OpBuilder &builder,
                     ObjectFifoCreateOp op, DMAChannelDir channelDir,
                     int channelIndex, int lockMode) {
    unsigned numBlocks = externalBuffersPerFifo[op].size();
    if (numBlocks == 0)
      return;
    TileOp objFifoTileOp = op.getProducerTileOp();
    assert([&] {
      const auto &target_model = xilinx::AIE::getTargetModel(objFifoTileOp);
      if (numBlocks > target_model.getNumBDs(objFifoTileOp.getCol(),
                                             objFifoTileOp.getRow())) {
        printf("Max number of BDs in a DMA channel exceeded.\n");
        return false;
      }
      return true;
    }());

    // search for ShimDMAOp
    ShimDMAOp *producerMem = nullptr;
    for (auto memOp : device.getOps<ShimDMAOp>()) {
      if (memOp.getTile() == op.getProducerTile()) {
        producerMem = &memOp;
        break;
      }
    }

    // if none exists, create one
    if (producerMem == nullptr) {
      builder.setInsertionPointToEnd(device.getBody());
      ShimDMAOp newMemOp = builder.create<ShimDMAOp>(
          builder.getUnknownLoc(), builder.getIndexType(), objFifoTileOp);
      producerMem = &newMemOp;
      Region &r = producerMem->getBody();
      r.push_back(new Block);
      // add terminator operation to end block
      Block &endBlock = r.back();
      builder.setInsertionPointToStart(&endBlock);
      builder.create<EndOp>(builder.getUnknownLoc());
    }

    Block *endBlock = findEndOpBlock(&(producerMem->getBody()));
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
    for (unsigned i = 0; i < numBlocks; i++) {
      if (i == numBlocks - 1) {
        succ = bdBlock;
      } else {
        succ = builder.createBlock(endBlock);
      }
      builder.setInsertionPointToStart(curr);
      createBdBlock<ExternalBufferOp>(builder, op, lockMode,
                                      externalBuffersPerFifo[op][blockIndex],
                                      channelDir, blockIndex, succ);
      curr = succ;
      blockIndex++;
    }
  }

  /// Function used to create a MemTileDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createMemTileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode) {
    unsigned numBlocks = op.size();
    if (numBlocks == 0)
      return;
    TileOp objFifoTileOp = op.getProducerTileOp();
    assert([&] {
      const auto &target_model = xilinx::AIE::getTargetModel(objFifoTileOp);
      if (numBlocks > target_model.getNumBDs(objFifoTileOp.getCol(),
                                             objFifoTileOp.getRow())) {
        printf("Max number of BDs in a DMA channel exceeded.\n");
        return false;
      }
      return true;
    }());

    // search for MemTileDMAOp
    MemTileDMAOp *producerDMA = nullptr;
    for (auto dmaOp : device.getOps<MemTileDMAOp>()) {
      if (dmaOp.getTile() == op.getProducerTile()) {
        producerDMA = &dmaOp;
        break;
      }
    }

    // if none exists, create one
    if (producerDMA == nullptr) {
      builder.setInsertionPointToEnd(device.getBody());
      MemTileDMAOp newDMAOp =
          builder.create<MemTileDMAOp>(builder.getUnknownLoc(), objFifoTileOp);
      producerDMA = &newDMAOp;
      Region &r = producerDMA->getBody();
      r.push_back(new Block);
      // add terminator operation to end block
      Block &endBlock = r.back();
      builder.setInsertionPointToStart(&endBlock);
      builder.create<EndOp>(builder.getUnknownLoc());
    }

    Block *endBlock = findEndOpBlock(&(producerDMA->getBody()));
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
    for (unsigned i = 0; i < numBlocks; i++) {
      if (i == numBlocks - 1) {
        succ = bdBlock;
      } else {
        succ = builder.createBlock(endBlock);
      }
      builder.setInsertionPointToStart(curr);
      createBdBlock<BufferOp>(builder, op, lockMode,
                              buffersPerFifo[op][blockIndex], channelDir,
                              blockIndex, succ);
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

  // Function to record operations in for loop body (without
  // terminator operation) and identify dependencies between them.
  void identifyDependencies(mlir::scf::ForOp forLoop,
                            std::vector<Operation *> &operations,
                            DenseMap<Operation *, int> &opIndex,
                            std::vector<std::vector<int>> &dependencies) {
    Block *body = forLoop.getBody();
    auto withoutTerminator = --body->end();
    int index = 0;
    for (auto op = body->begin(); op != withoutTerminator; op++) {
      operations.push_back(&(*op));
      opIndex[&(*op)] = index;

      // identify dependencies
      auto numOperands = (&(*op))->getNumOperands();
      std::vector<int> dependecyIndices;
      for (int i = 0; (unsigned)i < numOperands; i++) {
        auto operand = (&(*op))->getOperand(i);
        int dependencyIndex = -1;

        if (operand == forLoop.getInductionVar()) {
          dependencyIndex = LOOP_VAR_DEPENDENCY;
        } else {
          auto definingOp = operand.getDefiningOp();
          if (definingOp->getBlock()->getParentOp() == forLoop) {
            dependencyIndex = opIndex[definingOp];
          }
        }
        dependecyIndices.push_back(dependencyIndex);
      }
      dependencies.push_back(dependecyIndices);

      index++;
    }
  }

  // Function that duplicates given operations for the given number
  // of times. Assumes builder insertion point is set.
  // If there is a dependency on a loop induction variable, the given
  // base mlir::Value is used to resolve it.
  void duplicateBlock(OpBuilder &builder, int numDuplications,
                      std::vector<Operation *> &operations,
                      std::vector<std::vector<int>> &dependencies,
                      mlir::Value base, int64_t step, bool inLoop) {
    int originalIndex = 0;
    std::vector<Operation *> duplicatedOperations; // operations in current
                                                   // duplication iteration
    for (int i = 0; i < numDuplications; i++) {
      originalIndex = 0;
      duplicatedOperations.clear();
      for (auto op : operations) {
        // for each operand, check whether there was a dependecy
        auto clone = op->clone();
        auto numOperands = clone->getNumOperands();
        for (int operandIndex = 0; (unsigned)operandIndex < numOperands;
             operandIndex++) {
          int originalDependencyIndex =
              dependencies[originalIndex][operandIndex];
          if (originalDependencyIndex >= 0) {
            // replace the operand with the result of operation with
            // same index in current duplication
            clone->setOperand(
                operandIndex,
                duplicatedOperations[originalDependencyIndex]->getResult(
                    0)); // TODO: what if operation has
                         // multiple results?
          } else if (originalDependencyIndex == LOOP_VAR_DEPENDENCY) {
            int64_t increment_value = 0;
            if (inLoop) {
              // +1 because we do not duplicate original loop body
              increment_value = (i + 1) * step;
            } else {
              increment_value = i * step;
            }
            arith::ConstantOp increment = builder.create<arith::ConstantOp>(
                builder.getUnknownLoc(), builder.getIndexAttr(increment_value),
                builder.getIndexType());
            arith::AddIOp sum = builder.create<arith::AddIOp>(
                builder.getUnknownLoc(), builder.getIndexType(), base,
                increment->getResult(0));
            clone->setOperand(operandIndex, sum->getResult(0));
          }
        }

        builder.insert(clone);
        duplicatedOperations.push_back(clone);
        originalIndex++;
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
              checkSplitFifo<ObjectFifoAcquireOp>(
                  acqOp, coreOp.getTile().getDefiningOp<TileOp>());
              found = true;
              ObjectFifoCreateOp op =
                  acqOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();
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

            identifyDependencies(forLoop, operations, opIndex, dependencies);

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
                    builder.getIndexAttr(new_upper_bound),
                    old_upper_bound.getType());
                forLoop.setUpperBound(uBound);
              }
              arith::ConstantOp new_step = builder.create<arith::ConstantOp>(
                  builder.getUnknownLoc(), builder.getIndexAttr(new_step_value),
                  old_upper_bound.getType());
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
                      DenseMap<ObjectFifoCreateOp, int> &acc, int numLocks,
                      LockAction lockAction) {
    auto dev = op->getParentOfType<xilinx::AIE::DeviceOp>();
    auto &target = dev.getTargetModel();
    if (target.getTargetArch() == xilinx::AIE::AIEArch::AIE1) {
      int lockMode = 0;
      if ((port == ObjectFifoPort::Produce &&
           lockAction == LockAction::Release) ||
          (port == ObjectFifoPort::Consume &&
           lockAction == LockAction::Acquire))
        lockMode = 1;
      for (int i = 0; i < numLocks; i++) {
        int lockID = acc[op];
        builder.create<UseLockOp>(builder.getUnknownLoc(),
                                  locksPerFifo[op][lockID], lockMode,
                                  lockAction);
        acc[op] =
            (lockID + 1) % op.getElemNumber(); // update to next objFifo elem
      }
    } else {
      int lockMode = 1;
      for (int i = 0; i < numLocks; i++) {
        // search for the correct lock based on the port of the acq/rel
        // operation e.g. acq as consumer is the read lock (second)
        LockOp lock;
        if (lockAction == LockAction::AcquireGreaterEqual) {
          if (port == ObjectFifoPort::Produce)
            lock = locksPerFifo[op][0];
          else
            lock = locksPerFifo[op][1];
        } else {
          if (port == ObjectFifoPort::Produce)
            lock = locksPerFifo[op][1];
          else
            lock = locksPerFifo[op][0];
        }
        builder.create<UseLockOp>(builder.getUnknownLoc(), lock, lockMode,
                                  lockAction);
        acc[op] =
            (acc[op] + 1) % op.getElemNumber(); // update to next objFifo elem
      }
    }
  }

  /// Function used to check whether op is already contained in map.
  /// If it is then return the associated int, if not create new entry and
  /// return 0.
  int updateAndReturnIndex(DenseMap<ObjectFifoCreateOp, int> &map,
                           ObjectFifoCreateOp op) {
    if (map.find(op) == map.end()) {
      map[op] = 0;
      return 0;
    }
    return map[op];
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
      if (regOp.getTile() == tile && regOp.getFifo() == parent) {
        for (auto extBuff : regOp.getExternalBuffers())
          addExternalBuffer(child, extBuff.getDefiningOp<ExternalBufferOp>());
      }
    }
  }

  /// Function used to check whether objectFifo accessed by op has been split.
  /// If yes, it replaces the parent objectFifo with the correct consumer
  /// child based on the tile it is on.
  template <typename MyOp> void checkSplitFifo(MyOp op, TileOp tile) {
    ObjectFifoCreateOp parentFifo =
        op.getFifo().template getDefiningOp<ObjectFifoCreateOp>();
    if (splitFifos.find(parentFifo) != splitFifos.end()) {
      if (op.getPort() == ObjectFifoPort::Consume) {
        for (auto splitFifo : splitFifos[parentFifo]) {
          if (splitFifo.getProducerTile() == tile.getResult())
            op->replaceUsesOfWith(parentFifo, splitFifo);
        }
      }
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
        if (regOp.getTile() == tile && regOp.getFifo() == objFifo)
          return regOp.getExternalBuffers().size();
      }
    }

    int maxAcquire = 0;
    for (auto coreOp : device.getOps<CoreOp>()) {
      if (coreOp.getTile() == tile) {
        coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
          if (acqOp.getFifo().getDefiningOp<ObjectFifoCreateOp>() == objFifo)
            if (acqOp.acqNumber() > maxAcquire)
              maxAcquire = acqOp.acqNumber();
        });
      }
    }

    if (maxAcquire > 0) {
      if ((maxAcquire == 1) && (objFifo.size() == 1)) {
        return 1;
      }
      return maxAcquire + 1;
      // +1 because objectFifo size is always 1 bigger than maxAcquire to allow
      // for prefetching: simplest case scenario is at least a ping-pong buffer
    }

    return 0;
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    LockAnalysis lockAnalysis(device);
    DMAChannelAnalysis dmaAnalysis(device);
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

    //===----------------------------------------------------------------------===//
    // Create objectFifos
    //===----------------------------------------------------------------------===//
    std::set<TileOp>
        objectFifoTiles; // track cores to check for loops during unrolling

    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      objectFifoTiles.insert(createOp.getProducerTileOp());
      bool shared = false;
      std::vector<ObjectFifoCreateOp> splitConsumerFifos;
      int share_direction = 0;
      int consumerIndex = 0;

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
        int consMaxAcquire =
            findObjectFifoSize(device, consumerTileOp, createOp);
        builder.setInsertionPointAfter(createOp);
        AIEObjectFifoType datatype =
            createOp.getType().cast<AIEObjectFifoType>();

        ObjectFifoCreateOp consumerFifo = createObjectFifo(
            builder, datatype, consumerTile, consumerTile, consMaxAcquire);

        if (createOp.getConsumerTiles().size() > 1) {
          consumerFifo.getOperation()->setAttr(
              SymbolTable::getSymbolAttrName(),
              builder.getStringAttr(createOp.name()->getValue() + "_" +
                                    std::to_string(consumerIndex) + "_cons"));
          consumerIndex++;
        } else {
          consumerFifo.getOperation()->setAttr(
              SymbolTable::getSymbolAttrName(),
              builder.getStringAttr(createOp.name()->getValue() + "_cons"));
        }

        if (consumerTile.getDefiningOp<TileOp>().isShimTile())
          detectExternalBuffers(device, createOp, consumerFifo, consumerTile);

        // record that this objectFifo was split
        splitConsumerFifos.push_back(consumerFifo);
      }

      if (createOp.getProducerTileOp().isShimTile())
        detectExternalBuffers(device, createOp, createOp,
                              createOp.getProducerTile());

      // if split, the necessary size for producer fifo might change
      if (shared) {
        createObjectFifoElements(builder, lockAnalysis, createOp,
                                 share_direction);
      } else {
        int prodMaxAcquire =
            findObjectFifoSize(device, createOp.getProducerTileOp(), createOp);
        createOp->setAttr("elemNumber",
                          builder.getI32IntegerAttr(prodMaxAcquire));
        createObjectFifoElements(builder, lockAnalysis, createOp,
                                 share_direction);
        // register split consumer objectFifos
        splitFifos[createOp] = splitConsumerFifos;
      }
    }

    //===----------------------------------------------------------------------===//
    // Create flows and tile DMAs
    //===----------------------------------------------------------------------===//
    for (auto [producer, consumers] : splitFifos) {
      // create producer tile DMA
      xilinx::AIE::DMAChannel producerChan =
          dmaAnalysis.getMasterDMAChannel(producer.getProducerTile());
      createDMA(device, builder, producer, producerChan.first,
                producerChan.second, 0);

      for (auto consumer : consumers) {
        // create consumer tile DMA
        xilinx::AIE::DMAChannel consumerChan =
            dmaAnalysis.getSlaveDMAChannel(consumer.getProducerTile());
        createDMA(device, builder, consumer, consumerChan.first,
                  consumerChan.second, 1);

        // create flow
        builder.setInsertionPointAfter(producer);
        builder.create<FlowOp>(builder.getUnknownLoc(),
                               producer.getProducerTile(), WireBundle::DMA,
                               producerChan.second, consumer.getProducerTile(),
                               WireBundle::DMA, consumerChan.second);
      }
    }

    //===----------------------------------------------------------------------===//
    // Unroll for loops
    //===----------------------------------------------------------------------===//
    unrollForLoops(device, builder, objectFifoTiles);

    //===----------------------------------------------------------------------===//
    // Replace ops
    //===----------------------------------------------------------------------===//
    for (auto coreOp : device.getOps<CoreOp>()) {
      DenseMap<ObjectFifoAcquireOp, std::vector<BufferOp *>>
          subviews; // maps each "subview" to its buffer references (subviews
                    // are created by AcquireOps)
      DenseMap<ObjectFifoCreateOp, std::vector<int>>
          acquiresPerFifo; // maps each objFifo to indices of buffers acquired
                           // in latest subview of that objFifo (useful to
                           // cascade acquired elements to next AcquireOp)
      DenseMap<ObjectFifoCreateOp, std::vector<ObjectFifoReleaseOp>>
          releaseOps; // useful to check which ReleaseOp has taken place before
                      // an AcquireOp per objFifo
      DenseMap<ObjectFifoCreateOp, int>
          acqPerFifo; // maps each objFifo to its next index to acquire within
                      // this CoreOp
      DenseMap<ObjectFifoCreateOp, int>
          relPerFifo; // maps each objFifo to its next index to release within
                      // this CoreOp

      //===----------------------------------------------------------------------===//
      // Replace objectFifo.release ops
      //===----------------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoReleaseOp releaseOp) {
        // if objectFifo was split, replace with correct child
        checkSplitFifo<ObjectFifoReleaseOp>(
            releaseOp, coreOp.getTile().getDefiningOp<TileOp>());

        builder.setInsertionPointAfter(releaseOp);
        ObjectFifoCreateOp op =
            releaseOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();
        auto port = releaseOp.getPort();

        // update index of next element to release for this objectFifo
        updateAndReturnIndex(relPerFifo, op);

        // release locks
        int numLocks = releaseOp.relNumber();
        createUseLocks(builder, op, port, relPerFifo, numLocks,
                       LockAction::Release);

        // register release op
        if (releaseOps.find(op) != releaseOps.end())
          releaseOps[op].push_back(releaseOp);
        else {
          std::vector<ObjectFifoReleaseOp> release = {releaseOp};
          releaseOps[op] = release;
        }
      });

      //===----------------------------------------------------------------------===//
      // Replace objectFifo.acquire ops
      //===----------------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoAcquireOp acquireOp) {
        // if objectFifo was split, replace with correct child
        checkSplitFifo<ObjectFifoAcquireOp>(
            acquireOp, coreOp.getTile().getDefiningOp<TileOp>());

        builder.setInsertionPointAfter(acquireOp);
        auto port = acquireOp.getPort();
        ObjectFifoCreateOp op =
            acquireOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();

        // index of next element to acquire for this objectFifo
        int start = updateAndReturnIndex(
            acqPerFifo,
            op); // useful for keeping track of which indices are acquired

        // check how many elements have been released in between this AcquireOp
        // and the previous one
        int numRel = 0;
        for (auto relOp : releaseOps[op]) {
          ObjectFifoCreateOp otherOp =
              relOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();
          // TODO: operations may not be in the same block: currently only
          // support one block level of difference
          if (op == otherOp) {
            if (acquireOp.getOperation()->getBlock() ==
                relOp.getOperation()->getBlock()) {
              if (!acquireOp->isBeforeInBlock(relOp)) {
                releaseOps[op].erase(
                    releaseOps[op].begin()); // to ensure that we do not account
                                             // the ReleaseOps again later,
                                             // after the subview is created
                numRel += relOp.relNumber();
              }
            } else {
              Operation *acqBlockDefOp =
                  acquireOp.getOperation()->getBlock()->getParentOp();
              if (relOp.getOperation()->getBlock() ==
                  acqBlockDefOp->getBlock()) {
                if (!acqBlockDefOp->isBeforeInBlock(relOp)) {
                  releaseOps[op].erase(
                      releaseOps[op]
                          .begin()); // to ensure that we do not account
                                     // the ReleaseOps again later, after
                                     // the subview is created
                  numRel += relOp.relNumber();
                }
              } else {
                Operation *relBlockDefOp =
                    relOp.getOperation()->getBlock()->getParentOp();
                if (acquireOp.getOperation()->getBlock() ==
                    relBlockDefOp->getBlock()) {
                  if (!acquireOp->isBeforeInBlock(relBlockDefOp)) {
                    releaseOps[op].erase(
                        releaseOps[op]
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
        if (acquiresPerFifo[op].size() != 0) {
          // take into account what has already been acquired by previous
          // AcquireOp in program order
          acquiredIndices = acquiresPerFifo[op];
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
        auto &target = dev.getTargetModel();
        if (target.getTargetArch() == xilinx::AIE::AIEArch::AIE1)
          createUseLocks(builder, op, port, acqPerFifo, numCreate,
                         LockAction::Acquire);
        else
          createUseLocks(builder, op, port, acqPerFifo, numCreate,
                         LockAction::AcquireGreaterEqual);

        // create subview: buffers that were already acquired + new acquires
        for (int i = 0; i < numCreate; i++) {
          acquiredIndices.push_back(start);
          start = (start + 1) % op.getElemNumber();
        }
        std::vector<BufferOp *> subviewRefs;
        for (auto index : acquiredIndices)
          subviewRefs.push_back(&buffersPerFifo[op][index]);

        subviews[acquireOp] = subviewRefs;
        acquiresPerFifo[op] = acquiredIndices;
      });

      //===----------------------------------------------------------------------===//
      // Replace subview.access ops
      //===----------------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoSubviewAccessOp accessOp) {
        ObjectFifoAcquireOp acqOp =
            accessOp.getSubview().getDefiningOp<ObjectFifoAcquireOp>();
        accessOp.getOutput().replaceAllUsesWith(
            subviews[acqOp][accessOp.getIndex()]->getBuffer());
      });
    }

    //===----------------------------------------------------------------------===//
    // Remove old ops
    //===----------------------------------------------------------------------===//
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<AIEOpRemoval<ObjectFifoCreateOp>>(device.getContext());
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
