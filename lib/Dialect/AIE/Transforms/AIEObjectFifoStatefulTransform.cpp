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
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include <numeric>
#include <set>

#include <iostream>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-objectFifo-stateful-transform"

#define LOOP_VAR_DEPENDENCY (-2)

//===----------------------------------------------------------------------===//
// Lock Analysis
//===----------------------------------------------------------------------===//
class LockAnalysis {
  DenseMap<std::pair<Value, int>, int> locksPerTile;

public:
  LockAnalysis(DeviceOp &device) {
    // go over the locks created for each tile and update the index in
    // locksPerTile
    device.walk([&](LockOp lockOp) {
      auto tile = lockOp.getTile();
      auto lockID = lockOp.getLockIDValue();
      locksPerTile[{tile, lockID}] = 1;
    });
  }

  /// Given a tile, returns next usable lockID for that tile.
  int getLockID(TileOp &tileOp) {
    const auto &targetModel = getTargetModel(tileOp);
    for (unsigned i = 0;
         i < targetModel.getNumLocks(tileOp.getCol(), tileOp.getRow()); i++)
      if (int usageCnt = locksPerTile[{tileOp, i}]; usageCnt == 0) {
        locksPerTile[{tileOp, i}] = 1;
        return i;
      }
    return -1;
  }
};

//===----------------------------------------------------------------------===//
// DMA Channel Analysis
//===----------------------------------------------------------------------===//
class DMAChannelAnalysis {
  DenseMap<std::tuple<Value, DMAChannelDir, int>, int> channelsPerTile;

public:
  DMAChannelAnalysis(DeviceOp &device) {
    // go over the channels used for each tile and update channel map
    for (auto memOp : device.getOps<MemOp>()) {
      Region &r = memOp.getBody();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          channelsPerTile[{memOp.getTile(), op.getChannelDir(),
                           op.getChannelIndex()}] = 1;
        }
      }
    }
    for (auto memOp : device.getOps<MemTileDMAOp>()) {
      Region &r = memOp.getBody();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          channelsPerTile[{memOp.getTile(), op.getChannelDir(),
                           op.getChannelIndex()}] = 1;
        }
      }
    }
    for (auto memOp : device.getOps<ShimDMAOp>()) {
      Region &r = memOp.getBody();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          channelsPerTile[{memOp.getTile(), op.getChannelDir(),
                           op.getChannelIndex()}] = 1;
        }
      }
    }
  }

  /// Given a tile and DMAChannelDir, returns next usable channel index for
  /// that tile.
  int getDMAChannelIndex(TileOp tileOp, DMAChannelDir dir,
                         bool requiresAdjacentTileAccessChannels) {
    int maxChannelNum = 0;
    if (dir == DMAChannelDir::MM2S)
      maxChannelNum = tileOp.getNumSourceConnections(WireBundle::DMA);
    else
      maxChannelNum = tileOp.getNumDestConnections(WireBundle::DMA);

    const auto &targetModel = getTargetModel(tileOp);
    int maxChannelNumForAdjacentTile =
        targetModel.getMaxChannelNumForAdjacentMemTile(tileOp.getCol(),
                                                       tileOp.getRow());

    // if requires adjacent tile access channels, only allocate on channel 0-3,
    // and if cannot, return 0
    if (requiresAdjacentTileAccessChannels) {
      maxChannelNum = std::min(maxChannelNum, maxChannelNumForAdjacentTile);
    }

    for (int i = 0; i < maxChannelNum; i++) {
      if (int usageCnt = channelsPerTile[{tileOp.getResult(), dir, i}];
          usageCnt == 0) {
        channelsPerTile[{tileOp.getResult(), dir, i}] = 1;
        return i;
      }
    }
    return -1;
  }
};

//===----------------------------------------------------------------------===//
// Create objectFifos Pass
//===----------------------------------------------------------------------===//
struct AIEObjectFifoStatefulTransformPass
    : AIEObjectFifoStatefulTransformBase<AIEObjectFifoStatefulTransformPass> {
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
  std::vector<ObjectFifoCreateOp>
      splitBecauseLink; // objfifos which have been split because they are
  // part of a Link, not because they didn't have a shared memory module

  /// Function that returns true if two tiles in the AIE array share a memory
  /// module. share_direction is equal to:
  ///   * -1 if the shared memory module is that of the first input tile,
  ///   * 1 if it is that of the second input tile,
  ///   * 0 is no memory module is shared.
  bool isSharedMemory(TileOp a, TileOp b, int *share_direction) {
    const auto &targetModel = getTargetModel(a.getOperation());

    if ((a.isShimTile() && !b.isShimTile()) ||
        (!a.isShimTile() && b.isShimTile())) {
      *share_direction = 0;
      return false;
    }
    if ((targetModel.isMemTile(a.getCol(), a.getRow()) &&
         !targetModel.isMemTile(b.getCol(), b.getRow())) ||
        (!targetModel.isMemTile(a.getCol(), a.getRow()) &&
         targetModel.isMemTile(b.getCol(), b.getRow()))) {
      *share_direction = 0;
      return false;
    }
    bool rightShared = targetModel.isLegalMemAffinity(
        a.colIndex(), a.rowIndex(), b.colIndex(), b.rowIndex());

    bool leftShared = targetModel.isLegalMemAffinity(
        b.colIndex(), b.rowIndex(), a.colIndex(), a.rowIndex());

    if (leftShared)
      *share_direction = -1;
    else if (rightShared)
      *share_direction = 1;
    else
      *share_direction = 0;

    return leftShared || rightShared;
  }

  /// Function to retrieve ObjectFifoAllocateOp of ObjectFifoCreateOp,
  /// if it exists.
  std::optional<ObjectFifoAllocateOp>
  getOptionalAllocateOp(ObjectFifoCreateOp op) {
    ObjectFifoAllocateOp allocOp;
    auto device = op->getParentOfType<DeviceOp>();
    bool foundAlloc = false;
    for (ObjectFifoAllocateOp alloc : device.getOps<ObjectFifoAllocateOp>()) {
      if (alloc.getObjectFifo() == op) {
        if (foundAlloc)
          op.emitOpError("has more than one allocate operation");
        allocOp = alloc;
        foundAlloc = true;
      }
    }
    if (foundAlloc)
      return {allocOp};
    return {};
  }

  // Return true if the objectFifo created by createOp requires a DMA to be set
  // up. This is the case if the tiles are not adjacent (no shared memory), if
  // the objectFifo broadcasts to multiple tiles, if one of the consumers or
  // the producer wants to use the multi-dimensional address generation
  // features of the DMA, if the objectFifo is part of a LinkOp, or if the
  // via_DMA or repeatCount attributes of the objectFifo are set.
  bool requiresDMAs(ObjectFifoCreateOp createOp, int &share_direction) {
    bool hasSharedMemory = false;
    bool atLeastOneConsumerWantsTransform = false;
    bool isUsedInLinkOp = false;

    if (createOp.getVia_DMA())
      return true;

    if (createOp.getRepeatCount().has_value())
      return true;

    if (createOp.getConsumerTiles().size() == 1 &&
        createOp.getDimensionsToStream().empty()) {

      // Test for shared memory
      for (auto consumerTile : createOp.getConsumerTiles()) {
        if (auto consumerTileOp =
                dyn_cast<TileOp>(consumerTile.getDefiningOp())) {
          if (std::count(splitBecauseLink.begin(), splitBecauseLink.end(),
                         createOp))
            hasSharedMemory =
                isSharedMemory(createOp.getProducerTileOp(),
                               createOp.getProducerTileOp(), &share_direction);
          else
            hasSharedMemory = isSharedMemory(createOp.getProducerTileOp(),
                                             consumerTileOp, &share_direction);
        }
      }
    }

    // Only test for use of data layout transformations if we are in the shared
    // memory case; otherwise, we will return `true` in any case.
    if (hasSharedMemory) {
      // Even if just one of the consumers in the list of consumers wants to
      // perform a memory transform, we need to use DMAs.
      for (BDDimLayoutArrayAttr dims :
           createOp.getDimensionsFromStreamPerConsumer())
        if (!dims.empty()) {
          atLeastOneConsumerWantsTransform = true;
          break;
        }
    }

    // Check if the objectfifo operation can use shared memory for linking. If
    // the link operation is a distribute or a join operation, or if the link
    // has different memref types, DMAs are required even if shared memory is
    // available and the objectfifo should be split. Otherwise also check if the
    // via_shared_memory attribute of the objectfifo operation is set and try to
    // apply it.
    if (hasSharedMemory) {
      if (auto linkOp = getOptionalLinkOp(createOp)) {
        isUsedInLinkOp = true;
        if (!linkOp->isDistribute() && !linkOp->isJoin()) {
          auto fifoInType = llvm::cast<AIEObjectFifoType>(
              linkOp->getInputObjectFifos()[0].getElemType());
          auto producerType =
              llvm::cast<MemRefType>(fifoInType.getElementType());
          auto fifoOutType = llvm::cast<AIEObjectFifoType>(
              linkOp->getOutputObjectFifos()[0].getElemType());
          auto consumerType =
              llvm::cast<MemRefType>(fifoOutType.getElementType());
          if (consumerType != producerType) {
            // TODO: Support for different memref types through shared
            // memory without DMAs
            splitBecauseLink.push_back(createOp);
          }
          std::optional<ObjectFifoAllocateOp> opAlloc =
              getOptionalAllocateOp(createOp);
          if (opAlloc.has_value()) {
            TileOp delegate = opAlloc->getDelegateTileOp();
            int prodShareDir;
            int consShareDir;
            auto consumerTileOp = dyn_cast<TileOp>(
                createOp.getConsumerTiles()[0].getDefiningOp());
            isSharedMemory(delegate, createOp.getProducerTileOp(),
                           &prodShareDir);
            isSharedMemory(delegate, consumerTileOp, &consShareDir);
            if (prodShareDir == -1 && consShareDir == -1)
              isUsedInLinkOp = false;
            else
              splitBecauseLink.push_back(createOp);
          }
        } else {
          splitBecauseLink.push_back(createOp);
        }
      }
    }

    return !hasSharedMemory || atLeastOneConsumerWantsTransform ||
           isUsedInLinkOp;
  }

  /// Function to retrieve ObjectFifoLinkOp of ObjectFifoCreateOp,
  /// if it belongs to one.
  std::optional<ObjectFifoLinkOp> getOptionalLinkOp(ObjectFifoCreateOp op) {
    auto device = op->getParentOfType<DeviceOp>();
    for (ObjectFifoLinkOp linkOp : device.getOps<ObjectFifoLinkOp>()) {
      for (ObjectFifoCreateOp in : linkOp.getInputObjectFifos())
        if (in == op)
          return {linkOp};
      for (ObjectFifoCreateOp out : linkOp.getOutputObjectFifos())
        if (out == op)
          return {linkOp};
    }
    return {};
  }

  ObjectFifoCreateOp
  createObjectFifo(OpBuilder &builder, AIEObjectFifoType datatype,
                   std::string name, Value prodTile, Value consTile,
                   Attribute depth, BDDimLayoutArrayAttr dimensionsToStream,
                   BDDimLayoutArrayArrayAttr dimensionsFromStreamPerConsumer) {
    auto ofName = builder.getStringAttr(name);
    auto fifo = builder.create<ObjectFifoCreateOp>(
        builder.getUnknownLoc(), ofName, prodTile, consTile, depth, datatype,
        dimensionsToStream, dimensionsFromStreamPerConsumer);
    return fifo;
  }

  /// Function used to create objectFifo locks based on target architecture.
  /// Called by createObjectFifoElements().
  std::vector<LockOp> createObjectFifoLocks(OpBuilder &builder,
                                            LockAnalysis &lockAnalysis,
                                            ObjectFifoCreateOp op, int numElem,
                                            int joinDistribFactor,
                                            TileOp creation_tile,
                                            int repeatCount) {
    std::vector<LockOp> locks;
    if (op.getDisableSynchronization())
      return locks;
    auto dev = op->getParentOfType<DeviceOp>();
    auto &target = dev.getTargetModel();
    // if shimTile external buffers are collected from input code
    // create as many locks as there are external buffers
    if (creation_tile.isShimTile()) {
      numElem = 0;
      if (!externalBuffersPerFifo[op].empty())
        numElem = externalBuffersPerFifo[op].size();
    }
    if (target.getTargetArch() == AIEArch::AIE1) {
      for (int i = 0; i < numElem; i++) {
        // create corresponding aie1 locks
        int initValue = op.getInitValues().has_value() ? 1 : 0;
        int lockID = lockAnalysis.getLockID(creation_tile);
        assert(lockID >= 0 && "No more locks to allocate!");
        auto lock = builder.create<LockOp>(builder.getUnknownLoc(),
                                           creation_tile, lockID, initValue);
        lock.getOperation()->setAttr(SymbolTable::getSymbolAttrName(),
                                     builder.getStringAttr(op.name().str() +
                                                           "_lock_" +
                                                           std::to_string(i)));
        locks.push_back(lock);
      }
    } else {
      // create corresponding aie2 locks
      for (int i = 0; i < joinDistribFactor; i++) {
        auto initValues = op.getInitValues().has_value()
                              ? op.getInitValues().value().size()
                              : 0;
        int prodLockID = lockAnalysis.getLockID(creation_tile);
        assert(prodLockID >= 0 && "No more locks to allocate!");
        int prodLockValue = (numElem - initValues) * repeatCount;
        auto prodLock = builder.create<LockOp>(
            builder.getUnknownLoc(), creation_tile, prodLockID, prodLockValue);
        prodLock.getOperation()->setAttr(
            SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name().str() + "_prod_lock_" +
                                  std::to_string(i)));
        locks.push_back(prodLock);

        int consLockID = lockAnalysis.getLockID(creation_tile);
        assert(consLockID >= 0 && "No more locks to allocate!");
        int consLockValue = initValues * repeatCount;
        auto consLock = builder.create<LockOp>(
            builder.getUnknownLoc(), creation_tile, consLockID, consLockValue);
        consLock.getOperation()->setAttr(
            SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name().str() + "_cons_lock_" +
                                  std::to_string(i)));
        locks.push_back(consLock);
      }
    }
    return locks;
  }

  /// Function to calculate total memory usage on a specific tile
  /// based on all buffers allocated to that tile from buffersPerFifo map
  int calculateCurrentUsedMemory(
      TileOp targetTile,
      DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>> &buffersPerFifo,
      std::vector<BufferOp> &buffers) {
    int totalUsedMemory = 0;

    // Iterate through all ObjectFifos and their buffers
    for (auto &[fifoOp, bufferList] : buffersPerFifo) {
      for (auto &buffer : bufferList) {
        // Check if this buffer is allocated on the target tile
        if (buffer.getTile() == targetTile.getResult()) {
          auto bufferSizeBytes = buffer.getAllocationSize();
          totalUsedMemory += bufferSizeBytes;
        }
      }
    }

    // Also count buffers that are not in buffersPerFifo
    for (auto &buffer : buffers) {
      // Check if this buffer is allocated on the target tile
      if (buffer.getTile() == targetTile.getResult()) {
        auto bufferSizeBytes = buffer.getAllocationSize();
        totalUsedMemory += bufferSizeBytes;
      }
    }

    return totalUsedMemory;
  }

  /// Function to analyze cross-tile buffer allocations in splitFifos
  /// Returns a simple map of (ObjectFifoCreateOp, bool) indicating cross-tile
  /// issues
  std::map<ObjectFifoCreateOp, bool> analyzeCrossTileFIFOBuffers() {
    std::map<ObjectFifoCreateOp, bool> crossTileMap;

    for (size_t i = 0; i < splitFifos.size(); i++) {
      auto &[producerFifo, consumerFifos] = splitFifos[i];

      // Analyze producer buffers
      bool producerHasCrossTile = false;

      ObjectFifoCreateOp target = producerFifo;
      auto linkOp = getOptionalLinkOp(producerFifo);

      if (linkOp && objFifoLinks.find(*linkOp) != objFifoLinks.end()) {
        target = objFifoLinks[*linkOp]; // Use the linked target FIFO
      }

      if (buffersPerFifo.find(target) != buffersPerFifo.end()) {
        // For each FIFO (producer and consumer):
        auto &producerBuffers = buffersPerFifo[target];
        TileOp expectedTile = target.getProducerTileOp();
        for (auto &buffer : producerBuffers) {
          TileOp bufferTile = buffer.getTile().getDefiningOp<TileOp>();
          if (bufferTile != expectedTile) {
            producerHasCrossTile = true;
            break;
          }
        }
      }
      crossTileMap[producerFifo] = producerHasCrossTile;

      // Analyze consumer buffers
      for (auto &consumerFifo : consumerFifos) {
        bool consumerHasCrossTile = false;
        ObjectFifoCreateOp target = consumerFifo;
        auto linkOp = getOptionalLinkOp(consumerFifo);
        if (linkOp && objFifoLinks.find(*linkOp) != objFifoLinks.end()) {
          target = objFifoLinks[*linkOp]; // Use the linked target FIFO
        }

        if (buffersPerFifo.find(target) != buffersPerFifo.end()) {
          // For each FIFO (producer and consumer):
          auto &consumerBuffers = buffersPerFifo[target];
          TileOp expectedTile = target.getProducerTileOp();
          for (auto &buffer : consumerBuffers) {
            TileOp bufferTile = buffer.getTile().getDefiningOp<TileOp>();
            if (bufferTile != expectedTile) {
              consumerHasCrossTile = true;
              break;
            }
          }
        }
        crossTileMap[consumerFifo] = consumerHasCrossTile;
      }
    }
    return crossTileMap;
  }

  /// Helper function to find a tile at specific coordinates.
  /// If a tile is not found, it creates a new one and returns it.
  /// hostTile is the original tile from which we are searching for neighbors.
  /// we create the new tile below the hostTile
  TileOp findOrCreateTile(OpBuilder &builder, DeviceOp &dev, TileOp hostTile,
                          int col, int row) {
    // First, try to find an existing tile
    for (auto tile : dev.getOps<TileOp>()) {
      if (tile.getCol() == col && tile.getRow() == row) {
        return tile;
      }
    }

    // If not found, create a new one.
    OpBuilder::InsertionGuard g(builder);

    auto savedInsertionPoint = builder.saveInsertionPoint();

    // Find the last buffer operation after the host tile
    Operation *insertAfter = hostTile.getOperation();
    Operation *nextOp = insertAfter->getNextNode();
    while (nextOp && isa<BufferOp>(nextOp)) {
      insertAfter = nextOp;
      nextOp = nextOp->getNextNode();
    }

    builder.setInsertionPointAfter(insertAfter);
    auto newTile = builder.create<TileOp>(builder.getUnknownLoc(), col, row);

    builder.restoreInsertionPoint(savedInsertionPoint);

    return newTile;
  }

  /// Function used to create objectFifo elements and their locks.
  /// It maps the input objectFifo to associated buffers and locks.
  void createObjectFifoElements(OpBuilder &builder, LockAnalysis &lockAnalysis,
                                ObjectFifoCreateOp op, int share_direction) {
    if (!op.size())
      return;

    std::vector<BufferOp> buffers;
    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int numElem = op.size();
    int of_elem_index = 0; // used to give objectFifo elements a symbolic name

    // if this objectFifo is linked to another, check if the other's elements
    // have already been created: if none of the output objectfifos of the link
    // have initValues, then the elements that are created are those of the
    // objFifo with elements of bigger size
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
        // check if output objectfifo has initValues
        if (fifoOut.getInitValues().has_value()) {
          if (fifoOut.name() != op.name())
            return;
        } else {
          // check which objectfifo of the link has bigger size
          auto fifoInType = llvm::cast<AIEObjectFifoType>(fifoIn.getElemType());
          auto elemInType = llvm::cast<MemRefType>(fifoInType.getElementType());
          int inSize = elemInType.getNumElements();

          auto fifoOutType =
              llvm::cast<AIEObjectFifoType>(fifoOut.getElemType());
          auto elemOutType =
              llvm::cast<MemRefType>(fifoOutType.getElementType());

          if (int outSize = elemOutType.getNumElements(); inSize >= outSize) {
            if (op.name() != fifoIn.name())
              return;
          } else {
            if (fifoOut.name() != op.name())
              return;
          }
        }
      }
    }

    TileOp creation_tile;
    auto consumerTileOp =
        dyn_cast<TileOp>(op.getConsumerTiles()[0].getDefiningOp());
    if (share_direction == 0 || share_direction == -1)
      creation_tile = op.getProducerTileOp();
    else
      creation_tile = consumerTileOp;

    std::optional<ObjectFifoAllocateOp> opAlloc = getOptionalAllocateOp(op);
    if (opAlloc.has_value()) {
      TileOp delegate = opAlloc->getDelegateTileOp();
      int prodShareDir;
      int consShareDir;
      isSharedMemory(delegate, op.getProducerTileOp(), &prodShareDir);
      isSharedMemory(delegate, consumerTileOp, &consShareDir);
      if (prodShareDir == -1 && consShareDir == -1)
        creation_tile = delegate;
      else
        opAlloc->emitOpError("objectfifo has no shared memory access to "
                             "delegate tile's memory module");
    }

    // Reset opbuilder location to after the last tile declaration
    Operation *t = nullptr;
    auto dev = op->getParentOfType<DeviceOp>();
    for (auto tile_op : dev.getBody()->getOps<TileOp>()) {
      t = tile_op.getOperation();
    }

    builder.setInsertionPointAfter(t);
    for (int i = 0; i < numElem; i++) {

      mlir::ElementsAttr initValues = nullptr;
      if (!creation_tile.isShimTile()) {
        if (op.getInitValues().has_value()) {
          initValues =
              llvm::cast<mlir::ElementsAttr>(op.getInitValues().value()[i]);
        }

        auto elementType = elemType.getElementType();

        DataLayout dataLayout = DataLayout::closest(op.getOperation());
        int64_t elementBitWidth = dataLayout.getTypeSizeInBits(elementType);

        auto totalSizeBytes = elemType.getNumElements() * elementBitWidth / 8;
        auto &targetModel = dev.getTargetModel();

        int maxDataMemorySize = 0;
        if (creation_tile.isMemTile())
          maxDataMemorySize =
              targetModel.getMemTileSize(); // getMemTileSize returns in Bytes
        else
          maxDataMemorySize =
              targetModel
                  .getLocalMemorySize(); // getLocalMemorySize returns in Bytes

        // also need to count the buffers that are not in buffersPerFifo
        int currentUsedMemory =
            calculateCurrentUsedMemory(creation_tile, buffersPerFifo, buffers);

        // Check if current tile can hold the new buffer or not
        TileOp current_buf_allocation_tile =
            creation_tile; // used to keep track of the tile where the buffer is
                           // allocated
        if (static_cast<int>(currentUsedMemory + totalSizeBytes) >
            maxDataMemorySize) {
          // if not, check if the neighbour can hold the new buffer or not
          // Find neighbor tiles with shared memory
          std::vector<TileOp> neighborTiles;
          int currentCol = creation_tile.getCol();
          int currentRow = creation_tile.getRow();

          // Check tile to the left
          if (currentCol > 0) {
            TileOp leftTile = findOrCreateTile(builder, dev, creation_tile,
                                               currentCol - 1, currentRow);

            int share_direction = 0;
            if (isSharedMemory(creation_tile, leftTile, &share_direction)) {
              neighborTiles.push_back(leftTile);
            }
          }

          // Check tile to the right
          if (currentCol < (targetModel.columns() - 1)) {
            TileOp rightTile = findOrCreateTile(builder, dev, creation_tile,
                                                currentCol + 1, currentRow);
            int share_direction = 0;
            if (isSharedMemory(creation_tile, rightTile, &share_direction)) {
              neighborTiles.push_back(rightTile);
            }
          }

          // try to allocate on neighbor tiles
          if (!neighborTiles.empty()) {
            for (auto &tile : neighborTiles) {
              // Try to allocate on this neighbor tile
              int neighborUsedMemory =
                  calculateCurrentUsedMemory(tile, buffersPerFifo, buffers);
              if (static_cast<int>(neighborUsedMemory + totalSizeBytes) <=
                  maxDataMemorySize) {
                // Allocate buffer on neighbor tile, change creation_tile to be
                // this neighbour tile
                current_buf_allocation_tile = tile;
                break;
              }
            }
          }
        }
        auto buff = builder.create<BufferOp>(
            builder.getUnknownLoc(), elemType, current_buf_allocation_tile,
            builder.getStringAttr(op.name().str() + "_buff_" +
                                  std::to_string(of_elem_index)),
            /*address*/ nullptr, initValues,
            /*mem_bank*/ nullptr);
        buffers.push_back(buff);
      }
      of_elem_index++;
    }

    int repeatCount = 1;
    int joinDistribFactor = 1;
    if (op.getRepeatCount().has_value())
      repeatCount = op.getRepeatCount().value();
    if (linked) {
      if (linkOp->getRepeatCount().has_value())
        repeatCount = linkOp->getRepeatCount().value();
      if (linkOp->isDistribute())
        joinDistribFactor *= linkOp->getFifoOuts().size();
      else if (linkOp->isJoin())
        joinDistribFactor *= linkOp->getFifoIns().size();
      objFifoLinks[*linkOp] = op;
    }
    std::vector<LockOp> locks =
        createObjectFifoLocks(builder, lockAnalysis, op, numElem,
                              joinDistribFactor, creation_tile, repeatCount);
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
                MyOp buff, int offset, int len, Block *succ,
                BDDimLayoutArrayAttr dims, BDPadLayoutArrayAttr padDimensions,
                std::optional<PacketInfoAttr> bdPacket) {
    if (acqLock)
      builder.create<UseLockOp>(builder.getUnknownLoc(), acqLock, acqLockAction,
                                acqMode);
    if (bdPacket) {
      builder.create<DMABDPACKETOp>(builder.getUnknownLoc(),
                                    bdPacket->getPktType(),
                                    bdPacket->getPktId());
    }
    if (!dims.getValue().empty() && padDimensions) {
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, dims,
                              padDimensions);
    } else if (!dims.getValue().empty()) {
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, dims);
    } else {
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len);
    }
    if (acqLock)
      builder.create<UseLockOp>(builder.getUnknownLoc(), relLock,
                                LockAction::Release, relMode);
    builder.create<NextBDOp>(builder.getUnknownLoc(), succ);
  }

  /// Function used to create a Bd block.
  /// If lockMode is 0 we create a consumerDMA (i.e. on producer tile) else a
  /// producerDMA (i.e. on consumer tile).
  template <typename MyOp>
  void createBdBlock(OpBuilder &builder, ObjectFifoCreateOp op, int lockMode,
                     int acqNum, int relNum, MyOp buff, int offset, int len,
                     DMAChannelDir channelDir, size_t lockIndex, Block *succ,
                     BDDimLayoutArrayAttr dims,
                     BDPadLayoutArrayAttr padDimensions,
                     std::optional<PacketInfoAttr> bdPacket,
                     bool distribOrJoin = false) {
    LockOp acqLock;
    LockOp relLock;
    int acqMode = 1;
    int relMode = 1;
    auto acqLockAction = LockAction::Acquire;
    if (locksPerFifo[op].size() > 0) {
      auto dev = op->getParentOfType<DeviceOp>();
      if (auto &target = dev.getTargetModel();
          target.getTargetArch() == AIEArch::AIE1) {
        acqMode = lockMode == 0 ? 1 : 0;
        relMode = lockMode == 0 ? 0 : 1;
        acqLock = locksPerFifo[op][lockIndex];
        relLock = locksPerFifo[op][lockIndex];
      } else {
        acqMode = acqNum;
        relMode = relNum;
        acqLockAction = LockAction::AcquireGreaterEqual;
        int prodLockIndex = 0;
        int consLockIndex = 1;
        if (distribOrJoin) {
          prodLockIndex = lockIndex * 2;
          consLockIndex = lockIndex * 2 + 1;
        }
        acqLock = channelDir == DMAChannelDir::S2MM
                      ? locksPerFifo[op][prodLockIndex]
                      : locksPerFifo[op][consLockIndex];
        relLock = channelDir == DMAChannelDir::S2MM
                      ? locksPerFifo[op][consLockIndex]
                      : locksPerFifo[op][prodLockIndex];
      }
    }
    createBd(builder, acqLock, acqMode, acqLockAction, relLock, relMode, buff,
             offset, len, succ, dims, padDimensions, bdPacket);
  }

  /// Function that either calls createAIETileDMA(), createShimDMA() or
  /// createMemTileDMA() based on op tile row value.
  void createDMA(DeviceOp &device, OpBuilder &builder, ObjectFifoCreateOp op,
                 DMAChannelDir channelDir, int channelIndex, int lockMode,
                 BDDimLayoutArrayAttr dims, BDPadLayoutArrayAttr pad_dims,
                 std::optional<PacketInfoAttr> bdPacket) {
    if (op.getProducerTileOp().isShimTile()) {
      createShimDMA(device, builder, op, channelDir, channelIndex, lockMode,
                    dims, bdPacket);
    } else if (op.getProducerTileOp().isMemTile()) {
      BDPadLayoutArrayAttr padDims = nullptr;
      if (channelDir == DMAChannelDir::MM2S && pad_dims)
        padDims = pad_dims;
      createMemTileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       dims, padDims, bdPacket);
    } else {
      createAIETileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       dims, bdPacket);
    }
  }

  /// Function used to create a MemOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createAIETileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode,
                        BDDimLayoutArrayAttr dims,
                        std::optional<PacketInfoAttr> bdPacket) {
    size_t numBlocks = op.size();
    if (numBlocks == 0)
      return;

    int acqNum = 1;
    int relNum = 1;

    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int len = elemType.getNumElements();

    // check for repeat count
    int repeatCount = 1;
    if (op.getRepeatCount().has_value())
      repeatCount = op.getRepeatCount().value();

    // search for the buffers/locks (based on if this objFifo has a link)
    ObjectFifoCreateOp target = op;
    if (std::optional<ObjectFifoLinkOp> linkOp = getOptionalLinkOp(op);
        linkOp.has_value()) {
      if (objFifoLinks.find(linkOp.value()) != objFifoLinks.end()) {
        target = objFifoLinks[linkOp.value()];
        if (target == op) {
          if (linkOp->getRepeatCount().has_value()) {
            acqNum *= linkOp->getRepeatCount().value();
            relNum *= linkOp->getRepeatCount().value();
          }
        }
      }
    }

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
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(device.getBody()->getTerminator());
      auto newMemOp =
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
                               channelIndex, /*repeatCout*/ 0, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t elemIndex = 0;
    size_t totalBlocks = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (elemIndex >= buffersPerFifo[target].size())
        break;
      for (int r = 0; r < repeatCount; r++) {
        if (totalBlocks == numBlocks * repeatCount - 1)
          succ = bdBlock;
        else
          succ = builder.createBlock(endBlock);

        builder.setInsertionPointToStart(curr);
        createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                                buffersPerFifo[target][elemIndex], /*offset*/ 0,
                                len, channelDir, elemIndex, succ, dims, nullptr,
                                bdPacket);
        curr = succ;
        totalBlocks++;
      }
      elemIndex++;
    }
  }

  /// Function used to create a ShimDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createShimDMA(DeviceOp &device, OpBuilder &builder,
                     ObjectFifoCreateOp op, DMAChannelDir channelDir,
                     int channelIndex, int lockMode, BDDimLayoutArrayAttr dims,
                     std::optional<PacketInfoAttr> bdPacket) {
    size_t numBlocks = externalBuffersPerFifo[op].size();
    if (numBlocks == 0)
      return;

    int acqNum = 1;
    int relNum = 1;

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
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(device.getBody()->getTerminator());
      auto newDMAOp = builder.create<ShimDMAOp>(
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
                               channelIndex, /*repeatCout*/ 0, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t elemIndex = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (elemIndex >= externalBuffersPerFifo[op].size())
        break;
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      MemRefType buffer = externalBuffersPerFifo[op][elemIndex].getType();
      int len = buffer.getNumElements();
      builder.setInsertionPointToStart(curr);
      createBdBlock<ExternalBufferOp>(builder, op, lockMode, acqNum, relNum,
                                      externalBuffersPerFifo[op][elemIndex],
                                      /*offset*/ 0, len, channelDir, elemIndex,
                                      succ, dims, nullptr, bdPacket);
      curr = succ;
      elemIndex++;
    }
  }

  /// Function used to create a MemTileDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createMemTileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode,
                        BDDimLayoutArrayAttr dims,
                        BDPadLayoutArrayAttr padDimensions,
                        std::optional<PacketInfoAttr> bdPacket) {
    size_t numBlocks = op.size();
    if (numBlocks == 0)
      return;

    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int lenOut = elemType.getNumElements();
    int acqNum = 1;
    int relNum = 1;

    // check for repeat count
    int repeatCount = 1;
    if (op.getRepeatCount().has_value())
      repeatCount = op.getRepeatCount().value();

    // search for the buffers/locks (based on if this objFifo has a link)
    // identify size difference between input and output memrefs
    ObjectFifoCreateOp target = op;
    bool isDistribute = false;
    bool isJoin = false;
    int extraOffset = 0;
    int joinDistribFactor = 1;
    int joinDistribLockIndex = 0;
    auto linkOp = getOptionalLinkOp(op);
    if (linkOp) {
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end()) {
        target = objFifoLinks[*linkOp];
        auto srcOffsets = linkOp->getSrcOffsets();
        auto dstOffsets = linkOp->getDstOffsets();

        if (linkOp->getRepeatCount().has_value())
          if (linkOp->getInputObjectFifos()[0] == op) {
            acqNum *= linkOp->getRepeatCount().value();
            relNum *= linkOp->getRepeatCount().value();
          }

        if (linkOp->isJoin()) {
          // compute offset and length
          isJoin = true;
          if (target == op) {
            joinDistribFactor *= linkOp->getFifoIns().size();
          } else {
            int i = 0;
            for (auto fifoIn : linkOp->getInputObjectFifos()) {
              if (fifoIn.name() == op.name())
                break;
              i++;
            }
            extraOffset = *getConstantIntValue(srcOffsets[i]);
            lenOut = linkOp->getJoinTransferLengths()[i];
            joinDistribLockIndex = i;
          }
        } else if (linkOp->isDistribute()) {
          // compute offset and length
          isDistribute = true;
          if (target == op) {
            joinDistribFactor *= linkOp->getFifoOuts().size();
          } else {
            int i = 0;
            for (auto fifoOut : linkOp->getOutputObjectFifos()) {
              if (fifoOut.name() == op.name())
                break;
              i++;
            }
            extraOffset = *getConstantIntValue(dstOffsets[i]);
            lenOut = linkOp->getDistributeTransferLengths()[i];
            joinDistribLockIndex = i;
          }
        } else {
          if (target != op) {
            auto targetFifo =
                llvm::cast<AIEObjectFifoType>(target.getElemType());
            auto targetElemType =
                llvm::cast<MemRefType>(targetFifo.getElementType());
            lenOut = targetElemType.getNumElements();
          }
        }

        // check if current op is of smaller size in link
        if (target != op) {
          numBlocks = target.size();
        }
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
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(device.getBody()->getTerminator());
      auto newDMAOp =
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
                               channelIndex, /*repeatCout*/ 0, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t elemIndex = 0;
    size_t lockIndex = 0;
    size_t totalBlocks = 0;
    bool distribOrJoin = false;
    for (size_t i = 0; i < numBlocks; i++) {
      if (elemIndex >= buffersPerFifo[target].size())
        break;
      for (int r = 0; r < repeatCount * joinDistribFactor; r++) {
        if (totalBlocks == numBlocks * repeatCount * joinDistribFactor - 1) {
          succ = bdBlock;
        } else {
          succ = builder.createBlock(endBlock);
        }

        builder.setInsertionPointToStart(curr);
        int offset = 0;
        if (isDistribute || isJoin) {
          distribOrJoin = true;
          if (target == op) {
            if (isDistribute) {
              offset = *getConstantIntValue(linkOp->getDstOffsets()[r]);
              lenOut = linkOp->getDistributeTransferLengths()[r];
            } else {
              offset = *getConstantIntValue(linkOp->getSrcOffsets()[r]);
              lenOut = linkOp->getJoinTransferLengths()[r];
            }
            lockIndex = r % joinDistribFactor;
          } else {
            offset = extraOffset;
            lockIndex = joinDistribLockIndex;
          }
        } else {
          lockIndex = elemIndex;
        }

        createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                                buffersPerFifo[target][elemIndex], offset,
                                lenOut, channelDir, lockIndex, succ, dims,
                                padDimensions, bdPacket, distribOrJoin);
        curr = succ;
        totalBlocks++;
      }
      elemIndex++;
    }
  }

  // Function that computes the Least Common Multiplier of the values
  // of a vector.
  int computeLCM(std::set<int> values) {
    int lcm = 1;
    for (int i : values)
      lcm = i * lcm / std::gcd(i, lcm);
    return lcm;
  }

  // Function that unrolls for-loops that contain objectFifo operations.
  LogicalResult unrollForLoops(DeviceOp &device, OpBuilder &builder,
                               std::set<TileOp> objectFifoTiles) {
    for (auto coreOp : device.getOps<CoreOp>()) {
      if (objectFifoTiles.count(coreOp.getTileOp()) > 0) {
        std::vector<scf::ForOp> unrolledLoops;
        std::map<Operation *, bool> foundMap;
        std::map<Operation *, int64_t> remainderMap;
        std::map<Operation *, int64_t> tripCountMap;
        WalkResult res = coreOp.walk([&](scf::ForOp forLoop) {
          // look for operations on objectFifos
          // when multiple fifos in same loop, must use the smallest
          // common multiplier as the unroll factor
          foundMap[forLoop.getOperation()] = false;
          std::set<int> objFifoSizes;
          Block *body = forLoop.getBody();
          remainderMap[forLoop.getOperation()] = 0;
          for (auto acqOp : body->getOps<ObjectFifoAcquireOp>()) {
            if (acqOp.getOperation()->getParentOp() == forLoop) {
              foundMap[forLoop.getOperation()] = true;
              ObjectFifoCreateOp op = acqOp.getObjectFifo();
              objFifoSizes.insert(op.size());
            }
          }
          // If the loop doesn't have acquire and release locks
          // Push it to the unrolledLoops to avoid unrolling
          if (!foundMap[forLoop.getOperation()]) {
            unrolledLoops.push_back(forLoop);
            return WalkResult::advance();
          }
          // Walk in the loop region to unroll the loop and its remainder
          Region *region = forLoop->getParentRegion();
          scf::ForOp prevLoop;
          prevLoop = forLoop;
          tripCountMap[prevLoop.getOperation()] = 0;
          while (remainderMap[prevLoop.getOperation()] > 1 ||
                 foundMap[prevLoop.getOperation()]) {
            region->walk([&](scf::ForOp remLoop) {
              bool skipLoop = false;
              int64_t tripCount = 0;
              if (remLoop.getSingleLowerBound() &&
                  remLoop.getSingleUpperBound() && remLoop.getSingleStep()) {
                tripCount = constantTripCount(*(remLoop.getSingleLowerBound()),
                                              *(remLoop.getSingleUpperBound()),
                                              *(remLoop.getSingleStep()))
                                .value_or(0);
              }
              int unrollFactor =
                  computeLCM(objFifoSizes); // also counts original loop body
              // Loop ids are not unique.
              // Sometimes, immediately after unrolling, the unrolled loop
              // and the one next to it (can be the remainder loop or an
              // independent loop) will have the same ID. This makes it
              // difficult to identify which loop needs to be unrolled.
              // Once it restarts walking from start, it ends up allocating
              // new ID to each loop.
              if (remainderMap[prevLoop.getOperation()] > 1 &&
                  foundMap[remLoop.getOperation()] == false &&
                  prevLoop != remLoop) {
                skipLoop = true;
              }
              if (std::count(unrolledLoops.begin(), unrolledLoops.end(),
                             remLoop) == 0 &&
                  !skipLoop) {
                tripCountMap[remLoop.getOperation()] = tripCount;
                // if loop iterations < unrollFactor, unroll the loop fully
                if (tripCountMap[remLoop.getOperation()] < unrollFactor)
                  unrollFactor = tripCountMap[remLoop.getOperation()];
                // If unrollFactor = 0,divide by zero
                if (unrollFactor == 0) {
                  remLoop.emitOpError()
                      << "could not be unrolled with unrollFactor = 0, check "
                         "loop boundaries."
                      << "\n";
                  return WalkResult::interrupt();
                }
                remainderMap[remLoop.getOperation()] =
                    tripCountMap[remLoop.getOperation()] % unrollFactor;
                auto step = remLoop.getStep()
                                .getDefiningOp<arith::ConstantOp>()
                                .getValue();
                int64_t step_value = llvm::dyn_cast<IntegerAttr>(step).getInt();

                if (step_value < unrollFactor ||
                    foundMap[remLoop.getOperation()]) {
                  // Process the for loop
                  if (failed(mlir::loopUnrollByFactor(remLoop, unrollFactor))) {
                    remLoop.emitOpError()
                        << "could not be unrolled with unrollFactor: "
                        << unrollFactor << "\n";
                    return WalkResult::interrupt();
                  }
                  unrolledLoops.push_back(remLoop);
                  foundMap[remLoop.getOperation()] = false;
                } else {
                  remainderMap[remLoop.getOperation()] = 0;
                  foundMap[remLoop.getOperation()] = false;
                }
              } else {
                remainderMap[remLoop.getOperation()] = 0;
                foundMap[remLoop.getOperation()] = false;
              }
              prevLoop = remLoop;
              return WalkResult::advance();
            });
          }
          return WalkResult::advance();
        });
        if (res.wasInterrupted())
          return failure();
      }
    }
    return success();
  }

  // Function that generates the IR to update runtime state of objectfifo
  // accesses. Called by dynamicGlobalObjectFifos().
  void updateGlobalNextIndex(OpBuilder &builder, ObjectFifoReleaseOp relOp,
                             BufferOp globalNextIndex, arith::ConstantOp index,
                             arith::ConstantOp size) {
    builder.setInsertionPointAfter(relOp);
    Value oldCounter = builder.create<memref::LoadOp>(
        builder.getUnknownLoc(), globalNextIndex,
        ValueRange(ArrayRef({index.getResult()})));
    Value val = builder.create<arith::ConstantOp>(
        oldCounter.getLoc(), builder.getI32IntegerAttr(relOp.getSize()));
    Value sum = builder.create<arith::AddIOp>(val.getLoc(), oldCounter, val);
    Value isGreaterEqual = builder.create<arith::CmpIOp>(
        sum.getLoc(), arith::CmpIPredicate::sge, sum, size);
    Value newCounter = builder.create<arith::SelectOp>(
        sum.getLoc(), isGreaterEqual,
        builder.create<arith::SubIOp>(sum.getLoc(), sum, size), sum);
    builder.create<memref::StoreOp>(size.getLoc(), newCounter, globalNextIndex,
                                    ValueRange(ArrayRef({index.getResult()})));
  }

  // Function that generates the IR for objectfifo accesses to be handled at
  // runtime.
  LogicalResult dynamicGlobalObjectFifos(DeviceOp &device, OpBuilder &builder,
                                         std::set<TileOp> objectFifoTiles) {
    for (auto coreOp : device.getOps<CoreOp>()) {
      if (objectFifoTiles.count(coreOp.getTileOp()) <= 0)
        continue;
      if (objectFifoTiles.count(coreOp.getTileOp()) > 0) {
        // For each core: count the number of objectFifos and create
        // a global buffer just before the core to track index of
        // next object to access.
        // !! NOTE !! objectFifos with same producer / consumer tile
        // need two counters (accessed based on the ObjectFifoPort)
        std::map<std::pair<ObjectFifoCreateOp, ObjectFifoPort>, int> fifoSizes;
        // Also, keep a map of the ConstantOps for the indices per OF
        // and a map with the ConstantOps for the sizes per OF.
        std::map<std::pair<ObjectFifoCreateOp, ObjectFifoPort>,
                 arith::ConstantOp>
            globalIndices;
        std::map<std::pair<ObjectFifoCreateOp, ObjectFifoPort>,
                 arith::ConstantOp>
            constantSizes;

        int index = 0;
        builder.setInsertionPointToStart(&(coreOp.getBody().front()));
        Value initVal = builder.create<arith::ConstantOp>(
            builder.getUnknownLoc(), builder.getI32IntegerAttr(0));
        coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
          ObjectFifoCreateOp op = acqOp.getObjectFifo();
          ObjectFifoPort port = acqOp.getPort();
          if (fifoSizes.find({op, port}) == fifoSizes.end()) {
            fifoSizes[{op, port}] = op.size();
            auto indexOp = builder.create<arith::ConstantOp>(
                initVal.getLoc(), builder.getIndexAttr(index));
            globalIndices[{op, port}] = indexOp;
            index++;
            auto size = builder.create<arith::ConstantOp>(
                indexOp.getLoc(), builder.getI32IntegerAttr(op.size()));
            constantSizes[{op, port}] = size;
          }
        });
        builder.setInsertionPoint(coreOp);
        auto memrefTy =
            MemRefType::get(SmallVector<int64_t>{(int64_t)fifoSizes.size()},
                            builder.getI32Type());
        auto globalNextIndex = builder.create<BufferOp>(
            builder.getUnknownLoc(), memrefTy, coreOp.getTile(),
            /*sym_name*/ nullptr, /*address*/ nullptr,
            /*initial_value*/ nullptr, /*mem_bank*/ nullptr);

        // Initialize all counters in the global buffers to 0.
        for (auto i : constantSizes) {
          builder.setInsertionPointAfter(i.second);
          builder.create<memref::StoreOp>(
              builder.getUnknownLoc(), initVal, globalNextIndex,
              ValueRange(ArrayRef({globalIndices[i.first].getResult()})));
        }

        // Walk the code:
        // - after each ObjectFifoReleaseOp:
        //    - globalNextIndex: add #rel modulo objfifo depth
        // - before each ObjectFifoAcquireOp:
        //    - globalNextIndex: load index and use it to index_switch (one
        //    IndexSwithOp per AccessOp)
        WalkResult res = coreOp.walk([&](Operation *op) {
          if (auto relOp = dyn_cast<ObjectFifoReleaseOp>(op)) {
            ObjectFifoCreateOp createOp = relOp.getObjectFifo();
            ObjectFifoPort port = relOp.getPort();
            updateGlobalNextIndex(builder, relOp, globalNextIndex,
                                  globalIndices[{createOp, port}],
                                  constantSizes[{createOp, port}]);
          }
          if (auto acqOp = dyn_cast<ObjectFifoAcquireOp>(op)) {
            std::vector<ObjectFifoSubviewAccessOp> accessOps;
            for (auto u : acqOp->getUsers())
              if (auto accessOp = dyn_cast<ObjectFifoSubviewAccessOp>(u))
                accessOps.push_back(accessOp);

            for (auto accessOp : accessOps) {
              ObjectFifoCreateOp createOp = acqOp.getObjectFifo();
              ObjectFifoPort port = acqOp.getPort();

              // Single switch case
              if (fifoSizes[{createOp, port}] == 1)
                return WalkResult::advance();

              // Create a switch for each subview access
              builder.setInsertionPointAfter(accessOp);
              auto switchIndexAsInteger = builder.create<memref::LoadOp>(
                  builder.getUnknownLoc(), globalNextIndex,
                  ValueRange(
                      ArrayRef({globalIndices[{createOp, port}].getResult()})));
              auto switchIndex = builder.create<arith::IndexCastOp>(
                  builder.getUnknownLoc(), builder.getIndexType(),
                  switchIndexAsInteger);
              unsigned caseRegionCounts = fifoSizes[{createOp, port}];
              SmallVector<int64_t, 4> caseValues;
              for (int i = 0; i < fifoSizes[{createOp, port}]; ++i) {
                caseValues.push_back(i);
              }
              auto cases =
                  DenseI64ArrayAttr::get(builder.getContext(), caseValues);
              auto switchOp = builder.create<scf::IndexSwitchOp>(
                  switchIndex.getLoc(),
                  TypeRange({buffersPerFifo[createOp][0].getType()}),
                  switchIndex, cases, caseRegionCounts);
              // Create default case of IndexSwitchOp
              builder.createBlock(&switchOp.getDefaultRegion());
              auto bufferIndex = (accessOp.getIndex()) % createOp.size();
              builder.setInsertionPointToStart(&(switchOp.getDefaultBlock()));
              builder.create<scf::YieldOp>(
                  builder.getUnknownLoc(),
                  buffersPerFifo[createOp][bufferIndex].getResult());
              for (int i = 0; i < fifoSizes[{createOp, port}]; ++i) {
                // Create other cases of IndexSwitchOp
                builder.createBlock(&switchOp.getCaseRegions()[i]);
                builder.setInsertionPoint(&switchOp.getCaseBlock(i),
                                          switchOp.getCaseBlock(i).begin());
                int bufferToBeAccesed =
                    (accessOp.getIndex() + i) % fifoSizes[{createOp, port}];
                builder.create<scf::YieldOp>(
                    switchOp.getCaseRegions()[i].getLoc(),
                    buffersPerFifo[createOp][bufferToBeAccesed].getResult());
              }

              // Replace all uses of accessed objectfifo buffers with
              // results of switchOps
              accessOp.getOutput().replaceAllUsesWith(switchOp.getResult(0));
            }
          }
          return WalkResult::advance();
        });
        if (res.wasInterrupted())
          return failure();
      }
    }
    return success();
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
    auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
    if (auto linkOp = getOptionalLinkOp(op))
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
        target = objFifoLinks[*linkOp];

    auto dev = op->getParentOfType<DeviceOp>();
    if (!dev.getTargetModel().hasProperty(AIETargetModel::UsesSemaphoreLocks)) {

      if (locksPerFifo[target].size() == 0) {
        for (int i = 0; i < numLocks; i++) {
          int lockID = acc[{op, portNum}];
          acc[{op, portNum}] =
              (lockID + 1) % op.size(); // update to next objFifo elem
        }
        return;
      }

      int lockMode = 0;
      if ((port == ObjectFifoPort::Produce &&
           lockAction == LockAction::Release) ||
          (port == ObjectFifoPort::Consume &&
           lockAction == LockAction::Acquire))
        lockMode = 1;
      for (int i = 0; i < numLocks; i++) {
        int lockID = acc[{op, portNum}];
        builder.create<UseLockOp>(builder.getUnknownLoc(),
                                  locksPerFifo[target][lockID], lockAction,
                                  lockMode);
        acc[{op, portNum}] =
            (lockID + 1) % op.size(); // update to next objFifo elem
      }
    } else {
      if (numLocks == 0)
        return;

      if (locksPerFifo[target].size() == 0) {
        acc[{op, portNum}] = (acc[{op, portNum}] + numLocks) %
                             op.size(); // update to next objFifo elem
        return;
      }

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
      builder.create<UseLockOp>(builder.getUnknownLoc(), lock, lockAction,
                                numLocks);
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
    for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>())
      if (auto objFifo = regOp.getObjectFifo();
          regOp.getTile() == tile && objFifo == parent)
        for (auto extBuff : regOp.getExternalBuffers())
          addExternalBuffer(child, extBuff.getDefiningOp<ExternalBufferOp>());
  }

  /// Function used to replace uses of split objectFifos.
  void replaceSplitFifo(ObjectFifoCreateOp originalOp, ObjectFifoCreateOp newOp,
                        TileOp tile) {
    auto original =
        originalOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    auto newSymbol =
        newOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    for (auto user : tile->getUsers())
      if (isa<CoreOp>(user))
        if (auto res =
                SymbolTable::replaceAllSymbolUses(original, newSymbol, user);
            res.failed())
          llvm_unreachable("unreachable");
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
    if (tile.getDefiningOp<TileOp>().isShimTile())
      for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>()) {
        if (regOp.getTile() == tile)
          return regOp.getExternalBuffers().size();
      }

    int maxAcquire = 0;
    for (auto coreOp : device.getOps<CoreOp>())
      if (coreOp.getTile() == tile)
        coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
          if (auto createOp = acqOp.getObjectFifo(); createOp == objFifo)
            if (acqOp.acqNumber() > maxAcquire)
              maxAcquire = acqOp.acqNumber();
        });

    if (maxAcquire > 0) {
      if (maxAcquire == 1 && objFifo.size() == 1)
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
                                      int channelIndex, bool plio,
                                      std::optional<PacketInfoAttr> packet) {
    PacketInfoAttr packetInfo = nullptr;
    if (packet)
      packetInfo = *packet;
    builder.create<ShimDMAAllocationOp>(builder.getUnknownLoc(), obj_fifo,
                                        DMAChannelDirAttr::get(ctx, channelDir),
                                        builder.getI64IntegerAttr(channelIndex),
                                        builder.getI64IntegerAttr(colIndex),
                                        builder.getBoolAttr(plio), packetInfo);
  }

  /// Function used to verify that an objectfifo is present in at most one
  /// ObjectFifoLinkOp.
  void verifyObjectFifoLinks(DeviceOp &device) {
    DenseSet<ObjectFifoCreateOp> objectfifoset;
    for (ObjectFifoLinkOp link : device.getOps<ObjectFifoLinkOp>()) {
      for (ObjectFifoCreateOp inOf : link.getInputObjectFifos()) {
        if (objectfifoset.count(inOf))
          inOf.emitOpError("objectfifo cannot be in more than one "
                           "ObjectFifoLinkOp");
        objectfifoset.insert(inOf);
      }
      for (ObjectFifoCreateOp outOf : link.getOutputObjectFifos()) {
        if (objectfifoset.count(outOf))
          outOf.emitOpError("objectfifo cannot be in more than one "
                            "ObjectFifoLinkOp");
        objectfifoset.insert(outOf);
      }
    }
  }

  /// Account for already used packet IDs and return next available ID.
  int getStartPacketID(DeviceOp &device) {
    int packetID = 0;
    for (PacketFlowOp packetflow : device.getOps<PacketFlowOp>()) {
      if (packetflow.getID() > packetID) {
        // compute next available ID
        packetID = packetflow.getID() + 1;
      }
    }
    return packetID;
  }

  /// Helper function to assign DMA channel indices for FIFOs based on
  /// cross-tile conditions
  void assignDMAChannelIndices(
      DMAChannelAnalysis &dmaAnalysis,
      const std::map<ObjectFifoCreateOp, bool> &crossTileInfos,
      std::map<ObjectFifoCreateOp, int> &fifo_dma_channel_index,
      bool assignCrossTileOnly) {
    for (auto &[producer, consumers] : splitFifos) {
      // Check if we should process this producer based on cross-tile condition
      bool shouldProcessProducer = assignCrossTileOnly
                                       ? crossTileInfos.at(producer)
                                       : !crossTileInfos.at(producer);

      if (shouldProcessProducer) {
        bool requiresAdjacentTileAccessChannels = crossTileInfos.at(producer);
        int channelIndex = dmaAnalysis.getDMAChannelIndex(
            producer.getProducerTileOp(), DMAChannelDir::MM2S,
            requiresAdjacentTileAccessChannels);
        fifo_dma_channel_index[producer] = channelIndex;
      }

      for (auto consumer : consumers) {
        // Check if we should process this consumer based on cross-tile
        // condition
        bool shouldProcessConsumer = assignCrossTileOnly
                                         ? crossTileInfos.at(consumer)
                                         : !crossTileInfos.at(consumer);

        if (shouldProcessConsumer) {
          bool requiresAdjacentTileAccessChannels = crossTileInfos.at(consumer);
          int channelIndex = dmaAnalysis.getDMAChannelIndex(
              consumer.getProducerTileOp(), DMAChannelDir::S2MM,
              requiresAdjacentTileAccessChannels);
          fifo_dma_channel_index[consumer] = channelIndex;
        }
      }
    }
  }

  void runOnOperation() override {

    DeviceOp device = getOperation();
    LockAnalysis lockAnalysis(device);
    DMAChannelAnalysis dmaAnalysis(device);
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    auto ctx = device->getContext();
    auto producerWireType = WireBundle::DMA;
    auto consumerWireType = WireBundle::DMA;
    std::set<TileOp>
        objectFifoTiles; // track cores to check for loops during unrolling

    verifyObjectFifoLinks(device);

    //===------------------------------------------------------------------===//
    // Split objectFifos into a consumer end and producer end if needed
    //===------------------------------------------------------------------===//
    // We are going to create additional createObjectFifoOps, so get a copy of
    // all "original" ones before the loop to avoid looping over newly created
    // ones.
    std::vector<ObjectFifoCreateOp> createFifoOps;
    auto range = device.getOps<ObjectFifoCreateOp>();
    createFifoOps.insert(createFifoOps.end(), range.begin(), range.end());
    for (auto createOp : createFifoOps) {
      std::vector<ObjectFifoCreateOp> splitConsumerFifos;
      int consumerIndex = 0;
      int consumerDepth = createOp.size();
      ArrayRef<BDDimLayoutArrayAttr> consumerDims =
          createOp.getDimensionsFromStreamPerConsumer();

      // Only FIFOs using DMA are split into two ends;
      // skip in shared memory case
      if (int share_direction = 0; !requiresDMAs(createOp, share_direction)) {
        continue;
      }

      for (auto consumerTile : createOp.getConsumerTiles()) {
        auto consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());

        if (isa<ArrayAttr>(createOp.getElemNumber())) {
          // +1 to account for 1st depth (producer)
          consumerDepth = createOp.size(consumerIndex + 1);
        } else {
          consumerDepth = findObjectFifoSize(device, consumerTileOp, createOp);
        }

        builder.setInsertionPointAfter(createOp);
        auto datatype = llvm::cast<AIEObjectFifoType>(createOp.getElemType());
        auto consumerObjFifoSize =
            builder.getIntegerAttr(builder.getI32Type(), consumerDepth);
        // rename and replace split objectFifo
        std::string consumerFifoName;
        if (createOp.getConsumerTiles().size() > 1) {
          consumerFifoName = createOp.name().str() + "_" +
                             std::to_string(consumerIndex) + "_cons";
        } else {
          consumerFifoName = createOp.name().str() + "_cons";
        }
        BDDimLayoutArrayAttr emptyDims =
            BDDimLayoutArrayAttr::get(builder.getContext(), {});
        BDDimLayoutArrayAttr singletonFromStreamDims =
            BDDimLayoutArrayAttr::get(
                builder.getContext(),
                ArrayRef<BDDimLayoutAttr>{consumerDims[consumerIndex]});
        BDDimLayoutArrayArrayAttr fromStreamDims =
            BDDimLayoutArrayArrayAttr::get(builder.getContext(),
                                           singletonFromStreamDims);

        ObjectFifoCreateOp consumerFifo = createObjectFifo(
            builder, datatype, consumerFifoName, consumerTile, consumerTile,
            consumerObjFifoSize, emptyDims, fromStreamDims);
        if (createOp.getDisableSynchronization())
          consumerFifo.setDisableSynchronization(true);
        replaceSplitFifo(createOp, consumerFifo, consumerTileOp);

        // identify external buffers that were registered to the consumer fifo
        if (consumerTile.getDefiningOp<TileOp>().isShimTile())
          detectExternalBuffers(device, createOp, consumerFifo, consumerTile);

        // record that this objectFifo was split; it will require DMA config
        splitConsumerFifos.push_back(consumerFifo);

        // update the linkOp if the split objFifo was originally its start point
        if (auto linkOp = getOptionalLinkOp(createOp))
          for (ObjectFifoCreateOp fifoIn : linkOp->getInputObjectFifos())
            if (fifoIn.name() == createOp.name() &&
                consumerTile == *linkOp->getOptionalSharedTile())
              if (failed(SymbolTable::replaceAllSymbolUses(
                      createOp, consumerFifo.name(), linkOp->getOperation())))
                llvm::report_fatal_error("unable to update all symbol uses");

        consumerIndex++;
      }

      if (!splitConsumerFifos.empty()) {
        splitFifos.emplace_back(createOp, splitConsumerFifos);
      }
    }

    //===------------------------------------------------------------------===//
    // - Create objectFifo buffers and locks.
    // - Populate a list of tiles containing objectFifos for later processing of
    //   the acquires/releases (uses of the FIFO).
    // - Global release counter tracker to keep track of the objectFifo state
    //===------------------------------------------------------------------===//
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {

      int share_direction = 0;
      bool shared = !requiresDMAs(createOp, share_direction);

      // add all tiles that contain an objectFifo to objectFifoTiles for later
      // loop unrolling pass
      objectFifoTiles.insert(createOp.getProducerTileOp());
      for (auto consumerTile : createOp.getConsumerTiles()) {
        auto consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());
        objectFifoTiles.insert(consumerTileOp);
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
        if (isa<ArrayAttr>(createOp.getElemNumber()))
          createOp.setElemNumberAttr(
              builder.getI32IntegerAttr(createOp.size()));
        else {
          if (!createOp.getInitValues().has_value()) {

            int prodMaxAcquire = findObjectFifoSize(
                device, createOp.getProducerTileOp(), createOp);
            createOp.setElemNumberAttr(
                builder.getI32IntegerAttr(prodMaxAcquire));
          }
        }
        createObjectFifoElements(builder, lockAnalysis, createOp,
                                 share_direction);
      }
    }

    //===------------------------------------------------------------------===//
    // Create flows and tile DMAs
    //===------------------------------------------------------------------===//
    // Only the objectFifos we split above require DMA communication; the others
    // rely on shared memory and share the same buffers.

    // analyze cross-tile buffer allocations and print results
    auto crossTileInfos = analyzeCrossTileFIFOBuffers();

    // maps ends of split FIFO to DMA channels
    std::map<ObjectFifoCreateOp, int> fifo_dma_channel_index;

    // assign channel indices for FIFOs with cross-tile issues first
    assignDMAChannelIndices(dmaAnalysis, crossTileInfos, fifo_dma_channel_index,
                            true);
    // then assign channel indices for FIFOs without cross-tile issues
    assignDMAChannelIndices(dmaAnalysis, crossTileInfos, fifo_dma_channel_index,
                            false);

    int packetID = getStartPacketID(device);
    for (auto &[producer, consumers] : splitFifos) {
      int producerChanIndex = fifo_dma_channel_index[producer];
      if (producerChanIndex == -1)
        producer.getProducerTileOp().emitOpError(
            "number of output DMA channel exceeded!");
      DMAChannel producerChan = {DMAChannelDir::MM2S, producerChanIndex};
      std::optional<PacketInfoAttr> bdPacket = {};
      if (clPacketSwObjectFifos) {
        if (packetID > 31)
          device.emitOpError("max number of packet IDs reached");
        bdPacket = {
            AIE::PacketInfoAttr::get(ctx, /*pkt_type*/ 0, /*pkt_id*/ packetID)};
        packetID++;
      }
      createDMA(device, builder, producer, producerChan.direction,
                producerChan.channel, 0, producer.getDimensionsToStreamAttr(),
                producer.getPadDimensionsAttr(), bdPacket);
      // generate objectFifo allocation info
      builder.setInsertionPoint(device.getBody()->getTerminator());

      if (producer.getProducerTileOp().isShimTile())
        createObjectFifoAllocationInfo(
            builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
            producer.getProducerTileOp().colIndex(), producerChan.direction,
            producerChan.channel, producer.getPlio(), bdPacket);

      PacketFlowOp packetflow;
      if (clPacketSwObjectFifos) {
        // create packet flow
        builder.setInsertionPointAfter(producer);
        packetflow = builder.create<PacketFlowOp>(
            builder.getUnknownLoc(),
            builder.getIntegerAttr(builder.getI8Type(), bdPacket->getPktId()),
            nullptr, nullptr);
        {
          OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(
              &packetflow.getRegion().emplaceBlock());
          builder.create<EndOp>(builder.getUnknownLoc());
        }
      }

      for (auto consumer : consumers) {
        int consumerChanIndex = fifo_dma_channel_index[consumer];
        if (consumerChanIndex == -1)
          consumer.getProducerTileOp().emitOpError(
              "number of input DMA channel exceeded!");
        DMAChannel consumerChan = {DMAChannelDir::S2MM, consumerChanIndex};

        // If we have PLIO then figure out the direction and make that a PLIO
        if (producer.getPlio()) {
          producerWireType = producer.getProducerTileOp().isShimTile()
                                 ? WireBundle::PLIO
                                 : WireBundle::DMA;
          consumerWireType = consumer.getProducerTileOp().isShimTile()
                                 ? WireBundle::PLIO
                                 : WireBundle::DMA;
        } else {
          producerWireType = WireBundle::DMA;
          consumerWireType = WireBundle::DMA;
        }
        if (clPacketSwObjectFifos) {
          builder.setInsertionPointToStart(&packetflow.getPorts().front());
          builder.create<PacketDestOp>(builder.getUnknownLoc(),
                                       consumer.getProducerTile(),
                                       WireBundle::DMA, consumerChan.channel);
        }

        BDDimLayoutArrayAttr consumerDims =
            consumer.getDimensionsFromStreamPerConsumer()[0];
        createDMA(device, builder, consumer, consumerChan.direction,
                  consumerChan.channel, 1, consumerDims, nullptr, {});
        // generate objectFifo allocation info
        builder.setInsertionPoint(device.getBody()->getTerminator());

        if (consumer.getProducerTileOp().isShimTile())
          createObjectFifoAllocationInfo(
              builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
              consumer.getProducerTileOp().colIndex(), consumerChan.direction,
              consumerChan.channel, producer.getPlio(), {});

        if (!clPacketSwObjectFifos) {
          // create flow
          builder.setInsertionPointAfter(producer);
          builder.create<FlowOp>(builder.getUnknownLoc(),
                                 producer.getProducerTile(), producerWireType,
                                 producerChan.channel,
                                 consumer.getProducerTile(), consumerWireType,
                                 consumerChan.channel);
        }
      }

      if (clPacketSwObjectFifos) {
        builder.setInsertionPointToStart(&packetflow.getPorts().front());
        builder.create<PacketSourceOp>(builder.getUnknownLoc(),
                                       producer.getProducerTile(),
                                       WireBundle::DMA, producerChan.channel);
      }
    }

    //===------------------------------------------------------------------===//
    // Statically unroll for loops or use dynamic objectFifos
    //===------------------------------------------------------------------===//
    if (clDynamicObjectFifos) {
      if (failed(dynamicGlobalObjectFifos(device, builder, objectFifoTiles)))
        signalPassFailure();
    } else {
      std::set<TileOp> dynamicTiles;
      std::set<TileOp> unrollTiles;
      for (auto c : device.getOps<CoreOp>()) {
        TileOp t = c.getTileOp();
        if (objectFifoTiles.count(t) > 0) {
          if (c.getDynamicObjfifoLowering().has_value()) {
            if (c.getDynamicObjfifoLowering().value())
              dynamicTiles.insert(t);
            else
              unrollTiles.insert(t);
          } else {
            unrollTiles.insert(t);
          }
        }
      }
      if (failed(dynamicGlobalObjectFifos(device, builder, dynamicTiles)))
        signalPassFailure();
      if (failed(unrollForLoops(device, builder, unrollTiles)))
        signalPassFailure();
    }

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
        auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
        auto core = releaseOp->getParentOfType<CoreOp>();

        if (auto linkOp = getOptionalLinkOp(op)) {
          if (core.getTile() == *linkOp->getOptionalSharedTile()) {
            releaseOp->emitOpError("currently cannot access objectFifo used in "
                                   "ObjectFifoLinkOp");
            return;
          }
        }

        // update index of next element to release for this objectFifo
        updateAndReturnIndex(relPerFifo, {op, portNum});

        // release locks
        int numLocks = releaseOp.relNumber();
        // account for repetition
        if (op.getRepeatCount().has_value())
          numLocks *= op.getRepeatCount().value();
        createUseLocks(builder, op, port, relPerFifo, numLocks,
                       LockAction::Release);

        // register release op
        if (releaseOps.find({op, portNum}) != releaseOps.end()) {
          releaseOps[{op, portNum}].push_back(releaseOp);
        } else {
          std::vector release = {releaseOp};
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
        auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
        auto core = acquireOp->getParentOfType<CoreOp>();

        auto linkOp = getOptionalLinkOp(op);
        if (linkOp) {
          if (core.getTile() == *linkOp->getOptionalSharedTile()) {
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
        // !!! operations may not be in the same block !!!
        int numRel = 0;
        for (std::vector<ObjectFifoReleaseOp>::iterator relOp =
                 releaseOps[{op, portNum}].begin();
             relOp != releaseOps[{op, portNum}].end();) {
          bool erased = false;
          Operation *acqBlockDefOp = acquireOp.getOperation();
          do {
            Operation *relBlockDefOp = (*relOp).getOperation();
            do {
              if (acqBlockDefOp->getBlock() == relBlockDefOp->getBlock()) {
                if (relBlockDefOp->isBeforeInBlock(acqBlockDefOp)) {
                  numRel += (*relOp).relNumber();
                  relOp = releaseOps[{op, portNum}].erase(relOp);
                  // to ensure that we do not account
                  // the ReleaseOps again later,
                  // after the subview is created
                  erased = true;
                }
              }
            } while ((relBlockDefOp = relBlockDefOp->getParentOp()) &&
                     !isa<DeviceOp>(relBlockDefOp) && !erased);
          } while ((acqBlockDefOp = acqBlockDefOp->getParentOp()) &&
                   !isa<DeviceOp>(acqBlockDefOp) && !erased);
          if (!erased)
            ++relOp;
        }

        // track indices of elements to acquire
        std::vector<int> acquiredIndices;
        if (!acquiresPerFifo[{op, portNum}].empty()) {
          // take into account what has already been acquired by previous
          // AcquireOp in program order
          acquiredIndices = acquiresPerFifo[{op, portNum}];
          // take into account what has been released in-between
          if (static_cast<size_t>(numRel) > acquiredIndices.size()) {
            acquireOp->emitOpError("cannot release more elements than are "
                                   "already acquired");
            return;
          }
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

        // account for repetition
        if (op.getRepeatCount().has_value())
          numCreate *= op.getRepeatCount().value();

        auto dev = op->getParentOfType<DeviceOp>();
        if (auto &targetArch = dev.getTargetModel();
            targetArch.getTargetArch() == AIEArch::AIE1)
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
        subviewRefs.reserve(acquiredIndices.size());
        for (auto index : acquiredIndices)
          subviewRefs.push_back(&buffersPerFifo[target][index]);

        subviews[acquireOp] = subviewRefs;
        acquiresPerFifo[{op, portNum}] = acquiredIndices;
      });

      //===----------------------------------------------------------------===//
      // Replace subview.access ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoSubviewAccessOp accessOp) {
        auto acqOp = accessOp.getSubview().getDefiningOp<ObjectFifoAcquireOp>();
        if (ObjectFifoCreateOp op = acqOp.getObjectFifo()) {
          if (auto linkOp = getOptionalLinkOp(op); linkOp.has_value()) {
            if (!linkOp->isDistribute() && !linkOp->isJoin()) {
              for (auto consumerTile : op.getConsumerTiles()) {
                if (auto consumerTileOp =
                        dyn_cast<TileOp>(consumerTile.getDefiningOp())) {
                  int share_dir_value = 0;
                  bool sharing = isSharedMemory(
                      op.getProducerTileOp(), consumerTileOp, &share_dir_value);
                  if (!sharing)
                    accessOp->emitOpError(
                        "currently cannot access objectFifo used in "
                        "ObjectFifoLinkOp if the tiles don't share memory");
                }
              }
            } else
              accessOp->emitOpError(
                  "currently cannot access objectFifo used in "
                  "ObjectFifoLinkOp if it is a distribute or join link");
          }
        }
        accessOp.getOutput().replaceAllUsesWith(
            subviews[acqOp][accessOp.getIndex()]->getBuffer());
      });
    }
    // make global symbols to replace the to be erased ObjectFifoCreateOps
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      builder.setInsertionPointToStart(device.getBody());
      auto sym_name = createOp.getName();
      createOp->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr("__erase_" + sym_name));
      auto memrefType = llvm::cast<AIEObjectFifoType>(createOp.getElemType())
                            .getElementType();
      builder.create<memref::GlobalOp>(builder.getUnknownLoc(), sym_name,
                                       builder.getStringAttr("public"),
                                       memrefType, nullptr, false, nullptr);
    }

    //===------------------------------------------------------------------===//
    // Remove old ops
    //===------------------------------------------------------------------===//
    SetVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<ObjectFifoCreateOp, ObjectFifoLinkOp,
              ObjectFifoRegisterExternalBuffersOp, ObjectFifoAcquireOp,
              ObjectFifoSubviewAccessOp, ObjectFifoReleaseOp,
              ObjectFifoAllocateOp>(op))
        opsToErase.insert(op);
    });
    SmallVector<Operation *> sorted{opsToErase.begin(), opsToErase.end()};
    computeTopologicalSorting(sorted);
    for (auto *op : llvm::reverse(sorted))
      op->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEObjectFifoStatefulTransformPass() {
  return std::make_unique<AIEObjectFifoStatefulTransformPass>();
}
