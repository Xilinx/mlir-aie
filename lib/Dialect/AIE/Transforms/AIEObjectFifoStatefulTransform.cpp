//===- AIEObjectFifoStatefulTransform.cpp ----------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "mlir/Transforms/Mem2Reg.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <numeric>
#include <set>

#include <iostream>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOSTATEFULTRANSFORM
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-objectFifo-stateful-transform"

#define LOOP_VAR_DEPENDENCY (-2)

// Marker for `memref.alloca`s emitted by this pass for bookkeeping only (number
// of locks held, current buffer index). We use memrefs for these bookkeeping
// values because it enables easier threading through loop/control-flow
// structures. A `mem2reg` pass at the end converts them back to SSA values;
// this marker ensures that we convert _all_ allocas back to SSA values but
// touch _no_ allocas that were not emitted by us.
static constexpr llvm::StringLiteral kBookkeepingSlotAttrName =
    "aie.objectfifo.bookkeeping_slot";

//===----------------------------------------------------------------------===//
// DMA Channel Analysis
//===----------------------------------------------------------------------===//
class DMAChannelAnalysis {
  DenseMap<std::tuple<Value, DMAChannelDir, int>, int> channelsPerTile;
  DenseMap<std::tuple<Value, DMAChannelDir, int>, int> aieStreamsPerTile;

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
    for (auto flowOp : device.getOps<FlowOp>()) {
      if (flowOp.getSourceBundle() == WireBundle::Core)
        aieStreamsPerTile[{flowOp.getSource(), DMAChannelDir::MM2S,
                           flowOp.getSourceChannel()}] = 1;
      if (flowOp.getDestBundle() == WireBundle::Core)
        aieStreamsPerTile[{flowOp.getDest(), DMAChannelDir::S2MM,
                           flowOp.getDestChannel()}] = 1;
    }
    // Scan ShimDMAAllocationOps so that channels already claimed (e.g. by
    // the control packet overlay) are marked used in channelsPerTile and are
    // therefore skipped by getDMAChannelIndex when it auto-assigns channels
    // for objectFIFO lowering.
    for (auto allocOp : device.getOps<ShimDMAAllocationOp>()) {
      auto tile = allocOp.getTileOp();
      if (!tile)
        continue;
      channelsPerTile[{tile.getResult(), allocOp.getChannelDir(),
                       (int)allocOp.getChannelIndex()}] = 1;
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

  /// Reserve a user-pinned DMA channel for (tileOp, dir). Returns the channel
  /// on success; returns -1 if the channel is out of range for the tile or is
  /// already in use (the caller emits a diagnostic). Reserving up-front ensures
  /// first-free auto-assignment never steals a pinned channel.
  int reservePinnedChannel(TileOp tileOp, DMAChannelDir dir, int channel) {
    int maxChannelNum = (dir == DMAChannelDir::MM2S)
                            ? tileOp.getNumSourceConnections(WireBundle::DMA)
                            : tileOp.getNumDestConnections(WireBundle::DMA);
    if (channel < 0 || channel >= maxChannelNum)
      return -1;
    if (channelsPerTile[{tileOp.getResult(), dir, channel}] != 0)
      return -1;
    channelsPerTile[{tileOp.getResult(), dir, channel}] = 1;
    return channel;
  }

  /// Given a tile and DMAChannel, adds entry to aieStreamsPerTile or
  /// throws an error if the stream is already used.
  void checkAIEStreamIndex(TileOp tileOp, DMAChannel chan) {
    if (aieStreamsPerTile.find({tileOp.getResult(), chan.direction,
                                chan.channel}) == aieStreamsPerTile.end()) {
      aieStreamsPerTile[{tileOp.getResult(), chan.direction, chan.channel}] = 1;
    } else {
      if (chan.direction == DMAChannelDir::MM2S)
        tileOp.emitOpError("number of output Core channels exceeded!");
      else
        tileOp.emitOpError("number of input Core channels exceeded!");
    }
  }
};

//===----------------------------------------------------------------------===//
// Create objectFifos Pass
//===----------------------------------------------------------------------===//

/// Struct to hold per-device state for the objectFifo transformation.
/// This is passed to helper functions to avoid member variable pollution
/// between different device operations.
struct ObjectFifoState {
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
                        // part of a Link, not because they didn't have a shared
                        // memory module
  DenseMap<Operation *, DenseMap<std::pair<ObjectFifoCreateOp, int>, Value>>
      counterSlotsPerCore; // core -> (fifo, port) -> bookkeeping counter;
                           // the counter is used for both the runtime buffer
                           // index_switch and (on binary-lock architectures)
                           // the runtime lock index_switch
};

struct AIEObjectFifoStatefulTransformPass
    : xilinx::AIE::impl::AIEObjectFifoStatefulTransformBase<
          AIEObjectFifoStatefulTransformPass> {

  /// Function that returns true if two tiles in the AIE array share a memory
  /// module. share_direction is equal to:
  ///   * 2 if the memory modules on both tiles can be shared,
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

    if (leftShared && rightShared)
      *share_direction = 2;
    else if (leftShared)
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
  bool requiresDMAs(ObjectFifoCreateOp createOp, int &share_direction,
                    ObjectFifoState &state) {
    bool hasSharedMemory = false;
    bool atLeastOneConsumerWantsTransform = false;
    bool isUsedInLinkOp = false;

    if (createOp.getVia_DMA())
      return true;

    if (createOp.getRepeatCount().has_value())
      return true;

    if (createOp.getAieStream())
      return true;

    if (createOp.getConsumerElemType().has_value())
      return true;

    if (createOp.getConsumerTiles().size() == 1 &&
        createOp.getDimensionsToStream().empty()) {

      // Test for shared memory
      for (auto consumerTile : createOp.getConsumerTiles()) {
        if (auto consumerTileOp =
                dyn_cast<TileOp>(consumerTile.getDefiningOp())) {
          if (std::count(state.splitBecauseLink.begin(),
                         state.splitBecauseLink.end(), createOp))
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
            state.splitBecauseLink.push_back(createOp);
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
            if ((prodShareDir == -1 || prodShareDir == 2) &&
                (consShareDir == -1 || consShareDir == 2))
              isUsedInLinkOp = false;
            else
              state.splitBecauseLink.push_back(createOp);
          }
        } else {
          state.splitBecauseLink.push_back(createOp);
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
  createObjectFifo(OpBuilder &builder, Location loc, AIEObjectFifoType datatype,
                   std::string name, Value prodTile, Value consTile,
                   Attribute depth, BDDimLayoutArrayAttr dimensionsToStream,
                   BDDimLayoutArrayArrayAttr dimensionsFromStreamPerConsumer) {
    auto ofName = builder.getStringAttr(name);
    auto fifo = ObjectFifoCreateOp::create(
        builder, loc, ofName, prodTile, consTile, depth, datatype,
        dimensionsToStream, dimensionsFromStreamPerConsumer);
    return fifo;
  }

  /// Function used to create objectFifo locks based on target architecture.
  /// Called by createObjectFifoElements(). Locks are created without a lock ID;
  /// AIEAssignLockIDs assigns concrete IDs later in the pipeline.
  std::vector<LockOp>
  createObjectFifoLocks(OpBuilder &builder, ObjectFifoCreateOp op, int numElem,
                        int joinDistribFactor, TileOp creation_tile,
                        int repeatCount, ObjectFifoState &state) {
    std::vector<LockOp> locks;
    if (op.getDisableSynchronization())
      return locks;
    // Static-init no-link producer cycled via iter_count: source side needs
    // no sync; skip allocation to free the lock IDs.
    if (op.getInitValues().has_value() && op.getIterCount().has_value() &&
        op.getIterCount().value() > 1 && !getOptionalLinkOp(op).has_value() &&
        static_cast<int>(op.getInitValues().value().size()) == numElem)
      return locks;
    auto dev = op->getParentOfType<DeviceOp>();
    auto &target = dev.getTargetModel();
    // if shimTile external buffers are collected from input code
    // create as many locks as there are external buffers
    if (creation_tile.isShimTile()) {
      numElem = 0;
      if (!state.externalBuffersPerFifo[op].empty())
        numElem = state.externalBuffersPerFifo[op].size();
    }
    Location ofLoc = op.getLoc();
    if (target.getTargetArch() == AIEArch::AIE1) {
      for (int i = 0; i < numElem; i++) {
        // create corresponding aie1 locks
        int initValue = op.getInitValues().has_value() ? 1 : 0;
        auto lock = LockOp::create(builder, ofLoc, creation_tile, initValue);
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
        int prodLockValue = (numElem - initValues) * repeatCount;
        auto prodLock =
            LockOp::create(builder, ofLoc, creation_tile, prodLockValue);
        prodLock.getOperation()->setAttr(
            SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name().str() + "_prod_lock_" +
                                  std::to_string(i)));
        locks.push_back(prodLock);

        int consLockValue = initValues * repeatCount;
        auto consLock =
            LockOp::create(builder, ofLoc, creation_tile, consLockValue);
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
  std::map<ObjectFifoCreateOp, bool>
  analyzeCrossTileFIFOBuffers(ObjectFifoState &state) {
    std::map<ObjectFifoCreateOp, bool> crossTileMap;

    for (size_t i = 0; i < state.splitFifos.size(); i++) {
      auto &[producerFifo, consumerFifos] = state.splitFifos[i];

      // Analyze producer buffers
      bool producerHasCrossTile = false;

      ObjectFifoCreateOp target = producerFifo;
      auto linkOp = getOptionalLinkOp(producerFifo);

      if (linkOp &&
          state.objFifoLinks.find(*linkOp) != state.objFifoLinks.end()) {
        target = state.objFifoLinks[*linkOp]; // Use the linked target FIFO
      }

      if (state.buffersPerFifo.find(target) != state.buffersPerFifo.end()) {
        // For each FIFO (producer and consumer):
        auto &producerBuffers = state.buffersPerFifo[target];
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
        if (linkOp &&
            state.objFifoLinks.find(*linkOp) != state.objFifoLinks.end()) {
          target = state.objFifoLinks[*linkOp]; // Use the linked target FIFO
        }

        if (state.buffersPerFifo.find(target) != state.buffersPerFifo.end()) {
          // For each FIFO (producer and consumer):
          auto &consumerBuffers = state.buffersPerFifo[target];
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
    auto newTile = TileOp::create(builder, hostTile.getLoc(), col, row);

    builder.restoreInsertionPoint(savedInsertionPoint);

    return newTile;
  }

  /// Function used to create objectFifo elements and their locks.
  /// It maps the input objectFifo to associated buffers and locks.
  void createObjectFifoElements(OpBuilder &builder, ObjectFifoCreateOp op,
                                int share_direction, ObjectFifoState &state) {
    if (!op.size())
      return;

    if (op.getAieStream())
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
      if (state.objFifoLinks.find(*linkOp) != state.objFifoLinks.end())
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
            // When output has padDimensions, MemTile buffer should use
            // input (smaller) size — padding is applied on-the-fly by DMA
            bool outHasPadding = fifoOut.getPadDimensions().has_value();
            if (outHasPadding) {
              if (op.name() != fifoIn.name())
                return;
            } else {
              if (fifoOut.name() != op.name())
                return;
            }
          }
        }
      }
    }

    TileOp creation_tile;
    auto consumerTileOp =
        cast<TileOp>(op.getConsumerTiles()[0].getDefiningOp());
    if (share_direction != 1)
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
      if ((prodShareDir == -1 || prodShareDir == 2) &&
          (consShareDir == -1 || consShareDir == 2))
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
        int currentUsedMemory = calculateCurrentUsedMemory(
            creation_tile, state.buffersPerFifo, buffers);

        // Check if current tile can hold the new buffer or not
        TileOp current_buf_allocation_tile =
            creation_tile; // used to keep track of the tile where the buffer is
                           // allocated
        if (creation_tile.isMemTile()) {
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
              if (isSharedMemory(creation_tile, leftTile, &share_direction) &&
                  (share_direction == 1 || share_direction == 2)) {
                neighborTiles.push_back(leftTile);
              }
            }

            // Check tile to the right
            if (currentCol < (targetModel.columns() - 1)) {
              TileOp rightTile = findOrCreateTile(builder, dev, creation_tile,
                                                  currentCol + 1, currentRow);
              int share_direction = 0;
              if (isSharedMemory(creation_tile, rightTile, &share_direction) &&
                  (share_direction == 1 || share_direction == 2)) {
                neighborTiles.push_back(rightTile);
              }
            }

            // Try neighbor with more remaining capacity first to avoid
            // blocking adjacent MemTiles that also need spill space.
            if (!neighborTiles.empty()) {
              llvm::stable_sort(neighborTiles, [&](TileOp a, TileOp b) {
                return calculateCurrentUsedMemory(a, state.buffersPerFifo,
                                                  buffers) <
                       calculateCurrentUsedMemory(b, state.buffersPerFifo,
                                                  buffers);
              });
              for (auto &tile : neighborTiles) {
                int neighborUsedMemory = calculateCurrentUsedMemory(
                    tile, state.buffersPerFifo, buffers);
                if (static_cast<int>(neighborUsedMemory + totalSizeBytes) <=
                    maxDataMemorySize) {
                  // Allocate buffer on neighbor tile, change creation_tile to
                  // be this neighbour tile
                  current_buf_allocation_tile = tile;
                  break;
                }
              }
            }
          }
        }
        auto buff = BufferOp::create(
            builder, op.getLoc(), elemType, current_buf_allocation_tile,
            builder.getStringAttr(op.name().str() + "_buff_" +
                                  std::to_string(of_elem_index)),
            /*address*/ nullptr, initValues,
            /*mem_bank*/ nullptr, /*aligned*/ nullptr);
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
      state.objFifoLinks[*linkOp] = op;
    }
    std::vector<LockOp> locks = createObjectFifoLocks(
        builder, op, numElem, joinDistribFactor, creation_tile, repeatCount,
        state);
    state.buffersPerFifo[op] = buffers;
    state.locksPerFifo[op] = locks;
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
  void createBd(OpBuilder &builder, Location loc, LockOp acqLock, int acqMode,
                LockAction acqLockAction, LockOp relLock, int relMode,
                MyOp buff, int offset, int len, Block *succ,
                BDDimLayoutArrayAttr dims, BDPadLayoutArrayAttr padDimensions,
                std::optional<PacketInfoAttr> bdPacket) {
    if (acqLock)
      UseLockOp::create(builder, loc, acqLock, acqLockAction, acqMode);
    if (bdPacket) {
      DMABDPACKETOp::create(builder, loc, bdPacket->getPktType(),
                            bdPacket->getPktId());
    }
    if (!dims.getValue().empty() && padDimensions) {
      DMABDOp::create(builder, loc, buff, offset, len, dims, padDimensions);
    } else if (!dims.getValue().empty()) {
      DMABDOp::create(builder, loc, buff, offset, len, dims);
    } else {
      DMABDOp::create(builder, loc, buff, offset, len);
    }
    if (acqLock)
      UseLockOp::create(builder, loc, relLock, LockAction::Release, relMode);
    NextBDOp::create(builder, loc, succ);
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
                     ObjectFifoState &state, bool distribOrJoin = false) {
    LockOp acqLock;
    LockOp relLock;
    int acqMode = 1;
    int relMode = 1;
    auto acqLockAction = LockAction::Acquire;
    // Static-init producer cycled via iter_count > 1 with no upstream link:
    // skip source-side locks. The BD chain restarts via the channel's
    // task_count, but the per-BD lock state never gets replenished (no
    // upstream S2MM refills the buffers) so the chain would deadlock on
    // the second pass. Back-pressure to the downstream consumer is handled
    // by the DMA stream's flow control; source-side locking is unnecessary
    // for correctness in this configuration.
    bool isCycledStaticInitProducer =
        channelDir == DMAChannelDir::MM2S && op.getInitValues().has_value() &&
        op.getIterCount().has_value() && op.getIterCount().value() > 1 &&
        !getOptionalLinkOp(op).has_value();
    if (state.locksPerFifo[op].size() > 0 && !isCycledStaticInitProducer) {
      auto dev = op->getParentOfType<DeviceOp>();
      if (auto &target = dev.getTargetModel();
          target.getTargetArch() == AIEArch::AIE1) {
        acqMode = lockMode == 0 ? 1 : 0;
        relMode = lockMode == 0 ? 0 : 1;
        acqLock = state.locksPerFifo[op][lockIndex];
        relLock = state.locksPerFifo[op][lockIndex];
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
                      ? state.locksPerFifo[op][prodLockIndex]
                      : state.locksPerFifo[op][consLockIndex];
        relLock = channelDir == DMAChannelDir::S2MM
                      ? state.locksPerFifo[op][consLockIndex]
                      : state.locksPerFifo[op][prodLockIndex];
      }
    }
    createBd(builder, op.getLoc(), acqLock, acqMode, acqLockAction, relLock,
             relMode, buff, offset, len, succ, dims, padDimensions, bdPacket);
  }

  /// Function that either calls createAIETileDMA(), createShimDMA() or
  /// createMemTileDMA() based on op tile row value.
  void createDMA(DeviceOp &device, OpBuilder &builder, ObjectFifoCreateOp op,
                 DMAChannelDir channelDir, int channelIndex, int lockMode,
                 BDDimLayoutArrayAttr dims, BDPadLayoutArrayAttr pad_dims,
                 std::optional<PacketInfoAttr> bdPacket,
                 ObjectFifoState &state) {
    if (op.getProducerTileOp().isShimTile()) {
      createShimDMA(device, builder, op, channelDir, channelIndex, lockMode,
                    dims, bdPacket, state);
    } else if (op.getProducerTileOp().isMemTile()) {
      BDPadLayoutArrayAttr padDims = nullptr;
      if (channelDir == DMAChannelDir::MM2S && pad_dims)
        padDims = pad_dims;
      createMemTileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       dims, padDims, bdPacket, state);
    } else {
      createAIETileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       dims, bdPacket, state);
    }
  }

  /// Function used to create a MemOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createAIETileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode,
                        BDDimLayoutArrayAttr dims,
                        std::optional<PacketInfoAttr> bdPacket,
                        ObjectFifoState &state) {
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
      if (state.objFifoLinks.find(linkOp.value()) != state.objFifoLinks.end()) {
        target = state.objFifoLinks[linkOp.value()];
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
      auto newMemOp = MemOp::create(builder, op.getLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newMemOp.getRegion().emplaceBlock());
        EndOp::create(builder, op.getLoc());
      }
      producerMem = newMemOp.getOperation();
    }
    Block *endBlock = findEndOpBlock(producerMem->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    // With a single buffer, the DMA hardware repeat_count avoids
    // duplicating identical BDs.
    bool useHwRepeat = (repeatCount > 1 && numBlocks == 1);
    int dmaRepeatCount = useHwRepeat ? repeatCount - 1 : 0;
    int bdRepeatCount = useHwRepeat ? 1 : repeatCount;
    builder.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(builder, op.getLoc(), channelDir, channelIndex,
                       dmaRepeatCount, bdBlock, endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t elemIndex = 0;
    size_t totalBlocks = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (elemIndex >= state.buffersPerFifo[target].size())
        break;
      for (int r = 0; r < bdRepeatCount; r++) {
        if (totalBlocks == numBlocks * bdRepeatCount - 1)
          succ = bdBlock;
        else
          succ = builder.createBlock(endBlock);

        builder.setInsertionPointToStart(curr);
        createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                                state.buffersPerFifo[target][elemIndex],
                                /*offset*/ 0, len, channelDir, elemIndex, succ,
                                dims, nullptr, bdPacket, state);
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
                     std::optional<PacketInfoAttr> bdPacket,
                     ObjectFifoState &state) {
    size_t numBlocks = state.externalBuffersPerFifo[op].size();
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
      auto newDMAOp = ShimDMAOp::create(builder, op.getLoc(),
                                        builder.getIndexType(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        EndOp::create(builder, op.getLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    DMAStartOp::create(builder, op.getLoc(), channelDir, channelIndex,
                       /*repeatCout*/ 0, bdBlock, endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t elemIndex = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (elemIndex >= state.externalBuffersPerFifo[op].size())
        break;
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      MemRefType buffer = state.externalBuffersPerFifo[op][elemIndex].getType();
      int len = buffer.getNumElements();
      builder.setInsertionPointToStart(curr);
      createBdBlock<ExternalBufferOp>(
          builder, op, lockMode, acqNum, relNum,
          state.externalBuffersPerFifo[op][elemIndex],
          /*offset*/ 0, len, channelDir, elemIndex, succ, dims, nullptr,
          bdPacket, state);
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
                        std::optional<PacketInfoAttr> bdPacket,
                        ObjectFifoState &state) {
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

    // check for BD chain repeat count
    auto bdChainIterCount = op.getIterCount();

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
      if (state.objFifoLinks.find(*linkOp) != state.objFifoLinks.end()) {
        target = state.objFifoLinks[*linkOp];
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
            int targetLen = targetElemType.getNumElements();
            // Only override when target is larger or equal. When target
            // is smaller (padDimensions size mismatch after buffer
            // ownership change), op's own element count is correct.
            if (targetLen >= lenOut)
              lenOut = targetLen;
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
      auto newDMAOp = MemTileDMAOp::create(builder, op.getLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        EndOp::create(builder, op.getLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);

    int taskCount = 0;
    bool isBdChainMode = false;
    if (bdChainIterCount.has_value()) {
      taskCount = bdChainIterCount.value() - 1;
      isBdChainMode = true;
    }
    // With a single buffer and no join/distribute, the DMA hardware
    // repeat_count avoids duplicating identical BDs.
    bool useHwRepeat = (repeatCount > 1 && numBlocks == 1 &&
                        joinDistribFactor == 1 && !isBdChainMode);
    if (useHwRepeat)
      taskCount = repeatCount - 1;
    int bdRepeatFactor = useHwRepeat ? 1 : repeatCount;
    DMAStartOp::create(builder, op.getLoc(), channelDir, channelIndex,
                       taskCount, bdBlock, endBlock);
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
      if (elemIndex >= state.buffersPerFifo[target].size())
        break;
      for (int r = 0; r < bdRepeatFactor * joinDistribFactor; r++) {
        if (totalBlocks == numBlocks * bdRepeatFactor * joinDistribFactor - 1) {
          // If iter_count attribute is set (BD chain mode), create a
          // dedicated terminating block
          if (isBdChainMode) {
            succ = builder.createBlock(endBlock);
            // Create a separate terminating block with aie.end for this
            // specific DMA channel
            builder.setInsertionPointToStart(succ);
            EndOp::create(builder, op.getLoc());
          } else {
            succ = bdBlock;
          }
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
                                state.buffersPerFifo[target][elemIndex], offset,
                                lenOut, channelDir, lockIndex, succ, dims,
                                padDimensions, bdPacket, state, distribOrJoin);
        curr = succ;
        totalBlocks++;
      }
      elemIndex++;
    }
  }

  // Function that generates the IR to update runtime state of objectfifo
  // accesses. Called by dynamicGlobalObjectFifos().
  void updateGlobalNextIndex(OpBuilder &builder, ObjectFifoReleaseOp relOp,
                             Value counterSlot, arith::ConstantOp size) {
    builder.setInsertionPointAfter(relOp);
    Value oldCounter = memref::LoadOp::create(builder, relOp.getLoc(),
                                              counterSlot, ValueRange{});
    Value val =
        arith::ConstantOp::create(builder, oldCounter.getLoc(),
                                  builder.getI32IntegerAttr(relOp.getSize()));
    Value sum = arith::AddIOp::create(builder, val.getLoc(), oldCounter, val);
    Value isGreaterEqual = arith::CmpIOp::create(
        builder, sum.getLoc(), arith::CmpIPredicate::sge, sum, size);
    Value newCounter = arith::SelectOp::create(
        builder, sum.getLoc(), isGreaterEqual,
        arith::SubIOp::create(builder, sum.getLoc(), sum, size), sum);
    memref::StoreOp::create(builder, size.getLoc(), newCounter, counterSlot,
                            ValueRange{});
  }

  // Function that generates the IR for objectfifo accesses to be handled at
  // runtime.
  LogicalResult dynamicGlobalObjectFifos(DeviceOp &device, OpBuilder &builder,
                                         std::set<TileOp> objectFifoTiles,
                                         ObjectFifoState &state) {
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
        // Keep a map with the ConstantOps for the sizes per OF and a map with
        // the per-OF scalar counter slots.
        std::map<std::pair<ObjectFifoCreateOp, ObjectFifoPort>,
                 arith::ConstantOp>
            constantSizes;
        std::map<std::pair<ObjectFifoCreateOp, ObjectFifoPort>, Value>
            counterSlots;

        builder.setInsertionPointToStart(&(coreOp.getBody().front()));
        Value initVal = arith::ConstantOp::create(builder, coreOp.getLoc(),
                                                  builder.getI32IntegerAttr(0));
        // Each objectFifo/port gets its own scalar (rank-0) counter slot that
        // tracks the index of the next object to access. Each slot is a
        // promotable memref.alloca (rather than a multi-element buffer or an
        // aie.buffer), so that a subsequent -mem2reg threads it through the
        // enclosing scf.for loops as an iter_arg. This makes the per-access
        // index_switch select on a loop-carried SSA value, which lets constant
        // folding resolve the accessed buffer once the loops are unrolled.
        auto scalarTy = MemRefType::get(SmallVector<int64_t>{}, // rank-0
                                        builder.getI32Type());
        coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
          ObjectFifoCreateOp op = acqOp.getObjectFifo();
          ObjectFifoPort port = acqOp.getPort();
          if (fifoSizes.find({op, port}) == fifoSizes.end()) {
            fifoSizes[{op, port}] = op.size();
            auto size =
                arith::ConstantOp::create(builder, initVal.getLoc(),
                                          builder.getI32IntegerAttr(op.size()));
            constantSizes[{op, port}] = size;
            Value slot =
                memref::AllocaOp::create(builder, coreOp.getLoc(), scalarTy);
            slot.getDefiningOp()->setAttr(kBookkeepingSlotAttrName,
                                          builder.getUnitAttr());
            counterSlots[{op, port}] = slot;
            int portNum = port == ObjectFifoPort::Produce ? 0 : 1;
            state.counterSlotsPerCore[coreOp.getOperation()][{op, portNum}] =
                slot;
          }
        });

        // Initialize all counter slots to 0.
        for (auto &i : counterSlots) {
          builder.setInsertionPointAfterValue(i.second);
          memref::StoreOp::create(builder, coreOp.getLoc(), initVal, i.second,
                                  ValueRange{});
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
            updateGlobalNextIndex(builder, relOp,
                                  counterSlots[{createOp, port}],
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
              auto switchIndexAsInteger = memref::LoadOp::create(
                  builder, acqOp.getLoc(), counterSlots[{createOp, port}],
                  ValueRange{});
              auto switchIndex = arith::IndexCastOp::create(
                  builder, acqOp.getLoc(), builder.getIndexType(),
                  switchIndexAsInteger);
              unsigned caseRegionCounts = fifoSizes[{createOp, port}];
              SmallVector<int64_t, 4> caseValues;
              for (int i = 0; i < fifoSizes[{createOp, port}]; ++i) {
                caseValues.push_back(i);
              }
              auto cases =
                  DenseI64ArrayAttr::get(builder.getContext(), caseValues);
              auto switchOp = scf::IndexSwitchOp::create(
                  builder, switchIndex.getLoc(),
                  TypeRange({state.buffersPerFifo[createOp][0].getType()}),
                  switchIndex, cases, caseRegionCounts);
              // Create default case of IndexSwitchOp
              builder.createBlock(&switchOp.getDefaultRegion());
              auto bufferIndex = (accessOp.getIndex()) % createOp.size();
              builder.setInsertionPointToStart(&(switchOp.getDefaultBlock()));
              scf::YieldOp::create(
                  builder, accessOp.getLoc(),
                  state.buffersPerFifo[createOp][bufferIndex].getResult());
              for (int i = 0; i < fifoSizes[{createOp, port}]; ++i) {
                // Create other cases of IndexSwitchOp
                builder.createBlock(&switchOp.getCaseRegions()[i]);
                builder.setInsertionPoint(&switchOp.getCaseBlock(i),
                                          switchOp.getCaseBlock(i).begin());
                int bufferToBeAccesed =
                    (accessOp.getIndex() + i) % fifoSizes[{createOp, port}];
                scf::YieldOp::create(
                    builder, switchOp.getCaseRegions()[i].getLoc(),
                    state.buffersPerFifo[createOp][bufferToBeAccesed]
                        .getResult());
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
                      int numLocks, LockAction lockAction,
                      ObjectFifoState &state) {
    ObjectFifoCreateOp target = op;
    auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
    if (auto linkOp = getOptionalLinkOp(op))
      if (state.objFifoLinks.find(*linkOp) != state.objFifoLinks.end())
        target = state.objFifoLinks[*linkOp];

    auto dev = op->getParentOfType<DeviceOp>();
    if (!dev.getTargetModel().hasProperty(AIETargetModel::UsesSemaphoreLocks)) {

      if (state.locksPerFifo[target].size() == 0) {
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
        UseLockOp::create(builder, op.getLoc(),
                          state.locksPerFifo[target][lockID], lockAction,
                          lockMode);
        acc[{op, portNum}] =
            (lockID + 1) % op.size(); // update to next objFifo elem
      }
    } else {
      if (numLocks == 0)
        return;

      if (state.locksPerFifo[target].size() == 0) {
        acc[{op, portNum}] = (acc[{op, portNum}] + numLocks) %
                             op.size(); // update to next objFifo elem
        return;
      }

      // search for the correct lock based on the port of the acq/rel
      // operation e.g. acq as consumer is the read lock (second)
      LockOp lock;
      if (lockAction == LockAction::AcquireGreaterEqual) {
        if (port == ObjectFifoPort::Produce)
          lock = state.locksPerFifo[target][0];
        else
          lock = state.locksPerFifo[target][1];
      } else {
        if (port == ObjectFifoPort::Produce)
          lock = state.locksPerFifo[target][1];
        else
          lock = state.locksPerFifo[target][0];
      }
      UseLockOp::create(builder, op.getLoc(), lock, lockAction, numLocks);
      acc[{op, portNum}] = (acc[{op, portNum}] + numLocks) %
                           op.size(); // update to next objFifo elem
    }
  }

  /// Emit a UseLockOp whose lock value is a runtime-computed SSA i32.
  void createUseLocksRuntime(OpBuilder &builder, ObjectFifoCreateOp op,
                             ObjectFifoPort port, Value count,
                             LockAction lockAction, ObjectFifoState &state) {
    ObjectFifoCreateOp target = op;
    if (auto linkOp = getOptionalLinkOp(op))
      if (state.objFifoLinks.find(*linkOp) != state.objFifoLinks.end())
        target = state.objFifoLinks[*linkOp];

    if (state.locksPerFifo[target].size() == 0)
      return;

    // Select the correct lock based on the port and action, mirroring the
    // semaphore-lock branch of createUseLocks().
    LockOp lock;
    if (lockAction == LockAction::AcquireGreaterEqual) {
      if (port == ObjectFifoPort::Produce)
        lock = state.locksPerFifo[target][0];
      else
        lock = state.locksPerFifo[target][1];
    } else {
      if (port == ObjectFifoPort::Produce)
        lock = state.locksPerFifo[target][1];
      else
        lock = state.locksPerFifo[target][0];
    }
    UseLockOp::create(builder, op.getLoc(), lock, lockAction, count);
  }

  /// Emit UseLockOps for the dynamic lowering on binary-lock architectures
  /// (AIE1). Unlike semaphore locks, AIE1 has one binary lock per objectFifo
  /// element; the lock to use rotates in lockstep with the buffer. The starting
  /// lock index is only known at runtime (it depends on how many elements have
  /// been released), so each of the `numLocks` locks is selected with an
  /// scf.index_switch keyed on the same rotating counter used for buffer
  /// addressing. Once the enclosing loops are unrolled the counter folds to a
  /// constant and each switch collapses to a single concrete lock.
  /// `baseOffset` is the offset (within the
  /// rotation) of the first lock relative to the counter.
  void createUseLocksDynamicBinary(OpBuilder &builder, ObjectFifoCreateOp op,
                                   ObjectFifoPort port, Value counterSlot,
                                   int baseOffset, int numLocks,
                                   LockAction lockAction,
                                   ObjectFifoState &state) {
    if (numLocks == 0)
      return;
    ObjectFifoCreateOp target = op;
    if (auto linkOp = getOptionalLinkOp(op))
      if (state.objFifoLinks.find(*linkOp) != state.objFifoLinks.end())
        target = state.objFifoLinks[*linkOp];

    auto &locks = state.locksPerFifo[target];
    if (locks.empty())
      return;
    int size = op.size();

    // Binary lock mode, mirroring the non-semaphore branch of createUseLocks().
    int lockMode = 0;
    if ((port == ObjectFifoPort::Produce &&
         lockAction == LockAction::Release) ||
        (port == ObjectFifoPort::Consume && lockAction == LockAction::Acquire))
      lockMode = 1;

    Location loc = op.getLoc();

    // For single-element fifos there is no rotation: use the only lock.
    if (size == 1) {
      for (int i = 0; i < numLocks; i++)
        UseLockOp::create(builder, loc, locks[0], lockAction, lockMode);
      return;
    }

    Value counterI32 =
        memref::LoadOp::create(builder, loc, counterSlot, ValueRange{});
    Value counterIdx = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), counterI32);
    auto lockTy = locks[0].getType();

    SmallVector<int64_t, 4> caseValues;
    for (int c = 0; c < size; c++)
      caseValues.push_back(c);
    auto cases = DenseI64ArrayAttr::get(builder.getContext(), caseValues);

    for (int i = 0; i < numLocks; i++) {
      auto switchOp = scf::IndexSwitchOp::create(
          builder, loc, TypeRange({lockTy}), counterIdx, cases, size);
      // Default case: counter out of [0, size) should not happen; yield the
      // lock at the base offset.
      builder.createBlock(&switchOp.getDefaultRegion());
      builder.setInsertionPointToStart(&(switchOp.getDefaultBlock()));
      scf::YieldOp::create(builder, loc,
                           locks[(baseOffset + i) % size].getResult());
      for (int c = 0; c < size; c++) {
        builder.createBlock(&switchOp.getCaseRegions()[c]);
        builder.setInsertionPointToStart(&switchOp.getCaseBlock(c));
        scf::YieldOp::create(builder, loc,
                             locks[(baseOffset + i + c) % size].getResult());
      }
      builder.setInsertionPointAfter(switchOp);
      UseLockOp::create(builder, loc, switchOp.getResult(0), lockAction,
                        lockMode);
    }
  }

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
  void addExternalBuffer(ObjectFifoCreateOp fifo, ExternalBufferOp buff,
                         ObjectFifoState &state) {
    if (state.externalBuffersPerFifo.find(fifo) ==
        state.externalBuffersPerFifo.end()) {
      std::vector<ExternalBufferOp> buffs;
      state.externalBuffersPerFifo[fifo] = buffs;
    }
    state.externalBuffersPerFifo[fifo].push_back(buff);
  }

  /// Function used to detect all external buffers associated with parent
  /// objectFifo and tile then map them to child objectFifo.
  void detectExternalBuffers(DeviceOp &device, ObjectFifoCreateOp parent,
                             ObjectFifoCreateOp child, Value tile,
                             ObjectFifoState &state) {
    for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>())
      if (auto objFifo = regOp.getObjectFifo();
          regOp.getTile() == tile && objFifo == parent)
        for (auto extBuff : regOp.getExternalBuffers())
          addExternalBuffer(child, extBuff.getDefiningOp<ExternalBufferOp>(),
                            state);
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
  /// shimTile reference assigned by the objectFifo lowering.
  void createObjectFifoAllocationInfo(OpBuilder &builder, MLIRContext *ctx,
                                      ObjectFifoCreateOp &objFifoOp,
                                      TileOp shimTile, DMAChannelDir channelDir,
                                      int channelIndex, bool plio,
                                      std::optional<PacketInfoAttr> packet) {
    PacketInfoAttr packetInfo = nullptr;
    if (packet)
      packetInfo = *packet;
    std::string alloc_name = getShimAllocationName(objFifoOp.getName());
    // SymbolRefAttr::get(ctx, objFifoOp.getName())
    ShimDMAAllocationOp::create(
        builder, objFifoOp.getLoc(), StringAttr::get(ctx, alloc_name),
        shimTile.getResult(), DMAChannelDirAttr::get(ctx, channelDir),
        builder.getI64IntegerAttr(channelIndex), builder.getBoolAttr(plio),
        packetInfo);
  }

  static std::string getShimAllocationName(llvm::StringRef objFifoName) {
    return (objFifoName + "_shim_alloc").str();
  }

  /// Function used to verify that an objectfifo is present in at most one
  /// ObjectFifoLinkOp.
  LogicalResult verifyObjectFifoLinks(DeviceOp &device) {
    DenseSet<ObjectFifoCreateOp> objectfifoset;
    bool hasError = false;
    for (ObjectFifoLinkOp link : device.getOps<ObjectFifoLinkOp>()) {
      for (ObjectFifoCreateOp inOf : link.getInputObjectFifos()) {
        if (objectfifoset.count(inOf)) {
          inOf.emitOpError("objectfifo cannot be in more than one "
                           "ObjectFifoLinkOp");
          hasError = true;
        }
        objectfifoset.insert(inOf);
      }
      for (ObjectFifoCreateOp outOf : link.getOutputObjectFifos()) {
        if (objectfifoset.count(outOf)) {
          outOf.emitOpError("objectfifo cannot be in more than one "
                            "ObjectFifoLinkOp");
          hasError = true;
        }
        objectfifoset.insert(outOf);
      }
    }
    return hasError ? failure() : success();
  }

  /// This pass assumes every objectFifo producer/consumer tile is a placed
  /// `aie.tile`; the rest of the pass reaches them via getProducerTileOp() /
  /// cast<TileOp>(...), which dereferences a null if a tile is still an
  /// unplaced `aie.logical_tile` (or any non-TileOp). Emit a diagnostic and
  /// fail cleanly instead of crashing when a design reaches this pass before
  /// placement (e.g. --aie-place-tiles was not run first).
  LogicalResult verifyObjectFifoTilesArePlaced(DeviceOp &device) {
    auto isPlacedTile = [](Value tile) {
      return isa_and_nonnull<TileOp>(tile.getDefiningOp());
    };
    bool hasError = false;
    for (ObjectFifoCreateOp createOp : device.getOps<ObjectFifoCreateOp>()) {
      if (!isPlacedTile(createOp.getProducerTile())) {
        createOp.emitOpError("producer tile is not a placed aie.tile; run "
                             "--aie-place-tiles before this pass");
        hasError = true;
      }
      for (Value consumerTile : createOp.getConsumerTiles()) {
        if (!isPlacedTile(consumerTile)) {
          createOp.emitOpError("consumer tile is not a placed aie.tile; run "
                               "--aie-place-tiles before this pass");
          hasError = true;
        }
      }
    }
    return hasError ? failure() : success();
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
      bool assignCrossTileOnly, ObjectFifoState &state) {
    for (auto &[producer, consumers] : state.splitFifos) {
      // Check if we should process this producer based on cross-tile condition
      bool shouldProcessProducer = assignCrossTileOnly
                                       ? crossTileInfos.at(producer)
                                       : !crossTileInfos.at(producer);

      if (shouldProcessProducer) {
        // A pinned channel was already reserved in reservePinnedChannels();
        // honor it here instead of first-free assignment.
        if (auto pin = producer.getProdDmaChannel()) {
          fifo_dma_channel_index[producer] = *pin;
        } else {
          bool requiresAdjacentTileAccessChannels = crossTileInfos.at(producer);
          int channelIndex = dmaAnalysis.getDMAChannelIndex(
              producer.getProducerTileOp(), DMAChannelDir::MM2S,
              requiresAdjacentTileAccessChannels);
          fifo_dma_channel_index[producer] = channelIndex;
        }
      }

      for (auto consumer : consumers) {
        // Check if we should process this consumer based on cross-tile
        // condition
        bool shouldProcessConsumer = assignCrossTileOnly
                                         ? crossTileInfos.at(consumer)
                                         : !crossTileInfos.at(consumer);

        if (shouldProcessConsumer) {
          // Post-split each consumer is its own single-endpoint fifo carrying
          // its pin (if any) as prod_dma_channel; honor it here.
          if (auto pin = consumer.getProdDmaChannel()) {
            fifo_dma_channel_index[consumer] = *pin;
          } else {
            bool requiresAdjacentTileAccessChannels =
                crossTileInfos.at(consumer);
            int channelIndex = dmaAnalysis.getDMAChannelIndex(
                consumer.getProducerTileOp(), DMAChannelDir::S2MM,
                requiresAdjacentTileAccessChannels);
            fifo_dma_channel_index[consumer] = channelIndex;
          }
        }
      }
    }
  }

  /// Reserve every user-pinned DMA channel before any first-free assignment,
  /// so auto-assignment cannot steal a pinned channel. Emits a clean diagnostic
  /// on an out-of-range channel, a collision between two pins on the same
  /// (tile, dir), or a pin combined with aie_stream (which bypasses DMA).
  /// Returns failure if any conflict was found.
  LogicalResult reservePinnedChannels(DMAChannelAnalysis &dmaAnalysis,
                                      ObjectFifoState &state) {
    for (auto &[producer, consumers] : state.splitFifos) {
      if (auto pin = producer.getProdDmaChannel()) {
        if (producer.getAieStream())
          return producer.emitOpError(
              "cannot pin a DMA channel on an objectfifo that also uses "
              "aie_stream (stream ports bypass DMA channels)");
        if (dmaAnalysis.reservePinnedChannel(producer.getProducerTileOp(),
                                             DMAChannelDir::MM2S, *pin) < 0)
          return producer.emitOpError("pinned MM2S DMA channel ")
                 << *pin << " is out of range or already in use on this tile";
      }
      for (auto consumer : consumers) {
        if (auto pin = consumer.getProdDmaChannel()) {
          if (consumer.getAieStream())
            return consumer.emitOpError(
                "cannot pin a DMA channel on an objectfifo that also uses "
                "aie_stream (stream ports bypass DMA channels)");
          if (dmaAnalysis.reservePinnedChannel(consumer.getProducerTileOp(),
                                               DMAChannelDir::S2MM, *pin) < 0)
            return consumer.emitOpError("pinned S2MM DMA channel ")
                   << *pin << " is out of range or already in use on this tile";
        }
      }
    }
    return success();
  }

  // Promote the rank-0 bookkeeping counters (memref.alloca marked with
  // kBookkeepingSlotAttrName) that the dynamic lowering emits into loop-carried
  // SSA values, using mem2reg restricted to exactly those slots. Nothing else
  // in the IR is touched. It is a hard requirement that none of these slots
  // survive: they exist solely to let the standard mem2reg machinery thread the
  // counters through the surrounding control flow. If any cannot be promoted,
  // the pass fails rather than silently leaving memory-based bookkeeping.
  LogicalResult promoteBookkeepingSlots(DeviceOp device) {
    SmallVector<PromotableAllocationOpInterface> allocators;
    WalkResult collect = device.walk([&](memref::AllocaOp allocaOp) {
      if (!allocaOp->hasAttr(kBookkeepingSlotAttrName))
        return WalkResult::advance();
      auto promotable =
          dyn_cast<PromotableAllocationOpInterface>(allocaOp.getOperation());
      if (!promotable) {
        allocaOp.emitOpError()
            << "objectFifo bookkeeping slot does not implement the promotable "
               "allocation interface and cannot be lowered to SSA";
        return WalkResult::interrupt();
      }
      allocators.push_back(promotable);
      return WalkResult::advance();
    });
    if (collect.wasInterrupted())
      return failure();

    if (allocators.empty())
      return success();

    DataLayout dataLayout = DataLayout::closest(device);
    DominanceInfo dominance(device);
    OpBuilder promoteBuilder(device.getContext());
    (void)tryToPromoteMemorySlots(allocators, promoteBuilder, dataLayout,
                                  dominance);

    // Hardening: the lowering guarantees SSA-only bookkeeping. If any of our
    // slots could not be threaded through the surrounding control flow, fail
    // loudly instead of emitting memory-based bookkeeping.
    WalkResult leftover = device.walk([&](memref::AllocaOp allocaOp) {
      if (allocaOp->hasAttr(kBookkeepingSlotAttrName)) {
        allocaOp.emitOpError()
            << "objectFifo bookkeeping slot could not be promoted to SSA "
               "(mem2reg left it in place); the objectFifo lowering requires "
               "all bookkeeping counters to become loop-carried SSA values";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (leftover.wasInterrupted())
      return failure();
    return success();
  }

  void runOnOperation() override {

    DeviceOp device = getOperation();

    // Create local state for this device operation - ensures thread and
    // multi-device safety
    ObjectFifoState state;

    DMAChannelAnalysis dmaAnalysis(device);
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    auto ctx = device->getContext();
    auto producerWireType = WireBundle::DMA;
    auto consumerWireType = WireBundle::DMA;
    std::set<TileOp>
        objectFifoTiles; // track cores to check for loops during unrolling

    if (failed(verifyObjectFifoLinks(device)))
      return signalPassFailure();

    if (failed(verifyObjectFifoTilesArePlaced(device)))
      return signalPassFailure();

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
      if (int share_direction = 0;
          !requiresDMAs(createOp, share_direction, state)) {
        continue;
      }

      for (auto consumerTile : createOp.getConsumerTiles()) {
        auto consumerTileOp = cast<TileOp>(consumerTile.getDefiningOp());

        if (isa<ArrayAttr>(createOp.getElemNumber())) {
          // +1 to account for 1st depth (producer)
          consumerDepth = createOp.size(consumerIndex + 1);
        } else {
          consumerDepth = findObjectFifoSize(device, consumerTileOp, createOp);
        }

        builder.setInsertionPointAfter(createOp);
        // Use consumer element type if specified (asymmetric transfer)
        auto datatype = llvm::cast<AIEObjectFifoType>(
            createOp.getConsumerElemTypeOrDefault());
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

        ObjectFifoCreateOp consumerFifo =
            createObjectFifo(builder, createOp.getLoc(), datatype,
                             consumerFifoName, consumerTile, consumerTile,
                             consumerObjFifoSize, emptyDims, fromStreamDims);
        if (createOp.getDisableSynchronization())
          consumerFifo.setDisableSynchronization(true);
        // Propagate iter_count attribute from the original createOp
        // to the new consumerFifo
        if (auto bdChainIterCount = createOp.getIterCount()) {
          consumerFifo.setIterCountAttr(
              builder.getI32IntegerAttr(*bdChainIterCount));
        }
        replaceSplitFifo(createOp, consumerFifo, consumerTileOp);
        if (createOp.getAieStream()) {
          int streamEnd = createOp.getAieStream().value();
          if (streamEnd > 0) {
            consumerFifo->setAttr("aie_stream",
                                  builder.getI32IntegerAttr(streamEnd));
            consumerFifo->setAttr(
                "aie_stream_port",
                builder.getI32IntegerAttr(createOp.getAieStreamPort().value()));
          }
          if (streamEnd == 1) {
            createOp->removeAttr("aie_stream");
            createOp->removeAttr("aie_stream_port");
          }
        }
        // Propagate this consumer's pinned DMA channel (if any) from the
        // original op's cons_dma_channels array onto the fresh consumer fifo.
        // Post-split each consumer is its own single-endpoint fifo, so it
        // carries the pin as prod_dma_channel (the split consumer fifo's tile
        // is the one that owns the S2MM channel).
        if (auto consChans = createOp.getConsDmaChannels()) {
          ArrayRef<int> chans = *consChans;
          if (consumerIndex < (int)chans.size() && chans[consumerIndex] >= 0)
            consumerFifo.setProdDmaChannelAttr(
                builder.getI32IntegerAttr(chans[consumerIndex]));
        }

        // identify external buffers that were registered to the consumer fifo
        if (consumerTile.getDefiningOp<TileOp>().isShimTile())
          detectExternalBuffers(device, createOp, consumerFifo, consumerTile,
                                state);

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
        state.splitFifos.emplace_back(createOp, splitConsumerFifos);
      }
    }

    //===------------------------------------------------------------------===//
    // - Create objectFifo buffers and locks.
    // - Populate a list of tiles containing objectFifos for later processing of
    //   the acquires/releases (uses of the FIFO).
    // - Global release counter tracker to keep track of the objectFifo state
    //===------------------------------------------------------------------===//
    // Process MemTile ObjectFifos largest-first so large buffers get
    // priority for home placement and spill targets are chosen before
    // smaller fifos consume neighbor capacity.
    SmallVector<ObjectFifoCreateOp> sortedCreateOps(
        device.getOps<ObjectFifoCreateOp>());
    if (!sortedCreateOps.empty()) {
      DataLayout dataLayout = DataLayout::closest(sortedCreateOps[0]);
      // Sort only among MemTile-producer fifos by buffer size descending.
      // Non-MemTile fifos keep their IR-order positions undisturbed.
      auto getBufSize = [&](ObjectFifoCreateOp op) -> int64_t {
        auto fifoType = llvm::cast<AIEObjectFifoType>(op.getElemType());
        auto elemType = llvm::cast<MemRefType>(fifoType.getElementType());
        int64_t bits = dataLayout.getTypeSizeInBits(elemType.getElementType());
        return elemType.getNumElements() * bits / 8;
      };
      SmallVector<size_t> memTileSlots;
      SmallVector<ObjectFifoCreateOp> memTileFifos;
      for (size_t i = 0; i < sortedCreateOps.size(); i++) {
        auto prodTile = dyn_cast<TileOp>(
            sortedCreateOps[i].getProducerTile().getDefiningOp());
        if (prodTile && prodTile.isMemTile()) {
          memTileSlots.push_back(i);
          memTileFifos.push_back(sortedCreateOps[i]);
        }
      }
      llvm::stable_sort(memTileFifos,
                        [&](ObjectFifoCreateOp a, ObjectFifoCreateOp b) {
                          return getBufSize(a) > getBufSize(b);
                        });
      for (size_t i = 0; i < memTileSlots.size(); i++)
        sortedCreateOps[memTileSlots[i]] = memTileFifos[i];
    }
    for (auto createOp : sortedCreateOps) {

      int share_direction = 0;
      bool shared = !requiresDMAs(createOp, share_direction, state);

      // add all tiles that contain an objectFifo to objectFifoTiles for later
      // loop unrolling pass
      objectFifoTiles.insert(createOp.getProducerTileOp());
      for (auto consumerTile : createOp.getConsumerTiles()) {
        auto consumerTileOp = cast<TileOp>(consumerTile.getDefiningOp());
        objectFifoTiles.insert(consumerTileOp);
      }

      // identify external buffers that were registered to
      // the producer objectFifo
      if (createOp.getProducerTileOp().isShimTile())
        detectExternalBuffers(device, createOp, createOp,
                              createOp.getProducerTile(), state);

      // if split, the necessary size for producer fifo might change
      if (shared) {
        createObjectFifoElements(builder, createOp, share_direction, state);
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
        createObjectFifoElements(builder, createOp, share_direction, state);
      }
    }

    //===------------------------------------------------------------------===//
    // Create flows and tile DMAs
    //===------------------------------------------------------------------===//
    // Only the objectFifos we split above require DMA communication; the others
    // rely on shared memory and share the same buffers.

    // analyze cross-tile buffer allocations and print results
    auto crossTileInfos = analyzeCrossTileFIFOBuffers(state);

    // maps ends of split FIFO to DMA channels
    std::map<ObjectFifoCreateOp, int> fifo_dma_channel_index;

    // Reserve user-pinned channels before any first-free assignment so
    // auto-assignment cannot steal a pinned channel.
    if (failed(reservePinnedChannels(dmaAnalysis, state)))
      return signalPassFailure();

    // assign channel indices for FIFOs with cross-tile issues first
    assignDMAChannelIndices(dmaAnalysis, crossTileInfos, fifo_dma_channel_index,
                            true, state);
    // then assign channel indices for FIFOs without cross-tile issues
    assignDMAChannelIndices(dmaAnalysis, crossTileInfos, fifo_dma_channel_index,
                            false, state);

    int packetID = getStartPacketID(device);
    for (auto &[producer, consumers] : state.splitFifos) {
      int producerChanIndex = -1;
      DMAChannel producerChan;
      PacketFlowOp packetflow;
      if (producer.getAieStream()) {
        int prodStreamEnd = producer.getAieStream().value();
        if (prodStreamEnd == 0 || prodStreamEnd == 2) {
          producerChanIndex = producer.getAieStreamPort().value();
          producerChan = {DMAChannelDir::MM2S, producerChanIndex};
          dmaAnalysis.checkAIEStreamIndex(producer.getProducerTileOp(),
                                          producerChan);
        }
      } else {
        producerChanIndex = fifo_dma_channel_index[producer];
        if (producerChanIndex == -1) {
          producer.getProducerTileOp().emitOpError(
              "number of output DMA channel exceeded!");
          return signalPassFailure();
        }
        producerChan = {DMAChannelDir::MM2S, producerChanIndex};
        std::optional<PacketInfoAttr> bdPacket = {};
        if (clPacketSwObjectFifos) {
          if (packetID > 31) {
            device.emitOpError("max number of packet IDs reached");
            return signalPassFailure();
          }
          bdPacket = {AIE::PacketInfoAttr::get(ctx, /*pkt_type*/ 0,
                                               /*pkt_id*/ packetID)};
          packetID++;
        }
        createDMA(device, builder, producer, producerChan.direction,
                  producerChan.channel, 0, producer.getDimensionsToStreamAttr(),
                  producer.getPadDimensionsAttr(), bdPacket, state);

        // generate objectFifo allocation info
        builder.setInsertionPoint(device.getBody()->getTerminator());
        if (producer.getProducerTileOp().isShimTile())
          createObjectFifoAllocationInfo(
              builder, ctx, producer, producer.getProducerTileOp(),
              producerChan.direction, producerChan.channel, producer.getPlio(),
              bdPacket);

        if (clPacketSwObjectFifos) {
          // create packet flow
          builder.setInsertionPointAfter(producer);
          packetflow = builder.create<PacketFlowOp>(
              producer.getLoc(),
              builder.getIntegerAttr(builder.getI8Type(), bdPacket->getPktId()),
              nullptr, nullptr);
          {
            OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToStart(
                &packetflow.getRegion().emplaceBlock());
            builder.create<EndOp>(producer.getLoc());
          }
        }
      }

      for (auto consumer : consumers) {
        // if not aie stream, create consumer tile DMA
        int consumerChanIndex = -1;
        DMAChannel consumerChan;
        if (consumer.getAieStream()) {
          int consStreamEnd = consumer.getAieStream().value();
          if (consStreamEnd == 1 || consStreamEnd == 2) {
            consumerChanIndex = consumer.getAieStreamPort().value();
            consumerChan = {DMAChannelDir::S2MM, consumerChanIndex};
            dmaAnalysis.checkAIEStreamIndex(consumer.getProducerTileOp(),
                                            consumerChan);
          }
        } else {
          consumerChanIndex = fifo_dma_channel_index[consumer];
          if (consumerChanIndex == -1) {
            consumer.getProducerTileOp().emitOpError(
                "number of input DMA channel exceeded!");
            return signalPassFailure();
          }
          consumerChan = {DMAChannelDir::S2MM, consumerChanIndex};
          BDDimLayoutArrayAttr consumerDims =
              consumer.getDimensionsFromStreamPerConsumer()[0];
          createDMA(device, builder, consumer, consumerChan.direction,
                    consumerChan.channel, 1, consumerDims, nullptr, {}, state);

          // generate objectFifo allocation info
          builder.setInsertionPoint(device.getBody()->getTerminator());
          if (!consumer.getAieStream()) {
            // generate objectFifo allocation info
            builder.setInsertionPoint(device.getBody()->getTerminator());
            if (consumer.getProducerTileOp().isShimTile())
              createObjectFifoAllocationInfo(
                  builder, ctx, producer, consumer.getProducerTileOp(),
                  consumerChan.direction, consumerChan.channel,
                  producer.getPlio(), {});
          }

          if (clPacketSwObjectFifos) {
            builder.setInsertionPointToStart(&packetflow.getPorts().front());
            builder.create<PacketDestOp>(consumer.getLoc(),
                                         consumer.getProducerTile(),
                                         WireBundle::DMA, consumerChan.channel);
          }
        }

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
          if (producer.getAieStream()) {
            int prodStreamEnd = producer.getAieStream().value();
            if (prodStreamEnd == 0 || prodStreamEnd == 2)
              producerWireType = WireBundle::Core;
          }
          if (consumer.getAieStream()) {
            int consumerStreamEnd = consumer.getAieStream().value();
            if (consumerStreamEnd == 1 || consumerStreamEnd == 2)
              consumerWireType = WireBundle::Core;
          }
        }

        if (!clPacketSwObjectFifos) {
          // create flow
          builder.setInsertionPointAfter(producer);
          FlowOp::create(builder, producer.getLoc(), producer.getProducerTile(),
                         producerWireType, producerChan.channel,
                         consumer.getProducerTile(), consumerWireType,
                         consumerChan.channel);
        }
      }

      if (clPacketSwObjectFifos) {
        builder.setInsertionPointToStart(&packetflow.getPorts().front());
        PacketSourceOp::create(builder, producer.getLoc(),
                               producer.getProducerTile(), WireBundle::DMA,
                               producerChan.channel);
      }
    }

    //===------------------------------------------------------------------===//
    // Validate objectFifo accesses before lowering
    //===------------------------------------------------------------------===//
    // These checks must run before the buffer/lock lowering below: the dynamic
    // lowering assumes local buffers exist, which is not the case for stream
    // ports or shared tiles of a link, so validate up front and bail out with a
    // clean diagnostic instead of crashing in the lowering.
    for (auto coreOp : device.getOps<CoreOp>()) {
      auto validateAccess = [&](Operation *accessOp, ObjectFifoCreateOp op,
                                ObjectFifoPort port,
                                StringRef verb) -> LogicalResult {
        int portNum = port == ObjectFifoPort::Produce ? 0 : 1;
        if (auto linkOp = getOptionalLinkOp(op)) {
          if (coreOp.getTile() == *linkOp->getOptionalSharedTile())
            return accessOp->emitOpError(
                "currently cannot access objectFifo used in ObjectFifoLinkOp");
        }
        if (op.getAieStream().has_value()) {
          int streamEnd = op.getAieStream().value();
          if (streamEnd == 2 || streamEnd == portNum)
            return accessOp->emitOpError("cannot ")
                   << verb << " objectfifo stream port";
          return failure();
        }
        return success();
      };
      WalkResult vres = coreOp.walk([&](Operation *o) {
        if (auto acq = dyn_cast<ObjectFifoAcquireOp>(o)) {
          if (failed(validateAccess(acq, acq.getObjectFifo(), acq.getPort(),
                                    "acquire from")))
            return WalkResult::interrupt();
        } else if (auto rel = dyn_cast<ObjectFifoReleaseOp>(o)) {
          if (failed(validateAccess(rel, rel.getObjectFifo(), rel.getPort(),
                                    "release from")))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (vres.wasInterrupted())
        return signalPassFailure();
    }

    //===------------------------------------------------------------------===//
    // Annotate objectFifo loops with an unroll hint
    //===------------------------------------------------------------------===//
    // Record, on each scf.for that (transitively) carries an objectFifo access,
    // the factor by which it must be unrolled so that each iteration maps to a
    // fixed rotation of buffers/locks: the least common multiple of the depths
    // of the objectFifos accessed within it. This is captured here, while the
    // acquire/release ops still exist; the separate `aie-objectFifo-unroll`
    // pass consumes the hint after lowering (once those ops have been replaced
    // by the runtime bookkeeping) to drive `scf` loop unrolling. The hint is
    // only emitted when dynamic objectFifos are disabled, since that is the
    // only case where the unroll pass runs (and thus removes the hint
    // afterwards).
    if (!clDynamicObjectFifos) {
      for (auto coreOp : device.getOps<CoreOp>()) {
        coreOp.walk([&](scf::ForOp forOp) {
          int64_t lcm = 1;
          bool hasAccess = false;
          auto addDepth = [&](ObjectFifoCreateOp createOp) {
            hasAccess = true;
            lcm = std::lcm(lcm, static_cast<int64_t>(createOp.size()));
          };
          // Only account for accesses whose innermost enclosing scf.for is this
          // loop; accesses nested inside a child scf.for belong to (and drive
          // the unroll factor of) that child, not this loop. This mirrors the
          // legacy behavior of unrolling each loop by its own rotation period
          // and prevents over-unrolling ancestor loops.
          auto directlyIn = [&](Operation *op) {
            return op->getParentOfType<scf::ForOp>() == forOp;
          };
          forOp.getBody()->walk([&](ObjectFifoAcquireOp a) {
            if (directlyIn(a))
              addDepth(a.getObjectFifo());
          });
          forOp.getBody()->walk([&](ObjectFifoReleaseOp r) {
            if (directlyIn(r))
              addDepth(r.getObjectFifo());
          });
          if (hasAccess)
            forOp->setAttr(kObjectFifoUnrollHintAttrName,
                           builder.getI64IntegerAttr(lcm));
        });
      }
    }

    //===------------------------------------------------------------------===//
    // Statically-decidable over-release verification
    //===------------------------------------------------------------------===//
    // Independent of the lowering strategy: if a loop body releases more
    // elements of an objectFifo than it acquires, the number of held elements
    // underflows as the loop repeats, i.e. it releases more than were ever
    // acquired. This is always an error and is decidable without knowing the
    // trip count. Cases that depend on runtime/arbitrary control flow are not
    // statically decidable and are intentionally left unchecked.
    for (auto coreOp : device.getOps<CoreOp>()) {
      WalkResult vres = coreOp.walk([&](scf::ForOp forOp) {
        // Only account for accesses whose innermost enclosing scf.for is this
        // loop; nested accesses belong to (and are checked as) their own loop.
        auto directlyIn = [&](Operation *op) {
          return op->getParentOfType<scf::ForOp>() == forOp;
        };
        DenseMap<std::pair<ObjectFifoCreateOp, int>, int64_t> acquired;
        DenseMap<std::pair<ObjectFifoCreateOp, int>, int64_t> released;
        DenseMap<std::pair<ObjectFifoCreateOp, int>, ObjectFifoAcquireOp>
            firstAcquire;
        DenseMap<std::pair<ObjectFifoCreateOp, int>, ObjectFifoReleaseOp>
            firstRelease;
        // A (fifo, port) whose acquires/releases are split across loop nesting
        // levels (e.g. acquired in a nested loop, released in this body) cannot
        // be balanced without knowing the nested trip counts, so it is not
        // statically decidable and must be excluded from the check.
        llvm::DenseSet<std::pair<ObjectFifoCreateOp, int>> spansNestedLoop;
        forOp.getBody()->walk([&](ObjectFifoAcquireOp a) {
          auto key =
              std::make_pair(a.getObjectFifo(),
                             a.getPort() == ObjectFifoPort::Produce ? 0 : 1);
          if (!directlyIn(a)) {
            spansNestedLoop.insert(key);
            return;
          }
          acquired[key] += a.acqNumber();
          if (!firstAcquire.count(key))
            firstAcquire[key] = a;
        });
        forOp.getBody()->walk([&](ObjectFifoReleaseOp r) {
          auto key =
              std::make_pair(r.getObjectFifo(),
                             r.getPort() == ObjectFifoPort::Produce ? 0 : 1);
          if (!directlyIn(r)) {
            spansNestedLoop.insert(key);
            return;
          }
          released[key] += r.relNumber();
          if (!firstRelease.count(key))
            firstRelease[key] = r;
        });
        for (auto &entry : released) {
          if (spansNestedLoop.contains(entry.first))
            continue;
          if (entry.second > acquired.lookup(entry.first)) {
            // Attach the diagnostic to the acquire op when present (matching
            // the legacy behavior); otherwise to the offending release op.
            if (auto acq = firstAcquire.lookup(entry.first))
              acq->emitOpError(
                  "cannot release more elements than are already acquired");
            else
              firstRelease.lookup(entry.first)
                  ->emitOpError(
                      "cannot release more elements than are already acquired");
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (vres.wasInterrupted())
        return signalPassFailure();
    }

    //===------------------------------------------------------------------===//
    // Select the lowering strategy for each tile
    //===------------------------------------------------------------------===//
    // objectFifo accesses are lowered dynamically: runtime buffer addressing
    // (an scf.index_switch selecting the rotating buffer) and runtime lock
    // bookkeeping, keeping the loops rolled. Loop unrolling and the subsequent
    // constant folding of this runtime bookkeeping (which reproduces the legacy
    // static, unrolled lowering) is handled by the separate
    // `aie-objectFifo-unroll` pass followed by `-mem2reg`/`-canonicalize`.
    //
    // All objectFifo tiles use the dynamic buffer addressing. Lock bookkeeping
    // differs by architecture: semaphore locks (AIE2+) use a runtime "held"
    // counter with AcquireGreaterEqual/Release by count; binary locks (AIE1)
    // rotate one lock per element, selected at runtime with an index_switch.
    std::set<TileOp> dynamicLoweringTiles;
    {
      std::set<TileOp> dynamicTiles;
      for (auto c : device.getOps<CoreOp>()) {
        TileOp t = c.getTileOp();
        if (objectFifoTiles.count(t) > 0)
          dynamicTiles.insert(t);
      }
      dynamicLoweringTiles = dynamicTiles;
      if (failed(
              dynamicGlobalObjectFifos(device, builder, dynamicTiles, state)))
        return signalPassFailure();
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
      // Dynamic lock bookkeeping setup
      //===----------------------------------------------------------------===//
      // For tiles using the dynamic objectFifo lowering, the number of locks
      // to acquire is computed at runtime from a per-(fifo,port) "held" counter
      // stored in a local buffer.
      // Runtime lock counts require semaphore locks (AIE2+); on architectures
      // with binary locks (AIE1) the dynamic lowering keeps static lock counts.
      bool isDynamicCore = dynamicLoweringTiles.count(coreOp.getTileOp()) > 0 &&
                           device.getTargetModel().hasProperty(
                               AIETargetModel::UsesSemaphoreLocks);
      // Binary-lock (AIE1) tiles also use the dynamic buffer addressing, but
      // their locks rotate one-per-element and are selected with a runtime
      // index_switch keyed on the shared per-(fifo, port) counter rather than a
      // "held" count.
      bool isDynamicBinaryCore =
          dynamicLoweringTiles.count(coreOp.getTileOp()) > 0 &&
          !device.getTargetModel().hasProperty(
              AIETargetModel::UsesSemaphoreLocks);
      DenseMap<std::pair<ObjectFifoCreateOp, int>, Value> &counterSlots =
          state.counterSlotsPerCore[coreOp.getOperation()];
      // Per-(fifo, port) scalar "held" counter slots. Each is a promotable
      // rank-0 memref.alloca, so -mem2reg threads it through the enclosing
      // scf.for loops as an iter_arg and the computed lock counts fold to
      // constants once the loops are unrolled.
      DenseMap<std::pair<ObjectFifoCreateOp, int>, Value> heldSlots;
      if (isDynamicCore) {
        // Ordered list of (fifo, port) keys, so that the counter slots are
        // created deterministically (a plain DenseMap iteration order is not
        // stable).
        SmallVector<std::pair<ObjectFifoCreateOp, int>> slotOrder;
        auto assignSlot = [&](ObjectFifoCreateOp fifo, ObjectFifoPort p) {
          int pn = p == ObjectFifoPort::Produce ? 0 : 1;
          if (!heldSlots.count({fifo, pn})) {
            heldSlots[{fifo, pn}] = Value();
            slotOrder.push_back({fifo, pn});
          }
        };
        coreOp.walk([&](ObjectFifoAcquireOp a) {
          assignSlot(a.getObjectFifo(), a.getPort());
        });
        coreOp.walk([&](ObjectFifoReleaseOp r) {
          assignSlot(r.getObjectFifo(), r.getPort());
        });
        if (!slotOrder.empty()) {
          builder.setInsertionPointToStart(&(coreOp.getBody().front()));
          auto heldTy = MemRefType::get(SmallVector<int64_t>{}, // rank-0
                                        builder.getI32Type());
          Value zero = arith::ConstantOp::create(builder, coreOp.getLoc(),
                                                 builder.getI32IntegerAttr(0));
          for (auto &key : slotOrder) {
            Value slot =
                memref::AllocaOp::create(builder, coreOp.getLoc(), heldTy);
            slot.getDefiningOp()->setAttr(kBookkeepingSlotAttrName,
                                          builder.getUnitAttr());
            // Initialize the counter right after the alloca so it dominates all
            // of its stores.
            memref::StoreOp::create(builder, coreOp.getLoc(), zero, slot,
                                    ValueRange{});
            heldSlots[key] = slot;
          }
        }
      }

      //===----------------------------------------------------------------===//
      // Replace objectFifo.release ops
      //===----------------------------------------------------------------===//
      WalkResult res = coreOp.walk([&](ObjectFifoReleaseOp releaseOp) {
        builder.setInsertionPointAfter(releaseOp);
        ObjectFifoCreateOp op = releaseOp.getObjectFifo();
        auto port = releaseOp.getPort();
        auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
        auto core = releaseOp->getParentOfType<CoreOp>();

        if (auto linkOp = getOptionalLinkOp(op)) {
          if (core.getTile() == *linkOp->getOptionalSharedTile()) {
            releaseOp->emitOpError("currently cannot access objectFifo used in "
                                   "ObjectFifoLinkOp");
            return WalkResult::interrupt();
            ;
          }
        }

        if (op.getAieStream().has_value()) {
          int streamEnd = op.getAieStream().value();
          if (streamEnd == 2 || streamEnd == portNum)
            releaseOp->emitOpError("cannot release from objectfifo stream "
                                   "port");
          return WalkResult::interrupt();
        }

        // update index of next element to release for this objectFifo
        updateAndReturnIndex(relPerFifo, {op, portNum});

        // release locks
        int numLocks = releaseOp.relNumber();
        // account for repetition
        if (op.getRepeatCount().has_value())
          numLocks *= op.getRepeatCount().value();
        if (isDynamicBinaryCore && counterSlots.count({op, portNum})) {
          // Binary locks (AIE1): the released locks are those of the oldest
          // held elements, at rotation offset 0 relative to the counter (which
          // still points at the next element to release; the counter is
          // advanced afterwards by the buffer-addressing bookkeeping).
          createUseLocksDynamicBinary(
              builder, op, port, counterSlots[{op, portNum}], /*baseOffset=*/0,
              numLocks, LockAction::Release, state);
        } else {
          createUseLocks(builder, op, port, relPerFifo, numLocks,
                         LockAction::Release, state);
        }

        // For dynamic tiles, decrement the runtime "held" counter by the
        // number of released elements.
        if (isDynamicCore && heldSlots.count({op, portNum})) {
          Value slot = heldSlots[{op, portNum}];
          Value held = memref::LoadOp::create(builder, releaseOp.getLoc(), slot,
                                              ValueRange{});
          Value m = arith::ConstantOp::create(
              builder, releaseOp.getLoc(), builder.getI32IntegerAttr(numLocks));
          Value newHeld =
              arith::SubIOp::create(builder, releaseOp.getLoc(), held, m);
          memref::StoreOp::create(builder, releaseOp.getLoc(), newHeld, slot,
                                  ValueRange{});
        }

        // register release op
        if (releaseOps.find({op, portNum}) != releaseOps.end()) {
          releaseOps[{op, portNum}].push_back(releaseOp);
        } else {
          std::vector release = {releaseOp};
          releaseOps[{op, portNum}] = release;
        }
        return WalkResult::advance();
      });
      if (res.wasInterrupted())
        return signalPassFailure();

      //===----------------------------------------------------------------===//
      // Replace objectFifo.acquire ops
      //===----------------------------------------------------------------===//
      res = coreOp.walk([&](ObjectFifoAcquireOp acquireOp) {
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
            return WalkResult::interrupt();
            ;
          }
        }

        if (op.getAieStream().has_value()) {
          int streamEnd = op.getAieStream().value();
          if (streamEnd == 2 || streamEnd == portNum)
            acquireOp->emitOpError("cannot acquire from objectfifo stream "
                                   "port");
          return WalkResult::interrupt();
        }

        // index of next element to acquire for this objectFifo
        int start = updateAndReturnIndex(
            acqPerFifo, {op, portNum}); // useful for keeping track of which
        // indices are acquired

        // if objFifo was linked with others, find which objFifos
        // elements to use
        ObjectFifoCreateOp target = op;
        if (linkOp)
          if (state.objFifoLinks.find(*linkOp) != state.objFifoLinks.end())
            target = state.objFifoLinks[*linkOp];

        // For dynamic tiles, compute the number of locks to acquire at runtime
        // as max(0, acqNumber - held), using the runtime "held" counter. Buffer
        // addressing for size > 1 fifos is handled separately (via runtime
        // index_switch in dynamicGlobalObjectFifos); the subview references
        // built below are only used for size-1 fifos.
        if (isDynamicCore && heldSlots.count({op, portNum})) {
          builder.setInsertionPointAfter(acquireOp);
          int acqNum = acquireOp.acqNumber();
          int repeat = op.getRepeatCount().value_or(1);
          Value slot = heldSlots[{op, portNum}];
          Value held = memref::LoadOp::create(builder, acquireOp.getLoc(), slot,
                                              ValueRange{});
          Value nVal = arith::ConstantOp::create(
              builder, acquireOp.getLoc(), builder.getI32IntegerAttr(acqNum));
          Value zero = arith::ConstantOp::create(builder, acquireOp.getLoc(),
                                                 builder.getI32IntegerAttr(0));
          Value rawDelta =
              arith::SubIOp::create(builder, acquireOp.getLoc(), nVal, held);
          Value delta = arith::MaxSIOp::create(builder, acquireOp.getLoc(),
                                               rawDelta, zero);
          if (repeat > 1) {
            Value repeatVal = arith::ConstantOp::create(
                builder, acquireOp.getLoc(), builder.getI32IntegerAttr(repeat));
            delta = arith::MulIOp::create(builder, acquireOp.getLoc(), delta,
                                          repeatVal);
          }
          createUseLocksRuntime(builder, op, port, delta,
                                LockAction::AcquireGreaterEqual, state);
          Value newHeld =
              arith::AddIOp::create(builder, acquireOp.getLoc(), held, delta);
          memref::StoreOp::create(builder, acquireOp.getLoc(), newHeld, slot,
                                  ValueRange{});

          // Build the subview buffer references (used by the subview.access
          // replacement below for size-1 fifos).
          std::vector<BufferOp *> subviewRefs;
          subviewRefs.reserve(acqNum);
          for (int i = 0; i < acqNum; i++) {
            subviewRefs.push_back(&state.buffersPerFifo[target][start]);
            start = (start + 1) % op.size();
          }
          subviews[acquireOp] = subviewRefs;
          acqPerFifo[{op, portNum}] = start;
          return WalkResult::advance();
        }

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
            return WalkResult::interrupt();
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
        if (isDynamicBinaryCore && counterSlots.count({op, portNum})) {
          // Binary locks (AIE1): select the rotating lock at runtime. The new
          // locks are those of the elements acquired past the ones already
          // held, i.e. at rotation offset `alreadyAcq` relative to the counter.
          createUseLocksDynamicBinary(
              builder, op, port, counterSlots[{op, portNum}],
              /*baseOffset=*/alreadyAcq, numCreate, LockAction::Acquire, state);
        } else if (auto &targetArch = dev.getTargetModel();
                   targetArch.getTargetArch() == AIEArch::AIE1)
          createUseLocks(builder, op, port, acqPerFifo, numCreate,
                         LockAction::Acquire, state);
        else
          createUseLocks(builder, op, port, acqPerFifo, numCreate,
                         LockAction::AcquireGreaterEqual, state);

        // create subview: buffers that were already acquired + new acquires
        for (int i = 0; i < numCreate; i++) {
          acquiredIndices.push_back(start);
          start = (start + 1) % op.size();
        }
        std::vector<BufferOp *> subviewRefs;
        subviewRefs.reserve(acquiredIndices.size());
        for (auto index : acquiredIndices)
          subviewRefs.push_back(&state.buffersPerFifo[target][index]);

        subviews[acquireOp] = subviewRefs;
        acquiresPerFifo[{op, portNum}] = acquiredIndices;

        return WalkResult::advance();
      });
      if (res.wasInterrupted())
        return signalPassFailure();

      //===----------------------------------------------------------------===//
      // Replace subview.access ops
      //===----------------------------------------------------------------===//
      res = coreOp.walk([&](ObjectFifoSubviewAccessOp accessOp) {
        // Verifier guarantees the defining op is a direct acquire.
        auto acqOp = accessOp.getSubview().getDefiningOp<ObjectFifoAcquireOp>();
        assert(acqOp && "ObjectFifoSubviewAccessOp verifier should reject "
                        "non-direct subview operands");
        if (ObjectFifoCreateOp op = acqOp.getObjectFifo()) {
          if (auto linkOp = getOptionalLinkOp(op); linkOp.has_value()) {
            if (!linkOp->isDistribute() && !linkOp->isJoin()) {
              for (auto consumerTile : op.getConsumerTiles()) {
                if (auto consumerTileOp =
                        dyn_cast<TileOp>(consumerTile.getDefiningOp())) {
                  int share_dir_value = 0;
                  bool sharing = isSharedMemory(
                      op.getProducerTileOp(), consumerTileOp, &share_dir_value);
                  if (!sharing) {
                    accessOp->emitOpError(
                        "currently cannot access objectFifo used in "
                        "ObjectFifoLinkOp if the tiles don't share memory");
                    return WalkResult::interrupt();
                  }
                }
              }
            } else {
              accessOp->emitOpError(
                  "currently cannot access objectFifo used in "
                  "ObjectFifoLinkOp if it is a distribute or join link");
              return WalkResult::interrupt();
            }
          }
        }
        accessOp.getOutput().replaceAllUsesWith(
            subviews[acqOp][accessOp.getIndex()]->getBuffer());
        return WalkResult::advance();
      });
      if (res.wasInterrupted())
        return signalPassFailure();
    }

    //===------------------------------------------------------------------===//
    // Remove old ops
    //===------------------------------------------------------------------===//
    SetVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<ObjectFifoLinkOp, ObjectFifoRegisterExternalBuffersOp,
              ObjectFifoAcquireOp, ObjectFifoSubviewAccessOp,
              ObjectFifoReleaseOp, ObjectFifoAllocateOp>(op))
        opsToErase.insert(op);
    });
    SmallVector<Operation *> sorted{opsToErase.begin(), opsToErase.end()};
    computeTopologicalSorting(sorted);
    for (auto *op : llvm::reverse(sorted))
      op->erase();

    //===------------------------------------------------------------------===//
    // Replace any remaining uses of object fifo symbol with symbol of its shim
    // dma allocation.
    //===------------------------------------------------------------------===//
    opsToErase.clear();
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      std::string shimAllocName = getShimAllocationName(createOp.getName());
      if (failed(SymbolTable::replaceAllSymbolUses(
              createOp.getNameAttr(), builder.getStringAttr(shimAllocName),
              device))) {
        createOp.emitError(
            "failed to replace symbol uses with shim allocation");
        return signalPassFailure();
      }
      opsToErase.insert(createOp);
    }
    for (auto *op : opsToErase) {
      op->erase();
    }

    //===------------------------------------------------------------------===//
    // Promote bookkeeping counters to SSA
    //===------------------------------------------------------------------===//
    // The dynamic lowering above emits the per-(objectFifo, port) buffer-index
    // and lock-held counters as rank-0 memref.alloca slots with surrounding
    // load/store. Promote exactly those slots to loop-carried SSA values now so
    // that the pass never leaves memory-based bookkeeping behind; if any of our
    // slots cannot be promoted, this fails the pass.
    if (failed(promoteBookkeepingSlots(device)))
      return signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEObjectFifoStatefulTransformPass() {
  return std::make_unique<AIEObjectFifoStatefulTransformPass>();
}
