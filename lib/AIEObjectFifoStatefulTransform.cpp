//===- AIEObjectFifoStatefulTransform.cpp --------------------------*- MLIR
//-*-===//
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

#include "aie/AIEDialect.h"
#include "aie/AIETokenAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
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
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit), module(m) {}

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
  ModuleOp &module;
  DenseMap<std::pair<Value, int>, int> locksPerTile;

public:
  LockAnalysis(ModuleOp &m) : module(m) {
    // go over the locks created for each tile and update the index in
    // locksPerTile
    for (auto lockOp : module.getOps<LockOp>()) {
      auto tile = lockOp.getTile();
      auto lockID = lockOp.getLockIDValue();

      locksPerTile[std::make_pair(tile, lockID)] = 1;
    }
  }

  /// Given a tile, returns next usable lockID for that tile.
  int getLockID(Value tileOp) {
    for (unsigned i = 0; i < 16; i++) {
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
// Create objectFifos Pass
//===----------------------------------------------------------------------===//
struct AIEObjectFifoStatefulTransformPass
    : public AIEObjectFifoStatefulTransformBase<
          AIEObjectFifoStatefulTransformPass> {
  DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>>
      buffersPerFifo; // maps each objFifo to its corresponding elements
  DenseMap<ObjectFifoCreateOp, std::vector<LockOp>>
      locksPerFifo; // maps each objFifo to its corresponding locks
  DenseMap<ObjectFifoCreateOp,
           std::pair<ObjectFifoCreateOp, ObjectFifoCreateOp>>
      splitFifos;     // maps each objFifo between non-adjacent tiles to its
                      // corresponding producer and consumer objectFifos
  int buff_index = 0; // used to give objectFifo buffer elements a symbolic name

  /// Function used to create objectFifo elements and their locks.
  /// It maps the input objectFifo to associated buffers and locks.
  void createObjectFifoElements(OpBuilder &builder, LockAnalysis &analysis,
                                ObjectFifoCreateOp op) {
    std::vector<BufferOp> buffers;
    std::vector<LockOp> locks;
    AIEObjectFifoType fifo = op.getType().cast<AIEObjectFifoType>();
    MemRefType elemType = fifo.getElementType().cast<MemRefType>();

    builder.setInsertionPointAfter(op);
    for (int i = 0; i < op.size(); i++) {
      BufferOp buff = builder.create<BufferOp>(
          builder.getUnknownLoc(), elemType, op.getProducerTileOp());
      buff.getOperation()->setAttr(
          "sym_name",
          builder.getStringAttr("buff" + std::to_string(buff_index)));
      buffers.push_back(buff);
      buff_index++;

      int lockID = analysis.getLockID(op.getProducerTileOp());
      assert(lockID >= 0 && "No more locks to allocate!");
      LockOp lock = builder.create<LockOp>(builder.getUnknownLoc(),
                                           op.getProducerTileOp(), lockID);
      locks.push_back(lock);
    }

    buffersPerFifo[op] = buffers;
    locksPerFifo[op] = locks;
  }

  /// Function used to create a Bd block.
  /// If lockMode is 0 we create a consumerDMA (i.e. on producer tile) else a
  /// producerDMA (i.e. on consumer tile).
  void createBdBlock(OpBuilder &builder, int lockMode, BufferOp buff,
                     LockOp lock, Block *succ) {
    int acqMode = lockMode == 0 ? 1 : 0;
    int relMode = lockMode == 0 ? 0 : 1;
    int offset = 0;
    MemRefType buffer = buff.getType();
    int len = 1;
    for (auto i : buffer.getShape()) {
      len *= i;
    }

    builder.create<UseLockOp>(builder.getUnknownLoc(), lock, acqMode,
                              LockAction::Acquire);
    builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, 0);
    builder.create<UseLockOp>(builder.getUnknownLoc(), lock, relMode,
                              LockAction::Release);
    builder.create<cf::BranchOp>(builder.getUnknownLoc(), succ);
  }

  /// Function used to create a MemOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createDMA(OpBuilder &builder, ObjectFifoCreateOp op, DMAChan channelMode,
                 int lockMode) {
    int numBlocks = op.size();

    if (numBlocks == 0)
      return;

    assert(numBlocks <= 14 &&
           "Cannot have more than 16 blocks in a DMA channel.");

    // create MemOp
    builder.setInsertionPointAfter(locksPerFifo[op].back());
    MemOp producerMem =
        builder.create<MemOp>(builder.getUnknownLoc(), op.getProducerTileOp());
    Region &r = producerMem.getBody();
    r.push_back(new Block);
    Block &endBlock = r.back();
    Block *dmaBlock = builder.createBlock(&endBlock);
    Block *bdBlock = builder.createBlock(&endBlock);

    // add terminator operation to end block
    builder.setInsertionPointToStart(&endBlock);
    builder.create<EndOp>(builder.getUnknownLoc());

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelMode, bdBlock,
                               &endBlock);

    // create Bd blocks
    Block *succ = nullptr;
    Block *curr = bdBlock;
    int blockIndex = 0;
    for (int i = 0; i < numBlocks; i++) {
      if (i == numBlocks - 1) {
        succ = bdBlock;
      } else {
        succ = builder.createBlock(&endBlock);
      }
      builder.setInsertionPointToStart(curr);
      createBdBlock(builder, lockMode, buffersPerFifo[op][blockIndex],
                    locksPerFifo[op][blockIndex], succ);
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
  void unrollForLoops(ModuleOp &m, OpBuilder &builder,
                      std::vector<TileOp> objectFifoTiles) {
    for (auto coreOp : m.getOps<CoreOp>()) {
      if (std::find(objectFifoTiles.begin(), objectFifoTiles.end(),
                    coreOp.getTileOp()) != objectFifoTiles.end()) {

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
              checkSplitFifo(acqOp.getOperation());
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
                      DenseMap<ObjectFifoCreateOp, int> &acc, int numLocks,
                      int lockMode, LockAction lockAction) {
    for (int i = 0; i < numLocks; i++) {
      int lockID = acc[op];
      builder.create<UseLockOp>(builder.getUnknownLoc(),
                                locksPerFifo[op][lockID], lockMode, lockAction);
      acc[op] = (lockID + 1) % op.getElemNumber();
    }
  }

  /// Function used to check whether op is already contained in map.
  /// If it is then return the associated int, if not create new entry and
  /// return 0.
  int updateAndReturnIndex(DenseMap<ObjectFifoCreateOp, int> &map,
                           ObjectFifoCreateOp op) {
    if (!map[op]) {
      map[op] = 0;
      return 0;
    }
    return map[op];
  }

  /// Function used to check whether objectFifo accessed by op has been split.
  /// If yes, it replaces the parent objectFifo with the correct child based on
  /// port.
  void checkSplitFifo(Operation *op) {
    ObjectFifoCreateOp parentFifo;
    ObjectFifoPort port;
    if (isa<ObjectFifoAcquireOp>(op)) {
      ObjectFifoAcquireOp acqOp = dyn_cast<ObjectFifoAcquireOp>(op);
      parentFifo = acqOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();
      port = acqOp.getPort();
    } else if (isa<ObjectFifoReleaseOp>(op)) {
      ObjectFifoReleaseOp relOp = dyn_cast<ObjectFifoReleaseOp>(op);
      parentFifo = relOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();
      port = relOp.getPort();
    } else {
      assert(false && "checkSplitFifo() must be called on either "
                      "ObjectFifoAcquireOp or ObjectFifoReleaseOp");
    }

    if (splitFifos.find(parentFifo) != splitFifos.end()) {
      if (port == ObjectFifoPort::Produce) {
        op->replaceUsesOfWith(parentFifo, splitFifos[parentFifo].first);
      } else if (port == ObjectFifoPort::Consume) {
        op->replaceUsesOfWith(parentFifo, splitFifos[parentFifo].second);
      }
    }
  }

  /// Function used to check whether the process that is accessing the
  /// objectFifo is running on a tile matching the port of that
  /// objectFifo.
  void checkCorrectPort(Operation *op) {
    ObjectFifoCreateOp objFifo;
    ObjectFifoPort port;
    if (isa<ObjectFifoAcquireOp>(op)) {
      ObjectFifoAcquireOp acqOp = dyn_cast<ObjectFifoAcquireOp>(op);
      objFifo = acqOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();
      port = acqOp.getPort();
    } else if (isa<ObjectFifoReleaseOp>(op)) {
      ObjectFifoReleaseOp relOp = dyn_cast<ObjectFifoReleaseOp>(op);
      objFifo = relOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();
      port = relOp.getPort();
    } else {
      assert(false && "checkCorrectPort() must be called on either "
                      "ObjectFifoAcquireOp or ObjectFifoReleaseOp");
    }

    Operation *coreOp = op;
    while (!isa<CoreOp>(coreOp)) {
      coreOp = coreOp->getParentOp();
      if (coreOp == nullptr)
        assert(false && "ObjectFifoAcquireOp or ObjectFifoReleaseOp must be "
                        "used inside a CoreOp");
    }
    auto coreTile = dyn_cast<CoreOp>(coreOp).getTile();
    if (port == ObjectFifoPort::Produce) {
      if (coreTile != objFifo.getProducerTile())
        assert(false && "Producer port of objectFifo accessed by core running "
                        "on non-producer tile");
    } else if (port == ObjectFifoPort::Consume) {
      if (coreTile != objFifo.getConsumerTile())
        assert(false && "Consumer port of objectFifo accessed by core running "
                        "on non-consumer tile");
    }
  }

  /// Function used to find the size of an objectFifo after split based on
  /// the maximum number of elements (of the original objectFifo) acquired
  /// by a process running on given tile. If no CoreOp exists for this tile
  /// return 0.
  int findObjectFifoSize(ModuleOp &m, Value tile, ObjectFifoCreateOp objFifo) {

    if (objFifo.size() == 0)
      return 0;

    int maxAcquire = 0;
    for (auto coreOp : m.getOps<CoreOp>()) {
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
    ModuleOp m = getOperation();
    LockAnalysis analysis(m);
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    //===----------------------------------------------------------------------===//
    // Create objectFifos
    //===----------------------------------------------------------------------===//
    std::vector<TileOp> objectFifoTiles;

    for (auto createOp : m.getOps<ObjectFifoCreateOp>()) {
      objectFifoTiles.push_back(createOp.getProducerTileOp());
      objectFifoTiles.push_back(createOp.getConsumerTileOp());

      bool memoryAdjacent =
          isLegalMemAffinity(createOp.getProducerTileOp().colIndex(),
                             createOp.getProducerTileOp().rowIndex(),
                             createOp.getConsumerTileOp().colIndex(),
                             createOp.getConsumerTileOp().rowIndex());
      if (memoryAdjacent) {
        createObjectFifoElements(builder, analysis, createOp);

      } else {
        // Find max acquire number for producer and consumer of objectFifo.
        int prodMaxAcquire =
            findObjectFifoSize(m, createOp.getProducerTile(), createOp);
        int consMaxAcquire =
            findObjectFifoSize(m, createOp.getConsumerTile(), createOp);

        // objectFifos between non-adjacent tiles must be split into two new
        // ones, their elements will be created in next iterations
        builder.setInsertionPointAfter(createOp);
        AIEObjectFifoType fifo = createOp.getType().cast<AIEObjectFifoType>();
        ObjectFifoCreateOp producerFifo = builder.create<ObjectFifoCreateOp>(
            builder.getUnknownLoc(), fifo, createOp.getProducerTile(),
            createOp.getProducerTile(), prodMaxAcquire);
        ObjectFifoCreateOp consumerFifo = builder.create<ObjectFifoCreateOp>(
            builder.getUnknownLoc(), fifo, createOp.getConsumerTile(),
            createOp.getConsumerTile(), consMaxAcquire);

        // record that this objectFifo was split
        splitFifos[createOp] = std::make_pair(producerFifo, consumerFifo);
      }
    }

    //===----------------------------------------------------------------------===//
    // Create flows
    //===----------------------------------------------------------------------===//
    for (auto entry : splitFifos) {
      ObjectFifoCreateOp parentFifo = entry.first;
      ObjectFifoCreateOp childProducerFifo = entry.second.first;
      ObjectFifoCreateOp childConsumerFifo = entry.second.second;

      // create flow between tiles
      builder.setInsertionPointAfter(parentFifo);
      builder.create<FlowOp>(builder.getUnknownLoc(),
                             parentFifo.getProducerTile(), WireBundle::DMA, 0,
                             parentFifo.getConsumerTile(), WireBundle::DMA, 1);

      // create MemOps and DMA channels
      createDMA(builder, childProducerFifo, DMAChan::MM2S0, 0);
      createDMA(builder, childConsumerFifo, DMAChan::S2MM1, 1);
    }

    //===----------------------------------------------------------------------===//
    // Unroll for loops
    //===----------------------------------------------------------------------===//
    unrollForLoops(m, builder, objectFifoTiles);

    //===----------------------------------------------------------------------===//
    // Replace ops
    //===----------------------------------------------------------------------===//
    for (auto coreOp : m.getOps<CoreOp>()) {
      DenseMap<ObjectFifoAcquireOp, std::vector<BufferOp *>>
          subviews; // maps each "subview" to its buffer references (subviews
                    // are created by AcquireOps)
      DenseMap<ObjectFifoCreateOp, std::vector<int>>
          acquiresPerFifo; // maps each objFifo to indices of buffers acquired
                           // in latest subview of that objFifo (useful to
                           // cascade acquired elements to next AcquireOp)
      std::vector<ObjectFifoReleaseOp>
          releaseOps; // useful to check which ReleaseOp has taken place before
                      // an AcquireOp
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
        checkSplitFifo(releaseOp.getOperation());
        checkCorrectPort(releaseOp.getOperation());

        builder.setInsertionPointAfter(releaseOp);
        ObjectFifoCreateOp op =
            releaseOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();
        auto port = releaseOp.getPort();

        // update index of next element to release for this objectFifo
        updateAndReturnIndex(relPerFifo, op);

        // release locks
        int numLocks = releaseOp.relNumber();
        int lockMode = port == ObjectFifoPort::Produce ? 1 : 0;
        createUseLocks(builder, op, relPerFifo, numLocks, lockMode,
                       LockAction::Release);

        // add release op to list
        releaseOps.push_back(releaseOp);
      });

      //===----------------------------------------------------------------------===//
      // Replace objectFifo.acquire ops
      //===----------------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoAcquireOp acquireOp) {
        // if objectFifo was split, replace with correct child
        checkSplitFifo(acquireOp.getOperation());
        checkCorrectPort(acquireOp.getOperation());

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
        for (auto relOp : releaseOps) {
          ObjectFifoCreateOp otherOp =
              relOp.getFifo().getDefiningOp<ObjectFifoCreateOp>();
          // TODO: operations may not be in the same block: currently only
          // support one block level of difference
          if (op == otherOp) {
            if (acquireOp.getOperation()->getBlock() ==
                relOp.getOperation()->getBlock()) {
              if (!acquireOp->isBeforeInBlock(relOp)) {
                releaseOps.erase(
                    releaseOps.begin()); // to ensure that we do not account the
                                         // ReleaseOps again later, after the
                                         // subview is created
                numRel += relOp.relNumber();
              }
            } else {
              Operation *acqBlockDefOp =
                  acquireOp.getOperation()->getBlock()->getParentOp();
              if (relOp.getOperation()->getBlock() ==
                  acqBlockDefOp->getBlock()) {
                if (!acqBlockDefOp->isBeforeInBlock(relOp)) {
                  releaseOps.erase(
                      releaseOps.begin()); // to ensure that we do not account
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
                    releaseOps.erase(
                        releaseOps.begin()); // to ensure that we do not account
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
          // take into account what was already been acquired by previous
          // AcquireOp in program order
          acquiredIndices = acquiresPerFifo[op];
          // take into account what has been released in-between
          assert((size_t)numRel <= acquiredIndices.size() &&
                 "Cannot release more elements than are already acquired.");
          for (int i = 0; i < numRel; i++) {
            acquiredIndices.erase(acquiredIndices.begin());
          }
        }

        // acquire locks
        int numLocks = acquireOp.acqNumber();
        int lockMode = port == ObjectFifoPort::Produce ? 0 : 1;
        int alreadyAcq = acquiredIndices.size();
        int numCreate;
        if (numLocks > alreadyAcq) {
          numCreate = numLocks - alreadyAcq;
        } else {
          numCreate = 0;
        }
        createUseLocks(builder, op, acqPerFifo, numCreate, lockMode,
                       LockAction::Acquire);

        // create subview: buffers that were already acquired + new acquires
        for (int i = 0; i < numCreate; i++) {
          acquiredIndices.push_back(start);
          start = (start + 1) % op.getElemNumber();
        }
        std::vector<BufferOp *> subviewRefs;
        for (auto index : acquiredIndices) {
          subviewRefs.push_back(&buffersPerFifo[op][index]);
        }
        subviews[acquireOp] = subviewRefs;
        acquiresPerFifo[op] = acquiredIndices;
      });

      //===----------------------------------------------------------------------===//
      // Replace subview.access ops
      //===----------------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoSubviewAccessOp accessOp) {
        ObjectFifoAcquireOp acqOp =
            accessOp.getSubview().getDefiningOp<ObjectFifoAcquireOp>();
        auto users = accessOp.getOutput().getUsers();
        assert((size_t)accessOp.getIndex() < subviews[acqOp].size() &&
               "Index out of bounds for subview: accessed farther than number "
               "of acquired elements.");
        for (auto user : users) {
          user->replaceUsesOfWith(accessOp.getOutput(),
                                  *subviews[acqOp][accessOp.getIndex()]);
        }
      });
    }

    //===----------------------------------------------------------------------===//
    // Remove old ops
    //===----------------------------------------------------------------------===//
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<AIEOpRemoval<ObjectFifoCreateOp>>(m.getContext(), m);
    patterns.add<AIEOpRemoval<ObjectFifoAcquireOp>>(m.getContext(), m);
    patterns.add<AIEOpRemoval<ObjectFifoReleaseOp>>(m.getContext(), m);
    patterns.add<AIEOpRemoval<ObjectFifoSubviewAccessOp>>(m.getContext(), m);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEObjectFifoStatefulTransformPass() {
  return std::make_unique<AIEObjectFifoStatefulTransformPass>();
}