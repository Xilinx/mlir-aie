//===- AIECreateLocks.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "aie/AIETokenAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-create-locks"
using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct Token2LockLowering : public OpConversionPattern<UseTokenOp> {
  using OpConversionPattern<UseTokenOp>::OpConversionPattern;
  ModuleOp &module;
  DenseMap<std::pair<StringRef, int>, std::pair<LockOp, int>>
      &tokenValue2lockState;

  Token2LockLowering(MLIRContext *context, ModuleOp &m,
                     DenseMap<std::pair<StringRef, int>, std::pair<LockOp, int>>
                         &tokenValue2lockState,
                     PatternBenefit benefit = 1)
      : OpConversionPattern<UseTokenOp>(context, benefit), module(m),
        tokenValue2lockState(tokenValue2lockState) {}

  LogicalResult
  matchAndRewrite(UseTokenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lockState =
        tokenValue2lockState[std::make_pair(op.tokenName(), op.value())];

    LockAction action;
    if (op.acquire()) {
      action = LockAction::Acquire;
      LLVM_DEBUG(llvm::dbgs() << "Replacing Acquire: " << op << "\n");
    } else {
      action = LockAction::Release;
      LLVM_DEBUG(llvm::dbgs() << "Replacing Release: " << op << "\n");
    }

    rewriter.create<UseLockOp>(op.getLoc(), lockState.first, lockState.second,
                               action);
    LLVM_DEBUG(llvm::dbgs() << "with lock " << lockState.first << " in state "
                            << lockState.second << "\n");
    rewriter.eraseOp(op.getOperation());

    return success();
  }
};

static LockOp
allocateFullLock(DenseMap<int, DenseMap<int, StringRef>> &lockUsedStates,
                 DenseMap<int, LockOp> &tileLocks, TileOp tile,
                 OpBuilder &builder) {
  for (unsigned i = 0; i < 16; i++) {
    if (lockUsedStates.count(i)) {
      LLVM_DEBUG(llvm::dbgs() << "Lock " << i << " at " << tile
                              << "was already used, skipping...\n");
      continue;
    }
    auto lock = builder.create<LockOp>(builder.getUnknownLoc(), tile, i);
    tileLocks[i] = lock;
    LLVM_DEBUG(llvm::dbgs() << "lock " << lock << " allocated for tokens.\n");
    return lock;
  }
  return nullptr;
}

static std::pair<LockOp, int>
allocateLockState(DenseMap<int, DenseMap<int, StringRef>> &lockUsedStates,
                  DenseMap<int, LockOp> &tileLocks, TileOp tile,
                  StringRef tokenName, OpBuilder &builder) {

  // Try reusing a lock first
  for (unsigned i = 0; i < 16; i++) {
    if (!lockUsedStates.count(i))
      continue;

    // Avoid using the same physical lock for different tokens
    bool usedForDifferentToken = false;
    for (unsigned s = 0; s < 2; s++) {
      if (lockUsedStates[i].count(s) && lockUsedStates[i][s] != tokenName)
        usedForDifferentToken = true;
    }
    if (usedForDifferentToken)
      continue;

    for (unsigned s = 0; s < 2; s++) {
      if (lockUsedStates[i].count(s))
        continue;
      lockUsedStates[i][s] = tokenName;
      return std::make_pair(tileLocks[i], s);
    }
  }

  // Otherwise, allocate a new lock
  auto lock = allocateFullLock(lockUsedStates, tileLocks, tile, builder);
  if (lock == nullptr)
    return std::make_pair(lock, -1);
  lockUsedStates[lock.getLockID()][0] = tokenName;
  return std::make_pair(lock, 0 /* state */);
}

struct AIECreateLocksPass : public AIECreateLocksBase<AIECreateLocksPass> {

  // tileLockUsedStates[tile][lockId] = {state: user tokenName}
  DenseMap<TileOp, DenseMap<int, DenseMap<int, StringRef>>> tileLockUsedStates;
  // tileLocks[tileOp] = {lockId: LockOp}
  DenseMap<TileOp, DenseMap<int, LockOp>> tileLocks;

  // tokenValue2lockState[(tokenName, value)] = (lockOp, state)
  DenseMap<std::pair<StringRef, int>, std::pair<LockOp, int>>
      tokenValue2lockState;

  void reserveExistingLocks(TokenAnalysis &TA) {
    for (auto existingLocks : TA.getTileLocks()) {
      auto tile = existingLocks.first;
      auto locks = existingLocks.second;
      for (auto lock : locks) {
        int lockId = lock.first;
        // all states of exsiting locks shall be reserved
        tileLockUsedStates[tile][lockId][0] = "";
        tileLockUsedStates[tile][lockId][1] = "";
        tileLocks[tile][lockId] = lock.second;
      }
    }
  }

  void mapDMAAndMemcpyPairs(TokenAnalysis &TA, OpBuilder &builder) {
    // Phase 1: DMA and Memcpy pairs shall be mapped into 0/1 states of
    // a physical lock.

    for (auto pair : TA.getTokenPairs()) {
      auto acquire = pair.first;
      auto release = pair.second;
      auto acqUser = TA.getTokenUserOp(acquire);
      auto relUser = TA.getTokenUserOp(release);
      auto acqPair = TA.getTokenUseNameValue(acquire, true);
      auto relPair = TA.getTokenUseNameValue(release, false);

      // skip pairs if they are not in DMA or are MemcpyOp
      if ((!isa<MemcpyOp>(acquire) || !isa<MemcpyOp>(release)) &&
          (!isa<MemOp>(acqUser) || !isa<MemOp>(relUser)) &&
          (!isa<ShimDMAOp>(acqUser) || !isa<ShimDMAOp>(relUser)))
        continue;

      // if one of them is mapped but the other is not, it is impossible to
      // implement this scheme since the unmapped cannot be mapped to the
      // opposite state of the physical lock of the mapped
      assert(tokenValue2lockState.count(acqPair) ==
                 tokenValue2lockState.count(relPair) &&
             "Unable to resolve: DMA/memcpy chaining not supported.");

      // if both are mapped, they need to be in the same physical lock.
      if (tokenValue2lockState.count(acqPair)) {
        assert(tokenValue2lockState[acqPair].first ==
                   tokenValue2lockState[relPair].first &&
               "Unable to resolve: DMA/memcpy mapped to different locks.");
        continue; // no need to allocate :-)
      }

      // select the tile to place the physical lock
      auto tileOp = *(TA.getAccessibleTileOp(acqUser).begin());
      // the legality of the tile selection will be check in the next phase.
      // for now, we make a best selection to be located at the user tile,
      // as the user is the only possible sharable mem selection.

      // allocate a physical lock for the values, and use 0/1 as their states
      builder.setInsertionPointAfter(tileOp);
      auto lock = allocateFullLock(tileLockUsedStates[tileOp],
                                   tileLocks[tileOp], tileOp, builder);
      assert(lock && "No more locks to allocate!");
      tileLockUsedStates[tileOp][lock.getLockID()][1] = acqPair.first;
      tileLockUsedStates[tileOp][lock.getLockID()][0] = relPair.first;
      tokenValue2lockState[acqPair] = std::make_pair(lock, 1 /* state */);
      tokenValue2lockState[relPair] = std::make_pair(lock, 0 /* state */);
      LLVM_DEBUG(llvm::dbgs()
                 << "DMA token " << acqPair.first << " value " << acqPair.second
                 << " is replaced with lock " << lock << " state 1\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "DMA token " << relPair.first << " value " << relPair.second
                 << " is replaced with lock " << lock << " state 0\n");
    }
  }

  void mapTokenValues(TokenAnalysis &TA, OpBuilder &builder) {
    // Phase 2: Mapping all values to physical locks.

    for (auto tokenValues : TA.getTokenValues()) {
      auto tokenName = tokenValues.first;
      auto values = tokenValues.second;
      for (auto value : values) {
        auto valPair = std::make_pair(tokenName, value);

        SmallVector<Operation *> users;
        // locate all users
        for (auto acquire : TA.getTokenAcqMap()[tokenName]) {
          auto acqPair = TA.getTokenUseNameValue(acquire, true);
          // skip acquires with different values
          if (acqPair.second != valPair.second)
            continue;
          users.push_back(TA.getTokenUserOp(acquire));
        }
        for (auto release : TA.getTokenRelMap()[tokenName]) {
          auto relPair = TA.getTokenUseNameValue(release, false);
          // skip releases with different values
          if (relPair.second != valPair.second)
            continue;
          users.push_back(TA.getTokenUserOp(release));
        }

        // Skipping generation if the token is not used
        if (!users.size())
          continue;

        SmallSet<TileOp, 4> possibleTiles =
            TA.getAccessibleTileOp(*users.begin());

        for (auto user : users) {
          auto currPossibleTiles = TA.getAccessibleTileOp(user);

          // find the intersection of the possible tiles
          SmallSet<TileOp, 4> intersection;
          for (auto tileOp : currPossibleTiles)
            if (possibleTiles.count(tileOp))
              intersection.insert(tileOp);
          possibleTiles = intersection;

          assert(possibleTiles.size() && "Unable to place the lock.");
          LLVM_DEBUG(llvm::dbgs()
                     << "Token " << valPair.first << " value " << valPair.second
                     << " may be placed at " << *possibleTiles.begin()
                     << ", iterating...\n");
        }

        if (tokenValue2lockState.count(valPair)) {
          // if the token is already placed, check if it is a possible tile.
          auto tileOp =
              tokenValue2lockState[valPair].first.tile().getDefiningOp();
          assert(possibleTiles.count(dyn_cast<TileOp>(tileOp)) &&
                 "Failed to place the token to a physical tile.");

        } else {
          // otherwise, place it to a possible tile
          for (auto tileOp : possibleTiles) {
            // try all possible tiles until it is placed
            builder.setInsertionPointAfter(tileOp);
            auto lockState =
                allocateLockState(tileLockUsedStates[tileOp], tileLocks[tileOp],
                                  tileOp, valPair.first, builder);
            if (lockState.first != nullptr) {
              tokenValue2lockState[valPair] = lockState;
              LLVM_DEBUG(llvm::dbgs()
                         << "Token " << valPair.first << " value "
                         << valPair.second << "is replaced with lock "
                         << lockState.first << " state " << lockState.second
                         << "\n");
              break;
            }
          }
          assert(tokenValue2lockState.count(valPair) &&
                 "No more locks to allocate!");
        }
      }
    }
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    TokenAnalysis TA(m);
    TA.runAnalysis();
    LLVM_DEBUG(TA.print(llvm::dbgs()));
    reserveExistingLocks(TA);

    mapDMAAndMemcpyPairs(TA, builder);
    mapTokenValues(TA, builder);

    ConversionTarget target(getContext());
    target.addLegalOp<UseLockOp>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<Token2LockLowering>(m.getContext(), m,
                                        tokenValue2lockState);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIECreateLocksPass() {
  return std::make_unique<AIECreateLocksPass>();
}