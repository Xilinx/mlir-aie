//===- AIELLVMLoopOpt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.
//
//===----------------------------------------------------------------------===//
//
// This file implements loop strength reduction for LLVM dialect loops,
// specifically converting index-carried loops to pointer-carried loops.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aie-llvm-loop-opt"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

//===----------------------------------------------------------------------===//
// Index-to-pointer transformation
//===----------------------------------------------------------------------===//

/// Information about a candidate index-to-pointer transformation
struct LoopIndexCandidate {
  BlockArgument indexArg; // The index argument in loop header
  Operation *gepOp;       // The getelementptr using this index
  Operation *incrementOp; // The add operation incrementing the index
  Value basePtr;          // The base pointer (loop-invariant)
  Value stride;           // The increment stride (loop-invariant)
  unsigned argIndex;      // Position in block arguments
  Type gepResultType;     // Result type of the GEP
  Type gepElemType;       // Element type for the GEP
};

/// Information about a candidate bitcast fusion transformation
struct BitcastFusionCandidate {
  BlockArgument accArg; // The accumulator argument
  Type currentType;     // Current type (e.g., vector<64xi32>)
  Type nativeType;      // Native type for operations (e.g., vector<32xi64>)
  unsigned argIndex;    // Position in block arguments
  SmallVector<LLVM::BitcastOp> inputCasts;  // Bitcasts before operations
  SmallVector<LLVM::BitcastOp> outputCasts; // Bitcasts after operations
};

/// Information about an i64 induction variable that can be narrowed to i32
struct InductionVarNarrowingCandidate {
  BlockArgument indVar;                     // The induction variable (i64)
  unsigned argIndex;                        // Position in block arguments
  SmallVector<LLVM::ICmpOp> comparisons;    // Comparison operations
  SmallVector<LLVM::AddOp> increments;      // Increment operations
  SmallVector<LLVM::MulOp> multiplications; // Multiplication operations
};

/// Analyze a block argument to determine if it's a candidate for
/// index-to-pointer transformation
static std::optional<LoopIndexCandidate>
analyzeIndexArgument(BlockArgument arg, Block *loopHeader) {
  // Must be an integer type (i64, i32, etc.)
  if (!arg.getType().isInteger())
    return std::nullopt;

  LoopIndexCandidate candidate;
  candidate.indexArg = arg;
  candidate.argIndex = arg.getArgNumber();

  // Find the GEPOp that uses this index
  LLVM::GEPOp foundGEP = nullptr;
  for (Operation *user : arg.getUsers()) {
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(user)) {
      // Check if this is a simple GEP: base[index]
      // The index should be used as one of the dynamic indices
      bool indexUsedInGEP = false;
      for (Value dynamicIndex : gepOp.getDynamicIndices()) {
        if (dynamicIndex == arg) {
          indexUsedInGEP = true;
          break;
        }
      }

      if (indexUsedInGEP) {
        if (foundGEP) {
          // Multiple GEPs use this index - not a simple pattern
          return std::nullopt;
        }
        foundGEP = gepOp;
        candidate.gepOp = gepOp.getOperation();
        candidate.basePtr = gepOp.getBase();
        candidate.gepResultType = gepOp.getType();
        candidate.gepElemType = gepOp.getElemType();
      }
    }
  }

  if (!foundGEP)
    return std::nullopt;

  // Check that the GEP result is only used for loads
  for (Operation *user : foundGEP.getResult().getUsers()) {
    if (!isa<LLVM::LoadOp>(user)) {
      // GEP result used for something other than load - skip
      return std::nullopt;
    }
  }

  // Find the AddOp that increments this index
  LLVM::AddOp foundAdd = nullptr;
  for (Operation *user : arg.getUsers()) {
    if (auto addOp = dyn_cast<LLVM::AddOp>(user)) {
      // Check if one operand is the index
      if (addOp.getLhs() == arg || addOp.getRhs() == arg) {
        if (foundAdd) {
          // Multiple adds - not a simple pattern
          return std::nullopt;
        }
        foundAdd = addOp;
        candidate.incrementOp = addOp;
        // Get the stride (the other operand)
        candidate.stride =
            (addOp.getLhs() == arg) ? addOp.getRhs() : addOp.getLhs();
      }
    }
  }

  if (!foundAdd)
    return std::nullopt;

  // Verify the incremented value flows back to the loop header
  // (This is a simplified check - a full implementation would use dominance
  // analysis)
  bool flowsToBackEdge = false;
  for (Operation *user : foundAdd.getResult().getUsers()) {
    if (auto brOp = dyn_cast<LLVM::BrOp>(user)) {
      // Check if this branch goes to our loop header
      if (brOp->getSuccessor(0) == loopHeader) {
        // Check if our add result is passed as the corresponding argument
        if (brOp.getDestOperands()[candidate.argIndex] ==
            foundAdd.getResult()) {
          flowsToBackEdge = true;
          break;
        }
      }
    }
  }

  if (!flowsToBackEdge)
    return std::nullopt;

  return candidate;
}

/// Check if a value is loop-invariant (defined outside the loop)
static bool isLoopInvariant(Value val, Block *loopHeader) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return true; // Block argument or constant
  return defOp->getBlock() != loopHeader;
}

/// Find loop headers by looking for blocks with back-edges
/// A block is a loop header if it has a back-edge pointing to it
static SmallVector<Block *> findLoopHeaders(LLVM::LLVMFuncOp func) {
  SmallVector<Block *> loopHeaders;
  DenseSet<Block *> visited;

  // A block is a loop header if:
  // - It has block arguments (phi nodes)
  // - One of its successors branches back to it (creating a cycle)
  for (Block &block : func.getBlocks()) {
    if (block.getNumArguments() == 0)
      continue; // Loop headers have arguments (loop-carried values)

    // Check if any successor eventually branches back to this block
    for (Block *succ : block.getSuccessors()) {
      // Look for back-edge: does this successor branch back to us?
      for (Block *succOfSucc : succ->getSuccessors()) {
        if (succOfSucc == &block) {
          // Found a back-edge: succ -> block
          if (llvm::find(loopHeaders, &block) == loopHeaders.end()) {
            loopHeaders.push_back(&block);
            LLVM_DEBUG(llvm::dbgs() << "Found loop header block\n");
          }
          break;
        }
      }
    }
  }

  return loopHeaders;
}

/// Transform a loop by converting index arguments to pointer arguments
static void transformLoop(Block *loopHeader,
                          ArrayRef<LoopIndexCandidate> candidates,
                          OpBuilder &builder) {
  if (candidates.empty())
    return;

  LLVM_DEBUG(llvm::dbgs() << "Transforming loop with " << candidates.size()
                          << " candidates\n");

  // Step 1: Find the loop preheader (predecessor that enters the loop)
  Block *preheader = nullptr;
  for (Block *pred : loopHeader->getPredecessors()) {
    // Check if this is not a back-edge
    bool isBackEdge = false;
    for (Block *succ : loopHeader->getSuccessors()) {
      if (succ == pred) {
        isBackEdge = true;
        break;
      }
    }
    if (!isBackEdge && !preheader) {
      preheader = pred;
    }
  }

  if (!preheader) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find preheader\n");
    return; // Can't find preheader
  }

  // Step 2: For each candidate, compute initial pointer in preheader
  DenseMap<unsigned, Value> argIndexToInitPtr;

  builder.setInsertionPoint(preheader->getTerminator());

  for (const auto &candidate : candidates) {
    // Get the initial index value from preheader's branch
    Value initialIndex = nullptr;
    for (Operation &op : preheader->getOperations()) {
      if (auto brOp = dyn_cast<LLVM::BrOp>(&op)) {
        if (brOp->getSuccessor(0) == loopHeader) {
          initialIndex = brOp.getDestOperands()[candidate.argIndex];
          break;
        }
      }
    }

    if (!initialIndex)
      continue;

    // Create initial GEP: base[initial_index]
    auto initGEP = builder.create<LLVM::GEPOp>(
        preheader->getTerminator()->getLoc(), candidate.gepResultType,
        candidate.gepElemType, candidate.basePtr, ValueRange{initialIndex});

    argIndexToInitPtr[candidate.argIndex] = initGEP.getResult();
  }

  // Step 3: Change block argument types from i64 to !llvm.ptr
  for (const auto &candidate : candidates) {
    // Change the type of the block argument
    loopHeader->getArgument(candidate.argIndex)
        .setType(candidate.gepResultType);
  }

  // Step 4: Update preheader branch to pass pointers instead of indices
  builder.setInsertionPoint(preheader->getTerminator());
  auto preheaderBr = cast<LLVM::BrOp>(preheader->getTerminator());
  SmallVector<Value> newOperands;
  for (unsigned i = 0; i < preheaderBr.getDestOperands().size(); ++i) {
    if (argIndexToInitPtr.count(i)) {
      newOperands.push_back(argIndexToInitPtr[i]);
    } else {
      newOperands.push_back(preheaderBr.getDestOperands()[i]);
    }
  }
  builder.create<LLVM::BrOp>(preheaderBr.getLoc(), newOperands, loopHeader);
  preheaderBr.erase();

  // Step 5: Update loop body - replace GEP and Add operations
  for (const auto &candidate : candidates) {
    BlockArgument ptrArg = loopHeader->getArgument(candidate.argIndex);

    // Replace uses of GEP result with direct pointer
    candidate.gepOp->getResult(0).replaceAllUsesWith(ptrArg);

    // Replace Add with GEP for pointer arithmetic
    builder.setInsertionPoint(candidate.incrementOp);
    auto ptrIncrement = builder.create<LLVM::GEPOp>(
        candidate.incrementOp->getLoc(), ptrArg.getType(),
        candidate.gepElemType, ptrArg, ValueRange{candidate.stride});

    candidate.incrementOp->getResult(0).replaceAllUsesWith(
        ptrIncrement.getResult());

    // Clean up old operations
    candidate.incrementOp->erase();
    candidate.gepOp->erase();
  }
}

/// Fuse load-bitcast pairs in a block
static void fuseLoadBitcasts(Block *block, OpBuilder &builder) {
  SmallVector<std::pair<LLVM::LoadOp, LLVM::BitcastOp>> fusePairs;

  // Find all load-bitcast pairs
  for (Operation &op : block->getOperations()) {
    if (auto loadOp = dyn_cast<LLVM::LoadOp>(&op)) {
      // Check if this load has a single bitcast user
      if (loadOp.getResult().hasOneUse()) {
        if (auto bitcastOp = dyn_cast<LLVM::BitcastOp>(
                *loadOp.getResult().getUsers().begin())) {
          fusePairs.push_back({loadOp, bitcastOp});
        }
      }
    }
  }

  // Fuse pairs by replacing load result type and removing bitcast
  for (auto [loadOp, bitcastOp] : fusePairs) {
    builder.setInsertionPoint(loadOp);

    // Create new load with bitcast result type using the correct builder
    // signature
    auto newLoad = builder.create<LLVM::LoadOp>(
        loadOp.getLoc(), bitcastOp.getType(), loadOp.getAddr());

    // Copy attributes if present
    if (loadOp.getAlignmentAttr())
      newLoad.setAlignmentAttr(loadOp.getAlignmentAttr());
    if (loadOp.getVolatile_Attr())
      newLoad.setVolatile_Attr(loadOp.getVolatile_Attr());
    if (loadOp.getNontemporalAttr())
      newLoad.setNontemporalAttr(loadOp.getNontemporalAttr());

    // Replace bitcast uses with new load
    bitcastOp.getResult().replaceAllUsesWith(newLoad.getResult());

    // Erase old operations
    bitcastOp.erase();
    loadOp.erase();

    LLVM_DEBUG(llvm::dbgs() << "Fused load-bitcast pair\n");
  }
}

/// Optimize loop-carried accumulator bitcasts by changing block argument types
/// This handles the pattern:
///   arg (vector<64xi32>) -> bitcast to vector<32xi64> -> op -> bitcast back to
///   vector<64xi32> -> pass to next iteration
static void optimizeAccumulatorBitcasts(Block *loopHeader, OpBuilder &builder) {
  SmallVector<std::tuple<BlockArgument, Type, unsigned>> transformations;

  // Find block arguments that follow the bitcast pattern
  for (BlockArgument arg : loopHeader->getArguments()) {
    auto vecType = dyn_cast<VectorType>(arg.getType());
    if (!vecType)
      continue;

    // Check pattern: has bitcasts to same type (but may have other uses like
    // shufflevector)
    Type targetType = nullptr;
    SmallVector<LLVM::BitcastOp> directBitcasts;

    for (Operation *user : arg.getUsers()) {
      if (auto bitcastOp = dyn_cast<LLVM::BitcastOp>(user)) {
        if (!targetType) {
          targetType = bitcastOp.getType();
        } else if (targetType != bitcastOp.getType()) {
          // Bitcasts to different types - skip
          targetType = nullptr;
          break;
        }
        directBitcasts.push_back(bitcastOp);
      } else {
        // Allow other uses (e.g., shufflevector in exit blocks)
      }
    }

    // We need at least some bitcasts to a consistent type
    if (!targetType || directBitcasts.empty())
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Checking arg " << arg.getArgNumber()
                            << " has bitcasts to " << targetType << "\n");

    // Verify that values passed back to loop header are bitcast from targetType
    // Look at all branches to this header
    bool validPattern = true;
    for (auto it = loopHeader->pred_begin(); it != loopHeader->pred_end();
         ++it) {
      Block *pred = *it;
      // Skip preheader (entry to loop)
      if (pred == loopHeader->getSinglePredecessor()) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping preheader\n");
        continue;
      }

      if (auto brOp = dyn_cast_or_null<LLVM::BrOp>(pred->getTerminator())) {
        if (brOp->getSuccessor(0) == loopHeader) {
          Value passedValue = brOp.getDestOperands()[arg.getArgNumber()];

          LLVM_DEBUG(llvm::dbgs() << "Checking branch operand type: "
                                  << passedValue.getType() << "\n");

          // The passed value should have the same type as the current arg type
          // (it will be bitcast to targetType after we transform)
          if (passedValue.getType() != arg.getType()) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "Invalid pattern: operand type doesn't match arg type\n");
            validPattern = false;
            break;
          }

          // Optionally check if it's derived from an operation that produces
          // targetType This is a sanity check but not strictly required
          if (auto bitcastOp = passedValue.getDefiningOp<LLVM::BitcastOp>()) {
            if (bitcastOp.getArg().getType() == targetType) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Operand is bitcast from targetType - good!\n");
            }
          }
        }
      }
    }

    if (validPattern) {
      transformations.push_back({arg, targetType, arg.getArgNumber()});
      LLVM_DEBUG(llvm::dbgs() << "Found accumulator bitcast candidate at arg "
                              << arg.getArgNumber() << " from " << arg.getType()
                              << " to " << targetType << "\n");
    }
  }

  if (transformations.empty())
    return;

  // Apply transformations
  for (auto [arg, targetType, argNum] : transformations) {
    LLVM_DEBUG(llvm::dbgs() << "Transforming argument " << argNum << " from "
                            << arg.getType() << " to " << targetType << "\n");

    Type originalType = arg.getType();

    // Step 1: Change block argument type to targetType (e.g., vector<64xi32> ->
    // vector<32xi64>)
    arg.setType(targetType);

    // Step 2: Insert bitcast at start of loop to convert back to original type
    // This hoisted bitcast replaces all the scattered bitcasts
    builder.setInsertionPointToStart(loopHeader);
    auto hoistedBitcast = builder.create<LLVM::BitcastOp>(
        loopHeader->front().getLoc(), originalType, arg);

    LLVM_DEBUG(llvm::dbgs() << "Created hoisted bitcast from " << targetType
                            << " to " << originalType << "\n");

    // Step 3: Replace all uses of arg with the hoisted bitcast
    // (except the hoisted bitcast itself which uses arg)
    arg.replaceAllUsesExcept(hoistedBitcast.getResult(), hoistedBitcast);

    LLVM_DEBUG(llvm::dbgs() << "Replaced arg uses with hoisted bitcast\n");

    // Step 4: Update ALL branches (including preheader) to pass targetType
    for (auto it = loopHeader->pred_begin(); it != loopHeader->pred_end();
         ++it) {
      Block *pred = *it;

      if (auto brOp = dyn_cast<LLVM::BrOp>(pred->getTerminator())) {
        if (brOp->getSuccessor(0) == loopHeader) {
          Value operand = brOp.getDestOperands()[argNum];
          Value newOperand = operand;

          // If operand is bitcast from targetType to originalType, use the
          // source directly
          if (auto bitcastOp = operand.getDefiningOp<LLVM::BitcastOp>()) {
            if (bitcastOp.getArg().getType() == targetType) {
              newOperand = bitcastOp.getArg();
              LLVM_DEBUG(llvm::dbgs() << "Using bitcast source directly\n");
            }
          }

          // If operand type doesn't match target, insert bitcast
          if (newOperand.getType() != targetType) {
            builder.setInsertionPoint(brOp);
            auto bitcast = builder.create<LLVM::BitcastOp>(
                brOp.getLoc(), targetType, newOperand);
            newOperand = bitcast.getResult();
            LLVM_DEBUG(llvm::dbgs()
                       << "Inserted bitcast to targetType in branch\n");
          }

          // Update branch with new operand
          if (newOperand != operand) {
            builder.setInsertionPoint(brOp);
            SmallVector<Value> newOperands(brOp.getDestOperands());
            newOperands[argNum] = newOperand;
            builder.create<LLVM::BrOp>(brOp.getLoc(), newOperands, loopHeader);
            brOp.erase();
          }
        }
      }
    }
  }
}

struct AIELLVMLoopOptPass : AIELLVMLoopOptBase<AIELLVMLoopOptPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    OpBuilder builder(&getContext());

    SmallVector<Block *> loopHeaders = findLoopHeaders(func);

    LLVM_DEBUG(llvm::dbgs()
               << "Found " << loopHeaders.size() << " loop headers\n");

    // Phase 1: Index-to-pointer transformation
    for (Block *header : loopHeaders) {
      LLVM_DEBUG(llvm::dbgs() << "Analyzing loop header with "
                              << header->getNumArguments() << " arguments\n");

      SmallVector<LoopIndexCandidate> candidates;

      for (BlockArgument arg : header->getArguments()) {
        if (auto candidate = analyzeIndexArgument(arg, header)) {
          if (isLoopInvariant(candidate->basePtr, header)) {
            LLVM_DEBUG(llvm::dbgs() << "Found candidate at argument "
                                    << candidate->argIndex << "\n");
            candidates.push_back(*candidate);
          }
        }
      }

      if (!candidates.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Transforming " << candidates.size() << " candidates\n");
        transformLoop(header, candidates, builder);
      }
    }

    // Phase 2: Fuse load-bitcast pairs
    LLVM_DEBUG(llvm::dbgs() << "Phase 2: Fusing load-bitcast pairs\n");
    for (Block &block : func.getBlocks()) {
      fuseLoadBitcasts(&block, builder);
    }

    // Phase 3: Optimize loop-carried accumulator bitcasts
    LLVM_DEBUG(llvm::dbgs() << "Phase 3: Optimizing accumulator bitcasts\n");
    for (Block *header : loopHeaders) {
      optimizeAccumulatorBitcasts(header, builder);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<LLVM::LLVMFuncOp>>
AIE::createAIELLVMLoopOptPass() {
  return std::make_unique<AIELLVMLoopOptPass>();
}
