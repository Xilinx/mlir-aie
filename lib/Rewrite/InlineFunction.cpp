//===- InlineFunction.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Rewrite/InlineFunction.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include <list>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::func;
using namespace xilinx::phy::rewrite;

LogicalResult
FunctionInliner::matchAndRewrite(CallOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Inliner inliner(rewriter.getContext());

  // Get function symbol
  auto sym = op.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return failure();

  // Get FuncOp
  auto func = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(op, sym));
  if (!func)
    return failure();

  // Inline the function body to the location of the call operation.
  auto result = inlineCall(inliner, op, func, func.getCallableRegion());

  if (result.succeeded()) {
    auto *parent_region = op.getOperation()->getParentRegion();

    // Erase the function calling operation
    rewriter.eraseOp(op);

    // Erase blocks without predecessors after inlining to make sure all blocks
    // are reachable.  For example, if the inlined function is an infinite loop,
    // all the blocks following the function call are unreachable.
    std::list<Block *> dead_blocks;
    for (auto &block : *parent_region)
      if (!block.isEntryBlock() && block.hasNoPredecessors())
        dead_blocks.push_back(&block);

    for (auto *block : dead_blocks)
      rewriter.eraseBlock(block);
  }

  return result;
}
