//===- InlineFunction.h ------------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"

#ifndef MLIR_PHY_REWRITE_INLINE_FUNCTION_H
#define MLIR_PHY_REWRITE_INLINE_FUNCTION_H

namespace xilinx {
namespace phy {
namespace rewrite {

class Inliner : public mlir::InlinerInterface {
public:
  Inliner(mlir::MLIRContext *context) : InlinerInterface(context) {}
};

class FunctionInliner : public mlir::OpConversionPattern<mlir::func::CallOp> {
  using OpAdaptor = typename mlir::func::CallOp::Adaptor;

public:
  FunctionInliner(mlir::MLIRContext *context)
      : mlir::OpConversionPattern<mlir::func::CallOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

} // namespace rewrite
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_REWRITE_INLINE_FUNCTION_H
