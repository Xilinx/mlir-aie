//===- Dialects.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "aie-c/Dialects.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/XLLVM/XLLVMDialect.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIE, aie, xilinx::AIE::AIEDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIEX, aiex, xilinx::AIEX::AIEXDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIEVec, aievec,
                                      xilinx::aievec::AIEVecDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(XLLVM, xllvm, xilinx::xllvm::XLLVMDialect)

//===---------------------------------------------------------------------===//
// ObjectFifoType
//===---------------------------------------------------------------------===//

bool aieTypeIsObjectFifoType(MlirType type) {
  return llvm::isa<xilinx::AIE::AIEObjectFifoType>(unwrap(type));
}

MlirType aieObjectFifoTypeGet(MlirType type) {
  return wrap(xilinx::AIE::AIEObjectFifoType::get(
      llvm::cast<mlir::MemRefType>(unwrap(type))));
}

//===---------------------------------------------------------------------===//
// ObjectFifoSubviewType
//===---------------------------------------------------------------------===//

bool aieTypeIsObjectFifoSubviewType(MlirType type) {
  return llvm::isa<xilinx::AIE::AIEObjectFifoSubviewType>(unwrap(type));
}

MlirType aieObjectFifoSubviewTypeGet(MlirType type) {
  return wrap(xilinx::AIE::AIEObjectFifoSubviewType::get(
      llvm::cast<mlir::MemRefType>(unwrap(type))));
}

//===---------------------------------------------------------------------===//
// BlockFloatType
//===---------------------------------------------------------------------===//

bool aieTypeIsBlockFloatType(MlirType type) {
  return llvm::isa<xilinx::AIEX::BlockFloatType>(unwrap(type));
}

MlirType aieBlockFloatTypeGet(MlirContext ctx, const std::string &blockType) {
  return wrap(xilinx::AIEX::BlockFloatType::get(unwrap(ctx), blockType));
}
