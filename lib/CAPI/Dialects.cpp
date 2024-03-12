//===- Dialects.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  return unwrap(type).isa<xilinx::AIE::AIEObjectFifoType>();
}

MlirType aieObjectFifoTypeGet(MlirType type) {
  return wrap(xilinx::AIE::AIEObjectFifoType::get(
      unwrap(type).cast<mlir::MemRefType>()));
}

//===---------------------------------------------------------------------===//
// ObjectFifoSubviewType
//===---------------------------------------------------------------------===//

bool aieTypeIsObjectFifoSubviewType(MlirType type) {
  return unwrap(type).isa<xilinx::AIE::AIEObjectFifoSubviewType>();
}

MlirType aieObjectFifoSubviewTypeGet(MlirType type) {
  return wrap(xilinx::AIE::AIEObjectFifoSubviewType::get(
      unwrap(type).cast<mlir::MemRefType>()));
}
