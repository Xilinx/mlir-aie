//===- Dialects.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie-c/Dialects.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIE, aie, xilinx::AIE::AIEDialect)

//===---------------------------------------------------------------------===//
// ObjectFifoType
//===---------------------------------------------------------------------===//

bool aieTypeIsObjectFifoType(MlirType type) {
  return unwrap(type).isa<xilinx::AIE::AIEObjectFifoType>();
}

MlirType aieObjectFifoTypeGet(MlirType type) {
  return wrap(xilinx::AIE::AIEObjectFifoType::get(unwrap(type)));
}
