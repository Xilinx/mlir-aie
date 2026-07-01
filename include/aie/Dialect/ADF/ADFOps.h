//===- ADFOps.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021 Xilinx, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XILINX_DIALECT_ADF_H
#define XILINX_DIALECT_ADF_H

#include "aie/Dialect/ADF/ADFDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "aie/Dialect/ADF/ADF.h.inc"

#endif // XILINX_DIALECT_ADF_H