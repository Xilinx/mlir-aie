//===- ADFOps.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
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