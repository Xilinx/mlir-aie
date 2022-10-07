//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_PASSDETAIL_H
#define CONVERSION_PASSDETAIL_H

#include "aie/AIEDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "phy/Conversion/LayoutToPhysical.h"
#include "phy/Conversion/Nop.h"
#include "phy/Conversion/PhysicalToAie.h"
#include "phy/Dialect/Layout/LayoutDialect.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

#include <string>

namespace xilinx {
namespace phy {

// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "phy/Conversion/Passes.h.inc"

} // namespace phy
} // namespace xilinx

#endif // CONVERSION_PASSDETAIL_H
