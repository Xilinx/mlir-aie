//===- AIEDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIEX_DIALECT_H
#define MLIR_AIEX_DIALECT_H

#include <optional>

#include "aie/Dialect/AIE/IR/AIEDialect.h"

// Include dialect declarations such as parseAttributes, parseType
#include "aie/Dialect/AIEX/IR/AIEXDialect.h.inc"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "aie/Dialect/AIEX/IR/AIEX.h.inc"

#endif
