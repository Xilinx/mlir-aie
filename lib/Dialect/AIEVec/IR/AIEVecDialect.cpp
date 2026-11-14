//===- AIEVecDialect.cpp - AIE vector dialect ------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The dialect's constructor, which loads the LLVM dialect. LLVMDialect.h is
// included here alone: next to `using namespace llvm` it makes `Type`
// ambiguous.

#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "aie/Dialect/AIEVec/IR/AIEVecOpsDialect.cpp.inc"
