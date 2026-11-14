//===- AIEVecDialect.cpp - AIE vector dialect ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// The dialect's constructor, which loads the LLVM dialect for the FIFO
// stream address type. Kept apart from AIEVecOps.cpp: its using namespace
// llvm makes Type ambiguous once LLVMDialect.h is included.

#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "aie/Dialect/AIEVec/IR/AIEVecOpsDialect.cpp.inc"
