//===- XLLVMOps.td - XLLVM dialect op definitions ---------*- tablegen -*-====//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines external LLVM (XLLVM) dialect operations.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_XLLVM_IR_XLLVMOPS_TD
#define AIE_DIALECT_XLLVM_IR_XLLVMOPS_TD

include "aie/Dialect/XLLVM/IR/XLLVM.td"

// Include AIE2 intrinsics.
include "aie/Dialect/XLLVM/IR/XLLVMAIE2IntrOps.td"

#endif
