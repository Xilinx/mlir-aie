//===- XLLVMOps.td - XLLVM dialect op definitions ---------*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
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
