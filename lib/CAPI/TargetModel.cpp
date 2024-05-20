//===- TargetModel.cpp - C API for AIE TargetModel ------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie-c/TargetModel.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

using namespace mlir;

static inline AieTargetModel wrap(const xilinx::AIE::AIETargetModel &tm) {
  return AieTargetModel{static_cast<uint32_t>(tm.getDevice())};
}

static inline const xilinx::AIE::AIETargetModel& unwrap(AieTargetModel tm) {
  return xilinx::AIE::getTargetModel(static_cast<xilinx::AIE::AIEDevice>(tm.d));
}

AieTargetModel aieGetTargetModel(uint32_t device) {
  return wrap(xilinx::AIE::getTargetModel(
    static_cast<xilinx::AIE::AIEDevice>(device)));
}

int aieTargetModelColumns(AieTargetModel targetModel) {
  return unwrap(targetModel).columns();
}

int aieTargetModelRows(AieTargetModel targetModel) {
  return unwrap(targetModel).rows();
}

int aieTargetModelisNPU(AieTargetModel targetModel) {
  return unwrap(targetModel).isNPU();
}