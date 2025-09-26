//===- AIEUtils.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/AIEUtils.h"

using namespace mlir;
using namespace xilinx;

static unsigned cachedId = 0;

memref::GlobalOp AIEX::getOrCreateDataMemref(OpBuilder &builder,
                                             AIE::DeviceOp dev,
                                             mlir::Location loc,
                                             ArrayRef<uint32_t> words) {
  uint32_t num_words = words.size();
  MemRefType memrefType = MemRefType::get({num_words}, builder.getI32Type());
  TensorType tensorType =
      RankedTensorType::get({num_words}, builder.getI32Type());
  memref::GlobalOp global = nullptr;
  auto initVal = DenseElementsAttr::get<uint32_t>(tensorType, words);
  auto otherGlobals = dev.getOps<memref::GlobalOp>();
  for (auto g : otherGlobals) {
    if (g.getType() != memrefType)
      continue;
    auto otherValue = g.getInitialValue();
    if (!otherValue)
      continue;
    if (*otherValue != initVal)
      continue;
    global = g;
    break;
  }
  if (!global) {
    std::string name = "blockwrite_data_";
    while (dev.lookupSymbol(name + std::to_string(cachedId)))
      cachedId++;
    name += std::to_string(cachedId);
    global = builder.create<memref::GlobalOp>(
        loc, name, builder.getStringAttr("private"), memrefType, initVal, true,
        nullptr);
  }
  return global;
}
