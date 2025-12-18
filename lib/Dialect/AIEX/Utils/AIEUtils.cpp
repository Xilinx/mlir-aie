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

std::optional<AIEX::SubviewTraceResult>
AIEX::traceSubviewToBlockArgument(Value value) {
  int64_t offsetInBytes = 0;
  Value current = value;

  // Walk through the chain of operations until we reach a block argument
  while (current) {
    // Check if we've reached a block argument
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      return SubviewTraceResult{blockArg, offsetInBytes};
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      return std::nullopt;
    }

    // Handle memref.cast (just pass through)
    if (auto castOp = dyn_cast<memref::CastOp>(defOp)) {
      current = castOp.getSource();
      continue;
    }

    // Handle memref.reinterpret_cast (validate and pass through)
    if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      auto sourceType =
          dyn_cast<MemRefType>(reinterpretOp.getSource().getType());
      if (!sourceType) {
        return std::nullopt;
      }

      // Validate that source is contiguous (all strides must be 1)
      if (auto strided = dyn_cast<StridedLayoutAttr>(sourceType.getLayout())) {
        for (int64_t stride : strided.getStrides()) {
          if (stride != 1) {
            return std::nullopt; // Non-contiguous memory, cannot safely
                                 // reinterpret
          }
        }
      }

      current = reinterpretOp.getSource();
      continue;
    }

    // Handle memref.subview (accumulate offset and validate)
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      // Verify static offsets, sizes, strides
      if (!subviewOp.getStaticOffsets().empty() &&
          subviewOp.getStaticOffsets()[0] == ShapedType::kDynamic) {
        return std::nullopt;
      }
      if (!subviewOp.getStaticSizes().empty() &&
          subviewOp.getStaticSizes()[0] == ShapedType::kDynamic) {
        return std::nullopt;
      }
      if (!subviewOp.getStaticStrides().empty() &&
          subviewOp.getStaticStrides()[0] == ShapedType::kDynamic) {
        return std::nullopt;
      }

      // Only support rank-1 subviews
      if (subviewOp.getSourceType().getRank() != 1 ||
          subviewOp.getType().getRank() != 1) {
        return std::nullopt;
      }

      // Only support stride of 1 (contiguous)
      if (!subviewOp.getStaticStrides().empty() &&
          subviewOp.getStaticStrides()[0] != 1) {
        return std::nullopt;
      }

      // Calculate and accumulate offset in bytes
      auto sourceType = subviewOp.getSourceType();
      unsigned elemSizeInBits =
          sourceType.getElementType().getIntOrFloatBitWidth();
      if (elemSizeInBits % 8 != 0) {
        return std::nullopt;
      }
      unsigned elemSizeInBytes = elemSizeInBits / 8;
      int64_t offsetInElements = subviewOp.getStaticOffsets()[0];
      offsetInBytes += offsetInElements * elemSizeInBytes;

      current = subviewOp.getSource();
      continue;
    }

    // Encountered an unsupported operation
    return std::nullopt;
  }

  return std::nullopt;
}

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
    global = memref::GlobalOp::create(builder, loc, name,
                                      builder.getStringAttr("private"),
                                      memrefType, initVal, true, nullptr);
  }
  return global;
}
