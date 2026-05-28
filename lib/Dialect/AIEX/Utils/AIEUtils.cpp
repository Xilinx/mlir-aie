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

    // Handle memref.subview (accumulate base byte offset and pass through).
    //
    // Supports any source rank, including rank-reducing subviews. The byte
    // offset delta is read directly off the strided layouts: the subview
    // verifier already bakes Sum(offset[d] * sourceStride[d]) into the
    // result type offset, so (resultOffset - sourceOffset) is the delta in
    // elements.
    //
    // Result-slice contiguity is intentionally not enforced here. The caller
    // is responsible for supplying DMA size/stride descriptors that correctly
    // describe the access pattern on the root buffer; the aie/aiex dialect
    // verifiers (DMABDOp, NpuDmaMemcpyNdOp) enforce hardware legality of
    // those descriptors independently.
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      auto sourceType = subviewOp.getSourceType();
      auto resultType = subviewOp.getType();

      // All slicing parameters must be static.
      if (llvm::any_of(subviewOp.getStaticOffsets(), ShapedType::isDynamic) ||
          llvm::any_of(subviewOp.getStaticSizes(), ShapedType::isDynamic) ||
          llvm::any_of(subviewOp.getStaticStrides(), ShapedType::isDynamic))
        return std::nullopt;

      // Slicing strides must be 1 (no skipping in source).
      if (llvm::any_of(subviewOp.getStaticStrides(),
                       [](int64_t s) { return s != 1; }))
        return std::nullopt;

      // Element size must be byte-addressable.
      unsigned elemSizeInBits =
          sourceType.getElementType().getIntOrFloatBitWidth();
      if (elemSizeInBits % 8 != 0)
        return std::nullopt;

      // Read the byte-offset delta off the strided layouts.
      llvm::SmallVector<int64_t> srcStrides, resStrides;
      int64_t srcOff, resOff;
      if (failed(sourceType.getStridesAndOffset(srcStrides, srcOff)) ||
          failed(resultType.getStridesAndOffset(resStrides, resOff)))
        return std::nullopt;
      if (srcOff == ShapedType::kDynamic || resOff == ShapedType::kDynamic)
        return std::nullopt;

      offsetInBytes += (resOff - srcOff) * (elemSizeInBits / 8);

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
