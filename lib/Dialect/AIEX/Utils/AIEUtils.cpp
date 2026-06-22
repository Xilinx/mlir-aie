//===- AIEUtils.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

using namespace mlir;
using namespace xilinx;

// Counter used to name private blockwrite data globals when no caller-supplied
// next-index is available; the symbol-table probe below keeps names unique.
static unsigned blockwriteDataCounter = 0;

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

    // Handle memref.subview. Accepts any source rank; byte-offset delta is
    // (resultOffset - sourceOffset) from the strided layouts. The result
    // slice must remain row-major contiguous so callers can treat it as
    // linear from the returned base offset.
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      auto sourceType = subviewOp.getSourceType();
      auto resultType = subviewOp.getType();

      if (llvm::any_of(subviewOp.getStaticOffsets(), ShapedType::isDynamic) ||
          llvm::any_of(subviewOp.getStaticSizes(), ShapedType::isDynamic) ||
          llvm::any_of(subviewOp.getStaticStrides(), ShapedType::isDynamic))
        return std::nullopt;

      // No skipping in source.
      if (llvm::any_of(subviewOp.getStaticStrides(),
                       [](int64_t s) { return s != 1; }))
        return std::nullopt;

      unsigned elemSizeInBits =
          sourceType.getElementType().getIntOrFloatBitWidth();
      if (elemSizeInBits % 8 != 0)
        return std::nullopt;

      llvm::SmallVector<int64_t> srcStrides, resStrides;
      int64_t srcOff, resOff;
      if (failed(sourceType.getStridesAndOffset(srcStrides, srcOff)) ||
          failed(resultType.getStridesAndOffset(resStrides, resOff)))
        return std::nullopt;
      if (srcOff == ShapedType::kDynamic || resOff == ShapedType::kDynamic)
        return std::nullopt;

      // Result must be row-major contiguous: innermost stride == 1, and each
      // outer stride equals the product of all inner sizes. (Size-1 dims are
      // free: their stride is never stepped.) Without this, e.g. a column
      // slice of a 2D row-major memref would be accepted and patched as if
      // linear.
      ArrayRef<int64_t> resShape = resultType.getShape();
      if (!resShape.empty()) {
        if (resStrides.back() != 1)
          return std::nullopt;
        uint64_t product = 1;
        for (int d = static_cast<int>(resShape.size()) - 1; d > 0; --d) {
          product *= static_cast<uint64_t>(resShape[d]);
          if (resShape[d - 1] > 1 &&
              static_cast<uint64_t>(resStrides[d - 1]) != product)
            return std::nullopt;
        }
      }

      offsetInBytes += (resOff - srcOff) * (elemSizeInBits / 8);

      current = subviewOp.getSource();
      continue;
    }

    // Encountered an unsupported operation
    return std::nullopt;
  }

  return std::nullopt;
}

memref::GlobalOp AIEX::getOrCreateDataMemref(
    OpBuilder &builder, AIE::DeviceOp dev, mlir::Location loc,
    ArrayRef<uint32_t> words,
    llvm::DenseMap<mlir::Attribute, memref::GlobalOp> *dedupCache,
    unsigned *nextId) {
  uint32_t num_words = words.size();
  MemRefType memrefType = MemRefType::get({num_words}, builder.getI32Type());
  TensorType tensorType =
      RankedTensorType::get({num_words}, builder.getI32Type());
  auto initVal = DenseElementsAttr::get<uint32_t>(tensorType, words);

  // Dedup. The initial-value attribute is uniqued and encodes both the element
  // data and the (shaped) type, so it is a sufficient key on its own.
  memref::GlobalOp global = nullptr;
  if (dedupCache) {
    // O(1): the cache is seeded by the caller from the device's existing
    // globals and kept in sync below as we create new ones.
    auto it = dedupCache->find(initVal);
    if (it != dedupCache->end())
      return it->second;
  } else {
    // No cache supplied: scan existing globals for a match. Callers that create
    // many globals should pass a cache (see header).
    for (auto g : dev.getOps<memref::GlobalOp>()) {
      if (g.getType() != memrefType)
        continue;
      auto otherValue = g.getInitialValue();
      if (!otherValue)
        continue;
      if (*otherValue != initVal)
        continue;
      return g;
    }
  }

  // With a caller-supplied `nextId` (one past the largest existing
  // blockwrite_data_<n>) the name is unique by construction; otherwise probe
  // the symbol table for a free index.
  std::string name;
  if (nextId) {
    name = "blockwrite_data_" + std::to_string((*nextId)++);
  } else {
    name = "blockwrite_data_";
    while (dev.lookupSymbol(name + std::to_string(blockwriteDataCounter)))
      blockwriteDataCounter++;
    name += std::to_string(blockwriteDataCounter++);
  }
  global = memref::GlobalOp::create(builder, loc, name,
                                    builder.getStringAttr("private"),
                                    memrefType, initVal, true, nullptr);
  if (dedupCache)
    (*dedupCache)[initVal] = global;
  return global;
}

LogicalResult AIEX::emitUpdateBdAddressFromOffsetParameter(
    OpBuilder &builder, Operation *bdOp, BaseMemRefType bufType,
    uint64_t registerAddr) {
  auto idxAttr = bdOp->getAttrOfType<IntegerAttr>("offset_state_table_idx");
  assert(idxAttr && "emitUpdateBdAddressFromOffsetParameter called without "
                    "offset_state_table_idx attribute");

  uint8_t stateIdx = static_cast<uint8_t>(idxAttr.getUInt());
  uint32_t elemBytes = bufType.getElementTypeBitWidth() / 8;
  // Use func=mul with func_arg=elemBytes so the firmware computes
  // StateTable[idx] * elemBytes = byte offset, added into the BD address
  // register.
  AIEX::NpuUpdateFromScratchpadOp::create(
      builder, bdOp->getLoc(), stateIdx, AIEX::StateTableFunc::Mul,
      /*func_arg=*/elemBytes,
      /*address=*/static_cast<uint32_t>(registerAddr),
      /*buffer=*/nullptr, /*column=*/nullptr, /*row=*/nullptr);
  return success();
}
