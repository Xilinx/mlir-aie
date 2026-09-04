//===- Dialects.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "aie-c/Dialects.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/XLLVM/XLLVMDialect.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIE, aie, xilinx::AIE::AIEDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIEX, aiex, xilinx::AIEX::AIEXDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIEVec, aievec,
                                      xilinx::aievec::AIEVecDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(XLLVM, xllvm, xilinx::xllvm::XLLVMDialect)

//===---------------------------------------------------------------------===//
// ObjectFifoType
//===---------------------------------------------------------------------===//

bool aieTypeIsObjectFifoType(MlirType type) {
  return llvm::isa<xilinx::AIE::AIEObjectFifoType>(unwrap(type));
}

MlirType aieObjectFifoTypeGet(MlirType type) {
  return wrap(xilinx::AIE::AIEObjectFifoType::get(
      llvm::cast<mlir::MemRefType>(unwrap(type))));
}

//===---------------------------------------------------------------------===//
// BlockFloatType
//===---------------------------------------------------------------------===//

bool aieTypeIsBlockFloatType(MlirType type) {
  return llvm::isa<xilinx::AIEX::BlockFloatType>(unwrap(type));
}

MlirType aieBlockFloatTypeGet(MlirContext ctx, const std::string &blockType) {
  return wrap(xilinx::AIEX::BlockFloatType::get(unwrap(ctx), blockType));
}

//===---------------------------------------------------------------------===//
// TileLike Interface
//===---------------------------------------------------------------------===//

bool aieOpImplementsTileLike(MlirOperation op) {
  return llvm::isa<xilinx::AIE::TileLike>(unwrap(op));
}

bool aieTileLikeIsCoreTile(MlirOperation op) {
  auto tileLike = llvm::dyn_cast<xilinx::AIE::TileLike>(unwrap(op));
  return tileLike && tileLike.isCoreTile();
}

bool aieTileLikeIsMemTile(MlirOperation op) {
  auto tileLike = llvm::dyn_cast<xilinx::AIE::TileLike>(unwrap(op));
  return tileLike && tileLike.isMemTile();
}

bool aieTileLikeIsShimNOCTile(MlirOperation op) {
  auto tileLike = llvm::dyn_cast<xilinx::AIE::TileLike>(unwrap(op));
  return tileLike && tileLike.isShimNOCTile();
}

bool aieTileLikeIsShimPLTile(MlirOperation op) {
  auto tileLike = llvm::dyn_cast<xilinx::AIE::TileLike>(unwrap(op));
  return tileLike && tileLike.isShimPLTile();
}

bool aieTileLikeIsShimNOCorPLTile(MlirOperation op) {
  auto tileLike = llvm::dyn_cast<xilinx::AIE::TileLike>(unwrap(op));
  return tileLike && tileLike.isShimNOCorPLTile();
}

//===---------------------------------------------------------------------===//
// TraceBufferAttr / TraceSliceAttr
//===---------------------------------------------------------------------===//

bool aieAttrIsTraceBuffer(MlirAttribute attr) {
  return llvm::isa<xilinx::AIE::TraceBufferAttr>(unwrap(attr));
}

uint32_t aieTraceBufferGetArgIndex(MlirAttribute attr) {
  return llvm::cast<xilinx::AIE::TraceBufferAttr>(unwrap(attr)).getArgIndex();
}

uint32_t aieTraceBufferGetOffset(MlirAttribute attr) {
  return llvm::cast<xilinx::AIE::TraceBufferAttr>(unwrap(attr)).getOffset();
}

uint32_t aieTraceBufferGetSize(MlirAttribute attr) {
  return llvm::cast<xilinx::AIE::TraceBufferAttr>(unwrap(attr)).getSize();
}

bool aieTraceBufferGetDedicated(MlirAttribute attr) {
  return llvm::cast<xilinx::AIE::TraceBufferAttr>(unwrap(attr)).getDedicated();
}

bool aieAttrIsTraceSlice(MlirAttribute attr) {
  return llvm::isa<xilinx::AIE::TraceSliceAttr>(unwrap(attr));
}

MlirStringRef aieTraceSliceGetDevice(MlirAttribute attr) {
  return wrap(
      llvm::cast<xilinx::AIE::TraceSliceAttr>(unwrap(attr)).getDevice());
}

MlirStringRef aieTraceSliceGetSequence(MlirAttribute attr) {
  return wrap(
      llvm::cast<xilinx::AIE::TraceSliceAttr>(unwrap(attr)).getSequence());
}

uint32_t aieTraceSliceGetOffset(MlirAttribute attr) {
  return llvm::cast<xilinx::AIE::TraceSliceAttr>(unwrap(attr)).getOffset();
}

uint32_t aieTraceSliceGetSize(MlirAttribute attr) {
  return llvm::cast<xilinx::AIE::TraceSliceAttr>(unwrap(attr)).getSize();
}
