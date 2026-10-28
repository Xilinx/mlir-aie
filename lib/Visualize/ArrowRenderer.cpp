//===- ArrowRenderer.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ArrowRenderer.h"
#include "LayoutGeometry.h"

using namespace llvm;

namespace xilinx::AIE {

ArrowRenderer::~ArrowRenderer() = default;

namespace {

class StraightArrowRenderer : public ArrowRenderer {
public:
  llvm::SmallVector<ArrowPath> route(llvm::ArrayRef<ArrowRequest>,
                                     llvm::ArrayRef<RoutingObstacle>,
                                     const LayoutGeometry &) override {
    return {};
  }
};

} // namespace

std::unique_ptr<ArrowRenderer> createStraightArrowRenderer() {
  return std::make_unique<StraightArrowRenderer>();
}

} // namespace xilinx::AIE
