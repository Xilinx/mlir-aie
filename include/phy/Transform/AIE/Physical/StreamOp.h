//===- StreamOp.h -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Transform/AIE/LoweringPatterns.h"
#include "phy/Transform/Base/LoweringPatterns.h"

#include "mlir/Transforms/DialectConversion.h"

#ifndef MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_STREAMOP_H
#define MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_STREAMOP_H

namespace xilinx {
namespace phy {
namespace transform {
namespace aie {

class StreamOpLoweringPatternSet : public LoweringPatternSet {
  AIELoweringPatternSets *lowering;

public:
  StreamOpLoweringPatternSet(AIELoweringPatternSets *lowering)
      : lowering(lowering){};
  ~StreamOpLoweringPatternSet() override {}

  void populatePatternSet(mlir::RewritePatternSet &patterns) override;
};

} // namespace aie
} // namespace transform
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_STREAMOP_H
