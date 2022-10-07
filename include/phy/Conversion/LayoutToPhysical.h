//===- LayoutToPhysical.h -------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef PHY_CONVERSION_LAYOUT_TO_PHYSICAL_H_
#define PHY_CONVERSION_LAYOUT_TO_PHYSICAL_H_

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace phy {
std::unique_ptr<mlir::Pass> createLayoutToPhysical();
} // namespace phy
} // namespace xilinx

#endif // PHY_CONVERSION_LAYOUT_TO_PHYSICAL_H_
