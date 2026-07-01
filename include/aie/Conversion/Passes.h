//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_PASSES_H
#define AIE_CONVERSION_PASSES_H

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Conversion/AIEVecToLLVM/AIEVecToLLVM.h"
#include "aie/Conversion/PassesEnums.h.inc"

namespace xilinx {

#define GEN_PASS_DECL
#include "aie/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "aie/Conversion/Passes.h.inc"

} // namespace xilinx

#endif // AIE_CONVERSION_PASSES_H
