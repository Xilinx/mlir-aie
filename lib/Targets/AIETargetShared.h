//===- AIETargetShared.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

#include "aie/Dialect/AIE/AIENetlistAnalysis.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#ifndef AIETargetShared_XAIEV2_CDO_H
#define AIETargetShared_XAIEV2_CDO_H

namespace xilinx {
namespace AIE {

std::string tileLocStr(StringRef col, StringRef row);

std::string tileLocStr(uint32_t col, uint32_t row);

std::string tileDMAInstStr(StringRef col, StringRef row, StringRef bdNum);

std::string tileDMAInstStr(uint32_t col, uint32_t row, uint32_t bdNum);

std::string tileDMAInstRefStr(StringRef col, StringRef row, StringRef bdNum);

std::string tileDMAInstRefStr(uint32_t col, uint32_t row, uint32_t bdNum);

std::string packetStr(StringRef id, StringRef type);

std::string packetStr(uint32_t id, uint32_t type);

void generateXAieDmaSetMultiDimAddr(raw_ostream &output, uint32_t ndims,
                                    ArrayRef<DimTupleAttr> dims, uint32_t col,
                                    uint32_t row, uint32_t bdNum,
                                    uint32_t baseAddrA, uint32_t offsetA,
                                    uint32_t lenA, uint32_t bytesA,
                                    const char *error_ret);

} // namespace AIE
} // namespace xilinx

#endif
