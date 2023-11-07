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

std::string tileLocStr(llvm::StringRef col, llvm::StringRef row);

std::string tileLocStr(int col, int row);

std::string tileDMAInstStr(llvm::StringRef col, llvm::StringRef row,
                           llvm::StringRef bdNum);

std::string tileDMAInstStr(int col, int row, int bdNum);

std::string tileDMAInstRefStr(llvm::StringRef col, llvm::StringRef row,
                              llvm::StringRef bdNum);

std::string tileDMAInstRefStr(int col, int row, int bdNum);

std::string packetStr(llvm::StringRef id, llvm::StringRef type);

std::string packetStr(int id, int type);

void generateXAieDmaSetMultiDimAddr(llvm::raw_ostream &output, int ndims,
                                    llvm::ArrayRef<DimTupleAttr> dims, int col,
                                    int row, int bdNum, int baseAddrA,
                                    int offsetA, int lenA, int bytesA,
                                    const char *error_ret);

} // namespace AIE
} // namespace xilinx

#endif
