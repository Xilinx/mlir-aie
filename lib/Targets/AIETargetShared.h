//===- AIETargetShared.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIETargetShared_XAIEV2_CDO_H
#define AIETargetShared_XAIEV2_CDO_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

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
                                    llvm::ArrayRef<BDDimLayoutAttr> dims,
                                    int col, int row, int bdNum, int baseAddrA,
                                    int offsetA, int lenA,
                                    int elementWidthInBytes,
                                    const char *errorRet);

} // namespace AIE
} // namespace xilinx

#endif
