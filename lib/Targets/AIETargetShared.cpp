//===- AIETargetShared.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021 Xilinx Inc.
// Copyright (C) 2021-2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//===----------------------------------------------------------------------===//

#include "AIETargetShared.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace xilinx::AIE {

std::string tileLocStr(StringRef col, StringRef row) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "XAie_TileLoc(" << col << "," << row << ")";
  return str;
}

std::string tileLocStr(int col, int row) {
  return tileLocStr(std::to_string(col), std::to_string(row));
}

std::string tileDMAInstStr(StringRef col, StringRef row, StringRef bdNum) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "dma_tile" << col << row << "_bd" << bdNum;
  return str;
}

std::string tileDMAInstStr(int col, int row, int bdNum) {
  return tileDMAInstStr(std::to_string(col), std::to_string(row),
                        std::to_string(bdNum));
}

std::string tileDMAInstRefStr(StringRef col, StringRef row, StringRef bdNum) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "&(" << tileDMAInstStr(col, row, bdNum) << ")";
  return str;
}

std::string tileDMAInstRefStr(int col, int row, int bdNum) {
  return tileDMAInstRefStr(std::to_string(col), std::to_string(row),
                           std::to_string(bdNum));
}

std::string packetStr(StringRef id, StringRef type) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "XAie_PacketInit(" << id << "," << type << ")";
  return str;
}

std::string packetStr(int id, int type) {
  return packetStr(std::to_string(id), std::to_string(type));
}

static std::string tileDMATensorStr(StringRef col, StringRef row,
                                    StringRef bdNum) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "dma_tile_" << col << "_" << row << "_bd_" << bdNum << "_tensor";
  return str;
}

static std::string tileDMATensorStr(int col, int row, int bdNum) {
  return tileDMATensorStr(std::to_string(col), std::to_string(row),
                          std::to_string(bdNum));
}

void generateXAieDmaSetMultiDimAddr(raw_ostream &output, int ndims,
                                    ArrayRef<BDDimLayoutAttr> dims, int col,
                                    int row, int bdNum, int baseAddrA,
                                    int offsetA, int lenA, int bytesA,
                                    const char *errorRetval) {
  std::string tensor = tileDMATensorStr(col, row, bdNum);
  output << "XAie_DmaTensor " << tensor << " = {};\n";
  output << tensor << ".NumDim = " << std::to_string(ndims) << ";\n";
  output << tensor
         << ".Dim ="
            "__mlir_aie_alloc_dim_desc("
         << std::to_string(ndims) << ");\n";
  output << "if(NULL == " << tensor << ".Dim){\n"
         << "  return " << errorRetval << ";\n"
         << "}\n";
  for (int i = 0; i < ndims; i++) {
    // Pass down dimensions in reverse order; in the MLIR, this allows us
    // to specify strides/sizes in the same order as we would access a
    // multi-dim C array, with the highest dimension first.
    int j = ndims - i - 1;
    // Assume AIE-ML architecture; we assert this above
    output << tensor << ".Dim[" << std::to_string(j) << "].AieMlDimDesc"
           << " = { /* StepSize */ " << std::to_string(dims[i].getStride())
           << ", /* Size */ " << std::to_string(dims[i].getSize()) << "};\n";
  }
  output << "__mlir_aie_try(XAie_DmaSetMultiDimAddr("
         << tileDMAInstRefStr(col, row, bdNum) << ", "
         << "&" << tensor << ", "
         << "0x" << llvm::utohexstr(baseAddrA + offsetA) << ", "
         << " /* len */ " << lenA << " * " << bytesA << "));\n";
  // TODO: Probably need special handling for NOC
  // TODO: Might need to adjust strides / sizes by -1
}

} // namespace xilinx::AIE
