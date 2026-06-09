//===- AIETargets.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGETS_AIETARGETS_H
#define AIE_TARGETS_AIETARGETS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xilinx {
namespace AIE {

// One entry in the sidecar location map produced by AIETranslateNpuToBinary
// (and AIETranslateControlPacketsToUI32Vec). Each entry describes one
// transaction operation in the .bin: the byte range it occupies, its opcode,
// the absolute device address it touches (when applicable), the originating
// aiex.npu.* op's MLIR Location, and an optional regdb register name resolved
// from (address, tile-module). The vector is written alongside the .bin as a
// JSON sidecar (see emitNpuLocmapJSON) so consumers can correlate transaction
// words back to the source op's MLIR Location.
struct TxnLocEntry {
  uint32_t byteOffset = 0;
  uint32_t byteSize = 0;
  std::string opcodeName;          // "WRITE32", "BLOCKWRITE", etc.
  std::string sourceOpName;        // "aiex.npu.write32", etc.
  std::optional<uint64_t> address; // absolute device address when applicable
  std::string registerName; // regdb-resolved register name; empty if unknown
  std::string
      registerModule; // regdb module: "core", "memory", "memory_tile", "shim"
  std::optional<mlir::Location> loc;
};

// Serialize a vector of TxnLocEntry as JSON to `output`, with the given
// device name and binary file basename for the JSON header.
void emitNpuLocmapJSON(llvm::raw_ostream &output, llvm::StringRef deviceName,
                       llvm::StringRef binaryName,
                       const std::vector<TxnLocEntry> &locmap);

mlir::LogicalResult AIETranslateToXAIEV2(mlir::ModuleOp module,
                                         llvm::raw_ostream &output,
                                         llvm::StringRef deviceName = "");
mlir::LogicalResult AIETranslateToHSA(mlir::ModuleOp module,
                                      llvm::raw_ostream &output,
                                      llvm::StringRef deviceName = "");
mlir::LogicalResult AIEFlowsToJSON(mlir::ModuleOp module,
                                   llvm::raw_ostream &output,
                                   llvm::StringRef deviceName = "");
mlir::LogicalResult ADFGenerateCPPGraph(mlir::ModuleOp module,
                                        llvm::raw_ostream &output);
mlir::LogicalResult AIETranslateSCSimConfig(mlir::ModuleOp module,
                                            llvm::raw_ostream &output,
                                            llvm::StringRef deviceName = "");
mlir::LogicalResult AIETranslateShimSolution(mlir::ModuleOp module,
                                             llvm::raw_ostream &,
                                             llvm::StringRef deviceName = "");
mlir::LogicalResult AIETranslateGraphXPE(mlir::ModuleOp module,
                                         llvm::raw_ostream &, llvm::StringRef);
mlir::LogicalResult
AIETranslateNpuToBinary(mlir::ModuleOp, std::vector<uint32_t> &,
                        llvm::StringRef deviceName = "",
                        llvm::StringRef sequenceName = "",
                        std::vector<TxnLocEntry> *locmap = nullptr);
mlir::LogicalResult AIETranslateToUcDma(mlir::ModuleOp module,
                                        llvm::raw_ostream &output);
mlir::LogicalResult AIETranslateToUcDma(mlir::ModuleOp, std::string &assembly);
mlir::LogicalResult
AIETranslateControlPacketsToUI32Vec(mlir::ModuleOp, std::vector<uint32_t> &,
                                    llvm::StringRef deviceName = "",
                                    llvm::StringRef sequenceName = "",
                                    std::vector<TxnLocEntry> *locmap = nullptr);
mlir::LogicalResult AIETranslateToLdScript(mlir::ModuleOp module,
                                           llvm::raw_ostream &output,
                                           int tileCol, int tileRow,
                                           llvm::StringRef deviceName = "");
mlir::LogicalResult AIETranslateToBCF(mlir::ModuleOp module,
                                      llvm::raw_ostream &output, int tileCol,
                                      int tileRow,
                                      llvm::StringRef deviceName = "");
mlir::LogicalResult
AIELLVMLink(llvm::raw_ostream &output, std::vector<std::string> Files,
            bool DisableDITypeMap = false, bool NoVerify = false,
            bool Internalize = false, bool OnlyNeeded = false,
            bool PreserveAssemblyUseListOrder = false, bool Verbose = false);

mlir::LogicalResult AIETranslateToCDODirect(
    mlir::ModuleOp m, llvm::StringRef workDirPath, llvm::StringRef deviceName,
    bool bigEndian = false, bool emitUnified = false, bool cdoDebug = false,
    bool aieSim = false, bool xaieDebug = false, bool enableCores = true);

mlir::LogicalResult AIETranslateToTargetArch(mlir::ModuleOp module,
                                             llvm::raw_ostream &output,
                                             llvm::StringRef deviceName);

} // namespace AIE

namespace aievec {

/// Translates the AIE vector dialect MLIR to C++ code.
mlir::LogicalResult translateAIEVecToCpp(mlir::Operation *op, bool aie2,
                                         mlir::raw_ostream &os);

} // namespace aievec
} // namespace xilinx

#endif
