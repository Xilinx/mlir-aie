// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace xilinx {
namespace AIE {
mlir::LogicalResult AIETranslateToXAIEV1(mlir::ModuleOp module, llvm::raw_ostream &output);
mlir::LogicalResult AIEFlowsToJSON(mlir::ModuleOp module, llvm::raw_ostream &output);
}
}
