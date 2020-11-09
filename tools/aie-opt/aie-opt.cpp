// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

//#include "mlir/Analysis/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "AIEDialect.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
//  enableGlobalDialectRegistry(true);
//  registerAllDialects();
  registerAllPasses();
  xilinx::AIE::registerAIEFindFlowsPass();
  xilinx::AIE::registerAIECreateFlowsPass();
  xilinx::AIE::registerAIECreateCoresPass();
  xilinx::AIE::registerAIECreateLocksPass();
  xilinx::AIE::registerAIECoreToLLVMPass();
  xilinx::AIE::registerAIEHerdRoutingPass();
  xilinx::AIE::registerAIECreatePacketFlowsPass();
  xilinx::AIE::registerAIELowerMemcpyPass();

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<scf::SCFDialect>();
  registry.insert<xilinx::AIE::AIEDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
