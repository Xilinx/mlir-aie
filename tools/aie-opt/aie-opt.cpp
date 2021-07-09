// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

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

  registerAllPasses();
  xilinx::AIE::registerAIEAssignBufferAddressesPass();
  xilinx::AIE::registerAIECoreToLLVMPass();
  xilinx::AIE::registerAIECoreToStandardPass();
  xilinx::AIE::registerAIECreateCoresPass();
  xilinx::AIE::registerAIECreateLocksPass();
  xilinx::AIE::registerAIEFindFlowsPass();
  xilinx::AIE::registerAIEHerdRoutingPass();
  xilinx::AIE::registerAIELowerMemcpyPass();
  xilinx::AIE::registerAIENormalizeAddressSpacesPass();
  xilinx::AIE::registerAIERouteFlowsPass();
  xilinx::AIE::registerAIERoutePacketFlowsPass();
  xilinx::AIE::registerAIEVectorOptPass();

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<scf::SCFDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<xilinx::AIE::AIEDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
