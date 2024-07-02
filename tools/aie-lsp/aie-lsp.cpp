
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  registry.insert<scf::SCFDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<xilinx::AIE::AIEDialect>();
  registry.insert<xilinx::aievec::AIEVecDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  // registerMyDialects(registry);
  //registerMyPasses();
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
