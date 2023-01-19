
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"



int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  // registerMyDialects(registry);
  //registerMyPasses();
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
