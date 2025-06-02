#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "json.hpp"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

#define DEBUG_TYPE "aie-apply-coords"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using json = nlohmann::json;

struct AIEPlaceTilesPass : public AIEPlaceTilesBase<AIEPlaceTilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();

    std::ifstream jsonFile("netlist.json");
    if (!jsonFile.is_open()) {
        llvm::errs() << "Failed to open netlist.json\n";
        return;
    }

    json input = json::parse(jsonFile);

    auto tileOps = llvm::to_vector(device.getOps<TileOp>());

    for (size_t i = 0; i < tileOps.size(); ++i) {
      auto tileOp = tileOps[i];
      auto node = input["nodes"][i];

      int col = node["col_id"];
      int row = node["row_id"];

      // Replace attributes
      tileOp->setAttr("col", IntegerAttr::get(IntegerType::get(tileOp.getContext(), 32), col));
      tileOp->setAttr("row", IntegerAttr::get(IntegerType::get(tileOp.getContext(), 32), row));
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPlaceTilesPass() {
  return std::make_unique<AIEPlaceTilesPass>();
}
