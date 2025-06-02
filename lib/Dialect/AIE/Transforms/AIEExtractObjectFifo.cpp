#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "json.hpp"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorOr.h"

#define DEBUG_TYPE "aie-extract-fifo"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using json = nlohmann::ordered_json;

struct AIEExtractObjectFifoPass
    : AIEExtractObjectFifoBase<AIEExtractObjectFifoPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    int id = 0;

    std::error_code ec;
    llvm::raw_fd_ostream nFile("netlist.json", ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "Could not open netlist.json: " << ec.message() << "\n";
      return;
    }

    json output;
    output["nodes"] = json::array();
    output["nets"] = json::array();

    llvm::DenseMap<Value, int> tileIdMap;
    for (auto tileOp : device.getOps<TileOp>()) {
      int row = tileOp.getRow(), col = tileOp.getCol();
      std::string type = "Core";
      if (row == 0)
        type = "Shim";
      else if (row == 1)
        type = "Mem";

      json node = {
          {"id", id},
          {"type", type},
          {"col_id", col},
          {"row_id", row}};
      output["nodes"].push_back(node);
      tileIdMap[tileOp.getResult()] = id;
      ++id;
    }

    for (auto objectFifo : device.getOps<ObjectFifoCreateOp>()) {
      int sId = tileIdMap[objectFifo.getProducerTile()];
      std::vector<int> dIds;
      for (auto cTile : objectFifo.getConsumerTiles()) {
        dIds.push_back(tileIdMap[cTile]);
      }

      // Extract the memref element type
      auto memrefTy = objectFifo.getElemType().getElementType();

      // Compute number of elements in the memref shape
      int64_t shapeProduct = 1;
      for (auto dim : memrefTy.getShape())
        shapeProduct *= dim;

      // Compute size in bytes
      int64_t bits = memrefTy.getElementType().getIntOrFloatBitWidth();
      int64_t byteSize = shapeProduct * (bits / 8);

      std::vector<int64_t> depths;
      auto elemAttr = objectFifo.getElemNumberAttr();
      if (auto intAttr = dyn_cast<IntegerAttr>(elemAttr)) {
        depths.push_back(intAttr.getInt() * byteSize);
      } else if (auto arrayAttr = dyn_cast<ArrayAttr>(elemAttr)) {
        for (auto val : arrayAttr.getValue()) {
          depths.push_back(cast<IntegerAttr>(val).getInt() * byteSize);
        }
      } else {
        objectFifo.emitError("Unsupported elemNumber format");
        return;
      }

      json net = {
          {"src_id", sId},
          {"dst_ids", dIds},
          {"data_size", depths}};
      output["nets"].push_back(net);
    }

    nFile << output.dump(2);
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEExtractObjectFifoPass() {
  return std::make_unique<AIEExtractObjectFifoPass>();
}
