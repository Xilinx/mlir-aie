#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx::AIE;

namespace {

struct AIESequentialPlacerPass
    : AIESequentialPlacerBase<AIESequentialPlacerPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    // get the target model for the device, then make a vector of (row, col)
    // pairs, one for each core tile
    std::vector<std::pair<int, int>> computeTileLocations;
    uint32_t computeIdx = 0;
    auto &targetModel = device.getTargetModel();
    for (int col = 0; col < targetModel.columns(); ++col) {
      for (int row = 0; row < targetModel.rows(); ++row) {
        if (targetModel.isCoreTile(col, row)) {
          computeTileLocations.push_back({col, row});
        }
      }
    }

    for (auto tile : device.getOps<TileOp>()) {
      // if the row or column are -1, place the Tile
      if (tile.getRow() == -1 && tile.getCol() == -1) {
        // get the next available row and column
        // set the row and column of the TileOp
        if (computeIdx >= computeTileLocations.size()) {
          tile.emitError("Not enough compute tiles available for placement");
          return;
        }
        auto [col, row] = computeTileLocations[computeIdx++];
        tile.setCol(col);
        tile.setRow(row);
      }
    }

    for (auto fifo : device.getOps<ObjectFifoCreateOp>()) {
      // Check if the source and destination tiles are placed
      TileOp srcTile = fifo.getProducerTile().getDefiningOp<TileOp>();
      auto dstTiles = fifo.getConsumerTiles();
      int common_col = 0;
      if (srcTile.getRow() >= 0 && srcTile.getCol() < 0) {
        // If the source tile is not placed, place it at (0, 0)
        srcTile.setCol(common_col);
      }
      for (Value _dstTile : dstTiles) {
        TileOp dstTile = _dstTile.getDefiningOp<TileOp>();
        if (dstTile.getRow() >= 0 && dstTile.getCol() < 0) {
          // If the source tile is not placed, place it at (0, 0)
          dstTile.setCol(common_col);
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
xilinx::AIE::createAIESequentialPlacerPass() {
  return std::make_unique<AIESequentialPlacerPass>();
}
