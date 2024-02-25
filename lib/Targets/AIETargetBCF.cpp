//===- AIETargetBCF.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

std::string utohexstr(uint32_t u) { return "0x" + llvm::utohexstr(u); }

namespace xilinx {
namespace AIE {

LogicalResult AIETranslateToBCF(ModuleOp module, raw_ostream &output,
                                int tileCol, int tileRow) {
  DenseMap<TileID, Operation *> tiles;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;

  if (module.getOps<DeviceOp>().empty())
    module.emitOpError("expected aie.device operation at toplevel");
  DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

  collectTiles(targetOp, tiles);
  collectBuffers(targetOp, buffers);

  // _entry_point _main_init
  // _symbol      _main _after _main_init
  // _symbol      _main_init 0
  // _reserved DMb      0x00000 0x20000
  // _symbol   a        0x38000 0x2000
  // _extern   a
  // _stack    DM_stack 0x20000  0x400 //stack for core
  // _reserved DMb 0x40000 0xc0000 // And everything else the core can't
  // see
  // // Include all symbols from rom.c
  // _include _file rom.o
  for (auto tile : targetOp.getOps<TileOp>())
    if (tile.colIndex() == tileCol && tile.rowIndex() == tileRow) {
      const auto &targetModel = getTargetModel(tile);
      TileID srcCoord = {tile.colIndex(), tile.rowIndex()};

      std::string corefunc = std::string("core_") +
                             std::to_string(tile.getCol()) + "_" +
                             std::to_string(tile.getRow());
      output << "_entry_point _main_init\n";
      output << "_symbol " << corefunc << " _after _main_init\n";
      output << "_symbol _main_init 0\n";
      std::string initReserved = (targetModel.getTargetArch() == AIEArch::AIE2)
                                     ? "0x40000"
                                     : "0x20000";
      output << "_reserved DMb 0x00000 " << initReserved
             << " // Don't put data in code memory\n";

      int stacksize = 0;
      if (auto core = tile.getCoreOp())
        stacksize = core.getStackSize();
      output << "_stack DM_stack "
             << utohexstr(targetModel.getMemInternalBaseAddress(srcCoord))
             << " " << utohexstr(stacksize) << " // stack for core\n";

      auto doBuffer = [&](std::optional<TileID> tile, int offset,
                          const std::string &dir) {
        if (tile) {
          output << "// " + dir +
                        " -------------------------------------------------\n";
          uint32_t localMemSize = targetModel.getLocalMemorySize();
          if (tile != srcCoord)
            output << "_reserved DMb " << utohexstr(offset) << " "
                   << utohexstr(localMemSize) << " "
                   << " // Don't allocate variables in " << dir
                   << " neighbor\n\n";
          // TODO How to set as reserved if no buffer exists (or reserve
          // remaining buffer)
          if (tiles.count(*tile)) {
            for (auto buf : buffers[tiles[*tile]]) {
              std::string bufName(buf.name().getValue());
              int bufferBaseAddr = getBufferBaseAddress(buf);
              int numBytes = buf.getAllocationSize();
              if (buf.getInitialValue() && tile == srcCoord) {
                output << "_overlay " << bufName << " "
                       << utohexstr(offset + bufferBaseAddr) << " // "
                       << numBytes << " bytes\n";
              } else {
                output << "_symbol " << bufName << " "
                       << utohexstr(offset + bufferBaseAddr) << " " << numBytes
                       << '\n';
                output << "_extern " << bufName << "\n";
                output << "_reserved DMb " << utohexstr(offset + bufferBaseAddr)
                       << " " << numBytes << '\n';
              }
              output << "\n";
            }
          }
        } else {
          uint32_t localMemSize = targetModel.getLocalMemorySize();
          output << "_reserved DMb " << utohexstr(offset) << " "
                 << utohexstr(localMemSize) << " "
                 << " // No tile with memory exists to the " << dir << ".\n";
        }
      };

      output << "\n// mapping neighbors tile memory\n";
      doBuffer(targetModel.getMemSouth(srcCoord),
               targetModel.getMemSouthBaseAddress(), std::string("south"));
      doBuffer(targetModel.getMemWest(srcCoord),
               targetModel.getMemWestBaseAddress(), std::string("west"));
      doBuffer(targetModel.getMemNorth(srcCoord),
               targetModel.getMemNorthBaseAddress(), std::string("north"));
      doBuffer(targetModel.getMemEast(srcCoord),
               targetModel.getMemEastBaseAddress(), std::string("east"));
      output << "// end mapping neighbors tile memory\n\n";

      if (targetModel.getTargetArch() == AIEArch::AIE2) {
        output << "_reserved DMb 0x80000 0x80000 // And everything else "
                  "the core can't see\n";
      } else {
        output << "_reserved DMb 0x40000 0xc0000 // And everything else "
                  "the core can't see\n";
      }
      if (tile.getCoreOp() && tile.getCoreOp().getLinkWith())
        output << "_include _file "
               << tile.getCoreOp().getLinkWith().value().str() << "\n";
      output << "_resolve _main core_" << tile.getCol() << "_" << tile.getRow()
             << "\n";
    }

  return success();
}
} // namespace AIE
} // namespace xilinx
