//===- AIETargetLdScript.cpp -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Targets/AIETargets.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// Output the memorymap in gnu linker format for the given buffer operations,
// with the given offset. The offset is different depending on where the buffers
// are accessed from.
static void writeLDScriptMap(raw_ostream &output, BufferOp buf, int offset) {
  std::string bufName(buf.name().getValue());
  int bufferBaseAddr = getBufferBaseAddress(buf);
  int numBytes = buf.getAllocationSize();
  output << ". = 0x" << llvm::utohexstr(offset + bufferBaseAddr) << ";\n";
  output << bufName << " = .;\n";
  output << ". += 0x" << llvm::utohexstr(numBytes) << ";\n";
}

///// ld.script format:
//
// MEMORY
// {
//    program (RX) : ORIGIN = 0, LENGTH = 0x0020000
//    data (!RX) : ORIGIN = 0x20000, LENGTH = 0x0020000
// }
// ENTRY(_main_init)
// INPUT(something.o)
// SECTIONS
// {
//   . = 0x0;
//   .text : {
//      // the _main_init symbol from me_basic.o has to come at address zero.
//      *me_basic.o(.text)
//      . = 0x200;
//      __ctors_start__ = .;
//      __init_array_start = .;
//      KEEP(SORT(*)(.init_array))
//      __ctors_end__ = .;
//      __init_array_end = .;
//      __dtors_start__ = .;
//      __dtors_end__ = .;
//      *(.text)
//   } > program
//   .data : { *(.data) } > data
//   . = 0x20000;
//   _sp_start_value_DM_stack = .;
//   . = 0x24000;
//   a = .;
//   . += 1024;
//   .bss : { *(.bss) } > data
// }
LogicalResult xilinx::AIE::AIETranslateToLdScript(ModuleOp module,
                                                  raw_ostream &output,
                                                  int tileCol, int tileRow) {
  DenseMap<TileID, Operation *> tiles;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;

  if (module.getOps<DeviceOp>().empty()) {
    module.emitOpError("expected AIE.device operation at toplevel");
  }
  DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

  collectTiles(targetOp, tiles);
  collectBuffers(targetOp, buffers);

  for (auto tile : targetOp.getOps<TileOp>())
    if (tile.colIndex() == tileCol && tile.rowIndex() == tileRow) {
      TileID srcCoord = {tile.colIndex(), tile.rowIndex()};
      const auto &targetModel = getTargetModel(tile);

      // Figure out how much memory we have left for random allocations
      auto core = tile.getCoreOp();
      int max = core.getStackSize();
      for (auto buf : buffers[tiles[srcCoord]]) {
        int bufferBaseAddr = getBufferBaseAddress(buf);
        int numBytes = buf.getAllocationSize();
        max = std::max(max, bufferBaseAddr + numBytes);
      }
      int origin = targetModel.getMemInternalBaseAddress(srcCoord) + max;
      int length = targetModel.getLocalMemorySize() - max;
      output << R"THESCRIPT(
MEMORY
{
   program (RX) : ORIGIN = 0, LENGTH = 0x0020000
)THESCRIPT";
      output << "   data (!RX) : ORIGIN = 0x" << llvm::utohexstr(origin)
             << ", LENGTH = 0x" << llvm::utohexstr(length);
      output << R"THESCRIPT(
}
ENTRY(_main_init)
SECTIONS
{
  . = 0x0;
  .text : {
     /* the _main_init symbol has to come at address zero. */
     *crt0.o(.text)
     . = 0x200;
     _ctors_start = .;
     _init_array_start = .;
     KEEP(SORT(*.init_array))
     _ctors_end = .;
     _init_array_end = .;
     _dtors_start = .;
     _dtors_end = .;
     *(.text)
  } > program
  .data : {
     *(.data*);
     *(.rodata*)
  } > data
)THESCRIPT";
      auto doBuffer = [&](std::optional<TileID> tile, int offset,
                          std::string dir) {
        if (tile) {
          if (tiles.count(*tile))
            for (auto buf : buffers[tiles[*tile]])
              writeLDScriptMap(output, buf, offset);
        } else {
          output << "/* No tile with memory exists to the " << dir << ". */\n";
          output << ". = 0x" << llvm::utohexstr(offset) << ";\n";
          uint32_t localMemSize = targetModel.getLocalMemorySize();
          output << ". += 0x" << llvm::utohexstr(localMemSize) << ";\n";
        }
      };

      // Stack
      output << ". = 0x"
             << llvm::utohexstr(targetModel.getMemInternalBaseAddress(srcCoord))
             << ";\n";
      output << "_sp_start_value_DM_stack = .;\n";

      if (auto core = tile.getCoreOp())
        output << ". += 0x" << llvm::utohexstr(core.getStackSize())
               << "; /* stack */\n";
      else
        output << "/* no stack allocated */\n";

      doBuffer(targetModel.getMemSouth(srcCoord),
               targetModel.getMemSouthBaseAddress(), std::string("south"));
      doBuffer(targetModel.getMemWest(srcCoord),
               targetModel.getMemWestBaseAddress(), std::string("west"));
      doBuffer(targetModel.getMemNorth(srcCoord),
               targetModel.getMemNorthBaseAddress(), std::string("north"));
      doBuffer(targetModel.getMemEast(srcCoord),
               targetModel.getMemEastBaseAddress(), std::string("east"));

      output << "  .bss : { *(.bss) } > data\n";
      output << "  .bss.DMb.4 : { *(.bss.DMb.4) } > data\n";
      output << "}\n";
      if (auto coreOp = tile.getCoreOp()) {
        if (auto fileAttr = coreOp.getLinkWith())
          output << "INPUT(" << fileAttr.value().str() << ")\n";

        output << "PROVIDE(main = core_" << tile.getCol() << "_"
               << tile.getRow() << ");\n";
      }
    }
  return success();
}
