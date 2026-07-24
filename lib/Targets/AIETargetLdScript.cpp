//===- AIETargetLdScript.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// ENTRY(__start)
// INPUT(something.o)
// SECTIONS
// {
//   . = 0x0;
//   .text : {
//      // the __start symbol has to come at address zero.
//      *crt0.o(.text*)
//      __ctors_start__ = .;
//      __init_array_start = .;
//      KEEP(SORT(*)(.init_array))
//      __ctors_end__ = .;
//      __init_array_end = .;
//      __dtors_start__ = .;
//      __dtors_end__ = .;
//      *(.text*)
//   } > program
//   .data : { *(.data*) } > data
//   . = 0x20000;
//   _sp_start_value_DM_stack = .;
//   . = 0x24000;
//   a = .;
//   . += 1024;
//   .bss : { *(.bss*) } > data
// }
// PROVIDE(main = core_3_3);

LogicalResult xilinx::AIE::AIETranslateToLdScript(ModuleOp module,
                                                  raw_ostream &output,
                                                  int tileCol, int tileRow,
                                                  llvm::StringRef deviceName) {
  DenseMap<TileID, Operation *> tiles;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;

  DeviceOp targetOp =
      AIE::DeviceOp::getForSymbolInModuleOrError(module, deviceName);

  if (!targetOp) {
    return failure();
  }

  collectTiles(targetOp, tiles);
  collectBuffers(targetOp, buffers);

  for (auto tile : targetOp.getOps<TileOp>())
    if (tile.colIndex() == tileCol && tile.rowIndex() == tileRow) {
      TileID srcCoord = {tile.colIndex(), tile.rowIndex()};
      const auto &targetModel = getTargetModel(tile);

      // Figure out how much memory we have left for compiler-generated
      // sections (.data/.rodata/.bss) that are not explicitly placed; these are
      // emitted into the "data" region below. Buffers are placed by the
      // buffer-address allocator, which (in bank-aware mode) can leave the free
      // space fragmented -- pick the largest free gap across the stack and this
      // tile's buffers within the tile's local memory.
      auto core = tile.getCoreOp();
      int localMemSize = targetModel.getLocalMemorySize();

      // Collect occupied [start, end) intervals in tile-local coordinates: the
      // stack sits at the bottom of memory, followed by the placed buffers.
      SmallVector<std::pair<int, int>, 8> occupied;
      occupied.push_back({0, core.getStackSize()});
      for (auto buf : buffers[tiles[srcCoord]]) {
        int bufferBaseAddr = getBufferBaseAddress(buf);
        int numBytes = buf.getAllocationSize();
        occupied.push_back({bufferBaseAddr, bufferBaseAddr + numBytes});
      }
      std::sort(occupied.begin(), occupied.end());

      // Sweep the intervals to find the largest free gap not covered by any of
      // them within [0, localMemSize).
      int bestGapStart = 0;
      int bestGapLen = 0;
      int cursor = 0;
      auto considerGap = [&](int gapStart, int gapEnd) {
        if (gapEnd - gapStart > bestGapLen) {
          bestGapLen = gapEnd - gapStart;
          bestGapStart = gapStart;
        }
      };
      for (auto &iv : occupied) {
        if (iv.first > cursor)
          considerGap(cursor, iv.first);
        cursor = std::max(cursor, iv.second);
      }
      // Trailing gap above the highest occupied address.
      if (cursor < localMemSize)
        considerGap(cursor, localMemSize);

      int origin =
          targetModel.getMemInternalBaseAddress(srcCoord) + bestGapStart;
      int length = bestGapLen;
      output << R"THESCRIPT(
MEMORY
{
   program (RX) : ORIGIN = 0, LENGTH = 0x0020000
)THESCRIPT";
      output << "   data (!RX) : ORIGIN = 0x" << llvm::utohexstr(origin)
             << ", LENGTH = 0x" << llvm::utohexstr(length);
      output << R"THESCRIPT(
}
ENTRY(__start)
SECTIONS
{
  . = 0x0;
  .text : {
     /* the __start symbol has to come at address zero. */
     *crt0.o(.text*)
     _ctors_start = .;
     _init_array_start = .;
     KEEP(SORT(*.init_array))
     _ctors_end = .;
     _init_array_end = .;
     _dtors_start = .;
     _dtors_end = .;
     *(.text*)
  } > program
  .data : {
     *(.data*)
     *(.rodata*)
  } > data
  .comment : {
     *(.comment*)
  }
  .symtab : {
     *(.symtab)
  }
  .shstrtab : {
     *(.shstrtab)
  }
  .strtab : {
     *(.strtab)
  }
  .stack_sizes : {
     *(.stack_sizes)
  }

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

      output << "  .bss : { *(.bss*) } > data\n";
      // INPUT() directives must follow the closing brace of SECTIONS; placing
      // them inside SECTIONS is invalid linker script syntax.
      output << "}\n";
      if (auto coreOp = tile.getCoreOp()) {
        if (auto filesAttr = coreOp.getLinkFiles()) {
          // Canonical path: link_files populated by aie-assign-core-link-files.
          for (auto f : filesAttr->getAsRange<mlir::StringAttr>())
            output << "INPUT(" << f.getValue() << ")\n";
        } else if (auto fileAttr = coreOp.getLinkWith()) {
          // Deprecated fallback: core-level link_with was not migrated by
          // aie-assign-core-link-files (e.g., the pass was not run).
          output << "INPUT(" << fileAttr.value().str() << ")\n";
        }

        output << "PROVIDE(main = core_" << tile.getCol() << "_"
               << tile.getRow() << ");\n";
      }
    }
  return success();
}
