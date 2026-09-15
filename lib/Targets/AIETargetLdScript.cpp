//===- AIETargetLdScript.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIECoreMemory.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Targets/AIETargetShared.h"
#include "aie/Targets/AIETargets.h"

#include <optional>

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

  for (auto tile : targetOp.getOps<TileOp>()) {
    if (tile.colIndex() == tileCol && tile.rowIndex() == tileRow) {
      TileID srcCoord = {tile.colIndex(), tile.rowIndex()};
      // The "data" region below holds the sections the core compiler generates
      // itself (.data, .rodata and .bss) and that nothing places explicitly. It
      // is one contiguous region, so its size bounds what the core can link,
      // and the total free memory on the tile does not.
      const auto &targetModel = getTargetModel(tile);
      MemoryRun dataRun = coreDataRegion(tile, buffers[tiles[srcCoord]]);

      // A pass that adds or moves a buffer after the allocator ran leaves the
      // region aliasing it, and the core compiler and that buffer would then
      // write over each other. Report the alias here.
      for (auto buf : buffers[tiles[srcCoord]]) {
        if (buf.getCoreData() || buf.getAllocationSize() == 0 ||
            dataRun.size == 0) {
          continue;
        }
        int64_t bufStart = getBufferBaseAddress(buf);
        int64_t bufEnd = bufStart + buf.getAllocationSize();
        if (bufStart < dataRun.end() && dataRun.start < bufEnd) {
          return tile.emitOpError("data region 0x")
                 << llvm::utohexstr(dataRun.start) << "-0x"
                 << llvm::utohexstr(dataRun.end() - 1) << " overlaps buffer '"
                 << buf.name().getValue() << "' at 0x"
                 << llvm::utohexstr(bufStart)
                 << "; the buffer allocator's placement is stale. Re-run "
                    "--aie-assign-buffer-addresses";
        }
      }

      // Was hardcoded to 0x20000 -- eight times the real 0x4000 -- which let
      // an overflowing core link cleanly and fail much later in aie-rt's ELF
      // loader instead of here, at the linker, naming the section.
      int origin =
          targetModel.getMemInternalBaseAddress(srcCoord) + dataRun.start;
      int length = dataRun.size;
      output << R"THESCRIPT(
MEMORY
{
)THESCRIPT";
      output << "   program (RX) : ORIGIN = 0, LENGTH = 0x"
             << llvm::utohexstr(targetModel.getProgramMemorySize()) << "\n";
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
                          const std::string &dir) {
        if (tile) {
          if (tiles.count(*tile)) {
            for (auto buf : buffers[tiles[*tile]]) {
              if (!buf.getCoreData()) {
                writeLDScriptMap(output, buf, offset);
              }
            }
          }
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

      if (auto core = tile.getCoreOp()) {
        output << ". += 0x" << llvm::utohexstr(core.getEffectiveStackSize())
               << "; /* stack */\n";
      } else {
        output << "/* no stack allocated */\n";
      }

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
        // `link_files` holds the ordinary final-link inputs (object files)
        if (auto filesAttr = coreOp.getLinkFiles()) {
          // Canonical path: link_files populated by aie-assign-core-link-files.
          for (auto f : filesAttr->getAsRange<mlir::StringAttr>()) {
            output << "INPUT(" << f.getValue() << ")\n";
          }
        } else if (auto fileAttr = coreOp.getLinkWith()) {
          // Deprecated fallback: core-level link_with was not migrated by
          // aie-assign-core-link-files (e.g., the pass was not run). It carries
          // no mode, so it is always an ordinary link input.
          output << "INPUT(" << fileAttr.value().str() << ")\n";
        }

        output << "PROVIDE(main = core_" << tile.getCol() << "_"
               << tile.getRow() << ");\n";
      }
    }
  }
  return success();
}
