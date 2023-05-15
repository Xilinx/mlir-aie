//===- AIETargets.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "AIETargets.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/AIE/AIENetlistAnalysis.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::vector;
using namespace xilinx;
using namespace xilinx::AIE;

static llvm::cl::opt<int>
    tileCol("tilecol", llvm::cl::desc("column coordinate of core to translate"),
            llvm::cl::init(0));
static llvm::cl::opt<int>
    tileRow("tilerow", llvm::cl::desc("row coordinate of core to translate"),
            llvm::cl::init(0));

namespace xilinx {
namespace AIE {

static void registerDialects(DialectRegistry &registry) {
  registry.insert<xilinx::AIE::AIEDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<DLTIDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<VectorDialect>();
  registry.insert<LLVM::LLVMDialect>();
}

// Output the buffer map for the given buffer operations, with the given offset.
// The offset is different depending on where the buffers are accessed from.
void writeBufferMap(raw_ostream &output, BufferOp buf, int offset,
                    NetlistAnalysis &NL) {
  std::string bufName(buf.name().getValue());
  int bufferBaseAddr = NL.getBufferBaseAddress(buf);
  int numBytes = buf.getAllocationSize();
  output << "_symbol " << bufName << " "
         << "0x" << llvm::utohexstr(offset + bufferBaseAddr) << " " << numBytes
         << '\n';
}
// Output the memorymap in BCF format for the given buffer operations, with the
// given offset. The offset is different depending on where the buffers are
// accessed from.
void writeBCFMap(raw_ostream &output, BufferOp buf, int offset,
                 NetlistAnalysis &NL) {
  std::string bufName(buf.name().getValue());
  int bufferBaseAddr = NL.getBufferBaseAddress(buf);
  int numBytes = buf.getAllocationSize();
  output << "_symbol " << bufName << " "
         << "0x" << llvm::utohexstr(offset + bufferBaseAddr) << " "
         << "0x" << llvm::utohexstr(numBytes) << '\n';
  output << "_extern " << bufName << "\n";
  output << "_reserved DMb "
         << "0x" << llvm::utohexstr(offset + bufferBaseAddr) << " "
         << "0x" << llvm::utohexstr(numBytes) << '\n';
}
// Output the memorymap in gnu linker format for the given buffer operations,
// with the given offset. The offset is different depending on where the buffers
// are accessed from.
void writeLDScriptMap(raw_ostream &output, BufferOp buf, int offset,
                      NetlistAnalysis &NL) {
  std::string bufName(buf.name().getValue());
  int bufferBaseAddr = NL.getBufferBaseAddress(buf);
  int numBytes = buf.getAllocationSize();
  output << ". = 0x" << llvm::utohexstr(offset + bufferBaseAddr) << ";\n";
  output << bufName << " = .;\n";
  output << ". += 0x" << llvm::utohexstr(numBytes) << ";\n";
}

void registerAIETranslations() {
  TranslateFromMLIRRegistration registrationMMap(
      "aie-generate-mmap", "Generate AIE memory map",
      [](ModuleOp module, raw_ostream &output) {
        DenseMap<std::pair<int, int>, Operation *> tiles;
        DenseMap<Operation *, CoreOp> cores;
        DenseMap<Operation *, MemOp> mems;
        DenseMap<std::pair<Operation *, int>, LockOp> locks;
        DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;
        DenseMap<Operation *, SwitchboxOp> switchboxes;

        if (module.getOps<DeviceOp>().empty()) {
          module.emitOpError("expected AIE.device operation at toplevel");
        }
        DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

        NetlistAnalysis NL(targetOp, tiles, cores, mems, locks, buffers,
                           switchboxes);
        NL.collectTiles(tiles);
        NL.collectBuffers(buffers);

        for (auto tile : tiles) {
          Operation *srcTileOp = tile.second;
          std::pair<int, int> srcCoord = NL.getCoord(srcTileOp);
          int srcCol = srcCoord.first;
          int srcRow = srcCoord.second;

          output << "// Tile(" << srcCol << ", " << srcRow << ")\n";
          output << "// Memory map: name base_address num_bytes\n";

          auto doBuffer = [&](Optional<TileID> tile, int offset) {
            if (tiles.count(*tile))
              for (auto buf : buffers[tiles[*tile]])
                writeBufferMap(output, buf, offset, NL);
          };

          const auto &target_model = xilinx::AIE::getTargetModel(srcTileOp);

          if (auto tile = target_model.getMemSouth(srcCoord))
            doBuffer(tile, target_model.getMemSouthBaseAddress());
          if (auto tile = target_model.getMemWest(srcCoord))
            doBuffer(tile, target_model.getMemWestBaseAddress());
          if (auto tile = target_model.getMemNorth(srcCoord))
            doBuffer(tile, target_model.getMemNorthBaseAddress());
          if (auto tile = target_model.getMemEast(srcCoord))
            doBuffer(tile, target_model.getMemEastBaseAddress());
        }
        return success();
      },
      registerDialects);

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

  TranslateFromMLIRRegistration registrationLDScript(
      "aie-generate-ldscript", "Generate AIE loader script",
      [](ModuleOp module, raw_ostream &output) {
        DenseMap<std::pair<int, int>, Operation *> tiles;
        DenseMap<Operation *, CoreOp> cores;
        DenseMap<Operation *, MemOp> mems;
        DenseMap<std::pair<Operation *, int>, LockOp> locks;
        DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;
        DenseMap<Operation *, SwitchboxOp> switchboxes;

        if (module.getOps<DeviceOp>().empty()) {
          module.emitOpError("expected AIE.device operation at toplevel");
        }
        DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

        NetlistAnalysis NL(targetOp, tiles, cores, mems, locks, buffers,
                           switchboxes);
        NL.collectTiles(tiles);
        NL.collectBuffers(buffers);

        for (auto tile : targetOp.getOps<TileOp>())
          if (tile.colIndex() == tileCol && tile.rowIndex() == tileRow) {
            auto srcCoord = std::make_pair(tile.colIndex(), tile.rowIndex());
            const auto &target_model = getTargetModel(tile);

            // Figure out how much memory we have left for random allocations
            auto core = tile.getCoreOp();
            int max = core.getStackSize();
            for (auto buf : buffers[tiles[srcCoord]]) {
              int bufferBaseAddr = NL.getBufferBaseAddress(buf);
              int numBytes = buf.getAllocationSize();
              max = std::max(max, bufferBaseAddr + numBytes);
            }
            int origin = target_model.getMemInternalBaseAddress(srcCoord) + max;
            int length = target_model.getLocalMemorySize() - max;
            // output << "// Tile(" << tileCol << ", " << tileRow << ")\n";
            // output << "// Memory map: name base_address num_bytes\n";
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
     /* the _main_init symbol from me_basic.o has to come at address zero. */
     *me_basic.o(.text)
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
            auto doBuffer = [&](Optional<TileID> tile, int offset,
                                std::string dir) {
              if (tile) {
                if (tiles.count(*tile))
                  for (auto buf : buffers[tiles[*tile]])
                    writeLDScriptMap(output, buf, offset, NL);
              } else {
                output << "/* No tile with memory exists to the " << dir
                       << ". */\n";
                output << ". = 0x" << llvm::utohexstr(offset) << ";\n";
                uint32_t localMemSize = target_model.getLocalMemorySize();
                output << ". += 0x" << llvm::utohexstr(localMemSize) << ";\n";
              }
            };

            // Stack
            output << ". = 0x"
                   << llvm::utohexstr(
                          target_model.getMemInternalBaseAddress(srcCoord))
                   << ";\n";
            output << "_sp_start_value_DM_stack = .;\n";

            if (auto core = tile.getCoreOp())
              output << ". += 0x" << llvm::utohexstr(core.getStackSize())
                     << "; /* stack */\n";
            else
              output << "/* no stack allocated */\n";

            doBuffer(target_model.getMemSouth(srcCoord),
                     target_model.getMemSouthBaseAddress(),
                     std::string("south"));
            doBuffer(target_model.getMemWest(srcCoord),
                     target_model.getMemWestBaseAddress(), std::string("west"));
            doBuffer(target_model.getMemNorth(srcCoord),
                     target_model.getMemNorthBaseAddress(),
                     std::string("north"));
            doBuffer(target_model.getMemEast(srcCoord),
                     target_model.getMemEastBaseAddress(), std::string("east"));

            output << "  .bss : { *(.bss) } > data\n";
            output << "  .bss.DMb.4 : { *(.bss.DMb.4) } > data\n";
            output << "}\n";
            if (auto coreOp = tile.getCoreOp()) {
              if (auto fileAttr =
                      coreOp->getAttrOfType<StringAttr>("link_with")) {
                auto fileName = std::string(fileAttr.getValue());
                output << "INPUT(" << fileName << ")\n";
              }
              output << "PROVIDE(_main = core_" << tile.getCol() << "_"
                     << tile.getRow() << ");\n";
            }
          }
        return success();
      },
      registerDialects);

  //   _entry_point _main_init
  // _symbol      _main _after _main_init
  // _symbol      _main_init 0
  // _reserved DMb      0x00000 0x20000
  // _symbol   a        0x38000 0x2000
  // _extern   a
  // _stack    DM_stack 0x20000  0x400 //stack for core
  // _reserved DMb 0x40000 0xc0000 // And everything else the core can't see

  TranslateFromMLIRRegistration registrationBCF(
      "aie-generate-bcf", "Generate AIE bcf",
      [](ModuleOp module, raw_ostream &output) {
        DenseMap<std::pair<int, int>, Operation *> tiles;
        DenseMap<Operation *, CoreOp> cores;
        DenseMap<Operation *, MemOp> mems;
        DenseMap<std::pair<Operation *, int>, LockOp> locks;
        DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;
        DenseMap<Operation *, SwitchboxOp> switchboxes;

        if (module.getOps<DeviceOp>().empty()) {
          module.emitOpError("expected AIE.device operation at toplevel");
        }
        DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

        NetlistAnalysis NL(targetOp, tiles, cores, mems, locks, buffers,
                           switchboxes);
        NL.collectTiles(tiles);
        NL.collectBuffers(buffers);

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
            const auto &target_model = getTargetModel(tile);

            std::string corefunc = std::string("core_") +
                                   std::to_string(tile.getCol()) + "_" +
                                   std::to_string(tile.getRow());
            output << "_entry_point _main_init\n";
            output << "_symbol " << corefunc << " _after _main_init\n";
            output << "_symbol      _main_init 0\n";
            std::string initReserved =
                (target_model.getTargetArch() == AIEArch::AIE2) ? "0x40000"
                                                                : "0x20000";
            output << "_reserved DMb      0x00000 " << initReserved
                   << " //Don't put data in code memory\n";

            auto srcCoord = std::make_pair(tile.colIndex(), tile.rowIndex());
            auto doBuffer = [&](Optional<TileID> tile, int offset,
                                std::string dir) {
              if (tile) {
                if (tiles.count(*tile))
                  for (auto buf : buffers[tiles[*tile]])
                    writeBCFMap(output, buf, offset, NL);
                uint32_t localMemSize = target_model.getLocalMemorySize();
                if (tile != srcCoord)
                  output << "_reserved DMb 0x" << llvm::utohexstr(offset) << " "
                         << "0x" << llvm::utohexstr(localMemSize) << " "
                         << " // Don't allocate variables outside of local "
                            "memory.\n";
                // TODO How to set as reserved if no buffer exists (or reserve
                // remaining buffer)
              } else {
                uint32_t localMemSize = target_model.getLocalMemorySize();
                output << "_reserved DMb 0x" << llvm::utohexstr(offset) << " "
                       << "0x" << llvm::utohexstr(localMemSize) << " "
                       << " // No tile with memory exists to the " << dir
                       << ".\n";
              }
            };

            doBuffer(target_model.getMemSouth(srcCoord),
                     target_model.getMemSouthBaseAddress(),
                     std::string("south"));
            doBuffer(target_model.getMemWest(srcCoord),
                     target_model.getMemWestBaseAddress(), std::string("west"));
            doBuffer(target_model.getMemNorth(srcCoord),
                     target_model.getMemNorthBaseAddress(),
                     std::string("north"));
            doBuffer(target_model.getMemEast(srcCoord),
                     target_model.getMemEastBaseAddress(), std::string("east"));

            int stacksize = 0;
            if (auto core = tile.getCoreOp())
              stacksize = core.getStackSize();
            output << "_stack    DM_stack 0x"
                   << llvm::utohexstr(
                          target_model.getMemInternalBaseAddress(srcCoord))
                   << "  0x" << llvm::utohexstr(stacksize)
                   << " //stack for core\n";

            if (target_model.getTargetArch() == AIEArch::AIE2) {
              output << "_reserved DMb 0x80000 0x80000 // And everything else "
                        "the core can't see\n";
            } else {
              output << "_reserved DMb 0x40000 0xc0000 // And everything else "
                        "the core can't see\n";
            }
            if (auto coreOp = tile.getCoreOp()) {
              if (auto fileAttr =
                      coreOp->getAttrOfType<StringAttr>("link_with")) {
                auto fileName = std::string(fileAttr.getValue());
                output << "_include _file " << fileName << "\n";
              }
            }
            output << "_resolve _main core_" << tile.getCol() << "_"
                   << tile.getRow() << "\n";
          }
        return success();
      },
      registerDialects);

  TranslateFromMLIRRegistration registrationCoreList(
      "aie-generate-corelist", "Generate python list of cores",
      [](ModuleOp module, raw_ostream &output) {
        if (module.getOps<DeviceOp>().empty()) {
          module.emitOpError("expected AIE.device operation at toplevel");
        }
        DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

        output << "[";
        for (auto tileOp : targetOp.getOps<TileOp>()) {
          int col = tileOp.colIndex();
          int row = tileOp.rowIndex();
          if (auto coreOp = tileOp.getCoreOp()) {
            std::string elf_file = "None";
            if (auto fileAttr = coreOp->getAttrOfType<StringAttr>("elf_file"))
              elf_file = "\"" + std::string(fileAttr.getValue()) + "\"";
            output << '(' << std::to_string(col) << ',' << std::to_string(row)
                   << ',' << elf_file << "),";
          }
        }
        output << "]\n";
        return success();
      },
      registerDialects);

  TranslateFromMLIRRegistration registrationXADF(
      "adf-generate-cpp-graph", "Translate ADFDialect to C++ graph",
      ADFGenerateCPPGraph, [](DialectRegistry &registry) {
        registry.insert<xilinx::ADF::ADFDialect>();
        registerDialects(registry);
      });
  TranslateFromMLIRRegistration registrationXAIE(
      "aie-generate-xaie", "Generate libxaie configuration",
      [](ModuleOp module, raw_ostream &output) {
        return AIETranslateToXAIEV2(module, output);
      },
      registerDialects);
  TranslateFromMLIRRegistration registrationXJSON(
      "aie-flows-to-json", "Translate AIE flows to JSON", AIEFlowsToJSON,
      registerDialects);
  TranslateFromMLIRRegistration registrationXPE(
      "aie-mlir-to-xpe", "Translate AIE design to XPE file for simulation",
      AIETranslateGraphXPE, registerDialects);
  TranslateFromMLIRRegistration registrationSCSimConfig(
      "aie-mlir-to-scsim-config",
      "Translate AIE design to SCSimConfig file for simulation",
      AIETranslateSCSimConfig, registerDialects);
  TranslateFromMLIRRegistration registrationShimSolution(
      "aie-mlir-to-shim-solution",
      "Translate AIE design to ShimSolution file for simulation",
      AIETranslateShimSolution, registerDialects);
}
} // namespace AIE
} // namespace xilinx
