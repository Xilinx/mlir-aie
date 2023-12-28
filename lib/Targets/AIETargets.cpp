//===- AIETargets.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"
#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/JSON.h"

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

llvm::json::Value attrToJSON(Attribute &attr) {
  if (auto a = llvm::dyn_cast<StringAttr>(attr))
    return {a.getValue().str()};

  if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
    llvm::json::Array arrayJSON;
    for (auto a : arrayAttr)
      arrayJSON.push_back(attrToJSON(a));
    return llvm::json::Value(std::move(arrayJSON));
  }

  if (auto dictAttr = llvm::dyn_cast<DictionaryAttr>(attr)) {
    llvm::json::Object dictJSON;
    for (auto a : dictAttr) {
      auto ident = a.getName();
      auto attr = a.getValue();
      dictJSON[ident.str()] = attrToJSON(attr);
    }
    return llvm::json::Value(std::move(dictJSON));
  }

  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
    return llvm::json::Value(intAttr.getInt());

  return llvm::json::Value(std::string(""));
}

namespace xilinx::AIE {

static void registerDialects(DialectRegistry &registry) {
  registry.insert<xilinx::AIE::AIEDialect>();
  registry.insert<xilinx::AIEX::AIEXDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<DLTIDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<VectorDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<emitc::EmitCDialect>();
}

// Output the buffer map for the given buffer operations, with the given offset.
// The offset is different depending on where the buffers are accessed from.
void writeBufferMap(raw_ostream &output, BufferOp buf, int offset) {
  std::string bufName(buf.name().getValue());
  int bufferBaseAddr = getBufferBaseAddress(buf);
  int numBytes = buf.getAllocationSize();
  output << "_symbol " << bufName << " "
         << "0x" << llvm::utohexstr(offset + bufferBaseAddr) << " " << numBytes
         << '\n';
}
void registerAIETranslations() {
  TranslateFromMLIRRegistration registrationMMap(
      "aie-generate-mmap", "Generate AIE memory map",
      [](ModuleOp module, raw_ostream &output) {
        DenseMap<TileID, Operation *> tiles;
        DenseMap<Operation *, CoreOp> cores;
        DenseMap<Operation *, MemOp> mems;
        DenseMap<std::pair<Operation *, int>, LockOp> locks;
        DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;
        DenseMap<Operation *, SwitchboxOp> switchboxes;

        if (module.getOps<DeviceOp>().empty()) {
          module.emitOpError("expected AIE.device operation at toplevel");
        }
        DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

        collectTiles(targetOp, tiles);
        collectBuffers(targetOp, buffers);

        for (auto tile : tiles) {
          Operation *srcTileOp = tile.second;
          TileID srcCoord = cast<TileOp>(srcTileOp).getTileID();
          int srcCol = srcCoord.col;
          int srcRow = srcCoord.row;

          output << "// Tile(" << srcCol << ", " << srcRow << ")\n";
          output << "// Memory map: name base_address num_bytes\n";

          auto doBuffer = [&](std::optional<TileID> tile, int offset) {
            if (tiles.count(*tile))
              for (auto buf : buffers[tiles[*tile]])
                writeBufferMap(output, buf, offset);
          };

          const auto &targetModel = xilinx::AIE::getTargetModel(srcTileOp);

          if (auto tile = targetModel.getMemSouth(srcCoord))
            doBuffer(tile, targetModel.getMemSouthBaseAddress());
          if (auto tile = targetModel.getMemWest(srcCoord))
            doBuffer(tile, targetModel.getMemWestBaseAddress());
          if (auto tile = targetModel.getMemNorth(srcCoord))
            doBuffer(tile, targetModel.getMemNorthBaseAddress());
          if (auto tile = targetModel.getMemEast(srcCoord))
            doBuffer(tile, targetModel.getMemEastBaseAddress());
        }
        return success();
      },
      registerDialects);

  TranslateFromMLIRRegistration registrationShimDMAToJSON(
      "aie-generate-json", "Transform AIE shim DMA allocation info into JSON",
      [](ModuleOp module, raw_ostream &output) {
        for (auto d : module.getOps<DeviceOp>()) {
          llvm::json::Object moduleJSON;
          for (auto shimDMAMeta : d.getOps<ShimDMAAllocationOp>()) {
            llvm::json::Object shimJSON;
            auto channelDir = shimDMAMeta.getChannelDirAttr();
            shimJSON["channelDir"] = attrToJSON(channelDir);
            auto channelIndex = shimDMAMeta.getChannelIndexAttr();
            shimJSON["channelIndex"] = attrToJSON(channelIndex);
            auto col = shimDMAMeta.getColAttr();
            shimJSON["col"] = attrToJSON(col);
            moduleJSON[shimDMAMeta.getSymName()] =
                llvm::json::Value(std::move(shimJSON));
          }
          llvm::json::Value topv(std::move(moduleJSON));
          std::string ret;
          llvm::raw_string_ostream ss(ret);
          ss << llvm::formatv("{0:2}", topv) << "\n";
          output << ss.str();
        }
        return success();
      },
      registerDialects);

  TranslateFromMLIRRegistration registrationLDScript(
      "aie-generate-ldscript", "Generate AIE loader script",
      [](ModuleOp module, raw_ostream &output) {
        return AIETranslateToLdScript(module, output, tileCol, tileRow);
      },
      registerDialects);

  // _entry_point _main_init
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
        return AIETranslateToBCF(module, output, tileCol, tileRow);
      },
      registerDialects);

  TranslateFromMLIRRegistration registrationTargetArch(
      "aie-generate-target-arch", "Get the target architecture",
      [](ModuleOp module, raw_ostream &output) {
        AIEArch arch = AIEArch::AIE1;
        if (!module.getOps<DeviceOp>().empty()) {
          DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());
          arch = targetOp.getTargetModel().getTargetArch();
        }
        if (arch == AIEArch::AIE1)
          output << "AIE\n";
        else
          output << stringifyEnum(arch) << "\n";
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
            std::string elfFile = "None";
            if (auto fileAttr = coreOp.getElfFile())
              elfFile = "\"" + fileAttr.value().str() + "\"";
            output << '(' << col << ',' << row << ',' << elfFile << "),";
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
  TranslateFromMLIRRegistration registrationCDO(
      "aie-generate-cdo", "Generate libxaie for CDO",
      [](ModuleOp module, raw_ostream &output) {
        return AIETranslateToCDO(module, output);
      },
      registerDialects);
  TranslateFromMLIRRegistration registrationIPU(
      "aie-ipu-instgen", "Generate instructions for IPU",
      [](ModuleOp module, raw_ostream &output) {
        return AIETranslateToIPU(module, output);
      },
      registerDialects);
}
} // namespace xilinx::AIE
