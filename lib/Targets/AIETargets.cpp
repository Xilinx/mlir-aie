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
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"

#include <set>

#define DEBUG_TYPE "aie-targets"

using namespace mlir;
using namespace mlir::vector;
using namespace xilinx;
using namespace xilinx::AIE;

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
  registry.insert<index::IndexDialect>();
}

// Output the buffer map for the given buffer operations, with the given offset.
// The offset is different depending on where the buffers are accessed from.
void writeBufferMap(raw_ostream &output, BufferOp buf, int offset) {
  std::string bufName(buf.name().getValue());
  int bufferBaseAddr = getBufferBaseAddress(buf);
  int numBytes = buf.getAllocationSize();
  output << "_symbol " << bufName << " " << "0x"
         << llvm::utohexstr(offset + bufferBaseAddr) << " " << numBytes << '\n';
}

LogicalResult AIETranslateToTargetArch(ModuleOp module, raw_ostream &output) {
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
}

void registerAIETranslations() {
  static llvm::cl::opt<int> tileCol(
      "tilecol", llvm::cl::desc("column coordinate of core to translate"),
      llvm::cl::init(0));
  static llvm::cl::opt<int> tileRow(
      "tilerow", llvm::cl::desc("row coordinate of core to translate"),
      llvm::cl::init(0));

#ifdef AIE_ENABLE_AIRBIN
  static llvm::cl::opt<std::string> outputFilename(
      "airbin-output-filepath",
      llvm::cl::desc("Output airbin file path (including filename)"),
      llvm::cl::value_desc("airbin-output-filepath"),
      llvm::cl::init("airbin.elf"));

  static llvm::cl::opt<std::string> coreFilesDir(
      "airbin-aux-core-dir-path",
      llvm::cl::desc("Auxiliary core elf files dir path"),
      llvm::cl::value_desc("airbin-aux-core-dir-path"), llvm::cl::init("."));
#endif

  static llvm::cl::opt<std::string> workDirPath(
      "work-dir-path", llvm::cl::Optional,
      llvm::cl::desc("Absolute path to working directory"));

  static llvm::cl::opt<bool> bigEndian("big-endian", llvm::cl::init(false),
                                       llvm::cl::desc("Endianness"));

  static llvm::cl::opt<bool> cdoUnified(
      "cdo-unified", llvm::cl::init(false),
      llvm::cl::desc("Emit unified CDO bin (or separate bins)"));
  static llvm::cl::opt<bool> cdoDebug("cdo-debug", llvm::cl::init(false),
                                      llvm::cl::desc("Emit cdo debug info"));
  static llvm::cl::opt<bool> cdoAieSim(
      "cdo-aiesim", llvm::cl::init(false),
      llvm::cl::desc("AIESIM target cdo generation"));
  static llvm::cl::opt<bool> cdoXaieDebug(
      "cdo-xaie-debug", llvm::cl::init(false),
      llvm::cl::desc("Emit libxaie debug info"));
  static llvm::cl::opt<size_t> cdoEnableCores(
      "cdo-enable-cores", llvm::cl::init(true),
      llvm::cl::desc("Enable cores in CDO"));

  static llvm::cl::opt<bool> outputBinary(
      "aie-output-binary", llvm::cl::init(false),
      llvm::cl::desc(
          "Select binary (true) or text (false) output for supported "
          "translations. e.g. aie-npu-instgen, aie-ctrlpkt-to-bin"));

  TranslateFromMLIRRegistration registrationMMap(
      "aie-generate-mmap", "Generate AIE memory map",
      [](ModuleOp module, raw_ostream &output) {
        DenseMap<TileID, Operation *> tiles;
        DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;

        if (module.getOps<DeviceOp>().empty()) {
          module.emitOpError("expected AIE.device operation at toplevel");
        }
        DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

        collectTiles(targetOp, tiles);
        // sort the tiles for deterministic output
        using tileType = std::pair<TileID, Operation *>;
        struct tileCmp {
          bool operator()(const tileType &lhs, const tileType &rhs) const {
            return lhs.first < rhs.first;
          }
        };
        std::set<tileType, tileCmp> sortedTiles;
        for (auto tile : tiles)
          sortedTiles.insert(tileType{tile.first, tile.second});

        collectBuffers(targetOp, buffers);

        for (auto tile : sortedTiles) {
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

  TranslateFromMLIRRegistration registrationBCF(
      "aie-generate-bcf", "Generate AIE bcf",
      [](ModuleOp module, raw_ostream &output) {
        return AIETranslateToBCF(module, output, tileCol, tileRow);
      },
      registerDialects);

  TranslateFromMLIRRegistration registrationTargetArch(
      "aie-generate-target-arch", "Get the target architecture",
      AIETranslateToTargetArch, registerDialects);

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
#ifdef AIE_ENABLE_AIRBIN
  TranslateFromMLIRRegistration registrationAirbin(
      "aie-generate-airbin", "Generate configuration binary blob",
      [](ModuleOp module, raw_ostream &) {
        return AIETranslateToAirbin(module, outputFilename, coreFilesDir);
      },
      registerDialects);
#endif
  TranslateFromMLIRRegistration registrationXAIE(
      "aie-generate-xaie", "Generate libxaie configuration",
      [](ModuleOp module, raw_ostream &output) {
        return AIETranslateToXAIEV2(module, output);
      },
      registerDialects);
  TranslateFromMLIRRegistration registrationHSA(
      "aie-generate-hsa", "Generate hsa data movement configuration",
      [](ModuleOp module, raw_ostream &output) {
        return AIETranslateToHSA(module, output);
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
  TranslateFromMLIRRegistration registrationCDODirect(
      "aie-generate-cdo", "Generate libxaie for CDO directly",
      [](ModuleOp module, raw_ostream &) {
        SmallString<128> workDirPath_;
        if (workDirPath.getNumOccurrences() == 0) {
          if (llvm::sys::fs::current_path(workDirPath_))
            llvm::report_fatal_error(
                "couldn't get cwd to use as work-dir-path");
        } else
          workDirPath_ = workDirPath.getValue();
        LLVM_DEBUG(llvm::dbgs() << "work-dir-path: " << workDirPath_ << "\n");
        return AIETranslateToCDODirect(module, workDirPath_.c_str(), bigEndian,
                                       cdoUnified, cdoDebug, cdoAieSim,
                                       cdoXaieDebug, cdoEnableCores);
      },
      registerDialects);
  TranslateFromMLIRRegistration registrationCDOWithTxn(
      "aie-generate-txn", "Generate TXN configuration",
      [](ModuleOp module, raw_ostream &) {
        SmallString<128> workDirPath_;
        if (workDirPath.getNumOccurrences() == 0) {
          if (llvm::sys::fs::current_path(workDirPath_))
            llvm::report_fatal_error(
                "couldn't get cwd to use as work-dir-path");
        } else
          workDirPath_ = workDirPath.getValue();
        LLVM_DEBUG(llvm::dbgs() << "work-dir-path: " << workDirPath_ << "\n");
        return AIETranslateToTxn(module, workDirPath_.c_str(), cdoAieSim,
                                 cdoXaieDebug, cdoEnableCores);
      },
      registerDialects);
  TranslateFromMLIRRegistration registrationNPU(
      "aie-npu-instgen", "Translate npu instructions to binary",
      [](ModuleOp module, raw_ostream &output) {
        if (outputBinary == true) {
          std::vector<uint32_t> instructions;
          auto r = AIETranslateToNPU(module, instructions);
          if (failed(r))
            return r;
          output.write(reinterpret_cast<const char *>(instructions.data()),
                       instructions.size() * sizeof(uint32_t));
          return success();
        }
        return AIETranslateToNPU(module, output);
      },
      registerDialects);
  TranslateFromMLIRRegistration registrationCtrlPkt(
      "aie-ctrlpkt-to-bin", "Translate aiex.control_packet ops to binary",
      [](ModuleOp module, raw_ostream &output) {
        if (outputBinary == true) {
          std::vector<uint32_t> instructions;
          auto r = AIETranslateToControlPackets(module, instructions);
          if (failed(r))
            return r;
          output.write(reinterpret_cast<const char *>(instructions.data()),
                       instructions.size() * sizeof(uint32_t));
          return success();
        }
        return AIETranslateToControlPackets(module, output);
      },
      registerDialects);
}
} // namespace xilinx::AIE
