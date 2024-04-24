//===- XCLBinGen.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===---------------------------------------------------------------------===//

#include "XCLBinGen.h"

#include "aie/Conversion/AIEVecToLLVM/AIEVecToLLVM.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"

#include <regex>
#include <sstream>
#include <unordered_map>

#ifdef _WIN32
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

using namespace llvm;
using namespace mlir;
using namespace xilinx;

namespace {

// Apply the pass manager specific options of the XCLBinGenConfig to the pass
// manager. These control when (if ever) and what IR gets printed between
// passes, and whether the pass manager uses multi-theading.
void applyConfigToPassManager(XCLBinGenConfig &TK, PassManager &pm) {

  pm.getContext()->disableMultithreading(TK.DisableThreading);

  bool printBefore = TK.PrintIRBeforeAll;
  auto shouldPrintBeforePass = [printBefore](Pass *, Operation *) {
    return printBefore;
  };

  bool printAfter = TK.PrintIRAfterAll;
  auto shouldPrintAfterPass = [printAfter](Pass *, Operation *) {
    return printAfter;
  };

  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      TK.PrintIRModuleScope);
}
} // namespace

void xilinx::findVitis(XCLBinGenConfig &TK) {
  const char *env_vitis = ::getenv("VITIS");
  if (env_vitis == nullptr) {
    if (auto vpp = sys::findProgramByName("v++")) {
      SmallString<64> real_vpp;
      std::error_code err = sys::fs::real_path(vpp.get(), real_vpp);
      if (!err) {
        sys::path::remove_filename(real_vpp);
        sys::path::remove_filename(real_vpp);
        ::setenv("VITIS", real_vpp.c_str(), 1);
        dbgs() << "Found Vitis at " << real_vpp.c_str() << "\n";
      }
    }
  }
  env_vitis = ::getenv("VITIS");
  if (env_vitis != nullptr) {
    SmallString<64> vitis_path(env_vitis);
    SmallString<64> vitis_bin_path(vitis_path);
    sys::path::append(vitis_bin_path, "bin");

    SmallString<64> aietools_path(vitis_path);
    sys::path::append(aietools_path, "aietools");
    if (!sys::fs::exists(aietools_path)) {
      aietools_path = vitis_path;
      sys::path::append(aietools_path, "cardano");
    }
    TK.AIEToolsDir = std::string(aietools_path);
    ::setenv("AIETOOLS", TK.AIEToolsDir.c_str(), 1);

    SmallString<64> aietools_bin_path(aietools_path);
    sys::path::append(aietools_bin_path, "bin");
    const char *env_path = ::getenv("PATH");
    if (env_path == nullptr)
      env_path = "";
    SmallString<128> new_path(env_path);
    if (new_path.size())
      new_path += sys::EnvPathSeparator;
    new_path += aietools_bin_path;
    new_path += sys::EnvPathSeparator;
    new_path += vitis_bin_path;
    ::setenv("PATH", new_path.c_str(), 1);
  } else {
    errs() << "VITIS not found ...\n";
  }
}

static void addAIELoweringPasses(OpPassManager &pm) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(AIE::createAIECanonicalizeDevicePass());
  OpPassManager &devicePM = pm.nest<AIE::DeviceOp>();
  devicePM.addPass(AIE::createAIEAssignLockIDsPass());
  devicePM.addPass(AIE::createAIEAssignBufferDescriptorIDsPass());
  devicePM.addPass(AIE::createAIEObjectFifoRegisterProcessPass());
  devicePM.addPass(AIE::createAIEObjectFifoStatefulTransformPass());
  devicePM.addPass(AIEX::createAIEBroadcastPacketPass());
  devicePM.addPass(AIE::createAIERoutePacketFlowsPass());
  devicePM.addPass(AIEX::createAIELowerMulticastPass());
  devicePM.addPass(AIE::createAIEAssignBufferAddressesPass());
  pm.addPass(createConvertSCFToCFPass());
}

static void addLowerToLLVMPasses(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(xilinx::aievec::createConvertAIEVecToLLVMPass());

  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  ConvertFuncToLLVMPassOptions opts;
  opts.useBarePtrCallConv = true;
  pm.addPass(createConvertFuncToLLVMPass(opts));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

int runTool(StringRef Program, ArrayRef<std::string> Args, bool Verbose,
            std::optional<ArrayRef<StringRef>> Env = std::nullopt) {
  if (Verbose) {
    llvm::outs() << "Run:";
    if (Env)
      for (auto &s : *Env)
        llvm::outs() << " " << s;
    llvm::outs() << " " << Program;
    for (auto &s : Args)
      llvm::outs() << " " << s;
    llvm::outs() << "\n";
  }
  std::string err_msg;
  sys::ProcessStatistics stats;
  std::optional<sys::ProcessStatistics> opt_stats(stats);
  SmallVector<StringRef, 8> PArgs = {Program};
  PArgs.append(Args.begin(), Args.end());
  int result = sys::ExecuteAndWait(Program, PArgs, Env, {}, 0, 0, &err_msg,
                                   nullptr, &opt_stats);
  if (Verbose)
    llvm::outs() << (result == 0 ? "Succeeded " : "Failed ") << "in "
                 << std::chrono::duration_cast<std::chrono::duration<float>>(
                        stats.TotalTime)
                        .count()
                 << " code: " << result << "\n";
  return result;
}

template <unsigned N>
static void aieTargetDefines(SmallVector<std::string, N> &Args,
                             std::string aie_target) {
  if (aie_target == "AIE2")
    Args.push_back("-D__AIEARCH__=20");
  else
    Args.push_back("-D__AIEARCH__=10");
}

// Generate the elf files for the core
static LogicalResult generateCoreElfFiles(ModuleOp moduleOp,
                                          const StringRef objFile,
                                          XCLBinGenConfig &TK) {
  auto deviceOps = moduleOp.getOps<AIE::DeviceOp>();
  if (!llvm::hasSingleElement(deviceOps))
    return moduleOp.emitOpError("expected a single device op");

  AIE::DeviceOp deviceOp = *deviceOps.begin();
  auto tileOps = deviceOp.getOps<AIE::TileOp>();

  std::string errorMessage;

  for (auto tileOp : tileOps) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    auto coreOp = tileOp.getCoreOp();
    if (!coreOp)
      continue;

    std::string elfFileName;
    if (auto fileAttr = coreOp.getElfFileAttr()) {
      elfFileName = std::string(fileAttr.getValue());
    } else {
      elfFileName = std::string("core_") + std::to_string(col) + "_" +
                    std::to_string(row) + ".elf";
      coreOp.setElfFile(elfFileName);
    }

    SmallString<64> elfFile(TK.TempDir);
    sys::path::append(elfFile, elfFileName);

    if (TK.UseChess) {
      // Use xbridge (to remove any peano dependency with use-chess option)
      SmallString<64> bcfPath(TK.TempDir);
      sys::path::append(bcfPath, elfFileName + ".bcf");

      {
        auto bcfOutput = openOutputFile(bcfPath, &errorMessage);
        if (!bcfOutput)
          return coreOp.emitOpError(errorMessage);

        if (failed(AIE::AIETranslateToBCF(moduleOp, bcfOutput->os(), col, row)))
          return coreOp.emitOpError("Failed to generate BCF");
        bcfOutput->keep();
      }

      std::vector<std::string> extractedIncludes;
      {
        auto bcfFileIn = openInputFile(bcfPath, &errorMessage);
        if (!bcfFileIn)
          moduleOp.emitOpError(errorMessage);

        std::string bcfFile = std::string(bcfFileIn->getBuffer());
        std::regex r("_include _file (.*)");
        auto begin = std::sregex_iterator(bcfFile.begin(), bcfFile.end(), r);
        auto end = std::sregex_iterator();
        for (std::sregex_iterator i = begin; i != end; ++i)
          extractedIncludes.push_back(i->str(1));
      }

      SmallString<64> chessWrapperBin(TK.InstallDir);
      sys::path::append(chessWrapperBin, "bin", "xchesscc_wrapper");
      SmallString<64> chessworkDir(TK.TempDir);
      sys::path::append(chessworkDir, "chesswork");

      SmallVector<std::string> flags{StringRef(TK.TargetArch).lower(),
                                     "+w",
                                     std::string(chessworkDir),
                                     "-d",
                                     "+l",
                                     std::string(bcfPath),
                                     "-o",
                                     std::string(elfFile),
                                     "-f",
                                     std::string(objFile)};
      for (const auto &inc : extractedIncludes)
        flags.push_back(inc);

      if (runTool(chessWrapperBin, flags, TK.Verbose) != 0)
        coreOp.emitOpError("Failed to link with xbridge");
    } else {
      SmallString<64> ldscript_path(TK.TempDir);
      sys::path::append(ldscript_path, elfFileName + ".ld");
      {
        auto ldscript_output = openOutputFile(ldscript_path, &errorMessage);
        if (!ldscript_output)
          return coreOp.emitOpError(errorMessage);

        if (failed(AIE::AIETranslateToLdScript(moduleOp, ldscript_output->os(),
                                               col, row)))
          return coreOp.emitOpError("failed to generate ld script for core (")
                 << col << "," << row << ")";
        ldscript_output->keep();
      }

      // We are running a clang command for now, but really this is an lld
      // command.
      {
        std::string targetLower = StringRef(TK.TargetArch).lower();
        SmallVector<std::string, 10> flags;
        flags.push_back("-O2");
#ifdef _WIN32
        // TODO: Windows tries to load the wrong builtins path.
        std::string targetFlag = "--target=" + targetLower;
#else
        std::string targetFlag = "--target=" + targetLower + "-none-elf";
#endif
        flags.push_back(targetFlag);
        flags.emplace_back(objFile);
        SmallString<64> meBasicPath(TK.InstallDir);
        sys::path::append(meBasicPath, "aie_runtime_lib", TK.TargetArch,
                          "me_basic.o");
        flags.emplace_back(meBasicPath);
#ifndef _WIN32
        // TODO: No libc build on windows
        SmallString<64> libcPath(TK.PeanoDir);
        sys::path::append(libcPath, "lib", targetLower + "-none-unknown-elf",
                          "libc.a");
        flags.emplace_back(libcPath);
#endif
        flags.push_back("-Wl,--gc-sections");
        std::string ldScriptFlag = "-Wl,-T," + std::string(ldscript_path);
        flags.push_back(ldScriptFlag);
        flags.push_back("-o");
        flags.emplace_back(elfFile);
        SmallString<64> clangBin(TK.PeanoDir);
        sys::path::append(clangBin, "bin", "clang");
        if (runTool(clangBin, flags, TK.Verbose) != 0)
          return coreOp.emitOpError("failed to link elf file for core(")
                 << col << "," << row << ")";
      }
    }
  }
  return success();
}

static LogicalResult generateCDO(MLIRContext *context, ModuleOp moduleOp,
                                 XCLBinGenConfig &TK) {
  ModuleOp copy = moduleOp.clone();
  std::string errorMessage;
  // This corresponds to `process_host_cgen`, which is listed as host
  // compilation in aiecc.py... not sure we need this.
  PassManager passManager(context, ModuleOp::getOperationName());
  applyConfigToPassManager(TK, passManager);

  passManager.addNestedPass<AIE::DeviceOp>(AIE::createAIEPathfinderPass());
  passManager.addNestedPass<AIE::DeviceOp>(
      AIEX::createAIEBroadcastPacketPass());
  passManager.addNestedPass<AIE::DeviceOp>(
      AIE::createAIERoutePacketFlowsPass());
  passManager.addNestedPass<AIE::DeviceOp>(AIEX::createAIELowerMulticastPass());
  if (failed(passManager.run(copy)))
    return moduleOp.emitOpError(
        "failed to run passes to prepare of XCLBin generation");

  if (failed(AIE::AIETranslateToCDODirect(copy, TK.TempDir)))
    return moduleOp.emitOpError("failed to emit CDO");

  copy->erase();
  return success();
}

static json::Object makeKernelJSON(std::string name, std::string id,
                                   std::string instance) {
  return json::Object{
      {"name", name},
      {"type", "dpu"},
      {"extended-data", json::Object{{"subtype", "DPU"},
                                     {"functional", "1"},
                                     {"dpu_kernel_id", id}}},
      {"arguments", json::Array{json::Object{{"name", "instr"},
                                             {"memory-connection", "SRAM"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "char *"},
                                             {"offset", "0x00"}},
                                json::Object{{"name", "ninstr"},
                                             {"address-qualifier", "SCALAR"},
                                             {"type", "uint64_t"},
                                             {"offset", "0x08"}},
                                json::Object{{"name", "in"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "char *"},
                                             {"offset", "0x10"}},
                                json::Object{{"name", "tmp"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "char *"},
                                             {"offset", "0x18"}},
                                json::Object{{"name", "out"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "char *"},
                                             {"offset", "0x20"}}}},
      {"instances", json::Array{json::Object{{"name", instance}}}}};
}

static LogicalResult generateXCLBin(MLIRContext *context, ModuleOp moduleOp,
                                    XCLBinGenConfig &TK,
                                    const StringRef &Output) {
  std::string errorMessage;
  // Create mem_topology.json.
  SmallString<64> memTopologyJsonFile(TK.TempDir);
  sys::path::append(memTopologyJsonFile, "mem_topology.json");
  {
    auto memTopologyJsonOut =
        openOutputFile(memTopologyJsonFile, &errorMessage);
    if (!memTopologyJsonOut)
      return moduleOp.emitOpError(errorMessage);

    std::string mem_topology_data = R"({
      "mem_topology": {
          "m_count": "2",
          "m_mem_data": [
              {
                  "m_type": "MEM_DRAM",
                  "m_used": "1",
                  "m_sizeKB": "0x10000",
                  "m_tag": "HOST",
                  "m_base_address": "0x4000000"
              },
              {
                  "m_type": "MEM_DRAM",
                  "m_used": "1",
                  "m_sizeKB": "0xc000",
                  "m_tag": "SRAM",
                  "m_base_address": "0x4000000"
              }
          ]
      }
    })";
    memTopologyJsonOut->os() << mem_topology_data;
    memTopologyJsonOut->keep();
  }

  // Create aie_partition.json.
  SmallString<64> aiePartitionJsonFile(TK.TempDir);
  sys::path::append(aiePartitionJsonFile, "aie_partition.json");
  {
    auto aiePartitionJsonOut =
        openOutputFile(aiePartitionJsonFile, &errorMessage);
    if (!aiePartitionJsonOut)
      return moduleOp.emitOpError(errorMessage);

    std::string aie_partition_json_data = R"(
      {
        "aie_partition": {
          "name": "QoS",
          "operations_per_cycle": "2048",
          "inference_fingerprint": "23423",
          "pre_post_fingerprint": "12345",
          "partition": {
            "column_width": 4,
            "start_columns": [
              1
            ]
          },
          "PDIs": [
            {
              "uuid": "00000000-0000-0000-0000-000000008025",
              "file_name": "./design.pdi",
              "cdo_groups": [
                {
                  "name": "DPU",
                  "type": "PRIMARY",
                  "pdi_id": "0x01",
                  "dpu_kernel_ids": [
                    "0x901"
                  ],
                  "pre_cdo_groups": [
                    "0xC1"
                  ]
                }
              ]
            }
          ]
        }
      }
    )";
    aiePartitionJsonOut->os() << aie_partition_json_data;
    aiePartitionJsonOut->keep();
  }

  // Create kernels.json.
  SmallString<64> kernelsJsonFile(TK.TempDir);
  sys::path::append(kernelsJsonFile, "kernels.json");
  {
    auto kernelsJsonOut = openOutputFile(kernelsJsonFile, &errorMessage);
    if (!kernelsJsonOut)
      return moduleOp.emitOpError(errorMessage);

    json::Object kernels_data{
        {"ps-kernels",
         json::Object{
             {"kernels",
              json::Array{// TODO: Support for multiple kernels
                          makeKernelJSON(TK.XCLBinKernelName, TK.XCLBinKernelID,
                                         TK.XCLBinInstanceName)}}}}};
    kernelsJsonOut->os() << formatv("{0:2}",
                                    json::Value(std::move(kernels_data)));
    kernelsJsonOut->keep();
  }
  // Create design.bif.
  SmallString<64> designBifFile(TK.TempDir);
  sys::path::append(designBifFile, "design.bif");
  {
    auto designBifOut = openOutputFile(designBifFile, &errorMessage);
    if (!designBifOut)
      return moduleOp.emitOpError(errorMessage);

    designBifOut->os() << "all:\n"
                       << "{\n"
                       << "\tid_code = 0x14ca8093\n"
                       << "\textended_id_code = 0x01\n"
                       << "\timage\n"
                       << "\t{\n"
                       << "\t\tname=aie_image, id=0x1c000000\n"
                       << "\t\t{ type=cdo\n"
                       << "\t\t  file=" << TK.TempDir
                       << "/aie_cdo_error_handling.bin\n"
                       << "\t\t  file=" << TK.TempDir << "/aie_cdo_elfs.bin\n"
                       << "\t\t  file=" << TK.TempDir << "/aie_cdo_init.bin\n"
                       << "\t\t  file=" << TK.TempDir << "/aie_cdo_enable.bin\n"
                       << "\t\t}\n"
                       << "\t}\n"
                       << "}";
    designBifOut->keep();
  }

  // Execute the bootgen command.
  SmallString<64> designPdiFile(TK.TempDir);
  sys::path::append(designPdiFile, "design.pdi");
  {
    SmallVector<std::string, 7> flags{"-arch",  "versal",
                                      "-image", std::string(designBifFile),
                                      "-o",     std::string(designPdiFile),
                                      "-w"};

    SmallString<64> bootgenBin(TK.InstallDir);
    sys::path::append(bootgenBin, "bin", "bootgen");
    if (runTool(bootgenBin, flags, TK.Verbose) != 0)
      return moduleOp.emitOpError("failed to execute bootgen");
  }

  // Execute the xclbinutil command.
  {
    std::string memArg =
        "MEM_TOPOLOGY:JSON:" + std::string(memTopologyJsonFile);
    std::string partArg =
        "AIE_PARTITION:JSON:" + std::string(aiePartitionJsonFile);
    SmallVector<std::string, 20> flags{"--add-replace-section",
                                       memArg,
                                       "--add-kernel",
                                       std::string(kernelsJsonFile),
                                       "--add-replace-section",
                                       partArg,
                                       "--force",
                                       "--output",
                                       std::string(Output)};

    if (auto xclbinutil = sys::findProgramByName("xclbinutil")) {
      if (runTool(*xclbinutil, flags, TK.Verbose) != 0)
        return moduleOp.emitOpError("failed to execute xclbinutil");
    } else {
      return moduleOp.emitOpError("could not find xclbinutil");
    }
  }
  return success();
}

static std::string chesshack(const std::string &input) {
  std::string result(input);
  static const std::unordered_map<std::string, std::string> substitutions{
      {"memory\\(none\\)", "readnone"},
      {"memory\\(read\\)", "readonly"},
      {"memory\\(write\\)", "writeonly"},
      {"memory\\(argmem: readwrite\\)", "argmemonly"},
      {"memory\\(argmem: read\\)", "argmemonly readonly"},
      {"memory\\(argmem: write\\)", "argmemonly writeonly"},
      {"memory\\(inaccessiblemem: write\\)", "inaccessiblememonly writeonly"},
      {"memory\\(inaccessiblemem: readwrite\\)", "inaccessiblememonly"},
      {"memory\\(inaccessiblemem: read\\)", "inaccessiblememonly readonly"},
      {"memory(argmem: readwrite, inaccessiblemem: readwrite)",
       "inaccessiblemem_or_argmemonly"},
      {"memory(argmem: read, inaccessiblemem: read)",
       "inaccessiblemem_or_argmemonly readonly"},
      {"memory(argmem: write, inaccessiblemem: write)",
       "inaccessiblemem_or_argmemonly writeonly"},
  };
  for (const auto &pair : substitutions)
    result = std::regex_replace(result, std::regex(pair.first), pair.second);
  return result;
}

// A pass which removes the alignment attribute from llvm load operations, if
// the alignment is less than 4 (2 or 1).
//
// Example replaces:
//
// ```
//  %113 = llvm.load %112 {alignment = 2 : i64} : !llvm.ptr -> vector<32xbf16>
// ```
//
// with
//
// ```
//  %113 = llvm.load %112 : !llvm.ptr -> vector<32xbf16>
// ```
//
// If this pass is not included in the pipeline, there is an alignment error
// later in the compilation. This is a temporary workaround while a better
// solution is found: propagation of memref.assume_alignment is one option. See
// also https://jira.xilinx.com/projects/AIECC/issues/AIECC-589
namespace {
struct RemoveAlignment2FromLLVMLoadPass
    : public PassWrapper<RemoveAlignment2FromLLVMLoadPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
        auto alignmentAttr = loadOp.getAlignmentAttr();
        if (alignmentAttr) {
          int alignmentVal = alignmentAttr.getValue().getSExtValue();
          if (alignmentVal == 2 || alignmentVal == 1) {
            loadOp.setAlignment(std::optional<uint64_t>());
          }
        }
      }
    });
  }

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RemoveAlignment2FromLLVMLoadPass);
};
} // namespace

static LogicalResult generateUnifiedObject(MLIRContext *context,
                                           ModuleOp moduleOp,
                                           XCLBinGenConfig &TK,
                                           const std::string &outputFile) {
  PassManager pm(context, moduleOp.getOperationName());
  applyConfigToPassManager(TK, pm);

  xilinx::xllvm::registerXLLVMDialectTranslation(*context);
  pm.addNestedPass<AIE::DeviceOp>(AIE::createAIELocalizeLocksPass());
  pm.addNestedPass<AIE::DeviceOp>(AIE::createAIENormalizeAddressSpacesPass());
  pm.addPass(AIE::createAIECoreToStandardPass());
  pm.addPass(AIEX::createAIEXToStandardPass());

  // Convert specific vector dialect ops (like vector.contract) to the AIEVec
  // dialect
  {
    xilinx::aievec::ConvertVectorToAIEVecOptions vectorToAIEVecOptions{};

    std::string optionsString = [&]() {
      std::ostringstream optionsStringStream;
      optionsStringStream << "target-backend=";
      optionsStringStream << (TK.UseChess ? "cpp" : "llvmir");
      optionsStringStream << ' ' << "aie-target=aieml";
      return optionsStringStream.str();
    }();

    if (failed(vectorToAIEVecOptions.parseFromString(optionsString))) {
      return moduleOp.emitOpError("Failed to parse options from '")
             << optionsString
             << "': Failed to construct ConvertVectorToAIEVecOptions.";
    }
    xilinx::aievec::buildConvertVectorToAIEVec(pm, vectorToAIEVecOptions);
  }

  addLowerToLLVMPasses(pm);
  pm.addPass(std::make_unique<RemoveAlignment2FromLLVMLoadPass>());

  if (TK.Verbose) {
    llvm::outs() << "Running: ";
    pm.printAsTextualPipeline(llvm::outs());
    llvm::outs() << "\n";
  }

  ModuleOp copy = moduleOp.clone();
  if (failed(pm.run(copy)))
    return moduleOp.emitOpError("Failed to lower to LLVM");

  SmallString<64> LLVMIRFile(TK.TempDir);
  sys::path::append(LLVMIRFile, "input.ll");

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(copy, llvmContext);
  if (!llvmModule)
    return moduleOp.emitOpError("Failed to translate module to LLVMIR");

  std::string errorMessage;
  {
    auto output = openOutputFile(LLVMIRFile, &errorMessage);
    if (!output)
      return moduleOp.emitOpError(errorMessage);
    llvmModule->print(output->os(), nullptr);
    output->keep();
  }

  if (TK.UseChess) {
    SmallString<64> chessWrapperBin(TK.InstallDir);
    sys::path::append(chessWrapperBin, "bin", "xchesscc_wrapper");

    SmallString<64> chessworkDir(TK.TempDir);
    sys::path::append(chessworkDir, "chesswork");

    SmallString<64> chessIntrinsicsCpp(TK.InstallDir);
    sys::path::append(chessIntrinsicsCpp, "aie_runtime_lib", TK.TargetArch,
                      "chess_intrinsic_wrapper.cpp");

    SmallString<64> chessIntrinsicsLL(TK.TempDir);
    sys::path::append(chessIntrinsicsLL, "chess_intrinsic_wrapper.ll");

    if (runTool(chessWrapperBin,
                {StringRef(TK.TargetArch).lower(), "+w",
                 std::string(chessworkDir), "-c", "-d", "-f", "+f", "+P", "4",
                 std::string(chessIntrinsicsCpp), "-o",
                 std::string(chessIntrinsicsLL)},
                TK.Verbose) != 0)
      return moduleOp.emitOpError("Failed to compile chess intrinsics");

    std::string newIntrinsicsLL;
    {
      auto chessIntrinsicIn = openInputFile(chessIntrinsicsLL, &errorMessage);
      if (!chessIntrinsicIn)
        moduleOp.emitOpError(errorMessage);

      newIntrinsicsLL =
          std::regex_replace(std::string(chessIntrinsicIn->getBuffer()),
                             std::regex("target datalayout.*"), "");
      newIntrinsicsLL = std::regex_replace(newIntrinsicsLL,
                                           std::regex("target triple.*"), "");
    }
    {
      auto chessIntrinsicOut = openOutputFile(chessIntrinsicsLL);
      if (!chessIntrinsicOut)
        moduleOp.emitOpError(errorMessage);

      chessIntrinsicOut->os() << newIntrinsicsLL;
      chessIntrinsicOut->keep();
    }

    std::string llvmirString;
    {
      raw_string_ostream llvmirStream(llvmirString);
      llvmModule->print(llvmirStream, nullptr);
    }

    SmallString<64> chesslinkedFile(TK.TempDir);
    sys::path::append(chesslinkedFile, "input.chesslinked.ll");
    SmallString<64> llvmLinkBin(TK.PeanoDir);
    sys::path::append(llvmLinkBin, "bin", "llvm-link");
    if (!sys::fs::exists(llvmLinkBin)) {
      if (auto llvmLink = sys::findProgramByName("llvm-link"))
        llvmLinkBin = *llvmLink;
      else
        moduleOp.emitOpError("Can't find llvm-link");
    }
    if (runTool(llvmLinkBin,
                {std::string(LLVMIRFile), std::string(chessIntrinsicsLL), "-S",
                 "-o", std::string(chesslinkedFile)},
                TK.Verbose) != 0)
      moduleOp.emitOpError("Couldn't link in the intrinsics");

    std::string mungedLLVMIR;
    {
      auto chesslinkedIn = openInputFile(chesslinkedFile, &errorMessage);
      if (!chesslinkedIn)
        moduleOp.emitOpError(errorMessage);

      mungedLLVMIR = std::string(chesslinkedIn->getBuffer());
      mungedLLVMIR = chesshack(mungedLLVMIR);
    }
    {
      auto chesslinkedOut = openOutputFile(chesslinkedFile);
      if (!chesslinkedOut)
        moduleOp.emitOpError(errorMessage);

      chesslinkedOut->os() << mungedLLVMIR;
      chesslinkedOut->keep();
    }

    if (runTool(chessWrapperBin,
                {StringRef(TK.TargetArch).lower(), "+w",
                 std::string(chessworkDir), "-c", "-d", "-f", "+P", "4",
                 std::string(chesslinkedFile), "-o", std::string(outputFile)},
                TK.Verbose) != 0)
      return moduleOp.emitOpError("Failed to assemble with chess");
  } else {
    SmallString<64> peanoOptBin(TK.PeanoDir);
    sys::path::append(peanoOptBin, "bin", "opt");
    SmallString<64> peanoLLCBin(TK.PeanoDir);
    sys::path::append(peanoLLCBin, "bin", "llc");

    SmallString<64> OptLLVMIRFile(TK.TempDir);
    sys::path::append(OptLLVMIRFile, "input.opt.ll");
    if (runTool(peanoOptBin,
                {"-O2", "--inline-threshold=10", "-S", std::string(LLVMIRFile),
                 "--disable-builtin=memset", "-o", std::string(OptLLVMIRFile)},
                TK.Verbose) != 0)
      return moduleOp.emitOpError("Failed to optimize");

    if (runTool(peanoLLCBin,
                {std::string(OptLLVMIRFile), "-O2",
                 "--march=" + StringRef(TK.TargetArch).lower(),
                 "--function-sections", "--filetype=obj", "-o",
                 std::string(outputFile)},
                TK.Verbose) != 0)
      return moduleOp.emitOpError("Failed to assemble");
  }
  copy->erase();
  return success();
}

LogicalResult xilinx::aie2xclbin(MLIRContext *ctx, ModuleOp moduleOp,
                                 XCLBinGenConfig &TK, StringRef OutputNPU,
                                 StringRef OutputXCLBin) {
  PassManager pm(ctx, moduleOp.getOperationName());
  applyConfigToPassManager(TK, pm);

  addAIELoweringPasses(pm);

  if (TK.Verbose) {
    llvm::outs() << "Running: ";
    pm.printAsTextualPipeline(llvm::outs());
    llvm::outs() << "\n";
  }

  if (failed(pm.run(moduleOp)))
    return moduleOp.emitOpError("AIE lowering pipline failed");

  raw_string_ostream target_arch_os(TK.TargetArch);
  if (failed(AIE::AIETranslateToTargetArch(moduleOp, target_arch_os)))
    return moduleOp.emitOpError("Couldn't detect target architure");

  TK.TargetArch = StringRef(TK.TargetArch).trim();

  std::regex target_regex("AIE.?");
  if (!std::regex_search(TK.TargetArch, target_regex))
    return moduleOp.emitOpError()
           << "Unexpected target architecture: " << TK.TargetArch;

  // generateNPUInstructions
  {
    PassManager pm(ctx, moduleOp.getOperationName());
    applyConfigToPassManager(TK, pm);

    pm.addNestedPass<AIE::DeviceOp>(AIEX::createAIEDmaToNpuPass());
    ModuleOp copy = moduleOp.clone();
    if (failed(pm.run(copy)))
      return moduleOp.emitOpError("NPU Instruction pipeline failed");

    std::string errorMessage;
    auto output = openOutputFile(OutputNPU, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      return moduleOp.emitOpError("");
    }

    if (failed(AIE::AIETranslateToNPU(copy, output->os())))
      return moduleOp.emitOpError("NPU Instruction translation failed");

    output->keep();
    copy->erase();
  }

  SmallString<64> unifiedObj(TK.TempDir);
  sys::path::append(unifiedObj, "input.o");
  if (failed(generateUnifiedObject(ctx, moduleOp, TK, std::string(unifiedObj))))
    return moduleOp.emitOpError("Failed to generate unified object");

  if (failed(generateCoreElfFiles(moduleOp, unifiedObj, TK)))
    return moduleOp.emitOpError("Failed to generate core ELF file(s)");

  if (failed(generateCDO(ctx, moduleOp, TK)))
    return moduleOp.emitOpError("Failed to generate CDO");

  if (failed(generateXCLBin(ctx, moduleOp, TK, OutputXCLBin)))
    return moduleOp.emitOpError("Failed to generate XCLBin");

  return success();
}
