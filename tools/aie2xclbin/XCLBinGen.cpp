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

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/InitialAllDialect.h"
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
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"

#include <regex>

extern "C" {
#include "cdo_driver.h"
}

#ifdef _WIN32
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

using namespace llvm;
using namespace mlir;
using namespace xilinx;

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
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  ConvertFuncToLLVMPassOptions opts;
  opts.useBarePtrCallConv = true;
  pm.addPass(createConvertFuncToLLVMPass(opts));
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

int runTool(StringRef Program, ArrayRef<std::string> Args, bool Verbose,
            std::optional<ArrayRef<StringRef>> Env = std::nullopt) {
  if (Verbose) {
    llvm::outs() << "Run:";
    if (Env) {
      for (auto &s : *Env) {
        llvm::outs() << " " << s;
      }
    }
    llvm::outs() << " " << Program;
    for (auto &s : Args) {
      llvm::outs() << " " << s;
    }
    llvm::outs() << "\n";
  }
  std::string err_msg;
  sys::ProcessStatistics stats;
  std::optional<sys::ProcessStatistics> opt_stats(stats);
  SmallVector<StringRef, 8> PArgs = {Program};
  PArgs.append(Args.begin(), Args.end());
  int result = sys::ExecuteAndWait(Program, PArgs, Env, {}, 0, 0, &err_msg,
                                   nullptr, &opt_stats);
  if (Verbose) {
    llvm::outs() << (result == 0 ? "Succeeded " : "Failed ") << "in "
                 << std::chrono::duration_cast<std::chrono::duration<float>>(
                        stats.TotalTime)
                        .count()
                 << " code: " << result << "\n";
  }
  return result;
}

template <unsigned N>
static void aieTargetDefines(SmallVector<std::string, N> &Args,
                             std::string aie_target) {
  if (aie_target == "AIE2") {
    Args.push_back("-D__AIEARCH__=20");
  } else {
    Args.push_back("-D__AIEARCH__=10");
  }
}

// Generate the elf files for the core
static LogicalResult generateCoreElfFiles(ModuleOp moduleOp,
                                          const StringRef objFile,
                                          XCLBinGenConfig &TK) {
  auto deviceOps = moduleOp.getOps<AIE::DeviceOp>();
  if (!llvm::hasSingleElement(deviceOps)) {
    return moduleOp.emitOpError("expected a single device op");
  }

  AIE::DeviceOp deviceOp = *deviceOps.begin();
  auto tileOps = deviceOp.getOps<AIE::TileOp>();

  std::string errorMessage;

  for (auto tileOp : tileOps) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    auto coreOp = tileOp.getCoreOp();
    if (!coreOp) {
      continue;
    }

    std::string elfFileName;
    if (auto fileAttr = coreOp.getElfFileAttr()) {
      elfFileName = std::string(fileAttr.getValue());
    } else {
      elfFileName = std::string("core_") + std::to_string(col) + "_" +
                    std::to_string(row) + ".elf";
      std::cout << "assigned core elf name " << elfFileName << std::endl;
      coreOp.setElfFile(elfFileName);
    }

    SmallString<64> ldscript_path(TK.TempDir);
    sys::path::append(ldscript_path, elfFileName + ".ld");
    {
      auto ldscript_output = openOutputFile(ldscript_path, &errorMessage);
      if (!ldscript_output) {
        return coreOp.emitOpError(errorMessage);
      }

      if (failed(AIE::AIETranslateToLdScript(moduleOp, ldscript_output->os(),
                                             col, row))) {
        return coreOp.emitOpError("failed to generate ld script for core (")
               << col << "," << row << ")";
      }
      ldscript_output->keep();
    }

    // We are running a clang command for now, but really this is an lld
    // command.
    SmallString<64> elfFile(TK.TempDir);
    sys::path::append(elfFile, elfFileName);
    {
      std::string targetLower = StringRef(TK.TargetArch).lower();
      SmallVector<std::string, 10> flags;
      flags.push_back("-O2");
      std::string targetFlag = "--target=" + targetLower + "-none-elf";
      flags.push_back(targetFlag);
      flags.emplace_back(objFile);
      SmallString<64> meBasicPath(TK.InstallDir);
      sys::path::append(meBasicPath, "aie_runtime_lib", TK.TargetArch,
                        "me_basic.o");
      flags.emplace_back(meBasicPath);
      SmallString<64> libcPath(TK.PeanoDir);
      sys::path::append(libcPath, "lib", targetLower + "-none-unknown-elf",
                        "libc.a");
      flags.emplace_back(libcPath);
      flags.push_back("-Wl,--gc-sections");
      std::string ldScriptFlag = "-Wl,-T," + std::string(ldscript_path);
      flags.push_back(ldScriptFlag);
      flags.push_back("-o");
      flags.emplace_back(elfFile);
      SmallString<64> clangBin(TK.PeanoDir);
      sys::path::append(clangBin, "bin", "clang");
      if (runTool(clangBin, flags, TK.Verbose) != 0) {
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
  passManager.addNestedPass<AIE::DeviceOp>(AIE::createAIEPathfinderPass());
  passManager.addNestedPass<AIE::DeviceOp>(
      AIEX::createAIEBroadcastPacketPass());
  passManager.addNestedPass<AIE::DeviceOp>(
      AIE::createAIERoutePacketFlowsPass());
  passManager.addNestedPass<AIE::DeviceOp>(AIEX::createAIELowerMulticastPass());
  if (failed(passManager.run(copy)))
    return moduleOp.emitOpError(
        "failed to run passes to prepare of XCLBin generation");

  if (failed(AIE::AIETranslateToCDODirect(
          copy, TK.TempDir, byte_ordering::Little_Endian, false, false, false)))
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
    if (!memTopologyJsonOut) {
      return moduleOp.emitOpError(errorMessage);
    }

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
    if (!aiePartitionJsonOut) {
      return moduleOp.emitOpError(errorMessage);
    }
    std::string aie_partition_json_data = R"(
      {
        "aie_partition": {
          "name": "QoS",
          "operations_per_cycle": "2048",
          "inference_fingerprint": "23423",
          "pre_post_fingerprint": "12345",
          "partition": {
            "column_width": 1,
            "start_columns": [
              1,
              2,
              3,
              4
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
    if (!kernelsJsonOut) {
      return moduleOp.emitOpError(errorMessage);
    }
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
    if (!designBifOut) {
      return moduleOp.emitOpError(errorMessage);
    }

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

    if (auto bootgen = sys::findProgramByName("bootgen")) {
      if (runTool(*bootgen, flags, TK.Verbose) != 0) {
        return moduleOp.emitOpError("failed to execute bootgen");
      }
    } else {
      return moduleOp.emitOpError("could not find bootgen");
    }
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
      if (runTool(*xclbinutil, flags, TK.Verbose) != 0) {
        return moduleOp.emitOpError("failed to execute xclbinutil");
      }
    } else {
      return moduleOp.emitOpError("could not find xclbinutil");
    }
  }
  return success();
}

LogicalResult xilinx::aie2xclbin(MLIRContext *ctx, ModuleOp moduleOp,
                                 XCLBinGenConfig &TK, StringRef OutputIPU,
                                 StringRef OutputXCLBin) {
  PassManager pm(ctx, moduleOp.getOperationName());
  addAIELoweringPasses(pm);

  if (TK.Verbose) {
    llvm::outs() << "Running: ";
    pm.printAsTextualPipeline(llvm::outs());
    llvm::outs() << "\n";
  }

  if (failed(pm.run(moduleOp))) {
    return moduleOp.emitOpError("AIE lowering pipline failed");
  }

  raw_string_ostream target_arch_os(TK.TargetArch);
  if (failed(AIE::AIETranslateToTargetArch(moduleOp, target_arch_os))) {
    return moduleOp.emitOpError("Couldn't detect target architure");
  }

  TK.TargetArch = StringRef(TK.TargetArch).trim();

  std::regex target_regex("AIE.?");
  if (!std::regex_search(TK.TargetArch, target_regex)) {
    return moduleOp.emitOpError()
           << "Unexpected target architecture: " << TK.TargetArch;
  }

  // generateIPUInstructions
  {
    PassManager pm(ctx, moduleOp.getOperationName());
    pm.addNestedPass<AIE::DeviceOp>(AIEX::createAIEDmaToIpuPass());
    ModuleOp copy = moduleOp.clone();
    if (failed(pm.run(copy))) {
      return moduleOp.emitOpError("IPU Instruction pipeline failed");
    }

    std::string errorMessage;
    auto output = openOutputFile(OutputIPU, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      return moduleOp.emitOpError("");
    }

    if (failed(AIE::AIETranslateToIPU(copy, output->os()))) {
      return moduleOp.emitOpError("IPU Instruction translation failed");
    }

    output->keep();
    copy->erase();
  }

  SmallString<64> peanoOptBin(TK.PeanoDir);
  sys::path::append(peanoOptBin, "bin", "opt");
  SmallString<64> peanoLLCBin(TK.PeanoDir);
  sys::path::append(peanoLLCBin, "bin", "llc");

  // generateObjectFile
  SmallString<64> unifiedObj(TK.TempDir);
  sys::path::append(unifiedObj, "input.o");
  {
    PassManager pm(ctx, moduleOp.getOperationName());
    pm.addNestedPass<AIE::DeviceOp>(AIE::createAIELocalizeLocksPass());
    pm.addNestedPass<AIE::DeviceOp>(AIE::createAIENormalizeAddressSpacesPass());
    pm.addPass(AIE::createAIECoreToStandardPass());
    pm.addPass(AIEX::createAIEXToStandardPass());
    addLowerToLLVMPasses(pm);

    if (TK.Verbose) {
      llvm::outs() << "Running: ";
      pm.printAsTextualPipeline(llvm::outs());
      llvm::outs() << "\n";
    }

    ModuleOp copy = moduleOp.clone();
    if (failed(pm.run(copy))) {
      return moduleOp.emitOpError("Failed to lower to LLVM");
    }

    SmallString<64> LLVMIRFile(TK.TempDir);
    sys::path::append(LLVMIRFile, "input.ll");

    std::string errorMessage;
    auto output = openOutputFile(LLVMIRFile, &errorMessage);
    if (!output) {
      return moduleOp.emitOpError(errorMessage);
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(copy, llvmContext);
    if (!llvmModule)
      return moduleOp.emitOpError("Failed to translate module to LLVMIR");

    llvmModule->print(output->os(), nullptr);
    output->keep();

    SmallString<64> OptLLVMIRFile(TK.TempDir);
    sys::path::append(OptLLVMIRFile, "input.opt.ll");
    if (runTool(peanoOptBin,
                {"-O2", "--inline-threshold=10", "-S", std::string(LLVMIRFile),
                 "-o", std::string(OptLLVMIRFile)},
                TK.Verbose) != 0) {
      return moduleOp.emitOpError("Failed to optimize");
    }
    if (runTool(peanoLLCBin,
                {std::string(OptLLVMIRFile), "-O2",
                 "--march=" + StringRef(TK.TargetArch).lower(),
                 "--function-sections", "--filetype=obj", "-o",
                 std::string(unifiedObj)},
                TK.Verbose) != 0) {
      return moduleOp.emitOpError("Failed to assemble");
    }
    copy->erase();
  }

  if (failed(generateCoreElfFiles(moduleOp, unifiedObj, TK))) {
    return moduleOp.emitOpError("Failed to generate core ELF file(s)");
  }

  if (failed(generateCDO(ctx, moduleOp, TK))) {
    return moduleOp.emitOpError("Failed to generate CDO");
  }

  if (failed(generateXCLBin(ctx, moduleOp, TK, OutputXCLBin))) {
    return moduleOp.emitOpError("Failed to generate XCLBin");
  }

  return success();
}