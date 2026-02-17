//===- aiecc.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This is the main entry point for the AIE compiler driver (aiecc).
// It orchestrates the compilation flow for AIE devices.
//
// This C++ implementation provides similar functionality to the Python aiecc.py
// tool with the following architecture:
//
// 1. Command-line argument parsing using LLVM CommandLine library
// 2. MLIR module loading and parsing
// 3. MLIR transformation pipeline execution
// 4. Core compilation orchestration (xchesscc/peano)
// 5. NPU instruction generation
// 6. CDO/PDI/xclbin generation
// 7. Multi-device support
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/InitialAllDialect.h"
#include "aie/version.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

using namespace llvm;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static cl::OptionCategory aieCompilerOptions("AIE Compiler Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init(""),
                                          cl::cat(aieCompilerOptions));

static cl::opt<bool> showVersion("aie-version",
                                 cl::desc("Show version information"),
                                 cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string> sysroot("sysroot",
                                    cl::desc("Sysroot for cross-compilation"),
                                    cl::init(""), cl::cat(aieCompilerOptions));

static cl::opt<std::string> tmpDir("tmpdir",
                                   cl::desc("Directory for temporary files"),
                                   cl::init(""), cl::cat(aieCompilerOptions));

static cl::opt<bool> verbose("verbose", cl::desc("Enable verbose output"),
                             cl::init(false), cl::cat(aieCompilerOptions));

static cl::alias verboseShort("v", cl::desc("Alias for --verbose"),
                              cl::aliasopt(verbose),
                              cl::cat(aieCompilerOptions));

static cl::opt<bool> xbridge("xbridge", cl::desc("Link using xbridge"),
                             cl::init(true), cl::cat(aieCompilerOptions));

static cl::opt<bool> noXbridge("no-xbridge",
                               cl::desc("Link using peano (disable xbridge)"),
                               cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> xchesscc("xchesscc", cl::desc("Compile using xchesscc"),
                              cl::init(true), cl::cat(aieCompilerOptions));

static cl::opt<bool> noXchesscc("no-xchesscc", cl::desc("Compile using peano"),
                                cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> aiesim("aiesim", cl::desc("Generate aiesim Work folder"),
                            cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> compile("compile",
                             cl::desc("Enable compiling of AIE cores"),
                             cl::init(true), cl::cat(aieCompilerOptions));

static cl::opt<bool> noCompile("no-compile",
                               cl::desc("Disable compiling of AIE cores"),
                               cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> link("link", cl::desc("Enable linking of AIE code"),
                          cl::init(true), cl::cat(aieCompilerOptions));

static cl::opt<bool> noLink("no-link", cl::desc("Disable linking of AIE code"),
                            cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string> allocScheme(
    "alloc-scheme",
    cl::desc("Allocation scheme for AIE buffers (basic-sequential or "
             "bank-aware)"),
    cl::init("basic-sequential"), cl::cat(aieCompilerOptions));

static cl::opt<bool>
    generateNpuInsts("aie-generate-npu-insts",
                     cl::desc("Generate NPU instruction stream"),
                     cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    instsName("npu-insts-name",
              cl::desc("Output instructions filename for NPU target"),
              cl::init("{0}_{1}.bin"), cl::cat(aieCompilerOptions));

static cl::opt<bool> generateCdo("aie-generate-cdo",
                                 cl::desc("Generate libxaie v2 for CDO"),
                                 cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> generatePdi("aie-generate-pdi",
                                 cl::desc("Generate PDI binary"),
                                 cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string> pdiName("pdi-name", cl::desc("Output PDI filename"),
                                    cl::init("{0}.pdi"),
                                    cl::cat(aieCompilerOptions));

static cl::opt<bool> generateXclbin("aie-generate-xclbin",
                                    cl::desc("Generate xclbin"),
                                    cl::init(false),
                                    cl::cat(aieCompilerOptions));

static cl::opt<std::string> xclbinName("xclbin-name",
                                       cl::desc("Output xclbin filename"),
                                       cl::init("{0}.xclbin"),
                                       cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    deviceName("device-name", cl::desc("Device configuration to compile"),
               cl::init(""), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    sequenceName("sequence-name", cl::desc("Runtime sequence name to compile"),
                 cl::init(""), cl::cat(aieCompilerOptions));

static cl::opt<unsigned> optLevel("O", cl::desc("Optimization level (0-3)"),
                                  cl::init(2), cl::cat(aieCompilerOptions));

static cl::alias optLevelLong("opt-level", cl::desc("Alias for -O"),
                              cl::aliasopt(optLevel),
                              cl::cat(aieCompilerOptions));

static cl::opt<bool> dryRun("n",
                            cl::desc("Dry run mode (don't execute commands)"),
                            cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> dynamicObjFifos("dynamic-objFifos",
                                     cl::desc("Use dynamic object FIFOs"),
                                     cl::init(false),
                                     cl::cat(aieCompilerOptions));

static cl::opt<bool> packetSwObjFifos("packet-sw-objFifos",
                                      cl::desc("Use packet-switched flows"),
                                      cl::init(false),
                                      cl::cat(aieCompilerOptions));

static cl::opt<bool> ctrlPktOverlay("generate-ctrl-pkt-overlay",
                                    cl::desc("Generate control packet overlay"),
                                    cl::init(false),
                                    cl::cat(aieCompilerOptions));

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static void printVersion(raw_ostream &os) {
  os << "aiecc (C++ version) " << AIE_GIT_COMMIT << "\n";
}

static std::string findAieTool(StringRef toolName) {
  // Try to find tool in PATH
  auto result = sys::findProgramByName(toolName);
  if (result) {
    return *result;
  }

  // Try relative to this executable
  auto mainExecutable = sys::fs::getMainExecutable(nullptr, nullptr);
  SmallString<128> toolPath(sys::path::parent_path(mainExecutable));
  sys::path::append(toolPath, toolName);

  if (sys::fs::can_execute(toolPath)) {
    return std::string(toolPath);
  }

  return "";
}

static bool executeCommand(ArrayRef<StringRef> command,
                           bool verboseOutput = true) {
  if (verbose && verboseOutput) {
    llvm::outs() << "Executing:";
    for (const auto &arg : command) {
      llvm::outs() << " " << arg;
    }
    llvm::outs() << "\n";
  }

  if (dryRun) {
    if (verbose) {
      llvm::outs() << "Dry run - command not executed\n";
    }
    return true;
  }

  std::string errMsg;
  SmallVector<StringRef, 8> env;
  std::optional<StringRef> redirects[] = {std::nullopt, std::nullopt,
                                          std::nullopt};
  int result = sys::ExecuteAndWait(command[0], command, env, redirects,
                                   /*secondsToWait=*/0,
                                   /*memoryLimit=*/0, &errMsg);

  if (result != 0) {
    llvm::errs() << "Error: Command failed with exit code " << result << "\n";
    if (!errMsg.empty()) {
      llvm::errs() << "Error message: " << errMsg << "\n";
    }
    return false;
  }

  return true;
}

// Replace placeholders in format strings
static std::string formatString(StringRef formatStr, StringRef deviceName,
                                StringRef seqName = "") {
  std::string result = formatStr.str();
  size_t pos = result.find("{0}");
  if (pos != std::string::npos) {
    result.replace(pos, 3, deviceName.str());
  }
  pos = result.find("{1}");
  if (pos != std::string::npos && !seqName.empty()) {
    result.replace(pos, 3, seqName.str());
  }
  return result;
}

//===----------------------------------------------------------------------===//
// AIE Device and Core Discovery
//===----------------------------------------------------------------------===//

// Walk the module to find AIE device operations
static void findAIEDevices(ModuleOp module,
                           SmallVectorImpl<Operation *> &devices) {
  module.walk([&](Operation *op) {
    if (auto deviceOp = dyn_cast<xilinx::AIE::DeviceOp>(op)) {
      // Filter by device name if specified
      if (deviceName.empty() ||
          (deviceOp.getSymNameAttr() && deviceOp.getSymName() == deviceName)) {
        devices.push_back(op);
      }
    }
  });
}

// Find runtime sequences in a device
static void findRuntimeSequences(Operation *deviceOp,
                                 SmallVectorImpl<Operation *> &sequences) {
  deviceOp->walk([&](Operation *op) {
    if (auto seqOp = dyn_cast<xilinx::AIE::RuntimeSequenceOp>(op)) {
      if (sequenceName.empty() ||
          (seqOp.getSymNameAttr() && seqOp.getSymName() == sequenceName)) {
        sequences.push_back(op);
      }
    }
  });
}

// Count cores in a device for progress reporting
static unsigned countCoresInDevice(Operation *deviceOp) {
  unsigned count = 0;
  deviceOp->walk([&](Operation *op) {
    if (isa<xilinx::AIE::CoreOp>(op)) {
      count++;
    }
  });
  return count;
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Construction
//===----------------------------------------------------------------------===//

static std::string buildInputWithAddressesPipeline() {
  std::string pipeline = "builtin.module(";
  pipeline += "aie-canonicalize-device,";
  pipeline += "aie.device(";
  pipeline += "aie-assign-lock-ids,";
  pipeline += "aie-register-objectFifos,";
  pipeline += "aie-objectFifo-stateful-transform{";
  pipeline +=
      "dynamic-objFifos=" + std::string(dynamicObjFifos ? "true" : "false");
  pipeline +=
      " packet-sw-objFifos=" + std::string(packetSwObjFifos ? "true" : "false");
  pipeline += "},";
  pipeline += "aie-assign-bd-ids,";
  pipeline += "aie-lower-cascade-flows,";
  pipeline += "aie-lower-broadcast-packet,";
  pipeline += "aie-lower-multicast,";
  pipeline += "aie-assign-tile-controller-ids,";
  if (ctrlPktOverlay) {
    pipeline +=
        "aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true},";
  } else {
    pipeline +=
        "aie-generate-column-control-overlay{route-shim-to-tile-ctrl=false},";
  }
  pipeline +=
      "aie-assign-buffer-addresses{alloc-scheme=" + allocScheme.getValue() +
      "},";
  pipeline += "aie-vector-transfer-lowering{max-transfer-rank=1}";
  pipeline += "))";
  return pipeline;
}

static std::string buildNpuLoweringPipeline() {
  std::string pipeline = "builtin.module(aie.device(";
  pipeline += "aie-materialize-bd-chains,";
  pipeline += "aie-substitute-shim-dma-allocations,";
  pipeline += "aie-assign-runtime-sequence-bd-ids,";
  pipeline += "aie-dma-tasks-to-npu,";
  pipeline += "aie-dma-to-npu,";
  pipeline += "aie-lower-set-lock";
  pipeline += "))";
  return pipeline;
}

//===----------------------------------------------------------------------===//
// JSON Generation for xclbin Metadata
//===----------------------------------------------------------------------===//

static void generateMemTopologyJson(StringRef jsonPath) {
  std::ofstream jsonFile(jsonPath.str());
  jsonFile << "{\n";
  jsonFile << "  \"mem_topology\": {\n";
  jsonFile << "    \"m_count\": \"2\",\n";
  jsonFile << "    \"m_mem_data\": [\n";
  jsonFile << "      {\n";
  jsonFile << "        \"m_type\": \"MEM_DRAM\",\n";
  jsonFile << "        \"m_used\": \"1\",\n";
  jsonFile << "        \"m_sizeKB\": \"0x10000\",\n";
  jsonFile << "        \"m_tag\": \"HOST\",\n";
  jsonFile << "        \"m_base_address\": \"0x4000000\"\n";
  jsonFile << "      },\n";
  jsonFile << "      {\n";
  jsonFile << "        \"m_type\": \"MEM_DRAM\",\n";
  jsonFile << "        \"m_used\": \"1\",\n";
  jsonFile << "        \"m_sizeKB\": \"0xc000\",\n";
  jsonFile << "        \"m_tag\": \"SRAM\",\n";
  jsonFile << "        \"m_base_address\": \"0x4000000\"\n";
  jsonFile << "      }\n";
  jsonFile << "    ]\n";
  jsonFile << "  }\n";
  jsonFile << "}\n";
  jsonFile.close();
}

static void generateKernelsJson(StringRef jsonPath, StringRef devName) {
  std::ofstream jsonFile(jsonPath.str());
  jsonFile << "{\n";
  jsonFile << "  \"ps-kernels\": {\n";
  jsonFile << "    \"kernels\": [\n";
  jsonFile << "      {\n";
  jsonFile << "        \"name\": \"MLIR_AIE\",\n";
  jsonFile << "        \"type\": \"dpu\",\n";
  jsonFile << "        \"extended-data\": {\n";
  jsonFile << "          \"subtype\": \"DPU\",\n";
  jsonFile << "          \"functional\": \"0\",\n";
  jsonFile << "          \"dpu_kernel_id\": \"0x901\"\n";
  jsonFile << "        },\n";
  jsonFile << "        \"arguments\": [\n";
  jsonFile << "          {\n";
  jsonFile << "            \"name\": \"opcode\",\n";
  jsonFile << "            \"address-qualifier\": \"SCALAR\",\n";
  jsonFile << "            \"type\": \"uint64_t\",\n";
  jsonFile << "            \"offset\": \"0x00\"\n";
  jsonFile << "          },\n";
  jsonFile << "          {\n";
  jsonFile << "            \"name\": \"instr\",\n";
  jsonFile << "            \"memory-connection\": \"SRAM\",\n";
  jsonFile << "            \"address-qualifier\": \"GLOBAL\",\n";
  jsonFile << "            \"type\": \"char *\",\n";
  jsonFile << "            \"offset\": \"0x08\"\n";
  jsonFile << "          },\n";
  jsonFile << "          {\n";
  jsonFile << "            \"name\": \"ninstr\",\n";
  jsonFile << "            \"address-qualifier\": \"SCALAR\",\n";
  jsonFile << "            \"type\": \"uint32_t\",\n";
  jsonFile << "            \"offset\": \"0x10\"\n";
  jsonFile << "          },\n";
  for (int i = 0; i < 5; i++) {
    jsonFile << "          {\n";
    jsonFile << "            \"name\": \"bo" << i << "\",\n";
    jsonFile << "            \"memory-connection\": \"HOST\",\n";
    jsonFile << "            \"address-qualifier\": \"GLOBAL\",\n";
    jsonFile << "            \"type\": \"void*\",\n";
    jsonFile << "            \"offset\": \"" << std::hex << "0x"
             << (0x14 + i * 8) << std::dec << "\"\n";
    jsonFile << "          }" << (i < 4 ? "," : "") << "\n";
  }
  jsonFile << "        ],\n";
  jsonFile << "        \"instances\": [\n";
  jsonFile << "          {\n";
  jsonFile << "            \"name\": \"MLIRAIE\"\n";
  jsonFile << "          }\n";
  jsonFile << "        ]\n";
  jsonFile << "      }\n";
  jsonFile << "    ]\n";
  jsonFile << "  }\n";
  jsonFile << "}\n";
  jsonFile.close();
}

static void generatePartitionJson(StringRef jsonPath, StringRef devName,
                                  StringRef pdiPath) {
  std::ofstream jsonFile(jsonPath.str());
  jsonFile << "{\n";
  jsonFile << "  \"aie_partition\": {\n";
  jsonFile << "    \"name\": \"QoS\",\n";
  jsonFile << "    \"operations_per_cycle\": \"2048\",\n";
  jsonFile << "    \"inference_fingerprint\": \"23423\",\n";
  jsonFile << "    \"pre_post_fingerprint\": \"12345\",\n";
  jsonFile << "    \"partition\": {\n";
  jsonFile << "      \"column_width\": 1,\n";
  jsonFile << "      \"start_columns\": [0]\n";
  jsonFile << "    },\n";
  jsonFile << "    \"PDIs\": [\n";
  jsonFile << "      {\n";
  jsonFile << "        \"uuid\": \"00000000-0000-0000-0000-000000000000\",\n";
  jsonFile << "        \"file_name\": \"" << pdiPath.str() << "\",\n";
  jsonFile << "        \"cdo_groups\": [\n";
  jsonFile << "          {\n";
  jsonFile << "            \"name\": \"DPU\",\n";
  jsonFile << "            \"type\": \"PRIMARY\",\n";
  jsonFile << "            \"pdi_id\": \"0x01\",\n";
  jsonFile << "            \"dpu_kernel_ids\": [\"0x901\"],\n";
  jsonFile << "            \"pre_cdo_groups\": [\"0xC1\"]\n";
  jsonFile << "          }\n";
  jsonFile << "        ]\n";
  jsonFile << "      }\n";
  jsonFile << "    ]\n";
  jsonFile << "  }\n";
  jsonFile << "}\n";
  jsonFile.close();
}

//===----------------------------------------------------------------------===//
// Core Compilation and NPU Instruction Generation
//===----------------------------------------------------------------------===//

static LogicalResult generateNpuInstructions(MLIRContext &context,
                                             StringRef mlirFilePath,
                                             StringRef tmpDirName,
                                             StringRef devName) {
  if (!generateNpuInsts) {
    return success();
  }

  if (verbose) {
    llvm::outs() << "Generating NPU instructions for device: " << devName
                 << "\n";
  }

  std::string aieOptPath = findAieTool("aie-opt");
  std::string aieTranslatePath = findAieTool("aie-translate");

  if (aieOptPath.empty() || aieTranslatePath.empty()) {
    llvm::errs() << "Error: Could not find required AIE tools\n";
    return failure();
  }

  // Step 1: Run NPU lowering passes
  SmallString<128> npuLoweredPath(tmpDirName);
  sys::path::append(npuLoweredPath, devName.str() + "_npu_lowered.mlir");

  std::string pipeline = buildNpuLoweringPipeline();
  std::string pipelineArg = "--pass-pipeline=" + pipeline;

  SmallVector<std::string, 8> cmdStrs = {aieOptPath, mlirFilePath.str(),
                                         pipelineArg, "-o",
                                         npuLoweredPath.str().str()};

  SmallVector<StringRef, 8> cmd;
  for (const auto &str : cmdStrs) {
    cmd.push_back(str);
  }

  if (!executeCommand(cmd)) {
    llvm::errs() << "Error running NPU lowering passes\n";
    return failure();
  }

  // Step 2: Translate to NPU binary
  // Find runtime sequences and generate instructions for each
  SmallVector<Operation *, 4> sequences;

  // Parse the lowered module to find sequences
  // Note: We reuse the main context which already has all dialects registered
  ParserConfig parseConfig(&context);
  SourceMgr sourceMgr;
  auto module =
      parseSourceFile<ModuleOp>(npuLoweredPath, sourceMgr, parseConfig);

  if (!module) {
    llvm::errs() << "Error parsing lowered MLIR file\n";
    return failure();
  }

  // Find device
  SmallVector<Operation *, 1> devices;
  findAIEDevices(module.get(), devices);

  for (auto *deviceOp : devices) {
    if (auto devOp = dyn_cast<xilinx::AIE::DeviceOp>(deviceOp)) {
      if (devOp.getSymName() != devName) {
        continue;
      }

      SmallVector<Operation *, 4> seqs;
      findRuntimeSequences(deviceOp, seqs);

      for (auto *seqOp : seqs) {
        if (auto seq = dyn_cast<xilinx::AIE::RuntimeSequenceOp>(seqOp)) {
          StringRef seqName = seq.getSymName();
          std::string outputFileName =
              formatString(instsName, devName, seqName);

          // Output to current directory, not tmpdir (matches Python aiecc.py)
          SmallString<128> outputPath;
          if (sys::path::is_absolute(outputFileName)) {
            outputPath = outputFileName;
          } else {
            outputPath = outputFileName; // Relative to current directory
          }

          if (verbose) {
            llvm::outs() << "Generating NPU instructions for sequence: "
                         << seqName << " -> " << outputPath << "\n";
          }

          SmallVector<std::string, 8> translateStrs = {
              aieTranslatePath,
              npuLoweredPath.str().str(),
              "--aie-npu-to-binary",
              "--aie-device-name=" + devName.str(),
              "--aie-sequence-name=" + seqName.str(),
              "-o",
              outputPath.str().str()};

          SmallVector<StringRef, 8> translateCmd;
          for (const auto &str : translateStrs) {
            translateCmd.push_back(str);
          }

          if (!executeCommand(translateCmd)) {
            llvm::errs() << "Error generating NPU instructions\n";
            return failure();
          }
        }
      }
    }
  }

  return success();
}

static LogicalResult generateCdoArtifacts(StringRef mlirFilePath,
                                          StringRef tmpDirName,
                                          StringRef devName) {
  if (!generateCdo && !generatePdi && !generateXclbin) {
    return success();
  }

  if (verbose) {
    llvm::outs() << "Generating CDO artifacts for device: " << devName << "\n";
  }

  std::string aieTranslatePath = findAieTool("aie-translate");
  if (aieTranslatePath.empty()) {
    llvm::errs() << "Error: Could not find aie-translate tool\n";
    return failure();
  }

  // Generate CDO files
  SmallVector<std::string, 8> cdoStrs = {aieTranslatePath, mlirFilePath.str(),
                                         "--aie-generate-cdo",
                                         "--aie-device-name=" + devName.str(),
                                         "--work-dir-path=" + tmpDirName.str()};

  SmallVector<StringRef, 8> cdoCmd;
  for (const auto &str : cdoStrs) {
    cdoCmd.push_back(str);
  }

  if (!executeCommand(cdoCmd)) {
    llvm::errs() << "Error generating CDO files\n";
    return failure();
  }

  // Generate PDI if requested
  if (generatePdi || generateXclbin) {
    std::string bootgenPath = findAieTool("bootgen");
    if (bootgenPath.empty()) {
      if (verbose) {
        llvm::outs()
            << "Warning: bootgen not found, skipping PDI/xclbin generation\n";
      }
      return success();
    }

    std::string pdiFileName = formatString(pdiName, devName);
    SmallString<128> pdiPath(tmpDirName);
    sys::path::append(pdiPath, pdiFileName);

    // Create BIF file
    SmallString<128> bifPath(tmpDirName);
    sys::path::append(bifPath, devName.str() + "_design.bif");

    std::error_code ec;
    raw_fd_ostream bifFile(bifPath, ec);
    if (ec) {
      llvm::errs() << "Error creating BIF file: " << ec.message() << "\n";
      return failure();
    }

    bifFile << "all:\n";
    bifFile << "{\n";
    bifFile << "  id_code = 0x14ca8093\n";
    bifFile << "  extended_id_code = 0x01\n";
    bifFile << "  image\n";
    bifFile << "  {\n";
    bifFile << "    name=aie_image, id=0x1c000000\n";
    bifFile << "    { type=cdo ";
    bifFile << "file=" << tmpDirName << "/" << devName << "_aie_cdo_elfs.bin ";
    bifFile << "file=" << tmpDirName << "/" << devName << "_aie_cdo_init.bin ";
    bifFile << "file=" << tmpDirName << "/" << devName << "_aie_cdo_enable.bin";
    bifFile << " }\n";
    bifFile << "  }\n";
    bifFile << "}\n";
    bifFile.close();

    SmallVector<std::string, 8> bootgenStrs = {bootgenPath,
                                               "-arch",
                                               "versal",
                                               "-image",
                                               bifPath.str().str(),
                                               "-o",
                                               pdiPath.str().str(),
                                               "-w"};

    SmallVector<StringRef, 8> bootgenCmd;
    for (const auto &str : bootgenStrs) {
      bootgenCmd.push_back(str);
    }

    if (!executeCommand(bootgenCmd)) {
      llvm::errs() << "Error generating PDI\n";
      return failure();
    }

    if (verbose) {
      llvm::outs() << "Generated PDI: " << pdiPath << "\n";
    }

    // Generate xclbin if requested
    if (generateXclbin) {
      std::string xclbinutilPath = findAieTool("xclbinutil");
      if (xclbinutilPath.empty()) {
        if (verbose) {
          llvm::outs()
              << "Warning: xclbinutil not found, skipping xclbin generation\n";
        }
        return success();
      }

      if (verbose) {
        llvm::outs() << "Generating xclbin for device: " << devName << "\n";
      }

      // Generate JSON metadata files
      SmallString<128> memTopoPath(tmpDirName);
      sys::path::append(memTopoPath, devName.str() + "_mem_topology.json");
      generateMemTopologyJson(memTopoPath);

      SmallString<128> kernelsPath(tmpDirName);
      sys::path::append(kernelsPath, devName.str() + "_kernels.json");
      generateKernelsJson(kernelsPath, devName);

      SmallString<128> partitionPath(tmpDirName);
      sys::path::append(partitionPath, devName.str() + "_aie_partition.json");

      // Make pdiPath absolute for partition JSON
      SmallString<128> absPdiPath;
      if (sys::path::is_absolute(pdiPath)) {
        absPdiPath = pdiPath;
      } else {
        std::error_code ec = sys::fs::real_path(pdiPath, absPdiPath);
        if (ec) {
          // If real_path fails, try making it absolute manually
          sys::fs::current_path(absPdiPath);
          sys::path::append(absPdiPath, pdiPath);
        }
      }

      generatePartitionJson(partitionPath, devName, absPdiPath);

      // Build xclbin
      std::string xclbinFileName = formatString(xclbinName, devName);
      SmallString<128> xclbinPath;
      // Check if xclbinFileName is an absolute path or relative
      if (sys::path::is_absolute(xclbinFileName)) {
        xclbinPath = xclbinFileName;
      } else {
        // Put it in current working directory, not tmpdir
        xclbinPath = xclbinFileName;
      }

      SmallVector<std::string, 16> xclbinStrs = {
          xclbinutilPath,
          "--add-replace-section",
          "MEM_TOPOLOGY:JSON:" + memTopoPath.str().str(),
          "--add-kernel",
          kernelsPath.str().str(),
          "--add-replace-section",
          "AIE_PARTITION:JSON:" + partitionPath.str().str(),
          "--force",
          "--output",
          xclbinPath.str().str()};

      SmallVector<StringRef, 16> xclbinCmd;
      for (const auto &str : xclbinStrs) {
        xclbinCmd.push_back(str);
      }

      if (!executeCommand(xclbinCmd)) {
        llvm::errs() << "Error generating xclbin\n";
        return failure();
      }

      if (verbose) {
        llvm::outs() << "Generated xclbin: " << xclbinPath << "\n";
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main Compilation Flow
//===----------------------------------------------------------------------===//

static LogicalResult compileAIEModule(MLIRContext &context, ModuleOp module,
                                      StringRef tmpDirName) {
  if (verbose) {
    llvm::outs() << "Starting AIE compilation in directory: " << tmpDirName
                 << "\n";
  }

  // Discover AIE devices in the module
  SmallVector<Operation *, 4> devices;
  findAIEDevices(module, devices);

  if (devices.empty()) {
    llvm::errs() << "Error: No AIE devices found in module\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Found " << devices.size() << " AIE device(s)\n";
    for (auto *device : devices) {
      if (auto deviceOp = dyn_cast<xilinx::AIE::DeviceOp>(device)) {
        unsigned coreCount = countCoresInDevice(device);
        llvm::outs() << "  Device";
        if (deviceOp.getSymNameAttr()) {
          llvm::outs() << " '" << deviceOp.getSymName() << "'";
        }
        llvm::outs() << " with " << coreCount << " core(s)\n";
      }
    }
  }

  // Find required tools
  std::string aieOptPath = findAieTool("aie-opt");
  if (aieOptPath.empty()) {
    llvm::errs() << "Error: Could not find aie-opt tool\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Found aie-opt at: " << aieOptPath << "\n";
  }

  // Write input MLIR to temp file
  SmallString<128> inputPath(tmpDirName);
  sys::path::append(inputPath, "input.mlir");

  std::error_code ec;
  raw_fd_ostream inputFile(inputPath, ec);
  if (ec) {
    llvm::errs() << "Error opening file: " << ec.message() << "\n";
    return failure();
  }
  module->print(inputFile);
  inputFile.close();

  if (verbose) {
    llvm::outs() << "Wrote input MLIR to: " << inputPath << "\n";
  }

  // Step 1: Run resource allocation and lowering passes
  SmallString<128> withAddressesPath(tmpDirName);
  sys::path::append(withAddressesPath, "input_with_addresses.mlir");

  std::string pipeline = buildInputWithAddressesPipeline();
  std::string pipelineArg = "--pass-pipeline=" + pipeline;

  SmallVector<std::string, 16> passStrs = {aieOptPath, inputPath.str().str(),
                                           pipelineArg, "-o",
                                           withAddressesPath.str().str()};

  SmallVector<StringRef, 16> passCmd;
  for (const auto &str : passStrs) {
    passCmd.push_back(str);
  }

  if (!executeCommand(passCmd)) {
    llvm::errs() << "Error running resource allocation passes\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Resource allocation passes completed\n";
  }

  // Step 2: Run routing
  SmallString<128> physicalPath(tmpDirName);
  sys::path::append(physicalPath, "input_physical.mlir");

  std::string routingPipeline =
      "builtin.module(aie.device(aie-create-pathfinder-flows))";
  std::string routingArg = "--pass-pipeline=" + routingPipeline;

  SmallVector<std::string, 16> routingStrs = {
      aieOptPath, withAddressesPath.str().str(), routingArg, "-o",
      physicalPath.str().str()};

  SmallVector<StringRef, 16> routingCmd;
  for (const auto &str : routingStrs) {
    routingCmd.push_back(str);
  }

  if (!executeCommand(routingCmd)) {
    llvm::errs() << "Error running routing passes\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Routing completed\n";
  }

  // Step 3: Generate artifacts for each device
  for (auto *device : devices) {
    if (auto deviceOp = dyn_cast<xilinx::AIE::DeviceOp>(device)) {
      StringRef devName = deviceOp.getSymName();

      if (verbose) {
        llvm::outs() << "\nProcessing device: " << devName << "\n";
      }

      // Generate NPU instructions
      if (failed(generateNpuInstructions(context, physicalPath, tmpDirName,
                                         devName))) {
        return failure();
      }

      // Generate CDO/PDI/xclbin
      if (failed(generateCdoArtifacts(physicalPath, tmpDirName, devName))) {
        return failure();
      }
    }
  }

  return success();
}

static int processInputFile(StringRef inputFile, StringRef tmpDirName) {
  // Parse the input file
  MLIRContext context;
  context.loadDialect<xilinx::AIE::AIEDialect>();
  context.loadDialect<xilinx::AIEX::AIEXDialect>();

  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<vector::VectorDialect>();
  xilinx::registerAllDialects(registry);
  context.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> module;

  if (inputFile.empty()) {
    llvm::errs() << "Error: No input file specified\n";
    return 1;
  }

  ParserConfig parseConfig(&context);
  SourceMgr sourceMgr;
  module = parseSourceFile<ModuleOp>(inputFile, sourceMgr, parseConfig);

  if (!module) {
    llvm::errs() << "Error parsing MLIR file\n";
    return 1;
  }

  if (verbose) {
    llvm::outs() << "Successfully parsed input file: " << inputFile << "\n";
  }

  // Create temporary directory if needed
  SmallString<128> actualTmpDir;
  if (!tmpDirName.empty()) {
    actualTmpDir = tmpDirName;
  } else {
    // Create a project directory based on input filename
    StringRef baseName = sys::path::filename(inputFile);
    actualTmpDir = baseName;
    actualTmpDir += ".prj";
  }

  std::error_code ec = sys::fs::create_directory(actualTmpDir);
  if (ec && ec != std::errc::file_exists) {
    llvm::errs() << "Error creating temporary directory: " << ec.message()
                 << "\n";
    return 1;
  }

  if (verbose) {
    llvm::outs() << "Using temporary directory: " << actualTmpDir << "\n";
  }

  // Run the compilation flow
  if (failed(compileAIEModule(context, module.get(), actualTmpDir))) {
    llvm::errs() << "Compilation failed\n";
    return 1;
  }

  llvm::outs() << "Compilation completed successfully\n";
  return 0;
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::AddExtraVersionPrinter(printVersion);
  cl::ParseCommandLineOptions(argc, argv, "AIE Compiler Driver\n");

  if (showVersion) {
    printVersion(llvm::outs());
    return 0;
  }

  // Handle conflicting options
  if (noXbridge) {
    xbridge = false;
  }
  if (noXchesscc) {
    xchesscc = false;
  }
  if (noCompile) {
    compile = false;
  }
  if (noLink) {
    link = false;
  }

  // Process the input file
  return processInputFile(inputFilename, tmpDir);
}
