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
// 4. Core compilation (xchesscc/peano)
// 5. NPU instruction generation
// 6. CDO/PDI/xclbin generation
// 7. Multi-device support
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
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
#include <map>
#include <memory>
#include <optional>
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

static cl::opt<std::string>
    peanoInstallDir("peano", cl::desc("Peano compiler installation directory"),
                    cl::init(""), cl::cat(aieCompilerOptions));

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
  auto mainExecutable = sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&findAieTool));
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
  int result = sys::ExecuteAndWait(command[0], command, /*Env=*/std::nullopt,
                                   /*Redirects=*/{},
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

// Get Peano target triple from AIE target
static std::string getPeanoTarget(StringRef aieTarget) {
  std::string target = aieTarget.lower();
  return target + "-none-unknown-elf";
}

// Discover Peano installation directory (matching Python logic)
static std::string discoverPeanoInstallDir() {
  // 1. Check if --peano was specified
  if (!peanoInstallDir.empty()) {
    if (sys::fs::is_directory(peanoInstallDir)) {
      return peanoInstallDir;
    }
  }

  // 2. Check PEANO_INSTALL_DIR environment variable
  if (const char *peanoEnv = std::getenv("PEANO_INSTALL_DIR")) {
    if (sys::fs::is_directory(peanoEnv)) {
      return peanoEnv;
    }
  }

  // 3. Try relative to this executable (install area)
  auto mainExe = sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&discoverPeanoInstallDir));
  SmallString<256> exeDir(sys::path::parent_path(mainExe));
  // Go up from bin/ to install prefix
  SmallString<256> installPrefix(sys::path::parent_path(exeDir));

  SmallString<256> peanoDir(installPrefix);
  sys::path::append(peanoDir, "peano");
  if (sys::fs::is_directory(peanoDir)) {
    return std::string(peanoDir);
  }

  // 4. Try sibling peano directory (build area)
  SmallString<256> siblingPeano(sys::path::parent_path(installPrefix));
  sys::path::append(siblingPeano, "peano");
  if (sys::fs::is_directory(siblingPeano)) {
    return std::string(siblingPeano);
  }

  // 5. Try Python site-packages location (llvm-aie package)
  // This matches: sysconfig.get_path("platlib")/llvm-aie in Python
  // Common locations: ~/.local/lib/pythonX.Y/site-packages/llvm-aie
  //                   /path/to/venv/lib/pythonX.Y/site-packages/llvm-aie

  // Try VIRTUAL_ENV first if set
  if (const char *venvPath = std::getenv("VIRTUAL_ENV")) {
    SmallString<256> venvLlvmAie(venvPath);
    // Try common Python versions
    for (const char *pyver :
         {"python3.12", "python3.11", "python3.10", "python3.9"}) {
      SmallString<256> testPath(venvLlvmAie);
      sys::path::append(testPath, "lib", pyver, "site-packages", "llvm-aie");
      if (sys::fs::is_directory(testPath)) {
        return std::string(testPath);
      }
    }
  }

  // Try running python to get the actual site-packages path
  // This is a fallback that invokes python -c "import sysconfig;
  // print(sysconfig.get_path('platlib'))"
  SmallString<256> pythonSitePackages;
  {
    auto pythonPath = sys::findProgramByName("python3");
    if (!pythonPath) {
      pythonPath = sys::findProgramByName("python");
    }

    if (pythonPath) {
      // Use popen to capture output
      std::string cmd =
          *pythonPath + " -c \"import sysconfig; "
                        "print(sysconfig.get_path('platlib'))\" 2>/dev/null";
      FILE *pipe = popen(cmd.c_str(), "r");
      if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe)) {
          // Remove trailing newline
          std::string result(buffer);
          while (!result.empty() &&
                 (result.back() == '\n' || result.back() == '\r')) {
            result.pop_back();
          }
          SmallString<256> llvmAiePath(result);
          sys::path::append(llvmAiePath, "llvm-aie");
          if (sys::fs::is_directory(llvmAiePath)) {
            pclose(pipe);
            return std::string(llvmAiePath);
          }
        }
        pclose(pipe);
      }
    }
  }

  return "";
}

// Cached Peano install directory
static std::optional<std::string> cachedPeanoDir;

static StringRef getPeanoInstallDir() {
  if (!cachedPeanoDir.has_value()) {
    cachedPeanoDir = discoverPeanoInstallDir();
    if (verbose && !cachedPeanoDir->empty()) {
      llvm::outs() << "Discovered Peano installation: " << *cachedPeanoDir
                   << "\n";
    }
  }
  return *cachedPeanoDir;
}

// Find Peano compiler tools
static std::string findPeanoTool(StringRef toolName) {
  StringRef peanoDir = getPeanoInstallDir();

  if (!peanoDir.empty()) {
    SmallString<128> toolPath(peanoDir);
    sys::path::append(toolPath, "bin", toolName);
    if (sys::fs::can_execute(toolPath)) {
      return std::string(toolPath);
    }
  }

  // Try PATH as fallback
  auto result = sys::findProgramByName(toolName);
  if (result) {
    return *result;
  }

  return "";
}

// Helper to capture stdout from a command execution
static std::string executeAndCaptureOutput(ArrayRef<StringRef> command) {
  // Create temporary file for output
  SmallString<128> tmpPath;
  std::error_code ec =
      sys::fs::createTemporaryFile("aiecc_output", "txt", tmpPath);
  if (ec) {
    return "";
  }

  // Set up redirects: stdin=none, stdout=tmpPath, stderr=none
  std::optional<StringRef> redirects[3];
  redirects[0] = StringRef();        // stdin: none
  redirects[1] = StringRef(tmpPath); // stdout: our temp file
  redirects[2] = StringRef();        // stderr: none

  std::string errMsg;
  int result =
      sys::ExecuteAndWait(command[0], command, /*Env=*/std::nullopt, redirects,
                          /*secondsToWait=*/60,
                          /*memoryLimit=*/0, &errMsg);

  if (result != 0) {
    sys::fs::remove(tmpPath);
    return "";
  }

  // Read the output
  auto bufferOrErr = llvm::MemoryBuffer::getFile(tmpPath);
  sys::fs::remove(tmpPath);

  if (!bufferOrErr) {
    return "";
  }

  std::string output = bufferOrErr.get()->getBuffer().str();
  // Trim whitespace
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r' ||
                             output.back() == ' ' || output.back() == '\t')) {
    output.pop_back();
  }

  return output;
}

// Get the AIE target architecture for a device by running aie-translate
static std::string getAIETargetForDevice(StringRef mlirFilePath,
                                         StringRef deviceName) {
  std::string aieTranslatePath = findAieTool("aie-translate");
  if (aieTranslatePath.empty()) {
    if (verbose) {
      llvm::outs()
          << "Warning: Could not find aie-translate, defaulting to aie2\n";
    }
    return "aie2";
  }

  SmallVector<std::string, 6> cmdStrs = {
      aieTranslatePath, "--aie-generate-target-arch",
      "--aie-device-name=" + deviceName.str(), mlirFilePath.str()};

  SmallVector<StringRef, 6> cmd;
  for (const auto &str : cmdStrs) {
    cmd.push_back(str);
  }

  std::string target = executeAndCaptureOutput(cmd);

  if (target.empty()) {
    if (verbose) {
      llvm::outs() << "Warning: Could not determine target for device "
                   << deviceName << ", defaulting to aie2\n";
    }
    return "aie2";
  }

  if (verbose) {
    llvm::outs() << "Detected target architecture for device " << deviceName
                 << ": " << target << "\n";
  }

  return target;
}

//===----------------------------------------------------------------------===//
// AIE Device and Core Discovery
//===----------------------------------------------------------------------===//

struct CoreInfo {
  int col;
  int row;
  std::string linkWith; // External object files to link
  std::string elfFile;  // Generated ELF path (if already specified)
};

// Helper to extract core info from a CoreOp
static CoreInfo getCoreInfo(xilinx::AIE::CoreOp coreOp) {
  CoreInfo info;
  auto tileOp = dyn_cast<xilinx::AIE::TileOp>(coreOp.getTile().getDefiningOp());
  if (tileOp) {
    info.col = tileOp.getCol();
    info.row = tileOp.getRow();
  }

  if (auto linkWithAttr = coreOp.getLinkWithAttr()) {
    info.linkWith = linkWithAttr.getValue().str();
  }

  if (auto elfAttr = coreOp.getElfFileAttr()) {
    info.elfFile = elfAttr.getValue().str();
  }

  return info;
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Construction
//===----------------------------------------------------------------------===//

static std::string
buildInputWithAddressesPipeline(StringRef aieTarget = "aie2") {
  std::string pipeline = "builtin.module(";
  // These passes must come before the device passes (matching Python)
  pipeline += "convert-vector-to-aievec{aie-target=" + aieTarget.lower() +
              " target-backend=llvmir},";
  pipeline += "lower-affine,";
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
  pipeline += "),";                // close aie.device
  pipeline += "convert-scf-to-cf"; // Must come after device passes
  pipeline += ")";                 // close builtin.module
  return pipeline;
}

static std::string buildLLVMLoweringPipeline(StringRef deviceName,
                                             StringRef aieTarget = "aie2") {
  std::string deviceArg = "device=" + deviceName.str();

  // Matching Python's _create_aie_lower_to_llvm_pipeline +
  // LOWER_TO_LLVM_PIPELINE Note: Python does NOT pass tilecol/tilerow - the
  // pass processes all cores at once
  std::string pipeline = "builtin.module(";
  pipeline += "aie.device(aie-localize-locks,aie-normalize-address-spaces,aie-"
              "transform-bfp-types),";
  pipeline += "aie-standard-lowering{" + deviceArg + "},";
  pipeline += "aiex-standard-lowering,";
  pipeline += "convert-aievec-to-llvm{aie-target=" + aieTarget.lower() + "},";
  // LOWER_TO_LLVM_PIPELINE passes
  pipeline += "canonicalize,";
  pipeline += "cse,";
  pipeline += "expand-strided-metadata,";
  pipeline += "lower-affine,";
  pipeline += "arith-expand,";
  pipeline += "finalize-memref-to-llvm,";
  pipeline += "convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},";
  pipeline += "convert-to-llvm{dynamic=true},";
  pipeline += "canonicalize,";
  pipeline += "cse";
  pipeline += ")";
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
// Core Compilation
//===----------------------------------------------------------------------===//

struct CoreCompilationResult {
  std::string elfPath;
  bool success;
};

static LogicalResult compileCore(MLIRContext &context, StringRef deviceName,
                                 const CoreInfo &core,
                                 StringRef withAddressesPath,
                                 StringRef tmpDirName, StringRef aieTarget,
                                 std::string &outElfPath) {

  if (!compile) {
    // If we're not compiling, check if elf_file was already provided
    if (!core.elfFile.empty()) {
      outElfPath = core.elfFile;
      return success();
    }
    return success(); // Skip compilation
  }

  if (verbose) {
    llvm::outs() << "Compiling core (" << core.col << ", " << core.row << ")\n";
  }

  std::string aieOptPath = findAieTool("aie-opt");
  std::string aieTranslatePath = findAieTool("aie-translate");

  if (aieOptPath.empty() || aieTranslatePath.empty()) {
    llvm::errs() << "Error: Could not find required AIE tools\n";
    return failure();
  }

  // Step 1: Lower core to LLVM
  SmallString<128> coreLoweredPath(tmpDirName);
  sys::path::append(coreLoweredPath,
                    deviceName.str() + "_core_" + std::to_string(core.col) +
                        "_" + std::to_string(core.row) + "_lowered.mlir");

  std::string pipeline = buildLLVMLoweringPipeline(deviceName, aieTarget);
  std::string pipelineArg = "--pass-pipeline=" + pipeline;

  SmallVector<std::string, 8> lowerStrs = {aieOptPath, withAddressesPath.str(),
                                           pipelineArg, "-o",
                                           coreLoweredPath.str().str()};

  SmallVector<StringRef, 8> lowerCmd;
  for (const auto &str : lowerStrs) {
    lowerCmd.push_back(str);
  }

  if (!executeCommand(lowerCmd)) {
    llvm::errs() << "Error lowering core to LLVM\n";
    return failure();
  }

  // Step 2: Translate to LLVM IR
  SmallString<128> llvmIRPath(tmpDirName);
  sys::path::append(llvmIRPath, deviceName.str() + "_core_" +
                                    std::to_string(core.col) + "_" +
                                    std::to_string(core.row) + ".ll");

  SmallVector<std::string, 6> translateStrs = {
      aieTranslatePath, "--mlir-to-llvmir", coreLoweredPath.str().str(), "-o",
      llvmIRPath.str().str()};

  SmallVector<StringRef, 6> translateCmd;
  for (const auto &str : translateStrs) {
    translateCmd.push_back(str);
  }

  if (!executeCommand(translateCmd)) {
    llvm::errs() << "Error translating to LLVM IR\n";
    return failure();
  }

  // Step 3: Generate linker script
  SmallString<128> ldScriptPath(tmpDirName);
  sys::path::append(ldScriptPath, deviceName.str() + "_core_" +
                                      std::to_string(core.col) + "_" +
                                      std::to_string(core.row) + ".ld.script");

  SmallVector<std::string, 10> ldgenStrs = {
      aieTranslatePath,
      withAddressesPath.str(),
      "--aie-generate-ldscript",
      "--aie-device-name=" + deviceName.str(),
      "--tilecol=" + std::to_string(core.col),
      "--tilerow=" + std::to_string(core.row),
      "-o",
      ldScriptPath.str().str()};

  SmallVector<StringRef, 10> ldgenCmd;
  for (const auto &str : ldgenStrs) {
    ldgenCmd.push_back(str);
  }

  if (!executeCommand(ldgenCmd)) {
    llvm::errs() << "Error generating linker script\n";
    return failure();
  }

  // Step 4: Compile LLVM IR to object file
  SmallString<128> objPath(tmpDirName);
  sys::path::append(objPath, deviceName.str() + "_core_" +
                                 std::to_string(core.col) + "_" +
                                 std::to_string(core.row) + ".o");

  if (xchesscc) {
    // xchesscc compilation not yet implemented - would need chess-llvm-link +
    // xchesscc_wrapper
    llvm::errs()
        << "Error: xchesscc compilation not yet implemented in C++ aiecc\n";
    llvm::errs() << "Please use --no-xchesscc flag to compile with Peano\n";
    return failure();
  } else {
    // Use Peano toolchain
    std::string peanoOpt = findPeanoTool("opt");
    std::string peanoLlc = findPeanoTool("llc");

    if (peanoOpt.empty() || peanoLlc.empty()) {
      llvm::errs() << "Error: Could not find Peano compiler tools (opt/llc)\n";
      llvm::errs() << "Set PEANO_INSTALL_DIR or use --peano option\n";
      return failure();
    }

    // Run opt
    SmallString<128> optPath(tmpDirName);
    sys::path::append(optPath, deviceName.str() + "_core_" +
                                   std::to_string(core.col) + "_" +
                                   std::to_string(core.row) + ".opt.ll");

    std::string optLevelStr = std::to_string(optLevel);
    SmallVector<std::string, 12> optStrs = {peanoOpt,
                                            "--passes=default<O" + optLevelStr +
                                                ">,strip",
                                            "-inline-threshold=10",
                                            "-S",
                                            llvmIRPath.str().str(),
                                            "-o",
                                            optPath.str().str()};

    if (optLevel >= 3) {
      optStrs.insert(optStrs.begin() + 1, "-disable-loop-idiom-memset");
    }

    SmallVector<StringRef, 12> optCmd;
    for (const auto &str : optStrs) {
      optCmd.push_back(str);
    }

    if (!executeCommand(optCmd)) {
      llvm::errs() << "Error running Peano opt\n";
      return failure();
    }

    // Run llc
    SmallVector<std::string, 10> llcStrs = {peanoLlc,
                                            optPath.str().str(),
                                            "-O" + optLevelStr,
                                            "--march=" + aieTarget.lower(),
                                            "--function-sections",
                                            "--filetype=obj",
                                            "-o",
                                            objPath.str().str()};

    SmallVector<StringRef, 10> llcCmd;
    for (const auto &str : llcStrs) {
      llcCmd.push_back(str);
    }

    if (!executeCommand(llcCmd)) {
      llvm::errs() << "Error running Peano llc\n";
      return failure();
    }
  }

  // Step 5: Link to ELF
  if (!link) {
    outElfPath = objPath.str().str();
    return success();
  }

  SmallString<128> elfPath(tmpDirName);
  sys::path::append(elfPath, deviceName.str() + "_core_" +
                                 std::to_string(core.col) + "_" +
                                 std::to_string(core.row) + ".elf");

  // Make the ELF path absolute so CDO generation can find it
  SmallString<256> absElfPath;
  std::error_code ec = sys::fs::real_path(elfPath, absElfPath);
  if (ec) {
    // If real_path fails (file doesn't exist yet), make it absolute manually
    if (sys::path::is_absolute(elfPath)) {
      absElfPath = elfPath;
    } else {
      sys::fs::current_path(absElfPath);
      sys::path::append(absElfPath, elfPath);
      sys::path::remove_dots(absElfPath, /*remove_dot_dot=*/true);
    }
  }
  elfPath = absElfPath;

  if (xchesscc && xbridge) {
    // xchesscc + xbridge linking not yet implemented
    llvm::errs() << "Error: xchesscc linking not yet implemented\n";
    return failure();
  } else {
    // Link with Peano clang
    std::string peanoClang = findPeanoTool("clang");
    if (peanoClang.empty()) {
      llvm::errs() << "Error: Could not find Peano clang\n";
      return failure();
    }

    std::string peanoTarget = getPeanoTarget(aieTarget);
    std::string optLevelStr = std::to_string(optLevel);

    // Get Peano bin directory for the linker
    StringRef peanoDir = getPeanoInstallDir();
    SmallString<128> peanoBinDir;
    if (!peanoDir.empty()) {
      peanoBinDir = peanoDir;
      sys::path::append(peanoBinDir, "bin");
    } else {
      // Infer from clang path
      peanoBinDir = sys::path::parent_path(peanoClang);
    }

    // Find Peano's lld linker explicitly
    SmallString<256> peanoLld(peanoBinDir);
    sys::path::append(peanoLld, "ld.lld");

    SmallVector<std::string, 20> linkStrs = {peanoClang, "-O" + optLevelStr,
                                             "--target=" + peanoTarget};

    // Explicitly specify Peano's lld linker to avoid using system ld
    if (sys::fs::can_execute(peanoLld)) {
      linkStrs.push_back("-fuse-ld=" + peanoLld.str().str());
    } else {
      // Fallback: try to use lld from Peano bin via -B
      linkStrs.push_back("-B" + peanoBinDir.str().str());
      linkStrs.push_back("-fuse-ld=lld");
    }

    linkStrs.push_back(objPath.str().str());

    // Handle external object file if link_with attribute is specified
    // The linker script generated by aie-translate will include an INPUT()
    // directive for the link_with file, but it uses a relative path.
    // We need to copy the file to the .prj directory so the linker can find it.
    if (!core.linkWith.empty()) {
      // Resolve the link_with path relative to the input file
      SmallString<256> srcLinkWith;
      if (sys::path::is_absolute(core.linkWith)) {
        srcLinkWith = core.linkWith;
      } else {
        SmallString<256> inputDir = sys::path::parent_path(inputFilename);
        if (inputDir.empty()) {
          sys::fs::current_path(inputDir);
        }
        srcLinkWith = inputDir;
        sys::path::append(srcLinkWith, core.linkWith);
        sys::path::remove_dots(srcLinkWith, /*remove_dot_dot=*/true);
      }

      // Copy the object file to the .prj directory so the linker script's
      // INPUT() directive can find it
      SmallString<256> destLinkWith(tmpDirName);
      sys::path::append(destLinkWith, sys::path::filename(core.linkWith));

      std::error_code ec = sys::fs::copy_file(srcLinkWith, destLinkWith);
      if (ec) {
        llvm::errs() << "Error: Could not copy link_with file: " << srcLinkWith
                     << " to " << destLinkWith << "\n";
        llvm::errs() << "Error: " << ec.message() << "\n";
        return failure();
      }

      if (verbose) {
        llvm::outs() << "Copied link_with object: " << srcLinkWith << " -> "
                     << destLinkWith << "\n";
      }

      // Note: We don't add the object file to linkStrs because the linker
      // script already has an INPUT() directive for it
    }

    // Make linker script path absolute
    SmallString<128> absLdScriptPath;
    if (sys::path::is_absolute(ldScriptPath)) {
      absLdScriptPath = ldScriptPath;
    } else {
      std::error_code ec = sys::fs::real_path(ldScriptPath, absLdScriptPath);
      if (ec) {
        sys::fs::current_path(absLdScriptPath);
        sys::path::append(absLdScriptPath, ldScriptPath);
      }
    }

    linkStrs.push_back("-Wl,--gc-sections");
    linkStrs.push_back("-Wl,--orphan-handling=error");
    linkStrs.push_back("-Wl,-T," + absLdScriptPath.str().str());
    linkStrs.push_back("-o");
    linkStrs.push_back(elfPath.str().str());

    SmallVector<StringRef, 20> linkCmd;
    for (const auto &str : linkStrs) {
      linkCmd.push_back(str);
    }

    if (!executeCommand(linkCmd)) {
      llvm::errs() << "Error linking ELF file\n";
      return failure();
    }
  }

  outElfPath = elfPath.str().str();
  if (verbose) {
    llvm::outs() << "Generated ELF: " << outElfPath << "\n";
  }

  return success();
}

static LogicalResult
compileCores(MLIRContext &context, Operation *deviceOp, StringRef deviceName,
             StringRef withAddressesPath, StringRef tmpDirName,
             StringRef aieTarget,
             std::map<std::pair<int, int>, std::string> &elfPaths) {

  SmallVector<CoreInfo> cores;
  deviceOp->walk([&](xilinx::AIE::CoreOp coreOp) {
    cores.push_back(getCoreInfo(coreOp));
  });

  if (cores.empty()) {
    if (verbose) {
      llvm::outs() << "No cores to compile in device " << deviceName << "\n";
    }
    return success();
  }

  if (verbose) {
    llvm::outs() << "Compiling " << cores.size() << " core(s)\n";
  }

  for (const auto &core : cores) {
    std::string elfPath;
    if (failed(compileCore(context, deviceName, core, withAddressesPath,
                           tmpDirName, aieTarget, elfPath))) {
      return failure();
    }

    if (!elfPath.empty()) {
      elfPaths[{core.col, core.row}] = elfPath;
    }
  }

  return success();
}

// Update MLIR module with ELF file paths
static LogicalResult
updateModuleWithElfs(MLIRContext &context, StringRef physicalPath,
                     StringRef tmpDirName, StringRef deviceName,
                     const std::map<std::pair<int, int>, std::string> &elfPaths,
                     SmallString<128> &outPath) {

  if (elfPaths.empty()) {
    outPath = physicalPath;
    return success();
  }

  if (verbose) {
    llvm::outs() << "Updating MLIR with ELF paths\n";
  }

  // Parse the physical MLIR
  ParserConfig parseConfig(&context);
  SourceMgr sourceMgr;
  auto module = parseSourceFile<ModuleOp>(physicalPath, sourceMgr, parseConfig);

  if (!module) {
    llvm::errs() << "Error parsing physical MLIR file\n";
    return failure();
  }

  // Update cores with ELF paths
  module->walk([&](xilinx::AIE::DeviceOp devOp) {
    if (devOp.getSymName() != deviceName) {
      return;
    }

    devOp.walk([&](xilinx::AIE::CoreOp coreOp) {
      auto tileOp =
          dyn_cast<xilinx::AIE::TileOp>(coreOp.getTile().getDefiningOp());
      if (!tileOp)
        return;

      int col = tileOp.getCol();
      int row = tileOp.getRow();

      auto it = elfPaths.find({col, row});
      if (it != elfPaths.end()) {
        // When elf_file is specified, create a new core with empty body
        // (matching Python's set_elf_file_for_core behavior)
        OpBuilder builder(coreOp->getContext());
        builder.setInsertionPoint(coreOp);

        auto newCore = xilinx::AIE::CoreOp::create(
            builder, coreOp.getLoc(), builder.getIndexType(), coreOp.getTile());

        // Copy all attributes from the old core
        for (auto attr : coreOp->getAttrs()) {
          newCore->setAttr(attr.getName(), attr.getValue());
        }
        // Set the elf_file attribute
        newCore.setElfFileAttr(builder.getStringAttr(it->second));

        // Create empty body with just aie.end
        Block *newBlock = builder.createBlock(&newCore.getBody());
        builder.setInsertionPointToEnd(newBlock);
        xilinx::AIE::EndOp::create(builder, coreOp.getLoc());

        // Erase the old core
        coreOp.erase();
      }
    });
  });

  // Write updated MLIR
  outPath = tmpDirName;
  sys::path::append(outPath, deviceName.str() + "_physical_with_elfs.mlir");

  std::error_code ec;
  raw_fd_ostream outFile(outPath, ec);
  if (ec) {
    llvm::errs() << "Error writing MLIR with ELFs: " << ec.message() << "\n";
    return failure();
  }
  module->print(outFile);
  outFile.close();

  return success();
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
// NPU Instruction Generation
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

  // In dry-run mode, just show what would be done and return
  if (dryRun) {
    if (verbose) {
      llvm::outs() << "Would generate NPU instructions for device: " << devName
                   << "\n";
    }
    return success();
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
  // Parse the lowered module to find sequences
  ParserConfig parseConfig(&context);
  SourceMgr sourceMgr;
  auto module =
      parseSourceFile<ModuleOp>(npuLoweredPath, sourceMgr, parseConfig);

  if (!module) {
    llvm::errs() << "Error parsing lowered MLIR file\n";
    return failure();
  }

  // Find device and generate instructions for each runtime sequence
  LogicalResult result = success();
  for (auto devOp : module->getOps<xilinx::AIE::DeviceOp>()) {
    if (!deviceName.empty() && devOp.getSymName() != devName) {
      continue;
    }

    devOp.walk([&](xilinx::AIE::RuntimeSequenceOp seq) {
      if (failed(result)) {
        return; // Skip if already failed
      }

      if (!sequenceName.empty() && seq.getSymName() != sequenceName) {
        return;
      }

      StringRef seqName = seq.getSymName();
      std::string outputFileName =
          formatString(instsName, devName.str(), seqName);

      // Output to current directory (matches Python aiecc.py)
      SmallString<128> outputPath;
      if (sys::path::is_absolute(outputFileName)) {
        outputPath = outputFileName;
      } else {
        outputPath = outputFileName;
      }

      if (verbose) {
        llvm::outs() << "Generating NPU instructions for sequence: " << seqName
                     << " -> " << outputPath << "\n";
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
        result = failure();
      }
    });
  }

  if (failed(result)) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CDO/PDI/xclbin Generation
//===----------------------------------------------------------------------===//

static LogicalResult generateCdoArtifacts(StringRef mlirFilePath,
                                          StringRef tmpDirName,
                                          StringRef devName) {
  if (!generateCdo && !generatePdi && !generateXclbin) {
    return success();
  }

  if (verbose) {
    llvm::outs() << "Generating CDO artifacts for device: " << devName << "\n";
  }

  // In dry-run mode, just show what would be done and return
  if (dryRun) {
    if (verbose) {
      llvm::outs() << "Would generate CDO artifacts for device: " << devName
                   << "\n";
    }
    return success();
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
      llvm::errs()
          << "Error: bootgen not found, cannot generate requested PDI/xclbin\n";
      return failure();
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
      if (sys::path::is_absolute(xclbinFileName)) {
        xclbinPath = xclbinFileName;
      } else {
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

  // Count devices and cores for verbose output
  unsigned deviceCount = 0;
  if (verbose) {
    for (auto deviceOp : module.getOps<xilinx::AIE::DeviceOp>()) {
      if (!deviceName.empty() && deviceOp.getSymName() != deviceName) {
        continue;
      }
      deviceCount++;
      unsigned coreCount = 0;
      deviceOp.walk([&](xilinx::AIE::CoreOp) { coreCount++; });
      llvm::outs() << "  Device";
      if (deviceOp.getSymNameAttr()) {
        llvm::outs() << " '" << deviceOp.getSymName() << "'";
      }
      llvm::outs() << " with " << coreCount << " core(s)\n";
    }

    if (deviceCount == 0) {
      llvm::errs() << "Error: No AIE devices found in module\n";
      return failure();
    }
    llvm::outs() << "Found " << deviceCount << " AIE device(s)\n";
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

  // Step 3: Compile cores and generate artifacts for each device
  for (auto deviceOp : module.getOps<xilinx::AIE::DeviceOp>()) {
    // Filter by device name if specified
    if (!deviceName.empty() && deviceOp.getSymName() != deviceName) {
      continue;
    }

    StringRef devName = deviceOp.getSymName();

    if (verbose) {
      llvm::outs() << "\nProcessing device: " << devName << "\n";
    }

    // Get AIE target for this device using aie-translate
    std::string aieTarget;
    if (!dryRun) {
      aieTarget = getAIETargetForDevice(physicalPath.str(), devName);
    } else {
      aieTarget = "aie2"; // Default for dry-run
    }

    if (verbose) {
      llvm::outs() << "Using AIE target: " << aieTarget << "\n";
    }

    // Compile cores
    std::map<std::pair<int, int>, std::string> elfPaths;
    if (failed(compileCores(context, deviceOp, devName, withAddressesPath,
                            tmpDirName, aieTarget, elfPaths))) {
      return failure();
    }

    // Update MLIR with ELF paths
    SmallString<128> physicalWithElfs;
    if (failed(updateModuleWithElfs(context, physicalPath, tmpDirName, devName,
                                    elfPaths, physicalWithElfs))) {
      return failure();
    }

    // Use the updated MLIR for artifact generation
    StringRef mlirForArtifacts;
    if (elfPaths.empty()) {
      mlirForArtifacts = physicalPath;
    } else {
      mlirForArtifacts = physicalWithElfs.str();
    }

    // Generate NPU instructions
    if (failed(generateNpuInstructions(context, mlirForArtifacts, tmpDirName,
                                       devName))) {
      return failure();
    }

    // Generate CDO/PDI/xclbin
    if (failed(generateCdoArtifacts(mlirForArtifacts, tmpDirName, devName))) {
      return failure();
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
