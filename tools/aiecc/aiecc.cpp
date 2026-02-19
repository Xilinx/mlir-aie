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
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "aie/Conversion/Passes.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEVec/Analysis/Passes.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/TransformOps/DialectExtension.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/InitialAllDialect.h"
#include "aie/Targets/AIETargets.h"
#include "aie/version.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
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

static cl::opt<std::string>
    aietoolsDir("aietools", cl::desc("Path to aietools installation directory"),
                cl::init(""), cl::cat(aieCompilerOptions));

static cl::opt<bool> dumpIntermediates(
    "dump-intermediates",
    cl::desc("Dump intermediate MLIR files for debugging (default: off)"),
    cl::init(false), cl::cat(aieCompilerOptions));

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static void printVersion(raw_ostream &os) {
  os << "aiecc (C++ version) " << AIE_GIT_COMMIT << "\n";
}

/// Dump an MLIR module to a file if --dump-intermediates is enabled.
/// Returns true on success, false on failure.
static bool dumpModuleToFile(ModuleOp moduleOp, StringRef filePath,
                             StringRef description) {
  if (!dumpIntermediates)
    return true;

  std::error_code ec;
  raw_fd_ostream file(filePath, ec);
  if (ec) {
    llvm::errs() << "Warning: Could not dump " << description << " to "
                 << filePath << ": " << ec.message() << "\n";
    return false;
  }
  moduleOp->print(file);
  file.close();

  if (verbose) {
    llvm::outs() << "Dumped " << description << " to: " << filePath << "\n";
  }
  return true;
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

// Overload to avoid the pattern: vector<string> -> vector<StringRef> -> execute
static bool executeCommand(ArrayRef<std::string> command,
                           bool verboseOutput = true) {
  SmallVector<StringRef, 16> cmdRefs;
  cmdRefs.reserve(command.size());
  for (const auto &arg : command) {
    cmdRefs.push_back(arg);
  }
  return executeCommand(ArrayRef<StringRef>(cmdRefs), verboseOutput);
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
  // Use LLVM's ExecuteAndWait with output redirection to avoid shell injection
  {
    auto pythonPath = sys::findProgramByName("python3");
    if (!pythonPath) {
      pythonPath = sys::findProgramByName("python");
    }

    if (pythonPath) {
      // Create a temporary file for output
      SmallString<128> tempFile;
      std::error_code ec =
          sys::fs::createTemporaryFile("peano-discover", "txt", tempFile);
      if (!ec) {
        StringRef nullFile;
#ifdef _WIN32
        nullFile = "NUL";
#else
        nullFile = "/dev/null";
#endif
        // Redirect stdout to temp file, stderr to /dev/null
        std::optional<StringRef> redirects[] = {/*stdin=*/nullFile,
                                                /*stdout=*/StringRef(tempFile),
                                                /*stderr=*/nullFile};

        SmallVector<StringRef, 4> args = {
            *pythonPath, "-c",
            "import sysconfig; print(sysconfig.get_path('platlib'))"};

        int result =
            sys::ExecuteAndWait(*pythonPath, args, /*Env=*/std::nullopt,
                                redirects, /*secondsToWait=*/5);

        if (result == 0) {
          // Read the output from the temp file
          auto bufOrErr = MemoryBuffer::getFile(tempFile);
          if (bufOrErr) {
            std::string output = (*bufOrErr)->getBuffer().str();
            // Remove trailing newline
            while (!output.empty() &&
                   (output.back() == '\n' || output.back() == '\r')) {
              output.pop_back();
            }
            SmallString<256> llvmAiePath(output);
            sys::path::append(llvmAiePath, "llvm-aie");
            sys::fs::remove(tempFile);
            if (sys::fs::is_directory(llvmAiePath)) {
              return std::string(llvmAiePath);
            }
          }
        }
        sys::fs::remove(tempFile);
      }
    }
  }

  return "";
}

// Cached Peano install directory
static std::optional<std::string> cachedPeanoDir;

// Discover aietools installation directory by finding xchesscc in PATH
static std::string discoverAietoolsDir() {
  // 1. Check if --aietools was specified
  if (!aietoolsDir.empty()) {
    if (sys::fs::is_directory(aietoolsDir)) {
      return aietoolsDir;
    }
  }

  // 2. Find xchesscc in PATH and derive aietools from it
  auto xchessccPath = sys::findProgramByName("xchesscc");
  if (xchessccPath) {
    // xchesscc is typically at <aietools>/bin/xchesscc or
    // <aietools>/bin/unwrapped/lnx64.o/xchesscc
    // Use std::string to avoid self-referencing issues with parent_path
    std::string binDir = std::string(sys::path::parent_path(*xchessccPath));

    // Handle unwrapped paths
    // (e.g., aietools/bin/unwrapped/lnx64.o/xchesscc)
    StringRef parentName = sys::path::filename(binDir);
    if (parentName == "lnx64.o") {
      binDir = std::string(sys::path::parent_path(binDir)); // up from lnx64.o
      binDir = std::string(sys::path::parent_path(binDir)); // up from unwrapped
      binDir = std::string(sys::path::parent_path(binDir)); // up from bin
    } else if (parentName == "bin") {
      binDir = std::string(sys::path::parent_path(binDir)); // up from bin
    } else {
      // Assume it's in bin/ directory
      binDir = std::string(sys::path::parent_path(binDir));
    }

    if (sys::fs::is_directory(binDir)) {
      return binDir;
    }
  }

  return "";
}

// Cached aietools install directory
static std::optional<std::string> cachedAietoolsDir;

static StringRef getAietoolsDir() {
  if (!cachedAietoolsDir.has_value()) {
    cachedAietoolsDir = discoverAietoolsDir();
    if (verbose && !cachedAietoolsDir->empty()) {
      llvm::outs() << "Discovered aietools installation: " << *cachedAietoolsDir
                   << "\n";
    }
  }
  return *cachedAietoolsDir;
}

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

// Downgrade LLVM IR for compatibility with Chess toolchain's older LLVM.
// The Chess LLVM is based on an older version that doesn't support modern
// memory/capture attributes. This function performs string replacements
// matching the Python downgrade_ir_for_chess() function.
static std::string downgradeIRForChess(StringRef llvmIR) {
  std::string result = llvmIR.str();

  // Replace memory attributes
  auto replace = [&result](const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
      result.replace(pos, from.length(), to);
      pos += to.length();
    }
  };

  replace("memory(none)", "readnone");
  replace("memory(read)", "readonly");
  replace("memory(write)", "writeonly");
  replace("memory(argmem: readwrite)", "argmemonly");
  replace("memory(argmem: read)", "argmemonly readonly");
  replace("memory(argmem: write)", "argmemonly writeonly");
  replace("memory(inaccessiblemem: readwrite)", "inaccessiblememonly");
  replace("memory(inaccessiblemem: read)", "inaccessiblememonly readonly");
  replace("memory(inaccessiblemem: write)", "inaccessiblememonly writeonly");
  replace("memory(argmem: readwrite, inaccessiblemem: readwrite)",
          "inaccessiblemem_or_argmemonly");
  replace("memory(argmem: read, inaccessiblemem: read)",
          "inaccessiblemem_or_argmemonly readonly");
  replace("memory(argmem: write, inaccessiblemem: write)",
          "inaccessiblemem_or_argmemonly writeonly");
  replace("captures(none)", "nocapture");
  replace("getelementptr inbounds nuw", "getelementptr inbounds");

  // Remove nocreateundeforpoison attribute using regex-like logic
  // Pattern: \bnocreateundeforpoison\s+
  // We'll do a simple find-and-replace for this attribute
  size_t pos = 0;
  while ((pos = result.find("nocreateundeforpoison", pos)) !=
         std::string::npos) {
    // Find the end of the attribute (skip trailing whitespace)
    size_t end = pos + strlen("nocreateundeforpoison");
    while (end < result.size() && (result[end] == ' ' || result[end] == '\t')) {
      end++;
    }
    result.erase(pos, end - pos);
  }

  return result;
}

// Get the chess-llvm-link target directory name for a given AIE target
static std::string getChessTarget(StringRef aieTarget) {
  std::string target = aieTarget.lower();
  if (target == "aie2") {
    return "target_aie_ml";
  } else if (target == "aie2p") {
    return "target_aie2p";
  }
  return "target"; // AIE1
}

// Get the install path for mlir-aie (for finding runtime libraries)
// Checks multiple possible locations to support both build and install areas
static std::string getInstallPath() {
  auto mainExe = sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&getInstallPath));
  SmallString<256> binDir(sys::path::parent_path(mainExe));
  // Go up from bin/ to install prefix
  SmallString<256> installPrefix(sys::path::parent_path(binDir));

  // Check if aie_runtime_lib exists at this location
  SmallString<256> runtimeLibPath(installPrefix);
  sys::path::append(runtimeLibPath, "aie_runtime_lib");
  if (sys::fs::is_directory(runtimeLibPath)) {
    return std::string(installPrefix);
  }

  // Try sibling "install" directory (for when running from build/bin)
  SmallString<256> siblingInstall(sys::path::parent_path(installPrefix));
  sys::path::append(siblingInstall, "install");
  SmallString<256> siblingRuntimeLib(siblingInstall);
  sys::path::append(siblingRuntimeLib, "aie_runtime_lib");
  if (sys::fs::is_directory(siblingRuntimeLib)) {
    return std::string(siblingInstall);
  }

  // Check MLIR_AIE_INSTALL_DIR environment variable
  if (const char *installEnv = std::getenv("MLIR_AIE_INSTALL_DIR")) {
    if (sys::fs::is_directory(installEnv)) {
      return installEnv;
    }
  }

  // Fallback: return the original calculated path
  return std::string(installPrefix);
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

// Run chess-llvm-link to link LLVM IR with chess_intrinsic_wrapper.ll
// Returns the path to the linked .ll file, or empty string on failure.
static std::string runChessLlvmLink(StringRef inputLLPath, StringRef outputPath,
                                    StringRef aieTarget, StringRef tmpDirName) {
  StringRef aietoolsPath = getAietoolsDir();
  if (aietoolsPath.empty()) {
    llvm::errs() << "Error: Could not find aietools (xchesscc not in PATH)\n";
    return "";
  }

  // Build chess-llvm-link path
  // Path: <aietools>/tps/lnx64/<target>/bin/LNa64bin/chess-llvm-link
  std::string chessTarget = getChessTarget(aieTarget);
  SmallString<256> chessLlvmLinkPath(aietoolsPath);
  sys::path::append(chessLlvmLinkPath, "tps", "lnx64", chessTarget, "bin");
  sys::path::append(chessLlvmLinkPath, "LNa64bin", "chess-llvm-link");

  if (!sys::fs::can_execute(chessLlvmLinkPath)) {
    llvm::errs() << "Error: Could not find chess-llvm-link at: "
                 << chessLlvmLinkPath << "\n";
    return "";
  }

  // Build chess_intrinsic_wrapper.ll path
  std::string installPath = getInstallPath();
  SmallString<256> wrapperPath(installPath);
  std::string aieTargetUpper = aieTarget.upper();
  sys::path::append(wrapperPath, "aie_runtime_lib", aieTargetUpper,
                    "chess_intrinsic_wrapper.ll");

  if (!sys::fs::exists(wrapperPath)) {
    llvm::errs() << "Error: Could not find chess_intrinsic_wrapper.ll at: "
                 << wrapperPath << "\n";
    return "";
  }

  // Run chess-llvm-link
  SmallVector<std::string, 8> cmd = {chessLlvmLinkPath.str().str(),
                                     inputLLPath.str(),
                                     wrapperPath.str().str(),
                                     "-S",
                                     "-o",
                                     outputPath.str()};

  if (!executeCommand(cmd)) {
    llvm::errs() << "Error running chess-llvm-link\n";
    return "";
  }

  return outputPath.str();
}

// Extract _include _file entries from a BCF file
// BCF files contain lines like: _include _file path/to/file.o
// These specify object files that need to be linked.
static std::vector<std::string> extractInputFilesFromBCF(StringRef bcfPath) {
  std::vector<std::string> files;

  auto bufOrErr = MemoryBuffer::getFile(bcfPath);
  if (!bufOrErr) {
    return files;
  }

  StringRef content = (*bufOrErr)->getBuffer();
  SmallVector<StringRef, 32> lines;
  content.split(lines, '\n');

  for (StringRef line : lines) {
    // Match: _include _file <path>
    line = line.trim();
    if (line.starts_with("_include _file ")) {
      StringRef filePath = line.drop_front(strlen("_include _file ")).trim();
      if (!filePath.empty()) {
        files.push_back(filePath.str());
      }
    }
  }

  return files;
}

// Get the AIE target architecture for a device using direct C++ API call.
// This replaces the subprocess call to aie-translate
// --aie-generate-target-arch.
static std::string getAIETargetForDevice(ModuleOp moduleOp,
                                         StringRef deviceName) {
  std::string targetArch;
  llvm::raw_string_ostream os(targetArch);

  if (failed(xilinx::AIE::AIETranslateToTargetArch(moduleOp, os, deviceName))) {
    if (verbose) {
      llvm::outs() << "Warning: Could not determine target for device "
                   << deviceName << ", defaulting to aie2\n";
    }
    return "aie2";
  }

  // Trim trailing newline/whitespace from the output
  while (!targetArch.empty() &&
         (targetArch.back() == '\n' || targetArch.back() == '\r' ||
          targetArch.back() == ' ' || targetArch.back() == '\t')) {
    targetArch.pop_back();
  }

  if (targetArch.empty()) {
    if (verbose) {
      llvm::outs() << "Warning: Empty target for device " << deviceName
                   << ", defaulting to aie2\n";
    }
    return "aie2";
  }

  if (verbose) {
    llvm::outs() << "Detected target architecture for device " << deviceName
                 << ": " << targetArch << "\n";
  }

  return targetArch;
}

//===----------------------------------------------------------------------===//
// AIE Device and Core Discovery
//===----------------------------------------------------------------------===//

struct CoreInfo {
  std::int32_t col;
  std::int32_t row;
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
// Pass Pipeline Construction (for subprocess calls)
//===----------------------------------------------------------------------===//

// Pipeline strings kept for reference - subprocess calls have been replaced
// with in-memory PassManager execution in runResourceAllocationPipeline()
[[maybe_unused]] static std::string
buildInputWithAddressesPipeline(StringRef aieTarget = "aie2") {
  std::ostringstream oss;
  oss << "builtin.module("
      << "convert-vector-to-aievec{aie-target=" << aieTarget.lower()
      << " target-backend=llvmir}," << "lower-affine,"
      << "aie-canonicalize-device," << "aie.device(" << "aie-assign-lock-ids,"
      << "aie-register-objectFifos," << "aie-objectFifo-stateful-transform{"
      << "dynamic-objFifos=" << (dynamicObjFifos ? "true" : "false")
      << " packet-sw-objFifos=" << (packetSwObjFifos ? "true" : "false") << "},"
      << "aie-assign-bd-ids," << "aie-lower-cascade-flows,"
      << "aie-lower-broadcast-packet," << "aie-lower-multicast,"
      << "aie-assign-tile-controller-ids,"
      << "aie-generate-column-control-overlay{route-shim-to-tile-ctrl="
      << (ctrlPktOverlay ? "true" : "false") << "},"
      << "aie-assign-buffer-addresses{alloc-scheme=" << allocScheme.getValue()
      << "}," << "aie-vector-transfer-lowering{max-transfer-rank=1}" << "),"
      << "convert-scf-to-cf" << ")";
  return oss.str();
}

// Pipeline string kept for reference - replaced by runLLVMLoweringPipeline()
[[maybe_unused]] static std::string
buildLLVMLoweringPipeline(StringRef deviceName, StringRef aieTarget = "aie2") {
  std::ostringstream oss;
  oss << "builtin.module("
      << "aie.device(aie-localize-locks,aie-normalize-address-spaces,"
      << "aie-transform-bfp-types),"
      << "aie-standard-lowering{device=" << deviceName.str() << "},"
      << "aiex-standard-lowering,"
      << "convert-aievec-to-llvm{aie-target=" << aieTarget.lower() << "},"
      << "canonicalize," << "cse," << "expand-strided-metadata,"
      << "lower-affine," << "arith-expand," << "finalize-memref-to-llvm,"
      << "convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},"
      << "convert-to-llvm{dynamic=true}," << "canonicalize," << "cse" << ")";
  return oss.str();
}

// Pipeline string kept for reference - replaced by runNpuLoweringPipeline()
[[maybe_unused]] static std::string buildNpuLoweringPipeline() {
  return R"(builtin.module(aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu,aie-lower-set-lock)))";
}

//===----------------------------------------------------------------------===//
// In-Memory Pass Execution
//===----------------------------------------------------------------------===//

/// Run the resource allocation pipeline in-memory using PassManager.
/// This replaces the subprocess call to aie-opt with direct API calls.
static LogicalResult runResourceAllocationPipeline(ModuleOp moduleOp,
                                                   StringRef aieTarget) {
  MLIRContext *ctx = moduleOp.getContext();
  PassManager pm(ctx);

  // Enable verification between passes if verbose
  if (verbose) {
    pm.enableVerifier(true);
    // Enable crash reproducer to help debug CI failures
    pm.enableCrashReproducerGeneration("resource_alloc_crash.mlir");
  }

  // Step 1: Convert vector to aievec (this is a pipeline, not a single pass)
  // Only add for AIE2 and later targets - AIE1 doesn't support
  // target_backend=llvmir
  std::string lowerTarget = aieTarget.lower();
  if (lowerTarget == "aie2" || lowerTarget == "aieml" ||
      lowerTarget == "aie2p") {
    xilinx::aievec::ConvertVectorToAIEVecOptions vecOptions;
    vecOptions.aieTarget = lowerTarget;
    vecOptions.targetBackend = "llvmir";
    xilinx::aievec::buildConvertVectorToAIEVec(pm, vecOptions);
  }

  // Step 2: Lower affine
  pm.addPass(createLowerAffinePass());

  // Step 3: Canonicalize device (module-level pass)
  pm.addPass(xilinx::AIE::createAIECanonicalizeDevicePass());

  // Step 4: Device-level passes - use nest<DeviceOp>()
  OpPassManager &devicePm = pm.nest<xilinx::AIE::DeviceOp>();
  devicePm.addPass(xilinx::AIE::createAIEAssignLockIDsPass());
  devicePm.addPass(xilinx::AIE::createAIEObjectFifoRegisterProcessPass());
  devicePm.addPass(xilinx::AIE::createAIEObjectFifoStatefulTransformPass());
  devicePm.addPass(xilinx::AIE::createAIEAssignBufferDescriptorIDsPass());
  devicePm.addPass(xilinx::AIE::createAIELowerCascadeFlowsPass());
  devicePm.addPass(xilinx::AIEX::createAIEBroadcastPacketPass());
  devicePm.addPass(xilinx::AIEX::createAIELowerMulticastPass());
  devicePm.addPass(xilinx::AIE::createAIEAssignTileCtrlIDsPass());
  devicePm.addPass(xilinx::AIE::createAIEGenerateColumnControlOverlayPass());

  // Create buffer address assignment pass with alloc-scheme option
  xilinx::AIE::AIEAssignBufferAddressesOptions bufferOpts;
  bufferOpts.clAllocScheme = allocScheme.getValue();
  devicePm.addPass(xilinx::AIE::createAIEAssignBufferAddressesPass(bufferOpts));

  devicePm.addPass(xilinx::AIE::createAIEVectorTransferLoweringPass());

  // Step 5: Convert SCF to CF (module-level pass)
  pm.addPass(createSCFToControlFlowPass());

  if (verbose) {
    llvm::outs() << "Running resource allocation pipeline in-memory "
                 << "(alloc-scheme=" << allocScheme.getValue() << ")\n";
    llvm::outs().flush();
  }

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Error: Resource allocation pipeline failed\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Resource allocation pipeline completed successfully\n";
  }

  return success();
}

/// Run the routing pipeline in-memory using PassManager.
/// This replaces the subprocess call to aie-opt with direct API calls.
static LogicalResult runRoutingPipeline(ModuleOp moduleOp) {
  MLIRContext *ctx = moduleOp.getContext();
  PassManager pm(ctx);

  if (verbose) {
    pm.enableVerifier(true);
  }

  // Routing is a device-level pass
  OpPassManager &devicePm = pm.nest<xilinx::AIE::DeviceOp>();
  devicePm.addPass(xilinx::AIE::createAIEPathfinderPass());

  if (verbose) {
    llvm::outs() << "Running routing pipeline in-memory\n";
  }

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Error: Routing pipeline failed\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Routing pipeline completed successfully\n";
  }

  return success();
}

/// Run the NPU lowering pipeline in-memory using PassManager.
/// This replaces the subprocess call to aie-opt with direct API calls.
static LogicalResult runNpuLoweringPipeline(ModuleOp moduleOp) {
  MLIRContext *ctx = moduleOp.getContext();
  PassManager pm(ctx);

  if (verbose) {
    pm.enableVerifier(true);
  }

  // All NPU lowering passes are device-level
  OpPassManager &devicePm = pm.nest<xilinx::AIE::DeviceOp>();
  devicePm.addPass(xilinx::AIEX::createAIEMaterializeBDChainsPass());
  devicePm.addPass(xilinx::AIEX::createAIESubstituteShimDMAAllocationsPass());
  devicePm.addPass(xilinx::AIEX::createAIEAssignRuntimeSequenceBDIDsPass());
  devicePm.addPass(xilinx::AIEX::createAIEDMATasksToNPUPass());
  devicePm.addPass(xilinx::AIEX::createAIEDmaToNpuPass());
  devicePm.addPass(xilinx::AIEX::createAIELowerSetLockPass());

  if (verbose) {
    llvm::outs() << "Running NPU lowering pipeline in-memory\n";
  }

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Error: NPU lowering pipeline failed\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "NPU lowering pipeline completed successfully\n";
  }

  return success();
}

/// Run the LLVM lowering pipeline in-memory using PassManager.
/// This replaces the subprocess calls to aie-opt and aie-translate for
/// core compilation. The pipeline:
/// 1. Runs nested passes inside aie.device (localize-locks,
/// normalize-address-spaces)
/// 2. Runs aie-standard-lowering with specific core coordinates to extract core
/// 3. Runs standard LLVM lowering passes
/// 4. Results in a pure LLVM dialect module ready for translation to LLVM IR
///
/// NOTE: This function modifies moduleOp destructively - it removes all cores
/// except the specified one. Caller should pass a cloned module.
static LogicalResult runLLVMLoweringPipeline(ModuleOp moduleOp,
                                             StringRef deviceName, int col,
                                             int row,
                                             StringRef aieTarget = "aie2") {
  MLIRContext *ctx = moduleOp.getContext();
  PassManager pm(ctx);

  if (verbose) {
    pm.enableVerifier(true);
  }

  // Step 1: Nested passes inside aie.device
  OpPassManager &devicePm = pm.nest<xilinx::AIE::DeviceOp>();
  devicePm.addPass(xilinx::AIE::createAIELocalizeLocksPass());
  devicePm.addPass(xilinx::AIE::createAIENormalizeAddressSpacesPass());
  // TODO: Add aie-transform-bfp-types if needed

  // Step 2: aie-standard-lowering with specific core coordinates
  // This extracts the specified core and removes the aie.device wrapper
  xilinx::AIE::AIECoreToStandardOptions coreOpts;
  coreOpts.deviceName = deviceName.str();
  coreOpts.tileCol = col;
  coreOpts.tileRow = row;
  pm.addPass(xilinx::AIE::createAIECoreToStandardPass(coreOpts));

  // Step 3: AIEX to standard lowering
  pm.addPass(xilinx::AIEX::createAIEXToStandardPass());

  // Step 4: AIEVec to LLVM conversion
  xilinx::ConvertAIEVecToLLVMOptions aievecOpts;
  aievecOpts.aieTarget = aieTarget.lower();
  pm.addPass(xilinx::aievec::createConvertAIEVecToLLVMPass(aievecOpts));

  // Step 5: Standard LLVM lowering passes
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(arith::createArithExpandOpsPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass(
      ConvertFuncToLLVMPassOptions{/*useBarePtrCallConv=*/true}));
  // convert-to-llvm - use the generic conversion pass
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  if (verbose) {
    llvm::outs() << "Running LLVM lowering pipeline in-memory for core (" << col
                 << ", " << row << ")\n";
  }

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Error: LLVM lowering pipeline failed for core (" << col
                 << ", " << row << ")\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "LLVM lowering pipeline completed successfully for core ("
                 << col << ", " << row << ")\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Core Compilation
//===----------------------------------------------------------------------===//

struct CoreCompilationResult {
  std::string elfPath;
  bool success;
};

static LogicalResult compileCore(MLIRContext &context, ModuleOp moduleOp,
                                 StringRef deviceName, const CoreInfo &core,
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

  // Step 1: Clone module and run LLVM lowering pipeline in-memory
  // The pipeline is destructive (removes other cores), so we clone first.
  OwningOpRef<ModuleOp> coreModule = moduleOp.clone();

  // Register LLVM IR translation dialects
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);

  if (failed(runLLVMLoweringPipeline(*coreModule, deviceName, core.col,
                                     core.row, aieTarget))) {
    llvm::errs() << "Error lowering core to LLVM\n";
    return failure();
  }

  // Step 2: Translate to LLVM IR in-memory using translateModuleToLLVMIR
  SmallString<128> llvmIRPath(tmpDirName);
  sys::path::append(llvmIRPath, deviceName.str() + "_core_" +
                                    std::to_string(core.col) + "_" +
                                    std::to_string(core.row) + ".ll");

  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(*coreModule, llvmCtx);
  if (!llvmModule) {
    llvm::errs() << "Error translating to LLVM IR for core (" << core.col
                 << ", " << core.row << ")\n";
    return failure();
  }

  // Write LLVM IR to file for Peano toolchain
  {
    std::error_code ec;
    raw_fd_ostream llvmIRFile(llvmIRPath, ec);
    if (ec) {
      llvm::errs() << "Error opening LLVM IR file: " << ec.message() << "\n";
      return failure();
    }
    llvmModule->print(llvmIRFile, nullptr);
    if (verbose) {
      llvm::outs() << "Generated LLVM IR: " << llvmIRPath << "\n";
    }
  }

  // Step 3: Generate linker script (for non-xbridge linking)
  // Note: BCF is generated later in the linking step when xbridge is enabled
  SmallString<128> ldScriptPath(tmpDirName);
  sys::path::append(ldScriptPath, deviceName.str() + "_core_" +
                                      std::to_string(core.col) + "_" +
                                      std::to_string(core.row) + ".ld.script");

  if (!xbridge) {
    // Generate linker script to file using the original (unmodified) module
    std::error_code ec;
    raw_fd_ostream ldScriptFile(ldScriptPath, ec);
    if (ec) {
      llvm::errs() << "Error opening linker script file: " << ec.message()
                   << "\n";
      return failure();
    }

    if (failed(xilinx::AIE::AIETranslateToLdScript(
            moduleOp, ldScriptFile, core.col, core.row, deviceName))) {
      llvm::errs() << "Error generating linker script\n";
      return failure();
    }

    if (verbose) {
      llvm::outs() << "Generated linker script: " << ldScriptPath << "\n";
    }
  }

  // Step 4: Compile LLVM IR to object file
  SmallString<128> objPath(tmpDirName);
  sys::path::append(objPath, deviceName.str() + "_core_" +
                                 std::to_string(core.col) + "_" +
                                 std::to_string(core.row) + ".o");

  if (xchesscc) {
    // xchesscc compilation: IR downgrade -> chess-llvm-link -> xchesscc_wrapper

    // Step 4a: Read LLVM IR and apply downgrade for Chess compatibility
    auto bufOrErr = MemoryBuffer::getFile(llvmIRPath);
    if (!bufOrErr) {
      llvm::errs() << "Error reading LLVM IR file: "
                   << bufOrErr.getError().message() << "\n";
      return failure();
    }
    std::string downgradedIR = downgradeIRForChess((*bufOrErr)->getBuffer());

    // Write downgraded IR to .chesshack.ll
    SmallString<128> chessHackPath(tmpDirName);
    sys::path::append(chessHackPath,
                      deviceName.str() + "_core_" + std::to_string(core.col) +
                          "_" + std::to_string(core.row) + ".chesshack.ll");
    {
      std::error_code ec;
      raw_fd_ostream chessHackFile(chessHackPath, ec);
      if (ec) {
        llvm::errs() << "Error writing chesshack file: " << ec.message()
                     << "\n";
        return failure();
      }
      chessHackFile << downgradedIR;
    }

    if (verbose) {
      llvm::outs() << "Applied IR downgrade for Chess: " << chessHackPath
                   << "\n";
    }

    // Step 4b: Run chess-llvm-link to link with intrinsic wrapper
    SmallString<128> chessLinkedPath(tmpDirName);
    sys::path::append(chessLinkedPath,
                      deviceName.str() + "_core_" + std::to_string(core.col) +
                          "_" + std::to_string(core.row) + ".chesslinked.ll");

    std::string linkedResult =
        runChessLlvmLink(chessHackPath, chessLinkedPath, aieTarget, tmpDirName);
    if (linkedResult.empty()) {
      return failure();
    }

    if (verbose) {
      llvm::outs() << "Linked with chess intrinsic wrapper: " << chessLinkedPath
                   << "\n";
    }

    // Step 4c: Compile with xchesscc_wrapper
    // Find xchesscc_wrapper in PATH
    auto xchessccWrapperPath = sys::findProgramByName("xchesscc_wrapper");
    if (!xchessccWrapperPath) {
      llvm::errs() << "Error: Could not find xchesscc_wrapper in PATH\n";
      return failure();
    }

    SmallString<128> workDir(tmpDirName);
    sys::path::append(workDir, "work");

    // xchesscc_wrapper <target> +w <work> -c -d +Wclang,-xir -f <input.ll> -o
    // <output.o>
    std::string aieTargetLower = aieTarget.lower();
    SmallVector<std::string, 16> xchessCmd = {*xchessccWrapperPath,
                                              aieTargetLower,
                                              "+w",
                                              workDir.str().str(),
                                              "-c",
                                              "-d",
                                              "+Wclang,-xir",
                                              "-f",
                                              chessLinkedPath.str().str(),
                                              "-o",
                                              objPath.str().str()};

    if (!executeCommand(xchessCmd)) {
      llvm::errs() << "Error running xchesscc_wrapper\n";
      return failure();
    }

    if (verbose) {
      llvm::outs() << "Compiled with xchesscc: " << objPath << "\n";
    }
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
    SmallVector<std::string, 12> optCmd = {peanoOpt,
                                           "--passes=default<O" + optLevelStr +
                                               ">,strip",
                                           "-inline-threshold=10",
                                           "-S",
                                           llvmIRPath.str().str(),
                                           "-o",
                                           optPath.str().str()};

    if (optLevel >= 3) {
      optCmd.insert(optCmd.begin() + 1, "-disable-loop-idiom-memset");
    }

    if (!executeCommand(optCmd)) {
      llvm::errs() << "Error running Peano opt\n";
      return failure();
    }

    // Run llc
    SmallVector<std::string, 10> llcCmd = {peanoLlc,
                                           optPath.str().str(),
                                           "-O" + optLevelStr,
                                           "--march=" + aieTarget.lower(),
                                           "--function-sections",
                                           "--filetype=obj",
                                           "-o",
                                           objPath.str().str()};

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

  if (xbridge) {
    // xbridge linking: generate BCF, extract link_with, link with
    // xchesscc_wrapper Note: xbridge works with both xchesscc and
    // peano-compiled object files

    // Generate BCF file
    SmallString<128> bcfPath(tmpDirName);
    sys::path::append(bcfPath, deviceName.str() + "_core_" +
                                   std::to_string(core.col) + "_" +
                                   std::to_string(core.row) + ".bcf");

    {
      std::error_code ec;
      raw_fd_ostream bcfFile(bcfPath, ec);
      if (ec) {
        llvm::errs() << "Error opening BCF file: " << ec.message() << "\n";
        return failure();
      }

      if (failed(xilinx::AIE::AIETranslateToBCF(moduleOp, bcfFile, core.col,
                                                core.row, deviceName))) {
        llvm::errs() << "Error generating BCF file\n";
        return failure();
      }
    }

    if (verbose) {
      llvm::outs() << "Generated BCF: " << bcfPath << "\n";
    }

    // Extract link_with files from BCF
    std::vector<std::string> linkWithFiles = extractInputFilesFromBCF(bcfPath);

    // Handle link_with files: copy to .prj directory if needed
    std::string linkWithArgs;
    for (const auto &linkWithFile : linkWithFiles) {
      SmallString<256> srcPath;
      if (sys::path::is_absolute(linkWithFile)) {
        srcPath = linkWithFile;
      } else {
        SmallString<256> inputDir = sys::path::parent_path(inputFilename);
        if (inputDir.empty()) {
          sys::fs::current_path(inputDir);
        }
        srcPath = inputDir;
        sys::path::append(srcPath, linkWithFile);
        sys::path::remove_dots(srcPath, /*remove_dot_dot=*/true);
      }

      // Copy to .prj directory
      SmallString<256> destPath(tmpDirName);
      sys::path::append(destPath, sys::path::filename(linkWithFile));

      sys::fs::remove(destPath);
      std::error_code ec = sys::fs::copy_file(srcPath, destPath);
      if (ec) {
        llvm::errs() << "Error: Could not copy link_with file: " << srcPath
                     << " to " << destPath << ": " << ec.message() << "\n";
        return failure();
      }

      if (verbose) {
        llvm::outs() << "Copied link_with: " << srcPath << " -> " << destPath
                     << "\n";
      }

      if (!linkWithArgs.empty()) {
        linkWithArgs += " ";
      }
      linkWithArgs += destPath.str().str();
    }

    // Find xchesscc_wrapper
    auto xchessccWrapperPath = sys::findProgramByName("xchesscc_wrapper");
    if (!xchessccWrapperPath) {
      llvm::errs() << "Error: Could not find xchesscc_wrapper in PATH for "
                      "xbridge linking\n";
      return failure();
    }

    SmallString<128> workDir(tmpDirName);
    sys::path::append(workDir, "work");

    // Link: xchesscc_wrapper <target> +w <work> -d -f <obj> [link_with_files]
    // +l <bcf> -o <elf>
    std::string aieTargetLower = aieTarget.lower();
    SmallVector<std::string, 20> linkCmd = {*xchessccWrapperPath,
                                            aieTargetLower,
                                            "+w",
                                            workDir.str().str(),
                                            "-d",
                                            "-f",
                                            objPath.str().str()};

    // Add link_with files if any
    for (const auto &linkWithFile : linkWithFiles) {
      SmallString<256> localPath(tmpDirName);
      sys::path::append(localPath, sys::path::filename(linkWithFile));
      linkCmd.push_back(localPath.str().str());
    }

    linkCmd.push_back("+l");
    linkCmd.push_back(bcfPath.str().str());
    linkCmd.push_back("-o");
    linkCmd.push_back(elfPath.str().str());

    if (!executeCommand(linkCmd)) {
      llvm::errs() << "Error linking with xbridge\n";
      return failure();
    }

    if (verbose) {
      llvm::outs() << "Linked with xbridge: " << elfPath << "\n";
    }
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

    SmallVector<std::string, 20> linkCmd = {peanoClang, "-O" + optLevelStr,
                                            "--target=" + peanoTarget};

    // Explicitly specify Peano's lld linker to avoid using system ld
    if (sys::fs::can_execute(peanoLld)) {
      linkCmd.push_back("-fuse-ld=" + peanoLld.str().str());
    } else {
      // Fallback: try to use lld from Peano bin via -B
      linkCmd.push_back("-B" + peanoBinDir.str().str());
      linkCmd.push_back("-fuse-ld=lld");
    }

    linkCmd.push_back(objPath.str().str());

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

      // Remove destination file first if it exists (to ensure overwrite)
      sys::fs::remove(destLinkWith);

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

    linkCmd.push_back("-Wl,--gc-sections");
    linkCmd.push_back("-Wl,--orphan-handling=error");
    linkCmd.push_back("-Wl,-T," + absLdScriptPath.str().str());
    linkCmd.push_back("-o");
    linkCmd.push_back(elfPath.str().str());

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
compileCores(MLIRContext &context, ModuleOp moduleOp, Operation *deviceOp,
             StringRef deviceName, StringRef tmpDirName, StringRef aieTarget,
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
    if (failed(compileCore(context, moduleOp, deviceName, core, tmpDirName,
                           aieTarget, elfPath))) {
      return failure();
    }

    if (!elfPath.empty()) {
      elfPaths[{core.col, core.row}] = elfPath;
    }
  }

  return success();
}

// Update MLIR module with ELF file paths
/// Update the in-memory module with ELF paths for compiled cores.
/// This modifies the module directly without disk I/O.
static LogicalResult updateModuleWithElfs(
    ModuleOp moduleOp, StringRef deviceName,
    const std::map<std::pair<int, int>, std::string> &elfPaths) {

  if (elfPaths.empty()) {
    return success();
  }

  if (verbose) {
    llvm::outs() << "Updating module with ELF paths for device: " << deviceName
                 << "\n";
  }

  // Update cores with ELF paths
  moduleOp.walk([&](xilinx::AIE::DeviceOp devOp) {
    if (devOp.getSymName() != deviceName) {
      return;
    }

    devOp.walk([&](xilinx::AIE::CoreOp coreOp) {
      auto tileOp =
          dyn_cast<xilinx::AIE::TileOp>(coreOp.getTile().getDefiningOp());
      if (!tileOp)
        return;

      std::int32_t col = tileOp.getCol();
      std::int32_t row = tileOp.getRow();

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

  return success();
}

//===----------------------------------------------------------------------===//
// JSON Generation for xclbin Metadata
//===----------------------------------------------------------------------===//

static LogicalResult generateMemTopologyJson(StringRef jsonPath) {
  std::ofstream jsonFile(jsonPath.str());
  if (!jsonFile.is_open()) {
    llvm::errs() << "Error: Could not open file for writing: " << jsonPath
                 << "\n";
    return failure();
  }
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
  return success();
}

static LogicalResult generateKernelsJson(StringRef jsonPath,
                                         StringRef devName) {
  std::ofstream jsonFile(jsonPath.str());
  if (!jsonFile.is_open()) {
    llvm::errs() << "Error: Could not open file for writing: " << jsonPath
                 << "\n";
    return failure();
  }
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
  return success();
}

static LogicalResult generatePartitionJson(StringRef jsonPath,
                                           StringRef devName,
                                           StringRef pdiPath) {
  std::ofstream jsonFile(jsonPath.str());
  if (!jsonFile.is_open()) {
    llvm::errs() << "Error: Could not open file for writing: " << jsonPath
                 << "\n";
    return failure();
  }
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
  return success();
}

//===----------------------------------------------------------------------===//
// NPU Instruction Generation
//===----------------------------------------------------------------------===//

/// Generate NPU instructions from an in-memory module.
/// This clones the module since NPU lowering is destructive.
static LogicalResult generateNpuInstructions(ModuleOp moduleOp,
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

  // Clone the module since NPU lowering is destructive
  OwningOpRef<ModuleOp> clonedModule = moduleOp.clone();

  // Run NPU lowering passes in-memory on the clone
  if (failed(runNpuLoweringPipeline(*clonedModule))) {
    return failure();
  }

  // Dump intermediate if requested
  SmallString<128> npuLoweredPath(tmpDirName);
  sys::path::append(npuLoweredPath, devName.str() + "_npu_lowered.mlir");
  dumpModuleToFile(*clonedModule, npuLoweredPath, "NPU lowered module");

  // Step 2: Translate to NPU binary
  // Find device and generate instructions for each runtime sequence
  LogicalResult result = success();
  for (auto devOp : clonedModule->getOps<xilinx::AIE::DeviceOp>()) {
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

      // Generate NPU instructions using direct C++ API call.
      // This replaces the subprocess call to aie-translate --aie-npu-to-binary.
      std::vector<uint32_t> instructions;
      if (failed(xilinx::AIE::AIETranslateNpuToBinary(
              *clonedModule, instructions, devName, seqName))) {
        llvm::errs() << "Error generating NPU instructions for sequence: "
                     << seqName << "\n";
        result = failure();
        return;
      }

      // Write instructions to binary file
      std::error_code ec;
      raw_fd_ostream binFile(outputPath, ec, sys::fs::OpenFlags::OF_None);
      if (ec) {
        llvm::errs() << "Error opening NPU instructions file: " << ec.message()
                     << "\n";
        result = failure();
        return;
      }

      binFile.write(reinterpret_cast<const char *>(instructions.data()),
                    instructions.size() * sizeof(uint32_t));

      if (verbose) {
        llvm::outs() << "Wrote " << instructions.size()
                     << " instructions to: " << outputPath << "\n";
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

/// Generate CDO/PDI/xclbin artifacts from an in-memory module.
static LogicalResult generateCdoArtifacts(ModuleOp moduleOp,
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

  // Generate CDO files using direct C++ API call.
  // This replaces the subprocess call to aie-translate --aie-generate-cdo.
  // AIETranslateToCDODirect generates CDO files directly to the work
  // directory: {devName}_aie_cdo_elfs.bin, {devName}_aie_cdo_init.bin,
  // {devName}_aie_cdo_enable.bin
  if (failed(xilinx::AIE::AIETranslateToCDODirect(moduleOp, tmpDirName, devName,
                                                  /*bigEndian=*/false,
                                                  /*emitUnified=*/false,
                                                  /*cdoDebug=*/false,
                                                  /*aieSim=*/aiesim,
                                                  /*xaieDebug=*/false,
                                                  /*enableCores=*/true))) {
    llvm::errs() << "Error generating CDO files\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Generated CDO files in: " << tmpDirName << "\n";
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

    SmallVector<std::string, 8> bootgenCmd = {bootgenPath,
                                              "-arch",
                                              "versal",
                                              "-image",
                                              bifPath.str().str(),
                                              "-o",
                                              pdiPath.str().str(),
                                              "-w"};

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
      if (failed(generateMemTopologyJson(memTopoPath)))
        return failure();

      SmallString<128> kernelsPath(tmpDirName);
      sys::path::append(kernelsPath, devName.str() + "_kernels.json");
      if (failed(generateKernelsJson(kernelsPath, devName)))
        return failure();

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

      if (failed(generatePartitionJson(partitionPath, devName, absPdiPath)))
        return failure();

      // Build xclbin
      std::string xclbinFileName = formatString(xclbinName, devName);
      SmallString<128> xclbinPath;
      if (sys::path::is_absolute(xclbinFileName)) {
        xclbinPath = xclbinFileName;
      } else {
        xclbinPath = xclbinFileName;
      }

      SmallVector<std::string, 16> xclbinCmd = {
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

static LogicalResult compileAIEModule(MLIRContext &context, ModuleOp moduleOp,
                                      StringRef tmpDirName) {
  if (verbose) {
    llvm::outs() << "Starting AIE compilation in directory: " << tmpDirName
                 << "\n";
  }

  // Count devices and cores for verbose output
  unsigned deviceCount = 0;
  if (verbose) {
    for (auto deviceOp : moduleOp.getOps<xilinx::AIE::DeviceOp>()) {
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

  // Dump input MLIR if requested
  SmallString<128> inputPath(tmpDirName);
  sys::path::append(inputPath, "input.mlir");
  dumpModuleToFile(moduleOp, inputPath, "input MLIR");

  // Detect AIE target early from the input module
  std::string aieTarget = "aie2"; // Default
  for (auto deviceOp : moduleOp.getOps<xilinx::AIE::DeviceOp>()) {
    if (!deviceName.empty() && deviceOp.getSymName() != deviceName) {
      continue;
    }
    aieTarget = getAIETargetForDevice(moduleOp, deviceOp.getSymName());
    break;
  }

  if (verbose) {
    llvm::outs() << "Detected AIE target: " << aieTarget << "\n";
  }

  // If device filtering is requested, remove non-matching devices from the
  // module before running passes. This prevents passes from running on devices
  // that won't be compiled, which can cause issues in some environments.
  if (!deviceName.empty()) {
    SmallVector<xilinx::AIE::DeviceOp> toErase;
    for (auto deviceOp : moduleOp.getOps<xilinx::AIE::DeviceOp>()) {
      if (deviceOp.getSymName() != deviceName) {
        toErase.push_back(deviceOp);
      }
    }
    for (auto deviceOp : toErase) {
      if (verbose) {
        llvm::outs() << "Removing non-matching device: "
                     << deviceOp.getSymName() << "\n";
      }
      deviceOp.erase();
    }
  }

  // Step 1: Run resource allocation and lowering passes in-memory
  if (failed(runResourceAllocationPipeline(moduleOp, aieTarget))) {
    return failure();
  }

  // Write intermediate file only if we're compiling cores (LLVM lowering
  // subprocess needs it)
  // TODO: Eliminate this when LLVM lowering is converted to in-memory
  SmallString<128> withAddressesPath(tmpDirName);
  sys::path::append(withAddressesPath, "input_with_addresses.mlir");
  if (compile) {
    std::error_code ec;
    raw_fd_ostream file(withAddressesPath, ec);
    if (ec) {
      llvm::errs() << "Error writing intermediate MLIR: " << ec.message()
                   << "\n";
      return failure();
    }
    moduleOp->print(file);

    if (verbose) {
      llvm::outs() << "Wrote module with addresses to: " << withAddressesPath
                   << "\n";
    }
  }

  // Step 2: Run routing in-memory
  if (failed(runRoutingPipeline(moduleOp))) {
    return failure();
  }

  // Dump physical module if requested (for debugging only)
  SmallString<128> physicalPath(tmpDirName);
  sys::path::append(physicalPath, "input_physical.mlir");
  dumpModuleToFile(moduleOp, physicalPath, "physical module");

  // Step 3: Compile cores and generate artifacts for each device
  for (auto deviceOp : moduleOp.getOps<xilinx::AIE::DeviceOp>()) {
    // Filter by device name if specified
    if (!deviceName.empty() && deviceOp.getSymName() != deviceName) {
      continue;
    }

    StringRef devName = deviceOp.getSymName();

    if (verbose) {
      llvm::outs() << "\nProcessing device: " << devName << "\n";
    }

    // Compile cores using in-memory LLVM lowering and translation
    std::map<std::pair<int, int>, std::string> elfPaths;
    if (failed(compileCores(context, moduleOp, deviceOp, devName, tmpDirName,
                            aieTarget, elfPaths))) {
      return failure();
    }

    // Update module with ELF paths in-memory (no disk I/O)
    if (failed(updateModuleWithElfs(moduleOp, devName, elfPaths))) {
      return failure();
    }

    // Dump module with ELFs if requested (for debugging only)
    SmallString<128> physicalWithElfsPath(tmpDirName);
    sys::path::append(physicalWithElfsPath,
                      devName.str() + "_physical_with_elfs.mlir");
    dumpModuleToFile(moduleOp, physicalWithElfsPath, "module with ELFs");

    // Generate NPU instructions from in-memory module
    if (failed(generateNpuInstructions(moduleOp, tmpDirName, devName))) {
      return failure();
    }

    // Generate CDO/PDI/xclbin from in-memory module
    if (failed(generateCdoArtifacts(moduleOp, tmpDirName, devName))) {
      return failure();
    }
  }

  return success();
}

static int processInputFile(StringRef inputFile, StringRef tmpDirName) {
  // Register passes for in-memory execution (must happen before context
  // creation)
  registerAllPasses();
  xilinx::registerConversionPasses();
  xilinx::AIE::registerAIEPasses();
  xilinx::AIEX::registerAIEXPasses();
  xilinx::aievec::registerAIEVecAnalysisPasses();
  xilinx::aievec::registerAIEVecPasses();
  xilinx::aievec::registerAIEVecPipelines();

  // Set up dialect registry with all MLIR and AIE dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  xilinx::registerAllDialects(registry);
  registerAllExtensions(registry);
  xilinx::aievec::registerTransformDialectExtension(registry);

  // Create context and attach registry
  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> inputModuleOp;

  if (inputFile.empty()) {
    llvm::errs() << "Error: No input file specified\n";
    return 1;
  }

  ParserConfig parseConfig(&context);
  SourceMgr sourceMgr;
  inputModuleOp = parseSourceFile<ModuleOp>(inputFile, sourceMgr, parseConfig);

  if (!inputModuleOp) {
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
  if (failed(compileAIEModule(context, inputModuleOp.get(), actualTmpDir))) {
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
