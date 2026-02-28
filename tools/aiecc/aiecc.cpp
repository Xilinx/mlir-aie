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
#include "mlir/IR/Diagnostics.h"
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
#include "llvm/Support/JSON.h"
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

#include "aiecc_aiesim.h"

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

// Optional library integrations for direct calls instead of subprocess
#ifdef AIECC_HAS_AIEBU_LIBRARY
// Use C API to avoid exception handling issues with LLVM's -fno-exceptions
#include <aiebu/aiebu.h>
#endif

#ifdef AIECC_HAS_BOOTGEN_LIBRARY
// Use C API wrapper that handles exceptions internally
#include "bootgen_c_api.h"
#endif

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

static cl::opt<bool> noAiesim("no-aiesim",
                              cl::desc("Do not generate aiesim Work folder"),
                              cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool>
    compileHost("compile-host",
                cl::desc("Enable compiling of the host program"),
                cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool>
    noCompileHost("no-compile-host",
                  cl::desc("Disable compiling of the host program"),
                  cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    hostTarget("host-target",
               cl::desc("Target architecture of the host program"),
               cl::init("x86_64-linux-gnu"), cl::cat(aieCompilerOptions));

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
    cl::desc(
        "Allocation scheme for AIE buffers (basic-sequential, bank-aware, "
        "or empty string for bank-aware with fallback to basic-sequential)"),
    cl::init(""), cl::cat(aieCompilerOptions));

static cl::opt<bool>
    generateNpuInsts("aie-generate-npu-insts",
                     cl::desc("Generate NPU instruction stream"),
                     cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    instsName("npu-insts-name",
              cl::desc("Output instructions filename for NPU target"),
              cl::init("{0}_{1}.bin"), cl::cat(aieCompilerOptions));

static cl::opt<bool> generateElf(
    "aie-generate-elf",
    cl::desc("Generate ELF for AIE control/configuration (via aiebu)"),
    cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    elfName("elf-name", cl::desc("Output ELF filename for instruction ELF"),
            cl::init("design.elf"), cl::cat(aieCompilerOptions));

static cl::opt<bool> generateFullElf(
    "generate-full-elf",
    cl::desc("Generate complete full ELF with PDIs using aiebu-asm"),
    cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    fullElfName("full-elf-name",
                cl::desc("Output filename for full ELF (default: aie.elf)"),
                cl::init("aie.elf"), cl::cat(aieCompilerOptions));

static cl::opt<bool> generateCdo("aie-generate-cdo",
                                 cl::desc("Generate libxaie v2 for CDO"),
                                 cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> generatePdi("aie-generate-pdi",
                                 cl::desc("Generate PDI binary"),
                                 cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string> pdiName("pdi-name", cl::desc("Output PDI filename"),
                                    cl::init("{0}.pdi"),
                                    cl::cat(aieCompilerOptions));

static cl::opt<bool> expandLoadPdis(
    "expand-load-pdis",
    cl::desc("Expand load_pdi operations to explicit configuration sequences"),
    cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> generateXclbin("aie-generate-xclbin",
                                    cl::desc("Generate xclbin"),
                                    cl::init(false),
                                    cl::cat(aieCompilerOptions));

static cl::opt<std::string> xclbinName("xclbin-name",
                                       cl::desc("Output xclbin filename"),
                                       cl::init("{0}.xclbin"),
                                       cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    xclbinInput("xclbin-input",
                cl::desc("Input xclbin to extend with additional kernel/PDI"),
                cl::init(""), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    xclbinKernelName("xclbin-kernel-name",
                     cl::desc("Kernel name in xclbin (default: MLIR_AIE)"),
                     cl::init("MLIR_AIE"), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    xclbinInstanceName("xclbin-instance-name",
                       cl::desc("Instance name in xclbin (default: MLIRAIE)"),
                       cl::init("MLIRAIE"), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    xclbinKernelId("xclbin-kernel-id",
                   cl::desc("Kernel ID in xclbin (default: 0x901)"),
                   cl::init("0x901"), cl::cat(aieCompilerOptions));

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

static cl::opt<bool>
    generateTxn("aie-generate-txn",
                cl::desc("Generate transaction binary MLIR for configuration"),
                cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    txnName("txn-name",
            cl::desc("Output filename for transaction MLIR. "
                     "`{0}` is replaced with device name."),
            cl::init("{0}_transaction.mlir"), cl::cat(aieCompilerOptions));

static cl::opt<bool>
    generateCtrlPkt("aie-generate-ctrlpkt",
                    cl::desc("Generate control packets for configuration"),
                    cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    ctrlPktName("ctrlpkt-name",
                cl::desc("Output filename for control packet binary data. "
                         "`{0}` is replaced with device name."),
                cl::init("{0}_ctrlpkt.bin"), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    ctrlPktDmaSeqName("ctrlpkt-dma-seq-name",
                      cl::desc("Output filename for control packet DMA "
                               "sequence. `{0}` is replaced with device name."),
                      cl::init("{0}_ctrlpkt_dma_seq.bin"),
                      cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    ctrlPktElfName("ctrlpkt-elf-name",
                   cl::desc("Output filename for control packet combined ELF. "
                            "`{0}` is replaced with device name."),
                   cl::init("{0}_ctrlpkt.elf"), cl::cat(aieCompilerOptions));

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

// Forward declarations
static std::string findAiebuAsm();

#ifdef AIECC_HAS_AIEBU_LIBRARY
static std::optional<std::vector<char>> readBinaryFile(StringRef path);
static std::optional<std::vector<char>> generateElfViaAiebuLibrary(
    aiebu_assembler_buffer_type type, const std::vector<char> &buffer1,
    const std::vector<char> &buffer2, const std::vector<char> &patchJson);
static std::optional<std::vector<char>>
generateElfViaAiebuLibraryConfig(const std::vector<char> &configJson);
static LogicalResult writeElfFile(StringRef path,
                                  const std::vector<char> &elfData);
#endif

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
    llvm::outs().flush();
  }

  if (dryRun) {
    if (verbose) {
      llvm::outs() << "Dry run - command not executed\n";
      llvm::outs().flush();
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

  // 2. Check AIETOOLS_ROOT environment variable
  if (const char *aietoolsEnv = std::getenv("AIETOOLS_ROOT")) {
    if (sys::fs::is_directory(aietoolsEnv)) {
      return aietoolsEnv;
    }
  }

  // 3. Find xchesscc in PATH and derive aietools from it
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

  // Remove nocreateundeforpoison attribute
  // This attribute appears in LLVM IR and may be followed by whitespace or EOL
  size_t pos = 0;
  while ((pos = result.find("nocreateundeforpoison", pos)) !=
         std::string::npos) {
    // Find the end of the attribute (skip trailing whitespace, but not
    // newlines)
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
  } else if (target == "aie" || target == "aie1") {
    return "target";
  }
  // Unknown target - warn and return empty
  llvm::errs() << "Warning: Unsupported AIE target for chess toolchain: "
               << aieTarget << "\n";
  return "";
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
                                    StringRef aieTarget) {
  StringRef aietoolsPath = getAietoolsDir();
  if (aietoolsPath.empty()) {
    llvm::errs() << "Error: Could not find aietools (xchesscc not in PATH)\n";
    return "";
  }

  // Build chess-llvm-link path
  // Path: <aietools>/tps/lnx64/<target>/bin/LNa64bin/chess-llvm-link
  std::string chessTarget = getChessTarget(aieTarget);
  if (chessTarget.empty()) {
    return "";
  }
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
  SmallVector<std::string, 8> cmd = {std::string(chessLlvmLinkPath),
                                     inputLLPath.str(),
                                     std::string(wrapperPath),
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
    llvm::errs() << "Warning: Could not read BCF file '" << bcfPath
                 << "': " << bufOrErr.getError().message() << "\n";
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
      << " target-backend=llvmir},"
      << "lower-affine,"
      << "aie-canonicalize-device,"
      << "aie.device("
      << "aie-assign-lock-ids,"
      << "aie-register-objectFifos,"
      << "aie-objectFifo-stateful-transform{"
      << "dynamic-objFifos=" << (dynamicObjFifos ? "true" : "false")
      << " packet-sw-objFifos=" << (packetSwObjFifos ? "true" : "false") << "},"
      << "aie-assign-bd-ids,"
      << "aie-lower-cascade-flows,"
      << "aie-lower-broadcast-packet,"
      << "aie-lower-multicast,"
      << "aie-assign-tile-controller-ids,"
      << "aie-generate-column-control-overlay{route-shim-to-tile-ctrl="
      << (ctrlPktOverlay ? "true" : "false") << "},"
      << "aie-assign-buffer-addresses{alloc-scheme=" << allocScheme.getValue()
      << "},"
      << "aie-vector-transfer-lowering{max-transfer-rank=1}"
      << "),"
      << "convert-scf-to-cf"
      << ")";
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
      << "canonicalize,"
      << "cse,"
      << "expand-strided-metadata,"
      << "lower-affine,"
      << "arith-expand,"
      << "finalize-memref-to-llvm,"
      << "convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},"
      << "convert-to-llvm{dynamic=true},"
      << "canonicalize,"
      << "cse"
      << ")";
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

  // Add expand-load-pdi pass at module level (after device nesting)
  // if --expand-load-pdis is specified
  if (expandLoadPdis) {
    pm.addPass(xilinx::AIEX::createAIEExpandLoadPdiPass());
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

/// Run the transaction generation pipeline in-memory using PassManager.
/// This converts the AIE device to a sequence of transaction binary operations.
/// The pipeline: convert-aie-to-transaction{elf-dir=... device-name=...}
static LogicalResult runTransactionPipeline(ModuleOp moduleOp, StringRef elfDir,
                                            StringRef devName) {
  MLIRContext *ctx = moduleOp.getContext();
  PassManager pm(ctx);

  if (verbose) {
    pm.enableVerifier(true);
  }

  // Build pass pipeline string with options
  std::string pipelineStr =
      "builtin.module(aie.device(convert-aie-to-transaction{elf-dir=" +
      elfDir.str() + " device-name=" + devName.str() + "}))";

  if (failed(parsePassPipeline(pipelineStr, pm))) {
    llvm::errs() << "Error: Failed to parse transaction pipeline\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Running transaction generation pipeline in-memory\n";
  }

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Error: Transaction generation pipeline failed\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Transaction generation pipeline completed successfully\n";
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
  // Note: aie-transform-bfp-types pass would go here if the design uses
  // block floating-point (BFP) types that require normalization before
  // lowering. Currently not needed for standard designs.

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
        runChessLlvmLink(chessHackPath, chessLinkedPath, aieTarget);
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
                                              std::string(workDir),
                                              "-c",
                                              "-d",
                                              "+Wclang,-xir",
                                              "-f",
                                              std::string(chessLinkedPath),
                                              "-o",
                                              std::string(objPath)};

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
                                           std::string(llvmIRPath),
                                           "-o",
                                           std::string(optPath)};

    if (optLevel >= 3) {
      optCmd.insert(optCmd.begin() + 1, "-disable-loop-idiom-memset");
    }

    if (!executeCommand(optCmd)) {
      llvm::errs() << "Error running Peano opt\n";
      return failure();
    }

    // Run llc
    SmallVector<std::string, 10> llcCmd = {peanoLlc,
                                           std::string(optPath),
                                           "-O" + optLevelStr,
                                           "--march=" + aieTarget.lower(),
                                           "--function-sections",
                                           "--filetype=obj",
                                           "-o",
                                           std::string(objPath)};

    if (!executeCommand(llcCmd)) {
      llvm::errs() << "Error running Peano llc\n";
      return failure();
    }
  }

  // Step 5: Link to ELF
  if (!link) {
    outElfPath = std::string(objPath);
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
    SmallVector<std::string, 20> linkCmd = {
        *xchessccWrapperPath, aieTargetLower, "+w",
        std::string(workDir), "-d",           "-f",
        std::string(objPath)};

    // Add link_with files if any
    for (const auto &linkWithFile : linkWithFiles) {
      SmallString<256> localPath(tmpDirName);
      sys::path::append(localPath, sys::path::filename(linkWithFile));
      linkCmd.push_back(std::string(localPath));
    }

    linkCmd.push_back("+l");
    linkCmd.push_back(std::string(bcfPath));
    linkCmd.push_back("-o");
    linkCmd.push_back(std::string(elfPath));

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
      linkCmd.push_back("-fuse-ld=" + std::string(peanoLld));
    } else {
      // Fallback: try to use lld from Peano bin via -B
      linkCmd.push_back("-B" + std::string(peanoBinDir));
      linkCmd.push_back("-fuse-ld=lld");
    }

    linkCmd.push_back(std::string(objPath));

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
    linkCmd.push_back("-Wl,-T," + std::string(absLdScriptPath));
    linkCmd.push_back("-o");
    linkCmd.push_back(std::string(elfPath));

    if (!executeCommand(linkCmd)) {
      llvm::errs() << "Error linking ELF file\n";
      return failure();
    }
  }

  outElfPath = std::string(elfPath);
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
  jsonFile << R"({
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
}
)";
  jsonFile.close();
  return success();
}

static LogicalResult generateKernelsJson(StringRef jsonPath,
                                         StringRef devName) {
  std::error_code ec;
  llvm::raw_fd_ostream jsonFile(jsonPath, ec);
  if (ec) {
    llvm::errs() << "Error: Could not open file for writing: " << jsonPath
                 << ": " << ec.message() << "\n";
    return failure();
  }

  // Build JSON using LLVM JSON API for proper escaping of user-provided values
  llvm::json::Object extendedData;
  extendedData["subtype"] = "DPU";
  extendedData["functional"] = "0";
  extendedData["dpu_kernel_id"] = xclbinKernelId.getValue();

  // Build arguments array
  llvm::json::Array arguments;

  llvm::json::Object arg0;
  arg0["name"] = "opcode";
  arg0["address-qualifier"] = "SCALAR";
  arg0["type"] = "uint64_t";
  arg0["offset"] = "0x00";
  arguments.push_back(std::move(arg0));

  llvm::json::Object arg1;
  arg1["name"] = "instr";
  arg1["memory-connection"] = "SRAM";
  arg1["address-qualifier"] = "GLOBAL";
  arg1["type"] = "char *";
  arg1["offset"] = "0x08";
  arguments.push_back(std::move(arg1));

  llvm::json::Object arg2;
  arg2["name"] = "ninstr";
  arg2["address-qualifier"] = "SCALAR";
  arg2["type"] = "uint32_t";
  arg2["offset"] = "0x10";
  arguments.push_back(std::move(arg2));

  // Add buffer object arguments bo0-bo4
  for (int i = 0; i < 5; i++) {
    llvm::json::Object boArg;
    boArg["name"] = "bo" + std::to_string(i);
    boArg["memory-connection"] = "HOST";
    boArg["address-qualifier"] = "GLOBAL";
    boArg["type"] = "void*";
    std::ostringstream offsetStr;
    offsetStr << "0x" << std::hex << (0x14 + i * 8);
    boArg["offset"] = offsetStr.str();
    arguments.push_back(std::move(boArg));
  }

  // Build instance
  llvm::json::Object instance;
  instance["name"] = xclbinInstanceName.getValue();

  llvm::json::Array instances;
  instances.push_back(std::move(instance));

  // Build kernel
  llvm::json::Object kernel;
  kernel["name"] = xclbinKernelName.getValue();
  kernel["type"] = "dpu";
  kernel["extended-data"] = std::move(extendedData);
  kernel["arguments"] = std::move(arguments);
  kernel["instances"] = std::move(instances);

  llvm::json::Array kernels;
  kernels.push_back(std::move(kernel));

  llvm::json::Object psKernels;
  psKernels["kernels"] = std::move(kernels);

  llvm::json::Object root;
  root["ps-kernels"] = std::move(psKernels);

  jsonFile << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)))
           << "\n";
  return success();
}

static LogicalResult generatePartitionJson(StringRef jsonPath,
                                           StringRef devName,
                                           StringRef pdiPath) {
  std::error_code ec;
  llvm::raw_fd_ostream jsonFile(jsonPath, ec);
  if (ec) {
    llvm::errs() << "Error: Could not open file for writing: " << jsonPath
                 << ": " << ec.message() << "\n";
    return failure();
  }

  // Build JSON using LLVM JSON API for proper escaping of paths and IDs
  llvm::json::Object partition;
  partition["column_width"] = 1;
  llvm::json::Array startColumns;
  startColumns.push_back(0);
  partition["start_columns"] = std::move(startColumns);

  // Build cdo_group
  llvm::json::Object cdoGroup;
  cdoGroup["name"] = "DPU";
  cdoGroup["type"] = "PRIMARY";
  cdoGroup["pdi_id"] = "0x01";
  llvm::json::Array dpuKernelIds;
  dpuKernelIds.push_back(xclbinKernelId.getValue());
  cdoGroup["dpu_kernel_ids"] = std::move(dpuKernelIds);
  llvm::json::Array preCdoGroups;
  preCdoGroups.push_back("0xC1");
  cdoGroup["pre_cdo_groups"] = std::move(preCdoGroups);

  llvm::json::Array cdoGroups;
  cdoGroups.push_back(std::move(cdoGroup));

  // Build PDI entry
  llvm::json::Object pdi;
  pdi["uuid"] = "00000000-0000-0000-0000-000000000000";
  pdi["file_name"] = pdiPath.str();
  pdi["cdo_groups"] = std::move(cdoGroups);

  llvm::json::Array pdis;
  pdis.push_back(std::move(pdi));

  // Build aie_partition
  llvm::json::Object aiePartition;
  aiePartition["name"] = "QoS";
  aiePartition["operations_per_cycle"] = "2048";
  aiePartition["inference_fingerprint"] = "23423";
  aiePartition["pre_post_fingerprint"] = "12345";
  aiePartition["partition"] = std::move(partition);
  aiePartition["PDIs"] = std::move(pdis);

  llvm::json::Object root;
  root["aie_partition"] = std::move(aiePartition);

  jsonFile << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)))
           << "\n";
  return success();
}

/// Extract AIE_PARTITION section from existing xclbin and merge new PDI.
/// Returns the merged partition JSON path, or empty on failure.
static LogicalResult extractAndMergePartition(StringRef inputXclbin,
                                              StringRef newPartitionPath,
                                              StringRef outputPartitionPath,
                                              StringRef tmpDirName) {
  // Find xclbinutil
  auto xclbinutilPath = sys::findProgramByName("xclbinutil");
  if (!xclbinutilPath) {
    llvm::errs() << "Error: xclbinutil not found\n";
    return failure();
  }

  // Extract partition from input xclbin
  SmallString<128> inputPartitionPath(tmpDirName);
  sys::path::append(inputPartitionPath, "input_aie_partition.json");

  SmallVector<std::string, 10> extractCmd = {
      *xclbinutilPath,
      "--dump-section",
      "AIE_PARTITION:JSON:" + std::string(inputPartitionPath),
      "--force",
      "--quiet",
      "--input",
      inputXclbin.str()};

  if (!executeCommand(extractCmd)) {
    llvm::errs() << "Error extracting AIE_PARTITION from input xclbin\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Extracted partition from input xclbin: "
                 << inputPartitionPath << "\n";
  }

  // Read input partition JSON
  auto inputBufOrErr = llvm::MemoryBuffer::getFile(inputPartitionPath);
  if (!inputBufOrErr) {
    llvm::errs() << "Error reading input partition JSON: "
                 << inputBufOrErr.getError().message() << "\n";
    return failure();
  }

  // Read new partition JSON
  auto newBufOrErr = llvm::MemoryBuffer::getFile(newPartitionPath);
  if (!newBufOrErr) {
    llvm::errs() << "Error reading new partition JSON: "
                 << newBufOrErr.getError().message() << "\n";
    return failure();
  }

  // Parse both JSON files
  auto inputJson = llvm::json::parse(inputBufOrErr.get()->getBuffer());
  if (!inputJson) {
    llvm::errs() << "Error parsing input partition JSON: "
                 << llvm::toString(inputJson.takeError()) << "\n";
    return failure();
  }

  auto newJson = llvm::json::parse(newBufOrErr.get()->getBuffer());
  if (!newJson) {
    llvm::errs() << "Error parsing new partition JSON: "
                 << llvm::toString(newJson.takeError()) << "\n";
    return failure();
  }

  // Get the PDIs arrays from both
  auto *inputObj = inputJson->getAsObject();
  auto *newObj = newJson->getAsObject();
  if (!inputObj || !newObj) {
    llvm::errs() << "Error: JSON files are not objects\n";
    return failure();
  }

  auto *inputPartition = inputObj->getObject("aie_partition");
  auto *newPartition = newObj->getObject("aie_partition");
  if (!inputPartition || !newPartition) {
    llvm::errs() << "Error: Missing aie_partition in JSON\n";
    return failure();
  }

  auto *inputPDIs = inputPartition->getArray("PDIs");
  auto *newPDIs = newPartition->getArray("PDIs");
  if (!inputPDIs || !newPDIs) {
    llvm::errs() << "Error: Missing PDIs array in partition JSON\n";
    return failure();
  }

  // Append new PDIs to input PDIs
  for (auto &pdi : *newPDIs) {
    inputPDIs->push_back(std::move(pdi));
  }

  // Write merged partition JSON
  std::error_code ec;
  raw_fd_ostream outFile(outputPartitionPath, ec);
  if (ec) {
    llvm::errs() << "Error writing merged partition JSON: " << ec.message()
                 << "\n";
    return failure();
  }
  outFile << llvm::formatv("{0:2}", *inputJson);

  if (verbose) {
    llvm::outs() << "Merged partition JSON: " << outputPartitionPath << "\n";
  }

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
  // Full ELF requires NPU instructions
  if (!generateNpuInsts && !generateFullElf) {
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

      // Determine output path:
      // - If generateNpuInsts is set, use the filename as-is (relative to cwd)
      //   This matches Python aiecc.py behavior where --npu-insts-name
      //   specifies the output path relative to the current directory.
      // - Otherwise (e.g., for generateFullElf), write to tmpDirName so
      //   the full ELF generation can find them.
      SmallString<128> outputPath;
      if (generateNpuInsts) {
        outputPath = outputFileName;
      } else {
        outputPath = tmpDirName;
        sys::path::append(outputPath, outputFileName);
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
// Transaction Generation
//===----------------------------------------------------------------------===//

/// Generate transaction MLIR output for a device.
/// This converts the device configuration to transaction binary operations.
static LogicalResult generateTransactionOutput(ModuleOp moduleOp,
                                               StringRef tmpDirName,
                                               StringRef devName) {
  if (!generateTxn) {
    return success();
  }

  if (verbose) {
    llvm::outs() << "Generating transaction MLIR for device: " << devName
                 << "\n";
  }

  // In dry-run mode, just show what would be done
  if (dryRun) {
    if (verbose) {
      llvm::outs() << "Would generate transaction MLIR for device: " << devName
                   << "\n";
    }
    return success();
  }

  // Clone the module since transaction generation is destructive
  OwningOpRef<ModuleOp> clonedModule = moduleOp.clone();

  // Run transaction pipeline
  if (failed(runTransactionPipeline(*clonedModule, tmpDirName, devName))) {
    return failure();
  }

  // Write the transaction MLIR to output file
  std::string outputFileName = formatString(txnName, devName);
  SmallString<128> outputPath;
  if (sys::path::is_absolute(outputFileName)) {
    outputPath = outputFileName;
  } else {
    outputPath = tmpDirName;
    sys::path::append(outputPath, outputFileName);
  }

  // Dump intermediate MLIR
  SmallString<128> txnMlirPath(tmpDirName);
  sys::path::append(txnMlirPath, devName.str() + "_txn.mlir");
  dumpModuleToFile(*clonedModule, txnMlirPath, "Transaction MLIR");

  // Copy to output location
  std::error_code ec;
  raw_fd_ostream outFile(outputPath, ec, sys::fs::OpenFlags::OF_None);
  if (ec) {
    llvm::errs() << "Error opening transaction output file: " << ec.message()
                 << "\n";
    return failure();
  }

  clonedModule->print(outFile);

  if (verbose) {
    llvm::outs() << "Wrote transaction MLIR to: " << outputPath << "\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Control Packet Generation
//===----------------------------------------------------------------------===//

/// Generate control packet output for a device.
/// This converts the device configuration to control packets and generates:
/// 1. Control packet binary data (ctrlpkt.bin)
/// 2. Control packet DMA sequence binary (ctrlpkt_dma_seq.bin)
/// 3. Combined ELF file (ctrlpkt.elf) if aiebu-asm is available
static LogicalResult generateControlPacketOutput(ModuleOp moduleOp,
                                                 StringRef tmpDirName,
                                                 StringRef devName) {
  if (!generateCtrlPkt) {
    return success();
  }

  if (verbose) {
    llvm::outs() << "Generating control packets for device: " << devName
                 << "\n";
  }

  // In dry-run mode, just show what would be done
  if (dryRun) {
    if (verbose) {
      llvm::outs() << "Would generate control packets for device: " << devName
                   << "\n";
    }
    return success();
  }

  // Clone the module since control packet generation is destructive
  OwningOpRef<ModuleOp> clonedModule = moduleOp.clone();

  // Run control packet pipeline (Step 1: convert to control packets)
  MLIRContext *ctx = clonedModule->getContext();
  {
    PassManager pm(ctx);
    if (verbose) {
      pm.enableVerifier(true);
    }

    // Build pass pipeline for control packet conversion
    std::string pipelineStr =
        "builtin.module(aie.device(convert-aie-to-transaction{elf-dir=" +
        tmpDirName.str() +
        "},aie-txn-to-ctrl-packet,aie-legalize-ctrl-packet))";

    if (failed(parsePassPipeline(pipelineStr, pm))) {
      llvm::errs()
          << "Error: Failed to parse control packet conversion pipeline\n";
      return failure();
    }

    if (verbose) {
      llvm::outs() << "Running control packet conversion pipeline in-memory\n";
    }

    if (failed(pm.run(*clonedModule))) {
      llvm::errs() << "Error: Control packet conversion pipeline failed\n";
      return failure();
    }
  }

  // Dump intermediate control packet MLIR
  SmallString<128> ctrlPktMlirPath(tmpDirName);
  sys::path::append(ctrlPktMlirPath, devName.str() + "_ctrlpkt.mlir");
  dumpModuleToFile(*clonedModule, ctrlPktMlirPath, "Control packet MLIR");

  // Generate control packet binary using AIETranslateControlPacketsToUI32Vec
  std::string ctrlPktBinFileName = formatString(ctrlPktName, devName);
  SmallString<128> ctrlPktBinPath;
  if (sys::path::is_absolute(ctrlPktBinFileName)) {
    ctrlPktBinPath = ctrlPktBinFileName;
  } else {
    ctrlPktBinPath = tmpDirName;
    sys::path::append(ctrlPktBinPath, ctrlPktBinFileName);
  }

  std::vector<uint32_t> ctrlPktInstructions;
  if (failed(xilinx::AIE::AIETranslateControlPacketsToUI32Vec(
          *clonedModule, ctrlPktInstructions, devName, ""))) {
    llvm::errs() << "Error generating control packet binary for device: "
                 << devName << "\n";
    return failure();
  }

  // Write control packet binary
  {
    std::error_code ec;
    raw_fd_ostream binFile(ctrlPktBinPath, ec, sys::fs::OpenFlags::OF_None);
    if (ec) {
      llvm::errs() << "Error opening control packet binary file: "
                   << ec.message() << "\n";
      return failure();
    }
    binFile.write(reinterpret_cast<const char *>(ctrlPktInstructions.data()),
                  ctrlPktInstructions.size() * sizeof(uint32_t));
  }

  if (verbose) {
    llvm::outs() << "Wrote " << ctrlPktInstructions.size()
                 << " control packet instructions to: " << ctrlPktBinPath
                 << "\n";
  }

  // Step 2: Convert control packets to DMA and NPU for DMA sequence
  {
    PassManager pm(ctx);
    if (verbose) {
      pm.enableVerifier(true);
    }

    OpPassManager &devicePm = pm.nest<xilinx::AIE::DeviceOp>();
    devicePm.addPass(xilinx::AIEX::createAIECtrlPacketToDmaPass());
    devicePm.addPass(xilinx::AIEX::createAIEDmaToNpuPass());

    if (verbose) {
      llvm::outs() << "Running control packet to DMA pipeline in-memory\n";
    }

    if (failed(pm.run(*clonedModule))) {
      llvm::errs() << "Error: Control packet to DMA pipeline failed\n";
      return failure();
    }
  }

  // Dump intermediate DMA sequence MLIR
  SmallString<128> dmaSeqMlirPath(tmpDirName);
  sys::path::append(dmaSeqMlirPath, devName.str() + "_ctrlpkt_dma_seq.mlir");
  dumpModuleToFile(*clonedModule, dmaSeqMlirPath,
                   "Control packet DMA sequence MLIR");

  // Generate DMA sequence binary using AIETranslateNpuToBinary
  std::string dmaSeqBinFileName = formatString(ctrlPktDmaSeqName, devName);
  SmallString<128> dmaSeqBinPath;
  if (sys::path::is_absolute(dmaSeqBinFileName)) {
    dmaSeqBinPath = dmaSeqBinFileName;
  } else {
    dmaSeqBinPath = tmpDirName;
    sys::path::append(dmaSeqBinPath, dmaSeqBinFileName);
  }

  std::vector<uint32_t> dmaSeqInstructions;
  // Use "seq" as sequence name to match Python behavior
  if (failed(xilinx::AIE::AIETranslateNpuToBinary(*clonedModule,
                                                  dmaSeqInstructions, devName,
                                                  "" /* all sequences */))) {
    llvm::errs() << "Error generating control packet DMA sequence for device: "
                 << devName << "\n";
    return failure();
  }

  // Write DMA sequence binary
  {
    std::error_code ec;
    raw_fd_ostream binFile(dmaSeqBinPath, ec, sys::fs::OpenFlags::OF_None);
    if (ec) {
      llvm::errs() << "Error opening DMA sequence binary file: " << ec.message()
                   << "\n";
      return failure();
    }
    binFile.write(reinterpret_cast<const char *>(dmaSeqInstructions.data()),
                  dmaSeqInstructions.size() * sizeof(uint32_t));
  }

  if (verbose) {
    llvm::outs() << "Wrote " << dmaSeqInstructions.size()
                 << " DMA sequence instructions to: " << dmaSeqBinPath << "\n";
  }

  // Step 3: Generate combined ELF using aiebu library or aiebu-asm subprocess
  std::string elfFileName = formatString(ctrlPktElfName, devName);
  SmallString<128> elfPath;
  if (sys::path::is_absolute(elfFileName)) {
    elfPath = elfFileName;
  } else {
    elfPath = tmpDirName;
    sys::path::append(elfPath, elfFileName);
  }

  // Count runtime sequence arguments to determine ctrl_pkt buffer index
  int ctrlIdx = 0;
  for (auto devOp : clonedModule->getOps<xilinx::AIE::DeviceOp>()) {
    if (!deviceName.empty() && devOp.getSymName() != devName) {
      continue;
    }
    for (auto seqOp : devOp.getOps<xilinx::AIE::RuntimeSequenceOp>()) {
      // Get the number of arguments in the runtime sequence
      if (!seqOp.getBody().empty()) {
        ctrlIdx = seqOp.getBody().front().getNumArguments();
        break;
      }
    }
    break;
  }

  // Build external_buffers JSON string
  uint64_t ctrlPktSize = ctrlPktInstructions.size() * sizeof(uint32_t);
  std::string extBufJsonStr;
  {
    llvm::json::Object bufferCtrl;
    bufferCtrl["xrt_id"] = ctrlIdx;
    bufferCtrl["logical_id"] = -1;
    bufferCtrl["size_in_bytes"] = static_cast<int64_t>(ctrlPktSize);
    bufferCtrl["ctrl_pkt_buffer"] = 1;
    bufferCtrl["name"] = "runtime_control_packet";

    llvm::json::Object externalBuffers;
    externalBuffers["buffer_ctrl"] = std::move(bufferCtrl);

    llvm::json::Object root;
    root["external_buffers"] = std::move(externalBuffers);

    llvm::raw_string_ostream os(extBufJsonStr);
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
  }

#ifdef AIECC_HAS_AIEBU_LIBRARY
  // Try using aiebu library directly
  {
    // Convert instructions to char buffers
    std::vector<char> instrBuffer(
        reinterpret_cast<const char *>(dmaSeqInstructions.data()),
        reinterpret_cast<const char *>(dmaSeqInstructions.data()) +
            dmaSeqInstructions.size() * sizeof(uint32_t));

    std::vector<char> ctrlPktBuffer(
        reinterpret_cast<const char *>(ctrlPktInstructions.data()),
        reinterpret_cast<const char *>(ctrlPktInstructions.data()) +
            ctrlPktInstructions.size() * sizeof(uint32_t));

    std::vector<char> patchJson(extBufJsonStr.begin(), extBufJsonStr.end());

    if (verbose) {
      llvm::outs() << "Using aiebu library for control packet ELF generation\n";
    }

    auto elfData = generateElfViaAiebuLibrary(
        aiebu_assembler_buffer_type_blob_instr_transaction, instrBuffer,
        ctrlPktBuffer, patchJson);
    if (elfData && !elfData->empty()) {
      if (succeeded(writeElfFile(elfPath, *elfData))) {
        if (verbose) {
          llvm::outs() << "Generated control packet ELF via library: "
                       << elfPath << "\n";
        }
        return success();
      }
    }
    // Fall through to subprocess on library failure
    if (verbose) {
      llvm::outs()
          << "aiebu library failed for control packet ELF, falling back\n";
    }
  }
#endif // AIECC_HAS_AIEBU_LIBRARY

  // Subprocess fallback: find aiebu-asm
  std::string aiebuAsmPath = findAiebuAsm();
  if (aiebuAsmPath.empty()) {
    if (verbose) {
      llvm::outs() << "aiebu-asm not found, skipping control packet ELF "
                      "generation\n";
    }
    return success();
  }

  // Generate external_buffers.json for aiebu-asm subprocess
  SmallString<128> extBufJsonPath(tmpDirName);
  sys::path::append(extBufJsonPath, "external_buffers.json");
  {
    std::error_code ec;
    raw_fd_ostream jsonFile(extBufJsonPath, ec, sys::fs::OpenFlags::OF_None);
    if (ec) {
      llvm::errs() << "Error creating external_buffers.json: " << ec.message()
                   << "\n";
      return failure();
    }
    jsonFile << extBufJsonStr;
  }

  // Run aiebu-asm to generate combined ELF
  std::vector<StringRef> args;
  args.push_back(aiebuAsmPath);
  args.push_back("-t");
  args.push_back("aie2txn");
  args.push_back("-c");
  args.push_back(dmaSeqBinPath);
  args.push_back("-o");
  args.push_back(elfPath);
  args.push_back("-j");
  args.push_back(extBufJsonPath);
  args.push_back("-p");
  args.push_back(ctrlPktBinPath);

  if (verbose) {
    llvm::outs() << "Running: ";
    for (const auto &arg : args) {
      llvm::outs() << arg << " ";
    }
    llvm::outs() << "\n";
  }

  std::string errMsg;
  int exitCode = sys::ExecuteAndWait(aiebuAsmPath, args, std::nullopt, {},
                                     /*SecondsToWait=*/0,
                                     /*MemoryLimit=*/0, &errMsg);
  if (exitCode != 0) {
    llvm::errs() << "Error running aiebu-asm for control packet ELF: " << errMsg
                 << "\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Generated control packet ELF: " << elfPath << "\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ELF Generation (via aiebu-asm)
//===----------------------------------------------------------------------===//

/// Find aiebu-asm binary in PATH or at default locations.
static std::string findAiebuAsm() {
  // First try PATH
  auto aiebuAsmPath = sys::findProgramByName("aiebu-asm");
  if (aiebuAsmPath) {
    return *aiebuAsmPath;
  }

  // Try XRT installation location (common case)
  std::string xrtPath = "/opt/xilinx/xrt/bin/aiebu-asm";
  if (sys::fs::can_execute(xrtPath)) {
    return xrtPath;
  }

  // Try standalone aiebu installation
  std::string defaultPath = "/opt/xilinx/aiebu/bin/aiebu-asm";
  if (sys::fs::can_execute(defaultPath)) {
    return defaultPath;
  }

  return "";
}

//===----------------------------------------------------------------------===//
// AIEBU Library Integration (optional, compile-time enabled)
// Uses the C API to avoid exception handling issues with LLVM's -fno-exceptions
//===----------------------------------------------------------------------===//

#ifdef AIECC_HAS_AIEBU_LIBRARY
/// Helper to read a binary file into a vector of chars.
static std::optional<std::vector<char>> readBinaryFile(StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    return std::nullopt;
  }
  auto &buffer = *bufferOrErr;
  const char *data = buffer->getBufferStart();
  size_t size = buffer->getBufferSize();
  return std::vector<char>(data, data + size);
}

/// Generate ELF using aiebu C API with transaction instructions.
/// Returns the ELF data or nullopt on failure.
static std::optional<std::vector<char>>
generateElfViaAiebuLibrary(aiebu_assembler_buffer_type type,
                           const std::vector<char> &buffer1,
                           const std::vector<char> &buffer2 = {},
                           const std::vector<char> &patchJson = {}) {
  void *elfBuf = nullptr;
  int result = aiebu_assembler_get_elf(
      type, buffer1.data(), buffer1.size(),
      buffer2.empty() ? nullptr : buffer2.data(), buffer2.size(), &elfBuf,
      patchJson.empty() ? nullptr : patchJson.data(), patchJson.size(),
      nullptr, // libs
      nullptr, // libpaths
      nullptr, // pm_ctrlpkts
      0);      // pm_ctrlpkt_size

  if (result <= 0 || elfBuf == nullptr) {
    if (verbose) {
      llvm::errs() << "aiebu library error: returned " << result << "\n";
    }
    if (elfBuf) {
      free(elfBuf);
    }
    return std::nullopt;
  }

  // Copy data and free the allocated buffer
  std::vector<char> elfData(static_cast<char *>(elfBuf),
                            static_cast<char *>(elfBuf) + result);
  free(elfBuf);
  return elfData;
}

/// Generate ELF using aiebu C API with config JSON (for aie2_config type).
/// Returns the ELF data or nullopt on failure.
static std::optional<std::vector<char>>
generateElfViaAiebuLibraryConfig(const std::vector<char> &configJson) {
  void *elfBuf = nullptr;
  int result = aiebu_assembler_get_elf(aiebu_assembler_buffer_type_aie2_config,
                                       configJson.data(), configJson.size(),
                                       nullptr, 0,          // buffer2
                                       &elfBuf, nullptr, 0, // patch_json
                                       nullptr,             // libs
                                       nullptr,             // libpaths
                                       nullptr,             // pm_ctrlpkts
                                       0);                  // pm_ctrlpkt_size

  if (result <= 0 || elfBuf == nullptr) {
    if (verbose) {
      llvm::errs() << "aiebu library error (config): returned " << result
                   << "\n";
    }
    if (elfBuf) {
      free(elfBuf);
    }
    return std::nullopt;
  }

  std::vector<char> elfData(static_cast<char *>(elfBuf),
                            static_cast<char *>(elfBuf) + result);
  free(elfBuf);
  return elfData;
}

/// Write ELF data to a file.
static LogicalResult writeElfFile(StringRef path,
                                  const std::vector<char> &elfData) {
  std::error_code ec;
  raw_fd_ostream elfFile(path, ec, sys::fs::OpenFlags::OF_None);
  if (ec) {
    llvm::errs() << "Error writing ELF file " << path << ": " << ec.message()
                 << "\n";
    return failure();
  }
  elfFile.write(elfData.data(), elfData.size());
  return success();
}
#endif // AIECC_HAS_AIEBU_LIBRARY

/// Generate ELF file from NPU instruction binary using aiebu-asm.
/// This generates an ELF that can be loaded using xrt::elf API.
static LogicalResult generateElfFromInsts(ModuleOp moduleOp,
                                          StringRef tmpDirName,
                                          StringRef devName) {
  if (!generateElf) {
    return success();
  }

  if (verbose) {
    llvm::outs() << "Generating ELF for device: " << devName << "\n";
  }

  if (dryRun) {
    if (verbose) {
      llvm::outs() << "Would generate ELF for device: " << devName << "\n";
    }
    return success();
  }

  // Clone and run NPU lowering (same as generateNpuInstructions)
  OwningOpRef<ModuleOp> clonedModule = moduleOp.clone();

  if (failed(runNpuLoweringPipeline(*clonedModule))) {
    llvm::errs() << "Error running NPU lowering pipeline for ELF generation\n";
    return failure();
  }

  // Generate instructions for each sequence and combine them
  std::vector<uint32_t> allInstructions;
  LogicalResult result = success();

  for (auto devOp : clonedModule->getOps<xilinx::AIE::DeviceOp>()) {
    if (!deviceName.empty() && devOp.getSymName() != devName) {
      continue;
    }

    devOp.walk([&](xilinx::AIE::RuntimeSequenceOp seq) {
      if (failed(result)) {
        return;
      }

      if (!sequenceName.empty() && seq.getSymName() != sequenceName) {
        return;
      }

      std::vector<uint32_t> instructions;
      if (failed(xilinx::AIE::AIETranslateNpuToBinary(
              *clonedModule, instructions, devName, seq.getSymName()))) {
        llvm::errs() << "Error generating NPU instructions for ELF: "
                     << seq.getSymName() << "\n";
        result = failure();
        return;
      }

      // Append to combined instructions
      allInstructions.insert(allInstructions.end(), instructions.begin(),
                             instructions.end());
    });
  }

  if (failed(result)) {
    return failure();
  }

  if (allInstructions.empty()) {
    llvm::errs() << "Warning: No NPU instructions generated for ELF\n";
    return success();
  }

  // Determine output ELF path
  std::string outputElfPath = formatString(elfName, devName.str(), "");

#ifdef AIECC_HAS_AIEBU_LIBRARY
  // Try using aiebu library directly (avoids subprocess overhead)
  {
    // Convert instructions to char buffer
    std::vector<char> instrBuffer(
        reinterpret_cast<const char *>(allInstructions.data()),
        reinterpret_cast<const char *>(allInstructions.data()) +
            allInstructions.size() * sizeof(uint32_t));

    if (verbose) {
      llvm::outs() << "Using aiebu library for ELF generation\n";
    }

    auto elfData = generateElfViaAiebuLibrary(
        aiebu_assembler_buffer_type_blob_instr_transaction, instrBuffer);
    if (elfData && !elfData->empty()) {
      if (succeeded(writeElfFile(outputElfPath, *elfData))) {
        if (verbose) {
          llvm::outs() << "Generated ELF via library: " << outputElfPath
                       << "\n";
        }
        return success();
      }
    }
    // Fall through to subprocess on library failure
    if (verbose) {
      llvm::outs() << "aiebu library failed, falling back to subprocess\n";
    }
  }
#endif // AIECC_HAS_AIEBU_LIBRARY

  // Write combined instructions to temporary binary file for subprocess
  SmallString<128> tempBinPath(tmpDirName);
  sys::path::append(tempBinPath, devName.str() + "_elf_insts.bin");

  {
    std::error_code ec;
    raw_fd_ostream binFile(tempBinPath, ec, sys::fs::OpenFlags::OF_None);
    if (ec) {
      llvm::errs() << "Error creating temp binary for ELF: " << ec.message()
                   << "\n";
      return failure();
    }
    binFile.write(reinterpret_cast<const char *>(allInstructions.data()),
                  allInstructions.size() * sizeof(uint32_t));
  }

  if (verbose) {
    llvm::outs() << "Wrote " << allInstructions.size()
                 << " instructions to temp file: " << tempBinPath << "\n";
  }

  // Find aiebu-asm binary for subprocess fallback
  std::string aiebuAsmBin = findAiebuAsm();
  if (aiebuAsmBin.empty()) {
    llvm::errs() << "Error: aiebu-asm not found in PATH or at "
                    "/opt/xilinx/aiebu/bin/aiebu-asm\n";
    return failure();
  }

  // Run aiebu-asm to convert binary to ELF
  // aiebu-asm -t aie2txn -c <input.bin> -o <output.elf>
  SmallVector<std::string, 10> aiebuCmd = {
      aiebuAsmBin, "-t",         "aie2txn", "-c", std::string(tempBinPath),
      "-o",        outputElfPath};

  if (!executeCommand(aiebuCmd)) {
    llvm::errs() << "Error running aiebu-asm\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Generated ELF: " << outputElfPath << "\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Full ELF Generation (via aiebu-asm aie2_config)
//===----------------------------------------------------------------------===//

/// Structure to hold device info for full ELF generation.
struct DeviceElfInfo {
  std::string deviceName;
  std::string pdiPath;
  std::vector<std::pair<std::string, std::string>>
      sequences; // (seqName, instsPath)
  int pdiId;
};

/// Generate full ELF containing PDIs and instruction sequences.
/// This creates a config.json and calls aiebu-asm -t aie2_config.
static LogicalResult
generateFullElfArtifact(ArrayRef<DeviceElfInfo> deviceInfos,
                        StringRef tmpDirName) {
  if (!generateFullElf) {
    return success();
  }

  if (deviceInfos.empty()) {
    return success();
  }

  if (verbose) {
    llvm::outs() << "Generating full ELF with " << deviceInfos.size()
                 << " device(s)\n";
  }

  if (dryRun) {
    if (verbose) {
      llvm::outs() << "Would generate full ELF\n";
    }
    return success();
  }

  // Build config.json structure using LLVM JSON API for proper escaping
  // Format: { "xrt-kernels": [ { "name": ..., "PDIs": [...], "instance": [...]
  // } ] }
  llvm::json::Array xrtKernels;

  for (const auto &info : deviceInfos) {
    llvm::json::Object kernel;
    kernel["name"] = info.deviceName;

    // Arguments - determine max arg count from sequences
    // For now, use a default set of arguments
    llvm::json::Array arguments;
    llvm::json::Object arg0;
    arg0["name"] = "arg_0";
    arg0["type"] = "char *";
    arg0["offset"] = "0x0";
    arguments.push_back(std::move(arg0));

    llvm::json::Object arg1;
    arg1["name"] = "arg_1";
    arg1["type"] = "char *";
    arg1["offset"] = "0x8";
    arguments.push_back(std::move(arg1));

    llvm::json::Object arg2;
    arg2["name"] = "arg_2";
    arg2["type"] = "char *";
    arg2["offset"] = "0x10";
    arguments.push_back(std::move(arg2));

    kernel["arguments"] = std::move(arguments);

    // PDIs
    llvm::json::Array pdis;
    llvm::json::Object pdi;
    pdi["id"] = info.pdiId;
    pdi["PDI_file"] = info.pdiPath;
    pdis.push_back(std::move(pdi));
    kernel["PDIs"] = std::move(pdis);

    // Instances (runtime sequences)
    llvm::json::Array instances;
    for (const auto &seq : info.sequences) {
      llvm::json::Object instance;
      instance["id"] = seq.first;
      instance["TXN_ctrl_code_file"] = seq.second;
      instances.push_back(std::move(instance));
    }
    kernel["instance"] = std::move(instances);

    xrtKernels.push_back(std::move(kernel));
  }

  llvm::json::Object root;
  root["xrt-kernels"] = std::move(xrtKernels);

  // Write config.json
  SmallString<128> configPath(tmpDirName);
  sys::path::append(configPath, "full_elf_config.json");

  {
    std::error_code ec;
    raw_fd_ostream configFile(configPath, ec);
    if (ec) {
      llvm::errs() << "Error writing config.json: " << ec.message() << "\n";
      return failure();
    }
    configFile << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
  }

  if (verbose) {
    llvm::outs() << "Generated config.json: " << configPath << "\n";
    llvm::outs().flush();
  }

#ifdef AIECC_HAS_AIEBU_LIBRARY
  // Try using aiebu library directly
  {
    // Read the config JSON file we just wrote
    auto configData = readBinaryFile(configPath);
    if (configData && !configData->empty()) {
      if (verbose) {
        llvm::outs() << "Using aiebu library for full ELF generation\n";
      }

      auto elfData = generateElfViaAiebuLibraryConfig(*configData);
      if (elfData && !elfData->empty()) {
        if (succeeded(writeElfFile(fullElfName.getValue(), *elfData))) {
          if (verbose) {
            llvm::outs() << "Generated full ELF via library: " << fullElfName
                         << "\n";
            llvm::outs().flush();
          }
          return success();
        }
      }
      // Fall through to subprocess on library failure
      if (verbose) {
        llvm::outs() << "aiebu library failed for full ELF, falling back\n";
      }
    }
  }
#endif // AIECC_HAS_AIEBU_LIBRARY

  // Subprocess fallback: find aiebu-asm
  std::string aiebuAsmBin = findAiebuAsm();
  if (aiebuAsmBin.empty()) {
    llvm::errs() << "Error: aiebu-asm not found for full ELF generation\n";
    return failure();
  }

  // Run aiebu-asm -t aie2_config -j config.json -o output.elf
  SmallVector<std::string, 10> aiebuCmd = {aiebuAsmBin,
                                           "-t",
                                           "aie2_config",
                                           "-j",
                                           std::string(configPath),
                                           "-o",
                                           fullElfName.getValue()};

  if (!executeCommand(aiebuCmd)) {
    llvm::errs() << "Error running aiebu-asm for full ELF\n";
    llvm::errs().flush();
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Generated full ELF: " << fullElfName << "\n";
    llvm::outs().flush();
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
  // Full ELF requires PDI generation
  bool needPdi = generatePdi || generateFullElf;
  if (!generateCdo && !needPdi && !generateXclbin) {
    return success();
  }

  if (verbose) {
    llvm::outs() << "Generating CDO artifacts for device: " << devName << "\n";
  }

  // Generate JSON metadata files for xclbin even in dry-run mode
  // (matching Python aiecc.py behavior where -n only skips command execution)
  if (generateXclbin) {
    SmallString<128> memTopoPath(tmpDirName);
    sys::path::append(memTopoPath, devName.str() + "_mem_topology.json");
    if (failed(generateMemTopologyJson(memTopoPath)))
      return failure();

    SmallString<128> kernelsPath(tmpDirName);
    sys::path::append(kernelsPath, devName.str() + "_kernels.json");
    if (failed(generateKernelsJson(kernelsPath, devName)))
      return failure();

    // Generate partition JSON (with placeholder PDI path in dry-run)
    SmallString<128> partitionPath(tmpDirName);
    sys::path::append(partitionPath, devName.str() + "_aie_partition.json");
    std::string pdiFileName = formatString(pdiName, devName);
    SmallString<128> pdiPath(tmpDirName);
    sys::path::append(pdiPath, pdiFileName);
    // Make pdiPath absolute for partition JSON
    SmallString<128> absPdiPath;
    if (sys::path::is_absolute(pdiPath)) {
      absPdiPath = pdiPath;
    } else {
      // In dry-run mode, just construct the path manually
      sys::fs::current_path(absPdiPath);
      sys::path::append(absPdiPath, pdiPath);
    }
    if (failed(generatePartitionJson(partitionPath, devName, absPdiPath)))
      return failure();

    if (verbose) {
      llvm::outs() << "Generated JSON metadata files in: " << tmpDirName
                   << "\n";
    }
  }

  // In dry-run mode, skip actual command execution
  if (dryRun) {
    if (verbose) {
      llvm::outs() << "Dry-run: would generate CDO/PDI/xclbin artifacts for "
                      "device: "
                   << devName << "\n";
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

  // Generate PDI if requested (also required for full ELF)
  if (needPdi || generateXclbin) {
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

    bool pdiGenerated = false;

#ifdef AIECC_HAS_BOOTGEN_LIBRARY
    // Try using bootgen library directly via C API
    if (verbose) {
      llvm::outs() << "Using bootgen library for PDI generation\n";
    }
    char errorMsg[256] = {0};
    int result = bootgen_generate_pdi(
        std::string(bifPath).c_str(), std::string(pdiPath).c_str(),
        BOOTGEN_ARCH_VERSAL, /*overwrite=*/1, errorMsg, sizeof(errorMsg));
    if (result == BOOTGEN_SUCCESS) {
      if (verbose) {
        llvm::outs() << "Generated PDI via library: " << pdiPath << "\n";
      }
      pdiGenerated = true;
    } else {
      if (verbose) {
        llvm::outs() << "bootgen library failed (" << errorMsg
                     << "), falling back to subprocess\n";
      }
    }
#endif // AIECC_HAS_BOOTGEN_LIBRARY

    // Subprocess fallback if library not available or failed
    if (!pdiGenerated) {
      std::string bootgenPath = findAieTool("bootgen");
      if (bootgenPath.empty()) {
        llvm::errs() << "Error: bootgen not found, cannot generate requested "
                        "PDI/xclbin\n";
        return failure();
      }

      SmallVector<std::string, 8> bootgenCmd = {bootgenPath,
                                                "-arch",
                                                "versal",
                                                "-image",
                                                std::string(bifPath),
                                                "-o",
                                                std::string(pdiPath),
                                                "-w"};

      if (!executeCommand(bootgenCmd)) {
        llvm::errs() << "Error generating PDI\n";
        return failure();
      }

      if (verbose) {
        llvm::outs() << "Generated PDI: " << pdiPath << "\n";
      }
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

      // JSON metadata files were already generated earlier (even in dry-run)
      SmallString<128> memTopoPath(tmpDirName);
      sys::path::append(memTopoPath, devName.str() + "_mem_topology.json");

      SmallString<128> kernelsPath(tmpDirName);
      sys::path::append(kernelsPath, devName.str() + "_kernels.json");

      SmallString<128> partitionPath(tmpDirName);
      sys::path::append(partitionPath, devName.str() + "_aie_partition.json");

      // Build xclbin
      std::string xclbinFileName = formatString(xclbinName, devName);
      SmallString<128> xclbinPath;
      if (sys::path::is_absolute(xclbinFileName)) {
        xclbinPath = xclbinFileName;
      } else {
        xclbinPath = xclbinFileName;
      }

      SmallVector<std::string, 16> xclbinCmd;

      // Handle xclbin-input: merge with existing xclbin
      if (!xclbinInput.empty()) {
        if (verbose) {
          llvm::outs() << "Extending existing xclbin: " << xclbinInput << "\n";
        }

        // Merge partition JSONs
        SmallString<128> mergedPartitionPath(tmpDirName);
        sys::path::append(mergedPartitionPath,
                          devName.str() + "_merged_partition.json");

        if (failed(extractAndMergePartition(xclbinInput, partitionPath,
                                            mergedPartitionPath, tmpDirName))) {
          return failure();
        }

        xclbinCmd = {xclbinutilPath,
                     "--input",
                     xclbinInput.getValue(),
                     "--add-kernel",
                     std::string(kernelsPath),
                     "--add-replace-section",
                     "AIE_PARTITION:JSON:" + std::string(mergedPartitionPath),
                     "--force",
                     "--output",
                     std::string(xclbinPath)};
      } else {
        // Create new xclbin from scratch
        xclbinCmd = {xclbinutilPath,
                     "--add-replace-section",
                     "MEM_TOPOLOGY:JSON:" + std::string(memTopoPath),
                     "--add-kernel",
                     std::string(kernelsPath),
                     "--add-replace-section",
                     "AIE_PARTITION:JSON:" + std::string(partitionPath),
                     "--force",
                     "--output",
                     std::string(xclbinPath)};
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
// AIE Simulation and Host Compilation (delegated to aiecc_aiesim.cpp)
//===----------------------------------------------------------------------===//

/// Create AiesimConfig from current command-line options
static xilinx::aiecc::AiesimConfig createAiesimConfig() {
  xilinx::aiecc::AiesimConfig config;
  config.enabled = aiesim && !noAiesim;
  config.compileHost = compileHost && !noCompileHost;
  config.verbose = verbose;
  config.dryRun = dryRun;
  config.hostTarget = hostTarget.getValue();
  config.aietoolsPath = getAietoolsDir();
  config.installPath = getInstallPath();
  return config;
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
  // Collect device info for full ELF generation if requested
  std::vector<DeviceElfInfo> deviceElfInfos;
  int devicePdiId = 1; // Start PDI IDs from 1

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

    // Generate transaction MLIR output if requested
    if (failed(generateTransactionOutput(moduleOp, tmpDirName, devName))) {
      return failure();
    }

    // Generate control packet output if requested
    if (failed(generateControlPacketOutput(moduleOp, tmpDirName, devName))) {
      return failure();
    }

    // Generate ELF from NPU instructions (via aiebu-asm)
    if (failed(generateElfFromInsts(moduleOp, tmpDirName, devName))) {
      return failure();
    }

    // Generate CDO/PDI/xclbin from in-memory module
    if (failed(generateCdoArtifacts(moduleOp, tmpDirName, devName))) {
      return failure();
    }

    // Generate aie_inc.cpp and aiesim work folder if requested
    auto aiesimConfig = createAiesimConfig();
    if (failed(xilinx::aiecc::generateAieIncCpp(moduleOp, tmpDirName, devName,
                                                aiesimConfig))) {
      return failure();
    }
    if (failed(xilinx::aiecc::generateAiesim(moduleOp, tmpDirName, devName,
                                             aieTarget, aiesimConfig))) {
      return failure();
    }

    // Collect info for full ELF generation
    if (generateFullElf) {
      DeviceElfInfo info;
      info.deviceName = devName.str();
      info.pdiId = devicePdiId++;

      // Get absolute path to tmpDir for aiebu-asm (it needs absolute paths)
      SmallString<256> absTmpDir;
      if (auto ec = sys::fs::real_path(tmpDirName, absTmpDir)) {
        // Fall back to current dir + tmpDirName
        sys::fs::current_path(absTmpDir);
        sys::path::append(absTmpDir, tmpDirName);
      }

      // PDI path (must match generateCdoArtifacts output)
      std::string pdiFileName = formatString(pdiName, devName);
      SmallString<256> pdiFullPath(absTmpDir);
      sys::path::append(pdiFullPath, pdiFileName);
      info.pdiPath = std::string(pdiFullPath);

      // Collect runtime sequence instruction paths (also absolute)
      for (auto seqOp : deviceOp.getOps<xilinx::AIE::RuntimeSequenceOp>()) {
        StringRef seqName = seqOp.getSymName();
        std::string instsFileName =
            formatString(instsName, devName.str(), seqName);
        SmallString<256> instsFullPath(absTmpDir);
        sys::path::append(instsFullPath, instsFileName);
        info.sequences.emplace_back(seqName.str(), std::string(instsFullPath));
      }

      deviceElfInfos.push_back(std::move(info));
    }
  }

  // Generate full ELF after all devices are processed
  if (failed(generateFullElfArtifact(deviceElfInfos, tmpDirName))) {
    return failure();
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

  // Set up diagnostic handler to print all diagnostics (including warnings)
  // This ensures that pass diagnostics like buffer allocation warnings are
  // visible to the user. The handler will print to stderr by default.
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

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
  if (noAiesim) {
    aiesim = false;
  }
  if (noCompileHost) {
    compileHost = false;
  }

  // Validate: aiesim requires xbridge
  if (aiesim && !xbridge) {
    llvm::errs()
        << "Error: AIE Simulation (--aiesim) currently requires --xbridge\n";
    return 1;
  }

  // Process the input file
  return processInputFile(inputFilename, tmpDir);
}
