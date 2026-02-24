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
#include "mlir/Support/ToolUtilities.h"

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

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "aiecc_aiesim.h"

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
                              cl::desc("Disable AIE simulation mode (default)"),
                              cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> compile("compile",
                             cl::desc("Enable compiling of AIE cores"),
                             cl::init(true), cl::cat(aieCompilerOptions));

static cl::opt<bool> noCompile("no-compile",
                               cl::desc("Disable compiling of AIE cores"),
                               cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> compileHost(
    "compile-host",
    cl::desc("Enable host compilation (not supported in C++ aiecc)"),
    cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> noCompileHost(
    "no-compile-host",
    cl::desc("Disable host compilation (host compilation not supported)"),
    cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    hostTarget("host-target",
               cl::desc("Host target architecture (deprecated, ignored)"),
               cl::init(""), cl::cat(aieCompilerOptions));

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

static cl::opt<unsigned> numThreads(
    "j", cl::Prefix,
    cl::desc("Number of parallel threads for core compilation (0 = auto-detect "
             "based on CPU count, default: 1 for sequential)"),
    cl::init(1), cl::cat(aieCompilerOptions));

static cl::alias numThreadsLong("nthreads", cl::desc("Alias for -j"),
                                cl::aliasopt(numThreads),
                                cl::cat(aieCompilerOptions));

static cl::opt<bool> unified(
    "unified",
    cl::desc("Compile all cores together in a single process (default: off)"),
    cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> noUnified(
    "no-unified",
    cl::desc("Compile cores independently in separate processes (default)"),
    cl::init(false), cl::cat(aieCompilerOptions));

// Backward compatibility flags (no-ops in C++ aiecc)
// These flags existed in Python aiecc.py but are not used in the core
// compilation flow. They are kept for backward compatibility so that
// existing scripts and test cases don't fail with "unknown argument" errors.

// Note: --vectorize was removed - it was defined but never used in Python
// aiecc.py

static cl::opt<bool> profile("profile",
                             cl::desc("Enable profiling (deprecated, ignored)"),
                             cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool>
    progress("progress", cl::desc("Show progress output (deprecated, ignored)"),
             cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> enableRepeaterScripts(
    "enable-repeater-scripts",
    cl::desc("Enable repeater scripts (deprecated, ignored)"), cl::init(false),
    cl::cat(aieCompilerOptions));

static cl::opt<bool> disableRepeaterScripts(
    "disable-repeater-scripts",
    cl::desc("Disable repeater scripts (deprecated, ignored)"), cl::init(false),
    cl::cat(aieCompilerOptions));

static cl::opt<std::string> repeaterOutputDir(
    "repeater-output-dir",
    cl::desc("Repeater output directory (deprecated, ignored)"), cl::init(""),
    cl::cat(aieCompilerOptions));

static cl::opt<bool> linkAgainstHsa(
    "link_against_hsa",
    cl::desc("Link against HSA runtime (-lhsa-runtime64 from ROCm)"),
    cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<bool> noMaterialize(
    "no-materialize",
    cl::desc("Skip aie-materialize-runtime-sequences pass in NPU lowering"),
    cl::init(false), cl::cat(aieCompilerOptions));

//===----------------------------------------------------------------------===//
// Thread-safe output
//===----------------------------------------------------------------------===//

// Global mutex for thread-safe verbose output during parallel compilation
static std::mutex outputMutex;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Forward declarations
static std::string findAiebuAsm();

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
    // Print command without prefix to match Python aiecc.py output format
    // (tests check for command patterns like "llc" or "xchesscc_wrapper" at
    // line start)
    bool first = true;
    for (const auto &arg : command) {
      if (!first)
        llvm::outs() << " ";
      llvm::outs() << arg;
      first = false;
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

  // Add materialize runtime sequences pass at module level (before device
  // nesting) unless --no-materialize is specified
  if (!noMaterialize) {
    pm.addPass(xilinx::AIEX::createAIEMaterializeRuntimeSequencesPass());
  }

  // Device-level passes
  OpPassManager &devicePm = pm.nest<xilinx::AIE::DeviceOp>();
  devicePm.addPass(xilinx::AIEX::createAIEMaterializeBDChainsPass());
  devicePm.addPass(xilinx::AIEX::createAIESubstituteShimDMAAllocationsPass());
  devicePm.addPass(xilinx::AIEX::createAIEAssignRuntimeSequenceBDIDsPass());
  devicePm.addPass(xilinx::AIEX::createAIEDMATasksToNPUPass());
  devicePm.addPass(xilinx::AIEX::createAIEDmaToNpuPass());
  devicePm.addPass(xilinx::AIEX::createAIELowerSetLockPass());

  // Add expand-load-pdi pass at module level (after device nesting)
  // if --expand-load-pdis is specified
  if (expandLoadPdis) {
    pm.addPass(xilinx::AIEX::createAIEExpandLoadPdiPass());
  }

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
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::outs() << "Running LLVM lowering pipeline in-memory for core (" << col
                 << ", " << row << ")\n";
  }

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Error: LLVM lowering pipeline failed for core (" << col
                 << ", " << row << ")\n";
    return failure();
  }

  if (verbose) {
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::outs() << "LLVM lowering pipeline completed successfully for core ("
                 << col << ", " << row << ")\n";
  }

  return success();
}

/// Run the LLVM lowering pipeline for unified compilation (all cores together).
/// Unlike runLLVMLoweringPipeline, this does not filter to a specific core.
/// All cores are lowered together into a single module.
///
/// NOTE: This function modifies moduleOp destructively. Caller should pass a
/// cloned module.
static LogicalResult runUnifiedLLVMLoweringPipeline(ModuleOp moduleOp,
                                                    StringRef deviceName,
                                                    StringRef aieTarget) {
  MLIRContext *ctx = moduleOp.getContext();
  PassManager pm(ctx);

  if (verbose) {
    pm.enableVerifier(true);
  }

  // Step 1: Nested passes inside aie.device
  OpPassManager &devicePm = pm.nest<xilinx::AIE::DeviceOp>();
  devicePm.addPass(xilinx::AIE::createAIELocalizeLocksPass());
  devicePm.addPass(xilinx::AIE::createAIENormalizeAddressSpacesPass());

  // Step 2: aie-standard-lowering WITHOUT specific core coordinates
  // This exports ALL cores by using default values (-1, -1)
  xilinx::AIE::AIECoreToStandardOptions coreOpts;
  coreOpts.deviceName = deviceName.str();
  // tileCol and tileRow default to -1 (process all cores)
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
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  if (verbose) {
    llvm::outs() << "Running unified LLVM lowering pipeline in-memory for all "
                    "cores\n";
  }

  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Error: Unified LLVM lowering pipeline failed\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Unified LLVM lowering pipeline completed successfully\n";
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

  // When --no-unified is explicitly set with xchesscc, compile and link are
  // combined into one step. If --no-link is also set, skip compilation entirely
  // since xchesscc cannot produce object files without linking in this mode.
  // Note: This optimization only applies when --no-unified is explicitly
  // passed, not when unified is just false by default.
  if (noUnified && xchesscc && !link) {
    if (verbose) {
      std::lock_guard<std::mutex> lock(outputMutex);
      llvm::outs() << "Skipping core (" << core.col << ", " << core.row
                   << ") compilation: xchesscc requires linking\n";
    }
    return success();
  }

  if (verbose) {
    std::lock_guard<std::mutex> lock(outputMutex);
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
      std::lock_guard<std::mutex> lock(outputMutex);
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
      std::lock_guard<std::mutex> lock(outputMutex);
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
      std::lock_guard<std::mutex> lock(outputMutex);
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
      std::lock_guard<std::mutex> lock(outputMutex);
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
      std::lock_guard<std::mutex> lock(outputMutex);
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
      std::lock_guard<std::mutex> lock(outputMutex);
      llvm::outs() << "Generated BCF: " << bcfPath << "\n";
    }

    // Extract link_with files from BCF
    std::vector<std::string> linkWithFiles = extractInputFilesFromBCF(bcfPath);

    // Handle link_with files: copy to .prj directory if needed
    // Search order: current working directory, then input file directory
    std::string linkWithArgs;
    for (const auto &linkWithFile : linkWithFiles) {
      SmallString<256> srcPath;
      if (sys::path::is_absolute(linkWithFile)) {
        srcPath = linkWithFile;
      } else {
        // First try current working directory
        SmallString<256> cwdPath;
        sys::fs::current_path(cwdPath);
        sys::path::append(cwdPath, linkWithFile);
        if (sys::fs::exists(cwdPath)) {
          srcPath = cwdPath;
        } else {
          // Fall back to input file directory
          SmallString<256> inputDir = sys::path::parent_path(inputFilename);
          if (inputDir.empty()) {
            sys::fs::current_path(inputDir);
          }
          srcPath = inputDir;
          sys::path::append(srcPath, linkWithFile);
          sys::path::remove_dots(srcPath, /*remove_dot_dot=*/true);
        }
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
        std::lock_guard<std::mutex> lock(outputMutex);
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
    SmallVector<std::string, 20> linkCmd = {
        *xchessccWrapperPath, aieTargetLower, "+w",
        workDir.str().str(),  "-d",           "-f",
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
      std::lock_guard<std::mutex> lock(outputMutex);
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
      // Resolve the link_with path - check multiple locations:
      // 1. If absolute, use as-is
      // 2. Relative to current working directory (common for test cases)
      // 3. Relative to input file directory (common for installed examples)
      SmallString<256> srcLinkWith;
      if (sys::path::is_absolute(core.linkWith)) {
        srcLinkWith = core.linkWith;
      } else {
        // First try current working directory
        SmallString<256> cwdPath;
        sys::fs::current_path(cwdPath);
        sys::path::append(cwdPath, core.linkWith);
        if (sys::fs::exists(cwdPath)) {
          srcLinkWith = cwdPath;
        } else {
          // Fall back to input file directory
          SmallString<256> inputDir = sys::path::parent_path(inputFilename);
          if (inputDir.empty()) {
            sys::fs::current_path(inputDir);
          }
          srcLinkWith = inputDir;
          sys::path::append(srcLinkWith, core.linkWith);
          sys::path::remove_dots(srcLinkWith, /*remove_dot_dot=*/true);
        }
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
        std::lock_guard<std::mutex> lock(outputMutex);
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

    // Add HSA runtime linking if requested
    if (linkAgainstHsa) {
      if (!sysroot.empty()) {
        SmallString<256> rocmLibPath(sysroot);
        sys::path::append(rocmLibPath, "opt", "rocm", "lib");
        linkCmd.push_back("-L" + rocmLibPath.str().str());
      } else {
        linkCmd.push_back("-L/opt/rocm/lib");
      }
      linkCmd.push_back("-lhsa-runtime64");
    }

    linkCmd.push_back("-o");
    linkCmd.push_back(elfPath.str().str());

    if (!executeCommand(linkCmd)) {
      llvm::errs() << "Error linking ELF file\n";
      return failure();
    }
  }

  outElfPath = elfPath.str().str();
  if (verbose) {
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::outs() << "Generated ELF: " << outElfPath << "\n";
  }

  return success();
}

/// Compile all cores in a device, optionally in parallel.
/// When numThreads > 1, cores are compiled in parallel using std::async.
/// Each parallel task gets its own MLIRContext to avoid threading issues.
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

  // Determine number of threads to use
  unsigned nThreads = numThreads;
  if (nThreads == 0) {
    nThreads = std::thread::hardware_concurrency();
    if (nThreads == 0) {
      nThreads = 4; // Reasonable default if hardware_concurrency() fails
    }
  }

  // Cap threads at number of cores
  nThreads = std::min(nThreads, static_cast<unsigned>(cores.size()));

  if (verbose) {
    if (nThreads > 1) {
      llvm::outs() << "Compiling " << cores.size() << " core(s) in parallel ("
                   << nThreads << " threads)\n";
    } else {
      llvm::outs() << "Compiling " << cores.size() << " core(s) sequentially\n";
    }
  }

  // Sequential compilation (default, nThreads == 1)
  if (nThreads <= 1) {
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

  // Parallel compilation using std::async
  // We need to serialize the module to string and deserialize in each thread
  // because MLIRContext is not thread-safe for multi-threaded mutation.

  // First, serialize the module to string for parallel workers
  std::string moduleStr;
  {
    llvm::raw_string_ostream os(moduleStr);
    moduleOp->print(os);
  }

  // Get the dialect registry to use for parallel contexts
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  xilinx::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  // Thread-safe results storage
  std::mutex resultsMutex;
  std::atomic<bool> hasFailure{false};

  // Create futures for parallel tasks
  std::vector<std::future<void>> futures;
  futures.reserve(cores.size());

  // Use a semaphore-like pattern with atomic counter for thread limiting
  std::atomic<unsigned> activeThreads{0};

  for (const auto &core : cores) {
    // Wait if we've reached the thread limit
    while (activeThreads.load() >= nThreads) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Check if any thread has failed
    if (hasFailure.load()) {
      break;
    }

    activeThreads++;

    // Create task for this core
    futures.push_back(std::async(
        std::launch::async,
        [&registry, &moduleStr, deviceName = deviceName.str(),
         tmpDirName = tmpDirName.str(), aieTarget = aieTarget.str(), core,
         &resultsMutex, &elfPaths, &hasFailure, &activeThreads]() {
          // Each thread creates its own MLIRContext
          MLIRContext threadContext;
          threadContext.appendDialectRegistry(registry);
          threadContext.loadAllAvailableDialects();

          // Register LLVM IR translation dialects
          mlir::registerBuiltinDialectTranslation(threadContext);
          mlir::registerLLVMDialectTranslation(threadContext);

          // Parse the module in this thread's context
          ParserConfig parseConfig(&threadContext);
          OwningOpRef<ModuleOp> threadModule =
              parseSourceString<ModuleOp>(moduleStr, parseConfig);

          if (!threadModule) {
            llvm::errs() << "Error: Failed to parse module for core ("
                         << core.col << ", " << core.row << ")\n";
            hasFailure.store(true);
            activeThreads--;
            return;
          }

          // Compile the core
          std::string elfPath;
          if (failed(compileCore(threadContext, *threadModule, deviceName, core,
                                 tmpDirName, aieTarget, elfPath))) {
            hasFailure.store(true);
            activeThreads--;
            return;
          }

          // Store result
          if (!elfPath.empty()) {
            std::lock_guard<std::mutex> lock(resultsMutex);
            elfPaths[{core.col, core.row}] = elfPath;
          }

          activeThreads--;
        }));
  }

  // Wait for all tasks to complete
  for (auto &future : futures) {
    future.wait();
  }

  if (hasFailure.load()) {
    return failure();
  }

  return success();
}

/// Compile all cores using unified compilation mode.
/// In unified mode:
/// 1. All cores are lowered together to a single LLVM IR file
/// 2. That IR is compiled to a single object file
/// 3. Each core is linked separately using its own linker script/BCF
/// This can be faster than per-core compilation when cores share code.
static LogicalResult
compileCoresUnified(MLIRContext &context, ModuleOp moduleOp,
                    Operation *deviceOp, StringRef deviceName,
                    StringRef tmpDirName, StringRef aieTarget,
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

  if (!compile) {
    // If we're not compiling, just collect existing ELF paths
    for (const auto &core : cores) {
      if (!core.elfFile.empty()) {
        elfPaths[{core.col, core.row}] = core.elfFile;
      }
    }
    return success();
  }

  if (verbose) {
    llvm::outs() << "Compiling " << cores.size()
                 << " core(s) using unified compilation\n";
  }

  // Step 1: Clone module and run unified LLVM lowering pipeline
  OwningOpRef<ModuleOp> unifiedModule = moduleOp.clone();

  // Register LLVM IR translation dialects
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);

  if (failed(runUnifiedLLVMLoweringPipeline(*unifiedModule, deviceName,
                                            aieTarget))) {
    llvm::errs() << "Error: Unified LLVM lowering failed\n";
    return failure();
  }

  // Step 2: Translate to LLVM IR
  SmallString<128> llvmIRPath(tmpDirName);
  sys::path::append(llvmIRPath, deviceName.str() + "_input.ll");

  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(*unifiedModule, llvmCtx);
  if (!llvmModule) {
    llvm::errs() << "Error translating unified module to LLVM IR\n";
    return failure();
  }

  // Write LLVM IR to file
  {
    std::error_code ec;
    raw_fd_ostream llvmIRFile(llvmIRPath, ec);
    if (ec) {
      llvm::errs() << "Error opening unified LLVM IR file: " << ec.message()
                   << "\n";
      return failure();
    }
    llvmModule->print(llvmIRFile, nullptr);
    if (verbose) {
      llvm::outs() << "Generated unified LLVM IR: " << llvmIRPath << "\n";
    }
  }

  // Step 3: Compile to single object file
  SmallString<128> objPath(tmpDirName);
  sys::path::append(objPath, deviceName.str() + "_input.o");

  if (xchesscc) {
    // xchesscc compilation
    auto bufOrErr = MemoryBuffer::getFile(llvmIRPath);
    if (!bufOrErr) {
      llvm::errs() << "Error reading unified LLVM IR: "
                   << bufOrErr.getError().message() << "\n";
      return failure();
    }
    std::string downgradedIR = downgradeIRForChess((*bufOrErr)->getBuffer());

    SmallString<128> chessHackPath(tmpDirName);
    sys::path::append(chessHackPath, deviceName.str() + "_input.chesshack.ll");
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

    SmallString<128> chessLinkedPath(tmpDirName);
    sys::path::append(chessLinkedPath,
                      deviceName.str() + "_input.chesslinked.ll");

    std::string linkedResult =
        runChessLlvmLink(chessHackPath, chessLinkedPath, aieTarget, tmpDirName);
    if (linkedResult.empty()) {
      return failure();
    }

    auto xchessccWrapperPath = sys::findProgramByName("xchesscc_wrapper");
    if (!xchessccWrapperPath) {
      llvm::errs() << "Error: Could not find xchesscc_wrapper in PATH\n";
      return failure();
    }

    SmallString<128> workDir(tmpDirName);
    sys::path::append(workDir, "work");

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
      llvm::errs()
          << "Error running xchesscc_wrapper for unified compilation\n";
      return failure();
    }

    if (verbose) {
      llvm::outs() << "Compiled unified object with xchesscc: " << objPath
                   << "\n";
    }
  } else {
    // Peano compilation
    std::string peanoOpt = findPeanoTool("opt");
    std::string peanoLlc = findPeanoTool("llc");

    if (peanoOpt.empty() || peanoLlc.empty()) {
      llvm::errs() << "Error: Could not find Peano compiler tools\n";
      return failure();
    }

    // Apply peanohack (strip debug info for compatibility)
    auto bufOrErr = MemoryBuffer::getFile(llvmIRPath);
    if (!bufOrErr) {
      llvm::errs() << "Error reading unified LLVM IR: "
                   << bufOrErr.getError().message() << "\n";
      return failure();
    }

    // Write peanohacked version
    SmallString<128> peanohackPath(tmpDirName);
    sys::path::append(peanohackPath,
                      deviceName.str() + "_input.llpeanohack.ll");
    {
      std::error_code ec;
      raw_fd_ostream peanohackFile(peanohackPath, ec);
      if (ec) {
        llvm::errs() << "Error writing peanohack file: " << ec.message()
                     << "\n";
        return failure();
      }
      // Simple peanohack: just copy for now (could add attribute stripping)
      peanohackFile << (*bufOrErr)->getBuffer();
    }

    // Run opt
    SmallString<128> optPath(tmpDirName);
    sys::path::append(optPath, deviceName.str() + "_input.opt.ll");

    std::string optLevelStr = std::to_string(optLevel);
    SmallVector<std::string, 12> optCmd = {peanoOpt,
                                           "--passes=default<O" + optLevelStr +
                                               ">,strip",
                                           "-inline-threshold=10",
                                           "-S",
                                           peanohackPath.str().str(),
                                           "-o",
                                           optPath.str().str()};

    if (optLevel >= 3) {
      optCmd.insert(optCmd.begin() + 1, "-disable-loop-idiom-memset");
    }

    if (!executeCommand(optCmd)) {
      llvm::errs() << "Error running Peano opt for unified compilation\n";
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
      llvm::errs() << "Error running Peano llc for unified compilation\n";
      return failure();
    }

    if (verbose) {
      llvm::outs() << "Compiled unified object with Peano: " << objPath << "\n";
    }
  }

  // Step 4: Link each core separately using the shared object file
  if (!link) {
    // If not linking, just return the object path for all cores
    for (const auto &core : cores) {
      elfPaths[{core.col, core.row}] = objPath.str().str();
    }
    return success();
  }

  for (const auto &core : cores) {
    if (verbose) {
      llvm::outs() << "Linking core (" << core.col << ", " << core.row
                   << ") from unified object\n";
    }

    SmallString<128> elfPath(tmpDirName);
    sys::path::append(elfPath, deviceName.str() + "_core_" +
                                   std::to_string(core.col) + "_" +
                                   std::to_string(core.row) + ".elf");

    // Make ELF path absolute
    SmallString<256> absElfPath;
    std::error_code ec = sys::fs::real_path(elfPath, absElfPath);
    if (ec) {
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
      // xbridge linking with BCF
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
          llvm::errs() << "Error generating BCF file for core (" << core.col
                       << ", " << core.row << ")\n";
          return failure();
        }
      }

      std::vector<std::string> linkWithFiles =
          extractInputFilesFromBCF(bcfPath);

      // Copy link_with files to .prj directory
      // Search order: current working directory, then input file directory
      for (const auto &linkWithFile : linkWithFiles) {
        SmallString<256> srcPath;
        if (sys::path::is_absolute(linkWithFile)) {
          srcPath = linkWithFile;
        } else {
          // First try current working directory
          SmallString<256> cwdPath;
          sys::fs::current_path(cwdPath);
          sys::path::append(cwdPath, linkWithFile);
          if (sys::fs::exists(cwdPath)) {
            srcPath = cwdPath;
          } else {
            // Fall back to input file directory
            SmallString<256> inputDir = sys::path::parent_path(inputFilename);
            if (inputDir.empty()) {
              sys::fs::current_path(inputDir);
            }
            srcPath = inputDir;
            sys::path::append(srcPath, linkWithFile);
            sys::path::remove_dots(srcPath, /*remove_dot_dot=*/true);
          }
        }

        SmallString<256> destPath(tmpDirName);
        sys::path::append(destPath, sys::path::filename(linkWithFile));
        sys::fs::remove(destPath);
        std::error_code ec = sys::fs::copy_file(srcPath, destPath);
        if (ec) {
          llvm::errs() << "Error copying link_with file: " << srcPath << "\n";
          return failure();
        }
      }

      auto xchessccWrapperPath = sys::findProgramByName("xchesscc_wrapper");
      if (!xchessccWrapperPath) {
        llvm::errs() << "Error: Could not find xchesscc_wrapper for xbridge\n";
        return failure();
      }

      SmallString<128> workDir(tmpDirName);
      sys::path::append(workDir, "work");

      std::string aieTargetLower = aieTarget.lower();
      SmallVector<std::string, 20> linkCmd = {
          *xchessccWrapperPath, aieTargetLower, "+w",
          workDir.str().str(),  "-d",           "-f",
          objPath.str().str()};

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
        llvm::errs() << "Error linking core (" << core.col << ", " << core.row
                     << ") with xbridge\n";
        return failure();
      }
    } else {
      // Peano linking with linker script
      SmallString<128> ldScriptPath(tmpDirName);
      sys::path::append(ldScriptPath,
                        deviceName.str() + "_core_" + std::to_string(core.col) +
                            "_" + std::to_string(core.row) + ".ld.script");

      {
        std::error_code ec;
        raw_fd_ostream ldScriptFile(ldScriptPath, ec);
        if (ec) {
          llvm::errs() << "Error opening linker script file: " << ec.message()
                       << "\n";
          return failure();
        }

        if (failed(xilinx::AIE::AIETranslateToLdScript(
                moduleOp, ldScriptFile, core.col, core.row, deviceName))) {
          llvm::errs() << "Error generating linker script for core ("
                       << core.col << ", " << core.row << ")\n";
          return failure();
        }
      }

      std::string peanoClang = findPeanoTool("clang");
      if (peanoClang.empty()) {
        llvm::errs() << "Error: Could not find Peano clang\n";
        return failure();
      }

      std::string peanoTarget = getPeanoTarget(aieTarget);
      std::string optLevelStr = std::to_string(optLevel);

      StringRef peanoDir = getPeanoInstallDir();
      SmallString<128> peanoBinDir;
      if (!peanoDir.empty()) {
        peanoBinDir = peanoDir;
        sys::path::append(peanoBinDir, "bin");
      } else {
        peanoBinDir = sys::path::parent_path(peanoClang);
      }

      SmallString<256> peanoLld(peanoBinDir);
      sys::path::append(peanoLld, "ld.lld");

      // Handle link_with if specified
      // Search order: current working directory, then input file directory
      if (!core.linkWith.empty()) {
        SmallString<256> srcLinkWith;
        if (sys::path::is_absolute(core.linkWith)) {
          srcLinkWith = core.linkWith;
        } else {
          // First try current working directory
          SmallString<256> cwdPath;
          sys::fs::current_path(cwdPath);
          sys::path::append(cwdPath, core.linkWith);
          if (sys::fs::exists(cwdPath)) {
            srcLinkWith = cwdPath;
          } else {
            // Fall back to input file directory
            SmallString<256> inputDir = sys::path::parent_path(inputFilename);
            if (inputDir.empty()) {
              sys::fs::current_path(inputDir);
            }
            srcLinkWith = inputDir;
            sys::path::append(srcLinkWith, core.linkWith);
            sys::path::remove_dots(srcLinkWith, /*remove_dot_dot=*/true);
          }
        }

        SmallString<256> destLinkWith(tmpDirName);
        sys::path::append(destLinkWith, sys::path::filename(core.linkWith));
        sys::fs::remove(destLinkWith);
        std::error_code ec = sys::fs::copy_file(srcLinkWith, destLinkWith);
        if (ec) {
          llvm::errs() << "Error copying link_with file: " << srcLinkWith
                       << "\n";
          return failure();
        }
      }

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

      SmallVector<std::string, 20> linkCmd = {peanoClang, "-O" + optLevelStr,
                                              "--target=" + peanoTarget};

      if (sys::fs::can_execute(peanoLld)) {
        linkCmd.push_back("-fuse-ld=" + peanoLld.str().str());
      } else {
        linkCmd.push_back("-B" + peanoBinDir.str().str());
        linkCmd.push_back("-fuse-ld=lld");
      }

      linkCmd.push_back(objPath.str().str());
      linkCmd.push_back("-Wl,--gc-sections");
      linkCmd.push_back("-Wl,--orphan-handling=error");
      linkCmd.push_back("-Wl,-T," + absLdScriptPath.str().str());

      // Add HSA runtime linking if requested
      if (linkAgainstHsa) {
        if (!sysroot.empty()) {
          SmallString<256> rocmLibPath(sysroot);
          sys::path::append(rocmLibPath, "opt", "rocm", "lib");
          linkCmd.push_back("-L" + rocmLibPath.str().str());
        } else {
          linkCmd.push_back("-L/opt/rocm/lib");
        }
        linkCmd.push_back("-lhsa-runtime64");
      }

      linkCmd.push_back("-o");
      linkCmd.push_back(elfPath.str().str());

      if (!executeCommand(linkCmd)) {
        llvm::errs() << "Error linking core (" << core.col << ", " << core.row
                     << ")\n";
        return failure();
      }
    }

    elfPaths[{core.col, core.row}] = elfPath.str().str();
    if (verbose) {
      llvm::outs() << "Generated ELF: " << elfPath << "\n";
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
  jsonFile << "        \"name\": \"" << xclbinKernelName.getValue() << "\",\n";
  jsonFile << "        \"type\": \"dpu\",\n";
  jsonFile << "        \"extended-data\": {\n";
  jsonFile << "          \"subtype\": \"DPU\",\n";
  jsonFile << "          \"functional\": \"0\",\n";
  jsonFile << "          \"dpu_kernel_id\": \"" << xclbinKernelId.getValue()
           << "\"\n";
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
  jsonFile << "            \"name\": \"" << xclbinInstanceName.getValue()
           << "\"\n";
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
  jsonFile << "            \"dpu_kernel_ids\": [\"" << xclbinKernelId.getValue()
           << "\"],\n";
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

  SmallVector<std::string, 10> extractCmd = {*xclbinutilPath,
                                             "--dump-section",
                                             "AIE_PARTITION:JSON:" +
                                                 inputPartitionPath.str().str(),
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

  // Step 3: Generate combined ELF using aiebu-asm (if available)
  std::string aiebuAsmPath = findAiebuAsm();
  if (aiebuAsmPath.empty()) {
    if (verbose) {
      llvm::outs() << "aiebu-asm not found, skipping control packet ELF "
                      "generation\n";
    }
    return success();
  }

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

  // Generate external_buffers.json for aiebu-asm
  SmallString<128> extBufJsonPath(tmpDirName);
  sys::path::append(extBufJsonPath, "external_buffers.json");

  // Get control packet file size
  uint64_t ctrlPktSize = 0;
  if (auto status = sys::fs::file_size(ctrlPktBinPath, ctrlPktSize)) {
    llvm::errs() << "Error getting control packet file size: "
                 << status.message() << "\n";
    return failure();
  }

  {
    std::error_code ec;
    raw_fd_ostream jsonFile(extBufJsonPath, ec, sys::fs::OpenFlags::OF_None);
    if (ec) {
      llvm::errs() << "Error creating external_buffers.json: " << ec.message()
                   << "\n";
      return failure();
    }
    jsonFile << "{\n";
    jsonFile << "  \"external_buffers\": {\n";
    jsonFile << "    \"buffer_ctrl\": {\n";
    jsonFile << "      \"xrt_id\": " << ctrlIdx << ",\n";
    jsonFile << "      \"logical_id\": -1,\n";
    jsonFile << "      \"size_in_bytes\": " << ctrlPktSize << ",\n";
    jsonFile << "      \"ctrl_pkt_buffer\": 1,\n";
    jsonFile << "      \"name\": \"runtime_control_packet\"\n";
    jsonFile << "    }\n";
    jsonFile << "  }\n";
    jsonFile << "}\n";
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

  // Find aiebu-asm
  std::string aiebuAsmBin = findAiebuAsm();
  if (aiebuAsmBin.empty()) {
    llvm::errs() << "Error: aiebu-asm not found in PATH or at "
                    "/opt/xilinx/aiebu/bin/aiebu-asm\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "Found aiebu-asm: " << aiebuAsmBin << "\n";
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

  // Write combined instructions to temporary binary file
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

  // Determine output ELF path
  std::string outputElfPath = formatString(elfName, devName.str(), "");

  // Run aiebu-asm to convert binary to ELF
  // aiebu-asm -t aie2txn -c <input.bin> -o <output.elf>
  SmallVector<std::string, 10> aiebuCmd = {
      aiebuAsmBin, "-t",         "aie2txn", "-c", tempBinPath.str().str(),
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

  // Find aiebu-asm
  std::string aiebuAsmBin = findAiebuAsm();
  if (aiebuAsmBin.empty()) {
    llvm::errs() << "Error: aiebu-asm not found for full ELF generation\n";
    return failure();
  }

  // Build config.json structure
  // Format: { "xrt-kernels": [ { "name": ..., "PDIs": [...], "instance": [...]
  // } ] }
  std::string configJson = "{\n  \"xrt-kernels\": [\n";

  for (size_t i = 0; i < deviceInfos.size(); ++i) {
    const auto &info = deviceInfos[i];

    if (i > 0)
      configJson += ",\n";
    configJson += "    {\n";
    configJson += "      \"name\": \"" + info.deviceName + "\",\n";

    // Arguments - determine max arg count from sequences
    // For now, use a default set of arguments
    configJson += "      \"arguments\": [\n";
    configJson += "        {\"name\": \"arg_0\", \"type\": \"char *\", "
                  "\"offset\": \"0x0\"},\n";
    configJson += "        {\"name\": \"arg_1\", \"type\": \"char *\", "
                  "\"offset\": \"0x8\"},\n";
    configJson += "        {\"name\": \"arg_2\", \"type\": \"char *\", "
                  "\"offset\": \"0x10\"}\n";
    configJson += "      ],\n";

    // PDIs
    configJson += "      \"PDIs\": [\n";
    configJson += "        {\"id\": " + std::to_string(info.pdiId) +
                  ", \"PDI_file\": \"" + info.pdiPath + "\"}\n";
    configJson += "      ],\n";

    // Instances (runtime sequences)
    configJson += "      \"instance\": [\n";
    for (size_t j = 0; j < info.sequences.size(); ++j) {
      if (j > 0)
        configJson += ",\n";
      configJson += "        {\"id\": \"" + info.sequences[j].first +
                    "\", \"TXN_ctrl_code_file\": \"" +
                    info.sequences[j].second + "\"}";
    }
    configJson += "\n      ]\n";
    configJson += "    }";
  }

  configJson += "\n  ]\n}\n";

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
    configFile << configJson;
  }

  if (verbose) {
    llvm::outs() << "Generated config.json: " << configPath << "\n";
    llvm::outs().flush();
  }

  // Run aiebu-asm -t aie2_config -j config.json -o output.elf
  SmallVector<std::string, 10> aiebuCmd = {aiebuAsmBin,
                                           "-t",
                                           "aie2_config",
                                           "-j",
                                           configPath.str().str(),
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
// AIESim Work Folder Generation (wrapper)
//===----------------------------------------------------------------------===//

/// Wrapper to call the external aiesim generation function.
/// Implementation is in aiecc_aiesim.cpp.
static LogicalResult generateAiesimWorkFolder(ModuleOp moduleOp,
                                              StringRef tmpDirName,
                                              StringRef devName,
                                              StringRef aieTarget) {
  if (!aiesim) {
    return success();
  }

  // Get aietools path (required for simulation)
  StringRef aietoolsPath = getAietoolsDir();
  if (aietoolsPath.empty()) {
    llvm::errs() << "Error: --aiesim requires aietools installation. "
                 << "Set AIETOOLS_ROOT or ensure xchesscc is in PATH.\n";
    return failure();
  }

  // Get install path for runtime libraries
  std::string installPath = getInstallPath();

  // Set up configuration
  aiecc::AiesimConfig config;
  config.enabled = aiesim;
  config.verbose = verbose;
  config.dryRun = dryRun;

  return aiecc::generateAiesimWorkFolder(moduleOp, tmpDirName, devName,
                                         aieTarget, aietoolsPath, installPath,
                                         config);
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

  // In dry-run mode, skip CDO generation (C++ API) but still print commands
  if (!dryRun) {
    // Generate CDO files using direct C++ API call.
    // This replaces the subprocess call to aie-translate --aie-generate-cdo.
    // AIETranslateToCDODirect generates CDO files directly to the work
    // directory: {devName}_aie_cdo_elfs.bin, {devName}_aie_cdo_init.bin,
    // {devName}_aie_cdo_enable.bin
    if (failed(xilinx::AIE::AIETranslateToCDODirect(moduleOp, tmpDirName,
                                                    devName,
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
  }

  // Generate PDI if requested (also required for full ELF)
  if (needPdi || generateXclbin) {
    std::string bootgenPath = findAieTool("bootgen");
    if (bootgenPath.empty()) {
      llvm::errs()
          << "Error: bootgen not found, cannot generate requested PDI/xclbin\n";
      return failure();
    }

    std::string pdiFileName = formatString(pdiName, devName);
    SmallString<128> pdiPath(tmpDirName);
    sys::path::append(pdiPath, pdiFileName);

    // Create BIF file (skip in dry-run)
    SmallString<128> bifPath(tmpDirName);
    sys::path::append(bifPath, devName.str() + "_design.bif");

    if (!dryRun) {
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
      bifFile << "file=" << tmpDirName << "/" << devName
              << "_aie_cdo_elfs.bin ";
      bifFile << "file=" << tmpDirName << "/" << devName
              << "_aie_cdo_init.bin ";
      bifFile << "file=" << tmpDirName << "/" << devName
              << "_aie_cdo_enable.bin";
      bifFile << " }\n";
      bifFile << "  }\n";
      bifFile << "}\n";
      bifFile.close();
    }

    SmallVector<std::string, 8> bootgenCmd = {bootgenPath,
                                              "-arch",
                                              "versal",
                                              "-image",
                                              bifPath.str().str(),
                                              "-o",
                                              pdiPath.str().str(),
                                              "-w"};

    // DEBUG: Print bootgen path to verify we reach this point
    llvm::outs() << bootgenPath << " -arch versal -image " << bifPath
                 << " -o " << pdiPath << " -w\n";
    llvm::outs().flush();

    // Execute bootgen command
    if (!executeCommand(bootgenCmd, /*verboseOutput=*/false)) {
      llvm::errs() << "Error generating PDI\n";
      return failure();
    }

    if (verbose && !dryRun) {
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
                     kernelsPath.str().str(),
                     "--add-replace-section",
                     "AIE_PARTITION:JSON:" + mergedPartitionPath.str().str(),
                     "--force",
                     "--output",
                     xclbinPath.str().str()};
      } else {
        // Create new xclbin from scratch
        xclbinCmd = {xclbinutilPath,
                     "--add-replace-section",
                     "MEM_TOPOLOGY:JSON:" + memTopoPath.str().str(),
                     "--add-kernel",
                     kernelsPath.str().str(),
                     "--add-replace-section",
                     "AIE_PARTITION:JSON:" + partitionPath.str().str(),
                     "--force",
                     "--output",
                     xclbinPath.str().str()};
      }

      if (!executeCommand(xclbinCmd)) {
        llvm::errs() << "Error generating xclbin\n";
        return failure();
      }

      if (verbose && !dryRun) {
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
    if (unified) {
      // Unified compilation: all cores compiled together into one object
      if (failed(compileCoresUnified(context, moduleOp, deviceOp, devName,
                                     tmpDirName, aieTarget, elfPaths))) {
        return failure();
      }
    } else {
      // Per-core compilation (default): each core compiled separately
      if (failed(compileCores(context, moduleOp, deviceOp, devName, tmpDirName,
                              aieTarget, elfPaths))) {
        return failure();
      }
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

    // Generate AIE simulation work folder if requested
    if (failed(generateAiesimWorkFolder(moduleOp, tmpDirName, devName,
                                        aieTarget))) {
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
      info.pdiPath = pdiFullPath.str().str();

      // Collect runtime sequence instruction paths (also absolute)
      for (auto seqOp : deviceOp.getOps<xilinx::AIE::RuntimeSequenceOp>()) {
        StringRef seqName = seqOp.getSymName();
        std::string instsFileName =
            formatString(instsName, devName.str(), seqName);
        SmallString<256> instsFullPath(absTmpDir);
        sys::path::append(instsFullPath, instsFileName);
        info.sequences.emplace_back(seqName.str(), instsFullPath.str().str());
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
  if (noAiesim) {
    aiesim = false;
  }
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
  if (noUnified) {
    unified = false;
  }

  // Process the input file
  return processInputFile(inputFilename, tmpDir);
}
