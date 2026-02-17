//===- aiecc.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This is the main entry point for the AIE compiler driver (aiecc).
// It orchestrates the compilation flow for AIE devices.
//
// This C++ implementation provides similar functionality to the Python aiecc.py
// tool, with the following architecture:
//
// 1. Command-line argument parsing using LLVM CommandLine library
// 2. MLIR module loading and parsing
// 3. MLIR transformation pipeline execution
// 4. Orchestration of external tools (aie-opt, aie-translate, xchesscc, etc.)
// 5. Generation of output artifacts (ELF files, NPU instructions, xclbin, etc.)
//
// This initial implementation provides the core infrastructure and can be
// extended with additional features such as parallel compilation, progress
// reporting, and comprehensive artifact generation.
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

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/InitialAllDialect.h"
#include "aie/version.h"

#include <cstdlib>
#include <memory>
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

static cl::opt<bool> showVersion("version",
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
    cl::init(""), cl::cat(aieCompilerOptions));

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

static cl::opt<bool> generateXclbin("aie-generate-xclbin",
                                   cl::desc("Generate xclbin"),
                                   cl::init(false), cl::cat(aieCompilerOptions));

static cl::opt<std::string>
    xclbinName("xclbin-name", cl::desc("Output xclbin filename"),
              cl::init("{0}.xclbin"), cl::cat(aieCompilerOptions));

static cl::opt<std::string> deviceName("device-name",
                                      cl::desc("Device configuration to compile"),
                                      cl::init(""), cl::cat(aieCompilerOptions));

static cl::opt<unsigned> optLevel("O",
                                 cl::desc("Optimization level (0-3)"),
                                 cl::init(2), cl::cat(aieCompilerOptions));

static cl::alias optLevelLong("opt-level", cl::desc("Alias for -O"),
                             cl::aliasopt(optLevel),
                             cl::cat(aieCompilerOptions));

static cl::opt<bool> execute("n",
                            cl::desc("Disable executing commands (dry run)"),
                            cl::init(true), cl::cat(aieCompilerOptions));

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static void printVersion(raw_ostream &os) {
  os << "aiecc (C++ version) " << AIE_GIT_COMMIT << "\n";
}

static std::string findAieTool(StringRef toolName) {
  // Try to find tool in PATH
  if (auto result = sys::findProgramByName(toolName)) {
    return result.get();
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

static bool executeCommand(ArrayRef<StringRef> command, bool verbose) {
  if (verbose) {
    llvm::outs() << "Executing:";
    for (const auto &arg : command) {
      llvm::outs() << " " << arg;
    }
    llvm::outs() << "\n";
  }

  if (!execute) {
    if (verbose) {
      llvm::outs() << "(Dry run - command not executed)\n";
    }
    return true; // Dry run mode
  }

  std::string errMsg;
  SmallVector<StringRef, 8> env;
  Optional<StringRef> redirects[] = {None, None, None};
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
          (deviceOp.getSymNameAttr() &&
           deviceOp.getSymName() == deviceName)) {
        devices.push_back(op);
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

  // Step 1: Run initial MLIR transformation passes
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

  // Step 2: Run MLIR passes (simplified version)
  SmallString<128> outputPath(tmpDirName);
  sys::path::append(outputPath, "input_with_addresses.mlir");

  std::string allocSchemeOpt = allocScheme;
  if (allocSchemeOpt.empty()) {
    allocSchemeOpt = "basic-sequential";
  }

  // Build pass pipeline
  std::string passPipeline =
      "builtin.module(aie.device(aie-assign-lock-ids,"
      "aie-register-objectFifos,"
      "aie-objectFifo-stateful-transform,"
      "aie-assign-bd-ids,"
      "aie-lower-cascade-flows,"
      "aie-lower-broadcast-packet,"
      "aie-lower-multicast,"
      "aie-assign-tile-controller-ids,"
      "aie-assign-buffer-addresses{alloc-scheme=" + allocSchemeOpt + "}))";

  SmallVector<StringRef, 16> aieOptCmd{
      aieOptPath, inputPath, "--pass-pipeline=" + passPipeline, "-o",
      outputPath};

  if (!executeCommand(aieOptCmd, verbose)) {
    llvm::errs() << "Error running aie-opt passes\n";
    return failure();
  }

  if (verbose) {
    llvm::outs() << "MLIR transformation passes completed successfully\n";
  }

  // Note: In a complete implementation, we would continue with:
  // - Core compilation (using xchesscc or peano)
  // - Linking
  // - Generation of output artifacts (NPU instructions, CDO, xclbin, etc.)
  // This is a simplified version showing the structure

  return success();
}

static int processInputFile(StringRef inputFile, StringRef tmpDirName) {
  // Parse the input file
  MLIRContext context;
  context.loadDialect<xilinx::AIE::AIEDialect>();
  context.loadDialect<xilinx::AIEX::AIEXDialect>();
  xilinx::registerAllDialects(context);

  OwningOpRef<ModuleOp> module;
  
  if (inputFile.empty()) {
    llvm::errs() << "Error: No input file specified\n";
    return 1;
  }

  std::string errorMessage;
  auto fileOrErr = openInputFile(inputFile, &errorMessage);
  if (!fileOrErr) {
    llvm::errs() << "Error opening input file: " << errorMessage << "\n";
    return 1;
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);

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

  cl::SetVersionPrinter(printVersion);
  cl::ParseCommandLineOptions(argc, argv, "AIE Compiler Driver\n");

  if (showVersion) {
    printVersion(llvm::outs());
    return 0;
  }

  // Handle conflicting options
  if (noXbridge) {
    xbridge = false;
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
