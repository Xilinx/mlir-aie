//===- aiecc_aiesim.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// AIE Simulation work folder generation for aiecc.
//
//===----------------------------------------------------------------------===//

#include "aiecc_aiesim.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace mlir;
using namespace llvm;

namespace aiecc {

/// Execute an external command and return true on success.
static bool executeCommand(ArrayRef<std::string> args, bool verbose) {
  if (args.empty())
    return false;

  if (verbose) {
    outs() << "Executing:";
    for (const auto &arg : args) {
      outs() << " " << arg;
    }
    outs() << "\n";
  }

  SmallVector<StringRef, 16> argsRef;
  for (const auto &arg : args) {
    argsRef.push_back(arg);
  }

  std::string errMsg;
  int result =
      sys::ExecuteAndWait(argsRef[0], argsRef, std::nullopt, {}, 0, 0, &errMsg);

  if (result != 0) {
    errs() << "Command failed";
    if (!errMsg.empty()) {
      errs() << ": " << errMsg;
    }
    errs() << "\n";
    return false;
  }

  return true;
}

LogicalResult generateAiesimWorkFolder(ModuleOp moduleOp, StringRef tmpDirName,
                                       StringRef devName, StringRef aieTarget,
                                       StringRef aietoolsPath,
                                       StringRef installPath,
                                       const AiesimConfig &config) {
  if (!config.enabled) {
    return success();
  }

  if (config.verbose) {
    outs() << "Generating AIE simulation work folder for device: " << devName
           << "\n";
  }

  // Validate aietools path
  if (aietoolsPath.empty()) {
    errs() << "Error: --aiesim requires aietools installation. "
           << "Set AIETOOLS_ROOT or ensure xchesscc is in PATH.\n";
    return failure();
  }

  // Create sim directory structure
  SmallString<256> simDir(tmpDirName);
  sys::path::append(simDir, "sim");

  // Remove existing sim directory if present
  if (sys::fs::exists(simDir)) {
    if (auto ec = sys::fs::remove_directories(simDir)) {
      errs() << "Warning: Could not remove existing sim directory: "
             << ec.message() << "\n";
    }
  }

  // Create subdirectories
  SmallString<256> archDir(simDir);
  sys::path::append(archDir, "arch");
  SmallString<256> reportsDir(simDir);
  sys::path::append(reportsDir, "reports");
  SmallString<256> configDir(simDir);
  sys::path::append(configDir, "config");
  SmallString<256> psDir(simDir);
  sys::path::append(psDir, "ps");

  for (const auto &dir : {simDir, archDir, reportsDir, configDir, psDir}) {
    if (auto ec = sys::fs::create_directories(dir)) {
      errs() << "Error creating directory " << dir << ": " << ec.message()
             << "\n";
      return failure();
    }
  }

  if (config.dryRun) {
    if (config.verbose) {
      outs() << "Would generate AIE simulation files in: " << simDir << "\n";
    }
    return success();
  }

  // Generate graph.xpe
  SmallString<256> graphXpePath(reportsDir);
  sys::path::append(graphXpePath, "graph.xpe");
  {
    std::error_code ec;
    raw_fd_ostream outFile(graphXpePath, ec);
    if (ec) {
      errs() << "Error creating graph.xpe: " << ec.message() << "\n";
      return failure();
    }
    if (failed(xilinx::AIE::AIETranslateGraphXPE(moduleOp, outFile, devName))) {
      errs() << "Error generating graph.xpe\n";
      return failure();
    }
  }
  if (config.verbose) {
    outs() << "Generated: " << graphXpePath << "\n";
  }

  // Generate aieshim_solution.aiesol
  SmallString<256> shimSolutionPath(archDir);
  sys::path::append(shimSolutionPath, "aieshim_solution.aiesol");
  {
    std::error_code ec;
    raw_fd_ostream outFile(shimSolutionPath, ec);
    if (ec) {
      errs() << "Error creating aieshim_solution.aiesol: " << ec.message()
             << "\n";
      return failure();
    }
    if (failed(xilinx::AIE::AIETranslateShimSolution(moduleOp, outFile,
                                                     devName))) {
      errs() << "Error generating aieshim_solution.aiesol\n";
      return failure();
    }
  }
  if (config.verbose) {
    outs() << "Generated: " << shimSolutionPath << "\n";
  }

  // Generate scsim_config.json
  SmallString<256> scsimConfigPath(configDir);
  sys::path::append(scsimConfigPath, "scsim_config.json");
  {
    std::error_code ec;
    raw_fd_ostream outFile(scsimConfigPath, ec);
    if (ec) {
      errs() << "Error creating scsim_config.json: " << ec.message() << "\n";
      return failure();
    }
    if (failed(
            xilinx::AIE::AIETranslateSCSimConfig(moduleOp, outFile, devName))) {
      errs() << "Error generating scsim_config.json\n";
      return failure();
    }
  }
  if (config.verbose) {
    outs() << "Generated: " << scsimConfigPath << "\n";
  }

  // Run aie-find-flows pass and output to flows_physical.mlir
  SmallString<256> flowsPath(simDir);
  sys::path::append(flowsPath, "flows_physical.mlir");
  {
    // Clone the module for the find-flows pass
    OwningOpRef<ModuleOp> flowsModule = moduleOp.clone();
    MLIRContext *ctx = flowsModule->getContext();

    PassManager pm(ctx);
    OpPassManager &devicePm = pm.nest<xilinx::AIE::DeviceOp>();
    devicePm.addPass(xilinx::AIE::createAIEFindFlowsPass());

    if (failed(pm.run(*flowsModule))) {
      errs() << "Error running aie-find-flows pass\n";
      return failure();
    }

    // Write the module with flows
    std::error_code ec;
    raw_fd_ostream outFile(flowsPath, ec);
    if (ec) {
      errs() << "Error creating flows_physical.mlir: " << ec.message() << "\n";
      return failure();
    }
    flowsModule->print(outFile);
  }
  if (config.verbose) {
    outs() << "Generated: " << flowsPath << "\n";
  }

  // Generate flows_physical.json from the flows MLIR
  SmallString<256> flowsJsonPath(simDir);
  sys::path::append(flowsJsonPath, "flows_physical.json");
  {
    // Parse the flows MLIR file we just wrote
    ParserConfig parseConfig(moduleOp.getContext());
    SourceMgr sourceMgr;
    OwningOpRef<ModuleOp> flowsModule =
        parseSourceFile<ModuleOp>(flowsPath.str(), sourceMgr, parseConfig);
    if (!flowsModule) {
      errs() << "Error parsing flows_physical.mlir\n";
      return failure();
    }

    std::error_code ec;
    raw_fd_ostream outFile(flowsJsonPath, ec);
    if (ec) {
      errs() << "Error creating flows_physical.json: " << ec.message() << "\n";
      return failure();
    }
    if (failed(xilinx::AIE::AIEFlowsToJSON(*flowsModule, outFile, devName))) {
      errs() << "Error generating flows_physical.json\n";
      return failure();
    }
  }
  if (config.verbose) {
    outs() << "Generated: " << flowsJsonPath << "\n";
  }

  // Compile genwrapper_for_ps.cpp to ps.so
  std::string aieTargetUpper = aieTarget.upper();
  SmallString<256> genwrapperPath(installPath);
  sys::path::append(genwrapperPath, "aie_runtime_lib", aieTargetUpper, "aiesim",
                    "genwrapper_for_ps.cpp");

  if (!sys::fs::exists(genwrapperPath)) {
    errs() << "Warning: genwrapper_for_ps.cpp not found at " << genwrapperPath
           << ", skipping ps.so generation\n";
  } else {
    // Find clang++
    auto clangPath = sys::findProgramByName("clang++");
    if (!clangPath) {
      errs() << "Warning: clang++ not found, skipping ps.so generation\n";
    } else {
      // Build paths for includes and libraries
      SmallString<256> xaiengineIncludePath(installPath);
      sys::path::append(xaiengineIncludePath, "runtime_lib", "x86_64",
                        "xaiengine", "include");
      SmallString<256> xaiengineLibPath(installPath);
      sys::path::append(xaiengineLibPath, "runtime_lib", "x86_64", "xaiengine",
                        "lib");
      SmallString<256> testLibIncludePath(installPath);
      sys::path::append(testLibIncludePath, "runtime_lib", "x86_64", "test_lib",
                        "include");
      SmallString<256> testLibPath(installPath);
      sys::path::append(testLibPath, "runtime_lib", "x86_64", "test_lib",
                        "lib");
      SmallString<256> memAllocatorPath(testLibPath);
      sys::path::append(memAllocatorPath, "libmemory_allocator_sim_aie.a");

      SmallString<256> psOutputPath(psDir);
      sys::path::append(psOutputPath, "ps.so");

      // Build AIE target define
      std::string aieTargetDefine = "-D__AIEARCH__=";
      if (aieTarget == "aie2" || aieTarget == "aie2p") {
        aieTargetDefine += "20";
      } else {
        aieTargetDefine += "10";
      }

      std::vector<std::string> compileCmd = {
          *clangPath,
          "-O2",
          "-fuse-ld=lld",
          "-shared",
          "-o",
          psOutputPath.str().str(),
          genwrapperPath.str().str(),
          aieTargetDefine,
          "-fPIC",
          "-flto",
          "-fpermissive",
          "-DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR",
          "-Wno-deprecated-declarations",
          "-Wno-enum-constexpr-conversion",
          "-Wno-format-security",
          "-DSC_INCLUDE_DYNAMIC_PROCESSES",
          "-D__AIESIM__",
          "-D__PS_INIT_AIE__",
          "-Og",
          "-Dmain(...)=ps_main(...)",
          "-I" + tmpDirName.str(),
          "-I" + aietoolsPath.str() + "/include",
          "-I" + xaiengineIncludePath.str().str(),
          "-I" + aietoolsPath.str() + "/data/osci_systemc/include",
          "-I" + aietoolsPath.str() + "/include/xtlm/include",
          "-I" + aietoolsPath.str() +
              "/include/common_cpp/common_cpp_v1_0/include",
          "-I" + testLibIncludePath.str().str(),
          memAllocatorPath.str().str(),
          "-L" + xaiengineLibPath.str().str(),
          "-lxaienginecdo",
          "-L" + aietoolsPath.str() + "/lib/lnx64.o",
          "-L" + aietoolsPath.str() + "/lib/lnx64.o/Ubuntu",
          "-L" + aietoolsPath.str() + "/data/osci_systemc/lib/lnx64",
          "-Wl,--as-needed",
          "-lsystemc",
          "-lxtlm"};

      if (!executeCommand(compileCmd, config.verbose)) {
        errs() << "Warning: Failed to compile ps.so\n";
      } else if (config.verbose) {
        outs() << "Generated: " << psOutputPath << "\n";
      }
    }
  }

  // Write aiesim.sh script
  SmallString<256> simScriptPath(tmpDirName);
  sys::path::append(simScriptPath, "aiesim.sh");
  {
    std::error_code ec;
    raw_fd_ostream outFile(simScriptPath, ec);
    if (ec) {
      errs() << "Error creating aiesim.sh: " << ec.message() << "\n";
      return failure();
    }
    outFile << "#!/bin/sh\n";
    outFile << "prj_name=$(basename $(dirname $(realpath $0)))\n";
    outFile << "root=$(dirname $(dirname $(realpath $0)))\n";
    outFile << "vcd_filename=foo\n";
    outFile << "if [ -n \"$1\" ]; then\n";
    outFile << "  vcd_filename=$1\n";
    outFile << "fi\n";
    outFile << "cd $root\n";
    outFile << "aiesimulator --pkg-dir=${prj_name}/sim --dump-vcd "
               "${vcd_filename}\n";
  }
  // Make script executable
  if (auto ec = sys::fs::setPermissions(
          simScriptPath,
          sys::fs::perms::owner_all | sys::fs::perms::group_read |
              sys::fs::perms::group_exe | sys::fs::perms::others_read |
              sys::fs::perms::others_exe)) {
    errs() << "Warning: Could not set execute permission on aiesim.sh: "
           << ec.message() << "\n";
  }
  if (config.verbose) {
    outs() << "Generated: " << simScriptPath << "\n";
  }

  // Write .target file
  SmallString<256> targetFilePath(simDir);
  sys::path::append(targetFilePath, ".target");
  {
    std::error_code ec;
    raw_fd_ostream outFile(targetFilePath, ec);
    if (ec) {
      errs() << "Error creating .target file: " << ec.message() << "\n";
      return failure();
    }
    outFile << "hw\n";
  }

  outs() << "Simulation generated...\n";
  outs() << "To run simulation: " << simScriptPath << "\n";

  return success();
}

} // namespace aiecc
