//===- aiecc_aiesim.cpp - AIE Simulation support for aiecc ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// AIE simulation work folder generation for the C++ aiecc compiler driver.
//
//===----------------------------------------------------------------------===//

#include "aiecc_aiesim.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

namespace xilinx {
namespace aiecc {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Find an AIE tool in PATH or relative to the executable.
static std::string findAieTool(StringRef toolName) {
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

/// Execute a command with optional verbose output and dry-run support.
static bool executeCommand(ArrayRef<std::string> command, bool verbose,
                           bool dryRun) {
  if (verbose) {
    outs() << "Executing:";
    for (const auto &arg : command) {
      outs() << " " << arg;
    }
    outs() << "\n";
    outs().flush();
  }

  if (dryRun) {
    if (verbose) {
      outs() << "Dry run - command not executed\n";
      outs().flush();
    }
    return true;
  }

  SmallVector<StringRef, 16> cmdRefs;
  cmdRefs.reserve(command.size());
  for (const auto &arg : command) {
    cmdRefs.push_back(arg);
  }

  std::string errMsg;
  int result = sys::ExecuteAndWait(cmdRefs[0], cmdRefs, /*Env=*/std::nullopt,
                                   /*Redirects=*/{},
                                   /*secondsToWait=*/0,
                                   /*memoryLimit=*/0, &errMsg);

  if (result != 0) {
    errs() << "Error: Command failed with exit code " << result << "\n";
    if (!errMsg.empty()) {
      errs() << "Error message: " << errMsg << "\n";
    }
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

SmallVector<std::string> getAieTargetDefines(StringRef aieTarget) {
  SmallVector<std::string> defines;
  std::string target = aieTarget.lower();
  if (target == "aie2" || target == "aieml") {
    defines.push_back("-D__AIEARCH__=20");
  } else if (target == "aie2p") {
    defines.push_back("-D__AIEARCH__=21");
  } else {
    defines.push_back("-D__AIEARCH__=10");
  }
  return defines;
}

LogicalResult generateAieIncCpp(ModuleOp moduleOp, StringRef tmpDirName,
                                StringRef devName, const AiesimConfig &config) {
  if (!config.compileHost && !config.enabled) {
    return success();
  }

  std::string aieTranslatePath = findAieTool("aie-translate");
  if (aieTranslatePath.empty()) {
    errs() << "Error: aie-translate not found in PATH\n";
    return failure();
  }

  // We need to write the module to disk for aie-translate
  SmallString<128> physicalWithElfsPath(tmpDirName);
  sys::path::append(physicalWithElfsPath,
                    devName.str() + "_physical_with_elfs.mlir");

  // Write the module to disk if it doesn't exist
  if (!sys::fs::exists(physicalWithElfsPath)) {
    std::error_code ec;
    raw_fd_ostream moduleFile(physicalWithElfsPath, ec);
    if (ec) {
      errs() << "Error writing module for aie_inc.cpp generation: "
             << ec.message() << "\n";
      return failure();
    }
    moduleOp->print(moduleFile);
    moduleFile.close();
    if (moduleFile.has_error()) {
      errs() << "Error finalizing module file for aie_inc.cpp generation: "
             << moduleFile.error().message() << "\n";
      return failure();
    }
  }

  SmallString<128> aieIncPath(tmpDirName);
  sys::path::append(aieIncPath, "aie_inc.cpp");

  if (config.verbose) {
    outs() << "Generating aie_inc.cpp for device: " << devName << "\n";
  }

  SmallVector<std::string, 8> cmd = {aieTranslatePath,
                                     "--aie-generate-xaie",
                                     "--aie-device-name",
                                     devName.str(),
                                     physicalWithElfsPath.str().str(),
                                     "-o",
                                     aieIncPath.str().str()};

  if (!executeCommand(cmd, config.verbose, config.dryRun)) {
    errs() << "Error generating aie_inc.cpp\n";
    return failure();
  }

  if (config.verbose) {
    outs() << "Generated: " << aieIncPath << "\n";
  }

  return success();
}

LogicalResult generateAiesim(ModuleOp moduleOp, StringRef tmpDirName,
                             StringRef devName, StringRef aieTarget,
                             const AiesimConfig &config) {
  if (!config.enabled) {
    return success();
  }

  if (config.verbose) {
    outs() << "Generating aiesim work folder for device: " << devName << "\n";
  }

  if (config.aietoolsPath.empty()) {
    errs() << "Error: aietools not found, cannot generate aiesim\n";
    return failure();
  }

  // Create sim directory structure
  SmallString<128> simDir(tmpDirName);
  sys::path::append(simDir, "sim");
  if (std::error_code ec = sys::fs::create_directories(simDir)) {
    errs() << "Error creating sim directory: " << ec.message() << "\n";
    return failure();
  }

  SmallString<128> simArchDir(simDir);
  sys::path::append(simArchDir, "arch");
  if (std::error_code ec = sys::fs::create_directories(simArchDir)) {
    errs() << "Error creating sim/arch directory: " << ec.message() << "\n";
    return failure();
  }

  SmallString<128> simReportsDir(simDir);
  sys::path::append(simReportsDir, "reports");
  if (std::error_code ec = sys::fs::create_directories(simReportsDir)) {
    errs() << "Error creating sim/reports directory: " << ec.message() << "\n";
    return failure();
  }

  SmallString<128> simConfigDir(simDir);
  sys::path::append(simConfigDir, "config");
  if (std::error_code ec = sys::fs::create_directories(simConfigDir)) {
    errs() << "Error creating sim/config directory: " << ec.message() << "\n";
    return failure();
  }

  SmallString<128> simPsDir(simDir);
  sys::path::append(simPsDir, "ps");
  if (std::error_code ec = sys::fs::create_directories(simPsDir)) {
    errs() << "Error creating sim/ps directory: " << ec.message() << "\n";
    return failure();
  }

  std::string aieTranslatePath = findAieTool("aie-translate");
  if (aieTranslatePath.empty()) {
    errs() << "Error: aie-translate not found\n";
    return failure();
  }

  // Get path to physical module with ELFs
  SmallString<128> physicalWithElfsPath(tmpDirName);
  sys::path::append(physicalWithElfsPath,
                    devName.str() + "_physical_with_elfs.mlir");

  // Write the module to disk if it doesn't exist
  if (!sys::fs::exists(physicalWithElfsPath)) {
    std::error_code ec;
    raw_fd_ostream moduleFile(physicalWithElfsPath, ec);
    if (ec) {
      errs() << "Error writing module for aiesim: " << ec.message() << "\n";
      return failure();
    }
    moduleOp->print(moduleFile);
    moduleFile.close();
    if (moduleFile.has_error()) {
      errs() << "Error finalizing module for aiesim: "
             << moduleFile.error().message() << "\n";
      return failure();
    }
  }

  // Generate graph.xpe
  SmallString<128> xpePath(simReportsDir);
  sys::path::append(xpePath, "graph.xpe");
  SmallVector<std::string, 8> xpeCmd = {aieTranslatePath,
                                        "--aie-mlir-to-xpe",
                                        "--aie-device-name",
                                        devName.str(),
                                        physicalWithElfsPath.str().str(),
                                        "-o",
                                        xpePath.str().str()};
  if (!executeCommand(xpeCmd, config.verbose, config.dryRun)) {
    errs() << "Error generating graph.xpe\n";
    return failure();
  }

  // Generate aieshim_solution.aiesol
  SmallString<128> aiesolPath(simArchDir);
  sys::path::append(aiesolPath, "aieshim_solution.aiesol");
  SmallVector<std::string, 8> aiesolCmd = {aieTranslatePath,
                                           "--aie-mlir-to-shim-solution",
                                           "--aie-device-name",
                                           devName.str(),
                                           physicalWithElfsPath.str().str(),
                                           "-o",
                                           aiesolPath.str().str()};
  if (!executeCommand(aiesolCmd, config.verbose, config.dryRun)) {
    errs() << "Error generating aieshim_solution.aiesol\n";
    return failure();
  }

  // Generate scsim_config.json
  SmallString<128> scsimConfigPath(simConfigDir);
  sys::path::append(scsimConfigPath, "scsim_config.json");
  SmallVector<std::string, 8> scsimCmd = {aieTranslatePath,
                                          "--aie-mlir-to-scsim-config",
                                          "--aie-device-name",
                                          devName.str(),
                                          physicalWithElfsPath.str().str(),
                                          "-o",
                                          scsimConfigPath.str().str()};
  if (!executeCommand(scsimCmd, config.verbose, config.dryRun)) {
    errs() << "Error generating scsim_config.json\n";
    return failure();
  }

  // Run aie-find-flows pass and generate flows_physical.mlir
  SmallString<128> flowsPath(simDir);
  sys::path::append(flowsPath, "flows_physical.mlir");

  std::string aieOptPath = findAieTool("aie-opt");
  if (!aieOptPath.empty()) {
    SmallVector<std::string, 8> flowsCmd = {
        aieOptPath,
        "--pass-pipeline=builtin.module(aie.device(aie-find-flows))",
        physicalWithElfsPath.str().str(), "-o", flowsPath.str().str()};
    if (!executeCommand(flowsCmd, config.verbose, config.dryRun)) {
      errs() << "Warning: aie-find-flows pass failed\n";
      // Non-fatal, continue
    }
  }

  // Generate flows_physical.json
  if (sys::fs::exists(flowsPath)) {
    SmallString<128> flowsJsonPath(simDir);
    sys::path::append(flowsJsonPath, "flows_physical.json");
    SmallVector<std::string, 8> flowsJsonCmd = {
        aieTranslatePath,         "--aie-flows-to-json",
        "--aie-device-name",      devName.str(),
        flowsPath.str().str(),    "-o",
        flowsJsonPath.str().str()};
    if (!executeCommand(flowsJsonCmd, config.verbose, config.dryRun)) {
      errs() << "Warning: Failed to generate flows_physical.json\n";
      // Non-fatal
    }
  }

  // Build ps.so
  // Find genwrapper_for_ps.cpp
  SmallString<256> genwrapperPath(config.installPath);
  sys::path::append(genwrapperPath, "aie_runtime_lib", aieTarget.upper(),
                    "aiesim", "genwrapper_for_ps.cpp");

  if (!sys::fs::exists(genwrapperPath)) {
    errs() << "Warning: genwrapper_for_ps.cpp not found at " << genwrapperPath
           << ", skipping ps.so generation\n";
  } else {
    // Get paths for includes and libraries
    std::string archName = config.hostTarget;
    size_t dashPos = archName.find('-');
    if (dashPos != std::string::npos) {
      archName = archName.substr(0, dashPos);
    }

    SmallString<256> xaiengineInclude(config.installPath);
    sys::path::append(xaiengineInclude, "runtime_lib", archName, "xaiengine",
                      "include");

    SmallString<256> xaiengineLib(config.installPath);
    sys::path::append(xaiengineLib, "runtime_lib", archName, "xaiengine",
                      "lib");

    SmallString<256> testLibInclude(config.installPath);
    sys::path::append(testLibInclude, "runtime_lib", archName, "test_lib",
                      "include");

    SmallString<256> testLibPath(config.installPath);
    sys::path::append(testLibPath, "runtime_lib", archName, "test_lib", "lib");

    SmallString<256> memAllocator(testLibPath);
    sys::path::append(memAllocator, "libmemory_allocator_sim_aie.a");

    SmallString<128> psSoPath(simPsDir);
    sys::path::append(psSoPath, "ps.so");

    // Find clang++ in PATH
    auto clangPath = sys::findProgramByName("clang++");
    if (!clangPath) {
      errs() << "Warning: clang++ not found in PATH, skipping ps.so "
                "generation\n";
    } else {
      // Build clang++ command
      SmallVector<std::string, 32> clangCmd;
      clangCmd.push_back(*clangPath);
      clangCmd.push_back("-O2");
      clangCmd.push_back("-fuse-ld=lld");
      clangCmd.push_back("-shared");
      clangCmd.push_back("-o");
      clangCmd.push_back(psSoPath.str().str());
      clangCmd.push_back(genwrapperPath.str().str());

      // Add compilation flags
      clangCmd.push_back("-fPIC");
      clangCmd.push_back("-flto");
      clangCmd.push_back("-fpermissive");
      clangCmd.push_back("-DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR");
      clangCmd.push_back("-Wno-deprecated-declarations");
      clangCmd.push_back("-Wno-enum-constexpr-conversion");
      clangCmd.push_back("-Wno-format-security");
      clangCmd.push_back("-DSC_INCLUDE_DYNAMIC_PROCESSES");
      clangCmd.push_back("-D__AIESIM__");
      clangCmd.push_back("-D__PS_INIT_AIE__");
      clangCmd.push_back("-Og");
      clangCmd.push_back("-Dmain(...)=ps_main(...)");

      // Include paths
      clangCmd.push_back("-I" + tmpDirName.str());
      clangCmd.push_back("-I" + config.aietoolsPath + "/include");
      clangCmd.push_back("-I" + xaiengineInclude.str().str());
      clangCmd.push_back("-I" + config.aietoolsPath +
                         "/data/osci_systemc/include");
      clangCmd.push_back("-I" + config.aietoolsPath + "/include/xtlm/include");
      clangCmd.push_back("-I" + config.aietoolsPath +
                         "/include/common_cpp/common_cpp_v1_0/include");
      clangCmd.push_back("-I" + testLibInclude.str().str());

      // Memory allocator library
      if (sys::fs::exists(memAllocator)) {
        clangCmd.push_back(memAllocator.str().str());
      }

      // Link libraries
      clangCmd.push_back("-L" + xaiengineLib.str().str());
      clangCmd.push_back("-lxaienginecdo");
      clangCmd.push_back("-L" + config.aietoolsPath + "/lib/lnx64.o");
      clangCmd.push_back("-L" + config.aietoolsPath + "/lib/lnx64.o/Ubuntu");
      clangCmd.push_back("-L" + config.aietoolsPath +
                         "/data/osci_systemc/lib/lnx64");
      clangCmd.push_back("-Wl,--as-needed");
      clangCmd.push_back("-lsystemc");
      clangCmd.push_back("-lxtlm");

      // Add AIE target defines
      auto defines = getAieTargetDefines(aieTarget);
      for (const auto &def : defines) {
        clangCmd.push_back(def);
      }

      if (!executeCommand(clangCmd, config.verbose, config.dryRun)) {
        errs() << "Warning: Failed to build ps.so\n";
        // Non-fatal - aiesim folder is still partially generated
      }
    }
  }

  // Create .target file
  SmallString<128> targetFilePath(simDir);
  sys::path::append(targetFilePath, ".target");
  {
    std::error_code ec;
    raw_fd_ostream targetFile(targetFilePath, ec);
    if (ec) {
      errs() << "Error creating .target: " << ec.message() << "\n";
    } else {
      targetFile << "hw\n";
      targetFile.close();
      if (targetFile.has_error()) {
        errs() << "Error writing .target: " << targetFile.error().message()
               << "\n";
      }
    }
  }

  // Generate aiesim.sh script
  SmallString<128> simScriptPath(tmpDirName);
  sys::path::append(simScriptPath, "aiesim.sh");
  {
    std::error_code ec;
    raw_fd_ostream scriptFile(simScriptPath, ec);
    if (ec) {
      errs() << "Error creating aiesim.sh: " << ec.message() << "\n";
      return failure();
    }
    scriptFile << R"(#!/bin/sh
prj_name=$(basename $(dirname $(realpath $0)))
root=$(dirname $(dirname $(realpath $0)))
vcd_filename=foo
if [ -n "$1" ]; then
  vcd_filename=$1
fi
cd $root
aiesimulator --pkg-dir=${prj_name}/sim --dump-vcd ${vcd_filename}
)";
    scriptFile.close();
    if (scriptFile.has_error()) {
      errs() << "Error writing aiesim.sh: " << scriptFile.error().message()
             << "\n";
      return failure();
    }
  }

  // Make script executable
  if (std::error_code ec = sys::fs::setPermissions(
          simScriptPath, sys::fs::perms::owner_all | sys::fs::perms::group_exe |
                             sys::fs::perms::others_exe)) {
    errs() << "Warning: Failed to set executable permissions on aiesim.sh: "
           << ec.message() << "\n";
  }

  outs() << "Simulation generated...\n";
  outs() << "To run simulation: " << simScriptPath << "\n";

  return success();
}

} // namespace aiecc
} // namespace xilinx
