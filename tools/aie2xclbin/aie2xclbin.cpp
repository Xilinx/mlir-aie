//===- aiecc.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===---------------------------------------------------------------------===//

#include "XCLBinGen.h"
#include "configure.h"

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"

#include <iostream>
#include <regex>
#include <stdlib.h>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace xilinx;

cl::OptionCategory AIE2XCLBinCat("AIE To XCLBin Options",
                                 "Options specific to the aie2xclbin tool");

cl::opt<std::string> FileName(cl::Positional, cl::desc("<input mlir>"),
                              cl::Required, cl::cat(AIE2XCLBinCat));

cl::opt<std::string>
    TmpDir("tmpdir", cl::desc("Directory used for temporary file storage"),
           cl::cat(AIE2XCLBinCat));

cl::opt<bool> Verbose("v", cl::desc("Trace commands as they are executed"),
                      cl::cat(AIE2XCLBinCat));

cl::opt<std::string>
    Peano("peano", cl::desc("Root directory where peano compiler is installed"),
          cl::cat(AIE2XCLBinCat));

cl::opt<std::string>
    HostArch("host-target", cl::desc("Target architecture of the host program"),
             cl::init(HOST_ARCHITECTURE), cl::cat(AIE2XCLBinCat));

cl::opt<std::string>
    NPUInstsName("ipu-insts-name",
                 cl::desc("Output instructions filename for NPU target"),
                 cl::init("ipu_insts.txt"), cl::cat(AIE2XCLBinCat));

cl::opt<bool>
    PrintIRAfterAll("print-ir-after-all",
                    cl::desc("Configure all pass managers in lowering from aie "
                             "to xclbin to print IR after all passes"),
                    cl::init(false), cl::cat(AIE2XCLBinCat));

cl::opt<bool>
    PrintIRBeforeAll("print-ir-before-all",
                     cl::desc("Configure all pass managers in lowering from "
                              "aie to xclbin to print IR before all passes"),
                     cl::init(false), cl::cat(AIE2XCLBinCat));

cl::opt<bool>
    DisableThreading("disable-threading",
                     cl::desc("Configure all pass managers in lowering from "
                              "aie to xclbin to disable multithreading"),
                     cl::init(false), cl::cat(AIE2XCLBinCat));

cl::opt<bool> PrintIRModuleScope(
    "print-ir-module-scope",
    cl::desc("Configure all pass managers in lowering from aie to xclbin to "
             "print IR at the module scope"),
    cl::init(false), cl::cat(AIE2XCLBinCat));

cl::opt<std::string>
    XCLBinName("xclbin-name",
               cl::desc("Output xclbin filename for CDO/XCLBIN target"),
               cl::init("final.xclbin"), cl::cat(AIE2XCLBinCat));

cl::opt<std::string> XCLBinKernelName("xclbin-kernel-name",
                                      cl::desc("Kernel name in xclbin file"),
                                      cl::init("MLIR_AIE"),
                                      cl::cat(AIE2XCLBinCat));

cl::opt<std::string>
    XCLBinInstanceName("xclbin-instance-name",
                       cl::desc("Instance name in xclbin metadata"),
                       cl::init("MLIRAIEV1"), cl::cat(AIE2XCLBinCat));

cl::opt<std::string> XCLBinKernelID("xclbin-kernel-id",
                                    cl::desc("Kernel id in xclbin file"),
                                    cl::init("0x901"), cl::cat(AIE2XCLBinCat));

cl::opt<std::string> InstallDir("install-dir",
                                cl::desc("Root of mlir-aie installation"),
                                cl::cat(AIE2XCLBinCat));

cl::opt<bool> UseChess("use-chess",
                       cl::desc("Use chess compiler instead of peano"),
                       cl::cat(AIE2XCLBinCat));

int main(int argc, char *argv[]) {
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerTranslationCLOptions();
  cl::ParseCommandLineOptions(argc, argv);
  XCLBinGenConfig TK;
  TK.Verbose = Verbose;
  TK.HostArch = HostArch;
  TK.XCLBinKernelName = XCLBinKernelName;
  TK.XCLBinKernelID = XCLBinKernelID;
  TK.XCLBinInstanceName = XCLBinInstanceName;
  TK.UseChess = UseChess;
  TK.DisableThreading = DisableThreading;
  TK.PrintIRAfterAll = PrintIRAfterAll;
  TK.PrintIRBeforeAll = PrintIRBeforeAll;
  TK.PrintIRModuleScope = PrintIRModuleScope;

  if (TK.UseChess)
    findVitis(TK);

  if (Verbose)
    llvm::dbgs() << "\nCompiling " << FileName << "\n";

  if (InstallDir.size()) {
    TK.InstallDir = InstallDir;
  } else {
    // Navigate up from install/bin/aie2xclbin to install/
    TK.InstallDir = sys::path::parent_path(sys::path::parent_path(argv[0]));
  }
  TK.PeanoDir = Peano.getValue();
  if (!TK.UseChess && !sys::fs::is_directory(TK.PeanoDir)) {
    llvm::errs() << "Peano path \"" << TK.PeanoDir << "\" is invalid\n";
    return 1;
  }

  if (TmpDir.size())
    TK.TempDir = TmpDir.getValue();
  else
    TK.TempDir = FileName + ".prj";

  std::error_code err;
  SmallString<64> tmpDir(TK.TempDir);
  err = sys::fs::make_absolute(tmpDir);
  if (err)
    llvm::errs() << "Failed to make absolute path: " << err.message() << "\n";

  TK.TempDir = std::string(tmpDir);

  err = sys::fs::create_directories(TK.TempDir);
  if (err) {
    llvm::errs() << "Failed to create temporary directory " << TK.TempDir
                 << ": " << err.message() << "\n";
    return 1;
  }

  if (Verbose)
    llvm::errs() << "Created temporary directory " << TK.TempDir << "\n";

  MLIRContext ctx;
  ParserConfig pcfg(&ctx);
  SourceMgr srcMgr;

  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<vector::VectorDialect>();
  xilinx::registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  xilinx::xllvm::registerXLLVMDialectTranslation(registry);
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> owning =
      parseSourceFile<ModuleOp>(FileName, srcMgr, pcfg);

  if (!owning)
    return 1;

  if (failed(aie2xclbin(&ctx, *owning, TK, NPUInstsName.getValue(),
                        XCLBinName.getValue())))
    return 1;

  return 0;
}
