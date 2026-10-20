//===- aie-visualize.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"
#include "aie/Visualize/AIEVisualize.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace xilinx;

static cl::opt<std::string> fileName(cl::Positional, cl::desc("<input mlir>"),
                                     cl::Required);
static cl::opt<bool> emitDot("emit-dot",
                             cl::desc("Emit a DOT route visualization"));
static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));
static cl::list<unsigned>
    onlyFlows("only-flow", cl::desc("Emit only a flow with this numeric ID"),
              cl::CommaSeparated, cl::ZeroOrMore);
static cl::list<std::string> contractedNodeKinds(
    "contract",
    cl::desc("Contract semantic node kinds: tile, port, buffer, "
             "runtime-argument, call"),
    cl::value_desc("kind"), cl::CommaSeparated, cl::ZeroOrMore);
static cl::opt<bool> noFollowBuffers(
    "no-follow-buffers",
    cl::desc("Do not group flows through buffers or shared endpoints"));
static cl::opt<std::string>
    deviceName("device", cl::desc("Select an aie.device by symbol name"),
               cl::value_desc("symbol"));
static cl::opt<std::string>
    runtimeSequenceName("runtime-sequence",
                        cl::desc("Select an aie.runtime_sequence by symbol"),
                        cl::value_desc("symbol"));
static cl::opt<bool>
    skipRuntime("skip-runtime",
                cl::desc("Do not annotate DMA endpoints from runtime setup"));
static cl::opt<bool>
    skipVerify("skip-verify",
               cl::desc("Do not verify analysis IR after parsing"));
static cl::opt<bool>
    weighEdges("weigh-edges",
               cl::desc("Scale edge width by the bytes each flow group moves"));
static cl::opt<double>
    minEdgeWidth("min-edge-width",
                 cl::desc("Edge width in points for the lightest flow group"),
                 cl::init(0.5));
static cl::opt<double>
    maxEdgeWidth("max-edge-width",
                 cl::desc("Edge width in points for the heaviest flow group"),
                 cl::init(7.0));
static cl::opt<std::string> arrowStyle(
    "arrows",
    cl::desc("How arrows run between tiles: straight (default) or channel, "
             "which keeps them in the gaps and off the tiles"),
    cl::value_desc("style"), cl::init("straight"));
static cl::opt<bool>
    tileRoles("tile-roles",
              cl::desc("Headline each tile with the kernels it runs and mark "
                       "tiles that run the same kernels"));
static cl::opt<unsigned>
    maxHeadlineKernels("max-headline-kernels",
                       cl::desc("Name at most this many kernels in a tile "
                                "headline, counting the rest"),
                       cl::init(4));
static cl::opt<double> tileWidth("tile-width", cl::desc("Tile width in inches"),
                                 cl::init(2.15));
static cl::opt<double>
    minimumTileHeight("tile-height", cl::desc("Minimum tile height in inches"),
                      cl::init(2.15));
static cl::opt<double>
    tileGapX("tile-gap-x", cl::desc("Horizontal gap between tiles in inches"),
             cl::init(0.85));
static cl::opt<double>
    tileGapY("tile-gap-y", cl::desc("Vertical gap between tiles in inches"),
             cl::init(0.85));
static cl::opt<double>
    localSpacing("local-spacing",
                 cl::desc("Vertical spacing between tile-local nodes"),
                 cl::init(0.4));

const std::string bold("\033[0;1m");
const std::string dim("\033[0;2m");
const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string blue("\033[1;34m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string bwhite("\033[0;47m");
const std::string reset("\033[0m");
const std::string bgray("\033[48;5;239m");

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv);

  MLIRContext ctx;
  ParserConfig pcfg(&ctx, /*verifyAfterParse=*/!skipVerify);
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
      parseSourceFile<ModuleOp>(fileName, srcMgr, pcfg);

  if (!owning)
    return 1;

  // Named here rather than in the library, which does not know what the user
  // typed to reach this state.
  if (skipRuntime && !runtimeSequenceName.empty()) {
    errs() << "--skip-runtime and --runtime-sequence are mutually exclusive\n";
    return 3;
  }

  AIE::VisualizeOptions options;
  if (arrowStyle == "channel") {
    options.arrows = AIE::ArrowStyle::Channel;
  } else if (arrowStyle != "straight") {
    errs() << "--arrows expects 'straight' or 'channel'\n";
    return 3;
  }
  options.deviceName = deviceName;
  options.runtimeSequenceName = runtimeSequenceName;
  options.skipRuntime = skipRuntime;
  options.onlyFlows.assign(onlyFlows.begin(), onlyFlows.end());
  options.contractedNodeKinds.assign(contractedNodeKinds.begin(),
                                     contractedNodeKinds.end());
  options.noFollowBuffers = noFollowBuffers;
  options.weighEdges = weighEdges;
  options.minEdgeWidth = minEdgeWidth;
  options.maxEdgeWidth = maxEdgeWidth;
  options.tileRoles = tileRoles;
  options.maxHeadlineKernels = maxHeadlineKernels;
  options.tileWidth = tileWidth;
  options.minimumTileHeight = minimumTileHeight;
  options.tileGapX = tileGapX;
  options.tileGapY = tileGapY;
  options.localSpacing = localSpacing;

  if (emitDot) {
    std::error_code error;
    ToolOutputFile output(outputFilename, error, sys::fs::OF_Text);
    if (error) {
      errs() << error.message() << '\n';
      return 3;
    }
    if (failed(AIE::emitRouteDot(*owning, output.os(), options)))
      return 4;
    output.keep();
    return 0;
  }

  SmallVector<AIE::DeviceOp> deviceOps(owning->getOps<AIE::DeviceOp>());
  AIE::DeviceOp deviceOp;
  if (!deviceName.empty()) {
    for (AIE::DeviceOp candidate : deviceOps) {
      if (candidate.getSymName() == deviceName) {
        deviceOp = candidate;
        break;
      }
    }
    if (!deviceOp) {
      errs() << "no aie.device named '" << deviceName << "'\n";
      return 2;
    }
  } else if (deviceOps.size() == 1) {
    deviceOp = deviceOps.front();
  } else if (deviceOps.empty()) {
    errs() << "input contains no aie.device operations\n";
    return 2;
  } else {
    errs() << "input contains multiple aie.device operations; select one with "
              "--device=<symbol>\n";
    return 2;
  }

  const xilinx::AIE::AIETargetModel &model = deviceOp.getTargetModel();

  model.validate();

  std::vector<bool> used(model.columns() * model.rows());
  for (int col = 0; col < model.columns(); col++) {
    for (int row = 0; row < model.rows(); row++) {
      used[col + model.columns() * row] = false;
    }
  }
  for (auto tile : deviceOp.getOps<AIE::TileOp>()) {
    used[tile.getCol() + model.columns() * tile.getRow()] = true;
  }

  std::cout << model.columns() << " Columns and " << model.rows() << " Rows\n";
  for (int row = model.rows() - 1; row >= 0; row--) {
    std::cout << reset << row % 10 << " ";
    for (int col = 0; col < model.columns(); col++) {
      if (used[col + model.columns() * row])
        std::cout << bgray;
      else
        std::cout << dim;
      std::string v = reset + ".";
      if (model.isCoreTile(col, row))
        v = green + 'C';
      else if (model.isMemTile(col, row))
        v = red + 'M';
      else if (model.isShimNOCTile(col, row))
        v = blue + 'D';
      else if (model.isShimPLTile(col, row))
        v = magenta + 'P';
      std::cout << v << reset;
    }
    std::cout << "\n";
  }

  std::cout << reset << "  ";
  for (int col = 0; col < model.columns(); col++)
    std::cout << col % 10;
  std::cout << "\n";

  std::cout << "  ";
  for (int col = 0; col < model.columns(); col++) {
    int coltens = col / 10;
    if (coltens > 0)
      std::cout << coltens;
    else
      std::cout << " ";
  }
  std::cout << "\n";

  return 0;
}
