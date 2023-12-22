//===- ADFGenerateCppGraph.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/ADF/ADFOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/FileSystem.h"

#include <unordered_map>
#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::ADF;

/// Manages the indentation as we traverse the IR nesting.
static int currentindent = 0;
struct Indent {
  int indent;
  Indent() : indent(1) { currentindent += indent; }
  ~Indent() { currentindent -= indent; }
};
static void resetIndent() { currentindent = 0; }

raw_ostream &operator<<(raw_ostream &os, const Indent &indent) {
  for (int i = 0; i < currentindent; ++i)
    os << "  ";
  return os;
}

struct GraphWriter {
  raw_ostream &output;
  GraphWriter(raw_ostream &output) : output(output) {}

  // maps KernelOp to the generated c++ variable name.
  std::unordered_map<Operation *, std::string> kernelOp2VarName;

  StringRef getCTypeString(const Type &type) {
    if (llvm::dyn_cast<int8Type>(type))
      return int8Type::getMnemonic();
    if (llvm::dyn_cast<int16Type>(type))
      return int16Type::getMnemonic();
    if (llvm::dyn_cast<int32Type>(type))
      return int32Type::getMnemonic();
    if (llvm::dyn_cast<int64Type>(type))
      return int64Type::getMnemonic();
    if (llvm::dyn_cast<uint8Type>(type))
      return uint8Type::getMnemonic();
    if (llvm::dyn_cast<uint16Type>(type))
      return uint16Type::getMnemonic();
    if (llvm::dyn_cast<uint32Type>(type))
      return uint32Type::getMnemonic();
    if (llvm::dyn_cast<uint64Type>(type))
      return uint64Type::getMnemonic();
    if (llvm::dyn_cast<floatType>(type))
      return floatType::getMnemonic();
    llvm::report_fatal_error("unknown type");
  }

  std::string getKernelTypeString(const std::string &direction, Type type) {
    if (auto window = llvm::dyn_cast<WindowType>(type))
      return (direction + "_window_" + getCTypeString(window.getType()) + " *")
          .str();
    if (auto stream = llvm::dyn_cast<StreamType>(type))
      return (direction + "_stream_" + getCTypeString(stream.getType()) + " *")
          .str();
    if (auto stream = llvm::dyn_cast<ParameterType>(type))
      return std::string(getCTypeString(stream.getType()));

    llvm::report_fatal_error("unknown kernel type");
  }

  std::string getConnectionTypeString(Type type) {
    if (auto windowType = llvm::dyn_cast<WindowType>(type))
      return std::string("window<") + std::to_string(windowType.getSize()) +
             "> ";
    if (llvm::dyn_cast<StreamType>(type))
      return "stream";
    if (llvm::dyn_cast<ParameterType>(type))
      return "parameter";
    llvm::report_fatal_error("unknown connection type");
  }

  std::string getTempNetName() {
    static uint32_t netCnt = 0;
    return std::string("n") + std::to_string(netCnt++);
  }

  void visitOpResultUsers(GraphInputOp driverOp) {
    Indent indent;
    for (auto indexedResult : llvm::enumerate(driverOp->getResults())) {
      Value result = indexedResult.value();
      for (OpOperand &userOperand : result.getUses()) {
        Operation *userOp = userOperand.getOwner();
        int targetIndex = userOperand.getOperandNumber();
        if (auto kernel = dyn_cast<KernelOp>(userOp)) {
          auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              driverOp, kernel.getCalleeAttr());
          Type opType = funcOp.getFunctionType().getInput(targetIndex);
          std::string targetKernelName = kernelOp2VarName[kernel];
          output << indent << "connect<" << getConnectionTypeString(opType)
                 << "> ";
          output << getTempNetName() << " (" << driverOp.getName() << ", "
                 << targetKernelName << ".in[" << targetIndex << "]);\n";
        }
        // todo: kernel should not drive graph input, add an mlir verifier
        // condition
      }
    }
  }

  void visitOpResultUsers(KernelOp source) {
    Indent indent;
    std::string sourceKernelName = kernelOp2VarName[source];

    unsigned sourceIndex = 0;
    for (auto indexedResult : llvm::enumerate(source->getResults())) {
      Value result = indexedResult.value();
      for (OpOperand &userOperand : result.getUses()) {
        Operation *userOp = userOperand.getOwner();
        int targetIndex = userOperand.getOperandNumber();
        if (auto kernel = dyn_cast<KernelOp>(userOp)) {
          auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              kernel, kernel.getCalleeAttr());
          Type opType = funcOp.getFunctionType().getInput(targetIndex);
          auto targetKernelName = kernelOp2VarName[kernel];
          output << indent << "connect<" << getConnectionTypeString(opType)
                 << "> ";
          output << getTempNetName() << " (" << sourceKernelName << ".out["
                 << sourceIndex << "], " << targetKernelName << ".in["
                 << targetIndex << "]);\n";
        } else if (auto outputOp = dyn_cast<GraphOutputOp>(userOp)) {
          auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              source, source.getCalleeAttr());
          Type opType = funcOp.getFunctionType().getInput(sourceIndex);
          output << indent << "connect<" << getConnectionTypeString(opType)
                 << "> ";
          output << getTempNetName() << " (" << sourceKernelName << ".out["
                 << sourceIndex << "], " << outputOp.getName() << ");\n";
        }
        // todo: kernel should not drive graph input, add an mlir verifier
        // condition
      }
      sourceIndex++;
    }
  }

  void writeKernelFunctions(ModuleOp module) {
    output << "#include <adf.h>\n";
    output << "#ifndef FUNCTION_KERNELS_H\n";
    output << "#define FUNCTION_KERNELS_H\n\n";

    for (Block &block : module.getBodyRegion())
      for (auto funcOp : block.getOps<func::FuncOp>()) {
        output << "void " << funcOp.getSymName() << "(";
        FunctionType type = funcOp.getFunctionType();
        for (unsigned i = 0; i < type.getNumInputs(); i++)
          output << getKernelTypeString("input", type.getInput(i)) << " in" << i
                 << ", ";

        for (unsigned i = 0; i < type.getNumResults(); i++) {
          output << getKernelTypeString("output", type.getResult(i)) << " out"
                 << i;
          if (i < type.getNumResults() - 1)
            output << ", ";
          else
            output << ");\n";
        }
      }
    output << "#endif\n\n";
  }

  void writeClass(GraphOp graph) {
    output << "#include <adf.h>\n";
    output << "using namespace adf;\n";
    output << "class " << graph.getName() << " : public graph {\n";
    output << "private:\n";
    int kCnt = 1;
    {
      Indent indent;
      for (Region &region : graph->getRegions())
        for (Block &block : region.getBlocks())
          for (const auto kernel : block.getOps<KernelOp>()) {
            // collect and initialize some kernel info
            std::string varName = "k" + std::to_string(kCnt);
            output << indent << "kernel " << varName << ";\n";
            kernelOp2VarName[kernel] = varName;
            kCnt++;
          }
    }

    output << "\npublic:\n";
    Indent indent;
    for (auto op : graph.getBody()->getOps<GraphInputOp>())
      output << indent << "input_port " << op.getName() << ";\n";
    for (auto op : graph.getBody()->getOps<GraphOutputOp>())
      output << indent << "output_port " << op.getName() << ";\n";
    for (auto op : graph.getBody()->getOps<GraphInOutOp>())
      output << indent << "inout_port " << op.getName() << ";\n";

    output << "\n" << indent << graph.getName() << "() {\n";
    // initialize the kernel instances in the adf c++ graph
    {
      Indent indent;
      for (Region &region : graph->getRegions())
        for (Block &block : region.getBlocks())
          for (auto kernel : block.getOps<KernelOp>()) {
            output << indent << kernelOp2VarName[kernel] << " = kernel::create("
                   << kernel.getCallee().str() << ");\n";
          }
    }

    output << "\n";

    for (Region &region : graph->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (Operation &op : block.getOperations()) {
          if (auto port = dyn_cast<GraphInputOp>(op)) {
            visitOpResultUsers(port);
          } else if (auto graph = dyn_cast<KernelOp>(op)) {
            visitOpResultUsers(graph);
          } else if (auto graph = dyn_cast<GraphOutputOp>(op)) {
            // the graph output should have no users in adf, do nothing here
          }
        } // all op visited
      }
    }

    {
      Indent indent;
      for (Region &region : graph->getRegions())
        for (Block &block : region.getBlocks())
          for (auto kernel : block.getOps<KernelOp>()) {
            output << indent << "source(" << kernelOp2VarName[kernel] << ") = "
                   << "\"kernels.cc\";\n";
            output << indent << "runtime<ratio>(" << kernelOp2VarName[kernel]
                   << ") = "
                   << "0.1;\n";
          }
    }

    output << indent << "}\n";
    output << "};\n\n";
  }
};

LogicalResult AIE::ADFGenerateCPPGraph(ModuleOp module, raw_ostream &output) {
  GraphWriter writer(output);
  resetIndent();

  writer.writeKernelFunctions(module);

  for (Block &block : module.getBodyRegion())
    for (auto graphOp : block.getOps<GraphOp>())
      writer.writeClass(graphOp);
  return success();
}