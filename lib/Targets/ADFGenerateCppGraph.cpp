//===- ADFGenerateCppGraph.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "AIETargets.h"
#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/ADF/ADFOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include <iostream>
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
  Indent(int &indent) : indent(indent) { currentindent += indent; }
  ~Indent() { currentindent -= indent; }
};
static void resetIndent() { currentindent = 0; }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const struct Indent &indent) {
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
    if (const auto &t = type.dyn_cast<int8Type>())
      return t.getMnemonic();
    if (const auto &t = type.dyn_cast<int16Type>())
      return t.getMnemonic();
    if (const auto &t = type.dyn_cast<int32Type>())
      return t.getMnemonic();
    if (const auto &t = type.dyn_cast<int64Type>())
      return t.getMnemonic();
    if (const auto &t = type.dyn_cast<uint8Type>())
      return t.getMnemonic();
    if (const auto &t = type.dyn_cast<uint16Type>())
      return t.getMnemonic();
    if (const auto &t = type.dyn_cast<uint32Type>())
      return t.getMnemonic();
    if (const auto &t = type.dyn_cast<uint64Type>())
      return t.getMnemonic();
    if (const auto &t = type.dyn_cast<floatType>())
      return t.getMnemonic();
    assert(false);
  }
  std::string getKernelTypeString(std::string direction, Type type) {
    if (auto window = type.dyn_cast<WindowType>()) {
      return (direction + "_window_" + getCTypeString(window.getType()) + " *")
          .str();
    } else if (auto stream = type.dyn_cast<StreamType>()) {
      return (direction + "_stream_" + getCTypeString(stream.getType()) + " *")
          .str();
    } else if (auto stream = type.dyn_cast<ParameterType>()) {
      return std::string(getCTypeString(stream.getType()));
    } else {
      assert(false);
    }
  }

  std::string getConnectionTypeString(Type type) {
    if (auto windowType = type.dyn_cast<WindowType>()) {
      return std::string("window<") + std::to_string(windowType.getSize()) +
             "> ";
    } else if (auto windowType = type.dyn_cast<StreamType>())
      return "stream";
    else if (auto windowType = type.dyn_cast<ParameterType>())
      return "parameter";
    else
      assert(false);
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
              driverOp, kernel.calleeAttr());
          Type opType = funcOp.getFunctionType().getInput(targetIndex);
          std::string targetKernelName = kernelOp2VarName[kernel];
          output << indent << "connect<" << getConnectionTypeString(opType)
                 << "> ";
          output << getTempNetName() << " (" << driverOp.name() << ", "
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
              kernel, kernel.calleeAttr());
          Type opType = funcOp.getFunctionType().getInput(targetIndex);
          auto targetKernelName = kernelOp2VarName[kernel];
          output << indent << "connect<" << getConnectionTypeString(opType)
                 << "> ";
          output << getTempNetName() << " (" << sourceKernelName << ".out["
                 << sourceIndex << "], " << targetKernelName << ".in["
                 << targetIndex << "]);\n";
        } else if (auto outputOp = dyn_cast<GraphOutputOp>(userOp)) {
          auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              source, source.calleeAttr());
          Type opType = funcOp.getFunctionType().getInput(sourceIndex);
          output << indent << "connect<" << getConnectionTypeString(opType)
                 << "> ";
          output << getTempNetName() << " (" << sourceKernelName << ".out["
                 << sourceIndex << "], " << outputOp.name() << ");\n";
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

        for (unsigned i = 0; i < type.getNumInputs(); i++) {
          output << getKernelTypeString("input", type.getInput(i)) << " in" << i
                 << ", ";
        }

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

  void writeClass(ADF::GraphOp graph) {
    output << "#include <adf.h>\n";
    output << "using namespace adf;\n";
    output << "class " << graph.name() << " : public graph {\n";
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
      output << indent << "input_port " << op.name() << ";\n";
    for (auto op : graph.getBody()->getOps<GraphOutputOp>())
      output << indent << "output_port " << op.name() << ";\n";
    for (auto op : graph.getBody()->getOps<GraphInOutOp>())
      output << indent << "inout_port " << op.name() << ";\n";

    output << "\n" << indent << graph.name() << "() {\n";
    // initialize the kernel instances in the adf c++ graph
    {
      Indent indent;
      for (Region &region : graph->getRegions())
        for (Block &block : region.getBlocks())
          for (auto kernel : block.getOps<KernelOp>()) {
            output << indent << kernelOp2VarName[kernel] << " = kernel::create("
                   << kernel.callee().str() << ");\n";
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
            ;
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

mlir::LogicalResult xilinx::AIE::ADFGenerateCPPGraph(ModuleOp module,
                                                     raw_ostream &output) {
  GraphWriter writer(output);
  resetIndent();

  writer.writeKernelFunctions(module);

  for (Block &block : module.getBodyRegion())
    for (auto graphOp : block.getOps<GraphOp>()) {
      writer.writeClass(graphOp);
    }
  return mlir::success();
}

// };

// std::unique_ptr<OperationPass<ModuleOp>>
// xilinx::ADF::createADFGenerateCppGraphPass() {
//   return std::make_unique<ADFGenerateCppGraphPass>();
// }