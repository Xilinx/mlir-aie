//===- AIETargetCppTxn.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This file implements translation from dynamic runtime sequences to templated
// C++ code for runtime transaction generation.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

/// Helper class to generate C++ code from MLIR operations
class CppTxnEmitter {
public:
  CppTxnEmitter(llvm::raw_ostream &os, int indentLevel = 0)
      : os(os), indentLevel(indentLevel), varCounter(0) {}

  /// Emit C++ code for a runtime sequence
  LogicalResult emitRuntimeSequence(AIE::RuntimeSequenceOp seqOp);

private:
  llvm::raw_ostream &os;
  int indentLevel;
  int varCounter;

  // Maps MLIR values to C++ variable names
  DenseMap<Value, std::string> valueNames;

  // Track function parameters from runtime sequence arguments
  std::vector<std::string> paramNames;
  std::vector<std::string> paramTypes;

  /// Emit indentation
  void indent() {
    for (int i = 0; i < indentLevel; ++i)
      os << "  ";
  }

  /// Get or create a C++ variable name for an MLIR value
  std::string getOrCreateValueName(Value val);

  /// Emit a value (either constant or variable reference)
  void emitValue(Value val);

  /// Emit type name for C++ template parameter
  std::string emitTypeName(Type type);

  /// Emit instruction encoding helpers
  void emitInstructionHelpers();

  /// Emit operation-specific code
  LogicalResult emitOp(Operation *op);

  /// Emit dynamic NPU operations
  LogicalResult emitNpuDynWrite32(AIEX::NpuDynWrite32Op op);
  LogicalResult emitNpuDynMaskWrite32(AIEX::NpuDynMaskWrite32Op op);
  LogicalResult emitNpuDynDmaMemcpyNd(AIEX::NpuDynDmaMemcpyNdOp op);
  LogicalResult emitNpuDynSync(AIEX::NpuDynSyncOp op);

  /// Emit static NPU operations (for mixed static/dynamic sequences)
  LogicalResult emitNpuWrite32(AIEX::NpuWrite32Op op);
  LogicalResult emitNpuSync(AIEX::NpuSyncOp op);

  /// Emit control flow
  LogicalResult emitScfFor(scf::ForOp op);
  LogicalResult emitScfIf(scf::IfOp op);

  /// Emit arithmetic operations
  LogicalResult emitArithOp(Operation *op);
};

std::string CppTxnEmitter::getOrCreateValueName(Value val) {
  auto it = valueNames.find(val);
  if (it != valueNames.end())
    return it->second;

  // Check if this is a block argument (function parameter)
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    unsigned argNum = blockArg.getArgNumber();
    if (argNum < paramNames.size()) {
      valueNames[val] = paramNames[argNum];
      return paramNames[argNum];
    }
  }

  // Create a new variable name
  std::string name = "v" + std::to_string(varCounter++);
  valueNames[val] = name;
  return name;
}

void CppTxnEmitter::emitValue(Value val) {
  // Check if this is a constant
  if (auto defOp = val.getDefiningOp()) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        os << intAttr.getInt();
        return;
      }
      if (auto indexAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        os << indexAttr.getInt();
        return;
      }
    }
  }

  // Otherwise emit the variable name
  os << getOrCreateValueName(val);
}

std::string CppTxnEmitter::emitTypeName(Type type) {
  if (type.isIndex())
    return "size_t";
  if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width <= 32)
      return "uint32_t";
    else
      return "uint64_t";
  }
  return "auto";
}

void CppTxnEmitter::emitInstructionHelpers() {
  os << "// Transaction opcode definitions\n";
  os << "enum XAie_TxnOpcode {\n";
  os << "  XAIE_IO_WRITE = 0,\n";
  os << "  XAIE_IO_BLOCKWRITE = 1,\n";
  os << "  XAIE_IO_BLOCKSET = 2,\n";
  os << "  XAIE_IO_MASKWRITE = 3,\n";
  os << "  XAIE_IO_MASKPOLL = 4,\n";
  os << "  XAIE_IO_NOOP = 5,\n";
  os << "  XAIE_IO_PREEMPT = 6,\n";
  os << "  XAIE_IO_MASKPOLL_BUSY = 7,\n";
  os << "  XAIE_IO_LOADPDI = 8,\n";
  os << "  XAIE_IO_CUSTOM_OP_TCT = 128,\n";
  os << "  XAIE_IO_CUSTOM_OP_DDR_PATCH = 129,\n";
  os << "};\n\n";

  os << "// Helper to append instruction words\n";
  os << "inline void append_words(std::vector<uint32_t>& txn, "
     << "std::initializer_list<uint32_t> words) {\n";
  os << "  txn.insert(txn.end(), words.begin(), words.end());\n";
  os << "}\n\n";

  os << "// Helper to prepend transaction header\n";
  os << "inline void prepend_header(std::vector<uint32_t>& txn, "
     << "uint32_t numRows = 6, uint32_t numCols = 5, uint32_t devGen = 4) {\n";
  os << "  uint32_t numOps = (txn.size() - 4) / 4; // Rough estimate\n";
  os << "  uint32_t totalSize = txn.size() * sizeof(uint32_t);\n";
  os << "  std::vector<uint32_t> header = {\n";
  os << "    (numRows << 24) | (devGen << 16) | (0 << 8) | 1,\n";
  os << "    (0 << 8) | numCols,\n";
  os << "    numOps,\n";
  os << "    totalSize\n";
  os << "  };\n";
  os << "  txn.insert(txn.begin(), header.begin(), header.end());\n";
  os << "}\n\n";
}

LogicalResult CppTxnEmitter::emitNpuDynWrite32(AIEX::NpuDynWrite32Op op) {
  indent();
  os << "// NPU write32: ";
  emitValue(op.getAddress());
  os << " = ";
  emitValue(op.getValue());
  os << "\n";

  indent();
  os << "append_words(txn, {\n";
  indent();
  os << "  XAIE_IO_WRITE,\n";
  indent();
  os << "  0, // Reserved\n";
  indent();
  os << "  static_cast<uint32_t>(";
  emitValue(op.getAddress());
  os << "),\n";
  indent();
  os << "  0, // Reg offset\n";
  indent();
  os << "  static_cast<uint32_t>(";
  emitValue(op.getValue());
  os << "),\n";
  indent();
  os << "  6 * sizeof(uint32_t) // Op size\n";
  indent();
  os << "});\n";

  return success();
}

LogicalResult CppTxnEmitter::emitNpuDynMaskWrite32(AIEX::NpuDynMaskWrite32Op op) {
  indent();
  os << "// NPU maskwrite32\n";

  indent();
  os << "append_words(txn, {\n";
  indent();
  os << "  XAIE_IO_MASKWRITE,\n";
  indent();
  os << "  0,\n";
  indent();
  os << "  static_cast<uint32_t>(";
  emitValue(op.getAddress());
  os << "),\n";
  indent();
  os << "  0,\n";
  indent();
  os << "  static_cast<uint32_t>(";
  emitValue(op.getValue());
  os << "),\n";
  indent();
  os << "  static_cast<uint32_t>(";
  emitValue(op.getMask());
  os << "),\n";
  indent();
  os << "  7 * sizeof(uint32_t)\n";
  indent();
  os << "});\n";

  return success();
}

LogicalResult CppTxnEmitter::emitNpuDynSync(AIEX::NpuDynSyncOp op) {
  indent();
  os << "// NPU sync (task completion token)\n";

  indent();
  os << "{\n";
  indentLevel++;

  indent();
  os << "uint32_t word2 = (static_cast<uint32_t>(";
  emitValue(op.getDirection());
  os << ") & 0xff) |\n";
  indent();
  os << "                 ((static_cast<uint32_t>(";
  emitValue(op.getRow());
  os << ") & 0xff) << 8) |\n";
  indent();
  os << "                 ((static_cast<uint32_t>(";
  emitValue(op.getColumn());
  os << ") & 0xff) << 16);\n";

  indent();
  os << "uint32_t word3 = ((static_cast<uint32_t>(";
  emitValue(op.getRowNum());
  os << ") & 0xff) << 8) |\n";
  indent();
  os << "                 ((static_cast<uint32_t>(";
  emitValue(op.getColumnNum());
  os << ") & 0xff) << 16) |\n";
  indent();
  os << "                 ((static_cast<uint32_t>(";
  emitValue(op.getChannel());
  os << ") & 0xff) << 24);\n";

  indent();
  os << "append_words(txn, {XAIE_IO_CUSTOM_OP_TCT, "
     << "4 * sizeof(uint32_t), word2, word3});\n";

  indentLevel--;
  indent();
  os << "}\n";

  return success();
}

LogicalResult CppTxnEmitter::emitNpuDynDmaMemcpyNd(AIEX::NpuDynDmaMemcpyNdOp op) {
  indent();
  os << "// TODO: Dynamic DMA memcpy N-D - requires complex BD programming\n";
  indent();
  os << "// This would generate BD configuration code based on dynamic sizes/strides\n";
  return success();
}

LogicalResult CppTxnEmitter::emitNpuWrite32(AIEX::NpuWrite32Op op) {
  indent();
  os << "// NPU write32 (static): " << op.getAddress() << " = " << op.getValue() << "\n";

  uint32_t addr = op.getAddress();
  if (auto absAddr = op.getAbsoluteAddress())
    addr = *absAddr;

  indent();
  os << "append_words(txn, {XAIE_IO_WRITE, 0, "
     << addr << ", 0, " << op.getValue() << ", 6 * sizeof(uint32_t)});\n";

  return success();
}

LogicalResult CppTxnEmitter::emitNpuSync(AIEX::NpuSyncOp op) {
  indent();
  os << "// NPU sync (static)\n";

  uint32_t word2 = (static_cast<uint32_t>(op.getDirection()) & 0xff) |
                   ((op.getRow() & 0xff) << 8) |
                   ((op.getColumn() & 0xff) << 16);

  uint32_t word3 = ((op.getRowNum() & 0xff) << 8) |
                   ((op.getColumnNum() & 0xff) << 16) |
                   ((op.getChannel() & 0xff) << 24);

  indent();
  os << "append_words(txn, {XAIE_IO_CUSTOM_OP_TCT, "
     << "4 * sizeof(uint32_t), " << word2 << ", " << word3 << "});\n";

  return success();
}

LogicalResult CppTxnEmitter::emitScfFor(scf::ForOp op) {
  indent();
  os << "for (";

  std::string inductionVar = getOrCreateValueName(op.getInductionVar());
  os << "auto " << inductionVar << " = ";
  emitValue(op.getLowerBound());
  os << "; " << inductionVar << " < ";
  emitValue(op.getUpperBound());
  os << "; " << inductionVar << " += ";
  emitValue(op.getStep());
  os << ") {\n";

  indentLevel++;

  // Emit loop body
  for (auto &bodyOp : op.getBody()->getOperations()) {
    if (failed(emitOp(&bodyOp)))
      return failure();
  }

  indentLevel--;
  indent();
  os << "}\n";

  return success();
}

LogicalResult CppTxnEmitter::emitScfIf(scf::IfOp op) {
  indent();
  os << "if (";
  emitValue(op.getCondition());
  os << ") {\n";

  indentLevel++;
  for (auto &thenOp : op.getThenRegion().front().getOperations()) {
    if (failed(emitOp(&thenOp)))
      return failure();
  }
  indentLevel--;

  if (!op.getElseRegion().empty()) {
    indent();
    os << "} else {\n";
    indentLevel++;
    for (auto &elseOp : op.getElseRegion().front().getOperations()) {
      if (failed(emitOp(&elseOp)))
        return failure();
    }
    indentLevel--;
  }

  indent();
  os << "}\n";

  return success();
}

LogicalResult CppTxnEmitter::emitArithOp(Operation *op) {
  std::string resultName;
  if (op->getNumResults() > 0) {
    resultName = getOrCreateValueName(op->getResult(0));
    indent();
    os << "auto " << resultName << " = ";
  }

  llvm::TypeSwitch<Operation *>(op)
    .Case<arith::AddIOp>([&](auto addOp) {
      emitValue(addOp.getLhs());
      os << " + ";
      emitValue(addOp.getRhs());
    })
    .Case<arith::SubIOp>([&](auto subOp) {
      emitValue(subOp.getLhs());
      os << " - ";
      emitValue(subOp.getRhs());
    })
    .Case<arith::MulIOp>([&](auto mulOp) {
      emitValue(mulOp.getLhs());
      os << " * ";
      emitValue(mulOp.getRhs());
    })
    .Case<arith::DivUIOp>([&](auto divOp) {
      emitValue(divOp.getLhs());
      os << " / ";
      emitValue(divOp.getRhs());
    })
    .Case<arith::ConstantOp>([&](auto constOp) {
      // Constants are handled inline in emitValue
      return;
    })
    .Default([&](Operation *) {
      os << "/* unsupported arith op */";
    });

  if (op->getNumResults() > 0 && !isa<arith::ConstantOp>(op))
    os << ";\n";

  return success();
}

LogicalResult CppTxnEmitter::emitOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
    // Dynamic NPU operations
    .Case<AIEX::NpuDynWrite32Op>([&](auto dynWriteOp) {
      return emitNpuDynWrite32(dynWriteOp);
    })
    .Case<AIEX::NpuDynMaskWrite32Op>([&](auto dynMaskWriteOp) {
      return emitNpuDynMaskWrite32(dynMaskWriteOp);
    })
    .Case<AIEX::NpuDynSyncOp>([&](auto dynSyncOp) {
      return emitNpuDynSync(dynSyncOp);
    })
    .Case<AIEX::NpuDynDmaMemcpyNdOp>([&](auto dynDmaOp) {
      return emitNpuDynDmaMemcpyNd(dynDmaOp);
    })
    // Static NPU operations (for mixed sequences)
    .Case<AIEX::NpuWrite32Op>([&](auto writeOp) {
      return emitNpuWrite32(writeOp);
    })
    .Case<AIEX::NpuSyncOp>([&](auto syncOp) {
      return emitNpuSync(syncOp);
    })
    // Control flow
    .Case<scf::ForOp>([&](auto forOp) {
      return emitScfFor(forOp);
    })
    .Case<scf::IfOp>([&](auto ifOp) {
      return emitScfIf(ifOp);
    })
    // Arithmetic
    .Case<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivUIOp,
          arith::ConstantOp>([&](auto arithOp) {
      return emitArithOp(arithOp);
    })
    // Terminators
    .Case<scf::YieldOp>([&](auto) { return success(); })
    .Default([&](Operation *unknownOp) {
      indent();
      os << "// Unsupported operation: " << unknownOp->getName().getStringRef() << "\n";
      return success();
    });
}

LogicalResult CppTxnEmitter::emitRuntimeSequence(AIE::RuntimeSequenceOp seqOp) {
  // Extract function signature from runtime sequence arguments
  Block &entryBlock = seqOp.getBody().front();

  for (auto arg : entryBlock.getArguments()) {
    std::string paramName = "arg" + std::to_string(arg.getArgNumber());
    paramNames.push_back(paramName);
    paramTypes.push_back(emitTypeName(arg.getType()));
    valueNames[arg] = paramName;
  }

  // Emit file header
  os << "// Auto-generated C++ transaction sequence\n";
  os << "// Generated from MLIR runtime sequence\n\n";

  os << "#include <cstdint>\n";
  os << "#include <vector>\n";
  os << "#include <initializer_list>\n\n";

  os << "namespace aie_runtime {\n\n";

  // Emit helper functions
  emitInstructionHelpers();

  // Emit main transaction generation function
  std::string seqName = seqOp.getSymName().str();
  os << "std::vector<uint32_t> generate_txn_" << seqName << "(";

  // Emit parameters
  for (size_t i = 0; i < paramNames.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << paramTypes[i] << " " << paramNames[i];
  }
  os << ") {\n";

  indentLevel++;
  indent();
  os << "std::vector<uint32_t> txn;\n";
  indent();
  os << "txn.reserve(1024); // Pre-allocate\n\n";

  // Emit body operations
  for (auto &op : entryBlock.getOperations()) {
    if (failed(emitOp(&op)))
      return failure();
  }

  os << "\n";
  indent();
  os << "// Prepend transaction header\n";
  indent();
  os << "prepend_header(txn);\n\n";

  indent();
  os << "return txn;\n";
  indentLevel--;
  os << "}\n\n";

  os << "} // namespace aie_runtime\n";

  return success();
}

} // anonymous namespace

namespace xilinx {
namespace AIE {

LogicalResult AIETranslateToCppTxn(ModuleOp module, llvm::raw_ostream &output) {
  // Find runtime sequence operations
  auto runtimeSeqs = module.getOps<AIE::RuntimeSequenceOp>();

  if (runtimeSeqs.empty()) {
    return module.emitError("No runtime sequences found in module");
  }

  CppTxnEmitter emitter(output);

  for (auto seqOp : runtimeSeqs) {
    if (failed(emitter.emitRuntimeSequence(seqOp)))
      return failure();
  }

  return success();
}

} // namespace AIE
} // namespace xilinx
