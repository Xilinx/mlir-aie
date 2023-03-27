//===- AIEDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIEX_DIALECT_H
#define MLIR_AIEX_DIALECT_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <set>

using namespace mlir;

namespace xilinx {
namespace AIEX {

// The Dialect
class AIEXDialect : public mlir::Dialect {
public:
  explicit AIEXDialect(mlir::MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "AIEX"; }
};

} // namespace AIEX
} // namespace xilinx

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "aie/Dialect/AIEX/IR/AIEX.h.inc"

namespace xilinx {
namespace AIEX {

#define GEN_PASS_CLASSES
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"

std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIECreateCoresPass();
std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIECreateLocksPass();
std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIEHerdRoutingPass();
std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIELowerMemcpyPass();
std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIELowerMulticastPass();
std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIEBroadcastPacketPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"

} // namespace AIEX
} // namespace xilinx

#endif
