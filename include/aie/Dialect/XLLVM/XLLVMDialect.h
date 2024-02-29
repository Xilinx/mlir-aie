//===- XLLVMDialect.h - External LLVM (xllvm) dialect --------------C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the XLLVM dialect, containing LLVM intrinsic operations
// for an external LLVM compiler.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_XLLVM_XLLVMDIALECT_H
#define AIE_DIALECT_XLLVM_XLLVMDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/ThreadLocalCache.h"
#include "mlir/Transforms/Mem2Reg.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#define GET_OP_CLASSES
#include "aie/Dialect/XLLVM/IR/XLLVMOps.h.inc"

#include "aie/Dialect/XLLVM/IR/XLLVMDialect.h.inc"

namespace llvm {

class CallInst;
class IRBuilderBase;
class StringRef;

} // namespace llvm

namespace mlir {

class Operation;

namespace LLVM {
class ModuleTranslation;
} // namespace LLVM

} // namespace mlir

namespace xilinx::xllvm {

llvm::CallInst *createExternalLLVMIntrinsicCall(
    llvm::IRBuilderBase &builder,
    mlir::LLVM::ModuleTranslation &moduleTranslation, mlir::Operation *intrOp,
    llvm::StringRef intrinsicName);

} // namespace xilinx::xllvm

#endif // AIE_DIALECT_XLLVM_XLLVMDIALECT_H
