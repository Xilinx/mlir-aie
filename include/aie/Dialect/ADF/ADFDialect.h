//===- ADFDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XILINX_ADF_DIALECT_H
#define XILINX_ADF_DIALECT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

using namespace mlir;


namespace xilinx {
namespace ADF {
namespace detail {
struct InterfaceTypeStorage;
}
}
}


#include "aie/Dialect/ADF/ADFDialect.h.inc"

//===----------------------------------------------------------------------===//
// ADF Types
//===----------------------------------------------------------------------===//
namespace xilinx {
namespace ADF {

/// This class defines the ADF interface type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (InterfaceType), the base class to use (Type), and the storage class
/// (InterfaceTypeStorage).
class InterfaceType : public mlir::Type::TypeBase<InterfaceType, mlir::Type,
                                               detail::InterfaceTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `InterfaceType` with the given element types.
  /// There *must* be atleast one element type.
  static InterfaceType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this Interface type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this Interface struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
} // namespace ADF
} // namespace xilinx


namespace xilinx {
namespace ADF {

#define GEN_PASS_CLASSES
#include "aie/Dialect/ADF/ADFPasses.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createADFGenerateCppGraphPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/ADF/ADFPasses.h.inc"

} // ADF
} // namespace xilinx


#endif //XILINX_ADF_DIALECT_H
