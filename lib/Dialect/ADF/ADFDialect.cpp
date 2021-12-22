//===- ADFDialect.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
//


#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/ADF/ADFOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"


#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"


using namespace xilinx;
using namespace ADF;


//===----------------------------------------------------------------------===//
// ADF Types
//===----------------------------------------------------------------------===//

namespace xilinx {
namespace ADF {
namespace detail {
struct InterfaceTypeStorage : public mlir::TypeStorage {

  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  InterfaceTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}


  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static InterfaceTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<InterfaceTypeStorage>())
        InterfaceTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the interface.
  llvm::ArrayRef<mlir::Type> elementTypes;
};
} // namespace detail
} // namespace ADF
} // namespace xilinx

/// Create an instance of a `InterfaceType` with the given element types. 
InterfaceType InterfaceType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after the context are forwarded to the storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

/// Returns the element types of this interface type.
llvm::ArrayRef<mlir::Type> InterfaceType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}

/// Parse an instance of a type registered to the toy dialect.
mlir::Type ADFDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("itf") || parser.parseLess())
    return mlir::Type();

  // Parse the element types of the struct.
  mlir::SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is an IntegerType.
    if (!elementType.isa<mlir::IntegerType>()) {
      parser.emitError(typeLoc, "element type for a ADF interface must "
                                "be an IntegerType, got: ")
          << elementType;
      return mlir::Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return mlir::Type();
  return InterfaceType::get(elementTypes);
}

/// Print an instance of a type registered to the toy dialect.
void ADFDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // Currently the only ADF type is a interface type.
  InterfaceType interfaceType = type.cast<InterfaceType>();

  // Print the interface type according to the parser format.
  printer << "itf<";
  llvm::interleaveComma(interfaceType.getElementTypes(), printer);
  printer << '>';
}


//===----------------------------------------------------------------------===//
// ADF Dialect
//===----------------------------------------------------------------------===//
void ADFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/ADF/ADF.cpp.inc"
    >();
  addTypes<InterfaceType>();
}


#include "aie/Dialect/ADF/ADFDialect.cpp.inc"