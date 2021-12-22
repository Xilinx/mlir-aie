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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace xilinx;
using namespace ADF;

//===----------------------------------------------------------------------===//
// ADF Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/ADF/ADFTypes.cpp.inc"

mlir::Type ADFDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef typeTag;
  if (failed(parser.parseKeyword(&typeTag)))
    return Type();

  {
    Type genType;
    auto parseResult =
        generatedTypeParser(getContext(), parser, typeTag, genType);
    if (parseResult.hasValue())
      return genType;
  }

  parser.emitError(parser.getNameLoc()) << "unknown type!";
  return Type();
}

/// Print an instance of a type registered to the toy dialect.
void ADFDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
}

Type InterfaceType::parse(MLIRContext *ctxt, DialectAsmParser &parser) {
  Type oneType;
  if (parser.parseLess() || parser.parseType(oneType) || parser.parseGreater())
    return Type();

  return get(ctxt, oneType);
}
void InterfaceType::print(DialectAsmPrinter &printer) const {
  printer << "interface<" << getType() << ">";
}
Type WindowType::parse(MLIRContext *ctxt, DialectAsmParser &parser) {
  Type oneType;
  int size;
  int overlap;
  if (parser.parseLess() || parser.parseType(oneType) || parser.parseComma() ||
      parser.parseInteger(size) || parser.parseComma() ||
      parser.parseInteger(overlap) || parser.parseGreater())
    return Type();

  return get(ctxt, oneType, size, overlap);
}
void WindowType::print(DialectAsmPrinter &printer) const {
  printer << "window<" << getType() << ", " << getSize() << ", " << getOverlap()
          << ">";
}
Type StreamType::parse(MLIRContext *ctxt, DialectAsmParser &parser) {
  Type oneType;
  if (parser.parseLess() || parser.parseType(oneType) || parser.parseGreater())
    return Type();

  return get(ctxt, oneType);
}
void StreamType::print(DialectAsmPrinter &printer) const {
  printer << "stream<" << getType() << ">";
}
Type ParameterType::parse(MLIRContext *ctxt, DialectAsmParser &parser) {
  Type oneType;
  if (parser.parseLess() || parser.parseType(oneType) || parser.parseGreater())
    return Type();

  return get(ctxt, oneType);
}
void ParameterType::print(DialectAsmPrinter &printer) const {
  printer << "interface<" << getType() << ">";
}

//===----------------------------------------------------------------------===//
// ADF Dialect
//===----------------------------------------------------------------------===//
void ADFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/ADF/ADF.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "aie/Dialect/ADF/ADFTypes.cpp.inc"
      >();
}

#include "aie/Dialect/ADF/ADFDialect.cpp.inc"
