//===- StreamTypes.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_STREAM_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_STREAM_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// IStreamType
//===----------------------------------------------------------------------===//

def IStreamType : Physical_Type<"IStream", "istream"> {
  let summary = "A type specifiying a streaming endpoint for input";
  let description = [{
    A stream endpoint that can be used for reading data from a stream
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class IStreamTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<xilinx::phy::physical::IStreamType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<xilinx::phy::physical::IStreamType>().getDatatype()",
      element.predicate>
  ]>,
  "a reading endpoint (istream) of a stream of " # element.summary
>;

def AnyIStreamType : Type<
  Or<[
    IStreamTypeOf<AnyType>.predicate,
  ]>,
  "a reading endpoint (istream) of a stream"
>;

//===----------------------------------------------------------------------===//
// OStreamType
//===----------------------------------------------------------------------===//

def OStreamType : Physical_Type<"OStream", "ostream"> {
  let summary = "A type specifiying a streaming endpoint for output";
  let description = [{
    A stream endpoint that can be used for writing data to a stream
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class OStreamTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<xilinx::phy::physical::OStreamType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<xilinx::phy::physical::OStreamType>().getDatatype()",
      element.predicate>
  ]>,
  "a writing endpoint (ostream) of a stream of " # element.summary
>;

def AnyOStreamType : Type<
  Or<[
    OStreamTypeOf<AnyType>.predicate,
  ]>,
  "a writing endpoint (ostream) of a stream"
>;

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def AnyStreamEndpointType : Type<
  Or<[
    AnyIStreamType.predicate,
    AnyOStreamType.predicate,
  ]>,
  "a stream endpoint"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_STREAM_TYPES_TD
