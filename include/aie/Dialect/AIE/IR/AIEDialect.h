//===- AIEDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIE_DIALECT_H
#define MLIR_AIE_DIALECT_H

#include "AIEEnums.h"

#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "llvm/ADT/StringRef.h"

namespace xilinx::AIE {

// Check that the given DMA-like op (e.g. MemOp, ShimDMAOp)
// has valid BDs.
template <typename ConcreteType>
struct HasValidBDs : mlir::OpTrait::TraitBase<ConcreteType, HasValidBDs> {
  static mlir::LogicalResult verifyTrait(mlir::Operation *op);
};

// Check that the given DMA-like op (e.g. MemOp, ShimDMAOp)
// has valid channels.
template <typename ConcreteType>
struct HasValidDMAChannels
    : mlir::OpTrait::TraitBase<ConcreteType, HasValidBDs> {
  static mlir::LogicalResult verifyTrait(mlir::Operation *op);
};

template <typename ConcreteType>
struct SkipAccessibilityCheckTrait
    : mlir::OpTrait::TraitBase<ConcreteType, SkipAccessibilityCheckTrait> {};

class TileOp;

uint32_t getShimBurstLengthBytes(const AIE::AIETargetModel &tm,
                                 uint32_t burstLength);
uint32_t getShimBurstLengthEncoding(const AIE::AIETargetModel &tm,
                                    uint32_t burstLength);

mlir::LogicalResult
verifyOffsetSizeAndStrideOp(mlir::OffsetSizeAndStrideOpInterface op);

} // namespace xilinx::AIE

/// Include the generated interface declarations.
#include "aie/Dialect/AIE/IR/AIEInterfaces.h.inc"

namespace xilinx::AIE {
mlir::LogicalResult
myVerifyOffsetSizeAndStrideOp(mlir::OffsetSizeAndStrideOpInterface op);
template <typename ConcreteOp>
struct MyOffsetSizeAndStrideOpInterfaceTrait
    : public ::mlir::detail::OffsetSizeAndStrideOpInterfaceTrait<ConcreteOp> {
  static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
    return myVerifyOffsetSizeAndStrideOp(
        ::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>(op));
  }
};

struct MyOffsetSizeAndStrideOpInterface
    : ::mlir::OffsetSizeAndStrideOpInterface {
  template <typename ConcreteOp>
  struct Trait : public MyOffsetSizeAndStrideOpInterfaceTrait<ConcreteOp> {};
};
} // namespace xilinx::AIE

// Include dialect declarations such as parseAttributes, parseType
#include "aie/Dialect/AIE/IR/AIEDialect.h.inc"

namespace xilinx::AIE {

void registerAIETranslations();

} // namespace xilinx::AIE

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/AIE/IR/AIETypes.h.inc"

////////////////////////////////////////////////////////////////////////////////
// Custom Attributes ///////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define GET_ATTRDEF_CLASSES
#include "aie/Dialect/AIE/IR/AIEAttrs.h.inc"

////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace xilinx::AIE {

WireBundle getConnectingBundle(WireBundle dir);

#define GENERATE_TO_STRING(TYPE_WITH_INSERTION_OP)                             \
  friend std::string to_string(const TYPE_WITH_INSERTION_OP &s) {              \
    std::ostringstream ss;                                                     \
    ss << s;                                                                   \
    return ss.str();                                                           \
  }

using Port = struct Port {
  WireBundle bundle;
  int channel;

  bool operator==(const Port &rhs) const {
    return std::tie(bundle, channel) == std::tie(rhs.bundle, rhs.channel);
  }

  bool operator!=(const Port &rhs) const { return !(*this == rhs); }

  bool operator<(const Port &rhs) const {
    return std::tie(bundle, channel) < std::tie(rhs.bundle, rhs.channel);
  }

  friend std::ostream &operator<<(std::ostream &os, const Port &port) {
    os << "(";
    os << stringifyWireBundle(port.bundle).str();
    os << ": " << std::to_string(port.channel) << ")";
    return os;
  }

  GENERATE_TO_STRING(Port)

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Port &port) {
    os << to_string(port);
    return os;
  }
};

using Connect = struct Connect {
  Port src;
  Port dst;

  bool operator==(const Connect &rhs) const {
    return std::tie(src, dst) == std::tie(rhs.src, rhs.dst);
  }

  bool operator!=(const Connect &rhs) const { return !(*this == rhs); }

  bool operator<(const Connect &rhs) const {
    return std::tie(src, dst) < std::tie(rhs.src, rhs.dst);
  }
};

using DMAChannel = struct DMAChannel {
  DMAChannelDir direction;
  int channel;

  bool operator==(const DMAChannel &rhs) const {
    return std::tie(direction, channel) == std::tie(rhs.direction, rhs.channel);
  }
};

const AIETargetModel &getTargetModel(mlir::Operation *op);
const AIETargetModel &getTargetModel(AIEDevice device);

mlir::ParseResult
parseObjectFifoProducerTile(mlir::OpAsmParser &parser,
                            mlir::OpAsmParser::UnresolvedOperand &operand,
                            BDDimLayoutArrayAttr &dimensions);

void printObjectFifoProducerTile(mlir::OpAsmPrinter &printer,
                                 mlir::Operation *op, mlir::Value tile,
                                 BDDimLayoutArrayAttr dimensions);

mlir::ParseResult parseObjectFifoConsumerTiles(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &tiles,
    BDDimLayoutArrayArrayAttr &dimensions);

void printObjectFifoConsumerTiles(mlir::OpAsmPrinter &printer,
                                  mlir::Operation *op, mlir::OperandRange tiles,
                                  BDDimLayoutArrayArrayAttr dimensions);

int32_t getBufferBaseAddress(mlir::Operation *bufOp);

// Trace Event Value Parsing/Printing (handles both string and typed enums)
mlir::ParseResult parseTraceEvent(mlir::AsmParser &parser,
                                  mlir::Attribute &result);
void printTraceEventEnum(mlir::AsmPrinter &printer, mlir::Attribute attr);

} // namespace xilinx::AIE

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "aie/Dialect/AIE/IR/AIEOps.h.inc"

namespace xilinx::AIE {

void collectTiles(DeviceOp &device,
                  llvm::DenseMap<TileID, mlir::Operation *> &tiles);

void collectBuffers(
    DeviceOp &device,
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<BufferOp, 4>> &buffers);
} // namespace xilinx::AIE

namespace llvm {
// Functions hash just like pointers.
template <>
struct DenseMapInfo<xilinx::AIE::ObjectFifoAcquireOp> {
  static xilinx::AIE::ObjectFifoAcquireOp getEmptyKey() {
    auto *pointer = DenseMapInfo<void *>::getEmptyKey();
    return xilinx::AIE::ObjectFifoAcquireOp::getFromOpaquePointer(pointer);
  }

  static xilinx::AIE::ObjectFifoAcquireOp getTombstoneKey() {
    auto *pointer = DenseMapInfo<void *>::getTombstoneKey();
    return xilinx::AIE::ObjectFifoAcquireOp::getFromOpaquePointer(pointer);
  }

  static unsigned getHashValue(xilinx::AIE::ObjectFifoAcquireOp val) {
    return hash_value(val.getAsOpaquePointer());
  }

  static bool isEqual(xilinx::AIE::ObjectFifoAcquireOp lhs,
                      xilinx::AIE::ObjectFifoAcquireOp rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace llvm {
// Functions hash just like pointers.
template <>
struct DenseMapInfo<xilinx::AIE::ObjectFifoCreateOp> {
  static xilinx::AIE::ObjectFifoCreateOp getEmptyKey() {
    auto *pointer = DenseMapInfo<void *>::getEmptyKey();
    return xilinx::AIE::ObjectFifoCreateOp::getFromOpaquePointer(pointer);
  }

  static xilinx::AIE::ObjectFifoCreateOp getTombstoneKey() {
    auto *pointer = DenseMapInfo<void *>::getTombstoneKey();
    return xilinx::AIE::ObjectFifoCreateOp::getFromOpaquePointer(pointer);
  }

  static unsigned getHashValue(xilinx::AIE::ObjectFifoCreateOp val) {
    return hash_value(val.getAsOpaquePointer());
  }

  static bool isEqual(xilinx::AIE::ObjectFifoCreateOp lhs,
                      xilinx::AIE::ObjectFifoCreateOp rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<xilinx::AIE::DMAChannel> {
  using FirstInfo = DenseMapInfo<xilinx::AIE::DMAChannelDir>;
  using SecondInfo = DenseMapInfo<int>;

  static xilinx::AIE::DMAChannel getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static xilinx::AIE::DMAChannel getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const xilinx::AIE::DMAChannel &d) {
    return detail::combineHashValue(FirstInfo::getHashValue(d.direction),
                                    SecondInfo::getHashValue(d.channel));
  }

  static bool isEqual(const xilinx::AIE::DMAChannel &lhs,
                      const xilinx::AIE::DMAChannel &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<xilinx::AIE::Port> {
  using FirstInfo = DenseMapInfo<xilinx::AIE::WireBundle>;
  using SecondInfo = DenseMapInfo<int>;

  static xilinx::AIE::Port getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static xilinx::AIE::Port getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const xilinx::AIE::Port &d) {
    return detail::combineHashValue(FirstInfo::getHashValue(d.bundle),
                                    SecondInfo::getHashValue(d.channel));
  }

  static bool isEqual(const xilinx::AIE::Port &lhs,
                      const xilinx::AIE::Port &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

template <>
struct std::less<xilinx::AIE::Port> {
  bool operator()(const xilinx::AIE::Port &a,
                  const xilinx::AIE::Port &b) const {
    return a.bundle == b.bundle ? a.channel < b.channel : a.bundle < b.bundle;
  }
};

template <>
struct std::hash<xilinx::AIE::Port> {
  std::size_t operator()(const xilinx::AIE::Port &p) const noexcept {
    std::size_t h1 = std::hash<xilinx::AIE::WireBundle>{}(p.bundle);
    std::size_t h2 = std::hash<int>{}(p.channel);
    return h1 ^ h2 << 1;
  }
};

#endif
