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

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
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

#include "AIEEnums.h"

namespace xilinx {
namespace AIE {
// template <typename ConcreteType>
// class FlowEndPoint : public OpTrait::TraitBase<ConcreteType, FlowEndPoint>
// {};

// Check that the given DMA-like op (e.g. MemOp, ShimDMAOp)
// has valid BDs.
template <typename ConcreteType>
struct HasValidBDs : public OpTrait::TraitBase<ConcreteType, HasValidBDs> {
  static LogicalResult verifyTrait(Operation *op);
};

// Check that the given DMA-like op (e.g. MemOp, ShimDMAOp)
// has valid channels.
template <typename ConcreteType>
struct HasValidDMAChannels
    : public OpTrait::TraitBase<ConcreteType, HasValidBDs> {
  static LogicalResult verifyTrait(Operation *op);
};

class TileOp;
} // namespace AIE
} // namespace xilinx

/// Include the generated interface declarations.
#include "aie/Dialect/AIE/IR/AIEInterfaces.h.inc"

// Include dialect declarations such as parseAttributes, parseType
#include "aie/Dialect/AIE/IR/AIEDialect.h.inc"

namespace xilinx {
namespace AIE {

void registerAIETranslations();

} // namespace AIE
} // namespace xilinx

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Include generated TableGen-generated type definitions
#define GET_TYPEDEF_CLASSES 1
#include "aie/Dialect/AIE/IR/AIETypes.h.inc"

namespace xilinx {
namespace AIE {
namespace detail {
struct AIEObjectFifoTypeStorage;
}

/// This class defines the AIE ObjectFifo type.
class AIEObjectFifoType
    : public mlir::Type::TypeBase<AIEObjectFifoType, mlir::Type,
                                  detail::AIEObjectFifoTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `ObjectFifoType` with the given element type.
  static AIEObjectFifoType get(mlir::Type elementType);

  /// This method is used to verify the construction invariants.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              mlir::Type elementType);

  /// Returns the element type of this ObjectFifoType.
  mlir::Type getElementType();
};

namespace detail {
struct AIEObjectFifoSubviewTypeStorage;
}

/// This class defines the AIE ObjectFifoSubview type.
class AIEObjectFifoSubviewType
    : public mlir::Type::TypeBase<AIEObjectFifoSubviewType, mlir::Type,
                                  detail::AIEObjectFifoSubviewTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `SubviewType` with the given element type.
  static AIEObjectFifoSubviewType get(mlir::Type elementType);

  /// This method is used to verify the construction invariants.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              mlir::Type elementType);

  /// Returns the element type of this SubviewType.
  mlir::Type getElementType();
};

} // namespace AIE
} // namespace xilinx

////////////////////////////////////////////////////////////////////////////////
// Custom Attributes ///////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define GET_ATTRDEF_CLASSES
#include "aie/Dialect/AIE/IR/AIEAttrDefs.h.inc"

////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace xilinx {
namespace AIE {

// #include "AIEOpInterfaces.h.inc"

typedef std::pair<WireBundle, int> Port;
typedef std::pair<Port, Port> Connect;
typedef std::pair<DMAChannelDir, int> DMAChannel;

struct AIEArchDesc {
  bool checkerboard;
};

// xcve2302 17x2, xcvc1902 50x8
struct AIEDevDesc {
  unsigned int rows;
  unsigned int cols;
  AIEArchDesc arch;
};

const xilinx::AIE::AIETargetModel &getTargetModel(Operation *op);

::mlir::ParseResult
parseObjectFifoProducerTile(::mlir::OpAsmParser &parser,
                            ::mlir::OpAsmParser::UnresolvedOperand &operand,
                            DimTupleArrayAttr &dimensions);
void printObjectFifoProducerTile(::mlir::OpAsmPrinter &_odsPrinter,
                                 Operation *op, Value tile,
                                 Attribute dimensions);

::mlir::ParseResult parseObjectFifoConsumerTiles(
    ::mlir::OpAsmParser &parser,
    SmallVectorImpl<::mlir::OpAsmParser::UnresolvedOperand> &tiles,
    DimTupleArrayArrayAttr &dimensions);
void printObjectFifoConsumerTiles(::mlir::OpAsmPrinter &_odsPrinter,
                                  Operation *op, OperandRange tiles,
                                  Attribute dimensions);

} // namespace AIE
} // namespace xilinx

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "aie/Dialect/AIE/IR/AIE.h.inc"

namespace xilinx {
namespace AIE {

#define GEN_PASS_CLASSES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

std::unique_ptr<OperationPass<DeviceOp>> createAIEAssignBufferAddressesPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIEAssignLockIDsPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIECanonicalizeDevicePass();
std::unique_ptr<OperationPass<ModuleOp>> createAIECoreToStandardPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIEFindFlowsPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIELocalizeLocksPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIENormalizeAddressSpacesPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIERouteFlowsPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIERoutePacketFlowsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createAIEVectorOptPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIEPathfinderPass();
std::unique_ptr<OperationPass<DeviceOp>>
createAIEObjectFifoStatefulTransformPass();
std::unique_ptr<OperationPass<DeviceOp>>
createAIEObjectFifoRegisterProcessPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

} // namespace AIE
} // namespace xilinx

namespace llvm {
// Functions hash just like pointers.
template <> struct DenseMapInfo<xilinx::AIE::ObjectFifoAcquireOp> {
  static xilinx::AIE::ObjectFifoAcquireOp getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return xilinx::AIE::ObjectFifoAcquireOp::getFromOpaquePointer(pointer);
  }
  static xilinx::AIE::ObjectFifoAcquireOp getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
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
template <> struct DenseMapInfo<xilinx::AIE::ObjectFifoCreateOp> {
  static xilinx::AIE::ObjectFifoCreateOp getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return xilinx::AIE::ObjectFifoCreateOp::getFromOpaquePointer(pointer);
  }
  static xilinx::AIE::ObjectFifoCreateOp getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
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
} // namespace llvm

#endif
