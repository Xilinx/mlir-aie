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

#include "aie/Dialect/AIE/IR/AIEEnums.h.inc"

namespace mlir {
namespace OpTrait {
template <typename ConcreteType>
class FlowEndPoint : public OpTrait::TraitBase<ConcreteType, FlowEndPoint> {};
} // namespace OpTrait
} // namespace mlir

/// Include the generated interface declarations.
#include "aie/Dialect/AIE/IR/AIEInterfaces.h.inc"

namespace xilinx {
namespace AIE {

void registerAIETranslations();

// FIXME: use this
//#include "AIEDialect.h.inc"

// The Dialect
class AIEDialect : public mlir::Dialect {
public:
  explicit AIEDialect(mlir::MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "AIE"; }

  /// Parse a type registered to this dialect. Overridding this method is
  /// required for dialects that have custom types.
  /// Technically this is only needed to be able to round-trip to textual IR.
  mlir::Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect. Overridding this method is
  /// only required for dialects that have custom types.
  /// Technically this is only needed to be able to round-trip to textual IR.
  void printType(mlir::Type type, DialectAsmPrinter &os) const override;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

// namespace detail {
// struct AIEListTypeStorage;
// }

// /// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
// enum AIETypeKind {
//   // The enum starts at the range reserved for this dialect.
//   AIE_TYPE = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
//   AIE_LIST,
// };

// /// Type for Toy arrays.
// /// In MLIR Types are reference to immutable and uniqued objects owned by the
// /// MLIRContext. As such `AIEListType` only wraps a pointer to an uniqued
// /// instance of `AIEListTypeStorage` (defined in our implementation file) and
// /// provides the public facade API to interact with the type.
// class AIEListType : public mlir::Type::TypeBase<AIEListType, mlir::Type,
//                                                  detail::AIEListTypeStorage>
//                                                  {
// public:
//   using Base::Base;

//   /// Return the type of individual elements in the array.
//   mlir::Type getElementType();

//   /// Get the unique instance of this Type from the context.
//   static AIEListType get(mlir::Type elementType);

//   /// Support method to enable LLVM-style RTTI type casting.
//   static bool kindof(unsigned kind) { return kind == AIETypeKind::AIE_LIST; }
// };

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

////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

//#include "AIEOpInterfaces.h.inc"

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
std::unique_ptr<OperationPass<ModuleOp>> createAIEFindFlowsPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIELocalizeLocksPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIENormalizeAddressSpacesPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIERouteFlowsPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIERoutePacketFlowsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createAIEVectorOptPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIEPathfinderPass();
std::unique_ptr<OperationPass<DeviceOp>>
createAIEObjectFifoStatefulTransformPass();
std::unique_ptr<OperationPass<DeviceOp>> createAIEObjectFifoLoopUnrollPass();
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
