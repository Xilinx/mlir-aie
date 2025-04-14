//===- AIETransformBfp16Type.cpp --------------------------------------*-
// C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "transform-bfp16-type"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

using namespace mlir;

class Bfp16ToI8TypeConverter : public mlir::TypeConverter {
public:
  Bfp16ToI8TypeConverter(const AIETargetModel &targetModel) {
    addTypeAttributeConversion([&](Type type, Attribute attr) {
      return AttributeConversionResult(TypeAttr::get(convertType(type)));
    });

    // Leave other types unchanged
    // Note that the most recently added conversions will be invoked first and
    // therefore this is just a default option that has to be put first
    addConversion([](Type type) { return type; });

    // Add a conversion for bfp16Type to an integer type
    addConversion([&](bfp16Type bfp16Type) -> IntegerType {
      if (targetModel.getBfpMantissaSizeInBits() == 0) {
        llvm::errs()
            << "Block Floating Point is unsupported in the specified model\n";
        return mlir::IntegerType::get(bfp16Type.getContext(), 0);
      }

      if (!targetModel.checkBfpBlockSize(bfp16Type.getBlockSize())) {
        llvm::errs() << "Block size " << bfp16Type.getBlockSize()
                     << " is not supported in the specified model\n";
        return mlir::IntegerType::get(bfp16Type.getContext(), 0);
      }

      return mlir::IntegerType::get(
          bfp16Type.getContext(),
          bfp16Type.getBlockSize() * targetModel.getBfpMantissaSizeInBits() +
              targetModel.getBfpExponentSizeInBits());
    });

    // Add a conversion for MemRefType
    addConversion([&](MemRefType memRefType) {
      auto newElementType = convertType(memRefType.getElementType());
      return MemRefType::get(memRefType.getShape(), newElementType,
                             memRefType.getLayout(),
                             memRefType.getMemorySpace());
    });

    // Add a conversion for ObjectFifoType
    addConversion([&](AIEObjectFifoType objectFifoType) {
      auto newElementType = convertType(objectFifoType.getElementType());
      if (!newElementType) {
        llvm::errs() << "Failed to convert ObjectFifoType element type\n";
        return objectFifoType;
      }

      if (auto newMemRef = dyn_cast<MemRefType>(newElementType))
        return AIEObjectFifoType::get(objectFifoType.getContext(), newMemRef);

      llvm::errs() << "ObjectFifoType element type is not a MemRefType\n";
      return objectFifoType;
    });

    // Add a conversion for ObjectFifoSubviewType
    addConversion([&](AIEObjectFifoSubviewType objectFifoSubviewType) {
      auto newElementType = convertType(objectFifoSubviewType.getElementType());
      if (!newElementType) {
        llvm::errs()
            << "Failed to convert ObjectFifoSubviewType element type\n";
        return objectFifoSubviewType;
      }

      if (auto newMemRef = dyn_cast<MemRefType>(newElementType))
        return AIEObjectFifoSubviewType::get(objectFifoSubviewType.getContext(),
                                             newMemRef);

      llvm::errs()
          << "ObjectFifoSubviewType element type is not a MemRefType\n";
      return objectFifoSubviewType;
    });

    // Add a conversion for FunctionType
    addConversion([&](FunctionType funcType) {
      llvm::SmallVector<Type> newInputTypes;
      auto check = convertTypes(funcType.getInputs(), newInputTypes);
      if (check.failed()) {
        llvm::errs() << "Failed to convert function input types\n";
        return funcType;
      }

      llvm::SmallVector<Type> newOutputTypes;
      check = convertTypes(funcType.getResults(), newOutputTypes);
      if (check.failed()) {
        llvm::errs() << "Failed to convert function output types\n";
        return funcType;
      }

      return FunctionType::get(funcType.getContext(), newInputTypes,
                               newOutputTypes);
    });
  }
};

class Bfp16ToI8ConversionPattern : public ConversionPattern {
public:
  Bfp16ToI8ConversionPattern(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: For now, the logic is simply to replace all bfp operations by an
    // integer of the appropriate width. Other considerations will be dealt with
    // later

    // Operation results
    std::for_each(
        op->getResults().begin(), op->getResults().end(), [&](Value result) {
          auto conversion = typeConverter->convertType(result.getType());
          if (!conversion) {
            llvm::errs() << "Failed to convert result type: "
                         << result.getType();
            return;
          }
          result.setType(conversion);
        });

    // Operation operands
    std::for_each(
        op->getOperands().begin(), op->getOperands().end(), [&](Value operand) {
          auto conversion = typeConverter->convertType(operand.getType());
          if (!conversion) {
            llvm::errs() << "Failed to convert operand type: "
                         << operand.getType();
            return;
          }
          operand.setType(conversion);
        });

    // Operation attributes
    // Note: For some reason, the attribute list looks like it is immutable and
    // needs to be recreated from scratch. Also note that type attributes cannot
    // access their type and must therefore be managed through
    // the convertTypeAttribute conversion instead
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : op->getAttrs()) {
      if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
        auto conversion = typeConverter->convertTypeAttribute(
            typeAttr.getValue(), attr.getValue());
        if (!conversion) {
          llvm::errs() << "Failed to convert attribute: " << typeAttr.getValue()
                       << "\n"
                       << "Attribute type: "
                       << typeAttr.getValue().getAbstractType().getName()
                       << "\n";
          newAttrs.push_back(attr);
          continue;
        }
        newAttrs.push_back(NamedAttribute(attr.getName(), conversion.value()));
      } else {
        newAttrs.push_back(attr);
      }
    }
    op->setAttrs(DictionaryAttr::get(op->getContext(), newAttrs));

    return success();
  }
};

class AIETransformBfp16TypePass
    : public AIETransformBfp16TypeBase<AIETransformBfp16TypePass> {
public:
  void runOnOperation() override {
    DeviceOp device = getOperation();
    MLIRContext *context = device.getContext();

    // Create the type converter
    Bfp16ToI8TypeConverter typeConverter(device.getTargetModel());

    // Set up an empty conversion target, since we have to iterate over all ops
    ConversionTarget target(*context);

    RewritePatternSet patterns(context);
    patterns.add<Bfp16ToI8ConversionPattern>(typeConverter, context);

    // Apply the conversion
    if (failed(applyPartialConversion(device, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIETransformBfp16TypePass() {
  return std::make_unique<AIETransformBfp16TypePass>();
}
