//===- AIETransformBfpTypes.cpp --------------------------------*- C++ -*-===//
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

#define DEBUG_TYPE "transform-bfp-types"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

using namespace mlir;

class BfpToIntegerConverter : public mlir::TypeConverter {
public:
  BfpToIntegerConverter(const AIETargetModel &targetModel) {
    addTypeAttributeConversion([&](Type type, Attribute attr) {
      auto newType = convertType(type);
      if (!newType) {
        llvm::errs() << "Failed to convert type: " << type << "\n";
        return AttributeConversionResult::abort();
      }
      return AttributeConversionResult(TypeAttr::get(newType));
    });

    // Note that the most recently added conversions will be invoked first

    // Leave other types unchanged
    addConversion([](Type type) -> std::optional<Type> { return type; });

    // Add a conversion for bfpTypes to an integer type
    addConversion([&](BlockFloatType blockType) -> std::optional<IntegerType> {
      bool isSupported =
          targetModel.isSupportedBlockFormat(blockType.getBlockType().str());
      if (!isSupported) {
        llvm::errs() << "Block type " << blockType.getBlockType()
                     << " is not supported in the specified model\n";
        // Note that returning a nullptr here will stop the conversion while
        // returning a std::nullopt will allow the converter to keep trying the
        // remaining conversions (thus reaching the default one in this case)
        return nullptr;
      }

      return mlir::IntegerType::get(blockType.getContext(),
                                    blockType.getTotalSizeInBits());
    });

    // Add a conversion for MemRefType
    addConversion([&](MemRefType memRefType) -> std::optional<MemRefType> {
      auto newElementType = convertType(memRefType.getElementType());
      if (!newElementType) {
        llvm::errs() << "Failed to convert memref element type\n";
        return nullptr;
      }
      return MemRefType::get(memRefType.getShape(), newElementType,
                             memRefType.getLayout(),
                             memRefType.getMemorySpace());
    });

    // Add a conversion for ObjectFifoType
    addConversion([&](AIEObjectFifoType objectFifoType)
                      -> std::optional<AIEObjectFifoType> {
      auto newElementType = convertType(objectFifoType.getElementType());
      if (!newElementType) {
        llvm::errs() << "Failed to convert ObjectFifoType element type\n";
        return nullptr;
      }

      if (auto newMemRef = dyn_cast<MemRefType>(newElementType))
        return AIEObjectFifoType::get(objectFifoType.getContext(), newMemRef);

      llvm::errs()
          << "ObjectFifoType converted element type is not a MemRefType\n";
      return nullptr;
    });

    // Add a conversion for ObjectFifoSubviewType
    addConversion([&](AIEObjectFifoSubviewType objectFifoSubviewType)
                      -> std::optional<AIEObjectFifoSubviewType> {
      auto newElementType = convertType(objectFifoSubviewType.getElementType());
      if (!newElementType) {
        llvm::errs()
            << "Failed to convert ObjectFifoSubviewType element type\n";
        return nullptr;
      }

      if (auto newMemRef = dyn_cast<MemRefType>(newElementType))
        return AIEObjectFifoSubviewType::get(objectFifoSubviewType.getContext(),
                                             newMemRef);

      llvm::errs()
          << "ObjectFifoSubviewType element type is not a MemRefType\n";
      return nullptr;
    });

    // Add a conversion for FunctionType
    addConversion([&](FunctionType funcType) -> std::optional<FunctionType> {
      llvm::SmallVector<Type> newInputTypes;
      auto check = convertTypes(funcType.getInputs(), newInputTypes);
      if (check.failed()) {
        llvm::errs() << "Failed to convert function input types\n";
        return nullptr;
      }

      llvm::SmallVector<Type> newOutputTypes;
      check = convertTypes(funcType.getResults(), newOutputTypes);
      if (check.failed()) {
        llvm::errs() << "Failed to convert function output types\n";
        return nullptr;
      }

      return FunctionType::get(funcType.getContext(), newInputTypes,
                               newOutputTypes);
    });

    // Add conversions for other types as needed (llvm arrays?)
  }
};

class BfpToIntegerConversionPattern : public ConversionPattern {
public:
  BfpToIntegerConversionPattern(TypeConverter &typeConverter,
                                MLIRContext *context, bool &conversionFailed)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context),
        conversionFailed(conversionFailed) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // The objective is to replace all bfp operations by an integer of the
    // appropriate width. This pass currently does not have any other
    // functionality.

    // Operation results
    for (auto result : op->getResults()) {
      auto conversion = typeConverter->convertType(result.getType());
      if (!conversion) {
        conversionFailed = true;
        return op->emitError()
               << "Failed to convert result type: " << result.getType();
      }
      result.setType(conversion);
    }

    // Operation operands
    for (auto operand : op->getOperands()) {
      auto conversion = typeConverter->convertType(operand.getType());
      if (!conversion) {
        conversionFailed = true;
        return op->emitError()
               << "Failed to convert operand type: " << operand.getType();
      }
      operand.setType(conversion);
    }

    // Operation attributes
    // Note that the attribute list is immutable and
    // needs to be recreated from scratch. Also note that type attributes cannot
    // access their type and must therefore be managed through
    // the convertTypeAttribute conversion instead
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : op->getAttrs()) {
      if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
        auto conversion = typeConverter->convertTypeAttribute(
            typeAttr.getValue(), attr.getValue());
        if (!conversion) {
          conversionFailed = true;
          return op->emitError()
                 << "Failed to convert attribute type: " << typeAttr.getValue();
        }
        newAttrs.push_back(NamedAttribute(attr.getName(), conversion.value()));
      } else {
        newAttrs.push_back(attr);
      }
    }
    op->setAttrs(DictionaryAttr::get(op->getContext(), newAttrs));

    return success();
  }

private:
  bool &conversionFailed;
};

class AIETransformBfpTypesPass
    : public AIETransformBfpTypesBase<AIETransformBfpTypesPass> {
public:
  void runOnOperation() override {
    DeviceOp device = getOperation();
    MLIRContext *context = device.getContext();

    BfpToIntegerConverter typeConverter(device.getTargetModel());

    // Set up an empty conversion target, since we have to iterate over all ops
    ConversionTarget target(*context);

    RewritePatternSet patterns(context);
    bool conversionFailed = false;
    patterns.add<BfpToIntegerConversionPattern>(typeConverter, context,
                                                conversionFailed);

    if (failed(applyPartialConversion(device, target, std::move(patterns))) ||
        conversionFailed) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIETransformBfpTypesPass() {
  return std::make_unique<AIETransformBfpTypesPass>();
}
