//===- CIRToAIEpasses.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//

#include <array>

#include "aie/CIR/CIRToAIEPasses.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using namespace std::string_literals;

namespace xilinx::AIE::CIR {

struct CIRToAIETypesAnalysis {
  // llvm::DenseMap<mlir::Type, std::optional<mlir::Type>> types;
  struct AIELikeTypesDeconstruction {
    // For example "aie::device<aie::npu1>"
    std::string fullName;
    // For example "aie::device"
    std::string base;
    // For example "npu1"
    std::vector<std::string> subMatches;

    std::string str() {
      return "Fullname = " + fullName + ", base = " + base +
             ", subMatches = " + llvm::join(subMatches, ", ");
    }
  };

  llvm::DenseMap<mlir::Type, std::optional<AIELikeTypesDeconstruction>>
      moduleTypes;

  void analyze() {
    // A struct with a name like "aie::device<aie::npu1>" (and the "npu1" is
    // used directly for the MLIR aie.device attribute) or aie::tile_t<8,50> for
    // example
    static const std::array typeNamePatterns{
        llvm::Regex{"^(aie::device)<aie::([^>]+)>$"},
        llvm::Regex{"^(aie::tile_t)<([[:digit:]]+), ([[:digit:]]+)>$"}};

    for (auto &[type, value] : moduleTypes) {
      if (auto maybePointerType = mlir::dyn_cast<mlir::cir::PointerType>(type))
        if (auto maybeStructType = mlir::dyn_cast<mlir::cir::StructType>(
                maybePointerType.getPointee()))
          for (auto &tnp : typeNamePatterns)
            if (llvm::SmallVector<llvm::StringRef> matches;
                tnp.match(maybeStructType.getName(), &matches)) {
              value = {.fullName = matches[0].str(), .base = matches[1].str()};
              for (auto &e : llvm::ArrayRef(matches.begin() + 2, matches.end()))
                value->subMatches.emplace_back(e.str());
              // No need to look for a next match, go for the next type to
              // categorize
              break;
            }
    }
  }

public:
  CIRToAIETypesAnalysis(mlir::ModuleOp module) {
    module->walk([this](mlir::Operation *op) {
      for (auto result : op->getResults()) {
        auto type = result.getType();
        moduleTypes.try_emplace(type, std::nullopt);
      }
    });
    analyze();
  }

  void dump() {
    for (auto &[type, value] : moduleTypes) {
      llvm::outs() << "Type: " << type << " value: ";
      if (value) {
        llvm::outs() << value->str() << '\n';
      } else
        llvm::outs() << "None\n";
    }
  }
};

namespace {
// Lower C++ code like \code aie::device<aie::npu1> into an \code
// aie.device(npu1){} operation
struct DeviceLowering : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
  using mlir::OpConversionPattern<mlir::cir::AllocaOp>::OpConversionPattern;

  // \todo Find a less ugly way to access the analysis. How is it possible for a
  // pattern to access some contextual information?
  // It should be OK since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // The struct has a name like "aie::device<aie::npu1>" and the "npu1"
    // is used directly for the MLIR aie.device attribute
    if (auto aieLike = cat->moduleTypes[op.getType()];
        aieLike && aieLike->base == "aie::device") {
      auto deviceName = aieLike->subMatches[0];
      auto deviceId =
          xilinx::AIE::symbolizeEnum<xilinx::AIE::AIEDevice>(deviceName);
      if (!deviceId)
        // Actually this test cannot happens since the API of
        // xilinx::AIE::symbolizeEnum is strange: even if it returns a
        // std::optional it errors without returning
        op.emitError() << "aie::device incorrect for '" << deviceName << "'";
      auto deviceOp =
          rewriter.create<xilinx::AIE::DeviceOp>(op.getLoc(), *deviceId);
      // The aie.device requires one block
      deviceOp.getRegion().emplaceBlock();
      // Replace the alloca of the aie::device by a temporary cast from
      // the aie.device to
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          op, op.getResult().getType(), deviceOp.getResult());
      return mlir::success();
    }
    return mlir::failure();
  }
};

// Rewrite something like
//
// %3 = cir.alloca !ty_aie3A3Atile_t3C12C_43E, // !cir.ptr<!ty_aie3A3Atile_t3C12C_43E>, ["t", init] {alignment = 1 : i64} %4 = // cir.alloca !ty_std3A3Aarray3Cint2C_8192UL3E, // !cir.ptr<!ty_std3A3Aarray3Cint2C_8192UL3E>, ["b", init] {alignment = 4 : i64}
// %5 = cir.call // @_ZN3aie6deviceILNS_3$_0E42EE4tileILi1ELi4EEENS_6tile_tIXT_EXT0_EEEv(%2) : // (!cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu13E>) -> !ty_aie3A3Atile_t3C12C_43E
// cir.store %5, %3 : !ty_aie3A3Atile_t3C12C_43E, // !cir.ptr<!ty_aie3A3Atile_t3C12C_43E>
//
// Into
//
// %3 = builtin.unrealized_conversion_cast %2 : // !cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu13E> to // !cir.ptr<!ty_aie3A3Atile_t3C12C_43E>
struct TileLowering : public mlir::OpConversionPattern<mlir::cir::CallOp> {
  using mlir::OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (auto calledFunc =
            mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
                op, op.getCalleeAttr())) {
      if (auto annnotations = calledFunc.getAnnotationsAttr())
        for (auto a : calledFunc.getAnnotationsAttr()) {
          // A call with this annotation is a tile construction from a device
          if (mlir::cast<mlir::cir::AnnotationAttr>(a).getName() ==
              "cir.aie.device.tile") {
            auto device = op.getOperand(0);
            auto user = op.getResult().getUsers().begin();
            // Track the alloca where the tiled is stored
            auto store = mlir::dyn_cast<mlir::cir::StoreOp>(*user);
            auto alloca = mlir::dyn_cast<mlir::cir::AllocaOp>(
                store.getOperand(1).getDefiningOp());
            // Replace the alloca by a conversion to be replaced later in
            // another pass
            rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
                alloca, alloca.getResult().getType(), device);
            // Remove the now useless original operations
            rewriter.eraseOp(store);
            rewriter.eraseOp(op);
            return mlir::success();
          }
        }
    }
    //

    return mlir::failure();
  }
};

struct CIRToAIE : CIRToAIEBase<CIRToAIE> {
  void runOnOperation() override {
    // Compute the analysis for the module since it is a module pass.
    // \todo Should this be a real pass?
    auto &cat = getAnalysis<CIRToAIETypesAnalysis>();
    // \todo Clean up this mess
    DeviceLowering::cat = &cat;
    // See mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
    mlir::ConversionTarget target{getContext()};
    target.addLegalDialect<xilinx::AIE::AIEDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<mlir::cir::AllocaOp>(
        [&](mlir::cir::AllocaOp op) {
          // If the struct has a name like "aie::device<aie::npu1>", mark the
          // operation illegal so it has to be rewritten
          auto aieLike = cat.moduleTypes[op.getType()];
          return !(aieLike && aieLike->base == "aie::device");
        });
    target.addDynamicallyLegalOp<mlir::cir::CallOp>([](mlir::cir::CallOp op) {
      if (auto calledFunc =
              mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
                  op, op.getCalleeAttr())) {
        if (auto annnotations = calledFunc.getAnnotationsAttr())
          for (auto a : calledFunc.getAnnotationsAttr()) {
            auto an = mlir::cast<mlir::cir::AnnotationAttr>(a);
            auto n = an.getName();
            if (n == "cir.aie.device.tile")
              return false;
          }
      }
      return true;
    });
    mlir::RewritePatternSet patterns{&getContext()};
    patterns.add<DeviceLowering>(&getContext());
    patterns.add<TileLowering>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCIRToAIEPass() {
  return std::make_unique<CIRToAIE>();
}

} // namespace xilinx::AIE::CIR
