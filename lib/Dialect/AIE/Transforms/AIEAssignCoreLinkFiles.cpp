//===- AIEAssignCoreLinkFiles.cpp --------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// This pass infers the per-core set of external object files required for
// linking by tracing call edges from each core to func.func declarations that
// carry a "link_with" attribute.
//
// After the pass runs, every CoreOp that needs external files will have a
// "link_files" StrArrayAttr containing the (de-duplicated) list of .o paths.
//
// Core-level "link_with" (deprecated) is also migrated: its value is added to
// the set and the attribute is removed from the CoreOp.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "aie-assign-core-link-files"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEAssignCoreLinkFilesPass
    : AIEAssignCoreLinkFilesBase<AIEAssignCoreLinkFilesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect, mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device.getContext());

    // Build map: func name -> list of .o files (from "link_with" attr on
    // func.func). Keys and values are interned in the MLIRContext so the
    // StringRefs remain valid for the lifetime of the pass.
    DenseMap<StringRef, SmallVector<StringRef, 2>> funcToObjs;
    for (auto funcOp : device.getOps<mlir::func::FuncOp>()) {
      if (auto attr = funcOp->getAttrOfType<mlir::StringAttr>("link_with")) {
        funcToObjs[funcOp.getName()].push_back(attr.getValue());
      }
    }

    // Track which funcs are actually called from any core.
    llvm::DenseSet<StringRef> usedFuncs;

    // Walk each core, collect all .o files needed.
    device.walk([&](CoreOp core) {
      // Always walk CallOps first to keep usedFuncs accurate even when the
      // idempotency guard fires below (prevents false "never called" warnings
      // on a second pass invocation).
      core.walk(
          [&](mlir::func::CallOp call) { usedFuncs.insert(call.getCallee()); });

      // Early-out: pass already ran on this core and migration is done.
      if (core.getLinkFiles() && !core.getLinkWith())
        return;

      // De-duplicate while preserving insertion order. StringRefs point into
      // the MLIRContext attribute storage and remain valid throughout the pass.
      llvm::SetVector<StringRef> needed;

      // Migrate deprecated core-level attr: warn, consume it, and add to set.
      if (auto lw = core.getLinkWith()) {
        core.emitWarning(
            "link_with on aie.core is deprecated; attach link_with to "
            "the func.func declaration instead");
        needed.insert(lw.value());
        core->removeAttr("link_with");
      }

      // Trace func::CallOp ops to accumulate needed .o files.
      core.walk([&](mlir::func::CallOp call) {
        auto it = funcToObjs.find(call.getCallee());
        if (it != funcToObjs.end())
          for (StringRef obj : it->second)
            needed.insert(obj);
      });

      // Warn on indirect calls: link_with cannot be statically resolved.
      core.walk([&](mlir::func::CallIndirectOp indCall) {
        indCall.emitWarning(
            "indirect call in core body — link_with attributes on "
            "indirectly-called functions are not automatically resolved; "
            "declare the required .o files via link_with on the aie.core "
            "or on a directly-called func.func");
      });

      if (!needed.empty())
        core.setLinkFilesAttr(builder.getStrArrayAttr(needed.getArrayRef()));
    });

    // Warn about funcs with link_with that are never called from any core.
    for (auto &[funcName, objs] : funcToObjs) {
      if (!usedFuncs.count(funcName)) {
        if (auto funcOp = device.lookupSymbol<mlir::func::FuncOp>(funcName))
          funcOp.emitWarning("func '")
              << funcName
              << "' has link_with but is never called from any core; "
                 "its .o file will not be linked";
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignCoreLinkFilesPass() {
  return std::make_unique<AIEAssignCoreLinkFilesPass>();
}
