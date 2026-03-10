//===- AIEAssignCoreLinkFiles.cpp -------------------------------*- C++ -*-===//
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
#define GEN_PASS_DEF_AIEASSIGNCORELINKFILES
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
    : xilinx::AIE::impl::AIEAssignCoreLinkFilesBase<
          AIEAssignCoreLinkFilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device.getContext());

    // Build a map from func name to the object file(s) it requires, sourced
    // from the "link_with" string attribute on func.func declarations.
    // StringRefs are views into MLIRContext-owned storage and remain valid
    // for the entire pass run.
    DenseMap<StringRef, SmallVector<StringRef, 2>> funcToObjs;
    for (auto funcOp : device.getOps<mlir::func::FuncOp>()) {
      if (auto attr = funcOp->getAttrOfType<mlir::StringAttr>("link_with")) {
        funcToObjs[funcOp.getName()].push_back(attr.getValue());
      }
    }

    // Tracks which func.func symbols are directly called from at least one
    // core; used to warn about link_with-bearing functions that are never
    // called and whose object files would otherwise be silently omitted.
    llvm::DenseSet<StringRef> usedFuncs;

    // Only direct func.call edges are traced.  func.call_indirect ops and
    // calls through intermediate wrapper functions are not followed.  To
    // handle transitive dependencies, attach link_with directly to every
    // func.func declaration that a core calls, even thin wrappers.
    // TODO: extend to transitive call resolution.
    device.walk([&](CoreOp core) {
      // De-duplicate while preserving insertion order.
      llvm::SetVector<StringRef> needed;

      // Migrate deprecated core-level attr: warn, consume it, and add to set.
      if (auto lw = core.getLinkWith()) {
        core.emitWarning(
            "link_with on aie.core is deprecated; attach link_with to "
            "the func.func declaration instead");
        needed.insert(lw.value());
        core->removeAttr("link_with");
      }

      // Single walk over the core body: collect required object files and
      // record called symbols (for the unused-func warning below).
      core.walk([&](Operation *op) {
        if (auto call = dyn_cast<mlir::func::CallOp>(op)) {
          usedFuncs.insert(call.getCallee());
          auto it = funcToObjs.find(call.getCallee());
          if (it != funcToObjs.end())
            for (StringRef obj : it->second)
              needed.insert(obj);
        } else if (auto indCall = dyn_cast<mlir::func::CallIndirectOp>(op)) {
          indCall.emitWarning(
              "indirect call in core body — link_with attributes on "
              "indirectly-called functions are not automatically resolved; "
              "add a direct func.call to the required func.func declaration "
              "so that aie-assign-core-link-files can trace the dependency");
        }
      });

      if (!needed.empty()) {
        // builder is used only for attribute construction; its insertion
        // point is irrelevant and no ops are inserted.
        core.setLinkFilesAttr(builder.getStrArrayAttr(needed.getArrayRef()));
      }
    });

    // Warn about funcs with link_with that are never called from any core.
    for (auto &[funcName, objs] : funcToObjs) {
      if (!usedFuncs.count(funcName)) {
        if (auto funcOp = device.lookupSymbol<mlir::func::FuncOp>(funcName))
          funcOp.emitWarning()
              << "func '" << funcName
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
