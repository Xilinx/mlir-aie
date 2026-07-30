//===- AIEAssignCoreLinkFiles.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass infers the per-core set of external link artifacts required by
// each core by tracing call edges from that core to func.func declarations
// that carry a "link_with" attribute.
//
// How an artifact is linked is taken from metadata, never from the file name:
// an optional "link_with_mode" attribute on the same func.func declaration
// selects the policy.
//
//   * absent            -> ordinary input to the core's final link
//   * "merge"           -> merged into the core's LLVM module (llvm-link)
//                          before codegen; never handed to the final link
//
// After the pass runs, every CoreOp that needs external artifacts will have a
// "link_files" and/or a "link_merge_files" StrArrayAttr containing the
// (de-duplicated) list of paths for that policy.
//
// Core-level "link_with" (deprecated) is also migrated: its value is added to
// the ordinary "link_files" set and the attribute is removed from the CoreOp.
// There is no core-level way to request merging.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#define GEN_PASS_DEF_AIEASSIGNCORELINKFILES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "aie-assign-core-link-files"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

/// One external artifact required by a func.func declaration, together with
/// the policy that decides how it reaches the core binary.
struct LinkArtifact {
  StringRef path;
  bool merge;
};

/// The artifacts a single core needs, split by policy.  Computed for every
/// core before any IR is mutated, so that the device-scope consistency check
/// below can fail the pass without leaving half-written attributes behind.
struct CoreLinkFiles {
  CoreOp core;
  bool hadDeprecatedLinkWith = false;
  // De-duplicated, insertion-ordered.
  llvm::SetVector<StringRef> linkFiles;
  llvm::SetVector<StringRef> mergeFiles;
};

} // namespace

struct AIEAssignCoreLinkFilesPass
    : xilinx::AIE::impl::AIEAssignCoreLinkFilesBase<
          AIEAssignCoreLinkFilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    // Builder is used only for attribute construction; no ops are inserted.
    Builder builder(device.getContext());

    // Build a map from func name to the artifact(s) it requires, sourced from
    // the "link_with"/"link_with_mode" attributes on func.func declarations.
    // StringRefs are views into MLIRContext-owned storage and remain valid
    // for the entire pass run.
    DenseMap<StringRef, SmallVector<LinkArtifact, 2>> funcToObjs;
    bool badMetadata = false;
    for (auto funcOp : device.getOps<mlir::func::FuncOp>()) {
      auto linkWith = funcOp->getAttrOfType<mlir::StringAttr>("link_with");
      auto modeAttr = funcOp->getAttrOfType<mlir::StringAttr>("link_with_mode");
      if (modeAttr && !linkWith) {
        funcOp.emitError()
            << "func '" << funcOp.getName()
            << "' has link_with_mode but no link_with; link_with_mode only "
               "describes how the link_with artifact is linked";
        badMetadata = true;
        continue;
      }
      if (!linkWith)
        continue;
      bool merge = false;
      if (modeAttr) {
        // "merge" is the only non-default policy; everything else is a typo.
        if (modeAttr.getValue() != "merge") {
          funcOp.emitError()
              << "func '" << funcOp.getName()
              << "' has unknown link_with_mode '" << modeAttr.getValue()
              << "'; the only supported value is 'merge'";
          badMetadata = true;
          continue;
        }
        merge = true;
      }
      funcToObjs[funcOp.getName()].push_back({linkWith.getValue(), merge});
    }
    if (badMetadata)
      return signalPassFailure();

    // Tracks which func.func symbols are directly called from at least one
    // core; used to warn about link_with-bearing functions that are never
    // called and whose artifacts would otherwise be silently omitted.
    llvm::DenseSet<StringRef> usedFuncs;

    // Only direct func.call edges are traced.  func.call_indirect ops and
    // calls through intermediate wrapper functions are not followed.  To
    // handle transitive dependencies, attach link_with directly to every
    // func.func declaration that a core calls, even thin wrappers.
    // TODO: extend to transitive call resolution.
    SmallVector<CoreLinkFiles> coreFiles;
    device.walk([&](CoreOp core) {
      CoreLinkFiles files;
      files.core = core;

      // Migrate deprecated core-level attr: warn and add to the ordinary set.
      // The attribute itself is removed in the mutation phase below.  There is
      // deliberately no core-level way to request merging.
      if (auto lw = core.getLinkWith()) {
        core.emitWarning(
            "link_with on aie.core is deprecated; attach link_with to "
            "the func.func declaration instead");
        files.linkFiles.insert(lw.value());
        files.hadDeprecatedLinkWith = true;
      }

      // Single walk over the core body: collect required artifacts and
      // record called symbols (for the unused-func warning below).
      core.walk([&](Operation *op) {
        if (auto call = dyn_cast<mlir::func::CallOp>(op)) {
          usedFuncs.insert(call.getCallee());
          auto it = funcToObjs.find(call.getCallee());
          if (it != funcToObjs.end())
            for (const LinkArtifact &artifact : it->second)
              (artifact.merge ? files.mergeFiles : files.linkFiles)
                  .insert(artifact.path);
        } else if (auto indCall = dyn_cast<mlir::func::CallIndirectOp>(op)) {
          indCall.emitWarning(
              "indirect call in core body — link_with attributes on "
              "indirectly-called functions are not automatically resolved; "
              "add a direct func.call to the required func.func declaration "
              "so that aie-assign-core-link-files can trace the dependency");
        }
      });

      coreFiles.push_back(std::move(files));
    });

    // Device-scope consistency check.  aiecc has a "unified" mode in which all
    // cores of a device share one LLVM module that is llvm-linked once; if one
    // core merges an artifact while another object-links it, the second core's
    // ELF would define the artifact's symbols twice.  A per-core check is
    // therefore not enough: the policy must agree across the whole device.
    llvm::MapVector<StringRef, CoreOp> linkedBy;
    for (CoreLinkFiles &files : coreFiles)
      for (StringRef path : files.linkFiles)
        linkedBy.insert({path, files.core});
    bool conflict = false;
    for (CoreLinkFiles &files : coreFiles)
      for (StringRef path : files.mergeFiles) {
        auto it = linkedBy.find(path);
        if (it == linkedBy.end())
          continue;
        conflict = true;
        InFlightDiagnostic diag =
            files.core.emitError()
            << "artifact '" << path
            << "' is merged into an LLVM module here but object-linked in the "
               "same aie.device; an artifact must use a single link mode "
               "across all cores of a device";
        diag.attachNote(it->second.getLoc())
            << "artifact '" << path << "' is object-linked here";
      }
    if (conflict)
      return signalPassFailure();

    // Mutation phase.
    for (CoreLinkFiles &files : coreFiles) {
      if (files.hadDeprecatedLinkWith)
        files.core->removeAttr("link_with");
      // builder is used only for attribute construction; its insertion
      // point is irrelevant and no ops are inserted.
      if (!files.linkFiles.empty())
        files.core.setLinkFilesAttr(
            builder.getStrArrayAttr(files.linkFiles.getArrayRef()));
      if (!files.mergeFiles.empty())
        files.core.setLinkMergeFilesAttr(
            builder.getStrArrayAttr(files.mergeFiles.getArrayRef()));
    }

    // Warn about funcs with link_with that are never called from any core.
    for (auto &[funcName, objs] : funcToObjs) {
      if (!usedFuncs.count(funcName)) {
        if (auto funcOp = device.lookupSymbol<mlir::func::FuncOp>(funcName))
          funcOp.emitWarning()
              << "func '" << funcName
              << "' has link_with but is never called from any core; "
                 "its artifact will not be linked or merged";
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignCoreLinkFilesPass() {
  return std::make_unique<AIEAssignCoreLinkFilesPass>();
}
