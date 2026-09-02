//===- IRTransforms.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MLIR-IR (and LLVM-IR text) helpers: pass-pipeline builders, clone-and-mutate
// utilities, and small in-place IR walks used by aiecc's graph edges.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_IRTRANSFORMS_H
#define AIECC_IRTRANSFORMS_H

#include "Graph.h"
#include "StackSizeAnalysis.h"
#include "Utils.h"

#include "aie/Conversion/Passes.h"
#include "aie/Dialect/AIE/IR/AIECoreSymbols.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace xilinx::aiecc {

inline void registerLLVMIRTranslations(mlir::DialectRegistry &registry) {
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  xilinx::xllvm::registerXLLVMDialectTranslation(registry);
}

// PDI ID attribute carried on each DeviceOp; readers (load_pdi stamping,
// full-ELF config.json) consult this rather than re-deriving walk order.
constexpr llvm::StringLiteral kPdiIdAttr = "aiecc.pdi_id";

//===----------------------------------------------------------------------===//
// IR inspection
//===----------------------------------------------------------------------===//

// Detect the AIE target arch (lowercase: "aie", "aie2", "aie2p", ...).
// Falls back to "aie2".
inline std::string detectAIETarget(mlir::ModuleOp m,
                                   llvm::StringRef deviceName = "") {
  for (auto devOp : m.getOps<xilinx::AIE::DeviceOp>()) {
    if (!deviceName.empty() && devOp.getSymName() != deviceName)
      continue;
    std::string s;
    llvm::raw_string_ostream os(s);
    if (mlir::succeeded(
            xilinx::AIE::AIETranslateToTargetArch(m, os, devOp.getSymName()))) {
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                            s.back() == ' ' || s.back() == '\t'))
        s.pop_back();
      if (!s.empty())
        return llvm::StringRef(s).lower();
    }
    break;
  }
  return "aie2";
}

// Per-core key `"<devName>_core_<col>_<row>"` — shared across compiled-elfs /
// pre-baked-elfs / ldscripts nodes for bundle-by-key zips.
inline std::string coreKey(xilinx::AIE::CoreOp coreOp) {
  auto tile = mlir::cast<xilinx::AIE::TileOp>(coreOp.getTile().getDefiningOp());
  auto dev = coreOp->getParentOfType<xilinx::AIE::DeviceOp>();
  return dev.getSymName().str() + "_core_" + std::to_string(tile.getCol()) +
         "_" + std::to_string(tile.getRow());
}

//===----------------------------------------------------------------------===//
// In-place IR mutations
//===----------------------------------------------------------------------===//

// Stamp `aiecc.pdi_id` 1-based on every DeviceOp. Call once on the
// pre-split module so the attribute propagates through clones.
inline void assignDevicePdiIds(mlir::ModuleOp module) {
  mlir::Builder b(module.getContext());
  int nextId = 1;
  for (auto d : module.getOps<xilinx::AIE::DeviceOp>())
    d->setAttr(kPdiIdAttr, b.getI32IntegerAttr(nextId++));
}

// Propagate each device's `aiecc.pdi_id` onto every load_pdi referencing it.
inline void assignLoadPdiIds(mlir::ModuleOp module) {
  module.walk([&](xilinx::AIEX::NpuLoadPdiOp lp) {
    auto ref = lp.getDeviceRefAttr();
    if (!ref)
      return;
    auto dev = module.lookupSymbol<xilinx::AIE::DeviceOp>(ref.getValue());
    if (!dev)
      return;
    if (auto id = dev->getAttrOfType<mlir::IntegerAttr>(kPdiIdAttr))
      lp.setId(static_cast<uint32_t>(id.getInt()));
  });
}

//===----------------------------------------------------------------------===//
// Clone-and-mutate helpers
//===----------------------------------------------------------------------===//

// Clone `src` and absolutize the `(col, row)` CoreOp's `link_files` so
// the emitted ld script's INPUT() entries are cwd-independent.
inline mlir::OwningOpRef<mlir::ModuleOp>
absolutizeLinkFiles(mlir::ModuleOp src, int col, int row,
                    llvm::StringRef inputFile, llvm::StringRef workDir) {
  mlir::OwningOpRef<mlir::ModuleOp> cloned = src.clone();
  cloned->walk([&](xilinx::AIE::CoreOp coreOp) {
    auto tileOp =
        mlir::dyn_cast<xilinx::AIE::TileOp>(coreOp.getTile().getDefiningOp());
    if (!tileOp || tileOp.getCol() != col || tileOp.getRow() != row)
      return;
    auto filesAttr = coreOp.getLinkFiles();
    if (!filesAttr)
      return;
    llvm::SmallVector<mlir::Attribute> absFiles;
    for (auto f : filesAttr->getAsRange<mlir::StringAttr>())
      absFiles.push_back(mlir::StringAttr::get(
          cloned->getContext(),
          resolveExternalPath(f.getValue(), inputFile, workDir)));
    coreOp.setLinkFilesAttr(
        mlir::ArrayAttr::get(cloned->getContext(), absFiles));
  });
  return cloned;
}

// Collect `coreOp`'s merge-mode link artifacts -- the entries of
// `link_merge_files`, populated by aie-assign-core-link-files from
// `link_with_mode = "merge"` on the func.func declaration -- resolved to
// absolute paths. These are llvm-linked into the core's LLVM module before
// codegen (see buildObjectSubgraph's peano path) and inlined, eliminating the
// func.call boundary and the separately object-linked kernel object. Routing
// is decided purely by this attribute, never by file suffix: an artifact in
// the sibling `link_files` list is an ordinary final-link input whatever its
// format. The ldscript/BCF emitters emit `link_files` only, so an artifact
// merged here is never also object-linked and each symbol is defined once.
//
// The deprecated core-level `link_with` has nowhere to carry a mode, so it can
// never request merging and is not consulted here.
inline std::vector<std::string>
collectCoreIRLinkFiles(xilinx::AIE::CoreOp coreOp, llvm::StringRef inputFile,
                       llvm::StringRef workDir) {
  std::vector<std::string> files;
  if (auto mergeAttr = coreOp.getLinkMergeFiles())
    for (auto f : mergeAttr->getAsRange<mlir::StringAttr>())
      files.push_back(resolveExternalPath(f.getValue(), inputFile, workDir));
  return files;
}

// Sets `stack_size = defaultStackSize` on every CoreOp without an explicit
// `stack_size`. A CoreOp with an explicit `stack_size` keeps it.
inline mlir::OwningOpRef<mlir::ModuleOp>
populateDefaultStackSize(mlir::ModuleOp src, int64_t defaultStackSize) {
  mlir::OwningOpRef<mlir::ModuleOp> cloned = src.clone();
  mlir::Builder b(cloned->getContext());
  cloned->walk([&](xilinx::AIE::CoreOp coreOp) {
    if (!coreOp.getStackSizeAttr())
      coreOp.setStackSizeAttr(
          b.getI32IntegerAttr(static_cast<int32_t>(defaultStackSize)));
  });
  return cloned;
}

inline void appendSkippedArtifacts(mlir::InFlightDiagnostic &diag,
                                   llvm::ArrayRef<std::string> skipped) {
  for (size_t i = 0; i < skipped.size(); ++i)
    diag << (i ? ", " : "") << skipped[i];
}

// Measures each core's stack requirement and writes it to
// `measured_stack_size`: the top-level frame that `measureOwnFrame` reads from
// the compiled core object, plus the deepest call path through the core's
// `link_files` (StackSizeAnalysis.h).
//
// A requirement above `stack_size` fails the build, as does a cycle in the call
// graph. An unmeasurable symbol warns. An unmeasurable core frame leaves the
// requirement a lower bound, which warns and writes no attribute.
inline mlir::LogicalResult checkStackSizeRequirements(
    mlir::ModuleOp module, llvm::StringRef inputFile, llvm::StringRef workDir,
    llvm::function_ref<std::optional<int64_t>(xilinx::AIE::CoreOp)>
        measureOwnFrame) {
  mlir::LogicalResult result = mlir::success();

  // Keyed by object, because the result depends on the object alone. Several
  // cores can list one object.
  llvm::StringMap<llvm::StringSet<>> definedFunctionsByPath;

  for (xilinx::AIE::DeviceOp device : module.getOps<xilinx::AIE::DeviceOp>()) {
    // Collected per device, because each DeviceOp is its own symbol table,
    // and a sibling device can bind one name to a different override. The
    // check for a negative value repeats the one in external_func(), which
    // hand-written MLIR skips.
    llvm::StringMap<int64_t> overrides;
    device.walk([&](mlir::func::FuncOp funcOp) {
      auto attr =
          funcOp->getAttrOfType<mlir::IntegerAttr>("stack_size_override");
      if (!attr)
        return;
      if (attr.getInt() < 0) {
        funcOp.emitError() << "stack_size_override must be >= 0, got "
                           << attr.getInt();
        result = mlir::failure();
        return;
      }
      overrides[funcOp.getName()] = attr.getInt();
    });

    device.walk([&](xilinx::AIE::CoreOp coreOp) {
      // The roots are the symbols the core body calls directly.
      // AIEAssignCoreLinkFilesPass walks the same set.
      llvm::SmallVector<llvm::StringRef, 4> roots;
      llvm::StringSet<> seenRoots;
      coreOp.walk([&](mlir::func::CallOp call) {
        if (seenRoots.insert(call.getCallee()).second)
          roots.push_back(call.getCallee());
      });

      // A merge-mode kernel carries link_merge_files alone, so link_files is
      // absent while its roots can still carry a stack_size_override. The loop
      // below then scans zero objects, and those overrides still satisfy the
      // roots. A root with no object and no override reaches the unmeasurable
      // warning below.
      std::vector<std::string> resolved;
      if (auto filesAttr = coreOp.getLinkFiles())
        for (auto f : filesAttr->getAsRange<mlir::StringAttr>())
          resolved.push_back(
              resolveExternalPath(f.getValue(), inputFile, workDir));

      llvm::StringSet<> knownFunctions;
      for (const std::string &path : resolved) {
        auto it = definedFunctionsByPath.find(path);
        if (it == definedFunctionsByPath.end())
          it = definedFunctionsByPath
                   .try_emplace(path,
                                [&] {
                                  llvm::StringSet<> names;
                                  xilinx::aiecc::collectDefinedFunctionNames(
                                      path, names);
                                  return names;
                                }())
                   .first;
        for (const auto &name : it->second)
          knownFunctions.insert(name.first());
      }

      xilinx::aiecc::StackGraph graph;
      std::vector<std::string> skipped;
      for (const std::string &path : resolved)
        if (!xilinx::aiecc::addObjectToStackGraph(path, knownFunctions, graph))
          skipped.push_back(path);
      xilinx::aiecc::resolveIndirectCallEdges(graph);

      auto stackRes =
          xilinx::aiecc::computeStackRequirement(graph, roots, overrides);
      if (!stackRes.bytes) {
        if (stackRes.failureKind ==
            xilinx::aiecc::StackRequirementFailure::Cycle) {
          coreOp.emitError()
              << "cannot determine this core's stack requirement: "
              << stackRes.error
              << "; set stack_size_override on the affected kernel's "
                 "external_func()/func.func declaration (Kernel(...)/"
                 "ExternalFunction(...) in IRON), or pass "
                 "--no-measure-stack-size to skip this check entirely";
          result = mlir::failure();
        } else {
          coreOp.emitWarning()
              << "cannot determine this core's stack requirement: "
              << stackRes.error
              << "; stack_size is not being validated for this core. Set "
                 "stack_size_override on the affected kernel's "
                 "external_func()/func.func declaration (Kernel(...)/"
                 "ExternalFunction(...) in IRON) to enable it";
        }
        return;
      }

      if (!skipped.empty()) {
        auto diag = coreOp.emitWarning()
                    << "stack requirement computed as " << *stackRes.bytes
                    << " bytes, but " << skipped.size()
                    << " link_files artifact(s) could not be inspected "
                       "(archive, bitcode, or unreadable), so this may be "
                       "incomplete: ";
        appendSkippedArtifacts(diag, skipped);
      }

      uint32_t effective = coreOp.getEffectiveStackSize();
      std::optional<int64_t> ownFrame = measureOwnFrame(coreOp);
      if (!ownFrame) {
        if (static_cast<int64_t>(effective) < *stackRes.bytes)
          coreOp.emitWarning()
              << "this core's callees need at least " << *stackRes.bytes
              << " bytes of stack (not counting the core body's own frame), "
                 "but "
              << (coreOp.getStackSizeAttr() ? "stack_size is only "
                                            : "the default stack_size is only ")
              << effective << " bytes";
        return;
      }

      // An unchecked narrowing to i32 wraps to a small or negative number.
      if (*ownFrame > INT32_MAX - *stackRes.bytes) {
        coreOp.emitWarning()
            << "stack requirement computed as " << (*ownFrame + *stackRes.bytes)
            << " bytes, which does not fit in the attribute's i32; "
               "stack_size is not being validated for this core";
        return;
      }

      int64_t required = *ownFrame + *stackRes.bytes;
      coreOp.setMeasuredStackSizeAttr(
          mlir::Builder(module.getContext())
              .getI32IntegerAttr(static_cast<int32_t>(required)));

      if (static_cast<int64_t>(effective) < required) {
        if (coreOp.getStackSizeAttr())
          coreOp.emitError() << "stack_size = " << effective
                             << " is insufficient: this core needs " << required
                             << " bytes; increase stack_size to " << required
                             << " (Worker(stack_size=...) in IRON), or pass "
                                "--no-measure-stack-size to skip this check";
        else
          coreOp.emitError()
              << "stack_size is absent, so this core uses the device default "
                 "of "
              << effective << " bytes, but it needs " << required
              << " bytes; set stack_size = " << required
              << " (Worker(stack_size=...) in IRON), or pass "
                 "--no-measure-stack-size to skip this check";
        result = mlir::failure();
      }
    });
  }
  return result;
}

// Clone `src` and replace each matched CoreOp with a stub that carries
// `elf_file = <path>` and an empty body (verifier requires empty body when
// elf_file is set).
inline mlir::OwningOpRef<mlir::ModuleOp>
patchCoreElfFiles(mlir::ModuleOp src,
                  const llvm::StringMap<std::string> &elfByKey) {
  mlir::OwningOpRef<mlir::ModuleOp> cloned = src.clone();
  cloned->walk([&](xilinx::AIE::CoreOp coreOp) {
    auto it = elfByKey.find(coreKey(coreOp));
    if (it == elfByKey.end())
      return;
    mlir::OpBuilder b(coreOp);
    auto stub = xilinx::AIE::CoreOp::create(b, coreOp.getLoc(),
                                            b.getIndexType(), coreOp.getTile());
    for (auto attr : coreOp->getAttrs())
      stub->setAttr(attr.getName(), attr.getValue());
    stub.setElfFileAttr(b.getStringAttr(it->second));
    mlir::Block *body = b.createBlock(&stub.getBody());
    b.setInsertionPointToEnd(body);
    xilinx::AIE::EndOp::create(b, coreOp.getLoc());
    coreOp.erase();
  });
  return cloned;
}

//===----------------------------------------------------------------------===//
// LLVM-IR text post-processing
//===----------------------------------------------------------------------===//

// Strip newer-LLVM features Peano's older opt/llc can't parse. aiecc's LLVM is
// 24; Peano's is 21, so the text handed between them needs the gap patched.
inline std::string downgradeIRForPeano(llvm::StringRef ir,
                                       bool stripAlign = true) {
  std::string result = ir.str();
  auto erasePattern = [&](llvm::StringRef pat, auto trail) {
    for (size_t p = 0; (p = result.find(pat.str(), p)) != std::string::npos;) {
      size_t end = p + pat.size();
      while (end < result.size() && trail(result[end]))
        ++end;
      result.erase(p, end - p);
    }
  };
  // Newer LLVM prints special floats as 'inf'/'-inf'/'nan'; Peano's opt only
  // accepts the hex form. Anchor the rewrite on the preceding type keyword to
  // pick the correct hex width, and require a non-identifier char before it so
  // 'float' does not match inside 'bfloat'.
  auto isIdentChar = [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
  };
  auto replaceTypedLiteral = [&](llvm::StringRef from, llvm::StringRef to) {
    for (size_t p = 0; (p = result.find(from.str(), p)) != std::string::npos;) {
      if (p == 0 || !isIdentChar(result[p - 1])) {
        result.replace(p, from.size(), to.str());
        p += to.size();
      } else {
        p += from.size();
      }
    }
  };
  erasePattern("nocreateundeforpoison",
               [](char c) { return c == ' ' || c == '\t'; });
  // LLVM 23 dropped the size operand of `llvm.lifetime.start`/`.end`; Peano
  // still declares it `immarg`, so the size-less form fails its verifier
  // ("immarg operand has non-immediate parameter"). Put it back -- `-1` is
  // "whole object", what LLVM's own auto-upgrade uses. Matching the marker name
  // covers every address space; already-sized calls are skipped, so this is
  // idempotent.
  for (llvm::StringRef marker :
       {"@llvm.lifetime.start.", "@llvm.lifetime.end."}) {
    for (size_t p = 0;
         (p = result.find(marker.str(), p)) != std::string::npos;) {
      size_t paren = result.find('(', p);
      size_t eol = result.find('\n', p);
      if (paren == std::string::npos ||
          (eol != std::string::npos && paren > eol)) {
        p += marker.size();
        continue;
      }
      // The size-less form is the one whose first operand is the pointer.
      size_t arg = paren + 1;
      if (arg + 3 > result.size() || result.compare(arg, 3, "ptr") != 0 ||
          (arg + 3 < result.size() && isIdentChar(result[arg + 3]))) {
        p = arg;
        continue;
      }
      size_t bol = result.rfind('\n', p);
      bol = (bol == std::string::npos) ? 0 : bol + 1;
      bool isDeclaration = llvm::StringRef(result)
                               .substr(bol, p - bol)
                               .ltrim()
                               .starts_with("declare");
      std::string sizeArg = isDeclaration ? "i64 immarg, " : "i64 -1, ";
      result.insert(arg, sizeArg);
      p = arg + sizeArg.size();
    }
  }
  replaceTypedLiteral("half -inf", "half 0xHFC00");
  replaceTypedLiteral("half inf", "half 0xH7C00");
  replaceTypedLiteral("half nan", "half 0xH7E00");
  replaceTypedLiteral("bfloat -inf", "bfloat 0xRFF80");
  replaceTypedLiteral("bfloat inf", "bfloat 0xR7F80");
  replaceTypedLiteral("bfloat nan", "bfloat 0xR7FC0");
  replaceTypedLiteral("float -inf", "float 0xFFF0000000000000");
  replaceTypedLiteral("float inf", "float 0x7FF0000000000000");
  replaceTypedLiteral("float nan", "float 0x7FF8000000000000");
  replaceTypedLiteral("double -inf", "double 0xFFF0000000000000");
  replaceTypedLiteral("double inf", "double 0x7FF0000000000000");
  replaceTypedLiteral("double nan", "double 0x7FF8000000000000");
  // LLVM 23 omits the type prefix for inf/NaN constants that appear as phi
  // operands (e.g. `phi float [ -inf, %entry ]`); Peano's older LLVM needs the
  // double-widened hex form. replaceTypedLiteral() cannot be reused: it rejects
  // a match whose preceding char is an identifier char, which would skip a
  // ", -inf" whose prior operand ends in one (e.g. "%x, -inf"). Instead match
  // on token boundaries around the bare literal itself.
  {
    auto rewriteBareLiteral = [&](llvm::StringRef from, llvm::StringRef to) {
      size_t pos = 0;
      while ((pos = result.find(from.data(), pos, from.size())) !=
             std::string::npos) {
        bool okBefore =
            pos == 0 ||
            !isIdentChar(static_cast<unsigned char>(result[pos - 1]));
        size_t after = pos + from.size();
        bool okAfter = after >= result.size() ||
                       !isIdentChar(static_cast<unsigned char>(result[after]));
        if (okBefore && okAfter) {
          result.replace(pos, from.size(), to.data(), to.size());
          pos += to.size();
        } else {
          pos += from.size();
        }
      }
    };
    rewriteBareLiteral("-inf", "0xFFF0000000000000");
    rewriteBareLiteral("inf", "0x7FF0000000000000");
    rewriteBareLiteral("nan", "0x7FF8000000000000");
  }
  // Strip ', align <N>' attributes. Retaining them causes Peano's capped-O1 opt
  // to skip vectorizing the matmul K-loop, scalarizing it into ~10x more
  // program memory and overflowing AIE core memory. Do not remove without
  // confirming the i8 matmul still fits program memory.
  //
  // Pre-link only. The merged module keeps its `align`: the kernel arrives
  // already annotated by its own clang, and re-stripping demotes an
  // over-aligned alloca to the type's ABI alignment (an `aie::linear_approx`
  // LUT falls from 64 to 4) and drops the load/store alignment the kernel was
  // compiled against, which miscompiles the core.
  if (stripAlign) {
    const std::string alignPat = ", align ";
    size_t pos = 0;
    while ((pos = result.find(alignPat, pos)) != std::string::npos) {
      size_t end = pos + alignPat.size();
      while (end < result.size() && result[end] >= '0' && result[end] <= '9')
        ++end;
      if (end > pos + alignPat.size())
        result.erase(pos, end - pos);
      else
        pos = end;
    }
  }
  // Rewrite 'f0x<8hex>' typed float literals (an LLVM 23 printing form) to the
  // double-widened '0x<16hex>' form Peano's older LLVM only accepts. Match only
  // at token boundaries: no identifier/sigil char before 'f' (avoids matching
  // value names like %f0xDEAD), and exactly 8 hex digits with a non-hex-digit
  // boundary after (avoids partial matches against longer hex strings).
  {
    const std::string f0xPfx = "f0x";
    size_t pos = 0;
    while ((pos = result.find(f0xPfx, pos)) != std::string::npos) {
      // Require a non-identifier, non-sigil character before 'f' to avoid
      // matching inside LLVM IR value names like '%f0xDEAD' or '@f0xBEEF'.
      if (pos > 0 && (isIdentChar(result[pos - 1]) || result[pos - 1] == '%' ||
                      result[pos - 1] == '@')) {
        pos += f0xPfx.size();
        continue;
      }
      size_t hexStart = pos + f0xPfx.size();
      size_t hexEnd = hexStart;
      while (hexEnd < result.size() && hexEnd < hexStart + 8 &&
             std::isxdigit(static_cast<unsigned char>(result[hexEnd])))
        ++hexEnd;
      // Require exactly 8 hex digits followed by a non-hex-digit boundary.
      bool trailingOk =
          hexEnd >= result.size() ||
          !std::isxdigit(static_cast<unsigned char>(result[hexEnd]));
      if (hexEnd - hexStart == 8 && trailingOk) {
        // Decode the 32-bit float bit pattern and re-encode as a double so
        // that Peano's older opt can parse the resulting hex literal.
        uint32_t fbits = static_cast<uint32_t>(
            std::stoul(result.substr(hexStart, 8), nullptr, 16));
        float fval;
        std::memcpy(&fval, &fbits, sizeof(fval));
        double dval = static_cast<double>(fval);
        uint64_t dbits;
        std::memcpy(&dbits, &dval, sizeof(dval));
        // Format as "0x" followed by 16 uppercase hex digits.
        std::string replacement = "0x";
        for (int shift = 60; shift >= 0; shift -= 4)
          replacement += "0123456789ABCDEF"[(dbits >> shift) & 0xFu];
        result.replace(pos, hexEnd - pos, replacement);
        pos += replacement.size();
      } else {
        pos = hexEnd;
      }
    }
  }
  // LLVM 24 prints a 'float'/'half' constant as a short decimal whenever that
  // decimal round-trips in the *narrow* type; older LLVM required it to round
  // trip as a double and printed hex otherwise. Peano's parser still demands
  // exact representability, so it rejects what LLVM 24 prints ("floating point
  // constant invalid for type") in both the typed position ('float
  // 1.100000e-01') and the bare operand one ('fmul float %x, 1.100000e-01').
  //
  // llvm/llvm-project@41c214f0b115 ("[AsmWriter] Change the output syntax of
  // floating-point literals", #190649) moved that round-trip check onto the
  // value's own semantics and retired the legacy '0x<16hex>' spelling for
  // 'f0x', which the pass below this one rewrites.
  //
  // A bare operand takes its type from the instruction, so tokenize and track
  // the last type keyword seen on the line. That also keeps a mixed-type line
  // ('call void @f(float 1.1, double 2.2)') from being rewritten under the
  // wrong semantics. 'bfloat' and 'double' set the type but are left alone:
  // double decimals always round-trip, and bfloat is handled below.
  {
    enum class FPTy { None, Float, Half };
    std::string out;
    out.reserve(result.size());
    FPTy lineTy = FPTy::None;
    size_t i = 0;
    auto isNameChar = [](char c) {
      return std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
             c == '.' || c == '$' || c == '-';
    };
    while (i < result.size()) {
      char c = result[i];
      if (c == '\n') {
        lineTy = FPTy::None;
        out += c;
        ++i;
        continue;
      }
      // Copy quoted strings verbatim: they can hold anything that looks like a
      // literal (a version string, an escaped byte array). A quote inside one
      // is printed as `\22`, never `\"`, so the next bare quote terminates it.
      if (c == '"') {
        size_t end = result.find('"', i + 1);
        end = (end == std::string::npos) ? result.size() : end + 1;
        out.append(result, i, end - i);
        i = end;
        continue;
      }
      // Names (%v, @g, !12) and keywords are consumed whole, so a digit inside
      // one is never mistaken for a constant, and 'bfloat' never matches as
      // 'float'.
      if (c == '%' || c == '@' || c == '!' || c == '#' ||
          std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        size_t end = i + 1;
        while (end < result.size() && isNameChar(result[end]))
          ++end;
        llvm::StringRef word(result.data() + i, end - i);
        if (word == "float")
          lineTy = FPTy::Float;
        else if (word == "half")
          lineTy = FPTy::Half;
        else if (word == "bfloat" || word == "double")
          lineTy = FPTy::None;
        out.append(result, i, end - i);
        i = end;
        continue;
      }
      bool isNumStart =
          std::isdigit(static_cast<unsigned char>(c)) ||
          ((c == '-' || c == '+') && i + 1 < result.size() &&
           std::isdigit(static_cast<unsigned char>(result[i + 1])));
      if (!isNumStart) {
        out += c;
        ++i;
        continue;
      }
      size_t end = i + 1;
      while (end < result.size() &&
             (std::isalnum(static_cast<unsigned char>(result[end])) ||
              result[end] == '.' ||
              ((result[end] == '+' || result[end] == '-') &&
               (result[end - 1] == 'e' || result[end - 1] == 'E'))))
        ++end;
      llvm::StringRef num(result.data() + i, end - i);
      // Only decimals are at risk; the hex forms already say exactly what they
      // mean, and an integer is not a float constant.
      bool isDecimal =
          num.contains('.') || num.contains('e') || num.contains('E');
      if (lineTy == FPTy::None || !isDecimal || num.starts_with("0x")) {
        out.append(num.data(), num.size());
        i = end;
        continue;
      }
      const llvm::fltSemantics &sem = lineTy == FPTy::Half
                                          ? llvm::APFloat::IEEEhalf()
                                          : llvm::APFloat::IEEEsingle();
      llvm::APFloat val(llvm::APFloat::IEEEdouble());
      auto parsed =
          val.convertFromString(num, llvm::APFloat::rmNearestTiesToEven);
      if (!parsed) {
        llvm::consumeError(parsed.takeError());
        out.append(num.data(), num.size());
        i = end;
        continue;
      }
      bool lost = false;
      llvm::APFloat narrow = val;
      narrow.convert(sem, llvm::APFloat::rmNearestTiesToEven, &lost);
      if (!lost) {
        // Exactly representable, so Peano accepts the decimal as printed.
        out.append(num.data(), num.size());
        i = end;
        continue;
      }
      // 'half' takes its own 16-bit hex form; 'float' is spelled as the double
      // it widens to.
      uint64_t bits;
      int digits;
      if (lineTy == FPTy::Half) {
        out += "0xH";
        bits = narrow.bitcastToAPInt().getZExtValue();
        digits = 4;
      } else {
        bool ignored = false;
        llvm::APFloat wide = narrow;
        wide.convert(llvm::APFloat::IEEEdouble(),
                     llvm::APFloat::rmNearestTiesToEven, &ignored);
        out += "0x";
        bits = wide.bitcastToAPInt().getZExtValue();
        digits = 16;
      }
      for (int shift = (digits - 1) * 4; shift >= 0; shift -= 4)
        out += "0123456789ABCDEF"[(bits >> shift) & 0xFu];
      i = end;
    }
    result = std::move(out);
  }
  // Rewrite decimal bfloat16 literals ('bfloat N.NNe+NN', an LLVM 23 printing
  // form) to the bit-exact '0xR<4hex>' form Peano's older LLVM only accepts.
  // The float32->bfloat16 conversion uses round-to-nearest-even so the encoded
  // bits match the original constant exactly.
  {
    // Match "bfloat" followed by a decimal number (not already 0x-prefixed).
    const std::string bfPfx = "bfloat ";
    size_t pos = 0;
    while ((pos = result.find(bfPfx, pos)) != std::string::npos) {
      size_t numStart = pos + bfPfx.size();
      // Skip if this is already a hex constant (0x / 0xR / 0xH …).
      if (numStart + 1 < result.size() && result[numStart] == '0' &&
          result[numStart + 1] == 'x') {
        pos = numStart;
        continue;
      }
      // Collect an optional leading '-' and then digits/dot/exponent chars.
      size_t numEnd = numStart;
      if (numEnd < result.size() && result[numEnd] == '-')
        ++numEnd;
      // Must start with a digit.
      if (numEnd >= result.size() ||
          !std::isdigit(static_cast<unsigned char>(result[numEnd]))) {
        pos = numStart;
        continue;
      }
      while (numEnd < result.size() &&
             (std::isdigit(static_cast<unsigned char>(result[numEnd])) ||
              result[numEnd] == '.' || result[numEnd] == 'e' ||
              result[numEnd] == 'E' || result[numEnd] == '+' ||
              result[numEnd] == '-'))
        ++numEnd;
      std::string numStr = result.substr(numStart, numEnd - numStart);
      // Parse as float32 and convert to bfloat16 via round-to-nearest-even.
      // bfloat16 shares the float32 exponent; its 16 bits are the top 16 bits
      // of float32 (after RNE rounding).
      char *endp = nullptr;
      float fval = std::strtof(numStr.c_str(), &endp);
      // Require that strtof consumed the *entire* numStr; if it stopped early
      // (e.g. on an unexpected character) we must not rewrite the token using
      // a partially-parsed value.
      if (!endp || endp != numStr.c_str() + numStr.size()) {
        pos = numEnd;
        continue;
      }
      uint32_t f32bits;
      std::memcpy(&f32bits, &fval, sizeof(f32bits));
      // Round-to-nearest-even: add 0x7FFF + the LSB of the bfloat16 position.
      uint32_t lsb = (f32bits >> 16) & 1u;
      uint16_t bf16bits =
          static_cast<uint16_t>((f32bits + 0x7FFFu + lsb) >> 16);
      // Format as "bfloat 0xR" followed by 4 uppercase hex digits.
      std::string replacement = "bfloat 0xR";
      for (int shift = 12; shift >= 0; shift -= 4)
        replacement += "0123456789ABCDEF"[(bf16bits >> shift) & 0xFu];
      result.replace(pos, numEnd - pos, replacement);
      pos += replacement.size();
    }
  }
  // Second pass: bfloat constants without an explicit type prefix (e.g.
  // 'fmul bfloat %x, 1.445310e+00'), where LLVM 23 omits the type keyword
  // before the constant operand. Scan line-by-line; on any line whose
  // instruction type is 'bfloat', convert every bare decimal float operand.
  {
    auto convertDecimalBf = [&](uint32_t f32bits) -> std::string {
      uint32_t lsb = (f32bits >> 16) & 1u;
      uint16_t bf16bits =
          static_cast<uint16_t>((f32bits + 0x7FFFu + lsb) >> 16);
      std::string r = "0xR";
      for (int sh = 12; sh >= 0; sh -= 4)
        r += "0123456789ABCDEF"[(bf16bits >> sh) & 0xFu];
      return r;
    };
    // We need to process line-by-line, so work on a copy split into lines.
    std::string out;
    out.reserve(result.size());
    size_t lineStart = 0;
    while (lineStart <= result.size()) {
      size_t lineEnd = result.find('\n', lineStart);
      bool hasNewline = (lineEnd != std::string::npos);
      if (!hasNewline)
        lineEnd = result.size();
      std::string line = result.substr(lineStart, lineEnd - lineStart);
      // Only process lines where 'bfloat' appears as a type (i.e., the word
      // 'bfloat' is in the instruction line, not as part of an identifier).
      // Simple heuristic: look for " bfloat " or " bfloat," or "= bfloat ".
      bool hasBfloatType = line.find(" bfloat ") != std::string::npos ||
                           line.find(" bfloat,") != std::string::npos ||
                           line.find("= bfloat\n") != std::string::npos;
      if (hasBfloatType) {
        // Scan for bare decimal float literals: must be preceded by ", " (or
        // "( ") and start with an optional '-' then a digit.
        std::string newLine;
        newLine.reserve(line.size());
        size_t lp = 0;
        while (lp < line.size()) {
          // Look for ", " or "( " before a potential decimal.
          size_t sep = line.find(", ", lp);
          size_t paren = line.find("( ", lp);
          size_t next =
              (sep < paren ? sep : paren); // take whichever comes first
          if (next == std::string::npos) {
            newLine += line.substr(lp);
            break;
          }
          size_t afterSep = next + 2; // skip ", " or "( "
          newLine += line.substr(lp, afterSep - lp);
          lp = afterSep;
          // Try to parse a decimal float starting here.
          size_t numStart = lp;
          size_t numEnd = numStart;
          if (numEnd < line.size() && line[numEnd] == '-')
            ++numEnd;
          if (numEnd >= line.size() ||
              !std::isdigit(static_cast<unsigned char>(line[numEnd]))) {
            continue; // not a decimal, keep scanning
          }
          while (numEnd < line.size() &&
                 (std::isdigit(static_cast<unsigned char>(line[numEnd])) ||
                  line[numEnd] == '.' || line[numEnd] == 'e' ||
                  line[numEnd] == 'E' || line[numEnd] == '+' ||
                  line[numEnd] == '-'))
            ++numEnd;
          std::string numStr = line.substr(numStart, numEnd - numStart);
          // Skip if it already looks like an integer (no '.', 'e', or 'E').
          bool isFloat = numStr.find('.') != std::string::npos ||
                         numStr.find('e') != std::string::npos ||
                         numStr.find('E') != std::string::npos;
          if (!isFloat) {
            newLine += numStr;
            lp = numEnd;
            continue;
          }
          char *ep = nullptr;
          float fv = std::strtof(numStr.c_str(), &ep);
          if (!ep || ep != numStr.c_str() + numStr.size()) {
            newLine += numStr;
            lp = numEnd;
            continue;
          }
          uint32_t f32bits;
          std::memcpy(&f32bits, &fv, sizeof(f32bits));
          newLine += convertDecimalBf(f32bits);
          lp = numEnd;
        }
        line = std::move(newLine);
      }
      out += line;
      if (hasNewline)
        out += '\n';
      lineStart = lineEnd + (hasNewline ? 1 : result.size() + 1);
    }
    result = std::move(out);
  }
  return result;
}

// Downgrade LLVM IR for the Chess toolchain, whose LLVM is older and rejects
// modern memory/capture attributes.
inline std::string downgradeIRForChess(llvm::StringRef ir) {
  std::string result = ir.str();
  auto replaceAll = [&](llvm::StringRef from, llvm::StringRef to) {
    for (size_t p = 0; (p = result.find(from.str(), p)) != std::string::npos;) {
      result.replace(p, from.size(), to.str());
      p += to.size();
    }
  };
  replaceAll("memory(none)", "readnone");
  replaceAll("memory(read)", "readonly");
  replaceAll("memory(write)", "writeonly");
  replaceAll("memory(argmem: readwrite)", "argmemonly");
  replaceAll("memory(argmem: read)", "argmemonly readonly");
  replaceAll("memory(argmem: write)", "argmemonly writeonly");
  replaceAll("memory(inaccessiblemem: readwrite)", "inaccessiblememonly");
  replaceAll("memory(inaccessiblemem: read)", "inaccessiblememonly readonly");
  replaceAll("memory(inaccessiblemem: write)", "inaccessiblememonly writeonly");
  replaceAll("memory(argmem: readwrite, inaccessiblemem: readwrite)",
             "inaccessiblemem_or_argmemonly");
  replaceAll("memory(argmem: read, inaccessiblemem: read)",
             "inaccessiblemem_or_argmemonly readonly");
  replaceAll("memory(argmem: write, inaccessiblemem: write)",
             "inaccessiblemem_or_argmemonly writeonly");
  replaceAll("captures(none)", "nocapture");
  replaceAll("getelementptr inbounds nuw", "getelementptr inbounds");
  // Drop `nocreateundeforpoison` along with its trailing whitespace.
  for (size_t p = 0;
       (p = result.find("nocreateundeforpoison", p)) != std::string::npos;) {
    size_t end = p + llvm::StringRef("nocreateundeforpoison").size();
    while (end < result.size() && (result[end] == ' ' || result[end] == '\t'))
      ++end;
    result.erase(p, end - p);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Pass-pipeline factories
//
// Each `getXyzPipeline` returns a fully-configured `PassManager` ready to be
// `run()` on the appropriate input. Returns nullptr on construction failure
// (e.g. `parsePassPipeline` rejected an option string).
//===----------------------------------------------------------------------===//

// Tile placement (`aie-place-tiles`), nested under DeviceOp.
inline std::unique_ptr<mlir::PassManager>
getPlacementPipeline(mlir::MLIRContext *ctx, int coresPerCol,
                     xilinx::AIE::PlacerType placerType, int saSeed) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  xilinx::AIE::AIEPlaceTilesOptions opts;
  opts.clPlacerType = placerType;
  opts.clCoresPerCol = coresPerCol;
  opts.clSASeed = saSeed;
  pm->nest<xilinx::AIE::DeviceOp>().addPass(
      xilinx::AIE::createAIEPlaceTilesPass(opts));
  return pm;
}

// Trace flow + trace-config emission, nested under DeviceOp.
inline std::unique_ptr<mlir::PassManager>
getTracePipeline(mlir::MLIRContext *ctx) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  auto &dpm = pm->nest<xilinx::AIE::DeviceOp>();
  dpm.addPass(xilinx::AIE::createAIEInsertTraceFlowsPass());
  dpm.addPass(xilinx::AIE::createAIETraceToConfigPass());
  dpm.addPass(xilinx::AIE::createAIETraceRegPackWritesPass());
  dpm.addPass(xilinx::AIEX::createAIEXInlineTraceConfigPass());
  return pm;
}

// Vector → AIEVec → buffer/lock/DMA setup → control-overlay → SCF lowering.
// Operates on the whole module; the inner pipeline nests under DeviceOp.
// Inspects `mod` for target arch (drives `convert-vector-to-aievec` opts).
inline std::unique_ptr<mlir::PassManager>
getInputWithAddressesPipeline(mlir::MLIRContext *ctx, mlir::ModuleOp mod,
                              llvm::StringRef allocScheme, bool dynamicObjFifos,
                              bool packetSwObjFifos, bool ctrlPktOverlay,
                              bool bf16Emulation, bool loadPdiToCtrlPkt = false,
                              bool skipObjectFifoVerify = false) {
  using namespace xilinx::AIE;
  namespace X = xilinx::AIEX;
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  std::string target = detectAIETarget(mod);
  if (target == "aie2" || target == "aieml" || target == "aie2p")
    if (mlir::failed(mlir::parsePassPipeline(
            llvm::formatv("convert-vector-to-aievec{{aie-target={0}{1}}",
                          target, bf16Emulation ? " bf16-emulation=true" : "")
                .str(),
            *pm)))
      return nullptr;
  pm->addPass(mlir::createLowerAffinePass());
  pm->addPass(createAIECanonicalizeDevicePass());
  // Lower scratchpad runtime parameters (module-level). Must run before
  // AIEAssignLockIDs (new locks need IDs) and before address assignment (new
  // buffers need addresses). params.txt is materialized as a separate graph
  // edge, so no `outputParamsFile` is set here.
  pm->addPass(X::createAIELowerScratchpadParametersPass());

  // The control-overlay pass is module-level (it may emit a standalone
  // `@ctrl_pkt_overlay` device). With `ctrlPktOverlay` it must run BEFORE
  // objectFIFO lowering so the overlay claims its shim DMA channels first and
  // the objectFIFO transform (DMAChannelAnalysis) works around them. Otherwise
  // it runs after objectFIFO + tile-ctrl-id assignment (below).
  if (ctrlPktOverlay) {
    if (mlir::failed(mlir::parsePassPipeline(
            llvm::formatv(
                "aie-generate-column-control-overlay{{route-shim-to-tile-ctrl="
                "true emit-standalone-overlay={0}}",
                loadPdiToCtrlPkt)
                .str(),
            *pm)))
      return nullptr;
  }

  mlir::OpPassManager &dpm = pm->nest<DeviceOp>();
  // The stateful transform always emits the dynamic (runtime) buffer addressing
  // and lock bookkeeping. When dynamic objectFifos are disabled, the
  // aie-objectFifo-unroll pass below unrolls the loops that carry objectFifo
  // accesses and folds the (now loop-invariant) runtime bookkeeping into a
  // static, unrolled lowering.
  if (mlir::failed(mlir::parsePassPipeline(
          llvm::formatv(
              "aie-objectFifo-stateful-transform{{packet-sw-objFifos={0} "
              "skip-verify={1}}",
              packetSwObjFifos, skipObjectFifoVerify)
              .str(),
          dpm)))
    return nullptr;
  // Unroll the objectFifo loops (folding the runtime bookkeeping into the
  // static lowering). `default-dynamic=true` flips the default to the
  // loop-preserving form; per-core `dynamic_objfifo_lowering` attributes
  // override it either way. Either way the unroll hints are stripped.
  if (mlir::failed(mlir::parsePassPipeline(
          llvm::formatv("aie-objectFifo-unroll{{default-dynamic={0}}",
                        dynamicObjFifos)
              .str(),
          dpm)))
    return nullptr;
  // Assign IDs to the ID-less locks the objectFifo lowering creates (and to any
  // user locks without an ID).
  dpm.addPass(createAIEAssignLockIDsPass());
  dpm.addPass(X::createAIEReserveRuntimeBDIDsPass());
  dpm.addPass(createAIEAssignBufferDescriptorIDsPass());
  dpm.addPass(createAIELowerCascadeFlowsPass());
  dpm.addPass(X::createAIEBroadcastPacketPass());
  dpm.addPass(X::createAIELowerMulticastPass());
  dpm.addPass(createAIEAssignTileCtrlIDsPass());

  // Without `ctrlPktOverlay`, the (module-level) overlay pass runs here, after
  // tile-ctrl-id assignment. Break out of the device nest to run it, then
  // resume with a new device nest for the remaining per-device passes.
  if (!ctrlPktOverlay) {
    if (mlir::failed(
            mlir::parsePassPipeline("aie-generate-column-control-overlay{route-"
                                    "shim-to-tile-ctrl=false}",
                                    *pm)))
      return nullptr;
  }

  mlir::OpPassManager &dpm2 = pm->nest<DeviceOp>();
  AIEAssignBufferAddressesOptions bufOpts;
  bufOpts.clAllocScheme = allocScheme.str();
  dpm2.addPass(createAIEAssignBufferAddressesPass(bufOpts));
  dpm2.addPass(createAIEAssignCoreLinkFilesPass());
  dpm2.addPass(createAIEVectorTransferLoweringPass());
  pm->addPass(xilinx::AIEX::createAIESCFToControlFlowPass());
  return pm;
}

// Routing (`aie-create-pathfinder-flows`), nested under DeviceOp.
inline std::unique_ptr<mlir::PassManager>
getRoutingPipeline(mlir::MLIRContext *ctx) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  pm->nest<xilinx::AIE::DeviceOp>().addPass(
      xilinx::AIE::createAIEPathfinderPass());
  return pm;
}

// Per-core LLVM-lowering pipeline. Destructive: extracts the CoreOp at
// (col, row) and removes the `aie.device` wrapper. col/row=-1 means
// "all cores" (unified mode).
inline std::unique_ptr<mlir::PassManager>
getCoreLLVMLoweringPipeline(mlir::MLIRContext *ctx, llvm::StringRef deviceName,
                            int col, int row, llvm::StringRef aieTarget) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  mlir::OpPassManager &devicePm = pm->nest<xilinx::AIE::DeviceOp>();
  devicePm.addPass(xilinx::AIE::createAIELocalizeLocksPass());
  devicePm.addPass(xilinx::AIE::createAIENormalizeAddressSpacesPass());
  devicePm.addPass(xilinx::AIEX::createAIETransformBfpTypesPass());

  xilinx::AIE::AIECoreToStandardOptions coreOpts;
  coreOpts.deviceName = deviceName.str();
  coreOpts.tileCol = col;
  coreOpts.tileRow = row;
  pm->addPass(xilinx::AIE::createAIECoreToStandardPass(coreOpts));

  pm->addPass(xilinx::AIEX::createAIEXToStandardPass());

  xilinx::ConvertAIEVecToLLVMOptions aievecOpts;
  aievecOpts.aieTarget = llvm::StringRef(aieTarget).lower();
  pm->addPass(xilinx::aievec::createConvertAIEVecToLLVMPass(aievecOpts));

  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::memref::createExpandStridedMetadataPass());
  pm->addPass(mlir::createLowerAffinePass());
  {
    // LLVM 24 moved the min/max expansion patterns behind arith-expand options
    // that default to false, on the grounds that arith-to-llvm can lower these
    // ops straight to the llvm.intr.{maxnum,minnum,...} intrinsics. Peano's
    // AIE2 GlobalISel has no rule for the resulting G_FMAXNUM/G_FMINNUM, so llc
    // aborts with "unable to legalize instruction" and the core never builds.
    // Keep the cmpf/select expansion for the floating-point ops.
    //
    // Only the float half is needed: scalar integer min/max is already taken
    // care of before this point, so include-min-max-i is left at its default.
    mlir::arith::ArithExpandOpsPassOptions arithOpts;
    arithOpts.includeMinMaxF = true;
    pm->addPass(mlir::arith::createArithExpandOpsPass(arithOpts));
  }
  pm->addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm->addPass(mlir::createConvertFuncToLLVMPass(
      mlir::ConvertFuncToLLVMPassOptions{/*useBarePtrCallConv=*/true}));
  {
    mlir::ConvertToLLVMPassOptions llvmOpts;
    llvmOpts.useDynamic = true;
    pm->addPass(mlir::createConvertToLLVMPass(llvmOpts));
  }
  pm->addPass(mlir::createConvertVectorToLLVMPass());
  pm->addPass(mlir::createUBToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  return pm;
}

// Translate a lowered ModuleOp to textual LLVM IR. Item-shaped so it can be
// used directly as a graph map action.
inline mlir::LogicalResult
translateToLLVMIR(const Item<mlir::OwningOpRef<mlir::ModuleOp>> &item,
                  Item<std::string> &out) {
  llvm::LLVMContext llvmCtx;
  auto llvmMod = mlir::translateModuleToLLVMIR(item.get().get(), llvmCtx);
  if (!llvmMod) {
    llvm::errs() << "aiecc: translateModuleToLLVMIR failed\n";
    return mlir::failure();
  }
  std::string txt;
  llvm::raw_string_ostream os(txt);
  llvmMod->print(os, nullptr);
  out.value = std::move(txt);
  return mlir::success();
}

// Apply the per-core LLVM lowering to a module clone. col/row=-1 means
// "all cores" (unified mode); otherwise the named core's body.
inline mlir::LogicalResult
loweringPipeline(mlir::ModuleOp src, llvm::StringRef devName, int col, int row,
                 Item<mlir::OwningOpRef<mlir::ModuleOp>> &out) {
  mlir::OwningOpRef<mlir::ModuleOp> clone = src.clone();
  auto pm = getCoreLLVMLoweringPipeline(clone->getContext(), devName, col, row,
                                        detectAIETarget(src, devName));
  if (mlir::failed(pm->run(*clone)))
    return mlir::failure();
  out.value = std::move(clone);
  return mlir::success();
}

// Lower a device once and carve the result into one module per core.
//
// Two things have to happen in the same place. The lowering is what we want to
// run once -- per-core lowering re-runs it against a clone of the whole design
// and throws away every core but one. The carve then needs to know which tile
// owns each buffer, and that is only legible before lowering: `memref.global`
// loses any attribute we might hang on it when it becomes `llvm.mlir.global`.
// So this takes the pre-lowering DeviceOp, reads ownership off its BufferOps,
// and applies it after.
//
// Ownership matters because `AIEBufferToStandard` drops the initializer for
// buffers a core does not own, so the data lands in exactly one core's object.
// It keys that off the tile coordinates it was given, and unified lowering
// gives it none, so every carved module starts with every initializer and the
// globals are public -- symbol DCE will not touch them. Reproducing the drop
// here is what keeps a core's object carrying only its own data.
//
// Cores the caller says it will not compile -- a pre-baked `elf_file` is used
// verbatim -- are skipped, so this keys the same set as `perCore`.
//
// Keys match `coreKey`, so the results feed the per-core object subgraph.
inline mlir::FailureOr<
    std::vector<std::pair<std::string, mlir::OwningOpRef<mlir::ModuleOp>>>>
splitLoweredCores(const Item<OpInModule<xilinx::AIE::DeviceOp>> &devItem,
                  llvm::function_ref<bool(xilinx::AIE::CoreOp)> shouldCompile) {
  xilinx::AIE::DeviceOp dev = devItem.get().op;
  std::string devName = dev.getSymName().str();

  // Cores this device will actually compile, by coordinate.
  llvm::DenseSet<std::pair<int, int>> compiled;
  dev.walk([&](xilinx::AIE::CoreOp c) {
    if (!shouldCompile(c))
      return;
    auto tile = mlir::cast<xilinx::AIE::TileOp>(c.getTile().getDefiningOp());
    compiled.insert({tile.getCol(), tile.getRow()});
  });

  // Buffer symbol -> owning tile, read before the lowering erases the tiles.
  llvm::StringMap<std::pair<int, int>> owner;
  dev.walk([&](xilinx::AIE::BufferOp buf) {
    auto tile = mlir::cast<xilinx::AIE::TileOp>(buf.getTile().getDefiningOp());
    owner[buf.name().getValue()] = {tile.getCol(), tile.getRow()};
  });

  Item<mlir::OwningOpRef<mlir::ModuleOp>> lowered;
  if (mlir::failed(loweringPipeline(devItem.get().module.get(), devName, -1, -1,
                                    lowered)))
    return mlir::failure();

  // `core_<col>_<row>` is what AIECoreToStandardFunc emits. Match the shape
  // rather than the prefix: a hand-written `core_helper` is not a core.
  auto coreCoords =
      [](llvm::StringRef name) -> std::optional<std::pair<int, int>> {
    if (!name.consume_front("core_"))
      return std::nullopt;
    llvm::StringRef colStr, rowStr;
    std::tie(colStr, rowStr) = name.split('_');
    int col, row;
    if (colStr.empty() || rowStr.empty() || colStr.getAsInteger(10, col) ||
        rowStr.getAsInteger(10, row))
      return std::nullopt;
    return std::make_pair(col, row);
  };

  // Name plus coordinates, so the loop below never has to re-parse the name
  // and unwrap the optional a second time.
  llvm::SmallVector<std::pair<std::string, std::pair<int, int>>> cores;
  lowered.get().get().walk([&](mlir::LLVM::LLVMFuncOp f) {
    auto coords = coreCoords(f.getSymName());
    if (coords && compiled.contains(*coords))
      cores.emplace_back(f.getSymName().str(), *coords);
  });

  std::vector<std::pair<std::string, mlir::OwningOpRef<mlir::ModuleOp>>> out;
  out.reserve(cores.size());
  for (const auto &core : cores) {
    llvm::StringRef keep = core.first;
    std::pair<int, int> keepCoords = core.second;
    mlir::OwningOpRef<mlir::ModuleOp> clone = lowered.get().get().clone();

    llvm::SmallVector<mlir::Operation *> drop;
    clone->walk([&](mlir::LLVM::LLVMFuncOp f) {
      if (coreCoords(f.getSymName()) && f.getSymName() != keep)
        drop.push_back(f);
    });
    for (mlir::Operation *op : drop)
      op->erase();

    clone->walk([&](mlir::LLVM::GlobalOp g) {
      auto it = owner.find(g.getSymName());
      if (it != owner.end() && it->second != keepCoords)
        g.removeValueAttr();
    });

    mlir::PassManager pm(clone->getContext());
    pm.addPass(mlir::createSymbolDCEPass());
    if (mlir::failed(pm.run(*clone)))
      return mlir::failure();
    out.emplace_back(devName + "_" + keep.str(), std::move(clone));
  }
  return out;
}

// DMA→NPU lowering. Expects runtime sequences to already be materialized
// (getMaterializeRuntimeSeqPipeline).
inline std::unique_ptr<mlir::PassManager>
getNpuDmaLoweringPipeline(mlir::MLIRContext *ctx) {
  namespace X = xilinx::AIEX;
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  auto &dpm = pm->nest<xilinx::AIE::DeviceOp>();
  dpm.addPass(X::createAIEMaterializeBDChainsPass());
  dpm.addPass(X::createAIESubstituteShimDMAAllocationsPass());
  dpm.addPass(X::createAIEUnrollRuntimeSequenceLoopsPass());
  dpm.addPass(mlir::createCanonicalizerPass());
  // Decompose oversized non-contiguous ND transfers (wrap/stride exceeding the
  // hardware BD field limits) into legal sub-transfers before BD lowering.
  dpm.addPass(X::createAIEDecomposeLargeDmaBdPass());
  // A runtime-bound scf.for that survived unroll takes the dynamic BD pool path
  // (rewritten to pool pop/push, ids drawn at runtime); the static allocator
  // below skips it. Straight-line sequences fall through unchanged.
  dpm.addPass(X::createAIELowerDynamicBDPoolPass());
  dpm.addPass(mlir::createCanonicalizerPass());
  dpm.addPass(X::createAIEAssignRuntimeSequenceBDIDsPass());
  dpm.addPass(X::createAIEDMATasksToNPUPass());
  // Expand dma_channel_reset_for into its re-arm trio (dma_channel_reset +
  // set_lock + a START_QUEUE re-push) and lower the resulting dma_channel_reset
  // ops to maskwrite32 -- one pass. Runs before aie-dma-to-npu so the emitted
  // push_queue is lowered with the other queue pushes, and before
  // aie-lower-set- lock so the emitted set_lock ops are lowered too. The head
  // bd_id + repeat it re-pushes were folded into the objectfifo_rearm_binding
  // by aie-assign-bd-ids.
  dpm.addPass(X::createAIELowerDmaChannelResetPass());
  dpm.addPass(X::createAIEDmaToNpuPass());
  dpm.addPass(X::createAIELowerSetLockPass());
  dpm.addPass(X::createAIELowerCoreResetPass());
  return pm;
}

// `load_pdi { device_ref }` → explicit write32 or control-packet sequences.
// With `ctrlPkt=false` the referenced device's configuration is emitted as
// `write32`/`blockwrite` ops; with `ctrlPkt=true` it is emitted as
// `aiex.npu.control_packet` ops (which a later ctrl-packet-to-dma pass streams
// in), preceded by a `load_pdi @ctrl_pkt_overlay`.
inline std::unique_ptr<mlir::PassManager>
getExpandLoadPdiPipeline(mlir::MLIRContext *ctx, bool ctrlPkt = false) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  std::string expandPipeline = std::string("aie-expand-load-pdi{ctrl-pkt=") +
                               (ctrlPkt ? "true" : "false") + "}";
  if (mlir::failed(mlir::parsePassPipeline(expandPipeline, *pm)))
    return nullptr;
  if (ctrlPkt)
    pm->nest<xilinx::AIE::DeviceOp>().addPass(
        xilinx::AIEX::createAIELegalizeControlPacketPass());
  return pm;
}

// Runtime-sequence materialization (module-level).
inline std::unique_ptr<mlir::PassManager>
getMaterializeRuntimeSeqPipeline(mlir::MLIRContext *ctx) {
  namespace X = xilinx::AIEX;
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  pm->addPass(X::createAIEMaterializeRuntimeSequencesPass());
  return pm;
}

// Per-device DMA→NPU lowering, for both user DMA ops and the DMA ops lowered
// from control packets.
inline std::unique_ptr<mlir::PassManager>
getPerDeviceDmaLoweringPipeline(mlir::MLIRContext *ctx) {
  namespace X = xilinx::AIEX;
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  auto &dpm = pm->nest<xilinx::AIE::DeviceOp>();
  dpm.addPass(X::createAIEMaterializeBDChainsPass());
  dpm.addPass(X::createAIESubstituteShimDMAAllocationsPass());
  dpm.addPass(X::createAIEAssignRuntimeSequenceBDIDsPass());
  dpm.addPass(mlir::createCanonicalizerPass());
  dpm.addPass(X::createAIEDMATasksToNPUPass());
  dpm.addPass(X::createAIEDmaToNpuPass());
  dpm.addPass(X::createAIELowerSetLockPass());
  return pm;
}

// Convert legalized control-packet ops into DMA task ops (device-nested). The
// subsequent DMA→NPU lowering is done by getPerDeviceDmaLoweringPipeline.
inline std::unique_ptr<mlir::PassManager>
getCtrlPktToDmaPipeline(mlir::MLIRContext *ctx) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  pm->nest<xilinx::AIE::DeviceOp>().addPass(
      xilinx::AIEX::createAIECtrlPacketToDmaPass());
  return pm;
}

// Transaction generation: `convert-aie-to-transaction{elf-dir device-name}`,
// nested at the device level. The pass embeds each core's compiled ELF; with
// absolute `elf_file` attributes already patched into the IR, `elfDir` is only
// a fallback for relative paths. Builds a `@configure` runtime sequence of
// write32/blockwrite ops describing the device configuration. Returns nullptr
// if the option string fails to parse.
inline std::unique_ptr<mlir::PassManager>
getTransactionPipeline(mlir::MLIRContext *ctx, llvm::StringRef elfDir,
                       llvm::StringRef devName) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  std::string pipelineStr = ("convert-aie-to-transaction{elf-dir=" + elfDir +
                             " device-name=" + devName + "}")
                                .str();
  auto &dpm = pm->nest<xilinx::AIE::DeviceOp>();
  if (mlir::failed(mlir::parsePassPipeline(pipelineStr, dpm)))
    return nullptr;
  return pm;
}

// CERT pipeline: transaction generation, followed by lowering the
// transaction into the microcontroller firmware's CERT representation. Mirrors
// `builtin.module(aie.device(convert-aie-to-transaction{elf-dir=<elfDir>}),
// aie-npu-to-cert, aie.device(cert-legalize-pages, aie-cert-verify))`.
// `aie-npu-to-cert` runs
// module-wide (its `device-name` option defaults to "main" and absorbs every
// other device's load-pdi references into that device's cert sections), so
// this pipeline is meant to run once over the whole module, not per device.
// Feeds `--aie-cert-to-asm` (AIETranslateToUcDma) to produce the control-code
// assembly consumed by the downstream CERT assembler. Returns nullptr on parse
// failure.
inline std::unique_ptr<mlir::PassManager>
getCertPipeline(mlir::MLIRContext *ctx, llvm::StringRef elfDir) {
  namespace X = xilinx::AIEX;
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  std::string txnPipelineStr =
      ("convert-aie-to-transaction{elf-dir=" + elfDir + "}").str();
  auto &dpm = pm->nest<xilinx::AIE::DeviceOp>();
  if (mlir::failed(mlir::parsePassPipeline(txnPipelineStr, dpm)))
    return nullptr;
  pm->addPass(X::createAIENpuToCertPass());
  auto &certDpm = pm->nest<xilinx::AIE::DeviceOp>();
  certDpm.addPass(X::createAIECertPagesPass());
  // aie-cert-verify runs last, after page legalization and placement lowering,
  // where final pages and uC (group) assignments are known. It turns CERT
  // firmware constraints (barrier co-location, one wait_tcts per actor, remote
  // barrier rendezvous, preempt-count parity, id budgets) into compile-time
  // diagnostics instead of hardware hangs.
  certDpm.addPass(X::createAIECertVerifyPass());
  return pm;
}

// Control-packet generation: transaction pipeline, then rewrite the
// transaction ops into control packets and legalize them — all in one device
// nest. Same `elfDir` / `devName` semantics as getTransactionPipeline. Returns
// nullptr on parse failure.
inline std::unique_ptr<mlir::PassManager>
getControlPacketPipeline(mlir::MLIRContext *ctx, llvm::StringRef elfDir,
                         llvm::StringRef devName) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  std::string pipelineStr = ("convert-aie-to-transaction{elf-dir=" + elfDir +
                             " device-name=" + devName + "}")
                                .str();
  auto &dpm = pm->nest<xilinx::AIE::DeviceOp>();
  if (mlir::failed(mlir::parsePassPipeline(pipelineStr, dpm)))
    return nullptr;
  dpm.addPass(xilinx::AIEX::createAIETxnToControlPacketPass());
  dpm.addPass(xilinx::AIEX::createAIELegalizeControlPacketPass());
  return pm;
}

// Lower legalized control packets into a DMA sequence the host streams in:
// `aie-ctrl-packet-to-dma` → `aie-dma-to-npu`.
inline std::unique_ptr<mlir::PassManager>
getControlPacketDmaPipeline(mlir::MLIRContext *ctx) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  auto &dpm = pm->nest<xilinx::AIE::DeviceOp>();
  dpm.addPass(xilinx::AIEX::createAIECtrlPacketToDmaPass());
  dpm.addPass(xilinx::AIEX::createAIEDmaToNpuPass());
  return pm;
}

} // namespace xilinx::aiecc

#endif // AIECC_IRTRANSFORMS_H
