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
#include "Utils.h"

#include "aie/Conversion/Passes.h"
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
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
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

// Strip LLVM-23-only features Peano's older opt/llc can't parse.
inline std::string downgradeIRForPeano(llvm::StringRef ir) {
  std::string result = ir.str();
  auto erasePattern = [&](llvm::StringRef pat, auto trail) {
    for (size_t p = 0; (p = result.find(pat.str(), p)) != std::string::npos;) {
      size_t end = p + pat.size();
      while (end < result.size() && trail(result[end]))
        ++end;
      result.erase(p, end - p);
    }
  };
  auto replaceAll = [&](llvm::StringRef from, llvm::StringRef to) {
    for (size_t p = 0; (p = result.find(from.str(), p)) != std::string::npos;) {
      result.replace(p, from.size(), to.str());
      p += to.size();
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
  replaceAll("getelementptr inbounds nuw", "getelementptr inbounds");
  erasePattern("nocreateundeforpoison",
               [](char c) { return c == ' ' || c == '\t'; });
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
  {
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
                              bool bf16Emulation) {
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
  mlir::OpPassManager &dpm = pm->nest<DeviceOp>();
  dpm.addPass(createAIEAssignLockIDsPass());
  // The stateful transform always emits the dynamic (runtime) buffer addressing
  // and lock bookkeeping. When dynamic objectFifos are disabled we then
  // statically unroll the loops that carry objectFifo accesses; the subsequent
  // mem2reg + canonicalize folds the (now loop-invariant) runtime bookkeeping
  // into the equivalent static, unrolled lowering.
  if (mlir::failed(mlir::parsePassPipeline(
          llvm::formatv("aie-objectFifo-stateful-transform{{dynamic-objFifos="
                        "{0} packet-sw-objFifos={1}}",
                        dynamicObjFifos, packetSwObjFifos)
              .str(),
          dpm)))
    return nullptr;
  // Unroll the objectFifo loops after the dynamic codegen so that each unrolled
  // iteration maps to a fixed rotation of buffers/locks.
  if (!dynamicObjFifos)
    dpm.addPass(createAIEObjectFifoUnrollPass());
  // Promote the objectFifo bookkeeping counters (memref.alloca inside the
  // cores) to loop-carried SSA values, then fold the resulting constant buffer
  // selection and lock arithmetic.
  dpm.addPass(mlir::createMem2Reg());
  dpm.addPass(mlir::createCanonicalizerPass());
  if (!dynamicObjFifos) {
    // After unrolling by the objectFifo rotation period, the rotating
    // buffer/lock counter returns to its entry value each iteration, i.e. it is
    // loop-invariant. SCCP proves this and propagates the constant, after which
    // canonicalize folds every buffer/lock index_switch and the counter
    // arithmetic away, yielding the static unrolled lowering.
    dpm.addPass(mlir::createSCCPPass());
    dpm.addPass(mlir::createCanonicalizerPass());
  }
  dpm.addPass(createAIEAssignBufferDescriptorIDsPass());
  dpm.addPass(createAIELowerCascadeFlowsPass());
  dpm.addPass(X::createAIEBroadcastPacketPass());
  dpm.addPass(X::createAIELowerMulticastPass());
  dpm.addPass(createAIEAssignTileCtrlIDsPass());
  if (mlir::failed(mlir::parsePassPipeline(
          llvm::formatv("aie-generate-column-control-overlay{{route-shim-to-"
                        "tile-ctrl={0}}",
                        ctrlPktOverlay)
              .str(),
          dpm)))
    return nullptr;
  AIEAssignBufferAddressesOptions bufOpts;
  bufOpts.clAllocScheme = allocScheme.str();
  dpm.addPass(createAIEAssignBufferAddressesPass(bufOpts));
  dpm.addPass(createAIEAssignCoreLinkFilesPass());
  dpm.addPass(createAIEVectorTransferLoweringPass());
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
  pm->addPass(mlir::arith::createArithExpandOpsPass());
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

// NPU lowering (runtime-sequence materialization + DMA-to-NPU lowering).
// Nests inside the DeviceOp where appropriate.
inline std::unique_ptr<mlir::PassManager>
getNpuLoweringPipeline(mlir::MLIRContext *ctx, bool materialize = true) {
  namespace X = xilinx::AIEX;
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  if (materialize)
    pm->addPass(X::createAIEMaterializeRuntimeSequencesPass());
  auto &dpm = pm->nest<xilinx::AIE::DeviceOp>();
  dpm.addPass(X::createAIEMaterializeBDChainsPass());
  dpm.addPass(X::createAIESubstituteShimDMAAllocationsPass());
  dpm.addPass(X::createAIEUnrollRuntimeSequenceLoopsPass());
  dpm.addPass(mlir::createCanonicalizerPass());
  dpm.addPass(X::createAIEAssignRuntimeSequenceBDIDsPass());
  dpm.addPass(X::createAIEDMATasksToNPUPass());
  dpm.addPass(X::createAIEDmaToNpuPass());
  dpm.addPass(X::createAIELowerSetLockPass());
  return pm;
}

// `load_pdi { device_ref }` → explicit write sequences, avoiding a per-switch
// full PDI reload.
inline std::unique_ptr<mlir::PassManager>
getExpandLoadPdiPipeline(mlir::MLIRContext *ctx) {
  auto pm = std::make_unique<mlir::PassManager>(ctx);
  pm->addPass(xilinx::AIEX::createAIEExpandLoadPdiPass());
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
