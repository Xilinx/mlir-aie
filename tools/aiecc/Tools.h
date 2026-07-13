//===- Tools.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// External-tool integration for aiecc: the glue that either invokes a tool
// library in-process or locates/configures an external toolchain. Two flavors
// live here:
//
//   * In-process tool-library invocations — assembleElf (aiebu ELF assembler)
//     and assemblePdi (bootgen PDI generator) — the in-memory counterparts to
//     the declarative `ShellCommand` edges, used when the corresponding library
//     is linked so no subprocess is spawned.
//   * Toolchain path / configuration resolution — Chess toolchain locations,
//     the host `__AIEARCH__` define, and the host runtime libraries.
//
// These wrap or point at external tools; none of them are MLIR transforms (see
// IRTransforms.h for those).
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_TOOLS_H
#define AIECC_TOOLS_H

#include "Graph.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <vector>

#ifdef AIECC_HAS_AIEBU_LIBRARY
#include <aiebu/aiebu.h>
#endif

#ifdef AIECC_HAS_BOOTGEN_LIBRARY
#include <bootgen_c_api.h>
#endif

namespace xilinx::aiecc {

//===----------------------------------------------------------------------===//
// In-process tool-library invocations
//===----------------------------------------------------------------------===//

#ifdef AIECC_HAS_AIEBU_LIBRARY
// Assemble a loadable ELF in-memory from a transaction instruction blob via the
// linked aiebu static library (`aiebu_assembler_get_elf` with
// `blob_instr_transaction`), avoiding a subprocess. `buffer1` is the
// instruction binary, `buffer2` an optional control-packet binary (for combined
// control-packet ELFs), and `patchJson` an optional external_buffers.json
// patch. The ELF bytes are written to `out.filePath`. Only compiled when the
// aiebu library is linked; otherwise a declarative `aiebu-asm` ShellCommand
// edge is used (see the `instElf`/`ctrlpktElf` edges).
inline mlir::LogicalResult assembleElf(llvm::ArrayRef<char> buffer1,
                                       llvm::ArrayRef<char> buffer2,
                                       llvm::StringRef patchJson,
                                       Item<File> &out) {
  void *elfBuf = nullptr;
  int result = aiebu_assembler_get_elf(
      aiebu_assembler_buffer_type_blob_instr_transaction, buffer1.data(),
      buffer1.size(), buffer2.empty() ? nullptr : buffer2.data(),
      buffer2.size(), &elfBuf, patchJson.empty() ? nullptr : patchJson.data(),
      patchJson.size(), /*libs=*/nullptr, /*libpaths=*/nullptr,
      /*pm_ctrlpkts=*/nullptr, /*pm_ctrlpkt_size=*/0);
  if (result > 0 && elfBuf) {
    std::error_code ec;
    llvm::raw_fd_ostream os(out.filePath, ec);
    if (ec) {
      free(elfBuf);
      llvm::errs() << "aiecc: cannot write ELF '" << out.filePath
                   << "': " << ec.message() << "\n";
      return mlir::failure();
    }
    os.write(static_cast<char *>(elfBuf), result);
    free(elfBuf);
    out.value = File{};
    return mlir::success();
  }
  if (elfBuf)
    free(elfBuf);
  llvm::errs() << "aiecc: aiebu_assembler_get_elf failed (code " << result
               << ")\n";
  return mlir::failure();
}
#endif // AIECC_HAS_AIEBU_LIBRARY

#ifdef AIECC_HAS_BOOTGEN_LIBRARY
// Assemble a device's PDI from its BIF using the in-process bootgen C API
// (`bootgen_generate_pdi`), avoiding a subprocess. The BIF references the
// device's CDOs by absolute path; bootgen reads it and writes the PDI to
// `out.filePath`. Hard-fails with a diagnostic if bootgen rejects the inputs.
// Only compiled when the bootgen library is linked; otherwise a declarative
// `bootgen` ShellCommand edge is used (see the `pdi` edge).
inline mlir::LogicalResult assemblePdi(const Item<std::string> &bifItem,
                                       Item<File> &out) {
  char errMsg[1024] = {0};
  int rc = bootgen_generate_pdi(bifItem.asFile().c_str(), out.filePath.c_str(),
                                BOOTGEN_ARCH_VERSAL, /*overwrite=*/1, errMsg,
                                sizeof(errMsg));
  if (rc != BOOTGEN_SUCCESS) {
    llvm::errs() << "aiecc: bootgen_generate_pdi failed (code " << rc << ")";
    if (errMsg[0] != '\0')
      llvm::errs() << ": " << errMsg;
    llvm::errs() << '\n';
    return mlir::failure();
  }
  out.value = File{};
  return mlir::success();
}
#endif // AIECC_HAS_BOOTGEN_LIBRARY

//===----------------------------------------------------------------------===//
// Toolchain path & configuration resolution
//===----------------------------------------------------------------------===//

// Map an AIE target arch (lowercase) to the Chess toolchain's per-target
// directory name under `<aietools>/tps/lnx64/`. Empty for unsupported targets.
inline std::string getChessTarget(llvm::StringRef aieTarget) {
  std::string t = aieTarget.lower();
  if (t == "aie2" || t == "aieml")
    return "target_aie_ml";
  if (t == "aie2p")
    return "target_aie2p";
  if (t == "aie" || t == "aie1")
    return "target";
  llvm::errs() << "aiecc: unsupported AIE target for Chess toolchain: "
               << aieTarget << "\n";
  return "";
}

// Path to the Chess LLVM linker (`chess-llvm-link`) for the given AIE target,
// found under the aietools install's per-target tree. Empty if `arch` is not a
// Chess-supported target.
inline std::string getChessLLVMLinkPath(llvm::StringRef arch,
                                        llvm::StringRef aietoolsRoot) {
  std::string chessTarget = getChessTarget(arch);
  if (chessTarget.empty())
    return "";
  return llvm::formatv("{0}/tps/lnx64/{1}/bin/LNa64bin/chess-llvm-link",
                       aietoolsRoot, chessTarget);
}

// Path to the Chess intrinsic-wrapper IR (`chess_intrinsic_wrapper.ll`) that
// the core IR is linked against, under the mlir-aie install's per-arch runtime
// library.
inline std::string getChessIntrinsicWrapperPath(llvm::StringRef arch,
                                                llvm::StringRef installDir) {
  return llvm::formatv("{0}/aie_runtime_lib/{1}/chess_intrinsic_wrapper.ll",
                       installDir, arch.upper());
}

// The `__AIEARCH__` define a host program is compiled with, keyed off the
// device's AIE generation (AIE1=10, AIE2/AIEML=20, AIE2P=21).
inline std::string aieArchDefine(llvm::StringRef target) {
  std::string t = target.lower();
  if (t == "aie2" || t == "aieml")
    return "-D__AIEARCH__=20";
  if (t == "aie2p")
    return "-D__AIEARCH__=21";
  return "-D__AIEARCH__=10";
}

// Statically-available host runtime libraries shipped with the toolchain (not
// compilation outputs), resolved under `<installDir>/runtime_lib/<archTag>`.
// `archTag` is the host arch (the target triple's first component) plus a
// "-hsa" suffix when linking against the HSA runtime.
struct HostRuntimeLibs {
  std::string xaiengineInclude;
  std::string xaiengineLib;
  std::string memoryAllocator;
};

inline HostRuntimeLibs getHostRuntimeLibs(llvm::StringRef installDir,
                                          llvm::StringRef target,
                                          bool linkAgainstHsa) {
  std::string archTag = target.split('-').first.str();
  if (linkAgainstHsa)
    archTag += "-hsa";
  std::string base = llvm::formatv("{0}/runtime_lib/{1}", installDir, archTag);
  return {llvm::formatv("{0}/xaiengine/include", base),
          llvm::formatv("{0}/xaiengine/lib", base),
          llvm::formatv("{0}/test_lib/lib/{1}", base,
                        linkAgainstHsa ? "libmemory_allocator_hsa.a"
                                       : "libmemory_allocator_ion.a")};
}

} // namespace xilinx::aiecc

#endif // AIECC_TOOLS_H
