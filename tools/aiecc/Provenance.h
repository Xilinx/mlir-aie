// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIECC_PROVENANCE_H
#define AIECC_PROVENANCE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <string>
#include <vector>

namespace xilinx::aiecc {

/// Return the identity embedded when aiecc was built.
llvm::json::Object compilerProvenance();

/// Query a selected compiler, retaining only its version and revision.
llvm::json::Object coreCompilerProvenance(llvm::StringRef name,
                                          llvm::StringRef executable,
                                          llvm::StringRef versionFlag);

/// Append a non-allocated note without relocating the ELF's existing payloads.
llvm::Expected<std::vector<char>> stampElfProvenance(llvm::StringRef elf,
                                                     llvm::StringRef json);

} // namespace xilinx::aiecc

#endif
