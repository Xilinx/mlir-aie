// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIECC_PROVENANCE_H
#define AIECC_PROVENANCE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace xilinx::aiecc {
/// Add a non-allocated provenance note to an assembled AIEBU ELF.
llvm::Expected<std::vector<char>> stampElfProvenance(llvm::StringRef filename,
                                                     llvm::StringRef json);
} // namespace xilinx::aiecc

#endif
