//===- AliasScopeDomains.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGETS_ALIASSCOPEDOMAINS_H
#define AIE_TARGETS_ALIASSCOPEDOMAINS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"

#include <string>

namespace xilinx::AIE {

// Peano's textual IR spells an alias scope domain `!{!N, !"name"}`; aiecc's
// LLVM requires the disjoint flag `!{!N, i1 false, !"name"}` and does not
// upgrade text. Add it. Bitcode is returned unchanged.
inline std::string upgradeAliasScopeDomains(llvm::StringRef ir) {
  std::string result = ir.str();
  if (llvm::isBitcode(ir.bytes_begin(), ir.bytes_end()))
    return result;
  for (size_t p = 0;
       (p = result.find("= distinct !{!", p)) != std::string::npos;) {
    size_t bol = result.rfind('\n', p);
    bol = (bol == std::string::npos) ? 0 : bol + 1;
    llvm::StringRef id = llvm::StringRef(result).slice(bol, p).trim();
    size_t first = p + llvm::StringRef("= distinct !{").size();
    p = first;
    if (id.starts_with("!") &&
        result.compare(first, id.size() + 4, id.str() + ", !\"") == 0)
      result.insert(first + id.size(), ", i1 false");
  }
  return result;
}

} // namespace xilinx::AIE

#endif // AIE_TARGETS_ALIASSCOPEDOMAINS_H
