//===- StackSizeAnalysis.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes a core's stack requirement from its linked ELF: the maximum over
// the frame-weighted paths that leave the ELF entry point, following the call
// graph that the `.stack_sizes` data and the relocations describe. One call
// chain is live at a time, so the maximum bounds the requirement. A symbol the
// analysis cannot measure, and a symbol that recurses, both end the walk with
// a failure, so the result is an upper bound.
//
// The linker decides what a core contains, so measuring its output covers the
// toolchain's own startup code.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_STACKSIZEANALYSIS_H
#define AIECC_STACKSIZEANALYSIS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xilinx::aiecc {

// Cycle: the requirement is unbounded, so the design must declare a
// stack_size_override. Unmeasurable: the ELF is unreadable, or its
// `.stack_sizes` data is malformed. The driver warns for this case.
enum class StackRequirementFailure { Cycle, Unmeasurable };

struct StackRequirementResult {
  std::optional<int64_t> bytes;
  std::string error;
  StackRequirementFailure failureKind = StackRequirementFailure::Unmeasurable;
  // Functions with no frame size: the ELF holds no `.stack_sizes` entry for
  // them and the fallback table does not name them. Their frames count as 0,
  // so `bytes` is a lower bound while this list is non-empty. A kernel
  // compiled without -fstack-size-section lands here.
  std::vector<std::string> unmeasured;
};

// Measures the stack that the linked core ELF at `elfPath` needs. `overrides`
// maps a function name to a declared requirement for its whole subtree, and
// the walk stops at such a function.
//
// The link must keep the relocations (`-Wl,--emit-relocs`), which carry the
// call edges.
StackRequirementResult
computeStackRequirement(llvm::StringRef elfPath,
                        const llvm::StringMap<int64_t> &overrides);

// Sums the allocated .data, .rodata and .bss of the linked core ELF at
// `elfPath`. The link runs --gc-sections, so the ELF holds the sections the
// core reaches. Returns nothing when the file does not parse as an object.
std::optional<int64_t> measureDataSectionBytes(llvm::StringRef elfPath);

// Bytes by which a section overran its MEMORY region, from the linker's
// "overflowed by N bytes" report. A failed link writes no ELF, so this report
// is the only account of what the core needed. Returns nothing when the log
// carries no such report.
std::optional<int64_t> parseLinkOverflowBytes(llvm::StringRef log);

// Which memory bank a symbol's storage was placed for. `aie::lut<4>` reads its
// two tables at once and needs them in separate banks.
struct BankAssertion {
  std::string symbol;
  // Quoted back in the diagnostic: a section name like ".bss.DM_bankB".
  std::string origin;
  // Banks the request permits, ascending; a paired resource permits two.
  llvm::SmallVector<int, 2> banks;
};

// Bank requests carried by the objects a core links. Both toolchains put the
// bank in the section name: chess from `chess_storage(DM_bankA)`, Peano from an
// explicit `__attribute__((section))`.
std::vector<BankAssertion>
readBankAssertionsFromObjects(llvm::ArrayRef<std::string> objectPaths);

// An assertion the linked ELF did not satisfy.
struct BankViolation {
  BankAssertion assertion;
  int64_t address; // tile-relative
  int actualBank;
};

// Reports the requests the linked ELF contradicts. A symbol the ELF does not
// define is dropped: --gc-sections removes what the core never reaches.
std::vector<BankViolation>
checkBankPlacements(llvm::StringRef elfPath,
                    llvm::ArrayRef<BankAssertion> assertions,
                    int64_t tileBaseAddress, int64_t bankSize, int numBanks);

// One end of an `aie::lut<4>` table pair. A table on the stack is called out
// separately: the stack is one contiguous run, so two locals cannot be given
// separate banks at all.
struct LutOperand {
  enum class Kind { Symbol, Param, Stack };
  Kind kind = Kind::Symbol;
  std::string symbol;  // Kind::Symbol
  int paramIndex = -1; // Kind::Param
};

struct LutPair {
  std::string function;
  LutOperand a, b;
};

// Recovers which two objects each `aie::lut<4>` gather reads from, out of the
// LLVM IR that `-fembed-bitcode` leaves in the object's `.llvmbc` section. The
// gather takes its addresses from a single vector-select of two broadcast
// pointers, so the pair is whatever those two resolve to.
//
// Returns nothing when the object carries no IR, which the caller reports
// rather than mistaking for "no pairs found". Peano only: chess emits IR from
// an LLVM old enough that this parser rejects it.
std::optional<std::vector<LutPair>>
readLutPairsFromObject(llvm::StringRef objectPath);

// Tile-relative addresses of the data symbols the linked core ELF defines.
llvm::StringMap<int64_t> readDataSymbolAddresses(llvm::StringRef elfPath,
                                                 int64_t tileBaseAddress);

} // namespace xilinx::aiecc

#endif // AIECC_STACKSIZEANALYSIS_H
