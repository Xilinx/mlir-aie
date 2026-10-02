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
  // them. Their frames count as 0, so `bytes` is a lower bound while this list
  // is non-empty. A kernel compiled without -fstack-size-section lands here.
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

/// One allocatable section of a core ELF that competes for the tile's data
/// memory: `.data`, `.rodata`, `.bss`, and the bank-pinned `.aie.bank<N>` or
/// `DM_bank<L>` sections. `address` is relative to `tileBaseAddress`.
///
/// Both ways of reserving a core's own memory read this. A core compiled here
/// is measured from a probe link, where only the sizes carry meaning because
/// placement has not run yet; a prebaked `elf_file` is read for its addresses,
/// which are already final. Sharing the walk is what keeps the two from
/// disagreeing about which sections count.
struct CoreDataSection {
  std::string name;
  int64_t address = 0;
  int64_t size = 0;
  int64_t align = 1;
  // The single bank this section is pinned to. Absent for ordinary data, and
  // for a request naming several banks, which no region can satisfy alone.
  std::optional<int> bank;
};
llvm::SmallVector<CoreDataSection>
readCoreDataSections(llvm::StringRef elfPath, int64_t tileBaseAddress = 0);

// Sums the allocated .data, .rodata and .bss of the linked core ELF at
// `elfPath`. The link runs --gc-sections, so the ELF holds the sections the
// core reaches. Returns nothing when the file does not parse as an object.
std::optional<int64_t> measureDataSectionBytes(llvm::StringRef elfPath);

/// Bytes and alignment each bank's pinned sections occupy in a linked ELF,
/// indexed by bank. Read from a probe link, this is what a core's objects need
/// reserved; read from a prebaked `elf_file`, it is what they already hold.
/// Alignment is not incidental: a region that starts unaligned loses bytes to
/// the linker's own padding and the section stops fitting. Sizes include
/// padding between output sections in linked address order.
struct BankSectionSize {
  int64_t size = 0;
  int64_t align = 1;
};

// Unpinned sections packed in linked address order, including padding between
// output sections (Chess can keep .rodata separate). The returned size fits
// only when the reservation starts at the returned alignment.
std::optional<BankSectionSize>
measureDataSectionDemand(llvm::StringRef elfPath);

llvm::SmallVector<BankSectionSize>
measureBankSectionBytes(llvm::StringRef elfPath, int numBanks);

// Bytes by which a section overran its MEMORY region, from the linker's
// "overflowed by N bytes" report. A failed link writes no ELF, so this report
// is the only account of what the core needed. Returns nothing when the log
// carries no such report.
std::optional<int64_t> parseLinkOverflowBytes(llvm::StringRef log,
                                              llvm::StringRef region);

// Which memory bank a symbol's storage was placed for. `aie::lut<4>` reads its
// two tables at once and needs them in separate banks.
struct BankAssertion {
  std::string symbol;
  // Quoted back in the diagnostic: a section name like ".bss.DM_bankB".
  std::string origin;
  // Banks the request permits, ascending; a paired resource permits two.
  llvm::SmallVector<int, 2> banks;
};

// Bank requests carried by the objects and archive members a core links. Both
// toolchains put the bank in the section name: chess from
// `chess_storage(DM_bankA)`, Peano from an explicit `__attribute__((section))`.
// Names defined by multiple input objects are omitted, even if only one
// definition is pinned: linking/GC cannot reliably associate their requests.
std::vector<BankAssertion>
readBankAssertionsFromObjects(llvm::ArrayRef<std::string> objectPaths);

// An assertion the linked ELF did not satisfy.
struct BankViolation {
  BankAssertion assertion;
  int64_t address; // tile-relative
  int actualBank;
  uint64_t size;
  bool crossesBank;
};

// Reports the requests the linked ELF contradicts. A symbol the ELF does not
// define is dropped: --gc-sections removes what the core never reaches.
// Ambiguous duplicate names are also skipped, not reported as contradictions.
// Nonzero symbol extents must fit wholly within one permitted bank.
std::vector<BankViolation>
checkBankPlacements(llvm::StringRef elfPath,
                    llvm::ArrayRef<BankAssertion> assertions,
                    int64_t tileBaseAddress, int64_t bankSize, int numBanks);

// One end of an `aie::lut<4>` table pair. Stack locals are called out
// separately: their final offsets, and hence bank separation, cannot be
// verified here.
struct LutOperand {
  enum class Kind { Unknown, Symbol, Param, Stack };
  Kind kind = Kind::Unknown;
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
// Unresolvable gather addresses produce Unknown operands, not an empty result.
// Direct LLVM IR inputs (.ll or .bc) are also accepted.
// With elfPath, omit functions emitted in the object but removed by the linker.
// Keep other functions: inlining or renaming can hide symbols, not their LUTs.
// Returns nothing for unreadable IR. Peano only: Chess IR is too old to parse.
std::optional<std::vector<LutPair>>
readLutPairsFromObject(llvm::StringRef objectPath,
                       llvm::StringRef elfPath = {});

// Inspects a textual or bitcode IR file, including optimized per-core IR.
// Unknown operands and unresolved parameters remain in the result so the
// caller can reject unverified gathers. Functions are not filtered by linked
// symbol names: LTO may inline their gathers and remove the original symbol.
std::optional<std::vector<LutPair>> readLutPairsFromIR(llvm::StringRef irPath);

// Tile-relative addresses of the data symbols the linked core ELF defines.
// Ambiguous names are omitted. If supplied, `sizes` receives ELF symbol sizes;
// callers must check that a table's whole extent lies in one memory bank.
llvm::StringMap<int64_t>
readDataSymbolAddresses(llvm::StringRef elfPath, int64_t tileBaseAddress,
                        llvm::StringMap<uint64_t> *sizes = nullptr);

} // namespace xilinx::aiecc

#endif // AIECC_STACKSIZEANALYSIS_H
