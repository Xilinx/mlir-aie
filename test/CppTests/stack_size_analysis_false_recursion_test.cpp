//===- stack_size_analysis_false_recursion_test.cpp -------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tools/aiecc/StackSizeAnalysis.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <cstdlib>
#include <elf.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifndef AIE_STACK_SIZE_ANALYSIS_TEST_CLANG
#define AIE_STACK_SIZE_ANALYSIS_TEST_CLANG "clang"
#endif

namespace {

std::string quote(llvm::StringRef s) { return "'" + s.str() + "'"; }

void writeFile(llvm::StringRef path, llvm::StringRef contents) {
  std::ofstream out(path.str());
  if (!out) {
    throw std::runtime_error("failed to open " + path.str());
  }
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!out) {
    throw std::runtime_error("failed to write " + path.str());
  }
}

void run(llvm::StringRef command) {
  if (std::system(command.str().c_str()) != 0) {
    throw std::runtime_error("command failed: " + command.str());
  }
}

void patchElfAsAieWithNumberedDataRelocation(llvm::StringRef elfPath) {
  auto rangeFits = [](size_t size, size_t offset, size_t length) {
    return offset <= size && length <= size - offset;
  };
  std::ifstream in(elfPath.str(), std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + elfPath.str());
  }
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  if (bytes.size() < sizeof(Elf64_Ehdr)) {
    throw std::runtime_error("ELF is too small to patch");
  }

  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(bytes.data());
  if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
    throw std::runtime_error("expected an ELF64 test fixture");
  }
  if (ehdr->e_shnum == 0) {
    throw std::runtime_error("ELF has no section headers");
  }
  if (ehdr->e_shstrndx >= ehdr->e_shnum) {
    throw std::runtime_error("ELF has an invalid section-name string table");
  }
  size_t sectionTableSize =
      static_cast<size_t>(ehdr->e_shnum) * ehdr->e_shentsize;
  if (ehdr->e_shentsize != sizeof(Elf64_Shdr) ||
      !rangeFits(bytes.size(), ehdr->e_shoff, sectionTableSize)) {
    throw std::runtime_error("ELF has a truncated section table");
  }
  ehdr->e_machine = llvm::ELF::EM_AIE;

  auto *sections = reinterpret_cast<Elf64_Shdr *>(bytes.data() + ehdr->e_shoff);
  const auto &shstr = sections[ehdr->e_shstrndx];
  if (!rangeFits(bytes.size(), shstr.sh_offset, shstr.sh_size)) {
    throw std::runtime_error("ELF has a truncated section-name string table");
  }
  llvm::StringRef shstrtab(bytes.data() + shstr.sh_offset, shstr.sh_size);
  bool patchedRelocation = false;
  for (unsigned i = 0; i < ehdr->e_shnum; ++i) {
    const Elf64_Shdr &sec = sections[i];
    if (sec.sh_name >= shstrtab.size()) {
      throw std::runtime_error("ELF has an invalid section name offset");
    }
    llvm::StringRef name = shstrtab.drop_front(sec.sh_name).split('\0').first;
    if (sec.sh_type != SHT_RELA || name != ".rela.text") {
      continue;
    }
    if (sec.sh_entsize != sizeof(Elf64_Rela) ||
        sec.sh_size % sizeof(Elf64_Rela) != 0 ||
        !rangeFits(bytes.size(), sec.sh_offset, sec.sh_size)) {
      throw std::runtime_error("ELF has a malformed .rela.text section");
    }
    auto *relocs = reinterpret_cast<Elf64_Rela *>(bytes.data() + sec.sh_offset);
    size_t count = sec.sh_size / sizeof(Elf64_Rela);
    for (size_t j = 0; j < count; ++j) {
      if (ELF64_R_TYPE(relocs[j].r_info) != llvm::ELF::R_X86_64_64) {
        continue;
      }
      relocs[j].r_info =
          ELF64_R_INFO(ELF64_R_SYM(relocs[j].r_info), 62 /* aie2p FK_Data_4 */);
      patchedRelocation = true;
    }
  }
  if (!patchedRelocation) {
    throw std::runtime_error("failed to find x86_64 absolute relocation");
  }

  std::ofstream out(elfPath.str(), std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to reopen " + elfPath.str());
  }
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) {
    throw std::runtime_error("failed to write patched ELF");
  }
}

std::string buildFalseRecursionElf(bool useAieNumberedDataRelocation = false) {
  llvm::SmallString<128> dir;
  if (std::error_code ec = llvm::sys::fs::createUniqueDirectory(
          "stack-size-analysis-false-recursion", dir)) {
    throw std::runtime_error("failed to create temp directory: " +
                             ec.message());
  }

  llvm::SmallString<128> asmPath = dir;
  llvm::sys::path::append(asmPath, "false_recursion.s");
  llvm::SmallString<128> objPath = dir;
  llvm::sys::path::append(objPath, "false_recursion.o");
  llvm::SmallString<128> elfPath = dir;
  llvm::sys::path::append(elfPath, "false_recursion.elf");

  writeFile(asmPath, R"ASM(
.globl _start
.type _start,@function
_start:
  call entry_real
  ret
.size _start, .-_start

.globl entry_real
.type entry_real,@function
entry_real:
  .quad inlined_entry
  ret
.size entry_real, .-entry_real

.globl helper_cycle
.type helper_cycle,@function
helper_cycle:
  call entry_real
  ret
.size helper_cycle, .-helper_cycle

.globl inlined_entry
.type inlined_entry,@function
.set inlined_entry, helper_cycle
.size inlined_entry, 0
)ASM");

  std::string clang = AIE_STACK_SIZE_ANALYSIS_TEST_CLANG;
  run(clang + " -c " + quote(asmPath) + " -o " + quote(objPath));
  run(clang + " " + quote(objPath) +
      " -nostdlib -no-pie -Wl,-e,_start -Wl,--emit-relocs -o " +
      quote(elfPath));
  if (useAieNumberedDataRelocation) {
    patchElfAsAieWithNumberedDataRelocation(elfPath);
  }
  return elfPath.str().str();
}

void checkZeroSizedTargetDoesNotCreateFalseCycle() {
  auto result = xilinx::aiecc::computeStackRequirement(
      buildFalseRecursionElf(), llvm::StringMap<int64_t>());
  if (!result.bytes) {
    throw std::runtime_error("unexpected failure: " + result.error);
  }
  if (!result.error.empty()) {
    throw std::runtime_error("unexpected error text: " + result.error);
  }
}

void checkAieNumberedDataRelocationDoesNotCreateFalseCycle() {
  auto result = xilinx::aiecc::computeStackRequirement(
      buildFalseRecursionElf(/*useAieNumberedDataRelocation=*/true),
      llvm::StringMap<int64_t>());
  if (!result.bytes) {
    throw std::runtime_error("unexpected AIE-numbered failure: " +
                             result.error);
  }
  if (!result.error.empty()) {
    throw std::runtime_error("unexpected AIE-numbered error text: " +
                             result.error);
  }
}

void checkZeroSizedMainInitStillConnectsCallGraph() {
  llvm::SmallString<128> dir;
  if (std::error_code ec = llvm::sys::fs::createUniqueDirectory(
          "stack-size-analysis-main-init", dir)) {
    throw std::runtime_error("failed to create temp directory: " +
                             ec.message());
  }

  llvm::SmallString<128> asmPath = dir;
  llvm::sys::path::append(asmPath, "main_init_cycle.s");
  llvm::SmallString<128> objPath = dir;
  llvm::sys::path::append(objPath, "main_init_cycle.o");
  llvm::SmallString<128> elfPath = dir;
  llvm::sys::path::append(elfPath, "main_init_cycle.elf");

  writeFile(asmPath, R"ASM(
.globl _start
.type _start,@function
_start:
  call _main_init
  ret
.size _start, .-_start

.globl entry_real
.type entry_real,@function
entry_real:
  call recurse
  ret
.size entry_real, .-entry_real

.globl recurse
.type recurse,@function
recurse:
  call recurse
  ret
.size recurse, .-recurse

.globl _main_init
.type _main_init,@function
.set _main_init, entry_real
.size _main_init, 0
)ASM");

  std::string clang = AIE_STACK_SIZE_ANALYSIS_TEST_CLANG;
  run(clang + " -c " + quote(asmPath) + " -o " + quote(objPath));
  run(clang + " " + quote(objPath) +
      " -nostdlib -no-pie -Wl,-e,_start -Wl,--emit-relocs -o " +
      quote(elfPath));

  auto result = xilinx::aiecc::computeStackRequirement(
      elfPath.str(), llvm::StringMap<int64_t>());
  if (result.bytes) {
    throw std::runtime_error("expected recursion through _main_init alias");
  }
  if (result.failureKind != xilinx::aiecc::StackRequirementFailure::Cycle) {
    throw std::runtime_error("expected recursion failure kind");
  }
  if (result.error.find("recurse@0x") == std::string::npos) {
    throw std::runtime_error(
        "expected preserved call graph through _main_init");
  }
}

void checkRecursionDiagnosticPrintsAddresses() {
  llvm::SmallString<128> dir;
  if (std::error_code ec = llvm::sys::fs::createUniqueDirectory(
          "stack-size-analysis-cycle", dir)) {
    throw std::runtime_error("failed to create temp directory: " +
                             ec.message());
  }

  llvm::SmallString<128> asmPath = dir;
  llvm::sys::path::append(asmPath, "cycle.s");
  llvm::SmallString<128> objPath = dir;
  llvm::sys::path::append(objPath, "cycle.o");
  llvm::SmallString<128> elfPath = dir;
  llvm::sys::path::append(elfPath, "cycle.elf");

  writeFile(asmPath, R"ASM(
.globl _start
.type _start,@function
_start:
  call recurse
  ret
.size _start, .-_start

.globl recurse
.type recurse,@function
recurse:
  call recurse
  ret
.size recurse, .-recurse
)ASM");

  std::string clang = AIE_STACK_SIZE_ANALYSIS_TEST_CLANG;
  run(clang + " -c " + quote(asmPath) + " -o " + quote(objPath));
  run(clang + " " + quote(objPath) +
      " -nostdlib -no-pie -Wl,-e,_start -Wl,--emit-relocs -o " +
      quote(elfPath));

  auto result = xilinx::aiecc::computeStackRequirement(
      elfPath.str(), llvm::StringMap<int64_t>());
  if (result.bytes) {
    throw std::runtime_error("expected recursion failure");
  }
  if (result.failureKind != xilinx::aiecc::StackRequirementFailure::Cycle) {
    throw std::runtime_error("expected recursion failure kind");
  }
  if (result.error.find("_start@0x") == std::string::npos ||
      result.error.find("recurse@0x") == std::string::npos) {
    throw std::runtime_error("recursion diagnostic is missing addresses: " +
                             result.error);
  }
}

} // namespace

int main() {
  checkZeroSizedTargetDoesNotCreateFalseCycle();
  checkAieNumberedDataRelocationDoesNotCreateFalseCycle();
  checkZeroSizedMainInitStillConnectsCallGraph();
  checkRecursionDiagnosticPrintsAddresses();
  return 0;
}
