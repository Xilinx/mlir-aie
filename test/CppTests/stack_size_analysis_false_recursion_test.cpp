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
#include <cstring>
#include <elf.h>
#include <fstream>
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

void align(std::vector<char> &bytes, size_t alignment) {
  size_t padded = llvm::alignTo(bytes.size(), alignment);
  bytes.resize(padded, '\0');
}

template <typename T>
size_t appendStruct(std::vector<char> &bytes, const T &value,
                    size_t alignment = alignof(T)) {
  align(bytes, alignment);
  size_t offset = bytes.size();
  bytes.resize(offset + sizeof(T));
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
  return offset;
}

size_t appendBytes(std::vector<char> &bytes, llvm::StringRef contents,
                   size_t alignment = 1) {
  align(bytes, alignment);
  size_t offset = bytes.size();
  bytes.insert(bytes.end(), contents.begin(), contents.end());
  return offset;
}

template <typename T>
void overwriteStruct(std::vector<char> &bytes, size_t offset, const T &value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

std::string buildFalseRecursionElf() {
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
  return elfPath.str().str();
}

std::string buildAieNumberedFalseRecursionElf() {
  enum SectionIndex : uint16_t {
    NullSection,
    TextSection,
    StackSizesSection,
    RelaTextSection,
    SymtabSection,
    StrtabSection,
    ShstrtabSection,
  };
  enum SymbolIndex : uint32_t {
    NullSymbol,
    StartSymbol,
    EntryRealSymbol,
    HelperCycleSymbol,
    InlinedEntrySymbol,
  };
  constexpr uint32_t startAddr = 0x0;
  constexpr uint32_t entryRealAddr = 0x10;
  constexpr uint32_t helperCycleAddr = 0x20;
  constexpr uint32_t textSize = 0x30;
  constexpr uint32_t aieCallReloc = 1;

  llvm::SmallString<128> dir;
  if (std::error_code ec = llvm::sys::fs::createUniqueDirectory(
          "stack-size-analysis-aie-false-recursion", dir)) {
    throw std::runtime_error("failed to create temp directory: " +
                             ec.message());
  }
  llvm::SmallString<128> elfPath = dir;
  llvm::sys::path::append(elfPath, "false_recursion_aie.elf");

  const std::string strtab =
      std::string("\0__start\0entry_real\0helper_cycle\0inlined_entry\0", 47);
  const size_t startName = 1;
  const size_t entryRealName = startName + std::strlen("__start") + 1;
  const size_t helperCycleName = entryRealName + std::strlen("entry_real") + 1;
  const size_t inlinedEntryName =
      helperCycleName + std::strlen("helper_cycle") + 1;
  const std::string shstrtab = std::string(
      "\0.text\0.stack_sizes\0.rela.text\0.symtab\0.strtab\0.shstrtab\0", 57);
  const size_t textName = 1;
  const size_t stackSizesName = textName + std::strlen(".text") + 1;
  const size_t relaTextName = stackSizesName + std::strlen(".stack_sizes") + 1;
  const size_t symtabName = relaTextName + std::strlen(".rela.text") + 1;
  const size_t strtabName = symtabName + std::strlen(".symtab") + 1;
  const size_t shstrtabName = strtabName + std::strlen(".strtab") + 1;

  std::string stackSizes;
  auto appendStackSize = [&](uint32_t addr, uint8_t bytesValue) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      stackSizes.push_back(static_cast<char>((addr >> shift) & 0xFF));
    }
    stackSizes.push_back(static_cast<char>(bytesValue));
  };
  appendStackSize(entryRealAddr, 8);
  appendStackSize(helperCycleAddr, 8);

  std::vector<Elf32_Rela> relocs = {
      {startAddr, ELF32_R_INFO(EntryRealSymbol, aieCallReloc), 0},
      {entryRealAddr,
       ELF32_R_INFO(InlinedEntrySymbol,
                    xilinx::aiecc::detail::aieData4RelocAie2p),
       0},
      {helperCycleAddr, ELF32_R_INFO(EntryRealSymbol, aieCallReloc), 0},
  };
  std::vector<Elf32_Sym> symbols = {
      {},
      {static_cast<Elf32_Word>(startName), startAddr, 4,
       ELF32_ST_INFO(STB_GLOBAL, STT_FUNC), 0, TextSection},
      {static_cast<Elf32_Word>(entryRealName), entryRealAddr, 4,
       ELF32_ST_INFO(STB_GLOBAL, STT_FUNC), 0, TextSection},
      {static_cast<Elf32_Word>(helperCycleName), helperCycleAddr, 4,
       ELF32_ST_INFO(STB_GLOBAL, STT_FUNC), 0, TextSection},
      {static_cast<Elf32_Word>(inlinedEntryName), helperCycleAddr, 0,
       ELF32_ST_INFO(STB_GLOBAL, STT_FUNC), 0, TextSection},
  };

  std::vector<char> bytes(sizeof(Elf32_Ehdr), '\0');
  size_t textOffset = appendBytes(bytes, std::string(textSize, '\0'), 4);
  size_t stackSizesOffset = appendBytes(bytes, stackSizes, 4);
  size_t relaTextOffset =
      appendBytes(bytes,
                  llvm::StringRef(reinterpret_cast<const char *>(relocs.data()),
                                  relocs.size() * sizeof(Elf32_Rela)),
                  4);
  size_t symtabOffset = appendBytes(
      bytes,
      llvm::StringRef(reinterpret_cast<const char *>(symbols.data()),
                      symbols.size() * sizeof(Elf32_Sym)),
      4);
  size_t strtabOffset = appendBytes(bytes, strtab);
  size_t shstrtabOffset = appendBytes(bytes, shstrtab);
  align(bytes, alignof(Elf32_Shdr));
  size_t shoff = bytes.size();

  std::vector<Elf32_Shdr> sections(ShstrtabSection + 1);
  sections[TextSection] = {static_cast<Elf32_Word>(textName),
                           SHT_PROGBITS,
                           SHF_ALLOC | SHF_EXECINSTR,
                           0,
                           textOffset,
                           textSize,
                           0,
                           0,
                           4,
                           0};
  sections[StackSizesSection] = {static_cast<Elf32_Word>(stackSizesName),
                                 SHT_PROGBITS,
                                 0,
                                 0,
                                 stackSizesOffset,
                                 static_cast<Elf32_Word>(stackSizes.size()),
                                 0,
                                 0,
                                 1,
                                 0};
  sections[RelaTextSection] = {
      static_cast<Elf32_Word>(relaTextName),
      SHT_RELA,
      0,
      0,
      relaTextOffset,
      static_cast<Elf32_Word>(relocs.size() * sizeof(Elf32_Rela)),
      SymtabSection,
      TextSection,
      4,
      sizeof(Elf32_Rela)};
  sections[SymtabSection] = {
      static_cast<Elf32_Word>(symtabName),
      SHT_SYMTAB,
      0,
      0,
      symtabOffset,
      static_cast<Elf32_Word>(symbols.size() * sizeof(Elf32_Sym)),
      StrtabSection,
      1,
      4,
      sizeof(Elf32_Sym)};
  sections[StrtabSection] = {
      static_cast<Elf32_Word>(strtabName),    SHT_STRTAB, 0, 0, strtabOffset,
      static_cast<Elf32_Word>(strtab.size()), 0,          0, 1, 0};
  sections[ShstrtabSection] = {static_cast<Elf32_Word>(shstrtabName),
                               SHT_STRTAB,
                               0,
                               0,
                               shstrtabOffset,
                               static_cast<Elf32_Word>(shstrtab.size()),
                               0,
                               0,
                               1,
                               0};
  for (const Elf32_Shdr &section : sections) {
    appendStruct(bytes, section, alignof(Elf32_Shdr));
  }

  Elf32_Ehdr ehdr = {};
  ehdr.e_ident[EI_MAG0] = ELFMAG0;
  ehdr.e_ident[EI_MAG1] = ELFMAG1;
  ehdr.e_ident[EI_MAG2] = ELFMAG2;
  ehdr.e_ident[EI_MAG3] = ELFMAG3;
  ehdr.e_ident[EI_CLASS] = ELFCLASS32;
  ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  ehdr.e_type = ET_EXEC;
  ehdr.e_machine = llvm::ELF::EM_AIE;
  ehdr.e_version = EV_CURRENT;
  ehdr.e_entry = startAddr;
  ehdr.e_ehsize = sizeof(Elf32_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_shentsize = sizeof(Elf32_Shdr);
  ehdr.e_shnum = sections.size();
  ehdr.e_shstrndx = ShstrtabSection;
  overwriteStruct(bytes, 0, ehdr);

  std::ofstream out(elfPath.str(), std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open " + elfPath.str());
  }
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) {
    throw std::runtime_error("failed to write " + elfPath.str());
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
      buildAieNumberedFalseRecursionElf(), llvm::StringMap<int64_t>());
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
