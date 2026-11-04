// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Provenance.h"
#include "AIECCVersion.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"

#include <array>
#include <cstring>
#include <limits>
#include <optional>

using namespace llvm;

llvm::json::Object xilinx::aiecc::compilerProvenance() {
  return json::Object{{"schema", 1},
                      {"producer", "mlir-aie/aiecc"},
                      {"revision", AIECC_GIT_SHA},
                      {"dirty", AIECC_GIT_DIRTY < 0
                                    ? json::Value(nullptr)
                                    : json::Value(bool(AIECC_GIT_DIRTY))}};
}

llvm::json::Object
xilinx::aiecc::coreCompilerProvenance(StringRef name, StringRef executable,
                                      StringRef versionFlag) {
  json::Object result{
      {"name", name}, {"version", nullptr}, {"revision", nullptr}};
  if (executable.empty())
    return result;
  SmallString<128> path;
  int fd;
  if (sys::fs::createTemporaryFile("aiecc-version", "txt", fd, path))
    return result;
  sys::Process::SafelyCloseFileDescriptor(fd);
  std::array<std::optional<StringRef>, 3> redirects = {
      StringRef(""), StringRef(path), StringRef(path)};
  int status =
      sys::ExecuteAndWait(executable, {executable, versionFlag}, std::nullopt,
                          redirects, /*SecondsToWait=*/5);
  auto output = MemoryBuffer::getFile(path);
  sys::fs::remove(path);
  if (status != 0 || !output)
    return result;
  SmallVector<StringRef> matches;
  StringRef banner = (*output)->getBuffer();
  if (Regex("[Vv]ersion[ :]+([A-Za-z]*-?[0-9][A-Za-z0-9.+_-]*)")
          .match(banner, &matches))
    result["version"] = matches[1].str();
  if (Regex("llvm-aie[^ \n]* ([0-9a-fA-F]{7,40})").match(banner, &matches))
    result["revision"] = matches[1].str();
  else if (name == "chess" &&
           Regex("#([0-9a-fA-F]{7,40})#").match(banner, &matches))
    result["revision"] = matches[1].str();
  return result;
}

llvm::Expected<std::vector<char>>
xilinx::aiecc::stampElfProvenance(StringRef data, StringRef json) {
  using ELFT = object::ELF32LE;
  using Ehdr = ELFT::Ehdr;
  using Shdr = ELFT::Shdr;
  constexpr StringLiteral sectionName = ".note.mlir-aie.provenance";
  constexpr StringLiteral owner = "MLIR-AIE";
  auto invalid = [](StringRef reason) {
    return createStringError(inconvertibleErrorCode(),
                             "cannot stamp ELF provenance: " + reason);
  };
  if (data.size() < sizeof(Ehdr) || !data.starts_with("\177ELF") ||
      data[ELF::EI_CLASS] != ELF::ELFCLASS32 ||
      data[ELF::EI_DATA] != ELF::ELFDATA2LSB)
    return invalid("expected a little-endian ELF32 container");
  auto elf = object::ELFFile<ELFT>::create(data);
  if (!elf)
    return elf.takeError();
  auto sections = elf->sections();
  if (!sections)
    return sections.takeError();
  Ehdr header = elf->getHeader();
  if (sections->empty() || sections->size() >= ELF::SHN_LORESERVE - 1 ||
      header.e_shstrndx == ELF::SHN_UNDEF ||
      header.e_shstrndx >= sections->size())
    return invalid("unsupported section table");
  auto names = elf->getSectionStringTable(*sections);
  if (!names)
    return names.takeError();
  for (const Shdr &section : *sections) {
    auto name = elf->getSectionName(section, *names);
    if (!name)
      return name.takeError();
    if (*name == sectionName)
      return invalid("provenance note already exists");
  }
  uint64_t added = names->size() + sectionName.size() + json.size() + 64 +
                   (sections->size() + 1) * sizeof(Shdr);
  if (data.size() + added > std::numeric_limits<uint32_t>::max())
    return invalid("ELF32 size limit exceeded");

  // XRT relocations and the UID refer to the assembled contents. Only append
  // metadata and move the section table; leave program headers and payloads
  // put.
  std::vector<char> output(data.begin(), data.end());
  auto append = [&](StringRef bytes) {
    output.insert(output.end(), bytes.begin(), bytes.end());
  };
  auto align = [&] { output.resize(alignTo(output.size(), 4), 0); };
  std::vector<Shdr> table(sections->begin(), sections->end());
  Shdr &strings = table[header.e_shstrndx];
  strings.sh_offset = output.size();
  append(*names);
  append(sectionName);
  output.push_back(0);
  strings.sh_size = names->size() + sectionName.size() + 1;

  align();
  Shdr note{};
  note.sh_name = names->size();
  note.sh_type = ELF::SHT_NOTE;
  note.sh_offset = output.size();
  note.sh_addralign = 4;
  auto word = [&](uint32_t value) {
    char bytes[4];
    support::endian::write32le(bytes, value);
    append(StringRef(bytes, sizeof(bytes)));
  };
  word(owner.size() + 1);
  word(json.size() + 1);
  word(1);
  append(owner);
  output.push_back(0);
  align();
  append(json);
  output.push_back(0);
  align();
  note.sh_size = output.size() - note.sh_offset;
  table.push_back(note);
  header.e_shoff = output.size();
  header.e_shnum = table.size();
  append(StringRef(reinterpret_cast<const char *>(table.data()),
                   table.size() * sizeof(Shdr)));
  std::memcpy(output.data(), &header, sizeof(header));
  return output;
}
