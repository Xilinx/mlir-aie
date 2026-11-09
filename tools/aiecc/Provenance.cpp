// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Provenance.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstring>

using namespace llvm;

llvm::Expected<std::vector<char>>
xilinx::aiecc::stampElfProvenance(StringRef filename, StringRef json) {
  using ELFT = object::ELF32LE;
  auto buffer = MemoryBuffer::getFile(filename);
  if (!buffer)
    return errorCodeToError(buffer.getError());
  StringRef data = (*buffer)->getBuffer();
  auto elf = object::ELFFile<ELFT>::create(data);
  if (!elf)
    return elf.takeError();
  auto sections = elf->sections();
  if (!sections)
    return sections.takeError();
  auto names = elf->getSectionStringTable(*sections);
  if (!names)
    return names.takeError();

  // LLVM objcopy rejects AIEBU's .rela.dyn link to .dynsym. Append the note
  // and a new section table, preserving the assembled sections and segments.
  constexpr StringLiteral sectionName = ".note.mlir-aie.provenance";
  constexpr StringLiteral owner = "MLIR-AIE";
  std::vector<char> output(data.begin(), data.end());
  auto append = [&](StringRef bytes) {
    output.insert(output.end(), bytes.begin(), bytes.end());
  };
  auto align = [&] { output.resize(alignTo(output.size(), 4), 0); };
  auto header = elf->getHeader();
  std::vector<ELFT::Shdr> table(sections->begin(), sections->end());
  auto &strings = table[header.e_shstrndx];
  strings.sh_offset = output.size();
  append(*names);
  append(sectionName);
  output.push_back(0);
  strings.sh_size = names->size() + sectionName.size() + 1;

  align();
  ELFT::Shdr note{};
  note.sh_name = names->size();
  note.sh_type = ELF::SHT_NOTE;
  note.sh_offset = output.size();
  note.sh_addralign = 4;
  ELFT::Nhdr description{};
  description.n_namesz = owner.size() + 1;
  description.n_descsz = json.size() + 1;
  description.n_type = 1;
  append(StringRef(reinterpret_cast<const char *>(&description),
                   sizeof(description)));
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
                   table.size() * sizeof(ELFT::Shdr)));
  std::memcpy(output.data(), &header, sizeof(header));
  return output;
}
