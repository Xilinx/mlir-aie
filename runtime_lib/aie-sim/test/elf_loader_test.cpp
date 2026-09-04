//===- elf_loader_test.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The loader's job is placement, so the test asserts where bytes landed -- and
// asserts the .bss gap reads as zero, which is the one rule a memcpy would get
// wrong (Xilinx/mlir-aie#3532).
//
// The images are built here rather than committed, so each test states the
// exact header field it is about. Needs no core engine: placement is the array
// model's own work.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"
#include "aiesim/Device.h"
#include "aiesim/ElfLoader.h"
#include "TestSupport.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

constexpr uint32_t kOwnDataBase = 0x70000;

struct Segment {
  uint32_t vaddr;
  std::vector<uint8_t> contents;
  uint32_t memsz; // >= contents.size(); the excess is the .bss gap
};

void put16(std::vector<uint8_t> &v, size_t at, uint16_t x) {
  v[at] = uint8_t(x);
  v[at + 1] = uint8_t(x >> 8);
}

void put32(std::vector<uint8_t> &v, size_t at, uint32_t x) {
  for (int i = 0; i < 4; ++i)
    v[at + i] = uint8_t(x >> (8 * i));
}

/// A minimal ELF32 LE image: header, one program header per segment, then the
/// segment contents.
std::vector<uint8_t> makeElf(const std::vector<Segment> &segs, uint32_t entry) {
  const size_t ehdr = 52, phdr = 32;
  const size_t phoff = ehdr;
  size_t off = phoff + phdr * segs.size();
  std::vector<uint8_t> image(off, 0);

  std::memcpy(image.data(), "\177ELF", 4);
  image[4] = 1; // ELFCLASS32
  image[5] = 1; // ELFDATA2LSB
  image[6] = 1; // EV_CURRENT
  put16(image, 16, 2);            // ET_EXEC
  put32(image, 24, entry);        // e_entry
  put32(image, 28, uint32_t(phoff));
  put16(image, 42, uint16_t(phdr));
  put16(image, 44, uint16_t(segs.size()));

  for (size_t i = 0; i < segs.size(); ++i) {
    const Segment &s = segs[i];
    const size_t ph = phoff + phdr * i;
    put32(image, ph + 0, 1); // PT_LOAD
    put32(image, ph + 4, uint32_t(off));
    put32(image, ph + 8, s.vaddr);
    put32(image, ph + 16, uint32_t(s.contents.size()));
    put32(image, ph + 20, s.memsz);
    image.insert(image.end(), s.contents.begin(), s.contents.end());
    off += s.contents.size();
  }
  return image;
}

uint8_t dataByte(Tile &tile, uint32_t off) {
  uint8_t b = 0xAA;
  AIESIM_CHECK(tile.memory()->read(off, &b, 1));
  return b;
}

} // namespace

int main() {
  std::string error;
  DeviceModel dev;
  AIESIM_CHECK(makeDeviceFromName("npu2", dev, error));
  Array array(dev, nullptr);
  Tile *tile = array.tile(7, 3);
  AIESIM_CHECK(tile != nullptr);
  if (!tile)
    return aiesim_test::summarize("elf_loader_test");

  // A .text segment and a mixed .data+.bss segment, which is the ordinary shape
  // and the one the gap rule is about: four initialised bytes claiming twelve.
  const std::vector<uint8_t> text = {0x11, 0x22, 0x33, 0x44};
  const std::vector<uint8_t> data = {0xDE, 0xAD, 0xBE, 0xEF};
  std::vector<uint8_t> image =
      makeElf({{0x00000, text, uint32_t(text.size())},
               {kOwnDataBase, data, 12}},
              0x20);

  // Poison the gap first, so zeroing has to be observed rather than inherited
  // from a fresh tile that was already zero.
  for (uint32_t i = 0; i < 12; ++i) {
    const uint8_t poison = 0x5A;
    AIESIM_CHECK(tile->memory()->write(i, &poison, 1));
  }

  uint32_t entry = 0;
  AIESIM_CHECK(loadCoreElf(*tile, image.data(), image.size(), entry, error));
  AIESIM_CHECK_EQ(entry, 0x20u);

  uint8_t got[4] = {};
  AIESIM_CHECK(tile->programMemory()->read(0, got, sizeof(got)));
  AIESIM_CHECK_EQ(int(got[0]), 0x11);
  AIESIM_CHECK_EQ(int(got[3]), 0x44);

  AIESIM_CHECK_EQ(int(dataByte(*tile, 0)), 0xDE);
  AIESIM_CHECK_EQ(int(dataByte(*tile, 3)), 0xEF);

  // The gap. Every byte of p_memsz - p_filesz reads zero, not the poison and
  // not whatever followed the segment in the image.
  for (uint32_t i = 4; i < 12; ++i)
    AIESIM_CHECK_EQ(int(dataByte(*tile, i)), 0x00);

  // Negative controls: each must be refused, with the reason.
  {
    std::vector<uint8_t> bad = image;
    bad[1] = 'X';
    uint32_t e = 0;
    std::string why;
    AIESIM_CHECK(!loadCoreElf(*tile, bad.data(), bad.size(), e, why));
    AIESIM_CHECK(!why.empty());
  }
  {
    // A neighbour's band is a placement the host loader never performs.
    std::vector<uint8_t> neighbour = makeElf({{0x50000, data, 4}}, 0x20);
    uint32_t e = 0;
    std::string why;
    AIESIM_CHECK(
        !loadCoreElf(*tile, neighbour.data(), neighbour.size(), e, why));
    AIESIM_CHECK(why.find("neighbour") != std::string::npos);
  }
  {
    // filesz running past the image is a truncated file, not a short segment.
    std::vector<uint8_t> truncated = image;
    truncated.resize(truncated.size() - 2);
    uint32_t e = 0;
    std::string why;
    AIESIM_CHECK(
        !loadCoreElf(*tile, truncated.data(), truncated.size(), e, why));
  }

  return aiesim_test::summarize("elf_loader_test");
}
