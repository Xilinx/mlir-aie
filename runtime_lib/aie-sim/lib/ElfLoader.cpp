//===- ElfLoader.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// AIE cores are 32-bit little-endian, so this reads ELF32 headers directly
// rather than pulling in a general object library: the whole format surface
// used here is the header, the program headers, and PT_LOAD.
//
//===----------------------------------------------------------------------===//

#include "aiesim/ElfLoader.h"

#include "CoreAddressMap.h"
#include "aiesim/Components.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using namespace aiesim;

namespace {

constexpr uint32_t kPtLoad = 1;
constexpr size_t kEhdrSize = 52;
constexpr size_t kPhdrSize = 32;

uint16_t read16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t read32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

std::string format(const char *fmt, ...) {
  char buf[256];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return std::string(buf);
}

/// Write `filesz` bytes at `off`, then zero up to `memsz`.
///
/// The zeroing is the whole reason this is not a memcpy. A PT_LOAD holds
/// `p_filesz` initialised bytes and claims `p_memsz` of address space; the gap
/// is how .bss rides in a loadable segment, and the System V gABI defines it to
/// read as zero. Copying `p_memsz` bytes from the image instead walks off the
/// end of the segment's file contents and lands whatever follows -- .comment,
/// .symtab -- on the zero-initialised statics. That is not hypothetical: it is
/// Xilinx/mlir-aie#3532, where the bytes of a linker version string reached a
/// kernel's ping/pong selector and inverted its double-buffer phase for a whole
/// run. A model that memcpy'd would reproduce the bug and call it hardware.
bool placeSegment(Memory &mem, uint32_t off, const uint8_t *src,
                  uint32_t filesz, uint32_t memsz, const char *what,
                  std::string &error) {
  if (!mem.inRange(off, memsz)) {
    error = format("%s segment at 0x%08X+0x%X does not fit", what, off, memsz);
    return false;
  }
  if (filesz && !mem.write(off, src, filesz)) {
    error = format("%s write of 0x%X bytes at 0x%08X failed", what, filesz,
                   off);
    return false;
  }
  if (memsz > filesz) {
    std::vector<uint8_t> zeros(memsz - filesz, 0);
    if (!mem.write(off + filesz, zeros.data(),
                   static_cast<uint32_t>(zeros.size()))) {
      error = format("%s zero-fill of 0x%X bytes at 0x%08X failed", what,
                     memsz - filesz, off + filesz);
      return false;
    }
  }
  return true;
}

} // namespace

bool aiesim::loadCoreElf(Tile &tile, const void *image, size_t size,
                         uint32_t &entry, std::string &error) {
  const uint8_t *base = static_cast<const uint8_t *>(image);
  if (size < kEhdrSize) {
    error = format("image is %zu bytes, shorter than an ELF32 header", size);
    return false;
  }
  if (std::memcmp(base, "\177ELF", 4) != 0) {
    error = "not an ELF image";
    return false;
  }
  if (base[4] != 1 || base[5] != 1) {
    error = "not a 32-bit little-endian ELF, which is what an AIE core is";
    return false;
  }

  entry = read32(base + 24);
  const uint32_t phoff = read32(base + 28);
  const uint16_t phentsize = read16(base + 42);
  const uint16_t phnum = read16(base + 44);
  if (phentsize < kPhdrSize) {
    error = format("program header entry is %u bytes, need %zu", phentsize,
                   kPhdrSize);
    return false;
  }
  if (phoff + static_cast<size_t>(phnum) * phentsize > size) {
    error = "program header table runs past the end of the image";
    return false;
  }

  unsigned placed = 0;
  for (uint16_t i = 0; i < phnum; ++i) {
    const uint8_t *ph = base + phoff + static_cast<size_t>(i) * phentsize;
    if (read32(ph) != kPtLoad)
      continue;
    const uint32_t offset = read32(ph + 4);
    const uint32_t vaddr = read32(ph + 8);
    const uint32_t filesz = read32(ph + 16);
    const uint32_t memsz = read32(ph + 20);
    if (memsz == 0)
      continue;
    if (static_cast<size_t>(offset) + filesz > size) {
      error = format("segment %u contents run past the end of the image", i);
      return false;
    }

    Memory *prog = tile.programMemory();
    if (prog && vaddr - kProgramBase < prog->size()) {
      if (!placeSegment(*prog, vaddr - kProgramBase, base + offset, filesz,
                        memsz, "program", error))
        return false;
      ++placed;
      continue;
    }

    bool routed = false;
    for (const MemBand &band : kAIE2Bands) {
      if (vaddr < band.base || vaddr >= band.base + kBandSize)
        continue;
      if (band.dir != MemDirection::Own) {
        error = format("segment %u at 0x%08X addresses a neighbour's data "
                       "band; a core ELF is loaded one tile at a time",
                       i, vaddr);
        return false;
      }
      Memory *data = tile.memory();
      if (!data) {
        error = "tile has no data memory";
        return false;
      }
      if (!placeSegment(*data, vaddr - band.base, base + offset, filesz, memsz,
                        "data", error))
        return false;
      routed = true;
      ++placed;
      break;
    }
    if (!routed) {
      error = format("segment %u at 0x%08X is outside the core's address map",
                     i, vaddr);
      return false;
    }
  }

  if (placed == 0) {
    error = "no PT_LOAD segment carried anything to place";
    return false;
  }
  return true;
}

bool aiesim::loadCoreElfFile(Tile &tile, const std::string &path,
                             uint32_t &entry, std::string &error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  std::vector<uint8_t> image((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  if (image.empty()) {
    error = path + " is empty";
    return false;
  }
  return loadCoreElf(tile, image.data(), image.size(), entry, error);
}
