//===- RegionMap.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Which bytes of a core's address space belong to what: stack, objectFIFO
// buffers, program, data. Parsed from the linker script aiecc already emits
// (ldScripts_<core>.ld.script), so nothing new has to be generated.
//
// Without a region map every byte of data memory is equally legitimate, so a
// stack overrun and the buffer's own producer are the same write. The hazard
// this addresses, and the measurement that it is the default arrangement
// rather than an unlucky one, are in docs/Readings.md.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_REGIONMAP_H
#define AIESIM_REGIONMAP_H

#include <cstdint>
#include <string>
#include <vector>

namespace aiesim {

enum class RegionKind {
  Program, ///< The MEMORY block's `program` region.
  Data,    ///< The MEMORY block's `data` region (.data/.rodata/.bss).
  Stack,   ///< The `_sp_start_value_*` reservation.
  Buffer,  ///< A named allocation, in practice an objectFIFO buffer.
};

struct Region {
  std::string name;
  uint32_t begin = 0;
  uint32_t size = 0;
  RegionKind kind = RegionKind::Buffer;

  uint32_t end() const { return begin + size; }
  bool contains(uint32_t addr) const { return addr >= begin && addr < end(); }
  /// True when [addr, addr+len) is not wholly inside this region.
  bool escapes(uint32_t addr, uint32_t len) const {
    return addr < begin || uint64_t(addr) + len > end();
  }
};

/// The regions of one core's address space, sorted by address.
class RegionMap {
public:
  const std::vector<Region> &regions() const { return items; }
  bool empty() const { return items.empty(); }

  /// Sorted insert. Keeping the vector ordered is what lets clearance and
  /// overlap be a single linear pass rather than a search per query.
  void add(Region r);

  const Region *stack() const;
  const Region *findContaining(uint32_t addr) const;

  /// Bytes between the stack's top and the next region above it. Zero means a
  /// one-byte frame overrun lands in that region. Absent when there is no
  /// stack, or nothing is allocated above it.
  bool stackClearance(uint32_t &bytesOut, std::string &nextRegionOut) const;

  /// Pairs of regions whose ranges intersect. Always a defect: two owners for
  /// one byte, and whichever writes second wins silently.
  struct Overlap {
    std::string a, b;
    uint32_t begin, end;
  };
  std::vector<Overlap> overlaps() const;

  /// False when `sp` is outside the stack reservation, with `why` naming what
  /// it hit. A store's address cannot say whether it is a stack access; the
  /// stack POINTER leaving its reservation can.
  bool checkStackPointer(uint32_t sp, std::string &why) const;

  /// False when a write of `len` bytes at `addr` starts inside one region and
  /// runs past its end. An address in no region is not a fault.
  bool checkWrite(uint32_t addr, uint32_t len, std::string &why) const;

private:
  std::vector<Region> items;
};

/// Parse a linker script as emitted by AIETargetLdScript.cpp. Reads the MEMORY
/// block and the `. = addr; sym = .; . += size;` triples that follow. Returns
/// false with `error` set only for input that is not a linker script at all;
/// an unrecognised statement is skipped, because this parser must not become a
/// second, stricter implementation of the linker's grammar.
bool parseLinkerScript(const std::string &text, RegionMap &out,
                       std::string &error);

const char *regionKindName(RegionKind kind);

} // namespace aiesim

#endif // AIESIM_REGIONMAP_H
