//===- region_map_test.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The region map, against a linker script aiecc really emitted. The fixture is
// verbatim output for one core of the block_datatypes whole_array_shuffle
// design, trimmed only of its INPUT() lines: a hand-written approximation
// would agree with the parser and with nothing else.
//
//===----------------------------------------------------------------------===//

#include "TestSupport.h"

#include "aiesim/RegionMap.h"

using namespace aiesim;

namespace {

// From ldScripts_main_core_0_5.ld.script. Stack at 0x70000+0xD00 and the
// first buffer of the tile's own memory band at 0x70D00: touching, which is
// the arrangement the guard exists for.
const char *kScript = R"LD(
MEMORY
{
   program (RX) : ORIGIN = 0, LENGTH = 0x0020000
   data (!RX) : ORIGIN = 0x7D20C, LENGTH = 0x2DF4
}
ENTRY(__start)
SECTIONS
{
  . = 0x0;
  .text : {
     *crt0.o(.text*)
     _ctors_start = .;
     *(.text*)
  } > program
  .data : {
     *(.data*)
  } > data

. = 0x70000;
_sp_start_value_DM_stack = .;
. += 0xD00; /* stack */
. = 0x40D00;
C_L1L2_2_0_buff_0 = .;
. += 0x1200;
/* No tile with memory exists to the west. */
. = 0x50000;
. += 0x10000;
. = 0x70D00;
C_L1L2_3_0_buff_0 = .;
. += 0x1200;
. = 0x74000;
C_L1L2_3_0_buff_1 = .;
. += 0x1200;
  .bss : { *(.bss*) } > data
}
)LD";

RegionMap parsed() {
  RegionMap map;
  std::string err;
  AIESIM_CHECK(parseLinkerScript(kScript, map, err));
  AIESIM_CHECK(err.empty());
  return map;
}

void testParsesMemoryBlockAndAllocations() {
  RegionMap map = parsed();
  const Region *prog = nullptr, *stack = nullptr, *buf = nullptr;
  for (const Region &r : map.regions()) {
    if (r.name == "program") prog = &r;
    if (r.kind == RegionKind::Stack) stack = &r;
    if (r.name == "C_L1L2_3_0_buff_0") buf = &r;
  }
  AIESIM_CHECK(prog && prog->begin == 0 && prog->size == 0x20000);
  AIESIM_CHECK(stack && stack->begin == 0x70000 && stack->size == 0xD00);
  AIESIM_CHECK(stack && stack->name == "_sp_start_value_DM_stack");
  AIESIM_CHECK(buf && buf->begin == 0x70D00 && buf->size == 0x1200);

  // The `. = 0x50000; . += 0x10000;` padding run carries no symbol and must
  // not become a region; treating it as one would fabricate a 64 KB
  // allocation that the design never made.
  for (const Region &r : map.regions())
    AIESIM_CHECK(r.begin != 0x50000 || r.kind == RegionKind::Data);
}

void testStackClearanceIsZeroHere() {
  RegionMap map = parsed();
  uint32_t gap = 0xFFFFFFFF;
  std::string next;
  AIESIM_CHECK(map.stackClearance(gap, next));
  // The whole point: one byte of frame overrun lands in a live buffer.
  AIESIM_CHECK_EQ(gap, 0u);
  AIESIM_CHECK(next == "C_L1L2_3_0_buff_0");
}

void testStackPointerGuard() {
  RegionMap map = parsed();
  std::string why;

  AIESIM_CHECK(map.checkStackPointer(0x70000, why));
  AIESIM_CHECK(map.checkStackPointer(0x70CFF, why));

  // One byte past the reservation is already inside the buffer.
  why.clear();
  AIESIM_CHECK(!map.checkStackPointer(0x70D00, why));
  AIESIM_CHECK(why.find("C_L1L2_3_0_buff_0") != std::string::npos);
  AIESIM_CHECK(why.find("past the end") != std::string::npos);

  why.clear();
  AIESIM_CHECK(!map.checkStackPointer(0x6FFFF, why));
  AIESIM_CHECK(why.find("below the start") != std::string::npos);
}

void testWriteOverrunGuard() {
  RegionMap map = parsed();
  std::string why;

  // Wholly inside the buffer.
  AIESIM_CHECK(map.checkWrite(0x70D00, 0x1200, why));
  // Starts in the stack, runs one byte past it.
  why.clear();
  AIESIM_CHECK(!map.checkWrite(0x70CFC, 5, why));
  AIESIM_CHECK(why.find("_sp_start_value_DM_stack") != std::string::npos);
  AIESIM_CHECK(why.find("C_L1L2_3_0_buff_0") != std::string::npos);
  // Unallocated addresses are not this check's business: most of data memory
  // is legitimately unnamed and faulting on it would make the guard unusable.
  AIESIM_CHECK(map.checkWrite(0x60000, 64, why));
}

void testOverlapDetection() {
  RegionMap map;
  Region a; a.name = "buf_a"; a.begin = 0x1000; a.size = 0x200;
  Region b; b.name = "buf_b"; b.begin = 0x1100; b.size = 0x200;
  map.add(a);
  map.add(b);
  auto over = map.overlaps();
  AIESIM_CHECK_EQ(over.size(), size_t(1));
  if (!over.empty()) {
    AIESIM_CHECK(over[0].begin == 0x1100 && over[0].end == 0x1200);
  }
  // The parsed script allocates each buffer once; it must report none.
  AIESIM_CHECK(parsed().overlaps().empty());
}

void testFindContainingPrefersTheSpecificRegion() {
  RegionMap map = parsed();
  // 0x70D00 sits inside the buffer. Answering with a broad container instead
  // would make every guard message name the wrong owner.
  const Region *r = map.findContaining(0x70D00);
  AIESIM_CHECK(r != nullptr);
  if (r)
    AIESIM_CHECK(r->name == "C_L1L2_3_0_buff_0");
}

void testRejectsNonScripts() {
  RegionMap map;
  std::string err;
  AIESIM_CHECK(!parseLinkerScript("hello world\n", map, err));
  AIESIM_CHECK(!err.empty());
}

} // namespace

int main() {
  testParsesMemoryBlockAndAllocations();
  testStackClearanceIsZeroHere();
  testStackPointerGuard();
  testWriteOverrunGuard();
  testOverlapDetection();
  testFindContainingPrefersTheSpecificRegion();
  testRejectsNonScripts();
  return aiesim_test::summarize("region_map_test");
}
