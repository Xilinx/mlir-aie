#include "aie_api/aie.hpp"
#include "aie_api/utils.hpp"
#include <cstdint>
#include <cstdio>

template <typename T> const char *tid() { return "@"; }

template <> const char *tid<int8_t>() { return "i"; }
template <> const char *tid<int16_t>() { return "i"; }
template <> const char *tid<int32_t>() { return "i"; }

template <int nlanes, typename elemtype, typename vtype> void printv(vtype v) {
  printf("vector<%dx%s%u>[ ", nlanes, tid<elemtype>(), 8 * sizeof(elemtype));
  aie::print(aie::vector<elemtype, nlanes>(v));
  printf("]\n");
}

void printv16xi32(v16int32 v) { printv<16, int32_t>(v); }

void printv8xi32(v8int32 v) { printv<8, int32_t>(v); }

void printv32xi16(v32int16 v) { printv<32, int16_t>(v); }

void printv16xi16(v16int16 v) { printv<16, int16_t>(v); }

void printv32xi8(v32int8 v) { printv<32, int8_t>(v); }

void printv64xi8(v64int8 v) { printv<64, int8_t>(v); }

alignas(32) int32_t buff_i32[64];
alignas(32) int16_t buff_i16[64];
alignas(32) int8_t buff_i8[128];

int32_t *loadA64xi32() {
  for (int i = 0; i < 64; ++i)
    buff_i32[i] = i;
  return buff_i32;
}

int16_t *loadA64xi16() {
  for (int i = 0; i < 64; ++i)
    buff_i16[i] = i;
  return buff_i16;
}

int8_t *loadA128xi8() {
  for (int i = 0; i < 128; ++i)
    buff_i8[i] = i;
  return buff_i8;
}
