#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_m
#error Please specify matrix size M at kernel compile time using -DM=123.
#endif
#ifndef DIM_n
#error Please specify matrix size N at kernel compile time using -DN=123.
#endif

#define DTYPE uint8_t
#define N_TILES 1

constexpr size_t SIZE = DIM_m * DIM_n;

// Constraints for DIM_m and DIM_n
// Judging from aie_api/detail/aie2/transpose.hpp, aie::transpose supports the
// following dimensions:
//  - row/col count must be a power of two
//  - for  8-bit data types, total matrix size must be 128, 64, 32 or 16 bytes
//  - for 16-bit data types, total matrix size must be      64, 32, 16 or 8
//  bytes
//  - for 32-bit data types, total matrix size must be          32, 16, 8 or 4
//  bytes
//  - for 64-bit data types, total matrix size must be              16, 8 or 4
//  bytes
// Let's enforce these constraints here, because the AIE API compilation errors
// are not very helpful; if you end up here because of a failing assertion, it's
// because the AIE API does not support a transpose of the size you requested.
#define IS_POWER_OF_TWO(x) ((x > 0) && ((x & (x - 1)) == 0))
#define IMPLIES(a, b) (!(a) || ((a) && (b))) // a implies b
static_assert(IS_POWER_OF_TWO(DIM_m) && IS_POWER_OF_TWO(DIM_n) &&
              "m and n must be powers of two");
static_assert(IMPLIES(sizeof(DTYPE) == 8,
                      SIZE == 128 || SIZE == 64 || SIZE == 32 || SIZE == 16));
static_assert(IMPLIES(sizeof(DTYPE) == 16,
                      SIZE == 64 || SIZE == 32 || SIZE == 16 || SIZE == 8));
static_assert(IMPLIES(sizeof(DTYPE) == 32,
                      SIZE == 32 || SIZE == 16 || SIZE == 8 || SIZE == 4));
static_assert(IMPLIES(sizeof(DTYPE) == 64,
                      SIZE == 16 || SIZE == 8 || SIZE == 4));

extern "C" {

void transpose(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr) {
  // in and out may not alias, i.e. you cannot transpose in-place with this
  // kernel
  for (int i = 0; i < N_TILES; i++) {
    aie::vector<DTYPE, SIZE> in = aie::load_v<128>(in_ptr);
    aie::vector<DTYPE, SIZE> out = aie::transpose(in, DIM_m, DIM_n);
    aie::store_v(out_ptr, out);
    in_ptr += SIZE;
    out_ptr += SIZE;
  }
}
}