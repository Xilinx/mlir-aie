//===- yolo_m10_concat_pool.cc -------------------------------*- C++ -*-===//
//
// Tiny i8 concatenation helper for the m10 2-tile split: concat(lo[N], hi[M])
// → full[N+M]. Used on the gemm tile to merge the two pool halves produced
// by the conv_pool tiles before feeding the 1280→2 Gemm. Vectorized 32-byte
// copies; both halves are 640 B in the m10 call site (lo_count = hi_count
// = 640), so 40 vec stores total per call.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

extern "C" {

void yolo_m10_concat_pool_i8(int8_t *__restrict lo, int8_t *__restrict hi,
                             int8_t *__restrict full, const int32_t lo_count,
                             const int32_t hi_count) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();
  for (int i = 0; i < lo_count; i += 32) {
    aie::vector<int8, 32> v = aie::load_v<32>(lo + i);
    aie::store_v(full + i, v);
  }
  for (int i = 0; i < hi_count; i += 32) {
    aie::vector<int8, 32> v = aie::load_v<32>(hi + i);
    aie::store_v(full + lo_count + i, v);
  }
  event1();
}

} // extern "C"
