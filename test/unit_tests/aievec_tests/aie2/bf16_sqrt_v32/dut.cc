#include "vec_math.h"
void dut(bfloat16 *restrict v1, bfloat16 *restrict v2) {
  size_t v3 = 0;
  size_t v4 = 1024;
  size_t v5 = 32;
  for (size_t v6 = v3; v6 < v4; v6 += v5)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32bfloat16 v7 = *(v32bfloat16 *)(v1 + v6);
      v32bfloat16 v8 = getSqrtBf16(v7);
      *(v32bfloat16 *)(v2 + v6) = v8;
    }
  return;
}
