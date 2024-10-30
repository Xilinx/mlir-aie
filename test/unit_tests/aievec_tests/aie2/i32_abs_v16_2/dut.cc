#include "vec_math.h"
void dut(int32_t *restrict v1, int32_t *restrict v2) {
  size_t v3 = 0;
  size_t v4 = 1024;
  size_t v5 = 16;
  for (size_t v6 = v3; v6 < v4; v6 += v5)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16int32 v7 = *(v16int32 *)(v1 + v6);
      v16int32 v8 = getAbs(v7);
      *(v16int32 *)(v2 + v6) = v8;
    }
  return;
}
