#include "vec_math.h"
void dut(int8_t *restrict v1, int8_t *restrict v2) {
  size_t v3 = 0;
  size_t v4 = 1024;
  size_t v5 = 64;
  for (size_t v6 = v3; v6 < v4; v6 += v5)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      v64int8 v7 = *(v64int8 *)(v1 + v6);
      v64int8 v8 = getAbs(v7);
      *(v64int8 *)(v2 + v6) = v8;
    }
  return;
}
