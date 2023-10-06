#include "vec_math.h"
void dut(int16_t *restrict v1, int16_t *restrict v2) {
  size_t v3 = 0;
  size_t v4 = 1024;
  size_t v5 = 32;
  for (size_t v6 = v3; v6 < v4; v6 += v5)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int16 v7 = *(v32int16 *)(v1 + v6);
      v32int16 v8 = getAbs(v7);
      *(v32int16 *)(v2 + v6) = v8;
    }
  return;
}
