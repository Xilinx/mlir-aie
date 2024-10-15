#include "lut_based_ops.h"
void dut(bfloat16 *restrict v1, float v2, bfloat16 *restrict v3) {
  bfloat16 v4 = 0.0e+00;
  bfloat16 v5 = getInvBf16(v2);
  v32bfloat16 v6 = broadcast_to_v32bfloat16(v5);
  v16bfloat16 v7 = extract_v16bfloat16(v6, 0);
  v32bfloat16 v8 = broadcast_to_v32bfloat16(v4);
  v16bfloat16 v9 = extract_v16bfloat16(v8, 0);
  v32bfloat16 v10 = concat(v7, v9);
  size_t v11 = 0;
  size_t v12 = 1024;
  size_t v13 = 16;
  for (size_t v14 = v11; v14 < v12; v14 += v13)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v15 = *(v16bfloat16 *)(v1 + v14);
      v32bfloat16 v16 = concat(v15, v9);
      v16accfloat v17 = mul_elem_16_2(v16, v10);
      v16bfloat16 v18 = to_v16bfloat16(v17);
      *(v16bfloat16 *)(v3 + v14) = v18;
    }
  return;
}
