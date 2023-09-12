// Cycle count: 3245
#include "lut_based_ops.h"

void dut(bfloat16 *restrict v1, bfloat16 *restrict v2) {
  int32_t v3 = 0;
  int32_t v4 = 4;
  int32_t v5 = 8;
  int32_t v6 = 16;
  int32_t v7 = 32;
  v16float v8 = broadcast_zero_float();
  bfloat16 v9 = 0.0e+00;
  size_t v10 = 0;
  size_t v11 = 1024;
  size_t v12 = 16;
  for (size_t v13 = v10; v13 < v11; v13 += v12)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v14 = *(v16bfloat16 *)(v1 + v13);
      v16accfloat v15 = getExpBf16(v14);
      v16bfloat16 v16 = to_v16bfloat16(v15);
      *(v16bfloat16 *)(v1 + v13) = v16;
    }
  size_t v17 = 0;
  size_t v18 = 1024;
  size_t v19 = 16;
  v16float v20;
  v16float v21 = v8;
  for (size_t v22 = v17; v22 < v18; v22 += v19)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v23 = *(v16bfloat16 *)(v1 + v22);
      v16accfloat v24 = ups_to_v16accfloat(v23);
      v16accfloat v25 = v16accfloat(v21);
      v16accfloat v26 = add(v24, v25);
      v16float v27 = v16float(v26);
      v21 = v27;
    }
  v20 = v21;
  v16float v28 = shift_bytes(v20, v20, v7);
  v16accfloat v29 = v16accfloat(v20);
  v16accfloat v30 = v16accfloat(v28);
  v16accfloat v31 = add(v29, v30);
  v16float v32 = v16float(v31);
  v16float v33 = shift_bytes(v32, v32, v6);
  v16accfloat v34 = v16accfloat(v32);
  v16accfloat v35 = v16accfloat(v33);
  v16accfloat v36 = add(v34, v35);
  v16float v37 = v16float(v36);
  v16float v38 = shift_bytes(v37, v37, v5);
  v16accfloat v39 = v16accfloat(v37);
  v16accfloat v40 = v16accfloat(v38);
  v16accfloat v41 = add(v39, v40);
  v16float v42 = v16float(v41);
  v16float v43 = shift_bytes(v42, v42, v4);
  v16accfloat v44 = v16accfloat(v42);
  v16accfloat v45 = v16accfloat(v43);
  v16accfloat v46 = add(v44, v45);
  v16float v47 = v16float(v46);
  float v48 = extract_elem(v47, v3);
  bfloat16 v49 = getInvBf16(v48);
  v32bfloat16 v50 = broadcast_to_v32bfloat16(v49);
  v16bfloat16 v51 = extract_v16bfloat16(v50, 0);
  v32bfloat16 v52 = broadcast_to_v32bfloat16(v9);
  v16bfloat16 v53 = extract_v16bfloat16(v52, 0);
  v32bfloat16 v54 = concat(v51, v53);
  size_t v55 = 0;
  size_t v56 = 1024;
  size_t v57 = 16;
  for (size_t v58 = v55; v58 < v56; v58 += v57)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v59 = *(v16bfloat16 *)(v1 + v58);
      v32bfloat16 v60 = concat(v59, v53);
      v16accfloat v61 = mul_elem_16_2(v54, v60);
      v16bfloat16 v62 = to_v16bfloat16(v61);
      *(v16bfloat16 *)(v2 + v58) = v62;
    }
  return;
}
