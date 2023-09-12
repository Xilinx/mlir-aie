// Cycle count: 3712
#include "lut_based_ops.h"

void dut(bfloat16 *restrict v1, bfloat16 *restrict v2) {
  int32_t v3 = 0;
  int32_t v4 = 2;
  int32_t v5 = 4;
  int32_t v6 = 8;
  int32_t v7 = 16;
  int32_t v8 = 32;
  v16float v9 = broadcast_zero_float();
  bfloat16 v10 = 0.0e+00;
  v32bfloat16 v11 = broadcast_to_v32bfloat16(
      (bfloat16)-338953138925153547590470800371487866880.000000);
  size_t v12 = 0;
  size_t v13 = 1024;
  size_t v14 = 32;
  v32bfloat16 v15;
  v32bfloat16 v16 = v11;
  for (size_t v17 = v12; v17 < v13; v17 += v14)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32bfloat16 v18 = *(v32bfloat16 *)(v1 + v17);
      v32bfloat16 v19 = max(v16, v18);
      v16 = v19;
    }
  v15 = v16;
  v32bfloat16 v20 = shift_bytes(v15, v15, v8);
  v32bfloat16 v21 = max(v15, v20);
  v32bfloat16 v22 = shift_bytes(v21, v21, v7);
  v32bfloat16 v23 = max(v21, v22);
  v32bfloat16 v24 = shift_bytes(v23, v23, v6);
  v32bfloat16 v25 = max(v23, v24);
  v32bfloat16 v26 = shift_bytes(v25, v25, v5);
  v32bfloat16 v27 = max(v25, v26);
  v32bfloat16 v28 = shift_bytes(v27, v27, v4);
  v32bfloat16 v29 = max(v27, v28);
  bfloat16 v30 = extract_elem(v29, v3);
  v32bfloat16 v31 = broadcast_to_v32bfloat16(v30);
  v16bfloat16 v32 = extract_v16bfloat16(v31, 0);
  v16accfloat v33 = ups_to_v16accfloat(v32);
  size_t v34 = 0;
  size_t v35 = 1024;
  size_t v36 = 16;
  for (size_t v37 = v34; v37 < v35; v37 += v36)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v38 = *(v16bfloat16 *)(v1 + v37);
      v16accfloat v39 = ups_to_v16accfloat(v38);
      v16accfloat v40 = sub(v39, v33);
      v16bfloat16 v41 = to_v16bfloat16(v40);
      v16accfloat v42 = getExpBf16(v41);
      v16bfloat16 v43 = to_v16bfloat16(v42);
      *(v16bfloat16 *)(v1 + v37) = v43;
    }
  size_t v44 = 0;
  size_t v45 = 1024;
  size_t v46 = 16;
  v16float v47;
  v16float v48 = v9;
  for (size_t v49 = v44; v49 < v45; v49 += v46)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v50 = *(v16bfloat16 *)(v1 + v49);
      v16accfloat v51 = ups_to_v16accfloat(v50);
      v16accfloat v52 = v16accfloat(v48);
      v16accfloat v53 = add(v51, v52);
      v16float v54 = v16float(v53);
      v48 = v54;
    }
  v47 = v48;
  v16float v55 = shift_bytes(v47, v47, v8);
  v16accfloat v56 = v16accfloat(v47);
  v16accfloat v57 = v16accfloat(v55);
  v16accfloat v58 = add(v56, v57);
  v16float v59 = v16float(v58);
  v16float v60 = shift_bytes(v59, v59, v7);
  v16accfloat v61 = v16accfloat(v59);
  v16accfloat v62 = v16accfloat(v60);
  v16accfloat v63 = add(v61, v62);
  v16float v64 = v16float(v63);
  v16float v65 = shift_bytes(v64, v64, v6);
  v16accfloat v66 = v16accfloat(v64);
  v16accfloat v67 = v16accfloat(v65);
  v16accfloat v68 = add(v66, v67);
  v16float v69 = v16float(v68);
  v16float v70 = shift_bytes(v69, v69, v5);
  v16accfloat v71 = v16accfloat(v69);
  v16accfloat v72 = v16accfloat(v70);
  v16accfloat v73 = add(v71, v72);
  v16float v74 = v16float(v73);
  float v75 = extract_elem(v74, v3);
  bfloat16 v76 = getInvBf16(v75);
  v32bfloat16 v77 = broadcast_to_v32bfloat16(v76);
  v16bfloat16 v78 = extract_v16bfloat16(v77, 0);
  v32bfloat16 v79 = broadcast_to_v32bfloat16(v10);
  v16bfloat16 v80 = extract_v16bfloat16(v79, 0);
  v32bfloat16 v81 = concat(v78, v80);
  size_t v82 = 0;
  size_t v83 = 1024;
  size_t v84 = 16;
  for (size_t v85 = v82; v85 < v83; v85 += v84)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v86 = *(v16bfloat16 *)(v1 + v85);
      v32bfloat16 v87 = concat(v86, v80);
      v16accfloat v88 = mul_elem_16_2(v81, v87);
      v16bfloat16 v89 = to_v16bfloat16(v88);
      *(v16bfloat16 *)(v2 + v85) = v89;
    }
  return;
}
