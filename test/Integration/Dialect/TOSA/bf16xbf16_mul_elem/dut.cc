void dut(bfloat16 *restrict v1, bfloat16 *restrict v2, bfloat16 *restrict v3) {
  size_t v4 = 16;
  size_t v5 = 1024;
  size_t v6 = 0;
  bfloat16 v7 = 0.0e+00;
  v32bfloat16 v8 = broadcast_to_v32bfloat16(v7);
  v16bfloat16 v9 = extract_v16bfloat16(v8, 0);
  for (size_t v10 = v6; v10 < v5; v10 += v4)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v11 = *(v16bfloat16 *)(v1 + v10);
      v16bfloat16 v12 = *(v16bfloat16 *)(v2 + v10);
      v32bfloat16 v13 = concat(v11, v9);
      v32bfloat16 v14 = concat(v12, v9);
      v16accfloat v15 = mul_elem_16_2(v14, v13);
      v16bfloat16 v16 = to_v16bfloat16(v15);
      *(v16bfloat16 *)(v3 + v10) = v16;
    }
  return;
}
