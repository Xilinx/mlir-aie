void dut(bfloat16 *restrict v1, bfloat16 *restrict v2) {
  v32bfloat16 v3 = broadcast_to_v32bfloat16((bfloat16)0.000000);
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 32;
  v32bfloat16 v7;
  v32bfloat16 v8 = v3;
  for (size_t v9 = v4; v9 < v5; v9 += v6)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32bfloat16 v10 = *(v32bfloat16 *)(v1 + v9);
      v32bfloat16 v11 = max(v8, v10);
      v8 = v11;
    }
  v7 = v8;
  v32bfloat16 v12 = shift_bytes(v7, undef_v32bfloat16(), 32);
  v32bfloat16 v13 = max(v7, v12);
  v32bfloat16 v14 = shift_bytes(v13, undef_v32bfloat16(), 16);
  v32bfloat16 v15 = max(v13, v14);
  v32bfloat16 v16 = shift_bytes(v15, undef_v32bfloat16(), 8);
  v32bfloat16 v17 = max(v15, v16);
  v32bfloat16 v18 = shift_bytes(v17, undef_v32bfloat16(), 4);
  v32bfloat16 v19 = max(v17, v18);
  v32bfloat16 v20 = shift_bytes(v19, undef_v32bfloat16(), 2);
  v32bfloat16 v21 = max(v19, v20);
  bfloat16 v22 = extract_elem(v21, 0);
  *(bfloat16 *)v2 = v22;
  return;
}
