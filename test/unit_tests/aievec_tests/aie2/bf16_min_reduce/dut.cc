void dut(bfloat16 *restrict v1, bfloat16 *restrict v2) {
  int32_t v3 = 0;
  v32bfloat16 v4 = broadcast_to_v32bfloat16(
      (bfloat16)338953138925153547590470800371487866880.000000);
  size_t v5 = 0;
  size_t v6 = 1024;
  size_t v7 = 32;
  v32bfloat16 v8;
  v32bfloat16 v9 = v4;
  for (size_t v10 = v5; v10 < v6; v10 += v7)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32bfloat16 v11 = *(v32bfloat16 *)(v1 + v10);
      v32bfloat16 v12 = min(v9, v11);
      v9 = v12;
    }
  v8 = v9;
  v32bfloat16 v13 = shift_bytes(v8, undef_v32bfloat16(), 32);
  v32bfloat16 v14 = min(v8, v13);
  v32bfloat16 v15 = shift_bytes(v14, undef_v32bfloat16(), 16);
  v32bfloat16 v16 = min(v14, v15);
  v32bfloat16 v17 = shift_bytes(v16, undef_v32bfloat16(), 8);
  v32bfloat16 v18 = min(v16, v17);
  v32bfloat16 v19 = shift_bytes(v18, undef_v32bfloat16(), 4);
  v32bfloat16 v20 = min(v18, v19);
  v32bfloat16 v21 = shift_bytes(v20, undef_v32bfloat16(), 2);
  v32bfloat16 v22 = min(v20, v21);
  bfloat16 v23 = extract_elem(v22, v3);
  *(bfloat16 *)v2 = v23;
  return;
}
