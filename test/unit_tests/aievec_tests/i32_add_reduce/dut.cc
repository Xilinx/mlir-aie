void dut(int32_t *restrict v1, int32_t *restrict v2) {
  v16int32 v3 = undef_v16int32();
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 16;
  v16int32 v7;
  v16int32 v8 = v3;
  for (size_t v9 = v4; v9 < v5; v9 += v6)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16int32 v10 = *(v16int32 *)(v1 + v9);
      v16int32 v11 = add(v8, v10);
      v8 = v11;
    }
  v7 = v8;
  v16int32 v12 = shift_bytes(v7, undef_v16int32(), 32);
  v16int32 v13 = add(v7, v12);
  v16int32 v14 = shift_bytes(v13, undef_v16int32(), 16);
  v16int32 v15 = add(v13, v14);
  v16int32 v16 = shift_bytes(v15, undef_v16int32(), 8);
  v16int32 v17 = add(v15, v16);
  v16int32 v18 = shift_bytes(v17, undef_v16int32(), 4);
  v16int32 v19 = add(v17, v18);
  int32_t v20 = extract_elem(v19, 0);
  *(int32_t *)v2 = v20;
  return;
}
