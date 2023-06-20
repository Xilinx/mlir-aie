void dut(int32_t *restrict v1, int32_t *restrict v2) {
  int32_t v3 = 0;
  v16int32 v4 = broadcast_to_v16int32((int32_t)-2147483648);
  size_t v5 = 0;
  size_t v6 = 1024;
  size_t v7 = 16;
  v16int32 v8;
  v16int32 v9 = v4;
  for (size_t v10 = v5; v10 < v6; v10 += v7)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16int32 v11 = *(v16int32 *)(v1 + v10);
      v16int32 v12 = max(v9, v11);
      v9 = v12;
    }
  v8 = v9;
  v16int32 v13 = shift_bytes(v8, undef_v16int32(), 32);
  v16int32 v14 = max(v8, v13);
  v16int32 v15 = shift_bytes(v14, undef_v16int32(), 16);
  v16int32 v16 = max(v14, v15);
  v16int32 v17 = shift_bytes(v16, undef_v16int32(), 8);
  v16int32 v18 = max(v16, v17);
  v16int32 v19 = shift_bytes(v18, undef_v16int32(), 4);
  v16int32 v20 = max(v18, v19);
  int32_t v21 = extract_elem(v20, v3);
  *(int32_t *)v2 = v21;
  return;
}
