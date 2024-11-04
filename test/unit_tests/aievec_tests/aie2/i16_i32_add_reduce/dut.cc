void dut(int16_t *restrict v1, int32_t *restrict v2) {
  int32_t v3 = 0;
  int32_t v4 = 4;
  int32_t v5 = 8;
  int32_t v6 = 16;
  int32_t v7 = 32;
  v16int32 v8 = broadcast_zero_s32();
  size_t v9 = 0;
  size_t v10 = 1024;
  size_t v11 = 16;
  v16int32 v12;
  v16int32 v13 = v8;
  for (size_t v14 = v9; v14 < v10; v14 += v11)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16int16 v15 = *(v16int16 *)(v1 + v14);
      v16acc64 v16 = ups_to_v16acc64(v13, 0);
      v16acc64 v17 = ups_to_v16acc64(v15, 0);
      v16acc64 v18 = add(v16, v17);
      v16int32 v19 = srs_to_v16int32(v18, 0);
      v13 = v19;
    }
  v12 = v13;
  v16int32 v20 = shift_bytes(v12, v12, v7);
  v16int32 v21 = add(v12, v20);
  v16int32 v22 = shift_bytes(v21, v21, v6);
  v16int32 v23 = add(v21, v22);
  v16int32 v24 = shift_bytes(v23, v23, v5);
  v16int32 v25 = add(v23, v24);
  v16int32 v26 = shift_bytes(v25, v25, v4);
  v16int32 v27 = add(v25, v26);
  int32_t v28 = extract_elem(v27, v3);
  *(int32_t *)v2 = v28;
  return;
}
