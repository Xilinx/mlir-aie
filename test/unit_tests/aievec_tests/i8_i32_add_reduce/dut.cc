void dut(int8_t *restrict v1, int32_t *restrict v2) {
  int32_t v3 = 0;
  int32_t v4 = 4;
  int32_t v5 = 8;
  int32_t v6 = 16;
  int32_t v7 = 32;
  v32int32 v8 = concat(broadcast_zero_s32(), broadcast_zero_s32());
  size_t v9 = 0;
  size_t v10 = 1024;
  size_t v11 = 32;
  v32int32 v12;
  v32int32 v13 = v8;
  for (size_t v14 = v9; v14 < v10; v14 += v11)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int8 v15 = *(v32int8 *)(v1 + v14);
      v32acc32 v16 = ups_to_v32acc32(v15, 0);
      v32acc32 v17 = v32acc32(v13);
      v32acc32 v18 = add(v16, v17);
      v32int32 v19 = v32int32(v18);
      v13 = v19;
    }
  v12 = v13;
  v16int32 v20 = extract_v16int32(v12, 0);
  v16int32 v21 = extract_v16int32(v12, 1);
  v16int32 v22 = add(v20, v21);
  v16int32 v23 = shift_bytes(v22, v22, v7);
  v16int32 v24 = add(v22, v23);
  v16int32 v25 = shift_bytes(v24, v24, v6);
  v16int32 v26 = add(v24, v25);
  v16int32 v27 = shift_bytes(v26, v26, v5);
  v16int32 v28 = add(v26, v27);
  v16int32 v29 = shift_bytes(v28, v28, v4);
  v16int32 v30 = add(v28, v29);
  int32_t v31 = extract_elem(v30, v3);
  *(int32_t *)v2 = v31;
  return;
}
