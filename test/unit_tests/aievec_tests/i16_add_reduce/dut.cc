void dut(int16_t *restrict v1, int16_t *restrict v2) {
  int32_t v3 = 0;
  v32int16 v4 = broadcast_zero_s16();
  size_t v5 = 0;
  size_t v6 = 1024;
  size_t v7 = 32;
  v32int16 v8;
  v32int16 v9 = v4;
  for (size_t v10 = v5; v10 < v6; v10 += v7)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int16 v11 = *(v32int16 *)(v1 + v10);
      v32int16 v12 = add(v9, v11);
      v9 = v12;
    }
  v8 = v9;
  v32int16 v13 = shift_bytes(v8, undef_v32int16(), 32);
  v32int16 v14 = add(v8, v13);
  v32int16 v15 = shift_bytes(v14, undef_v32int16(), 16);
  v32int16 v16 = add(v14, v15);
  v32int16 v17 = shift_bytes(v16, undef_v32int16(), 8);
  v32int16 v18 = add(v16, v17);
  v32int16 v19 = shift_bytes(v18, undef_v32int16(), 4);
  v32int16 v20 = add(v18, v19);
  v32int16 v21 = shift_bytes(v20, undef_v32int16(), 2);
  v32int16 v22 = add(v20, v21);
  int16_t v23 = extract_elem(v22, v3);
  *(int16_t *)v2 = v23;
  return;
}
