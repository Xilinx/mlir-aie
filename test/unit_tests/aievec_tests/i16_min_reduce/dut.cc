void dut(int16_t *restrict v1, int16_t *restrict v2) {
  v32int16 v3 = broadcast_to_v32int16((int16_t)32767);
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 32;
  v32int16 v7;
  v32int16 v8 = v3;
  for (size_t v9 = v4; v9 < v5; v9 += v6)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int16 v10 = *(v32int16 *)(v1 + v9);
      v32int16 v11 = min(v8, v10);
      v8 = v11;
    }
  v7 = v8;
  v32int16 v12 = shift_bytes(v7, undef_v32int16(), 32);
  v32int16 v13 = min(v7, v12);
  v32int16 v14 = shift_bytes(v13, undef_v32int16(), 16);
  v32int16 v15 = min(v13, v14);
  v32int16 v16 = shift_bytes(v15, undef_v32int16(), 8);
  v32int16 v17 = min(v15, v16);
  v32int16 v18 = shift_bytes(v17, undef_v32int16(), 4);
  v32int16 v19 = min(v17, v18);
  v32int16 v20 = shift_bytes(v19, undef_v32int16(), 2);
  v32int16 v21 = min(v19, v20);
  int16_t v22 = extract_elem(v21, 0);
  *(int16_t *)v2 = v22;
  return;
}
