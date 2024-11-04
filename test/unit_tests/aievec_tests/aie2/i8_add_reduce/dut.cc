void dut(int8_t *restrict v1, int8_t *restrict v2) {
  int32_t v3 = 0;
  v64int8 v4 = broadcast_zero_s8();
  size_t v5 = 0;
  size_t v6 = 1024;
  size_t v7 = 64;
  v64int8 v8;
  v64int8 v9 = v4;
  for (size_t v10 = v5; v10 < v6; v10 += v7)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      v64int8 v11 = *(v64int8 *)(v1 + v10);
      v64int8 v12 = add(v9, v11);
      v9 = v12;
    }
  v8 = v9;
  v64int8 v13 = shift_bytes(v8, undef_v64int8(), 32);
  v64int8 v14 = add(v8, v13);
  v64int8 v15 = shift_bytes(v14, undef_v64int8(), 16);
  v64int8 v16 = add(v14, v15);
  v64int8 v17 = shift_bytes(v16, undef_v64int8(), 8);
  v64int8 v18 = add(v16, v17);
  v64int8 v19 = shift_bytes(v18, undef_v64int8(), 4);
  v64int8 v20 = add(v18, v19);
  v64int8 v21 = shift_bytes(v20, undef_v64int8(), 2);
  v64int8 v22 = add(v20, v21);
  v64int8 v23 = shift_bytes(v22, undef_v64int8(), 1);
  v64int8 v24 = add(v22, v23);
  int8_t v25 = extract_elem(v24, v3);
  *(int8_t *)v2 = v25;
  return;
}
